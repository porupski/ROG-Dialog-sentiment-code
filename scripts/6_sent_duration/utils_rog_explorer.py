"""Helpers for the ROG explorer notebook (06_rog_dia_explorer.ipynb).

Keep plot/sampling code OUT of the notebook so the notebook stays scannable.
No side effects on import; callers pass the data in.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

LABELS_6 = [
    "predominantlyNegative", "mixedNegative", "neutralNegative",
    "neutralPositive",       "mixedPositive", "predominantlyPositive",
]
SENT_6_TO_3 = {
    "predominantlyNegative": "negative", "mixedNegative": "negative",
    "neutralNegative":       "neutral",  "neutralPositive": "neutral",
    "mixedPositive":         "positive", "predominantlyPositive": "positive",
}
SENT_6_ORD = {lbl: i for i, lbl in enumerate(LABELS_6)}
BACKBONES = ["xlm-roberta-large", "xlm-r-parla", "xlm-r-parlasent"]
MIXTURES  = ["S", "P", "R", "SP", "SR", "PR", "SPR"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_rog(rog_path: Path) -> pd.DataFrame:
    """Return a DataFrame with per-row duration, words, wps, sentiment, sentiment_3, split, text, audio_path."""
    rows = []
    with open(rog_path) as f:
        for line in f:
            r = json.loads(line)
            s = (r.get("labels") or {}).get("sentiment")
            if s is None:
                continue
            dur = r["end_t"] - r["start_t"]
            text = r["text"]
            words = len(text.split())
            wps = words / dur if dur > 0 else 0.0
            rows.append({
                "instance_id": r["instance_id"],
                "split":       r["split"],
                "speaker":     r["speaker"],
                "audio_path":  r.get("audio_path"),
                "text":        text,
                "sentiment":   s,
                "sentiment_3": SENT_6_TO_3[s],
                "sentiment_ord": SENT_6_ORD[s],
                "duration":    dur,
                "words":       words,
                "wps":         wps,
            })
    return pd.DataFrame(rows)


def load_matrix_predictions(matrix_dir: Path) -> dict:
    """Return {instance_id: [{cell, pred_6, pred_3, gold_6, gold_3, correct_6, correct_3}, ...]}"""
    out = defaultdict(list)
    for d in sorted(matrix_dir.iterdir()):
        if not d.is_dir() or "__" not in d.name:
            continue
        pf = d / "predictions_rog_dia.jsonl"
        if not pf.exists():
            continue
        with open(pf) as f:
            for line in f:
                r = json.loads(line)
                out[r["instance_id"]].append({
                    "cell":       d.name,
                    "pred_6":     r["pred_6cls"],
                    "gold_6":     r["gold_6cls"],
                    "pred_3":     r["pred_3cls"],
                    "gold_3":     r["gold_3cls"],
                    "correct_6":  r["pred_6cls"] == r["gold_6cls"],
                    "correct_3":  r["pred_3cls"] == r["gold_3cls"],
                })
    return out


# ---------------------------------------------------------------------------
# Plots — histograms + 3-D / 2-D topography over duration × WPS
# ---------------------------------------------------------------------------

def plot_histograms(df: pd.DataFrame, duration_floor: float = 2.0, wps_clip: float = 15.0):
    """Three-panel histogram of duration, word count, WPS. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))

    axes[0].hist(df["duration"], bins=60, color="#4472c4", edgecolor="#2a4a85", alpha=0.88)
    if duration_floor:
        axes[0].axvline(duration_floor, color="crimson", ls="--", lw=1.2,
                        label=f"{duration_floor:g} s floor")
        axes[0].legend()
    axes[0].set_xlabel("duration (s)"); axes[0].set_ylabel("count")
    axes[0].set_title(f"Duration · median {df['duration'].median():.2f} s · "
                      f"mean {df['duration'].mean():.2f} s")

    axes[1].hist(df["words"], bins=range(0, int(df["words"].max()) + 2),
                 color="#5eb45e", edgecolor="#2f6b2f", alpha=0.88)
    axes[1].set_xlabel("word count"); axes[1].set_ylabel("count")
    axes[1].set_title(f"Words · median {df['words'].median():.0f} · "
                      f"mean {df['words'].mean():.2f}")

    wps_c = df["wps"].clip(upper=wps_clip)
    axes[2].hist(wps_c, bins=60, color="#d08b1a", edgecolor="#7a4f10", alpha=0.88)
    axes[2].axvline(df["wps"].median(), color="crimson", ls="--", lw=1.2,
                    label=f"median {df['wps'].median():.2f}")
    axes[2].set_xlabel(f"WPS (clipped at {wps_clip:g})"); axes[2].set_ylabel("count")
    axes[2].set_title(f"Speech rate · median {df['wps'].median():.2f} · "
                      f"mean {df['wps'].mean():.2f}")
    axes[2].legend()

    fig.tight_layout()
    return fig


def _bin_edges(series, n_bins, mode="quantile", clip=None):
    vals = series.values if clip is None else np.clip(series.values, 0, clip)
    if mode == "quantile":
        q = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(vals, q)
    else:  # fixed
        edges = np.linspace(vals.min(), vals.max(), n_bins + 1)
    # ensure strict monotonic
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def sentiment_topography(df: pd.DataFrame, n_dur_bins=5, n_wps_bins=5,
                         wps_clip=10.0, mode="quantile"):
    """Return (dur_edges, wps_edges, mean_sent, count) arrays for the 2-D grid of mean
    sentiment ordinal (0-5) over duration × WPS. Empty cells return NaN."""
    dur_edges = _bin_edges(df["duration"], n_dur_bins, mode=mode)
    wps_edges = _bin_edges(df["wps"], n_wps_bins, mode=mode, clip=wps_clip)

    mean_sent = np.full((n_wps_bins, n_dur_bins), np.nan)
    count     = np.zeros((n_wps_bins, n_dur_bins), dtype=int)

    dur = df["duration"].values
    wps = np.clip(df["wps"].values, 0, wps_clip)
    sent = df["sentiment_ord"].values

    for i in range(n_wps_bins):
        in_wps = (wps >= wps_edges[i]) & (wps < wps_edges[i + 1] + (1e-6 if i == n_wps_bins - 1 else 0))
        for j in range(n_dur_bins):
            in_dur = (dur >= dur_edges[j]) & (dur < dur_edges[j + 1] + (1e-6 if j == n_dur_bins - 1 else 0))
            mask = in_wps & in_dur
            n = int(mask.sum())
            count[i, j] = n
            if n > 0:
                mean_sent[i, j] = float(sent[mask].mean())
    return dur_edges, wps_edges, mean_sent, count


def plot_sentiment_topography_2d(df: pd.DataFrame, n_dur_bins=5, n_wps_bins=5,
                                  wps_clip=10.0, mode="quantile"):
    """2-D heatmap over (duration, WPS). Cell color = mean sentiment ordinal (0-5).
    3 = neutralPositive (ROG's modal class), so the diverging colormap is centred there."""
    import matplotlib.pyplot as plt
    dur_edges, wps_edges, mean_sent, count = sentiment_topography(
        df, n_dur_bins=n_dur_bins, n_wps_bins=n_wps_bins, wps_clip=wps_clip, mode=mode)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(mean_sent, origin="lower", aspect="auto",
                   extent=[dur_edges[0], dur_edges[-1], wps_edges[0], wps_edges[-1]],
                   cmap="RdYlGn", vmin=1.0, vmax=4.0)
    # overlay n per cell
    for i in range(n_wps_bins):
        for j in range(n_dur_bins):
            if count[i, j] == 0:
                continue
            x = 0.5 * (dur_edges[j] + dur_edges[j + 1])
            y = 0.5 * (wps_edges[i] + wps_edges[i + 1])
            ax.text(x, y, f"{mean_sent[i, j]:.2f}\nn={count[i, j]}",
                    ha="center", va="center", fontsize=7.5, color="black")

    cbar = fig.colorbar(im, ax=ax, label="mean sentiment ordinal (0=predNeg … 5=predPos)")
    cbar.ax.axhline(3.0, color="black", lw=0.6)  # the neutral/positive boundary in ord space
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("WPS (clipped)")
    ax.set_title(f"Sentiment topography — mean class over duration × WPS "
                 f"({n_dur_bins}×{n_wps_bins} {mode} bins, n={len(df)})")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# F1 topography — model performance per (duration × WPS) bin.
# This is the "where does the model actually work?" view, not the raw-label view.
# ---------------------------------------------------------------------------

def _macro_f1(gold, pred, labels):
    import numpy as np
    f1s = []
    for lbl in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(gold, pred) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(gold, pred) if g == lbl and p != lbl)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def f1_topography(df: pd.DataFrame, cell_preds: dict, cell_key: str | None = None,
                   n_dur_bins=5, n_wps_bins=5, wps_clip=10.0, mode="quantile",
                   scheme: str = "3"):
    """Compute per-(duration × WPS)-bin macro-F1 for one matrix cell or the best-cell
    envelope.

    cell_key: either a '{backbone}__{mixture}' key (one specific cell) or None
              (envelope — best-F1-per-bin across all available cells, matching the
              convention used by 01/02 duration scripts).
    scheme:   '3' or '6' — which F1 to compute per bin.

    Returns (dur_edges, wps_edges, f1_matrix, count_matrix, chosen_cell_matrix).
    f1_matrix[i_wps, j_dur] = macro-F1 in that bin (NaN if n<min_n).
    chosen_cell_matrix[i, j] = the cell name picked by the envelope in that bin (None if
    cell_key was specified).
    """
    if scheme not in ("3", "6"):
        raise ValueError("scheme must be '3' or '6'")
    labels = [
        "negative", "neutral", "positive",
    ] if scheme == "3" else LABELS_6

    # Build an instance_id -> (duration, wps) lookup from the dataframe
    dur_map = dict(zip(df["instance_id"], df["duration"]))
    wps_map = dict(zip(df["instance_id"], df["wps"]))

    # Filter df to only rows that have matrix predictions (otherwise bins are built off
    # the whole corpus but graded on a fraction of it).
    scored_ids = set(cell_preds.keys())
    scored_df = df[df["instance_id"].isin(scored_ids)].copy()
    if scored_df.empty:
        raise RuntimeError("no overlap between df and cell_preds — did the matrix test "
                           "split change? Pass the matrix dir that produced these predictions.")

    dur_edges = _bin_edges(scored_df["duration"], n_dur_bins, mode=mode)
    wps_edges = _bin_edges(scored_df["wps"], n_wps_bins, mode=mode, clip=wps_clip)

    # For every scored instance_id, locate its (i_wps, j_dur) cell.
    def _locate(dur, wps_v):
        wps_v_c = min(max(wps_v, 0), wps_clip)
        j = int(np.searchsorted(dur_edges, dur, side="right") - 1)
        i = int(np.searchsorted(wps_edges, wps_v_c, side="right") - 1)
        j = min(max(j, 0), n_dur_bins - 1)
        i = min(max(i, 0), n_wps_bins - 1)
        return i, j

    # Pick a candidate cell list.
    if cell_key is not None:
        candidate_cells = [cell_key]
    else:
        all_cells = sorted({p["cell"] for preds in cell_preds.values() for p in preds})
        candidate_cells = all_cells

    # For each candidate cell, compute F1 per bin. Then take argmax per bin.
    f1_per_cell = {}
    for ck in candidate_cells:
        bins_gold = defaultdict(list)
        bins_pred = defaultdict(list)
        for iid, preds in cell_preds.items():
            p = next((x for x in preds if x["cell"] == ck), None)
            if p is None:
                continue
            dur = dur_map.get(iid); wps_v = wps_map.get(iid)
            if dur is None or wps_v is None:
                continue
            i, j = _locate(dur, wps_v)
            if scheme == "3":
                bins_gold[(i, j)].append(p["gold_3"])
                bins_pred[(i, j)].append(p["pred_3"])
            else:
                bins_gold[(i, j)].append(p["gold_6"])
                bins_pred[(i, j)].append(p["pred_6"])
        mat = np.full((n_wps_bins, n_dur_bins), np.nan)
        cnt = np.zeros((n_wps_bins, n_dur_bins), dtype=int)
        for (i, j), g in bins_gold.items():
            if len(g) == 0:
                continue
            mat[i, j] = _macro_f1(g, bins_pred[(i, j)], labels)
            cnt[i, j] = len(g)
        f1_per_cell[ck] = (mat, cnt)

    # Envelope: per bin, pick argmax F1 across candidate cells. Keep the cell name.
    f1 = np.full((n_wps_bins, n_dur_bins), np.nan)
    cnt_out = np.zeros((n_wps_bins, n_dur_bins), dtype=int)
    chosen = np.full((n_wps_bins, n_dur_bins), None, dtype=object)
    for i in range(n_wps_bins):
        for j in range(n_dur_bins):
            best_v, best_ck, best_n = np.nan, None, 0
            for ck, (mat, cnt) in f1_per_cell.items():
                v = mat[i, j]; n = cnt[i, j]
                if np.isnan(v):
                    continue
                if np.isnan(best_v) or v > best_v:
                    best_v, best_ck, best_n = v, ck, n
            if best_ck is not None:
                f1[i, j] = best_v
                cnt_out[i, j] = best_n
                chosen[i, j] = best_ck
    return dur_edges, wps_edges, f1, cnt_out, chosen


def plot_f1_topography_2d(df: pd.DataFrame, cell_preds: dict, cell_key: str | None = None,
                           scheme: str = "3", n_dur_bins=5, n_wps_bins=5,
                           wps_clip=10.0, mode="quantile"):
    """Heatmap of macro-F1 per (duration × WPS) bin. Cell=None → best-cell envelope."""
    import matplotlib.pyplot as plt
    dur_edges, wps_edges, f1, cnt, chosen = f1_topography(
        df, cell_preds, cell_key=cell_key, scheme=scheme,
        n_dur_bins=n_dur_bins, n_wps_bins=n_wps_bins, wps_clip=wps_clip, mode=mode)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(f1, origin="lower", aspect="auto",
                   extent=[dur_edges[0], dur_edges[-1], wps_edges[0], wps_edges[-1]],
                   cmap="RdYlGn", vmin=0.0, vmax=1.0)
    for i in range(n_wps_bins):
        for j in range(n_dur_bins):
            if cnt[i, j] == 0:
                continue
            x = 0.5 * (dur_edges[j] + dur_edges[j + 1])
            y = 0.5 * (wps_edges[i] + wps_edges[i + 1])
            ck = chosen[i, j] if cell_key is None else cell_key
            label = f"{f1[i, j]:.2f}\nn={cnt[i, j]}"
            if cell_key is None and ck is not None:
                # abbreviate "xlm-r-parla__SPR" -> "parla×SPR"
                bb, mix = ck.split("__", 1)
                short = bb.replace("xlm-roberta-", "").replace("xlm-r-", "")
                label += f"\n{short}×{mix}"
            ax.text(x, y, label, ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, label=f"macro-F1 ({scheme}-cls)")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("WPS (clipped)")
    title_cell = cell_key if cell_key is not None else "best-cell envelope"
    ax.set_title(f"F1 topography ({scheme}-cls) — {title_cell}  "
                 f"({n_dur_bins}×{n_wps_bins} {mode} bins)")
    fig.tight_layout()
    return fig


def plot_f1_topography_3d(df: pd.DataFrame, cell_preds: dict, cell_key: str | None = None,
                           scheme: str = "3", n_dur_bins=5, n_wps_bins=5,
                           wps_clip=10.0, mode="quantile"):
    """3-D surface of macro-F1 over (duration × WPS). Cell=None → best-cell envelope."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    dur_edges, wps_edges, f1, cnt, _ = f1_topography(
        df, cell_preds, cell_key=cell_key, scheme=scheme,
        n_dur_bins=n_dur_bins, n_wps_bins=n_wps_bins, wps_clip=wps_clip, mode=mode)

    dur_centres = 0.5 * (dur_edges[:-1] + dur_edges[1:])
    wps_centres = 0.5 * (wps_edges[:-1] + wps_edges[1:])
    X, Y = np.meshgrid(dur_centres, wps_centres)

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, f1, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                           edgecolor="#333", linewidth=0.3, alpha=0.95)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("WPS")
    ax.set_zlabel(f"macro-F1 ({scheme}-cls)")
    ax.set_zlim(0, 1)
    ax.view_init(elev=22, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.65, label="macro-F1")
    title_cell = cell_key if cell_key is not None else "best-cell envelope"
    ax.set_title(f"F1 topography (3-D, {scheme}-cls) — {title_cell}")
    fig.tight_layout()
    return fig


def plot_sentiment_topography_3d(df: pd.DataFrame, n_dur_bins=5, n_wps_bins=5,
                                  wps_clip=10.0, mode="quantile"):
    """3-D surface (duration × WPS × mean sentiment ordinal). Companion to the 2-D heatmap.
    Empty cells render as NaN gaps — do not interpolate across them."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
    dur_edges, wps_edges, mean_sent, count = sentiment_topography(
        df, n_dur_bins=n_dur_bins, n_wps_bins=n_wps_bins, wps_clip=wps_clip, mode=mode)

    dur_centres = 0.5 * (dur_edges[:-1] + dur_edges[1:])
    wps_centres = 0.5 * (wps_edges[:-1] + wps_edges[1:])
    X, Y = np.meshgrid(dur_centres, wps_centres)

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, mean_sent, cmap="RdYlGn", vmin=1.0, vmax=4.0,
                           edgecolor="#333", linewidth=0.3, alpha=0.95)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("WPS")
    ax.set_zlabel("mean sentiment ord (0..5)")
    ax.set_zlim(0, 5)
    ax.view_init(elev=22, azim=-135)
    fig.colorbar(surf, ax=ax, shrink=0.65, label="mean sentiment ord")
    ax.set_title(f"Sentiment topography (3-D) — {mode} bins, n={len(df)}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Random listen helpers
# ---------------------------------------------------------------------------

def matrix_grid_md(instance_id: str, cell_preds: dict) -> str:
    """Markdown grid of pred_3 (✓/✗) per (backbone, mixture) for one instance.
    Returns a short string if the instance is not in the matrix test set."""
    preds = cell_preds.get(instance_id, [])
    if not preds:
        return "_not in 2 s-filtered test split — no matrix predictions_"
    by_cell = {p["cell"]: p for p in preds}
    header = "| backbone \\ mixture | " + " | ".join(MIXTURES) + " |"
    sep    = "|---|" + "|".join([":---:"] * len(MIXTURES)) + "|"
    body = []
    for bb in BACKBONES:
        cells_out = []
        for m in MIXTURES:
            p = by_cell.get(f"{bb}__{m}")
            if p is None:
                cells_out.append("—")
            else:
                mark = "✓" if p["correct_3"] else "✗"
                cells_out.append(f"{mark} `{p['pred_3']}`")
        body.append(f"| {bb} | " + " | ".join(cells_out) + " |")
    gold3 = by_cell[list(by_cell)[0]]["gold_3"]
    return (f"**matrix 3-cls predictions** (gold = `{gold3}`)\n\n"
            + "\n".join([header, sep] + body))


# ---------------------------------------------------------------------------
# Context-sweep envelope — 3-D F1 surface over (context W × target feature).
#
# The 170-instance OR-probe subset is tested across W ∈ {0, 10, 20, 30, 50} and
# also an iso (no-context) baseline. For each W, we want to see how F1 varies
# with a target-side feature (duration seconds, or target-text word count).
#
# Shape: baseline plane (iso F1_3, flat gray, see-through) plus a signal surface
# rising above it where context helps. Smoothing is cosmetic — the raw grid is
# always printed in a 2-D companion plot.
# ---------------------------------------------------------------------------


# The 2026-11-11 OR run used terse 3-cls labels; the 2026-11-17 runs use full words.
# Normalize everything to the new scheme so iso + ctx batches score against a shared label set.
_3CLS_ALIAS = {
    "neg": "negative",  "negative": "negative",
    "neu": "neutral",   "neutral":  "neutral",
    "pos": "positive",  "positive": "positive",
}


def load_or_run_predictions(run_dir: Path) -> pd.DataFrame:
    """Load one OR run's predictions.jsonl into a DataFrame with:
    instance_id, gold_3, pred_3, parse_ok, duration, text, words, wps, gold_6, pred_6.

    - 3-cls labels are normalized to the full-word scheme (neg → negative) so the
      2026-11-11 short-label batch and the 2026-11-17 long-label batches share a
      label set.
    - `wps` = words / duration when duration is a positive number; NaN otherwise
      (e.g. ParlaSent-SL rows carry duration=None — their WPS plot defaults to the
      1-D word-count envelope).
    - gold_6 / pred_6 are included for optional 6-cls envelopes; empty when the
      predictions came from a 3-cls run (parsed_label is a 3-cls token, no 6-cls).
    """
    rows = []
    with open(run_dir / "predictions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pred = r.get("parsed_label")
            gold3 = r.get("gold_sent3") or r.get("gold_for_scheme")
            gold6 = r.get("gold_sent6")
            scheme = r.get("class_scheme")
            dur = r.get("duration")
            text = r.get("text") or ""
            if pred is None:
                continue
            words = len(text.split())
            try:
                wps = words / dur if dur and dur > 0 else float("nan")
            except TypeError:
                wps = float("nan")
            # Only populate pred_6 if this run was 6-cls (otherwise parsed_label is 3-cls).
            pred_6 = pred if scheme == "6cls" else None
            rows.append({
                "instance_id": r["instance_id"],
                "gold_3": _3CLS_ALIAS.get(gold3, gold3),
                "pred_3": _3CLS_ALIAS.get(pred, pred) if scheme != "6cls" else None,
                "gold_6": gold6,
                "pred_6": pred_6,
                "parse_ok": r.get("parse_ok", True),
                "duration": dur,
                "text": text,
                "words": words,
                "wps": wps,
            })
    return pd.DataFrame(rows)


def _pick_run_dir(batch_dir: Path, model: str, shot: str, scheme: str) -> Path | None:
    """Find a run dir inside a batch matching model/shot/scheme — tolerant of
    02_ (OR-text-...) and 03_ (OR-text-ctx..., OR-text-ctxw20-...) naming."""
    for d in sorted(batch_dir.iterdir()):
        if not d.is_dir():
            continue
        n = d.name
        if model in n and shot in n and scheme in n:
            return d
    return None


def load_context_sweep(iso_batch: Path, ctx_batches: dict, model: str = "flash",
                       shot: str = "0shot", scheme: str = "3cls") -> dict:
    """Load predictions for the iso baseline plus every W context batch for one
    (model, shot, scheme) config.

    ctx_batches: {W_value: Path(batch_dir_for_that_W)}
    Returns {W (-1 for iso): DataFrame}. Iso is keyed by W = -1 so the dict iters in a
    natural "no-context → W=0 → W=10 → ..." order when sorted.
    """
    out = {}
    iso_run = _pick_run_dir(iso_batch, model, shot, scheme)
    if iso_run is None:
        raise RuntimeError(f"iso run not found in {iso_batch} for {model}/{shot}/{scheme}")
    out[-1] = load_or_run_predictions(iso_run)
    for W, bdir in ctx_batches.items():
        rd = _pick_run_dir(bdir, model, shot, scheme)
        if rd is None:
            raise RuntimeError(f"ctx W={W} run not found in {bdir} for {model}/{shot}/{scheme}")
        out[W] = load_or_run_predictions(rd)
    return out


def context_envelope_grid(sweep: dict, feature: str = "duration",
                          n_bins: int = 5, mode: str = "quantile",
                          scheme: str = "3", wps_clip: float = 10.0) -> tuple:
    """Build the (W, feature-bin) grid of macro-F1.

    feature ∈ {'duration', 'words', 'wps'}:
      - 'duration' uses target utterance length in seconds.
      - 'words' uses target word count (len(text.split())).
      - 'wps' uses words / duration (speech rate). Values are clipped at `wps_clip`
        for bin construction so the histogram tail (ROG has outliers up to ~20)
        doesn't collapse all signal into the lower buckets. NaN WPS rows (e.g.
        duration==0 or None) are dropped from both the bin-edges pool and the
        per-W grid.

    Bin edges are quantile edges over the UNION of all sweep rows (so the x-axis
    is identical across W slices and surfaces are comparable).

    Returns (W_values_sorted, bin_edges, F1_matrix[n_W, n_bins], count_matrix[n_W, n_bins]).
    """
    if feature not in ("duration", "words", "wps"):
        raise ValueError("feature must be 'duration', 'words', or 'wps'")
    if scheme != "3":
        raise ValueError("only 3-cls envelope is implemented (6-cls is in the sweep CSV but "
                         "the sparse 29-per-class slice gets too thin per bin)")
    labels3 = ["negative", "neutral", "positive"]

    def _feat_vec(df):
        if feature == "duration":
            return df["duration"].values
        if feature == "words":
            return df["words"].values
        # wps: clip at wps_clip; preserve NaN for the drop-mask below.
        v = df["wps"].values
        return np.where(np.isnan(v), np.nan, np.minimum(v, wps_clip))

    all_feat = []
    for df in sweep.values():
        v = _feat_vec(df)
        v = v[~np.isnan(v)] if feature == "wps" else v
        all_feat.extend(list(v))
    edges = _bin_edges(pd.Series(all_feat), n_bins, mode=mode)

    Ws = sorted(sweep.keys())
    F1 = np.full((len(Ws), n_bins), np.nan)
    cnt = np.zeros((len(Ws), n_bins), dtype=int)
    for w_idx, W in enumerate(Ws):
        df = sweep[W]
        feat = _feat_vec(df)
        gold = df["gold_3"].tolist()
        pred = df["pred_3"].tolist()
        valid = ~np.isnan(feat) if feature == "wps" else np.ones(len(feat), dtype=bool)
        for j in range(n_bins):
            lo, hi = edges[j], edges[j + 1] + (1e-6 if j == n_bins - 1 else 0)
            mask = valid & (feat >= lo) & (feat < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            g = [gold[i] for i in range(len(gold)) if mask[i]]
            p = [pred[i] for i in range(len(pred)) if mask[i]]
            cnt[w_idx, j] = n
            F1[w_idx, j] = _macro_f1(g, p, labels3)
    return Ws, edges, F1, cnt


def plot_context_envelope_3d(sweep: dict, feature: str = "duration",
                             n_bins: int = 5, mode: str = "quantile",
                             scheme: str = "3", title_suffix: str = "",
                             smooth_sigma: float = 1.2, finer_grid: int = 60,
                             wps_clip: float = 10.0):
    """3-D surface of macro-F1 over (context-W × target-feature bin). Flat transparent
    gray baseline plane is drawn at iso F1 (W=-1) averaged across bins — signal piercing
    through the plane = context helped.

    smooth_sigma / finer_grid: cosmetic smoothing — the raw grid is drawn as a scatter
    with vertical stems so the eye can still see the 6×5 anchor points.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    Ws, edges, F1, cnt = context_envelope_grid(sweep, feature=feature, n_bins=n_bins,
                                               mode=mode, scheme=scheme, wps_clip=wps_clip)
    iso_f1_per_bin = F1[Ws.index(-1)]   # iso row
    iso_mean = float(np.nanmean(iso_f1_per_bin))

    # X axis display: use an ordinal index so iso (-1) and W=0..50 sit side by side.
    W_display = [max(w, 0) for w in Ws]
    W_labels = ["iso" if w == -1 else f"W={w}" for w in Ws]
    x_raw = np.arange(len(Ws), dtype=float)
    y_raw = 0.5 * (edges[:-1] + edges[1:])
    X_raw, Y_raw = np.meshgrid(x_raw, y_raw, indexing="ij")

    # Interpolate + smooth to a finer grid for the "soft hills" surface.
    F1_filled = np.where(np.isnan(F1), iso_mean, F1)   # fill NaNs with iso so hills don't dip
    interp = RegularGridInterpolator((x_raw, y_raw), F1_filled, method="linear",
                                     bounds_error=False, fill_value=iso_mean)
    xf = np.linspace(x_raw.min(), x_raw.max(), finer_grid)
    yf = np.linspace(y_raw.min(), y_raw.max(), finer_grid)
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    Zf = interp(np.stack([Xf.ravel(), Yf.ravel()], axis=-1)).reshape(Xf.shape)
    Zf = gaussian_filter(Zf, sigma=smooth_sigma)

    # Colormap vmin/vmax: anchor vmin to iso floor and vmax above the data peak so the
    # dynamic range inside the panel maps across the full colormap — stops the surface
    # from reading as a flat yellow-green.
    vmin = max(0.0, iso_mean - 0.10)
    vmax = min(1.0, max(float(np.nanmax(F1)) + 0.05, iso_mean + 0.25))
    zlo  = max(0.0, iso_mean - 0.15)
    zhi  = min(1.0, vmax + 0.02)

    fig = plt.figure(figsize=(11.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    # Signal surface.
    surf = ax.plot_surface(Xf, Yf, Zf, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                           edgecolor="none", linewidth=0, alpha=0.92,
                           rstride=1, cstride=1)

    # Baseline plane (transparent gray) at iso mean F1. Slight edge ring so it's visible
    # from above even when the signal surface sits close to it.
    Zb = np.full_like(Xf, iso_mean)
    ax.plot_surface(Xf, Yf, Zb, color="#6a6a6a", alpha=0.20,
                    edgecolor="none", linewidth=0)

    # Raw grid anchor points with vertical stems to the baseline — makes the "pierce
    # through the floor" shape explicit.
    for i in range(len(Ws)):
        for j in range(n_bins):
            if cnt[i, j] == 0:
                continue
            x, y, z = x_raw[i], y_raw[j], F1[i, j]
            ax.plot([x, x], [y, y], [iso_mean, z],
                    color=("#3a7a3a" if z >= iso_mean else "#a04040"),
                    lw=0.8, alpha=0.75)
            ax.scatter([x], [y], [z], s=26,
                       c=[z], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       edgecolor="black", linewidth=0.4, depthshade=False)

    ax.set_xticks(x_raw)
    ax.set_xticklabels(W_labels, fontsize=8.5)
    y_unit = {"duration": "s", "words": "words", "wps": f"words/s, clip={wps_clip:g}"}[feature]
    ax.set_xlabel("context width"); ax.set_ylabel(f"target {feature} ({y_unit})")
    ax.set_zlabel(f"macro-F1 ({scheme}-cls)")
    ax.set_zlim(zlo, zhi)
    ax.view_init(elev=28, azim=-120)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, label="macro-F1")
    _ = cbar
    ax.set_title(f"Context envelope — F1 vs (W, target {feature})  "
                 f"baseline={iso_mean:.3f}{('  ·  ' + title_suffix) if title_suffix else ''}")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig, (Ws, edges, F1, cnt, iso_mean)


# ---------------------------------------------------------------------------
# Single-run topography — one OR/local predictions.jsonl, bin by target feature(s).
#
# Phase 2 use: stock XLM-R-parlasent on full-ROG (3-D over duration × WPS) and on
# ParlaSent-SL (1-D over word count, since ParlaSent has no duration). Both consume
# the predictions.jsonl schema produced by 02_/04_, via load_or_run_predictions.
# ---------------------------------------------------------------------------


def predictions_topography_grid(df: pd.DataFrame, n_dur_bins: int = 5,
                                 n_wps_bins: int = 5, wps_clip: float = 10.0,
                                 mode: str = "quantile", scheme: str = "3") -> tuple:
    """Bin one predictions DataFrame by (duration, WPS) and compute per-bin macro-F1.

    Expects columns: duration, wps, gold_3, pred_3 (as emitted by
    load_or_run_predictions). Rows with NaN duration or NaN wps are dropped
    (e.g. ParlaSent-SL rows, which have no duration — those use
    `predictions_envelope_1d` instead).

    Returns (dur_edges, wps_edges, f1_matrix[n_wps, n_dur], count_matrix[n_wps, n_dur]).
    """
    if scheme != "3":
        raise ValueError("only 3-cls topography is implemented")
    labels3 = ["negative", "neutral", "positive"]

    mask = df["duration"].notna() & df["wps"].notna() & df["gold_3"].notna() & df["pred_3"].notna()
    d = df[mask].copy()
    if d.empty:
        raise RuntimeError("no rows with duration+wps+gold_3+pred_3; this predictions file "
                           "may be from a ParlaSent-style corpus (no duration). Use "
                           "`predictions_envelope_1d` for 1-D word-count envelopes.")
    d["wps_c"] = np.minimum(d["wps"].values, wps_clip)

    dur_edges = _bin_edges(d["duration"], n_dur_bins, mode=mode)
    wps_edges = _bin_edges(d["wps_c"], n_wps_bins, mode=mode)

    f1 = np.full((n_wps_bins, n_dur_bins), np.nan)
    cnt = np.zeros((n_wps_bins, n_dur_bins), dtype=int)
    dur_v = d["duration"].values
    wps_v = d["wps_c"].values
    g = d["gold_3"].tolist()
    p = d["pred_3"].tolist()
    for i in range(n_wps_bins):
        in_wps = (wps_v >= wps_edges[i]) & (wps_v < wps_edges[i + 1] + (1e-6 if i == n_wps_bins - 1 else 0))
        for j in range(n_dur_bins):
            in_dur = (dur_v >= dur_edges[j]) & (dur_v < dur_edges[j + 1] + (1e-6 if j == n_dur_bins - 1 else 0))
            m = in_wps & in_dur
            n = int(m.sum())
            if n == 0:
                continue
            gi = [g[k] for k in range(len(g)) if m[k]]
            pi = [p[k] for k in range(len(p)) if m[k]]
            cnt[i, j] = n
            f1[i, j] = _macro_f1(gi, pi, labels3)
    return dur_edges, wps_edges, f1, cnt


def plot_predictions_topography_3d(df: pd.DataFrame, title: str = "",
                                    n_dur_bins: int = 5, n_wps_bins: int = 5,
                                    wps_clip: float = 10.0, mode: str = "quantile",
                                    scheme: str = "3"):
    """3-D surface of per-(duration × WPS)-bin macro-F1 for one predictions run.

    Companion to `plot_context_envelope_3d` — same visual grammar (RdYlGn surface
    anchored to the data range, raw grid points with vertical stems to a baseline
    plane) but the baseline plane is the ROW's overall F1_3 instead of an iso
    baseline from a separate run.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    labels3 = ["negative", "neutral", "positive"]
    dur_edges, wps_edges, f1, cnt = predictions_topography_grid(
        df, n_dur_bins=n_dur_bins, n_wps_bins=n_wps_bins, wps_clip=wps_clip,
        mode=mode, scheme=scheme)

    # Overall F1 = baseline plane.
    mask = df["duration"].notna() & df["wps"].notna() & df["gold_3"].notna() & df["pred_3"].notna()
    d = df[mask]
    overall = _macro_f1(d["gold_3"].tolist(), d["pred_3"].tolist(), labels3)

    dur_centres = 0.5 * (dur_edges[:-1] + dur_edges[1:])
    wps_centres = 0.5 * (wps_edges[:-1] + wps_edges[1:])
    X, Y = np.meshgrid(dur_centres, wps_centres)

    f1_filled = np.where(np.isnan(f1), overall, f1)
    vmin = max(0.0, overall - 0.10)
    vmax = min(1.0, max(float(np.nanmax(f1)) + 0.05, overall + 0.25))

    fig = plt.figure(figsize=(10.5, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, f1_filled, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                           edgecolor="#333", linewidth=0.3, alpha=0.92)
    Zb = np.full_like(X, overall)
    ax.plot_surface(X, Y, Zb, color="#6a6a6a", alpha=0.20, edgecolor="none")
    for i in range(n_wps_bins):
        for j in range(n_dur_bins):
            if cnt[i, j] == 0:
                continue
            x, y, z = dur_centres[j], wps_centres[i], f1[i, j]
            ax.plot([x, x], [y, y], [overall, z],
                    color=("#3a7a3a" if z >= overall else "#a04040"),
                    lw=0.8, alpha=0.75)
            ax.scatter([x], [y], [z], s=28,
                       c=[z], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       edgecolor="black", linewidth=0.4, depthshade=False)
    ax.set_xlabel("duration (s)"); ax.set_ylabel(f"WPS (clip={wps_clip:g})")
    ax.set_zlabel(f"macro-F1 ({scheme}-cls)")
    ax.set_zlim(max(0.0, overall - 0.15), min(1.0, vmax + 0.02))
    ax.view_init(elev=28, azim=-120)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="macro-F1")
    ax.set_title(f"{title}  (baseline F1={overall:.3f}, n={int(cnt.sum())})")
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig, (dur_edges, wps_edges, f1, cnt, overall)


def predictions_envelope_1d(df: pd.DataFrame, feature: str = "words",
                             n_bins: int = 5, mode: str = "quantile",
                             scheme: str = "3", wps_clip: float = 10.0) -> tuple:
    """1-D envelope: macro-F1 across bins of one target feature. Use for ParlaSent-SL
    (no duration → the 3-D topography has nothing to bin on, so we fall back to
    word-count bins). feature ∈ {'duration', 'words', 'wps'}.

    Returns (edges, f1_per_bin, count_per_bin, overall_f1).
    """
    if scheme != "3":
        raise ValueError("only 3-cls envelope is implemented")
    if feature not in ("duration", "words", "wps"):
        raise ValueError("feature must be 'duration', 'words', or 'wps'")
    labels3 = ["negative", "neutral", "positive"]

    mask = df["gold_3"].notna() & df["pred_3"].notna()
    if feature == "duration":
        mask &= df["duration"].notna()
        feat = df["duration"].values
    elif feature == "words":
        feat = df["words"].values
    else:  # wps
        mask &= df["wps"].notna()
        feat = np.minimum(df["wps"].values, wps_clip)

    d_idx = np.where(mask.values)[0]
    feat = feat[d_idx]
    g = df["gold_3"].values[d_idx].tolist()
    p = df["pred_3"].values[d_idx].tolist()
    if len(g) == 0:
        raise RuntimeError(f"no rows with {feature}+gold_3+pred_3 for 1-D envelope")

    edges = _bin_edges(pd.Series(feat), n_bins, mode=mode)
    overall = _macro_f1(g, p, labels3)

    f1 = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    for j in range(n_bins):
        lo, hi = edges[j], edges[j + 1] + (1e-6 if j == n_bins - 1 else 0)
        m = (feat >= lo) & (feat < hi)
        n = int(m.sum())
        if n == 0:
            continue
        gi = [g[k] for k in range(len(g)) if m[k]]
        pi = [p[k] for k in range(len(p)) if m[k]]
        cnt[j] = n
        f1[j] = _macro_f1(gi, pi, labels3)
    return edges, f1, cnt, overall


def plot_predictions_envelope_1d(df: pd.DataFrame, feature: str = "words",
                                  title: str = "",
                                  n_bins: int = 5, mode: str = "quantile",
                                  scheme: str = "3", wps_clip: float = 10.0):
    """2-D line/bar envelope: macro-F1 vs feature-bin. Bar heights = per-bin F1;
    overlaid dashed line = overall F1 baseline. Annotate each bar with n per bin.
    """
    import matplotlib.pyplot as plt
    edges, f1, cnt, overall = predictions_envelope_1d(
        df, feature=feature, n_bins=n_bins, mode=mode, scheme=scheme, wps_clip=wps_clip)
    centres = 0.5 * (edges[:-1] + edges[1:])
    widths = 0.9 * (edges[1:] - edges[:-1])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#3a7a3a" if (not np.isnan(v) and v >= overall) else
              ("#a04040" if not np.isnan(v) else "#bbb") for v in f1]
    ax.bar(centres, np.where(np.isnan(f1), 0, f1), width=widths,
           color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.axhline(overall, color="#555", ls="--", lw=1.2, label=f"overall F1={overall:.3f}")
    for x, y, n in zip(centres, f1, cnt):
        if n == 0:
            ax.text(x, 0.02, "n=0", ha="center", va="bottom", fontsize=8, color="#555")
        else:
            ax.text(x, y + 0.012, f"{y:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
    unit = {"duration": "s", "words": "words", "wps": f"words/s, clip={wps_clip:g}"}[feature]
    ax.set_xlabel(f"target {feature} ({unit})")
    ax.set_ylabel(f"macro-F1 ({scheme}-cls)")
    ax.set_ylim(0, min(1.0, max(0.3, float(np.nanmax(f1)) * 1.25 if not np.all(np.isnan(f1)) else 1.0)))
    ax.set_title(f"{title}  (baseline F1={overall:.3f}, n={int(cnt.sum())})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, (edges, f1, cnt, overall)


def random_sample(df: pd.DataFrame, n=5, seed=None) -> pd.DataFrame:
    """Return n random rows from df, seeded. Empty df returns empty."""
    import random as _random
    if df.empty:
        return df
    if seed is None:
        seed = _random.randrange(10_000_000)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    out = df.iloc[list(map(int, idx))].copy()
    out.attrs["seed"] = seed
    return out
