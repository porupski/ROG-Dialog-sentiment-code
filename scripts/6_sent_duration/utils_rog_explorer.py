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
