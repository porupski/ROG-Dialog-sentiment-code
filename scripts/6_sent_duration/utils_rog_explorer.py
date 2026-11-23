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


def label_dist_by_duration_band(df: pd.DataFrame,
                                 edges: list[float] = (0, 2, 5, 10, float("inf")),
                                 scheme: str = "6") -> pd.DataFrame:
    """Return a long-form DataFrame of sentiment counts by duration band.

    One row per (duration_band × label). Columns: band, label, count,
    pct_within_band (count / n in band), pct_of_label_total (count of this label
    in band / total count of this label across all bands).

    Used to see WHO the 2 s duration floor actually drops — is the <2 s mass
    neutral filler, or is the model losing real sentiment signal to the floor?
    """
    if scheme not in ("6", "3"):
        raise ValueError("scheme must be '6' or '3'")
    labels = LABELS_6 if scheme == "6" else ["negative", "neutral", "positive"]
    lab_col = "sentiment" if scheme == "6" else "sentiment_3"

    edges = list(edges)
    band_names = []
    for a, b in zip(edges[:-1], edges[1:]):
        band_names.append(f"<{b:g} s" if a == 0 else
                          (f"≥{a:g} s" if b == float("inf") else f"{a:g}–{b:g} s"))
    band_of = np.full(len(df), -1, dtype=int)
    dur = df["duration"].values
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (dur >= a) & (dur < b)
        band_of[mask] = i

    rows = []
    totals_per_label = df[lab_col].value_counts().to_dict()
    for i, name in enumerate(band_names):
        sub = df[band_of == i]
        n_band = len(sub)
        for lbl in labels:
            c = int((sub[lab_col] == lbl).sum())
            rows.append({
                "band": name,
                "label": lbl,
                "count": c,
                "pct_within_band": round(100.0 * c / n_band, 2) if n_band > 0 else 0.0,
                "pct_of_label_total": round(100.0 * c / totals_per_label.get(lbl, 1), 2),
            })
    return pd.DataFrame(rows)


def plot_label_dist_by_duration_band(df: pd.DataFrame,
                                      edges: list[float] = (0, 2, 5, 10, float("inf")),
                                      scheme: str = "6"):
    """100%-stacked bar — each duration band sums to 100%; colour segments show the
    sentiment-class composition inside that band. The 2 s floor shows up as the gap
    between the leftmost bar (<2 s, dropped by the matrix's 2 s-filtered training)
    and the rest.

    Returns (fig, dist_df). dist_df is the long-form table from
    `label_dist_by_duration_band` (useful for the companion printed table).
    """
    import matplotlib.pyplot as plt
    labels = LABELS_6 if scheme == "6" else ["negative", "neutral", "positive"]
    dist = label_dist_by_duration_band(df, edges=edges, scheme=scheme)
    band_order = list(dict.fromkeys(dist["band"].tolist()))  # preserve insertion order

    # Diverging-ordinal palette: red = negative, yellow = neutral, green = positive.
    # 6-cls gets a smoother ramp; 3-cls uses three stops.
    if scheme == "6":
        palette = ["#8b0000", "#d8512d", "#f0b25e", "#aed07a", "#4faa54", "#206020"]
    else:
        palette = ["#c0392b", "#f1c40f", "#2ecc71"]

    totals = {b: dist[dist["band"] == b]["count"].sum() for b in band_order}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(band_order))
    x = np.arange(len(band_order))
    for lbl, color in zip(labels, palette):
        heights = []
        for b in band_order:
            row = dist[(dist["band"] == b) & (dist["label"] == lbl)].iloc[0]
            heights.append(row["pct_within_band"])
        heights = np.array(heights)
        bars = ax.bar(x, heights, bottom=bottom, color=color, edgecolor="black",
                      linewidth=0.4, label=lbl, width=0.78)
        for xi, h, bo in zip(x, heights, bottom):
            if h >= 4.0:  # only annotate segments that are big enough to read
                ax.text(xi, bo + h / 2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color=("white" if lbl in ("predominantlyNegative", "mixedNegative",
                                                   "predominantlyPositive", "negative") else "black"))
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={totals[b]})" for b in band_order])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% within duration band")
    ax.set_title(f"ROG v2 — sentiment-class composition by duration band ({scheme}-cls)")
    ax.legend(title=f"{scheme}-cls label", loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, title_fontsize=9, frameon=False)
    fig.tight_layout()
    return fig, dist


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
                          n_bins: int = 5, mode: str = "auto_p",
                          scheme: str = "3", wps_clip: float = 10.0,
                          edges_override: "list | np.ndarray | None" = None) -> tuple:
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
    is identical across W slices and surfaces are comparable). `edges_override`
    overrides with explicit edges when the quantile scheme collapses (e.g.
    words: 46% of ROG is 1-word, so quantile edges pile up at 1 and lose the
    mid-range resolution Ivan wants between 6 and 25).

    Returns (W_values_sorted, bin_edges, F1_matrix[n_W, n_bins], count_matrix[n_W, n_bins]).
    """
    if feature not in ("duration", "words", "wps"):
        raise ValueError("feature must be 'duration', 'words', or 'wps'")
    if scheme == "3":
        labels_cls = ["negative", "neutral", "positive"]
        gold_col, pred_col = "gold_3", "pred_3"
    elif scheme == "6":
        labels_cls = LABELS_6
        gold_col, pred_col = "gold_6", "pred_6"
    else:
        raise ValueError(f"scheme must be '3' or '6', got {scheme!r}")

    def _feat_vec(df):
        if feature == "duration":
            return df["duration"].values
        if feature == "words":
            return df["words"].values
        # wps: clip at wps_clip; preserve NaN for the drop-mask below.
        v = df["wps"].values
        return np.where(np.isnan(v), np.nan, np.minimum(v, wps_clip))

    if edges_override is not None:
        edges = np.asarray(edges_override, dtype=float)
    elif mode == "auto_p":
        edges = _auto_bin_edges_p(sweep, feature, _feat_vec, gold_col, pred_col,
                                  labels_cls)
    else:
        all_feat = []
        for df in sweep.values():
            v = _feat_vec(df)
            v = v[~np.isnan(v)] if feature == "wps" else v
            all_feat.extend(list(v))
        edges = _bin_edges(pd.Series(all_feat), n_bins, mode=mode)
    n_bins_eff = len(edges) - 1

    Ws = sorted(sweep.keys())
    F1 = np.full((len(Ws), n_bins_eff), np.nan)
    cnt = np.zeros((len(Ws), n_bins_eff), dtype=int)
    for w_idx, W in enumerate(Ws):
        df = sweep[W]
        feat = _feat_vec(df)
        gold = df[gold_col].tolist()
        pred = df[pred_col].tolist()
        valid = ~np.isnan(feat) if feature == "wps" else np.ones(len(feat), dtype=bool)
        for j in range(n_bins_eff):
            lo, hi = edges[j], edges[j + 1] + (1e-6 if j == n_bins_eff - 1 else 0)
            mask = valid & (feat >= lo) & (feat < hi)
            # Skip rows with None gold/pred (6-cls has sparse-column NaNs).
            g = [gold[i] for i in range(len(gold))
                 if mask[i] and gold[i] is not None and pred[i] is not None]
            p = [pred[i] for i in range(len(pred))
                 if mask[i] and gold[i] is not None and pred[i] is not None]
            n = len(g)
            if n == 0:
                continue
            cnt[w_idx, j] = n
            F1[w_idx, j] = _macro_f1(g, p, labels_cls)
    return Ws, edges, F1, cnt


def _auto_bin_edges_p(sweep: dict, feature: str, feat_vec_fn,
                      gold_col: str, pred_col: str, labels_cls: list,
                      min_count_per_bin: int = 30,
                      p_threshold: float = 0.05) -> np.ndarray:
    """Greedy autoselect of bin edges. Start fine, merge adjacent bins when
    any bin fails either constraint:
      - `n < min_count_per_bin` (UNION count across all Ws), or
      - chi-square independence test of (gold × pred) returns p >= p_threshold.

    Returns the finest edge array that passes both gates.
    """
    from scipy.stats import chi2_contingency

    # Union feature vector + gold/pred across all Ws.
    rows = []
    for df in sweep.values():
        v = feat_vec_fn(df)
        g = df[gold_col].tolist()
        p = df[pred_col].tolist()
        for fv, gv, pv in zip(v, g, p):
            if gv is None or pv is None:
                continue
            if feature == "wps" and (isinstance(fv, float) and np.isnan(fv)):
                continue
            rows.append((float(fv), gv, pv))
    if not rows:
        return np.array([0.0, 1.0])
    rows.sort(key=lambda r: r[0])
    xs = np.array([r[0] for r in rows])

    # Candidate edges: unique feature values (integers for `words`, finer for wps).
    if feature == "words":
        cands = sorted(set(int(round(x)) for x in xs))
        edges = np.array(cands + [cands[-1] + 1], dtype=float)
    else:
        q = np.linspace(0, 1, 15)
        edges = np.unique(np.quantile(xs, q))

    def _eval(edges):
        out = []
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1] + (1e-6 if j == len(edges) - 2 else 0)
            rs = [r for r in rows if lo <= r[0] < hi]
            n = len(rs)
            if n < 2:
                out.append((n, 1.0))
                continue
            # Chi-square on confusion matrix.
            ct = {}
            for _, gv, pv in rs:
                ct[(gv, pv)] = ct.get((gv, pv), 0) + 1
            table = np.array([[ct.get((g, p), 0) for p in labels_cls]
                              for g in labels_cls], dtype=float)
            if table.sum(0).min() == 0 or table.sum(1).min() == 0:
                out.append((n, 1.0))
                continue
            try:
                _, pval, _, _ = chi2_contingency(table)
            except Exception:
                pval = 1.0
            out.append((n, float(pval)))
        return out

    # Greedy merge until all bins pass. Merge the worst-offending bin with its
    # smaller-count neighbour; break ties by merging rightward.
    while len(edges) > 3:
        stats = _eval(edges)
        offenders = [(j, n, pval) for j, (n, pval) in enumerate(stats)
                     if n < min_count_per_bin or pval >= p_threshold]
        if not offenders:
            break
        # Pick the offender with the lowest count.
        offenders.sort(key=lambda t: (t[1], t[2]))
        j = offenders[0][0]
        if j == 0:
            new_edges = np.concatenate([edges[:1], edges[2:]])
        elif j == len(edges) - 2:
            new_edges = np.concatenate([edges[:-2], edges[-1:]])
        else:
            n_left = stats[j - 1][0]
            n_right = stats[j + 1][0]
            if n_left <= n_right:
                new_edges = np.concatenate([edges[:j], edges[j + 1:]])
            else:
                new_edges = np.concatenate([edges[:j + 1], edges[j + 2:]])
        if len(new_edges) == len(edges):
            break
        edges = new_edges
    return edges


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


def plot_context_envelope_3d_v2(sweep: dict, feature: str = "duration",
                                n_bins: int = 5, mode: str = "quantile",
                                scheme: str = "3", title_suffix: str = "",
                                smooth_sigma: float = 1.2, finer_grid: int = 100,
                                wps_clip: float = 10.0,
                                zlim: tuple | None = None,
                                elev: float = 28, azim: float = -120,
                                edges_override: "list | np.ndarray | None" = None,
                                iaa_ceiling: float | None = None,
                                iaa_label: str = "IAA ceiling",
                                water_color: str = "#5dade2",
                                ceiling_color: str = "#d7a4b3",
                                intersection_color: str = "#00e5ff",
                                ceiling_intersection_color: str = "#ff2ea6",
                                draw_intersection: bool = True):
    """Overhauled context-envelope renderer.

    2x3 figure: main 3-D occupies the left 2x2 cells; two 1-D profiles sit on
    the right column (top = F1 vs feature, one line per W; bottom = F1 vs
    context-W, one line per feature-bin). Baseline plane sits at iso F1 and
    behind the signal surface (water metaphor — mountains stick up through it).
    Vertical stems from the floor up to each anchor point let the eye read off
    (W, feature) coordinates. Intersection between signal and baseline is
    picked out with a gold contour.

    `zlim=(lo, hi)` forces the F1 axis range — used to match z-axis across
    paired plots (e.g. 3cls words vs 6cls words).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import gridspec
    from matplotlib.colors import LightSource, Normalize
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    Ws, edges, F1, cnt = context_envelope_grid(sweep, feature=feature, n_bins=n_bins,
                                               mode=mode, scheme=scheme, wps_clip=wps_clip,
                                               edges_override=edges_override)
    n_bins_eff = len(edges) - 1
    iso_f1_per_bin = F1[Ws.index(-1)]
    iso_mean = float(np.nanmean(iso_f1_per_bin))

    W_labels = ["iso" if w == -1 else f"W={w}" for w in Ws]
    x_raw = np.arange(len(Ws), dtype=float)
    y_raw = 0.5 * (edges[:-1] + edges[1:])

    # Fine grid for a soft surface.
    F1_filled = np.where(np.isnan(F1), iso_mean, F1)
    interp = RegularGridInterpolator((x_raw, y_raw), F1_filled, method="linear",
                                     bounds_error=False, fill_value=iso_mean)
    xf = np.linspace(x_raw.min(), x_raw.max(), finer_grid)
    yf = np.linspace(y_raw.min(), y_raw.max(), finer_grid)
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    Zf = interp(np.stack([Xf.ravel(), Yf.ravel()], axis=-1)).reshape(Xf.shape)
    Zf = gaussian_filter(Zf, sigma=smooth_sigma)

    # Color + Z normalisation.
    vmin = max(0.0, iso_mean - 0.10)
    vmax = min(1.0, max(float(np.nanmax(F1)) + 0.05, iso_mean + 0.25))
    if zlim is None:
        zlo = max(0.0, iso_mean - 0.15)
        zhi = min(1.0, vmax + 0.02)
    else:
        zlo, zhi = zlim

    # Figure + GridSpec. Wider canvas so legends can live outside the side
    # panels (bbox_to_anchor spills right of each axes); savefig with
    # bbox_inches="tight" catches them.
    fig = plt.figure(figsize=(16.5, 7.8))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 0.85],
                           hspace=0.32, wspace=0.28,
                           left=0.04, right=0.85, top=0.92, bottom=0.08)
    ax_main = fig.add_subplot(gs[:, :2], projection="3d")
    # Disable matplotlib's centroid-based painter sort — explicit zorder on
    # each artist is honoured instead. Required to keep the water/ceiling
    # planes above or below the signal surface consistently.
    try:
        ax_main.computed_zorder = False
    except Exception:
        pass
    ax_feat = fig.add_subplot(gs[0, 2])
    ax_ctx  = fig.add_subplot(gs[1, 2])

    # Signal surface — RdYlGn colormap over the smoothed Zf, shaded with a
    # top-left light source for relief.
    ls = LightSource(azdeg=315, altdeg=55)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap("RdYlGn")(norm(Zf))
    rgb = rgba[..., :3]
    try:
        shaded = ls.shade_rgb(rgb, Zf, vert_exag=0.5, blend_mode="soft")
    except Exception:
        shaded = rgb
    facecolors = np.concatenate([shaded, np.ones(shaded.shape[:2] + (1,))], axis=-1)

    # Per-polygon cell centres for the water/ceiling alpha masks. plot_surface
    # with rstride=cstride=1 emits one polygon per (Nx-1, Ny-1) grid cell; its
    # colour is taken from the (i, j) vertex by default. Using per-polygon
    # facecolors with alpha=0 for out-of-region cells avoids the NaN-masked
    # approach, which dropped mixed-sign polygons at the crossing and left
    # visible gaps where the signal oscillates across the baseline.
    Zf_centre = 0.25 * (Zf[:-1, :-1] + Zf[1:, :-1] + Zf[:-1, 1:] + Zf[1:, 1:])
    water_rgb = mcolors.to_rgb(water_color)
    ceiling_rgb = mcolors.to_rgb(ceiling_color)

    def _plane_facecolors(mask: np.ndarray, rgb: tuple, alpha: float) -> np.ndarray:
        fc = np.zeros(mask.shape + (4,), dtype=float)
        fc[..., :3] = rgb
        fc[..., 3] = np.where(mask, alpha, 0.0)
        return fc

    Zplane = np.full_like(Xf, iso_mean)

    # Water background: cells where the signal is above baseline. Drawn
    # BEFORE the signal surface and fainter — reads as the water line behind
    # the mountain silhouette.
    water_bg_fc = _plane_facecolors(Zf_centre > iso_mean, water_rgb, 0.26)
    ax_main.plot_surface(Xf, Yf, Zplane, facecolors=water_bg_fc,
                         rstride=1, cstride=1,
                         edgecolor="none", linewidth=0, antialiased=True,
                         shade=False, zorder=1)

    # Ceiling background (where signal has broken through the ceiling).
    if iaa_ceiling is not None and zlo <= iaa_ceiling <= zhi:
        Zceil = np.full_like(Xf, iaa_ceiling)
        ceiling_bg_fc = _plane_facecolors(Zf_centre > iaa_ceiling, ceiling_rgb, 0.22)
        ax_main.plot_surface(Xf, Yf, Zceil, facecolors=ceiling_bg_fc,
                             rstride=1, cstride=1,
                             edgecolor="none", linewidth=0, antialiased=True,
                             shade=False, zorder=2)

    # Signal surface itself.
    surf = ax_main.plot_surface(Xf, Yf, Zf, facecolors=facecolors,
                                rstride=1, cstride=1,
                                edgecolor="none", linewidth=0, antialiased=True,
                                shade=False, zorder=5)

    # Water foreground — covers the submerged region (cells where the signal
    # is at or below baseline).
    water_fg_fc = _plane_facecolors(Zf_centre <= iso_mean, water_rgb, 0.58)
    ax_main.plot_surface(Xf, Yf, Zplane, facecolors=water_fg_fc,
                         rstride=1, cstride=1,
                         edgecolor="none", linewidth=0, antialiased=True,
                         shade=False, zorder=12)

    # Ceiling foreground — sits above the signal surface wherever the signal
    # is below the ceiling.
    if iaa_ceiling is not None and zlo <= iaa_ceiling <= zhi:
        Zceil = np.full_like(Xf, iaa_ceiling)
        ceiling_fg_fc = _plane_facecolors(Zf_centre <= iaa_ceiling, ceiling_rgb, 0.40)
        ax_main.plot_surface(Xf, Yf, Zceil, facecolors=ceiling_fg_fc,
                             rstride=1, cstride=1,
                             edgecolor="none", linewidth=0, antialiased=True,
                             shade=False, zorder=14)

    # Intersection contours drawn as Line3D objects (ax.plot) rather than
    # ax.contour(offset=…). matplotlib's 3-D sort handles Line3D more
    # reliably than Poly3DCollection ties, so the highlight stays on top.
    if draw_intersection:
        try:
            import matplotlib.patheffects as pe
            import matplotlib.pyplot as _plt
            # Baseline crossing.
            _tmp_fig, _tmp_ax = _plt.subplots()
            _cs = _tmp_ax.contour(Xf, Yf, Zf, levels=[iso_mean])
            z_lift_base = iso_mean + 0.004 * max(zhi - zlo, 1e-6)
            try:
                segs_base = _cs.allsegs[0]
            except Exception:
                segs_base = []
            for seg in segs_base:
                if len(seg) < 2:
                    continue
                xs, ys = seg[:, 0], seg[:, 1]
                zs = np.full_like(xs, z_lift_base)
                ln, = ax_main.plot(xs, ys, zs, color=intersection_color,
                                   lw=2.6, zorder=40)
                ln.set_path_effects(
                    [pe.Stroke(linewidth=4.4, foreground="#101010", alpha=0.75),
                     pe.Normal()])
            _plt.close(_tmp_fig)

            # Ceiling crossing.
            if iaa_ceiling is not None and zlo <= iaa_ceiling <= zhi:
                _tmp_fig2, _tmp_ax2 = _plt.subplots()
                _cs2 = _tmp_ax2.contour(Xf, Yf, Zf, levels=[iaa_ceiling])
                z_lift_c = iaa_ceiling + 0.004 * max(zhi - zlo, 1e-6)
                try:
                    segs_c = _cs2.allsegs[0]
                except Exception:
                    segs_c = []
                for seg in segs_c:
                    if len(seg) < 2:
                        continue
                    xs, ys = seg[:, 0], seg[:, 1]
                    zs = np.full_like(xs, z_lift_c)
                    ln, = ax_main.plot(xs, ys, zs,
                                       color=ceiling_intersection_color,
                                       lw=2.4, zorder=42)
                    ln.set_path_effects(
                        [pe.Stroke(linewidth=4.2, foreground="#101010", alpha=0.75),
                         pe.Normal()])
                _plt.close(_tmp_fig2)
        except Exception:
            pass

    # Anchor pins: stem from the floor up to F1[i,j], marker AT the raw
    # (x, y, F1) point. Markers sit on the surface so the eye picks up each
    # cell's exact value; stems carry the sign-vs-iso cue.
    for i in range(len(Ws)):
        for j in range(n_bins_eff):
            if cnt[i, j] == 0:
                continue
            x, y, z = x_raw[i], y_raw[j], F1[i, j]
            ax_main.plot([x, x], [y, y], [zlo, z],
                         color=("#2e7d32" if z >= iso_mean else "#b73b3b"),
                         lw=0.6, alpha=0.55, zorder=15)
            ax_main.scatter([x], [y], [z], s=26,
                            c=[z], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                            edgecolor="black", linewidth=0.4,
                            depthshade=False, zorder=22)

    ax_main.set_xticks(x_raw)
    ax_main.set_xticklabels(W_labels, fontsize=9)
    y_unit = {"duration": "s", "words": "words", "wps": f"words/s, clip={wps_clip:g}"}[feature]
    ax_main.set_xlabel("context width"); ax_main.set_ylabel(f"target {feature} ({y_unit})")
    ax_main.set_zlabel(f"macro-F1 ({scheme}-cls)")
    ax_main.set_zlim(zlo, zhi)
    ax_main.view_init(elev=elev, azim=azim)
    # Words are integer-valued — pin the y ticks to integers so halves don't
    # show up on the y axis.
    if feature == "words":
        from matplotlib.ticker import MaxNLocator
        ax_main.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_main.set_title(f"Context envelope — F1 vs (W, target {feature})  "
                      f"baseline={iso_mean:.3f}"
                      f"{('  ·  ' + title_suffix) if title_suffix else ''}",
                      fontsize=11)

    # Colorbar for the main 3-D panel. ScalarMappable with the surface's cmap
    # + normalisation since plot_surface(facecolors=...) doesn't expose one.
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_main, shrink=0.55, pad=0.08,
                        label=f"macro-F1 ({scheme}-cls)")
    _ = cbar

    # ── Side profile A (top-right): F1 vs feature, one line per W ────────
    # X axis reversed so the direction matches the 3-D view (feature axis
    # goes "away from viewer" at this azimuth — small values on the right,
    # large values on the left as seen in the 3-D plot).
    cmap_W = plt.get_cmap("plasma")
    for i, W in enumerate(Ws):
        color = cmap_W(i / max(len(Ws) - 1, 1))
        valid = cnt[i] > 0
        if not valid.any():
            continue
        ax_feat.plot(y_raw[valid], F1[i, valid], marker="o", markersize=4, lw=1.7,
                     color=color, label=W_labels[i])
    ax_feat.axhline(iso_mean, color="#555", lw=1.0, ls="--", alpha=0.8,
                    label=f"iso mean = {iso_mean:.3f}")
    if iaa_ceiling is not None:
        ax_feat.axhline(iaa_ceiling, color=ceiling_color, lw=1.2, ls=":", alpha=0.95,
                        label=f"{iaa_label} = {iaa_ceiling:.3f}")
    ax_feat.set_xlabel(f"target {feature} ({y_unit})")
    ax_feat.set_ylabel("macro-F1")
    ax_feat.set_ylim(zlo, zhi)
    ax_feat.invert_xaxis()
    if feature == "words":
        from matplotlib.ticker import MaxNLocator
        ax_feat.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_feat.set_title(f"Profile — F1 vs {feature} (one line per W)", fontsize=10)
    ax_feat.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                   frameon=True, title="context")
    ax_feat.grid(True, alpha=0.25)

    # ── Side profile B (bottom-right): F1 vs context-W, one line per bin ──
    # Categorical palette (tab10) — distinct per bin, not a gradient (bins
    # are ordered but the per-line comparison is categorical).
    tab = plt.get_cmap("tab10")
    for j in range(n_bins_eff):
        color = tab(j % 10)
        valid = cnt[:, j] > 0
        if not valid.any():
            continue
        bin_label = f"{edges[j]:.1f}–{edges[j+1]:.1f}"
        ax_ctx.plot(x_raw[valid], F1[valid, j], marker="o", markersize=4, lw=1.7,
                    color=color, label=bin_label)
    ax_ctx.axhline(iso_mean, color="#555", lw=1.0, ls="--", alpha=0.8,
                   label=f"iso mean = {iso_mean:.3f}")
    if iaa_ceiling is not None:
        ax_ctx.axhline(iaa_ceiling, color=ceiling_color, lw=1.2, ls=":", alpha=0.95,
                       label=f"{iaa_label} = {iaa_ceiling:.3f}")
    ax_ctx.set_xticks(x_raw)
    ax_ctx.set_xticklabels(W_labels, fontsize=8, rotation=20)
    ax_ctx.set_xlabel("context width")
    ax_ctx.set_ylabel("macro-F1")
    ax_ctx.set_ylim(zlo, zhi)
    ax_ctx.set_title(f"Profile — F1 vs context (one line per {feature} bin)", fontsize=10)
    ax_ctx.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=True, title=f"{feature} bin ({y_unit})")
    ax_ctx.grid(True, alpha=0.25)

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
