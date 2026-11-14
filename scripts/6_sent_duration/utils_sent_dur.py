"""Shared data layer for the Phase-A sentiment × duration envelope study.

Responsibilities:
  - Load Phase-A config.json.
  - Enumerate cells in a 4_finetune_text matrix run and read their
    predictions_<src>.jsonl.
  - Join each prediction row with duration / word-count metadata from the
    source JSONL under data/ (via instance_id); fall back to text.split()
    when the source isn't on disk.
  - Fixed-boundary and equal-count quantile binners.
  - Per-bin metrics: F1_3 / F1_6 / R² / ρ / MAE / n.
  - Primary envelope plot (one line per backbone, best-mixture-avg per bin,
    self-labeled cells excluded) and a per-backbone facet audit plot.

The metrics code is a trimmed port of compute_metrics_from_preds in
scripts/4_finetune_text/utils_text_finetune.py, specialised for what the
envelope plot consumes.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Constants (mirror 4_finetune_text where it matters).
# ---------------------------------------------------------------------------

LABELS_6 = [
    "predominantlyNegative", "mixedNegative", "neutralNegative",
    "neutralPositive",       "mixedPositive", "predominantlyPositive",
]
LABELS_3 = ["negative", "neutral", "positive"]
SENT_6_TO_3_IDX = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}


def backbone_short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    bins_audio_sec: list[list[float]]
    bins_text_words: list[list[float]]
    test_source_axis: dict[str, list[str]]
    source_metadata: dict[str, dict]
    metrics_per_bin: list[str]
    primary_metric: str
    min_bin_n: int
    backbones_order: list[str]
    mixtures_order: list[str]
    self_labeled_backbone: str
    plot_layout: dict


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text())
    return Config(
        bins_audio_sec=raw["bins_audio_sec"],
        bins_text_words=raw["bins_text_words"],
        test_source_axis=raw["test_source_axis"],
        source_metadata=raw["source_metadata"],
        metrics_per_bin=raw["metrics_per_bin"],
        primary_metric=raw["primary_metric"],
        min_bin_n=raw["min_bin_n"],
        backbones_order=raw["backbones_order"],
        mixtures_order=raw["mixtures_order"],
        self_labeled_backbone=raw["self_labeled_backbone"],
        plot_layout=raw["plot_layout"],
    )


# ---------------------------------------------------------------------------
# Matrix-cell discovery
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    tag: str
    backbone_short: str
    mixture: str
    self_labeled: bool
    path: Path


def is_self_labeled(bb_short: str, mixture: str, self_labeled_backbone: str) -> bool:
    return bb_short == self_labeled_backbone and "P" in mixture


def discover_cells(matrix_dir: Path, cfg: Config) -> list[Cell]:
    """Walk <matrix_dir> for cell dirs that contain summary.json + at least one
    predictions_<src>.jsonl. A cell dir name is `<backbone_short>__<mixture>`."""
    cells: list[Cell] = []
    for sub in sorted(matrix_dir.iterdir()):
        if not sub.is_dir():
            continue
        if "__" not in sub.name:
            continue
        bb, mix = sub.name.split("__", 1)
        if bb not in cfg.backbones_order or mix not in cfg.mixtures_order:
            continue
        if not any(sub.glob("predictions_*.jsonl")):
            continue
        cells.append(Cell(
            tag=sub.name,
            backbone_short=bb,
            mixture=mix,
            self_labeled=is_self_labeled(bb, mix, cfg.self_labeled_backbone),
            path=sub,
        ))
    return cells


# ---------------------------------------------------------------------------
# Source-metadata index: {instance_id -> {"audio_sec": float|None, "text_words": int}}
# ---------------------------------------------------------------------------

def _word_count_from_text(text: str) -> int:
    return len(text.split()) if text else 0


def build_metadata_index(src_name: str, cfg: Config, root: Path) -> tuple[dict[str, dict], dict]:
    """Return (index, info). info['axes_supported'] lists what we can bin by for this src.
    If the source JSONL is missing, index is empty and callers will fall back to the
    predictions row's own text for word count; audio_sec binning becomes unavailable."""
    meta_cfg = cfg.source_metadata.get(src_name) or {}
    path = root / meta_cfg.get("path", "")
    info = {"path": str(path), "axes_supported": [], "missing": True, "n_rows": 0}
    index: dict[str, dict] = {}

    if not path or not path.exists():
        # Nothing we can look up; let the caller derive word_count from predictions.text.
        info["axes_supported"] = ["text_words"]
        return index, info

    info["missing"] = False
    audio_pair = meta_cfg.get("audio_field_pair")
    audio_len_field = meta_cfg.get("audio_length_field")

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row.get("instance_id") or row.get("id")
            if not iid:
                continue
            audio_sec = None
            if audio_pair:
                s = row.get(audio_pair[0])
                e = row.get(audio_pair[1])
                if s is not None and e is not None:
                    audio_sec = float(e) - float(s)
            elif audio_len_field and row.get(audio_len_field) is not None:
                audio_sec = float(row[audio_len_field])
            # Word count from the source's text, if present; fall back later.
            text = row.get("text") or row.get("sentence") or ""
            wc = _word_count_from_text(text)
            index[iid] = {"audio_sec": audio_sec, "text_words": wc}
            info["n_rows"] += 1

    axes = ["text_words"]
    has_audio = any(v["audio_sec"] is not None for v in index.values())
    if has_audio:
        axes.append("audio_sec")
    info["axes_supported"] = axes
    return index, info


# ---------------------------------------------------------------------------
# Joined prediction row
# ---------------------------------------------------------------------------

@dataclass
class PredRow:
    cell_tag: str
    backbone_short: str
    mixture: str
    self_labeled: bool
    test_source: str
    instance_id: str
    gold_scalar: float
    pred_scalar: float
    text_words: int
    audio_sec: float | None


def load_predictions_for_cell(
    cell: Cell, test_source: str, index: dict[str, dict]
) -> list[PredRow]:
    path = cell.path / f"predictions_{test_source}.jsonl"
    if not path.exists():
        return []
    rows: list[PredRow] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            iid = r.get("instance_id")
            if not iid:
                continue
            md = index.get(iid, {})
            audio_sec = md.get("audio_sec")
            text_words = md.get("text_words")
            if text_words is None:
                text_words = _word_count_from_text(r.get("text") or "")
            rows.append(PredRow(
                cell_tag=cell.tag,
                backbone_short=cell.backbone_short,
                mixture=cell.mixture,
                self_labeled=cell.self_labeled,
                test_source=test_source,
                instance_id=iid,
                gold_scalar=float(r["gold_scalar"]),
                pred_scalar=float(r["pred_scalar"]),
                text_words=int(text_words),
                audio_sec=audio_sec,
            ))
    return rows


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def assign_fixed_bin(value: float, bounds: list[list[float]]) -> int | None:
    """Half-open (lo, hi]. Returns the bin index or None if out of range."""
    for i, (lo, hi) in enumerate(bounds):
        if lo < value <= hi:
            return i
    return None


def derive_quantile_bounds(values: list[float], k: int = 5) -> list[list[float]]:
    """k equal-count bins from a list of values. Ascending, non-overlapping (lo, hi]."""
    if not values:
        return []
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return []
    n = len(vals)
    bounds: list[list[float]] = []
    prev_hi = vals[0] - 1e-9  # ensure (prev_hi, v]
    for q in range(1, k + 1):
        idx = min(int(round(q * n / k)) - 1, n - 1)
        hi = vals[idx]
        if hi <= prev_hi:
            hi = prev_hi + 1e-6
        bounds.append([prev_hi, hi])
        prev_hi = hi
    return bounds


def bin_center(lo: float, hi: float) -> float:
    if hi >= 999 and lo > 0:
        return lo + 5.0
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Metrics (on scalar predictions)
# ---------------------------------------------------------------------------

def _round6(y: float) -> int:
    return max(0, min(5, round(y)))


def _spearman(a: list[float], b: list[float]) -> float | None:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    if len(a) < 2:
        return None
    r = spearmanr(a, b).correlation
    if r is None or r != r:
        return None
    return float(r)


def metrics_for_bin(rows: list[PredRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "F1_3": None, "F1_6": None, "R2": None, "spearman": None, "MAE": None}
    g = [r.gold_scalar for r in rows]
    p = [r.pred_scalar for r in rows]
    mae = sum(abs(a - b) for a, b in zip(g, p)) / n
    mean_g = sum(g) / n
    ss_tot = sum((x - mean_g) ** 2 for x in g) or 1e-9
    ss_res = sum((x - y) ** 2 for x, y in zip(g, p))
    r2 = 1.0 - ss_res / ss_tot
    rho = _spearman(g, p)
    g6 = [_round6(x) for x in g]
    p6 = [_round6(x) for x in p]
    g3 = [SENT_6_TO_3_IDX[x] for x in g6]
    p3 = [SENT_6_TO_3_IDX[x] for x in p6]

    def macro_f1(y_true: list[int], y_pred: list[int], k: int) -> float:
        f1s = []
        for c in range(k):
            tp = sum(1 for t, q in zip(y_true, y_pred) if t == c and q == c)
            fp = sum(1 for t, q in zip(y_true, y_pred) if t != c and q == c)
            fn = sum(1 for t, q in zip(y_true, y_pred) if t == c and q != c)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return sum(f1s) / k

    return {
        "n": n,
        "F1_3": macro_f1(g3, p3, 3),
        "F1_6": macro_f1(g6, p6, 6),
        "R2": r2,
        "spearman": rho,
        "MAE": mae,
    }


# ---------------------------------------------------------------------------
# Per-cell per-bin computation
# ---------------------------------------------------------------------------

def rows_by_axis_value(rows: list[PredRow], axis: str) -> list[tuple[PredRow, float]]:
    """Attach the axis value to each row; skip rows where it's missing."""
    out = []
    for r in rows:
        v = r.text_words if axis == "text_words" else r.audio_sec
        if v is None:
            continue
        out.append((r, float(v)))
    return out


def bin_rows_fixed(rows_with_values: list[tuple[PredRow, float]],
                   bounds: list[list[float]]) -> dict[int, list[PredRow]]:
    out: dict[int, list[PredRow]] = {i: [] for i in range(len(bounds))}
    for r, v in rows_with_values:
        i = assign_fixed_bin(v, bounds)
        if i is None:
            continue
        out[i].append(r)
    return out


# ---------------------------------------------------------------------------
# Envelope: best-mixture-avg per bin, per backbone.
# ---------------------------------------------------------------------------

def envelope_best_mixture_avg(
    per_cell_metrics: dict[tuple[str, str], dict[int, dict]],
    backbones_order: list[str],
    primary_metric: str,
    n_bins: int,
    exclude_self_labeled: set[tuple[str, str]] | None = None,
) -> dict[str, list[float | None]]:
    """For each backbone, for each bin index, take the MAX of the primary metric across
    mixtures (excluding self-labeled cells). Returns {backbone -> [bin0_val, ...]}.
    Returns None for bins with no non-null entries."""
    exclude = exclude_self_labeled or set()
    out: dict[str, list[float | None]] = {bb: [] for bb in backbones_order}
    for bb in backbones_order:
        for b in range(n_bins):
            vals = []
            for (bb2, mix), binned in per_cell_metrics.items():
                if bb2 != bb:
                    continue
                if (bb2, mix) in exclude:
                    continue
                m = binned.get(b, {}).get(primary_metric)
                if m is None:
                    continue
                vals.append(m)
            out[bb].append(max(vals) if vals else None)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_envelope_primary(
    bin_centers: list[float],
    envelope: dict[str, list[float | None]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> Path | None:
    plt = _try_matplotlib()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    markers = {"xlm-roberta-large": "o", "xlm-r-parla": "s", "xlm-r-parlasent": "^"}
    for bb, ys in envelope.items():
        xs = [c for c, y in zip(bin_centers, ys) if y is not None]
        ys_f = [y for y in ys if y is not None]
        ax.plot(xs, ys_f, marker=markers.get(bb, "o"), label=bb)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_envelope_facet(
    bin_centers: list[float],
    per_cell_metrics: dict[tuple[str, str], dict[int, dict]],
    backbones_order: list[str],
    mixtures_order: list[str],
    primary_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    self_labeled: set[tuple[str, str]] | None = None,
) -> Path | None:
    plt = _try_matplotlib()
    if plt is None:
        return None
    self_labeled = self_labeled or set()
    n = len(backbones_order)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, bb in zip(axes, backbones_order):
        for mix in mixtures_order:
            binned = per_cell_metrics.get((bb, mix))
            if not binned:
                continue
            ys = [binned.get(b, {}).get(primary_metric) for b in range(len(bin_centers))]
            xs = [c for c, y in zip(bin_centers, ys) if y is not None]
            ys_f = [y for y in ys if y is not None]
            if not xs:
                continue
            style = "--" if (bb, mix) in self_labeled else "-"
            ax.plot(xs, ys_f, style, marker="o", label=mix, markersize=4)
        ax.set_title(bb, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Structure test — monotone / flat / U-shape
# ---------------------------------------------------------------------------

def classify_envelope_shape(bin_centers: list[float],
                            values: list[float | None]) -> dict:
    """Return {rho, slope, n_bins_used, shape}."""
    pairs = [(c, v) for c, v in zip(bin_centers, values) if v is not None]
    if len(pairs) < 3:
        return {"rho": None, "slope": None, "n_bins_used": len(pairs), "shape": "insufficient"}
    xs = [c for c, _ in pairs]
    ys = [v for _, v in pairs]
    rho = _spearman(xs, ys)
    # Linear slope via least-squares (closed form).
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs) or 1e-9
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    # Shape classification.
    if rho is None:
        shape = "unknown"
    elif rho >= 0.6:
        shape = "monotone+"
    elif rho <= -0.6:
        shape = "monotone-"
    elif abs(rho) < 0.3:
        # Check for U/inverted-U: max/min at an interior bin.
        mid_idx = ys.index(max(ys))
        if 0 < mid_idx < len(ys) - 1 and (max(ys) - min(ys)) > 0.05:
            shape = "inverted-U"
        else:
            mid_idx = ys.index(min(ys))
            if 0 < mid_idx < len(ys) - 1 and (max(ys) - min(ys)) > 0.05:
                shape = "U"
            else:
                shape = "flat"
    else:
        shape = "weak"
    return {"rho": rho, "slope": slope, "n_bins_used": len(pairs), "shape": shape}
