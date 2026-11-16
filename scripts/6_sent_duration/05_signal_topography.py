#!/usr/bin/env python3
"""3-D topography of sentiment signal over the (WPS × duration) plane.

For each (test_source × backbone), it:
  1. Pools all non-self-labelled predictions across the 7 training mixtures.
  2. Grids the test pool into K × K equal-count (WPS, duration) cells.
  3. Per grid cell, computes F1_3 using the BEST-mixture value (best-mixture-avg
     per cell — same convention as the envelope plots).
  4. Emits (a) a 2-D heatmap with bicubic smoothing and (b) a 3-D surface plot,
     one pair per (source × backbone).

The ridge/valley in the plots tells you: here is the (speech-rate, clip-length)
corner where sentiment signal is densest. Use it to motivate the training-pool
curation filter.

Outputs: <matrix_dir>/sent_duration/topography/*.png
         <matrix_dir>/sent_duration/topography/topography_long.csv

Usage:
  python3 scripts/6_sent_duration/05_signal_topography.py \\
      misc/run_results/finetune_20261113_162724 \\
      --sources rog_dia parlaspeech_si \\
      --k 4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from utils_sent_dur import (  # noqa: E402
    Config,
    PredRow,
    build_metadata_index,
    derive_quantile_bounds,
    discover_cells,
    load_config,
    load_predictions_for_cell,
    metrics_for_bin,
)


DEFAULT_SOURCES = ("rog_dia", "parlaspeech_si")


def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def grid_signal(
    per_mix_rows: dict[str, list[PredRow]],
    k: int,
    min_n_cell: int = 10,
) -> dict:
    """Build a K×K grid over (WPS, duration). Returns {
        'wps_bounds':   [(lo, hi), ...],
        'dur_bounds':   [(lo, hi), ...],
        'f1':           [[F1_3 or None]*K]*K,   # rows = WPS bin, cols = dur bin
        'n':            [[int]*K]*K,
        'best_mix':     [[str or None]*K]*K,
    }"""
    # Pool axis values from one mixture (all mixtures share the test set).
    sample = next(iter(per_mix_rows.values())) if per_mix_rows else []
    wps_vals = [r.speech_rate for r in sample if r.speech_rate is not None]
    dur_vals = [r.audio_sec for r in sample if r.audio_sec is not None]
    if not wps_vals or not dur_vals:
        return {"wps_bounds": [], "dur_bounds": [], "f1": [], "n": [], "best_mix": []}
    wps_bounds = derive_quantile_bounds(wps_vals, k=k)
    dur_bounds = derive_quantile_bounds(dur_vals, k=k)

    def _which_bin(v, bounds):
        if v is None:
            return None
        for i, (lo, hi) in enumerate(bounds):
            if lo < v <= hi:
                return i
        return None

    # Pre-bin rows per mixture.
    grid_f1 = [[None] * k for _ in range(k)]
    grid_n = [[0] * k for _ in range(k)]
    grid_mix = [[None] * k for _ in range(k)]
    # For each cell (i_wps, j_dur), compute F1 per mixture, take max.
    per_cell: dict[tuple[int, int], list[tuple[str, float, int]]] = {}
    for mix, rows in per_mix_rows.items():
        # Bucket this mixture's rows into (i, j).
        buckets: dict[tuple[int, int], list[PredRow]] = {}
        for r in rows:
            i = _which_bin(r.speech_rate, wps_bounds)
            j = _which_bin(r.audio_sec, dur_bounds)
            if i is None or j is None:
                continue
            buckets.setdefault((i, j), []).append(r)
        for (i, j), rs in buckets.items():
            if len(rs) < min_n_cell:
                continue
            m = metrics_for_bin(rs)
            if m.get("F1_3") is None:
                continue
            per_cell.setdefault((i, j), []).append((mix, m["F1_3"], len(rs)))

    for (i, j), entries in per_cell.items():
        best = max(entries, key=lambda t: t[1])
        grid_f1[i][j] = best[1]
        grid_n[i][j] = max(e[2] for e in entries)
        grid_mix[i][j] = best[0]

    return {
        "wps_bounds": wps_bounds, "dur_bounds": dur_bounds,
        "f1": grid_f1, "n": grid_n, "best_mix": grid_mix,
    }


def plot_heatmap(grid: dict, title: str, out_path: Path) -> Path | None:
    plt = _try_plt()
    if plt is None:
        return None
    import numpy as np
    f1 = np.array([[(v if v is not None else np.nan) for v in row] for row in grid["f1"]])
    if f1.size == 0:
        return None
    k = f1.shape[0]
    wps_centers = [(lo + hi) / 2.0 for lo, hi in grid["wps_bounds"]]
    dur_centers = [(lo + hi) / 2.0 for lo, hi in grid["dur_bounds"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    # Bicubic-smoothed heatmap.
    im = ax.imshow(
        f1, origin="lower", aspect="auto", cmap="viridis",
        interpolation="bicubic",
        extent=[dur_centers[0], dur_centers[-1], wps_centers[0], wps_centers[-1]],
    )
    # Overlay cell F1 values + n.
    for i in range(k):
        for j in range(k):
            v = grid["f1"][i][j]
            if v is None:
                continue
            ax.text(dur_centers[j], wps_centers[i],
                    f"{v:.2f}\nn={grid['n'][i][j]}",
                    ha="center", va="center", fontsize=7,
                    color="white" if v < 0.75 else "black")
    ax.set_xlabel("utterance duration (s)")
    ax.set_ylabel("speech rate (words/s)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="F1_3 (best mixture)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_surface_3d(grid: dict, title: str, out_path: Path) -> Path | None:
    plt = _try_plt()
    if plt is None:
        return None
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — side-effect import
    f1 = np.array([[(v if v is not None else np.nan) for v in row] for row in grid["f1"]])
    if f1.size == 0:
        return None
    wps_centers = np.array([(lo + hi) / 2.0 for lo, hi in grid["wps_bounds"]])
    dur_centers = np.array([(lo + hi) / 2.0 for lo, hi in grid["dur_bounds"]])
    X, Y = np.meshgrid(dur_centers, wps_centers)
    Z = np.where(np.isnan(f1), 0.0, f1)

    # Smooth with a Gaussian if scipy is available (nicer-looking surface);
    # otherwise just use the raw grid.
    try:
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=0.6, mode="nearest")
    except ImportError:
        pass

    fig = plt.figure(figsize=(7.5, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0.1,
                           edgecolor="white", antialiased=True)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("WPS (words/s)")
    ax.set_zlabel("F1_3")
    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=25, azim=-130)
    ax.set_title(title, fontsize=10)
    fig.colorbar(surf, ax=ax, label="F1_3", shrink=0.6, pad=0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("matrix_dir", type=Path)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES),
                   choices=["rog_dia", "parlaspeech_si"],
                   help="which test sources to produce topographies for")
    p.add_argument("--k", type=int, default=4,
                   help="grid size (K × K equal-count bins). Default 4.")
    p.add_argument("--min_n_cell", type=int, default=10,
                   help="minimum rows per grid cell to compute F1 (default 10).")
    args = p.parse_args()

    matrix_dir: Path = args.matrix_dir.resolve()
    if not matrix_dir.exists():
        raise SystemExit(f"matrix_dir not found: {matrix_dir}")

    cfg: Config = load_config(args.config)
    cells = discover_cells(matrix_dir, cfg)
    print(f"Discovered {len(cells)} cells")
    topo_dir = matrix_dir / "sent_duration" / "topography"
    topo_dir.mkdir(parents=True, exist_ok=True)

    long_rows = []

    for src in args.sources:
        index, info = build_metadata_index(src, cfg, ROOT)
        if "audio_sec" not in info["axes_supported"]:
            print(f"[{src}] audio_sec unavailable — skipping")
            continue
        print(f"\n[{src}] metadata rows={info['n_rows']}")

        # For each backbone, pool rows per mixture (non-self-labelled).
        for bb in cfg.backbones_order:
            per_mix_rows: dict[str, list[PredRow]] = {}
            for c in cells:
                if c.backbone_short != bb:
                    continue
                if c.self_labeled:
                    continue
                rs = load_predictions_for_cell(c, src, index)
                if rs:
                    per_mix_rows[c.mixture] = rs
            if not per_mix_rows:
                print(f"  [{bb}] no clean mixtures — skipping")
                continue

            grid = grid_signal(per_mix_rows, k=args.k, min_n_cell=args.min_n_cell)
            if not grid["f1"]:
                print(f"  [{bb}] empty grid — skipping")
                continue

            title_2d = (f"{src} · {bb} · sentiment signal topography\n"
                        f"(F1_3, best mixture per (WPS × duration) cell, "
                        f"self-labelled excluded)")
            heatmap_path = topo_dir / f"topography_{src}_{bb}_heatmap.png"
            plot_heatmap(grid, title_2d, heatmap_path)
            print(f"  [{bb}] wrote {heatmap_path.relative_to(ROOT)}")

            surface_path = topo_dir / f"topography_{src}_{bb}_surface.png"
            plot_surface_3d(grid, title_2d, surface_path)
            print(f"  [{bb}] wrote {surface_path.relative_to(ROOT)}")

            # Long-CSV
            for i, (wlo, whi) in enumerate(grid["wps_bounds"]):
                for j, (dlo, dhi) in enumerate(grid["dur_bounds"]):
                    long_rows.append({
                        "test_source": src, "backbone_short": bb,
                        "wps_bin_idx": i, "wps_lo": wlo, "wps_hi": whi,
                        "wps_center": (wlo + whi) / 2.0,
                        "dur_bin_idx": j, "dur_lo": dlo, "dur_hi": dhi,
                        "dur_center": (dlo + dhi) / 2.0,
                        "F1_3": grid["f1"][i][j], "n": grid["n"][i][j],
                        "best_mixture": grid["best_mix"][i][j],
                    })

    csv_path = topo_dir / "topography_long.csv"
    if long_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
            w.writeheader()
            w.writerows(long_rows)
        print(f"\nwrote {csv_path.relative_to(ROOT)}  ({len(long_rows)} rows)")


if __name__ == "__main__":
    main()
