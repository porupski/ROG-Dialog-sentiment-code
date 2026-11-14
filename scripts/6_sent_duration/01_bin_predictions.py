#!/usr/bin/env python3
"""Phase A — bin a finetune-matrix run's predictions by utterance duration / word count,
compute per-bin metrics, and emit an envelope plot per test source.

Consumes:
  <matrix_dir>/<backbone_short>__<mixture>/predictions_<test_src>.jsonl
  <matrix_dir>/matrix_summary.json  (optional; used for header meta only)
  data/<source>.jsonl               (for instance_id -> duration metadata join)
  scripts/6_sent_duration/config.json

Produces (under <matrix_dir>/sent_duration/):
  matrix_duration_long_<bin_mode>.csv        one row per (cell × test × axis × bin)
  envelope_<test>_<axis>_<bin_mode>.png      primary: one line per backbone, best-mixture-avg
  envelope_facet_<test>_<axis>_<bin_mode>.png  per-backbone subplots, one line per mixture

Usage:
  python3 scripts/6_sent_duration/01_bin_predictions.py <matrix_dir> [--bin_mode fixed|quantile]
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
    bin_center,
    bin_rows_fixed,
    build_metadata_index,
    classify_envelope_shape,
    derive_quantile_bounds,
    discover_cells,
    envelope_best_mixture_avg,
    load_config,
    load_predictions_for_cell,
    metrics_for_bin,
    plot_envelope_facet,
    plot_envelope_primary,
    rows_by_axis_value,
)


AXIS_LABELS = {
    "audio_sec":  ("utterance duration (s)", "audio"),
    "text_words": ("word count",             "words"),
}


def long_csv_rows(cell_tag, bb, mix, self_labeled, test_source, axis,
                  bin_idx, lo, hi, n_in_bin, metrics):
    center = bin_center(lo, hi)
    return {
        "cell_tag": cell_tag,
        "backbone_short": bb,
        "mixture": mix,
        "self_labeled": int(self_labeled),
        "test_source": test_source,
        "axis": axis,
        "bin_idx": bin_idx,
        "bin_lo": lo,
        "bin_hi": hi,
        "bin_center": center,
        "n": n_in_bin,
        "F1_3": metrics.get("F1_3"),
        "F1_6": metrics.get("F1_6"),
        "R2": metrics.get("R2"),
        "spearman": metrics.get("spearman"),
        "MAE": metrics.get("MAE"),
    }


LONG_CSV_FIELDS = [
    "cell_tag", "backbone_short", "mixture", "self_labeled",
    "test_source", "axis",
    "bin_idx", "bin_lo", "bin_hi", "bin_center", "n",
    "F1_3", "F1_6", "R2", "spearman", "MAE",
]


def compute_per_cell_binned_metrics(
    cells,
    index: dict,
    test_source: str,
    axis: str,
    bounds: list[list[float]],
):
    """Returns {(bb, mix) -> {bin_idx -> metrics}}."""
    out: dict[tuple[str, str], dict[int, dict]] = {}
    for c in cells:
        pred_rows = load_predictions_for_cell(c, test_source, index)
        if not pred_rows:
            continue
        rwv = rows_by_axis_value(pred_rows, axis)
        if not rwv:
            continue
        binned = bin_rows_fixed(rwv, bounds)
        out[(c.backbone_short, c.mixture)] = {
            b: metrics_for_bin(rs) for b, rs in binned.items()
        }
    return out


def axis_values_across_cells(cells, index, test_source: str, axis: str) -> list[float]:
    """Pool axis values across all cells' predictions for quantile-bound derivation.
    Since every cell evaluates on the same test set, we can pull axis values from
    just the first available cell."""
    for c in cells:
        rows = load_predictions_for_cell(c, test_source, index)
        if not rows:
            continue
        rwv = rows_by_axis_value(rows, axis)
        if rwv:
            return [v for _, v in rwv]
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("matrix_dir", type=Path, help="e.g. misc/run_results/finetune_20261113_162724")
    p.add_argument("--bin_mode", choices=("fixed", "quantile"), default="fixed")
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    args = p.parse_args()

    matrix_dir: Path = args.matrix_dir.resolve()
    if not matrix_dir.exists():
        raise SystemExit(f"matrix_dir not found: {matrix_dir}")

    cfg: Config = load_config(args.config)
    cells = discover_cells(matrix_dir, cfg)
    if not cells:
        raise SystemExit(f"no cells found under {matrix_dir}")
    print(f"Discovered {len(cells)} cells in {matrix_dir}")

    out_dir = matrix_dir / "sent_duration"
    out_dir.mkdir(exist_ok=True)

    # Build the long-CSV row collector and the envelope collector.
    long_rows = []
    envelope_notes = []  # for the report later

    for test_source, axes in cfg.test_source_axis.items():
        index, info = build_metadata_index(test_source, cfg, ROOT)
        print(f"\n[{test_source}] metadata: path={info['path']} "
              f"missing={info['missing']} n_rows={info['n_rows']} "
              f"axes_supported={info['axes_supported']}")
        # Filter axes by what we actually support given the data.
        usable_axes = [a for a in axes if a in info["axes_supported"]]
        if usable_axes != axes:
            dropped = [a for a in axes if a not in usable_axes]
            print(f"  [note] dropping unsupported axes: {dropped}")
        if not usable_axes:
            print(f"  [skip] no supported axes for {test_source}")
            continue

        for axis in usable_axes:
            if args.bin_mode == "fixed":
                bounds = (cfg.bins_audio_sec if axis == "audio_sec"
                          else cfg.bins_text_words)
            else:
                vals = axis_values_across_cells(cells, index, test_source, axis)
                if not vals:
                    print(f"  [skip] no values available for quantile bins on {axis}")
                    continue
                bounds = derive_quantile_bounds(vals, k=5)
            centers = [bin_center(lo, hi) for lo, hi in bounds]
            print(f"  [{axis}] bin_mode={args.bin_mode}  bounds={bounds}")

            per_cell = compute_per_cell_binned_metrics(
                cells, index, test_source, axis, bounds
            )

            # Long-CSV rows
            for (bb, mix), binned in per_cell.items():
                self_labeled = any(c.backbone_short == bb and c.mixture == mix
                                   and c.self_labeled for c in cells)
                for b, (lo, hi) in enumerate(bounds):
                    m = binned.get(b, {})
                    long_rows.append(long_csv_rows(
                        cell_tag=f"{bb}__{mix}",
                        bb=bb, mix=mix, self_labeled=self_labeled,
                        test_source=test_source, axis=axis,
                        bin_idx=b, lo=lo, hi=hi,
                        n_in_bin=m.get("n", 0),
                        metrics=m,
                    ))

            # Envelope: primary per-metric
            self_labeled_set = {(c.backbone_short, c.mixture) for c in cells if c.self_labeled}
            env = envelope_best_mixture_avg(
                per_cell, cfg.backbones_order, cfg.primary_metric,
                n_bins=len(bounds), exclude_self_labeled=self_labeled_set,
            )

            # Blank out bins that fail min_bin_n test — no primary point there.
            for bb in cfg.backbones_order:
                for b in range(len(bounds)):
                    cell_ns = [binned.get(b, {}).get("n", 0)
                               for (bb2, _), binned in per_cell.items() if bb2 == bb]
                    if not cell_ns or max(cell_ns) < cfg.min_bin_n:
                        env[bb][b] = None

            x_label, axis_tag = AXIS_LABELS[axis]
            title = (f"{test_source} — sentiment signal vs {x_label}\n"
                     f"(bin_mode={args.bin_mode}, {cfg.primary_metric}, best-mixture avg)")
            png_path = out_dir / f"envelope_{test_source}_{axis_tag}_{args.bin_mode}.png"
            written = plot_envelope_primary(
                centers, env,
                title=title, xlabel=x_label,
                ylabel=cfg.primary_metric,
                out_path=png_path,
            )
            if written:
                print(f"  wrote {written.relative_to(ROOT)}")
            else:
                print(f"  [skip] matplotlib not installed — no envelope PNG")

            facet_path = out_dir / f"envelope_facet_{test_source}_{axis_tag}_{args.bin_mode}.png"
            facet_written = plot_envelope_facet(
                centers, per_cell,
                cfg.backbones_order, cfg.mixtures_order,
                cfg.primary_metric,
                title=(f"{test_source} — all cells\n"
                       f"(bin_mode={args.bin_mode}, dashed = self-labeled)"),
                xlabel=x_label, ylabel=cfg.primary_metric,
                out_path=facet_path,
                self_labeled=self_labeled_set,
            )
            if facet_written:
                print(f"  wrote {facet_written.relative_to(ROOT)}")

            # Structure-test one-liner per backbone to print and park for report.
            for bb in cfg.backbones_order:
                shape = classify_envelope_shape(centers, env[bb])
                envelope_notes.append({
                    "test_source": test_source, "axis": axis, "bin_mode": args.bin_mode,
                    "backbone_short": bb,
                    "shape": shape["shape"], "rho": shape["rho"],
                    "slope": shape["slope"], "n_bins_used": shape["n_bins_used"],
                })
                rho_s = f"{shape['rho']:+.2f}" if shape["rho"] is not None else "n/a"
                slp_s = f"{shape['slope']:+.3f}" if shape["slope"] is not None else "n/a"
                print(f"    [{bb:<22}] shape={shape['shape']:<11} ρ={rho_s} slope={slp_s}  "
                      f"n_bins_used={shape['n_bins_used']}")

    # Write long CSV
    csv_path = out_dir / f"matrix_duration_long_{args.bin_mode}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_CSV_FIELDS)
        w.writeheader()
        w.writerows(long_rows)
    print(f"\nwrote {csv_path.relative_to(ROOT)}  ({len(long_rows)} rows)")

    # Also dump the structure-test notes as JSON — consumed by 02_envelope_report.py.
    import json as _json
    notes_path = out_dir / f"envelope_notes_{args.bin_mode}.json"
    notes_path.write_text(_json.dumps(envelope_notes, indent=2))
    print(f"wrote {notes_path.relative_to(ROOT)}  ({len(envelope_notes)} notes)")


if __name__ == "__main__":
    main()
