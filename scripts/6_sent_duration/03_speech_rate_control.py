#!/usr/bin/env python3
"""Phase A' — redo the ROG-Dialog duration envelope WITH speech-rate controlled.

Question we're controlling for:
  Does the "signal grows with duration" pattern on ROG survive once we hold
  speech rate (words-per-second) constant? If the signal still climbs within
  each WPS quartile, duration is the real explanatory variable. If the
  climb disappears, it was speech-rate riding on duration.

Approach:
  1. For each ROG-Dia test row, compute wps = text_words / audio_sec.
  2. Split the test pool into 4 WPS quartiles.
  3. For each WPS quartile × each axis (audio_sec, text_words), rerun the
     primary envelope (one line per backbone, best-mixture-avg per bin,
     self-labeled cells excluded).
  4. Compare per-backbone shape classification across quartiles — if all four
     quartiles still show monotone+, duration is the driver.

Only ROG-Dia is binned this way (it's the only source where we have both
audio_sec and text_words for every row). ParlaSent-SL has no audio;
ParlaSpeech-SI has no slim JSONL on disk right now.

Outputs (under <matrix_dir>/sent_duration/):
  speech_rate_distribution_rog_dia.png            histogram with Q1-4 cut lines
  envelope_rog_dia_<axis>_wps_q<k>.png            4 envelopes × 2 axes = 8 PNGs
  matrix_duration_long_speech_rate.csv            one row per (cell × axis × q × bin)
  speech_rate_report.md                           per-quartile shape classification
                                                  + verdict (confound confirmed/refuted)

Usage:
  python3 scripts/6_sent_duration/03_speech_rate_control.py \
      misc/run_results/finetune_20261113_162724
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from utils_sent_dur import (  # noqa: E402
    Config,
    PredRow,
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
    plot_envelope_primary,
)


DEFAULT_TEST_SOURCE = "rog_dia"


def _try_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_wps_distribution(wps_values: list[float], quartile_edges: list[float],
                          out_path: Path) -> Path | None:
    plt = _try_plt()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(wps_values, bins=40, color="#5a9bd4", edgecolor="#2f5e82")
    for e in quartile_edges:
        ax.axvline(e, color="#a00", linestyle="--", linewidth=1)
    ax.set_xlabel("speech rate (words / s)")
    ax.set_ylabel("count")
    ax.set_title(f"ROG-Dialog test — speech-rate distribution (n={len(wps_values)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("matrix_dir", type=Path)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--n_quartiles", type=int, default=4)
    p.add_argument("--n_duration_bins", type=int, default=3,
                   help="equal-count duration bins used within each WPS quartile. "
                        "ROG test n≈434 → 4 quartiles × 3 bins ≈ 36/bin (safely above min_bin_n=25). "
                        "Raise to 5 only if min_bin_n is relaxed via --min_bin_n.")
    p.add_argument("--min_bin_n", type=int, default=None,
                   help="override config.json min_bin_n for within-quartile slices. "
                        "Default: use config value (25).")
    p.add_argument("--test_source", default=DEFAULT_TEST_SOURCE,
                   choices=["rog_dia", "parlaspeech_si"],
                   help="test source to run the WPS control on. Requires both "
                        "audio_sec and text_words to be available for the source.")
    args = p.parse_args()
    TEST_SOURCE = args.test_source

    matrix_dir: Path = args.matrix_dir.resolve()
    if not matrix_dir.exists():
        raise SystemExit(f"matrix_dir not found: {matrix_dir}")
    cfg: Config = load_config(args.config)
    if args.min_bin_n is not None:
        cfg.min_bin_n = args.min_bin_n
    print(f"Using min_bin_n={cfg.min_bin_n}, "
          f"n_quartiles={args.n_quartiles}, n_duration_bins={args.n_duration_bins}")

    cells = discover_cells(matrix_dir, cfg)
    if not cells:
        raise SystemExit(f"no cells found under {matrix_dir}")
    print(f"Discovered {len(cells)} cells")

    out_dir = matrix_dir / "sent_duration"
    out_dir.mkdir(exist_ok=True)

    # 1) Load ROG-Dia metadata (gives audio_sec per instance_id).
    index, info = build_metadata_index(TEST_SOURCE, cfg, ROOT)
    print(f"[{TEST_SOURCE}] metadata rows={info['n_rows']} axes={info['axes_supported']}")
    if "audio_sec" not in info["axes_supported"]:
        raise SystemExit(f"audio_sec not available for {TEST_SOURCE} — nothing to control for")

    # 2) Pool prediction rows from the first available cell to compute WPS universe.
    #    (Every cell evaluates on the same test set, so axis values are identical.)
    sample_rows: list[PredRow] = []
    for c in cells:
        rs = load_predictions_for_cell(c, TEST_SOURCE, index)
        if rs:
            sample_rows = rs
            break
    wps_pool = [r.speech_rate for r in sample_rows if r.speech_rate is not None]
    if not wps_pool:
        raise SystemExit("no rows carried both audio_sec and text_words — can't compute WPS")

    # 3) WPS quartile edges.
    sorted_wps = sorted(wps_pool)
    n = len(sorted_wps)
    quartile_edges = []
    for q in range(1, args.n_quartiles):
        idx = min(int(round(q * n / args.n_quartiles)) - 1, n - 1)
        quartile_edges.append(sorted_wps[idx])
    q_lo = sorted_wps[0] - 1e-6
    q_hi = sorted_wps[-1] + 1e-6
    quartile_bounds = []
    prev = q_lo
    for e in quartile_edges + [q_hi]:
        quartile_bounds.append((prev, e))
        prev = e
    print(f"[WPS] n={n}  min={sorted_wps[0]:.2f}  max={sorted_wps[-1]:.2f}  "
          f"edges={[round(e,2) for e in quartile_edges]}")
    for i, (lo, hi) in enumerate(quartile_bounds):
        c_in_q = sum(1 for v in wps_pool if lo < v <= hi)
        print(f"  Q{i+1}: ({lo:.2f}, {hi:.2f}] → {c_in_q} rows")

    plot_wps_distribution(
        wps_pool, quartile_edges,
        out_dir / f"speech_rate_distribution_{TEST_SOURCE}.png",
    )

    # 4) For each quartile, for each axis, bin + score + plot envelope.
    axes = ["audio_sec", "text_words"]
    long_rows = []
    shape_notes = []
    self_labeled_set = {(c.backbone_short, c.mixture) for c in cells if c.self_labeled}

    for q_idx, (wps_lo, wps_hi) in enumerate(quartile_bounds):
        q_tag = f"q{q_idx + 1}"
        # Subset of test instance_ids within this WPS quartile.
        wps_filter = {
            r.instance_id for r in sample_rows
            if r.speech_rate is not None and wps_lo < r.speech_rate <= wps_hi
        }
        if not wps_filter:
            continue
        print(f"\n== WPS {q_tag}  ({wps_lo:.2f}, {wps_hi:.2f}]  n={len(wps_filter)} ==")

        # Pre-filter predictions per cell to this quartile, then bin + score per axis.
        per_cell_rows: dict[tuple[str, str], list[PredRow]] = {}
        for c in cells:
            rows = load_predictions_for_cell(c, TEST_SOURCE, index)
            rows_q = [r for r in rows if r.instance_id in wps_filter]
            if rows_q:
                per_cell_rows[(c.backbone_short, c.mixture)] = rows_q

        for axis in axes:
            # Pool axis values WITHIN the quartile for the shared per-source quantile bounds.
            axis_vals_pool = []
            for rows in per_cell_rows.values():
                for r in rows:
                    v = r.text_words if axis == "text_words" else r.audio_sec
                    if v is not None:
                        axis_vals_pool.append(v)
                break   # same test set; one cell suffices
            if not axis_vals_pool:
                continue
            bounds = derive_quantile_bounds(axis_vals_pool, k=args.n_duration_bins)
            centers = [bin_center(lo, hi) for lo, hi in bounds]

            # Bin each cell's rows.
            per_cell_metrics: dict[tuple[str, str], dict[int, dict]] = {}
            for (bb, mix), rows in per_cell_rows.items():
                rwv = []
                for r in rows:
                    v = r.text_words if axis == "text_words" else r.audio_sec
                    if v is None:
                        continue
                    rwv.append((r, float(v)))
                if not rwv:
                    continue
                binned = bin_rows_fixed(rwv, bounds)
                per_cell_metrics[(bb, mix)] = {
                    b: metrics_for_bin(rs) for b, rs in binned.items()
                }

            # Envelope + long-CSV rows.
            env = envelope_best_mixture_avg(
                per_cell_metrics, cfg.backbones_order, cfg.primary_metric,
                n_bins=len(bounds), exclude_self_labeled=self_labeled_set,
            )

            # Apply min_bin_n gate.
            for bb in cfg.backbones_order:
                for b in range(len(bounds)):
                    max_n = 0
                    for (bb2, _), binned in per_cell_metrics.items():
                        if bb2 != bb:
                            continue
                        n_b = binned.get(b, {}).get("n", 0)
                        if n_b > max_n:
                            max_n = n_b
                    if max_n < cfg.min_bin_n:
                        env[bb][b] = None

            # Plot
            title = (f"{TEST_SOURCE} · WPS {q_tag} ({wps_lo:.2f}, {wps_hi:.2f}] · {axis}\n"
                     f"(best-mixture avg, min_bin_n={cfg.min_bin_n})")
            png_path = out_dir / f"envelope_{TEST_SOURCE}_{axis}_wps_{q_tag}.png"
            plot_envelope_primary(
                centers, env,
                title=title,
                xlabel=("utterance duration (s)" if axis == "audio_sec" else "word count"),
                ylabel=cfg.primary_metric,
                out_path=png_path,
            )
            print(f"  [{axis}] wrote {png_path.relative_to(ROOT)}")

            # Shape classification per backbone.
            for bb in cfg.backbones_order:
                shape = classify_envelope_shape(centers, env[bb])
                shape_notes.append({
                    "wps_quartile": q_tag,
                    "wps_lo": wps_lo, "wps_hi": wps_hi,
                    "axis": axis, "backbone_short": bb,
                    "shape": shape["shape"], "rho": shape["rho"],
                    "slope": shape["slope"], "n_bins_used": shape["n_bins_used"],
                })

            # Long CSV
            for (bb, mix), binned in per_cell_metrics.items():
                self_labeled = (bb, mix) in self_labeled_set
                for b, (lo, hi) in enumerate(bounds):
                    m = binned.get(b, {})
                    long_rows.append({
                        "wps_quartile": q_tag,
                        "wps_lo": wps_lo, "wps_hi": wps_hi,
                        "backbone_short": bb, "mixture": mix,
                        "self_labeled": int(self_labeled),
                        "axis": axis,
                        "bin_idx": b, "bin_lo": lo, "bin_hi": hi,
                        "bin_center": bin_center(lo, hi),
                        "n": m.get("n", 0),
                        "F1_3": m.get("F1_3"), "F1_6": m.get("F1_6"),
                        "R2": m.get("R2"), "spearman": m.get("spearman"),
                        "MAE": m.get("MAE"),
                    })

    # 5) Write long CSV.
    csv_path = out_dir / f"matrix_duration_long_speech_rate_{TEST_SOURCE}.csv"
    if long_rows:
        fields = list(long_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(long_rows)
        print(f"\nwrote {csv_path.relative_to(ROOT)}  ({len(long_rows)} rows)")

    # 6) Report + verdict.
    report_lines = []
    report_lines.append(f"# Phase A' — speech-rate-controlled envelope on `{TEST_SOURCE}`\n")
    report_lines.append(f"**matrix_dir:** `{matrix_dir}`\n")
    report_lines.append("Question: does the Phase-A finding — that ROG-Dialog sentiment signal "
                        "grows with utterance duration — survive holding speech rate constant?\n")
    report_lines.append(f"**Speech-rate quartile edges (words/s):** "
                        f"{[round(e, 2) for e in quartile_edges]}\n")
    report_lines.append("## Shape classification by WPS quartile\n")
    # Table.
    report_lines.append("| axis | backbone | " + " | ".join(
        f"Q{i+1} shape (ρ)" for i in range(args.n_quartiles)) + " |")
    report_lines.append("|---|---|" + "---|" * args.n_quartiles)
    # Pivot.
    by_axis_bb: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for n in shape_notes:
        by_axis_bb[(n["axis"], n["backbone_short"])][n["wps_quartile"]] = n
    for (axis, bb), qd in sorted(by_axis_bb.items()):
        cells_md = []
        for q in range(1, args.n_quartiles + 1):
            e = qd.get(f"q{q}")
            if e is None:
                cells_md.append("—")
            else:
                rho_s = f"{e['rho']:+.2f}" if e["rho"] is not None else "—"
                cells_md.append(f"{e['shape']} ({rho_s})")
        report_lines.append(f"| `{axis}` | `{bb}` | " + " | ".join(cells_md) + " |")
    report_lines.append("")

    # Verdict.
    verdict_lines = []
    for (axis, bb), qd in sorted(by_axis_bb.items()):
        shapes = [qd.get(f"q{q}", {}).get("shape") for q in range(1, args.n_quartiles + 1)]
        shapes_nonnull = [s for s in shapes if s and s != "insufficient"]
        if not shapes_nonnull:
            verdict_lines.append(f"- `{axis}` · `{bb}`: all quartiles insufficient (bin-count gate).")
            continue
        all_mono_pos = all(s == "monotone+" for s in shapes_nonnull)
        any_mono_pos = any(s == "monotone+" for s in shapes_nonnull)
        if all_mono_pos:
            verdict_lines.append(f"- `{axis}` · `{bb}`: **monotone+ in every usable WPS quartile** — "
                                 "duration effect is robust to speech-rate control.")
        elif any_mono_pos:
            n_pos = sum(1 for s in shapes_nonnull if s == "monotone+")
            verdict_lines.append(f"- `{axis}` · `{bb}`: monotone+ in **{n_pos}/{len(shapes_nonnull)}** "
                                 f"usable quartiles — partial confound; report both aggregates and "
                                 "within-WPS slices.")
        else:
            verdict_lines.append(f"- `{axis}` · `{bb}`: the monotone+ effect **does not survive** "
                                 "within-WPS slicing — Phase A reading was speech-rate riding on duration.")
    report_lines.append("## Verdict\n")
    report_lines.extend(verdict_lines)
    report_lines.append("")
    report_lines.append("## Artefacts\n")
    report_lines.append(f"- `speech_rate_distribution_{TEST_SOURCE}.png` — WPS histogram + quartile cuts.")
    report_lines.append(f"- `envelope_{TEST_SOURCE}_<axis>_wps_q{{1..4}}.png` — one envelope per WPS quartile.")
    report_lines.append(f"- `matrix_duration_long_speech_rate_{TEST_SOURCE}.csv` — raw per-bin metrics.")

    report_path = out_dir / f"speech_rate_report_{TEST_SOURCE}.md"
    report_path.write_text("\n".join(report_lines))
    print(f"wrote {report_path.relative_to(ROOT)}")
    notes_path = out_dir / f"envelope_notes_speech_rate_{TEST_SOURCE}.json"
    notes_path.write_text(json.dumps(shape_notes, indent=2))
    print(f"wrote {notes_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
