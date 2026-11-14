#!/usr/bin/env python3
"""Phase A — write report_phaseA.md from the long-CSVs produced by 01_bin_predictions.py.

Consumes:
  <matrix_dir>/sent_duration/matrix_duration_long_<bin_mode>.csv
  <matrix_dir>/sent_duration/envelope_notes_<bin_mode>.json

Produces:
  <matrix_dir>/sent_duration/report_phaseA.md

Report contents:
  - Per-test, per-axis envelope tables (bin center × backbone, best-mixture-avg).
  - Structure-test one-liner per (backbone × test × axis × bin_mode): shape, ρ, slope.
  - Fixed-vs-quantile sanity cross-check: do shapes agree?
  - Self-labeled cell note.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

BACKBONES_ORDER = ["xlm-roberta-large", "xlm-r-parla", "xlm-r-parlasent"]
MIXTURES_ORDER = ["S", "P", "R", "SP", "SR", "PR", "SPR"]
TEST_SOURCES_ORDER = ["parlasent_sl", "parlaspeech_si", "rog_dia"]
AXES_ORDER = ["audio_sec", "text_words"]
SELF_LABELED_BB = "xlm-r-parlasent"


def _load_long(csv_path: Path) -> list[dict]:
    out = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            # Cast types.
            for num_f in ("bin_idx", "bin_lo", "bin_hi", "bin_center", "n",
                          "F1_3", "F1_6", "R2", "spearman", "MAE", "self_labeled"):
                v = row.get(num_f)
                if v == "" or v is None:
                    row[num_f] = None
                else:
                    try:
                        row[num_f] = float(v) if "." in v or num_f in {"bin_lo", "bin_hi", "bin_center",
                                                                        "F1_3", "F1_6", "R2",
                                                                        "spearman", "MAE"} else int(v)
                    except (TypeError, ValueError):
                        row[num_f] = None
            out.append(row)
    return out


def _best_mixture_avg(rows: list[dict], primary: str) -> dict:
    """{(test, axis, backbone, bin_idx) -> {center, value, best_mixture, n}}"""
    out = {}
    buckets = defaultdict(list)
    for r in rows:
        if r.get("self_labeled"):
            continue
        key = (r["test_source"], r["axis"], r["backbone_short"], r["bin_idx"])
        v = r.get(primary)
        if v is None:
            continue
        buckets[key].append((r["mixture"], v, r.get("n"), r.get("bin_center")))
    for key, entries in buckets.items():
        best = max(entries, key=lambda t: t[1])
        out[key] = {"center": best[3], "value": best[1],
                    "best_mixture": best[0], "n": best[2]}
    return out


def _bins_for(rows: list[dict], test_source: str, axis: str) -> list[tuple[int, float, float, float]]:
    """Return sorted (bin_idx, lo, hi, center) tuples for a (test, axis)."""
    seen = {}
    for r in rows:
        if r["test_source"] != test_source or r["axis"] != axis:
            continue
        seen[r["bin_idx"]] = (r["bin_lo"], r["bin_hi"], r["bin_center"])
    return [(b, *seen[b]) for b in sorted(seen)]


def _render_envelope_table(rows: list[dict], test_source: str, axis: str,
                           primary: str) -> str:
    bins = _bins_for(rows, test_source, axis)
    if not bins:
        return "_(no data)_"
    bma = _best_mixture_avg(rows, primary)
    lines = [
        f"| backbone \\ bin | " + " | ".join(
            f"{lo:g}–{hi:g}" for _, lo, hi, _ in bins
        ) + " |",
        "|" + "---|" * (len(bins) + 1),
    ]
    for bb in BACKBONES_ORDER:
        cells = []
        for b, _, _, _ in bins:
            r = bma.get((test_source, axis, bb, b))
            if r is None:
                cells.append(" — ")
            else:
                cells.append(f"{r['value']:+.3f}<br/>({r['best_mixture']})")
        lines.append(f"| `{bb}` | " + " | ".join(cells) + " |")
    # n per bin (max over backbones)
    n_cells = []
    for b, _, _, _ in bins:
        ns = [bma[(test_source, axis, bb, b)]["n"]
              for bb in BACKBONES_ORDER
              if (test_source, axis, bb, b) in bma]
        n_cells.append(str(max(ns)) if ns else "—")
    lines.append("| n (max across backbones) | " + " | ".join(n_cells) + " |")
    return "\n".join(lines)


def _render_shape_table(notes: list[dict], bin_mode: str) -> str:
    rows = [n for n in notes if n["bin_mode"] == bin_mode]
    if not rows:
        return "_(no notes)_"
    lines = [
        "| test | axis | backbone | shape | ρ | slope | bins used |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for n in rows:
        rho_s = f"{n['rho']:+.3f}" if n.get("rho") is not None else "—"
        slp_s = f"{n['slope']:+.4f}" if n.get("slope") is not None else "—"
        lines.append(
            f"| `{n['test_source']}` | {n['axis']} | `{n['backbone_short']}` | "
            f"{n['shape']} | {rho_s} | {slp_s} | {n['n_bins_used']} |"
        )
    return "\n".join(lines)


def _fixed_vs_quantile_crosscheck(notes_fixed: list[dict],
                                  notes_quant: list[dict]) -> str:
    """Compare per-(test, axis, backbone) shape label fixed vs quantile."""
    index_q = {(n["test_source"], n["axis"], n["backbone_short"]): n for n in notes_quant}
    rows = [
        "| test | axis | backbone | fixed shape (ρ) | quantile shape (ρ) | agrees? |",
        "|---|---|---|---|---|:---:|",
    ]
    agree_count = 0
    total = 0
    for n in notes_fixed:
        q = index_q.get((n["test_source"], n["axis"], n["backbone_short"]))
        if not q:
            continue
        total += 1
        agrees = n["shape"] == q["shape"]
        if agrees:
            agree_count += 1
        rho_f = f"{n['rho']:+.2f}" if n.get("rho") is not None else "—"
        rho_q = f"{q['rho']:+.2f}" if q.get("rho") is not None else "—"
        mark = "✅" if agrees else "⚠"
        rows.append(f"| `{n['test_source']}` | {n['axis']} | `{n['backbone_short']}` | "
                    f"{n['shape']} ({rho_f}) | {q['shape']} ({rho_q}) | {mark} |")
    rows.append(f"\n**Agreement: {agree_count} / {total}**")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("matrix_dir", type=Path)
    args = p.parse_args()

    matrix_dir = args.matrix_dir.resolve()
    sd_dir = matrix_dir / "sent_duration"
    if not sd_dir.exists():
        raise SystemExit(f"no sent_duration dir under {matrix_dir} — run 01_bin_predictions.py first")

    out = []
    out.append(f"# Phase A — sentiment × duration envelope report")
    out.append("")
    out.append(f"**matrix_dir:** `{matrix_dir}`")
    out.append("")
    out.append("Per-test envelope tables below show, for each bin, the **best-mixture** per "
               "(backbone × bin) by the primary metric (default `F1_3`). Self-labeled cells "
               "(xlm-r-parlasent trained on any mixture containing `P`) are **excluded**. "
               "`n` row reports the maximum bin count across backbones.")
    out.append("")

    fixed_csv = sd_dir / "matrix_duration_long_fixed.csv"
    quant_csv = sd_dir / "matrix_duration_long_quantile.csv"
    fixed_notes = sd_dir / "envelope_notes_fixed.json"
    quant_notes = sd_dir / "envelope_notes_quantile.json"

    bin_modes_present = []
    for label, csvp, notesp in (("fixed", fixed_csv, fixed_notes),
                                ("quantile", quant_csv, quant_notes)):
        if not csvp.exists():
            out.append(f"_(no `matrix_duration_long_{label}.csv` — rerun `01_bin_predictions.py --bin_mode {label}` to populate)_")
            out.append("")
            continue
        bin_modes_present.append(label)
        rows = _load_long(csvp)
        notes = json.loads(notesp.read_text()) if notesp.exists() else []

        out.append(f"## Bin mode: `{label}`")
        out.append("")
        for test_source in TEST_SOURCES_ORDER:
            for axis in AXES_ORDER:
                has_any = any(r["test_source"] == test_source and r["axis"] == axis
                              for r in rows)
                if not has_any:
                    continue
                out.append(f"### `{test_source}` · axis = `{axis}`")
                out.append("")
                out.append(_render_envelope_table(rows, test_source, axis, primary="F1_3"))
                out.append("")

        out.append(f"#### Structure classification — bin_mode = `{label}`")
        out.append("")
        out.append(_render_shape_table(notes, bin_mode=label))
        out.append("")

    # Fixed vs quantile cross-check if both present.
    if "fixed" in bin_modes_present and "quantile" in bin_modes_present:
        f_notes = json.loads(fixed_notes.read_text())
        q_notes = json.loads(quant_notes.read_text())
        out.append("## Fixed vs quantile sanity cross-check")
        out.append("")
        out.append("If a shape label survives the switch from fixed to equal-count bins, we treat "
                   "it as real signal rather than an artefact of boundary choice.")
        out.append("")
        out.append(_fixed_vs_quantile_crosscheck(f_notes, q_notes))
        out.append("")

    # Interpretation hints
    out.append("## How to read this")
    out.append("")
    out.append("- **monotone+ (ρ ≥ +0.6)** — signal grows with duration/word count. "
               "Supports the \"short ROG clips are inherently harder\" hypothesis.")
    out.append("- **monotone− (ρ ≤ −0.6)** — signal shrinks with duration. "
               "Unusual; would suggest the longer bins contain register-mismatched content.")
    out.append("- **flat (|ρ| < 0.3)** — signal is roughly constant across bins. "
               "Dominant interpretation: the parliament → dialogue gap is driven by register "
               "mismatch, not by utterance length.")
    out.append("- **U / inverted-U** — non-monotone; mid-length bin is the hardest or easiest. "
               "Expect to need a per-backbone follow-up.")
    out.append("")
    out.append("Envelope PNGs for each (test, axis, bin_mode) are saved next to this report.")

    report_path = sd_dir / "report_phaseA.md"
    report_path.write_text("\n".join(out))
    print(f"wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
