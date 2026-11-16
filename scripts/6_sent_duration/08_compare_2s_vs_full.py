#!/usr/bin/env python3
"""Side-by-side comparison of the 2s-filtered matrix vs the full-ROG rerun.

Run AFTER the user has launched:
    python3 scripts/4_finetune_text/run_matrix.py --mixtures R_full SR_full PR_full SPR_full --use_gpu

Reads summary.json from each cell dir in both matrix output directories and emits a
one-screen markdown table pairing R↔R_full, SR↔SR_full, PR↔PR_full, SPR↔SPR_full for
each backbone.

Comparisons shown:
  1. Train effect, same test (rog_dia, 2 s-filtered) — did training on full ROG help on
     the comparable test? This is the apples-to-apples contrast.
  2. Full-test picture (rog_dia_full) — only the _full cells can produce this. Baselines
     are read from runs/baselines/baselines_rog_test_full.md if present.

Output: runs/finetune_rogfull_<stamp>/compare_2s_vs_full.md  (or --out)
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

MIXTURE_PAIRS = [("R", "R_full"), ("SR", "SR_full"), ("PR", "PR_full"), ("SPR", "SPR_full")]
BACKBONES = ["xlm-roberta-large", "xlm-r-parla", "xlm-r-parlasent"]
METRICS = ["macro_f1_3cls", "macro_f1_6cls", "r2", "spearman"]


def load_summary(cell_dir: Path):
    sj = cell_dir / "summary.json"
    if not sj.exists():
        return None
    return json.loads(sj.read_text())


def cell_metrics(summary, test_src):
    m = (summary or {}).get("metrics_by_test", {}).get(test_src)
    if not m:
        return {k: None for k in METRICS}
    return {k: m.get(k) for k in METRICS}


def fmt(v, prec=3):
    return f"{v:+.{prec}f}" if isinstance(v, (int, float)) else "—"


def fmt_pair(a, b):
    """Return (a_fmt, b_fmt, delta_fmt) with delta bolded if |delta| >= 0.02."""
    if a is None or b is None:
        return fmt(a), fmt(b), "—"
    d = b - a
    tag = f"{d:+.3f}"
    if abs(d) >= 0.02:
        tag = f"**{tag}**"
    return f"{a:+.3f}", f"{b:+.3f}", tag


def load_baselines(root: Path):
    out = {}
    for name in ("baselines_rog_test_2s", "baselines_rog_test_full"):
        p = root / "runs" / "baselines" / f"{name}.md"
        if p.exists():
            out[name] = p
    bj = root / "runs" / "baselines" / "baselines.json"
    if bj.exists():
        try:
            out["json"] = json.loads(bj.read_text())
        except Exception:
            pass
    return out


def render(filtered_root: Path, full_root: Path, baselines: dict) -> str:
    L = []
    L.append(f"# Matrix comparison — 2 s-filtered vs full ROG")
    L.append("")
    L.append(f"- filtered matrix: `{filtered_root.relative_to(ROOT) if filtered_root.is_relative_to(ROOT) else filtered_root}`")
    L.append(f"- full-ROG matrix: `{full_root.relative_to(ROOT) if full_root.is_relative_to(ROOT) else full_root}`")
    L.append("")

    # Section 1 — same test, train effect
    L.append("## 1 · Apples-to-apples — same ROG test (2 s-filtered), different training data")
    L.append("Positive Δ means the full-ROG training beat the 2 s-filtered training on the "
             "comparable test set. Bold = |Δ| ≥ 0.02.")
    L.append("")
    L.append("| backbone | mixture pair | F1_3 (2s) | F1_3 (full) | Δ F1_3 | F1_6 (2s) | F1_6 (full) | Δ F1_6 | R² (2s) | R² (full) | Δ R² |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bb in BACKBONES:
        for (mx, mx_full) in MIXTURE_PAIRS:
            a_sum = load_summary(filtered_root / f"{bb}__{mx}")
            b_sum = load_summary(full_root / f"{bb}__{mx_full}")
            a = cell_metrics(a_sum, "rog_dia")
            b = cell_metrics(b_sum, "rog_dia")
            f3a, f3b, d3 = fmt_pair(a["macro_f1_3cls"], b["macro_f1_3cls"])
            f6a, f6b, d6 = fmt_pair(a["macro_f1_6cls"], b["macro_f1_6cls"])
            r2a, r2b, dr2 = fmt_pair(a["r2"], b["r2"])
            L.append(f"| {bb} | {mx} ↔ {mx_full} | {f3a} | {f3b} | {d3} | {f6a} | {f6b} | {d6} | {r2a} | {r2b} | {dr2} |")
    L.append("")

    # Baseline strip for Section 1
    bj = baselines.get("json")
    if bj:
        r2s = (bj.get("eval_sets", {}).get("rog_test_2s", {}) or {}).get("strategies", {})
        if r2s:
            L.append("Baselines on the SAME test set (ROG test, 2 s-filtered):")
            L.append("")
            L.append("| strategy | F1_3 | F1_6 | R² |")
            L.append("|---|---:|---:|---:|")
            for s, m in r2s.items():
                L.append(f"| {s} | {m['macro_f1_3cls']:.3f} | {m['macro_f1_6cls']:.3f} | {m['r2_ord_6cls']:+.3f} |")
            L.append("")

    # Section 2 — full-ROG test, only full cells can report
    L.append("## 2 · The new picture — full ROG test (no duration filter)")
    L.append("Only `_full` cells are evaluated here (the 2 s-filtered models never saw "
             "sub-2-s training rows and are not natively scored on the full test).")
    L.append("")
    L.append("| backbone | mixture | F1_3 | F1_6 | R² | ρ |")
    L.append("|---|---|---:|---:|---:|---:|")
    for bb in BACKBONES:
        for _, mx_full in MIXTURE_PAIRS:
            b_sum = load_summary(full_root / f"{bb}__{mx_full}")
            b = cell_metrics(b_sum, "rog_dia_full")
            L.append(
                f"| {bb} | {mx_full} "
                f"| {fmt(b['macro_f1_3cls'])} "
                f"| {fmt(b['macro_f1_6cls'])} "
                f"| {fmt(b['r2'])} "
                f"| {fmt(b['spearman'])} |"
            )
    L.append("")

    if bj:
        rfull = (bj.get("eval_sets", {}).get("rog_test_full", {}) or {}).get("strategies", {})
        if rfull:
            L.append("Baselines on the full ROG test (no duration filter):")
            L.append("")
            L.append("| strategy | F1_3 | F1_6 | R² |")
            L.append("|---|---:|---:|---:|")
            for s, m in rfull.items():
                L.append(f"| {s} | {m['macro_f1_3cls']:.3f} | {m['macro_f1_6cls']:.3f} | {m['r2_ord_6cls']:+.3f} |")
            L.append("")

    # Section 3 — winner pick
    L.append("## 3 · Winner suggestion")
    L.append("Pick the deliverable cell by argmax F1_3 on whichever test the bossman cares about.")
    L.append("")
    for test_src, label in [("rog_dia", "Filtered test (matches the HTML's current Section 2)"),
                            ("rog_dia_full", "Full test (the new picture)")]:
        candidates = []
        for bb in BACKBONES:
            # Pull from both matrices: filtered for rog_dia, full for both test sets
            for mx in ("R", "SR", "PR", "SPR"):
                s = load_summary(filtered_root / f"{bb}__{mx}")
                m = cell_metrics(s, test_src)
                if m["macro_f1_3cls"] is not None:
                    candidates.append((m["macro_f1_3cls"], f"{bb} × {mx} (filtered-train)", m))
            for mx in ("R_full", "SR_full", "PR_full", "SPR_full"):
                s = load_summary(full_root / f"{bb}__{mx}")
                m = cell_metrics(s, test_src)
                if m["macro_f1_3cls"] is not None:
                    candidates.append((m["macro_f1_3cls"], f"{bb} × {mx} (full-train)", m))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        best = candidates[0]
        L.append(f"**{label}** — top cell: `{best[1]}`  ·  F1_3 {best[0]:.3f}  ·  "
                 f"F1_6 {fmt(best[2]['macro_f1_6cls'])}  ·  R² {fmt(best[2]['r2'])}")
        L.append("")
        L.append("Top 5:")
        for score, tag, m in candidates[:5]:
            L.append(f"- {tag} — F1_3 {score:.3f} · F1_6 {fmt(m['macro_f1_6cls'])} · R² {fmt(m['r2'])}")
        L.append("")

    L.append("---")
    L.append("Next step: once a winner is agreed, replace the Section 2 table in "
             "`report_sentiment_signal_hunt.html` with its numbers.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--filtered", required=True, type=Path,
                    help="2s-filtered matrix output dir (e.g. runs/finetune_20261113_162724 "
                         "or misc/run_results/finetune_20261113_162724)")
    ap.add_argument("--full", required=True, type=Path,
                    help="Full-ROG matrix output dir (e.g. runs/finetune_rogfull_<stamp>)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Markdown output path (default: <full>/compare_2s_vs_full.md)")
    args = ap.parse_args()

    filtered_root = args.filtered if args.filtered.is_absolute() else (ROOT / args.filtered)
    full_root = args.full if args.full.is_absolute() else (ROOT / args.full)

    for p in (filtered_root, full_root):
        if not p.exists():
            raise SystemExit(f"Not found: {p}")

    baselines = load_baselines(ROOT)
    md = render(filtered_root, full_root, baselines)

    out = args.out or (full_root / "compare_2s_vs_full.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
