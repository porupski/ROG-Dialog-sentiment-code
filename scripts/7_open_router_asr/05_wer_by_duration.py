#!/usr/bin/env python3
"""05_wer_by_duration — WER/CER stratified by target duration.

Ivan's hypothesis: "most of ROG is 1 s and 1 s is godawful." This script
tests it: bin the full-ROG predictions by target duration, compute micro WER/CER
per bin for priority turns (no + won), and emit a table + a PNG plot.

Usage:
    mamba run -n ssp python3 scripts/7_open_router_asr/05_wer_by_duration.py \\
        --run_dir runs/from_remote/7_asr_run/chirp3_fullrog_2026-11-23_145526
"""

import argparse
import json
import re
import string
from pathlib import Path

try:
    import jiwer
except ImportError:
    print("ERROR: jiwer not available — mamba run -n ssp")
    raise

_PUNCT = string.punctuation + "„“”‚‘’«»…–—"
_PUNCT_RE = re.compile(f"[{re.escape(_PUNCT)}]")


def normalise(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DUR_BINS = [(0, 1.0, "<1 s"),
            (1.0, 2.0, "1–2 s"),
            (2.0, 4.0, "2–4 s"),
            (4.0, 8.0, "4–8 s"),
            (8.0, 1e9, "8+ s")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_dir", required=True,
                   help="run directory containing predictions_enriched.jsonl")
    return p.parse_args()


def bin_of(d):
    for lo, hi, name in DUR_BINS:
        if lo <= d < hi:
            return name
    return "8+ s"


def micro_wer_cer(rows, use_norm=True):
    if use_norm:
        g = [normalise(r.get("text_gold") or "") for r in rows]
        a = [normalise(r.get("transcript") or "") for r in rows]
    else:
        g = [r.get("text_gold") or "" for r in rows]
        a = [r.get("transcript") or "" for r in rows]
    g = [x if x.strip() else "." for x in g]
    a = [x if x.strip() else "." for x in a]
    return jiwer.wer(g, a), jiwer.cer(g, a)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    pred_path = run_dir / "predictions_enriched.jsonl"
    rows = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    print(f"Loaded {len(rows)} predictions from {pred_path.name}")

    priority = [r for r in rows
                if r.get("overlap") in ("no", "won")
                and (r.get("text_gold") or "").strip()]
    print(f"Priority scope (no+won, non-empty gold): {len(priority)}")

    binned = {name: [] for _, _, name in DUR_BINS}
    for r in priority:
        d = r.get("duration_s") or 0
        binned[bin_of(d)].append(r)

    print("\n| duration bin | turns | %turns | hours | WER_norm | WER_raw | CER_norm | CER_raw | words |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    out_rows = []
    total_t = len(priority)
    total_h = sum((r.get("duration_s") or 0) for r in priority) / 3600
    for lo, hi, name in DUR_BINS:
        br = binned[name]
        if not br:
            continue
        n = len(br)
        h = sum((r.get("duration_s") or 0) for r in br) / 3600
        words = sum(len((r.get("text_gold") or "").split()) for r in br)
        wer_norm, cer_norm = micro_wer_cer(br, use_norm=True)
        wer_raw,  cer_raw  = micro_wer_cer(br, use_norm=False)
        pct = 100 * n / max(total_t, 1)
        pct_h = 100 * h / max(total_h, 1e-9)
        line = (f"| {name} | {n} | {pct:.1f} % | {h:.3f} ({pct_h:.1f} %) | "
                f"{wer_norm:.3f} | {wer_raw:.3f} | {cer_norm:.3f} | {cer_raw:.3f} | {words} |")
        print(line)
        out_rows.append({"bin": name, "n": n, "pct_turns": pct, "hours": h,
                         "pct_hours": pct_h,
                         "wer_norm": wer_norm, "wer_raw": wer_raw,
                         "cer_norm": cer_norm, "cer_raw": cer_raw, "words": words})

    out_json = run_dir / "wer_by_duration.json"
    out_json.write_text(json.dumps({"priority_n": len(priority),
                                    "total_hours": total_h,
                                    "bins": out_rows}, indent=2))
    print(f"\nSaved: {out_json}")

    if plt is None:
        print("matplotlib not available, skipping plot")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    labels = [r["bin"] for r in out_rows]
    wers_n = [r["wer_norm"] for r in out_rows]
    cers_n = [r["cer_norm"] for r in out_rows]
    ns     = [r["n"] for r in out_rows]
    x = list(range(len(labels)))
    ax1.bar(x, ns, color="lightgray", width=0.6, label="turns (count)", zorder=1)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("target duration")
    ax1.set_ylabel("turns per bin", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    ax2 = ax1.twinx()
    ax2.plot(x, wers_n, "o-", color="#a00", label="WER_norm", lw=2, markersize=8, zorder=3)
    ax2.plot(x, cers_n, "s--", color="#2a7a4a", label="CER_norm", lw=1.5, markersize=7, zorder=3)
    ax2.axhline(0.303, color="#555", lw=1, ls=":", label="SloBench WER2 = 0.303")
    ax2.set_ylabel("error rate (normalised)", color="#333")
    ax2.set_ylim(0, max(1.0, max(wers_n) * 1.1))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    plt.title(f"Chirp3 WER/CER by target duration (priority no+won, n={len(priority)}, FULLROG)")
    plt.tight_layout()

    docs_img = Path(__file__).resolve().parent.parent.parent / "docs" / "img"
    docs_img.mkdir(parents=True, exist_ok=True)
    out_png = docs_img / "chirp3_wer_by_duration_FULLROG.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
