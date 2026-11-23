#!/usr/bin/env python3
"""03_compute_metrics — post-hoc WER/CER + overlap rollup for a Chirp 3 run.

Reads a run dir's `predictions.jsonl`, joins with the full ROG v2 JSONL to
compute per-instance overlap labels (`no` / `won` / `lost`) and `overlap_frac`,
then emits:

  <run_dir>/predictions_enriched.jsonl   # per-row: + wer, cer, overlap, overlap_frac
  <run_dir>/metrics.json                 # all aggregates as structured JSON
  <run_dir>/metrics.md                   # same aggregates as a human report

Judge WER_norm (prio) against SloBench WER2=0.303.

Usage:
  mamba run -n ssp python scripts/7_open_router_asr/03_compute_metrics.py \\
      --run_dir data/7_asr_runs/chirp3_fullrog_<timestamp> \\
      --rog_jsonl data/rog_dialog_v2_instance.jsonl
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import jiwer


_PUNCT = string.punctuation + "„“”‚‘’«»…–—"
_PUNCT_RE = re.compile(f"[{re.escape(_PUNCT)}]")


def normalise(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_wer_cer(gold: str, asr: str):
    gold, asr = gold or "", asr or ""
    if not gold.strip() and not asr.strip():
        return 0.0, 0.0
    if not gold.strip():
        return float("nan"), float("nan")
    try:
        return jiwer.wer(gold, asr), jiwer.cer(gold, asr)
    except Exception:
        return float("nan"), float("nan")


def compute_overlap(rog: list[dict]):
    by_file: dict[str, list[dict]] = {}
    for r in rog:
        by_file.setdefault(r["file_id"], []).append(r)
    labels, fracs = {}, {}
    for rows in by_file.values():
        rows_sorted = sorted(rows, key=lambda r: (r["start_t"], r.get("speaker", "")))
        for r in rows_sorted:
            st, et, spk = r["start_t"], r["end_t"], r.get("speaker")
            dur = max(et - st, 1e-9)
            neigh, total = [], 0.0
            for s in rows_sorted:
                if s is r or s.get("speaker") == spk:
                    continue
                if s["end_t"] <= st or s["start_t"] >= et:
                    continue
                o = min(et, s["end_t"]) - max(st, s["start_t"])
                if o > 0:
                    neigh.append(s)
                    total += o
            fracs[r["instance_id"]] = min(total / dur, 1.0)
            if not neigh:
                labels[r["instance_id"]] = "no"
                continue
            key = lambda x: (-(x["end_t"] - x["start_t"]), x["start_t"], x.get("speaker", ""))
            labels[r["instance_id"]] = "won" if sorted([r] + neigh, key=key)[0] is r else "lost"
    return labels, fracs


def _wc(s): return len((s or "").split())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", required=True,
                   help="run directory containing predictions.jsonl")
    p.add_argument("--rog_jsonl", required=True,
                   help="full ROG v2 instance JSONL (needed for overlap labelling)")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found", file=sys.stderr); sys.exit(2)
    rog_path = Path(args.rog_jsonl).resolve()
    if not rog_path.exists():
        print(f"ERROR: {rog_path} not found", file=sys.stderr); sys.exit(2)

    preds = [json.loads(l) for l in open(pred_path, encoding="utf-8")]
    rog = [json.loads(l) for l in open(rog_path, encoding="utf-8")]
    labels, fracs = compute_overlap(rog)

    for p in preds:
        iid = p["instance_id"]
        p["overlap"] = labels.get(iid, "unknown")
        p["overlap_frac"] = fracs.get(iid, float("nan"))
        g, a = p.get("text_gold") or "", p.get("transcript") or ""
        p["gold_norm"] = normalise(g)
        p["asr_norm"]  = normalise(a)
        p["wer_raw"], p["cer_raw"]   = safe_wer_cer(g, a)
        p["wer_norm"], p["cer_norm"] = safe_wer_cer(p["gold_norm"], p["asr_norm"])

    enriched = run_dir / "predictions_enriched.jsonl"
    with open(enriched, "w", encoding="utf-8") as f:
        for p in preds:
            keep = {k: p[k] for k in (
                "instance_id", "audio_path", "text_gold", "transcript",
                "duration_s", "model", "language", "status", "latency_s",
                "overlap", "overlap_frac",
                "wer_raw", "cer_raw", "wer_norm", "cer_norm",
            ) if k in p}
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")

    scored = [p for p in preds if not p.get("error") and (p.get("text_gold") or "").strip()]
    prio   = [p for p in scored if p["overlap"] in ("no", "won")]

    def micro(rows):
        if not rows: return None
        g  = [p["text_gold"] for p in rows]
        a  = [p["transcript"] or "" for p in rows]
        gn = [p["gold_norm"] for p in rows]
        an = [p["asr_norm"]  for p in rows]
        return {"n": len(rows),
                "wer_raw":  jiwer.wer(g, a),
                "wer_norm": jiwer.wer(gn, an),
                "cer_raw":  jiwer.cer(g, a),
                "cer_norm": jiwer.cer(gn, an)}

    all_m  = micro(scored)
    prio_m = micro(prio)

    total_h = sum((p.get("duration_s") or 0) for p in preds) / 3600
    total_t = len(preds)
    total_w = sum(_wc(p.get("text_gold")) for p in preds)
    oframe_num = sum((p.get("overlap_frac") or 0) * (p.get("duration_s") or 0) for p in preds)
    oframe_den = sum((p.get("duration_s") or 0) for p in preds) or 1.0
    overlap_frac_time = oframe_num / oframe_den

    bucket_stats = {}
    for k in ("no", "won", "lost", "unknown"):
        rows = [p for p in preds if p.get("overlap") == k]
        if not rows: continue
        h = sum((p.get("duration_s") or 0) for p in rows) / 3600
        w = sum(_wc(p.get("text_gold")) for p in rows)
        ofs = [p["overlap_frac"] for p in rows if p.get("overlap_frac") == p.get("overlap_frac")]
        bucket_stats[k] = {
            "turns": len(rows), "hours": h, "words": w,
            "turns_pct": 100 * len(rows) / max(total_t, 1),
            "hours_pct": 100 * h / max(total_h, 1e-9),
            "words_pct": 100 * w / max(total_w, 1),
            "overlap_frac_mean": (sum(ofs) / len(ofs)) if ofs else None,
        }

    total_cost = sum(((p.get("raw_response") or {}).get("usage") or {}).get("cost", 0) or 0
                     for p in preds)
    total_latency = sum(p.get("latency_s") or 0 for p in preds)

    metrics = {
        "run_dir": str(run_dir),
        "n_preds": total_t,
        "n_errors": sum(1 for p in preds if p.get("error")),
        "n_scored": len(scored),
        "n_priority": len(prio),
        "total_hours": total_h,
        "total_turns": total_t,
        "total_words": total_w,
        "overlap_frac_time_weighted_mean": overlap_frac_time,
        "wer_cer_all":  all_m,
        "wer_cer_prio": prio_m,
        "buckets": bucket_stats,
        "or_cost_usd": total_cost,
        "or_latency_s": total_latency,
        "or_latency_s_mean_per_row": total_latency / max(total_t, 1),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    lines = [
        f"# Chirp 3 run metrics — `{run_dir.name}`",
        "",
        f"- predictions: {total_t}  (scored {len(scored)}, errors {metrics['n_errors']})",
        f"- total audio: **{total_h:.3f} h**, {total_t} turns, {total_w} gold words",
        f"- time-weighted mean overlap_frac: **{overlap_frac_time:.3f}**",
        f"- OR cost: ${total_cost:.4f}   wall-time (summed per-row latency): {total_latency/60:.1f} min",
        "",
        "## WER / CER",
        "",
        "| scope | n | WER_raw | WER_norm | CER_raw | CER_norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if all_m:
        lines.append(f"| all | {all_m['n']} | {all_m['wer_raw']:.3f} | "
                     f"{all_m['wer_norm']:.3f} | {all_m['cer_raw']:.3f} | {all_m['cer_norm']:.3f} |")
    if prio_m:
        lines.append(f"| **priority (no+won)** | {prio_m['n']} | {prio_m['wer_raw']:.3f} | "
                     f"**{prio_m['wer_norm']:.3f}** | {prio_m['cer_raw']:.3f} | {prio_m['cer_norm']:.3f} |")
    lines += ["",
              "Judge **WER_norm (priority)** against SloBench WER2 = 0.303.",
              "",
              "## Overlap rollup",
              "",
              "| bucket | turns | %t | hours | %h | words | %w | overlap_frac̄ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("no", "won", "lost", "unknown"):
        b = bucket_stats.get(k)
        if not b: continue
        ofm = f"{b['overlap_frac_mean']:.3f}" if b["overlap_frac_mean"] is not None else "—"
        lines.append(f"| {k} | {b['turns']} | {b['turns_pct']:.1f}% | {b['hours']:.3f} | "
                     f"{b['hours_pct']:.1f}% | {b['words']} | {b['words_pct']:.1f}% | {ofm} |")
    lines += ["",
              f"Outputs written: `{(run_dir / 'metrics.json').name}`, "
              f"`{(run_dir / 'metrics.md').name}`, "
              f"`{(run_dir / 'predictions_enriched.jsonl').name}`.",
              ""]
    (run_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
