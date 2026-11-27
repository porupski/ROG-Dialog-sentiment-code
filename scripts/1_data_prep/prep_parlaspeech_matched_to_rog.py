#!/usr/bin/env python3
"""Sample ParlaSpeech-SI so its shape matches ROG-Dialog v2.

Goal: produce a ParlaSpeech-SI subset whose distribution over
    (3-cls sentiment label, audio duration, speech rate)
matches ROG-Dialog v2's distribution as closely as feasible. This lets us re-train
or re-evaluate on a ParlaSpeech subset where the clip-stat gap to ROG is removed,
so any remaining performance gap after training is attributable to REGISTER
(parliament vs colloquial dialogue), not to clip length or speech rate.

Matching strategy (joint, per-label):
  1. Bin duration and WPS into 5×5 ROG-quantile cells per 3-cls label.
  2. For each (label, dur_bin, wps_bin) cell count ROG's instances.
  3. Sample that many rows from ParlaSpeech's rows that fall in the same cell
     (using ROG's edges, so matching is on absolute values, not within-corpus rank).
  4. Where ParlaSpeech has fewer rows than ROG asks for in a cell, take what's there
     and record the deficit. Final subset is thus a subset of what ROG asks for.

Non-goals:
  - Exactly reproducing ROG's mean duration or mean WPS. ParlaSpeech below ~1.1 s does
    not exist (parliament utterances don't go that short) — expect a deficit in the
    shortest ROG bin and mean-duration drift upward.
  - Fine-grained speaker balance. Splits are preserved; speaker diversity is whatever
    ParlaSpeech already gives inside each sampled cell.

Output:
  data/parlaspeech_v4_si_matched_to_rog.jsonl   — slim rows with duration + wps + label
  data/parlaspeech_v4_si_matched_to_rog_report.md — joint-distribution before/after
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

ROG_PATH = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
PARLA_PATH = ROOT / "data" / "parlaspeech_v4_si_utterance_instance.jsonl"
OUT_JSONL = ROOT / "data" / "parlaspeech_v4_si_matched_to_rog.jsonl"
OUT_REPORT = ROOT / "data" / "parlaspeech_v4_si_matched_to_rog_report.md"

SENT_6_TO_3 = {
    "predominantlyNegative": "negative", "mixedNegative": "negative",
    "neutralNegative":       "neutral",  "neutralPositive": "neutral",
    "mixedPositive":         "positive", "predominantlyPositive": "positive",
}
LABELS_3 = ["negative", "neutral", "positive"]


def load_rog(rog_path: Path):
    rows = []
    with open(rog_path) as f:
        for line in f:
            r = json.loads(line)
            s = (r.get("labels") or {}).get("sentiment")
            if s is None:
                continue
            dur = r["end_t"] - r["start_t"]
            words = len(r["text"].split())
            wps = words / dur if dur > 0 else 0.0
            rows.append({
                "split":    r["split"],
                "label_3":  SENT_6_TO_3[s],
                "duration": dur,
                "wps":      wps,
            })
    return rows


def load_parla(parla_path: Path):
    """Keep only rows with a usable audio_length and ParlaSent_3 label. Build a SLIM
    record per row (flat schema — flatten sentiment.*, drop speaker_info / cap / pauses /
    etc.) so downstream readers don't have to walk the nested parlaspeech blob and the
    on-disk JSONL stays small."""
    rows = []
    kept, dropped = 0, 0
    with open(parla_path) as f:
        for line in f:
            r = json.loads(line)
            al = r.get("audio_length")
            sent = r.get("sentiment") or {}
            lab = sent.get("ParlaSent_3")
            logit = sent.get("ParlaSent_logit")
            sent6 = sent.get("ParlaSent_6")
            if al is None or lab is None or logit is None:
                dropped += 1
                continue
            try:
                dur = float(al)
            except (ValueError, TypeError):
                dropped += 1
                continue
            text = r.get("text") or ""
            words = len(text.split())
            if dur <= 0 or words == 0:
                dropped += 1
                continue
            lab3 = lab.strip().lower()
            if lab3 not in LABELS_3:
                dropped += 1
                continue
            slim = {
                "id":              r.get("id"),
                "split":           r.get("split", "train"),
                "label_3":         lab3,
                "duration":        dur,
                "wps":             words / dur,
                "text":            text,
                "sentiment_logit": float(logit),
                "sentiment_6":     sent6,
                "audio_start":     r.get("audio_start"),
                "audio_end":       r.get("audio_end"),
                "audio":           r.get("audio"),
            }
            rows.append({
                "_slim": slim,
                "label_3":  lab3,
                "duration": dur,
                "wps":      words / dur,
            })
            kept += 1
    return rows, kept, dropped


def quantile_edges(values, n_bins):
    q = np.linspace(0, 1, n_bins + 1)
    e = np.quantile(values, q)
    # make strictly monotonic
    for i in range(1, len(e)):
        if e[i] <= e[i - 1]:
            e[i] = e[i - 1] + 1e-6
    return e


def locate(x, edges):
    """Return bin index in [0, n_bins-1] for a single value against bin edges."""
    j = int(np.searchsorted(edges, x, side="right") - 1)
    return min(max(j, 0), len(edges) - 2)


def build_match(rog_rows, parla_rows, n_dur=5, n_wps=5, wps_clip=10.0, seed=42):
    """Return (sampled_parla_rows, report_dict)."""
    rng = random.Random(seed)

    # Reference edges from ROG — jointly over the whole corpus (not per-label), so the
    # grid is the same across labels. Per-label counts then drive sampling.
    rog_dur = np.array([r["duration"] for r in rog_rows])
    rog_wps = np.clip(np.array([r["wps"] for r in rog_rows]), 0, wps_clip)
    dur_edges = quantile_edges(rog_dur, n_dur)
    wps_edges = quantile_edges(rog_wps, n_wps)

    # Reference counts: ROG per (label, dur_bin, wps_bin)
    rog_count = defaultdict(int)
    for r in rog_rows:
        d_b = locate(r["duration"], dur_edges)
        w_b = locate(min(r["wps"], wps_clip), wps_edges)
        rog_count[(r["label_3"], d_b, w_b)] += 1

    # ParlaSpeech pool: rows indexed by (label, dur_bin, wps_bin)
    parla_pool = defaultdict(list)
    for r in parla_rows:
        d_b = locate(r["duration"], dur_edges)
        w_b = locate(min(r["wps"], wps_clip), wps_edges)
        parla_pool[(r["label_3"], d_b, w_b)].append(r)

    # Sample.
    sampled = []
    deficits = []
    for key, need in sorted(rog_count.items()):
        pool = parla_pool.get(key, [])
        got = min(len(pool), need)
        if got > 0:
            take = rng.sample(pool, got)
            sampled.extend(take)
        if got < need:
            deficits.append((key, need, got))

    # Report dict
    rpt = {
        "n_rog":                 len(rog_rows),
        "n_parla_available":     len(parla_rows),
        "n_matched":             len(sampled),
        "dur_edges":             [round(float(x), 3) for x in dur_edges],
        "wps_edges":             [round(float(x), 3) for x in wps_edges],
        "rog_label_prior":       dict(Counter(r["label_3"] for r in rog_rows)),
        "parla_label_prior_all": dict(Counter(r["label_3"] for r in parla_rows)),
        "matched_label_prior":   dict(Counter(r["label_3"] for r in sampled)),
        "deficits":              deficits,  # [(key, need, got)]
    }
    return sampled, rpt


def render_report(rpt, matched_rows, rog_rows, parla_rows_count):
    """Markdown report comparing ROG vs matched-Parla distributions."""
    L = []
    L.append("# ParlaSpeech-SI → ROG-Dialog v2 matched subsample")
    L.append("")
    L.append(f"- ROG rows considered:        **{rpt['n_rog']}**")
    L.append(f"- ParlaSpeech rows available: **{rpt['n_parla_available']}** (full file)")
    L.append(f"- Matched subset:             **{rpt['n_matched']}** rows (= {100*rpt['n_matched']/rpt['n_rog']:.1f}% of ROG size)")
    L.append("")
    L.append(f"- Duration bin edges (s):  `{rpt['dur_edges']}`")
    L.append(f"- WPS bin edges:           `{rpt['wps_edges']}`")
    L.append("")
    L.append("## 3-cls label prior")
    L.append("")
    L.append("| label | ROG | Parla (all) | Matched |")
    L.append("|---|---:|---:|---:|")
    for lab in LABELS_3:
        rn = rpt["rog_label_prior"].get(lab, 0)
        pn = rpt["parla_label_prior_all"].get(lab, 0)
        mn = rpt["matched_label_prior"].get(lab, 0)
        rogp = 100 * rn / rpt["n_rog"] if rpt["n_rog"] else 0
        parp = 100 * pn / rpt["n_parla_available"] if rpt["n_parla_available"] else 0
        matp = 100 * mn / rpt["n_matched"] if rpt["n_matched"] else 0
        L.append(f"| {lab} | {rn} ({rogp:.1f}%) | {pn} ({parp:.1f}%) | {mn} ({matp:.1f}%) |")
    L.append("")

    # Simple numeric summary of duration + wps distributions
    def _summary(name, rows):
        d = np.array([r["duration"] for r in rows])
        w = np.array([r["wps"] for r in rows])
        return (f"| {name} | {len(rows)} | {d.min():.2f} / {np.median(d):.2f} / "
                f"{d.mean():.2f} / {d.max():.2f} "
                f"| {w.min():.2f} / {np.median(w):.2f} / "
                f"{w.mean():.2f} / {w.max():.2f} |")

    L.append("## Duration + WPS summary (min / median / mean / max)")
    L.append("")
    L.append("| set | n | duration (s) | WPS |")
    L.append("|---|---:|---|---|")
    L.append(_summary("ROG", rog_rows))
    L.append(_summary("Matched", matched_rows))
    L.append("")

    # Deficits
    if rpt["deficits"]:
        L.append("## Deficits — bins where ParlaSpeech ran dry")
        L.append("")
        L.append("| label | dur_bin | wps_bin | ROG asked | Parla had |")
        L.append("|---|---:|---:|---:|---:|")
        for (lab, d_b, w_b), need, got in rpt["deficits"]:
            L.append(f"| {lab} | {d_b} | {w_b} | {need} | {got} |")
        L.append("")
        total_asked = sum(n for _, n, _ in rpt["deficits"])
        total_got = sum(g for _, _, g in rpt["deficits"])
        L.append(f"Total deficit across these bins: **{total_asked - total_got}** instances.")
    else:
        L.append("## Deficits")
        L.append("")
        L.append("None. ParlaSpeech had enough rows in every (label, dur_bin, wps_bin) ROG asked for.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rog", type=Path, default=ROG_PATH)
    ap.add_argument("--parla", type=Path, default=PARLA_PATH)
    ap.add_argument("--out", type=Path, default=OUT_JSONL)
    ap.add_argument("--report", type=Path, default=OUT_REPORT)
    ap.add_argument("--n-dur-bins", type=int, default=5)
    ap.add_argument("--n-wps-bins", type=int, default=5)
    ap.add_argument("--wps-clip", type=float, default=10.0,
                    help="Clip WPS at this value before binning (ROG has outliers >100 wps).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading ROG from {args.rog}")
    rog_rows = load_rog(args.rog)
    print(f"  {len(rog_rows)} ROG rows with labels")

    print(f"Loading ParlaSpeech from {args.parla}")
    parla_rows, kept, dropped = load_parla(args.parla)
    print(f"  {kept} ParlaSpeech rows kept · {dropped} dropped (missing audio_length / label)")

    sampled, rpt = build_match(
        rog_rows, parla_rows,
        n_dur=args.n_dur_bins, n_wps=args.n_wps_bins,
        wps_clip=args.wps_clip, seed=args.seed,
    )

    # Write out matched subset — slim/flat schema:
    #   {id, split, label_3, duration, wps, text, sentiment_logit, sentiment_6,
    #    audio_start, audio_end, audio}
    # Consumed by `_reader_parlaspeech_si_matched_to_rog` in utils_text_finetune.py
    # (flat sentiment_logit, not nested sentiment.*).
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in sampled:
            f.write(json.dumps(r["_slim"], ensure_ascii=False) + "\n")
    print(f"Wrote {len(sampled)} matched rows (slim schema) to {args.out}")

    # Report
    md = render_report(rpt, sampled, rog_rows, len(parla_rows))
    args.report.write_text(md)
    print(f"Wrote report to {args.report}")

    # Console summary
    print(f"\nROG size: {rpt['n_rog']} · matched: {rpt['n_matched']} ({100*rpt['n_matched']/rpt['n_rog']:.1f}%)")
    print(f"ROG     label prior: {rpt['rog_label_prior']}")
    print(f"Matched label prior: {rpt['matched_label_prior']}")
    print(f"ROG     duration median/mean: "
          f"{np.median([r['duration'] for r in rog_rows]):.2f}s / "
          f"{np.mean([r['duration'] for r in rog_rows]):.2f}s")
    print(f"Matched duration median/mean: "
          f"{np.median([r['duration'] for r in sampled]):.2f}s / "
          f"{np.mean([r['duration'] for r in sampled]):.2f}s")


if __name__ == "__main__":
    main()
