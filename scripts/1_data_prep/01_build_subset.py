#!/usr/bin/env python3
"""Build the 174-instance eval subset and 18-instance few-shot pool for the ROG-Dialog sentiment probe.

Rules:
- Keep only instances with 2.0 <= (end_t - start_t) <= 20.0 and a non-null `labels.sentiment`.
- Eval subset: 29 instances per 6-class label (174 total), drawn from dev+test, round-robin by speaker.
- Few-shot pool: 3 instances per 6-class label (18 total), drawn from train, 3 distinct speakers per class.
- Collapsed 3-class label (`sentiment_3`) added to every emitted record using the ParlaSent-style collapse.
- Seed fixed at 42 for all random choices.
- Splits are speaker-disjoint, so train <-> dev+test speaker contamination is structurally impossible.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# Defaults are v2 now. Pass --src data/rog_dialog_instance.jsonl --suffix "" to rebuild v1.
SRC = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
OUT_SUBSET = ROOT / "data" / "subset_174_seed42_v2.jsonl"
OUT_FEWSHOT = ROOT / "data" / "fewshot_pool_18_seed42_v2.jsonl"
OUT_REPORT = ROOT / "data" / "subset_report_v2.txt"
SCRIPT_LOG = ROOT / "runs" / "_script_log.jsonl"

SEED = 42
N_PER_CLASS_EVAL = 29
N_PER_CLASS_FEWSHOT = 3
MIN_DUR = 2.0
MAX_DUR = 20.0

LABELS_6 = [
    "predominantlyNegative",
    "mixedNegative",
    "neutralNegative",
    "neutralPositive",
    "mixedPositive",
    "predominantlyPositive",
]

SENT_6_TO_3 = {
    "predominantlyNegative": "negative",
    "mixedNegative": "negative",
    "neutralNegative": "neutral",
    "neutralPositive": "neutral",
    "mixedPositive": "positive",
    "predominantlyPositive": "positive",
}

SENT_6_ORDER = {lbl: i for i, lbl in enumerate(LABELS_6)}
SENT_3_ORDER = {"negative": 0, "neutral": 1, "positive": 2}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_filtered(path):
    kept, dropped = [], Counter()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            s = r.get("labels", {}).get("sentiment")
            if s is None:
                dropped["null_sentiment"] += 1
                continue
            d = r["end_t"] - r["start_t"]
            if d < MIN_DUR:
                dropped["too_short"] += 1
                continue
            if d > MAX_DUR:
                dropped["too_long"] += 1
                continue
            r["duration"] = d
            kept.append(r)
    return kept, dropped


def speaker_stratified(instances, n, seed):
    """Round-robin picks across speakers (shuffled), then within-speaker shuffled. Max diversity under n."""
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for inst in instances:
        by_spk[inst["speaker"]].append(inst)
    speakers = sorted(by_spk.keys())
    rng.shuffle(speakers)
    for spk in speakers:
        rng.shuffle(by_spk[spk])
    picked = []
    while len(picked) < n and any(by_spk[s] for s in speakers):
        for spk in speakers:
            if by_spk[spk]:
                picked.append(by_spk[spk].pop(0))
                if len(picked) == n:
                    break
    return picked


def fewshot_pick(instances, n, seed):
    """n instances from n distinct speakers (if possible). Deterministic."""
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for inst in instances:
        by_spk[inst["speaker"]].append(inst)
    speakers = sorted(by_spk.keys())
    rng.shuffle(speakers)
    picked = []
    for spk in speakers:
        rng.shuffle(by_spk[spk])
        picked.append(by_spk[spk][0])
        if len(picked) == n:
            break
    return picked


def emit(inst):
    """Project to a stable, trimmed record and add collapsed labels + ordered ints."""
    s6 = inst["labels"]["sentiment"]
    s3 = SENT_6_TO_3[s6]
    return {
        "instance_id": inst["instance_id"],
        "dataset": inst.get("dataset"),
        "file_id": inst.get("file_id"),
        "split": inst["split"],
        "speaker": inst["speaker"],
        "start_t": inst["start_t"],
        "end_t": inst["end_t"],
        "duration": round(inst["duration"], 3),
        "audio_path": inst.get("audio_path"),
        "text": inst["text"],
        "sentiment": s6,
        "sentiment_3": s3,
        "sentiment_6_ord": SENT_6_ORDER[s6],
        "sentiment_3_ord": SENT_3_ORDER[s3],
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def report(eval_records, fewshot_records, dropped, src_total):
    lines = []
    lines.append(f"ROG-Dialog sentiment probe — subset report")
    lines.append(f"generated: {now_iso()}")
    lines.append(f"source: {SRC}")
    lines.append(f"seed: {SEED}  duration_filter: [{MIN_DUR}, {MAX_DUR}] s")
    lines.append("")
    lines.append(f"source rows: {src_total}")
    lines.append(f"dropped: {dict(dropped)}")
    lines.append("")

    def dist(records, label="sentiment"):
        c = Counter(r[label] for r in records)
        return [(k, c[k]) for k in sorted(c, key=lambda x: SENT_6_ORDER.get(x, SENT_3_ORDER.get(x, 99)))]

    def speakers(records, label="sentiment"):
        by = defaultdict(set)
        by_c = defaultdict(Counter)
        for r in records:
            by[r[label]].add(r["speaker"])
            by_c[r[label]][r["speaker"]] += 1
        return by, by_c

    lines.append(f"=== EVAL SUBSET ({len(eval_records)}) ===")
    lines.append(f"file: {OUT_SUBSET.relative_to(ROOT)}")
    lines.append(f"split origin: " + str(dict(Counter(r["split"] for r in eval_records))))
    lines.append("per-6-class count:")
    for k, v in dist(eval_records, "sentiment"):
        lines.append(f"  {k:<24} {v}")
    lines.append("per-3-class count (collapsed):")
    for k, v in dist(eval_records, "sentiment_3"):
        lines.append(f"  {k:<24} {v}")
    by_spk, by_spk_c = speakers(eval_records, "sentiment")
    lines.append("per-6-class unique-speakers / max-single-speaker-contrib:")
    for k in LABELS_6:
        spks = by_spk[k]
        counts = by_spk_c[k]
        max_c = max(counts.values()) if counts else 0
        lines.append(f"  {k:<24} speakers={len(spks)} max_from_one_spk={max_c}")
    lines.append("")

    lines.append(f"=== FEW-SHOT POOL ({len(fewshot_records)}) ===")
    lines.append(f"file: {OUT_FEWSHOT.relative_to(ROOT)}")
    lines.append(f"split origin: " + str(dict(Counter(r["split"] for r in fewshot_records))))
    lines.append("per-6-class count:")
    for k, v in dist(fewshot_records, "sentiment"):
        lines.append(f"  {k:<24} {v}")
    lines.append("per-6-class speakers in pool:")
    by_spk, _ = speakers(fewshot_records, "sentiment")
    for k in LABELS_6:
        lines.append(f"  {k:<24} {sorted(by_spk[k])}")
    lines.append("")

    all_eval_spks = set(r["speaker"] for r in eval_records)
    all_pool_spks = set(r["speaker"] for r in fewshot_records)
    overlap = all_eval_spks & all_pool_spks
    lines.append(f"speaker overlap eval <-> fewshot: {len(overlap)}  (expected 0)")
    if overlap:
        lines.append(f"  OVERLAP: {sorted(overlap)}")
    return "\n".join(lines) + "\n"


def main():
    global OUT_SUBSET, OUT_FEWSHOT, OUT_REPORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC),
                    help=f"source JSONL (default: {SRC.name})")
    ap.add_argument("--out-subset", default=str(OUT_SUBSET))
    ap.add_argument("--out-fewshot", default=str(OUT_FEWSHOT))
    ap.add_argument("--out-report", default=str(OUT_REPORT))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT_SUBSET = Path(args.out_subset)
    OUT_FEWSHOT = Path(args.out_fewshot)
    OUT_REPORT = Path(args.out_report)

    started = now_iso()
    src = Path(args.src)

    with open(src) as f:
        src_total = sum(1 for _ in f)

    all_rows, dropped = load_filtered(src)

    by_class_eval_pool = defaultdict(list)
    by_class_train_pool = defaultdict(list)
    for r in all_rows:
        s = r["labels"]["sentiment"]
        if r["split"] in ("dev", "test"):
            by_class_eval_pool[s].append(r)
        elif r["split"] == "train":
            by_class_train_pool[s].append(r)

    eval_records, fewshot_records = [], []
    for i, lbl in enumerate(LABELS_6):
        pool = by_class_eval_pool.get(lbl, [])
        picked = speaker_stratified(pool, N_PER_CLASS_EVAL, seed=args.seed + i)
        if len(picked) < N_PER_CLASS_EVAL:
            print(f"WARN: class {lbl} yielded only {len(picked)} < {N_PER_CLASS_EVAL}")
        eval_records.extend(emit(r) for r in picked)

        train_pool = by_class_train_pool.get(lbl, [])
        few = fewshot_pick(train_pool, N_PER_CLASS_FEWSHOT, seed=args.seed + 1000 + i)
        if len(few) < N_PER_CLASS_FEWSHOT:
            print(f"WARN: few-shot class {lbl} yielded only {len(few)} < {N_PER_CLASS_FEWSHOT}")
        fewshot_records.extend(emit(r) for r in few)

    # Shuffle eval order (deterministic) so --test picks span classes instead of taking all
    # 29 from the first class. Few-shot pool is left in per-class order so prompt order is stable.
    random.Random(args.seed + 9999).shuffle(eval_records)

    write_jsonl(OUT_SUBSET, eval_records)
    write_jsonl(OUT_FEWSHOT, fewshot_records)

    rpt = report(eval_records, fewshot_records, dropped, src_total)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(rpt)
    print(rpt)

    SCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SCRIPT_LOG, "a") as f:
        f.write(json.dumps({
            "script": "01_build_subset.py",
            "ts_start": started,
            "ts_end": now_iso(),
            "seed": args.seed,
            "src": str(src),
            "n_eval": len(eval_records),
            "n_fewshot": len(fewshot_records),
            "out_subset": str(OUT_SUBSET),
            "out_fewshot": str(OUT_FEWSHOT),
            "out_report": str(OUT_REPORT),
            "dropped": dict(dropped),
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
