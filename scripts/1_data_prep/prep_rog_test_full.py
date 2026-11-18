#!/usr/bin/env python3
"""Emit data/subset_rog_test_full.jsonl — the full ROG v2 test split in the flat
`subset_174_seed42_v2.jsonl` schema so the OR probe runners (02_, 03_) consume it
via --eval-set without any code change.

Source: data/rog_dialog_v2_instance.jsonl, filtered to split=='test' AND
labels.sentiment present. No duration filter (natural-distribution eval set).

Row shape (fields that existing runners / analyze_probe.py / topography tools read):
  instance_id, dataset, file_id, split, speaker, start_t, end_t, duration,
  audio_path, text, sentiment, sentiment_3, sentiment_6_ord, sentiment_3_ord

The order is deterministic: original corpus order, then stable-sorted so --n-examples
caps aren't class-biased. We don't shuffle — the OR runners read sequentially and for
a full-test eval there's no subset to balance.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
OUT = ROOT / "data" / "subset_rog_test_full.jsonl"

LABELS_6 = [
    "predominantlyNegative",
    "mixedNegative",
    "neutralNegative",
    "neutralPositive",
    "mixedPositive",
    "predominantlyPositive",
]
SENT_6_TO_3 = {
    "predominantlyNegative": "negative", "mixedNegative": "negative",
    "neutralNegative": "neutral",        "neutralPositive": "neutral",
    "mixedPositive": "positive",         "predominantlyPositive": "positive",
}
SENT_6_ORD = {l: i for i, l in enumerate(LABELS_6)}
SENT_3_ORD = {"negative": 0, "neutral": 1, "positive": 2}


def main():
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    rows_out = []
    seen = set()
    for line in open(SRC):
        r = json.loads(line)
        if r.get("split") != "test":
            continue
        labs = r.get("labels") or {}
        s = labs.get("sentiment")
        if s is None or s not in SENT_6_ORD:
            continue
        iid = r["instance_id"]
        if iid in seen:
            continue
        seen.add(iid)
        s3 = SENT_6_TO_3[s]
        dur = r["end_t"] - r["start_t"]
        rows_out.append({
            "instance_id":      iid,
            "dataset":          r.get("dataset", "ROG-Dialog"),
            "file_id":          r["file_id"],
            "split":            r["split"],
            "speaker":          r["speaker"],
            "start_t":          r["start_t"],
            "end_t":            r["end_t"],
            "duration":         round(dur, 3),
            "audio_path":       r.get("audio_path"),
            "text":             r["text"],
            "sentiment":        s,
            "sentiment_3":      s3,
            "sentiment_6_ord":  SENT_6_ORD[s],
            "sentiment_3_ord":  SENT_3_ORD[s3],
        })

    with open(OUT, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    c6 = Counter(r["sentiment"] for r in rows_out)
    c3 = Counter(r["sentiment_3"] for r in rows_out)
    print(f"wrote {OUT.relative_to(ROOT)}  n={len(rows_out)}")
    print(f"  6-cls: {dict(c6)}")
    print(f"  3-cls: {dict(c3)}")


if __name__ == "__main__":
    main()
