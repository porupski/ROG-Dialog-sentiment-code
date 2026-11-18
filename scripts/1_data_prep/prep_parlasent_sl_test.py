#!/usr/bin/env python3
"""Emit data/subset_parlasent_sl_test.jsonl — the ParlaSent-SL TEST split in the
flat `subset_174_seed42_v2.jsonl` schema so 04_run_parlasent.py / analyze_probe.py
consume it via --eval-set without any code change.

Source: data/parlasent_sl_instance.jsonl, filtered to split=='test' AND
`sentiment` present.

No audio / no duration — those fields are emitted as None so the OR runners
degrade gracefully (02_/03_ pass duration through; the topography plotter
treats None duration → 1-D envelope over word count only).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "data" / "parlasent_sl_instance.jsonl"
OUT = ROOT / "data" / "subset_parlasent_sl_test.jsonl"

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
        s = r.get("sentiment")
        if s is None or s not in SENT_6_ORD:
            continue
        iid = r["instance_id"]
        if iid in seen:
            continue
        seen.add(iid)
        s3 = SENT_6_TO_3[s]
        rows_out.append({
            "instance_id":      iid,
            "dataset":          r.get("dataset", "ParlaSent"),
            "file_id":          r.get("file_id"),
            "split":            r["split"],
            "speaker":          (r.get("metadata") or {}).get("speaker_name"),
            "start_t":          None,
            "end_t":             None,
            "duration":         None,
            "audio_path":       None,
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
