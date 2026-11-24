#!/usr/bin/env python3
"""04_compute_full_rog_iaa — IAA (annotator ↔ curator) on the full ROG test split.

The 170-subset IAA = 0.699 was the headline ceiling. This script computes the
honest full-ROG ceiling by comparing `labels.sentiment` (final/curator) vs
`labels.sentiment_annotated` (raw annotator) on the 2233-row test split.

3-class collapse mapping:
    {predominantlyNegative, mixedNegative} → negative
    {neutralNegative, neutralPositive}     → neutral
    {mixedPositive, predominantlyPositive} → positive

Usage:
    mamba run -n ssp python3 scripts/6_sent_duration/04_compute_full_rog_iaa.py \\
        --rog_jsonl data/rog_dialog_v2_instance.jsonl \\
        --split test
"""

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
except ImportError:
    print("ERROR: sklearn not available — mamba run -n ssp")
    raise


COLLAPSE_6_TO_3 = {
    "predominantlyNegative": "negative",
    "mixedNegative":          "negative",
    "neutralNegative":        "neutral",
    "neutralPositive":        "neutral",
    "mixedPositive":          "positive",
    "predominantlyPositive":  "positive",
}
CLASSES_3 = ["negative", "neutral", "positive"]
CLASSES_6 = ["predominantlyNegative", "mixedNegative", "neutralNegative",
             "neutralPositive", "mixedPositive", "predominantlyPositive"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rog_jsonl", required=True)
    p.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    return p.parse_args()


def iaa_block(gold_6, pred_6, label):
    gold_3 = [COLLAPSE_6_TO_3[g] for g in gold_6]
    pred_3 = [COLLAPSE_6_TO_3[p] for p in pred_6]

    f1_3 = f1_score(gold_3, pred_3, labels=CLASSES_3, average="macro", zero_division=0)
    f1_6 = f1_score(gold_6, pred_6, labels=CLASSES_6, average="macro", zero_division=0)
    kappa_3 = cohen_kappa_score(gold_3, pred_3, labels=CLASSES_3)
    kappa_6 = cohen_kappa_score(gold_6, pred_6, labels=CLASSES_6)
    agr_3 = sum(a == b for a, b in zip(gold_3, pred_3)) / len(gold_3)
    agr_6 = sum(a == b for a, b in zip(gold_6, pred_6)) / len(gold_6)

    print(f"=== {label} (n={len(gold_6)}) ===")
    print(f"  F1_3 macro:   {f1_3:.3f}")
    print(f"  F1_6 macro:   {f1_6:.3f}")
    print(f"  κ (3-class):  {kappa_3:.3f}")
    print(f"  κ (6-class):  {kappa_6:.3f}")
    print(f"  raw agr (3):  {agr_3:.3f}  ({sum(a==b for a,b in zip(gold_3,pred_3))}/{len(gold_3)})")
    print(f"  raw agr (6):  {agr_6:.3f}  ({sum(a==b for a,b in zip(gold_6,pred_6))}/{len(gold_6)})")
    print()
    return {"label": label, "n": len(gold_6),
            "f1_3": f1_3, "f1_6": f1_6,
            "kappa_3": kappa_3, "kappa_6": kappa_6,
            "agr_3": agr_3, "agr_6": agr_6}


def main():
    args = parse_args()
    rog_path = Path(args.rog_jsonl).resolve()
    rows = [json.loads(l) for l in open(rog_path, encoding="utf-8")]
    if args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]
    print(f"Loaded {len(rows)} rows from {rog_path.name} (split={args.split})\n")

    paired = [(r["labels"].get("sentiment"), r["labels"].get("sentiment_annotated"))
              for r in rows
              if r["labels"].get("sentiment") in CLASSES_6
              and r["labels"].get("sentiment_annotated") in CLASSES_6]
    print(f"paired rows (both labels present and valid 6-class): {len(paired)}")
    print(f"dropped (missing/invalid): {len(rows) - len(paired)}\n")

    curator = [p[0] for p in paired]
    annotator = [p[1] for p in paired]

    print("Label distribution (curator):")
    for k, v in Counter(curator).most_common():
        print(f"  {k:30s} {v:6d}")
    print("Label distribution (annotator):")
    for k, v in Counter(annotator).most_common():
        print(f"  {k:30s} {v:6d}")
    print()

    result = iaa_block(curator, annotator,
                       f"IAA — curator (gold) vs annotator (raw), {args.split} split")

    diff_count = sum(c != a for c, a in zip(curator, annotator))
    print(f"Rows where curator ≠ annotator: {diff_count} ({100*diff_count/len(paired):.1f}%)")
    print()

    curator_3 = [COLLAPSE_6_TO_3[c] for c in curator]
    annotator_3 = [COLLAPSE_6_TO_3[a] for a in annotator]
    print("3-class confusion matrix (rows=annotator, cols=curator):")
    cm = confusion_matrix(annotator_3, curator_3, labels=CLASSES_3)
    print(f"              {' '.join(f'{c:>10s}' for c in CLASSES_3)}")
    for i, row_name in enumerate(CLASSES_3):
        print(f"  {row_name:12s} {' '.join(f'{v:>10d}' for v in cm[i])}")

    project_root = Path(__file__).resolve().parent.parent.parent  # scripts/6_.../x.py → project
    out_path = project_root / "runs" / "baselines" / f"iaa_fullrog_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
