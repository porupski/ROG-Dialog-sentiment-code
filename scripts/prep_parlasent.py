#!/usr/bin/env python3
"""Download and convert ParlaSent 1.0 into our canonical per-instance JSONL schema.

ParlaSent 1.0 is a 7-language parliamentary-sentence corpus manually annotated with
a 6-level sentiment schema (Negative / M_Negative / N_Negative / N_Positive /
M_Positive / Positive). Source: CLARIN.SI handle 11356/1868.

Downloads 8 files (~7.4 MB total) into misc/parlasent_raw/, then emits one
canonical JSONL per language into data/ in the same shape ROG-Dialog v2 uses,
so 06_finetune_text.py / 03_analyze_results.py can treat them uniformly.

Schema mapping (source -> canonical):
  sentence        -> text
  country         -> country (preserved as-is; "HR"/"SR"/"BA" in the BCS pool, etc.)
  label           -> sentiment  (predominantlyNegative … predominantlyPositive)
                   + sentiment_3 (negative / neutral / positive, same collapse as 01)
                   + sentiment_6_ord (0..5)
                   + sentiment_3_ord (0..2)
  split           -> split  (train / dev / test; already present per row)
  document_id     -> file_id
  sentence_id     -> plus document_id into instance_id
  name/party/     -> preserved under metadata
  gender/age/etc.

Multi-run safe: existing downloads are skipped unless --redownload is passed.
Idempotent conversion: existing <lang>_instance.jsonl is overwritten on each run
(cheap) — rerun whenever label mapping evolves.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

RAW_DIR = ROOT / "misc" / "parlasent_raw"
OUT_DIR = ROOT / "data"

BASE_URL = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1868"

# One stem per source file; the per-language canonical output stem is derived from it.
# BCS and EN each ship an extra held-out test file alongside the main train/dev/test one.
SOURCES = [
    # (source_filename, out_stem)
    ("ParlaSent_SL.jsonl",       "parlasent_sl"),
    ("ParlaSent_BCS.jsonl",      "parlasent_bcs"),
    ("ParlaSent_CZ.jsonl",       "parlasent_cz"),
    ("ParlaSent_EN.jsonl",       "parlasent_en"),
    ("ParlaSent_SK.jsonl",       "parlasent_sk"),
    ("ParlaSent_BCS_test.jsonl", "parlasent_bcs_test_extra"),
    ("ParlaSent_EN_test.jsonl",  "parlasent_en_test_extra"),
]
# README is downloaded but not converted.
AUX = ["README.txt"]

LABEL_6_MAP = {
    "Negative":   "predominantlyNegative",
    "M_Negative": "mixedNegative",
    "N_Negative": "neutralNegative",
    "N_Positive": "neutralPositive",
    "M_Positive": "mixedPositive",
    "Positive":   "predominantlyPositive",
}
LABELS_6_ORDER = [
    "predominantlyNegative", "mixedNegative", "neutralNegative",
    "neutralPositive",       "mixedPositive", "predominantlyPositive",
]
SENT_6_TO_3 = {
    "predominantlyNegative": "negative", "mixedNegative": "negative",
    "neutralNegative":       "neutral",  "neutralPositive": "neutral",
    "mixedPositive":         "positive", "predominantlyPositive": "positive",
}
LABELS_3_ORDER = ["negative", "neutral", "positive"]

SENT_6_ORD = {l: i for i, l in enumerate(LABELS_6_ORDER)}
SENT_3_ORD = {l: i for i, l in enumerate(LABELS_3_ORDER)}


def download(fname: str, dest: Path, redownload: bool) -> None:
    out = dest / fname
    if out.exists() and not redownload:
        print(f"  [skip] {fname} ({out.stat().st_size:,} B)")
        return
    url = f"{BASE_URL}/{fname}"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [get]  {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(out, "wb") as f:
        f.write(r.read())
    print(f"         -> {out} ({out.stat().st_size:,} B)")


def convert_one(src: Path, out: Path, out_stem: str) -> dict:
    """Emit the canonical per-instance JSONL. Returns a small summary dict."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "skipped_null_label": 0, "unknown_label": 0,
             "per_split": {"train": 0, "dev": 0, "test": 0, "other": 0}}
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            stats["total"] += 1
            label = r.get("label")
            if label is None:
                stats["skipped_null_label"] += 1
                continue
            sent6 = LABEL_6_MAP.get(label)
            if sent6 is None:
                stats["unknown_label"] += 1
                continue
            sent3 = SENT_6_TO_3[sent6]
            split = r.get("split", "other")
            if split not in stats["per_split"]:
                split = "other"
            stats["per_split"][split] += 1

            doc_id = r.get("document_id", "?")
            sent_id = r.get("sentence_id", "?")
            instance_id = f"ParlaSent_{out_stem}_{doc_id}_{sent_id}"

            fout.write(json.dumps({
                "instance_id": instance_id,
                "dataset": "ParlaSent",
                "source_stem": out_stem,
                "file_id": str(doc_id),
                "sentence_id": sent_id,
                "country": r.get("country"),
                "split": split,
                "text": r["sentence"],
                "sentiment": sent6,
                "sentiment_3": sent3,
                "sentiment_6_ord": SENT_6_ORD[sent6],
                "sentiment_3_ord": SENT_3_ORD[sent3],
                "metadata": {
                    "label_raw": label,
                    "reconciliation": r.get("reconciliation"),
                    "annotator1": r.get("annotator1"),
                    "annotator2": r.get("annotator2"),
                    "term": r.get("term"),
                    "date": r.get("date"),
                    "speaker_name": r.get("name"),
                    "party": r.get("party"),
                    "gender": r.get("gender"),
                    "birth_year": r.get("birth_year"),
                    "ruling": r.get("ruling"),
                },
            }, ensure_ascii=False) + "\n")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redownload", action="store_true",
                    help="Re-fetch source files even if they exist")
    ap.add_argument("--skip-download", action="store_true",
                    help="Convert whatever is already in misc/parlasent_raw/")
    args = ap.parse_args()

    if not args.skip_download:
        print(f"Downloading {len(SOURCES) + len(AUX)} files to {RAW_DIR.relative_to(ROOT)}/")
        for src, _ in SOURCES:
            download(src, RAW_DIR, args.redownload)
        for aux in AUX:
            download(aux, RAW_DIR, args.redownload)
        print()

    print(f"Converting to canonical per-instance JSONL in {OUT_DIR.relative_to(ROOT)}/")
    summary = {}
    for src_name, out_stem in SOURCES:
        src = RAW_DIR / src_name
        if not src.exists():
            print(f"  [warn] missing {src}; run without --skip-download first")
            continue
        out = OUT_DIR / f"{out_stem}_instance.jsonl"
        stats = convert_one(src, out, out_stem)
        summary[out_stem] = {"rows": stats["total"] - stats["skipped_null_label"] - stats["unknown_label"],
                             "splits": stats["per_split"],
                             "skipped_null_label": stats["skipped_null_label"],
                             "unknown_label": stats["unknown_label"]}
        s = summary[out_stem]
        print(f"  {out_stem}: kept={s['rows']} "
              f"(train={s['splits']['train']}, dev={s['splits']['dev']}, "
              f"test={s['splits']['test']}, other={s['splits']['other']})"
              f"  dropped_null={s['skipped_null_label']}  unknown_lbl={s['unknown_label']}")

    report = OUT_DIR / "parlasent_prep_report.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
