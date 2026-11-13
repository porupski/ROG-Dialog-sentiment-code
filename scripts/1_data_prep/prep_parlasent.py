#!/usr/bin/env python3
"""Download and convert ParlaSent 1.0 into our canonical per-instance JSONL schema.

ParlaSent 1.0 is a 7-language parliamentary-sentence corpus manually annotated with
a six-level sentiment schema (Batanović et al. 2020). Source: CLARIN.SI handle 11356/1868.

The six raw labels per Mochtak et al. 2023 §3.4:
  Negative   (−1)  predominantly negative
  M_Negative (−M)  mixed, leans negative
  N_Neutral  (−NS) non-sentiment statement, leans negative
  P_Neutral  (+NS) non-sentiment statement, leans positive
  M_Positive (+M)  mixed, leans positive
  Positive   (+1)  predominantly positive

The published `label` field is the **3-class** collapse ("neutral / negative / positive"),
NOT a 6-level label — so reading it as 6-level drops ~half the corpus. The 6-level signal
lives in `reconciliation` (populated for all training rows, consensus after disagreement
discussion) with `annotator1` as the fallback on the _test.jsonl files that ship with a
single annotator only.

Downloads 8 files (~7.4 MB total) into misc/parlasent_raw/, then emits one canonical
JSONL per language into data/ in the same shape ROG-Dialog v2 uses, so
4_finetune_text/finetune_text.py / analyze_results.py can treat them uniformly.

Schema mapping (source -> canonical):
  sentence                       -> text
  country                        -> country (preserved as-is)
  reconciliation|annotator1      -> sentiment      (predominantlyNegative ... predominantlyPositive)
                                 + sentiment_3    (negative / neutral / positive)
                                 + sentiment_6_ord (0..5)
                                 + sentiment_3_ord (0..2)
  (published `label`)            -> metadata.published_3class_label (kept for audit)
  split                          -> split  (train / dev / test; already present per row)
  document_id                    -> file_id
  sentence_id                    -> plus document_id into instance_id
  name/party/gender/age/etc.     -> preserved under metadata

Multi-run safe: existing downloads are skipped unless --redownload is passed.
Idempotent conversion: existing <lang>_instance.jsonl is overwritten on each run.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

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
    "Negative":   "predominantlyNegative",   # −1
    "M_Negative": "mixedNegative",           # −M
    "N_Neutral":  "neutralNegative",         # −NS (non-sentiment, leans negative)
    "P_Neutral":  "neutralPositive",         # +NS (non-sentiment, leans positive)
    "M_Positive": "mixedPositive",           # +M
    "Positive":   "predominantlyPositive",   # +1
}

# Fields to try, in order, when sourcing the 6-level label. Training files carry
# `reconciliation` (consensus after disagreement discussion); the single-annotator
# _test files only carry `annotator1`.
LABEL_6_FIELDS = ("reconciliation", "annotator1", "annotator2")
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


def pick_six_level(r: dict) -> tuple[str | None, str | None]:
    """Return (raw_6level_label, source_field) using LABEL_6_FIELDS fallback order.
    Only accepts values present in LABEL_6_MAP so a stray 3-class string (e.g. the
    published `label`) can't accidentally shadow the real 6-level field."""
    for field in LABEL_6_FIELDS:
        v = r.get(field)
        if isinstance(v, str) and v in LABEL_6_MAP:
            return v, field
    return None, None


def convert_one(src: Path, out: Path, out_stem: str) -> dict:
    """Emit the canonical per-instance JSONL. Returns a small summary dict."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "total": 0,
        "skipped_no_6level": 0,
        "per_split": {"train": 0, "dev": 0, "test": 0, "other": 0},
        "label_source_counts": {f: 0 for f in LABEL_6_FIELDS},
    }
    with open(src) as fin, open(out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            r = json.loads(line)
            stats["total"] += 1

            raw_6, src_field = pick_six_level(r)
            if raw_6 is None:
                stats["skipped_no_6level"] += 1
                continue
            stats["label_source_counts"][src_field] += 1

            sent6 = LABEL_6_MAP[raw_6]
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
                    "label_6_raw": raw_6,
                    "label_6_source": src_field,
                    "published_3class_label": r.get("label"),
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
        kept = stats["total"] - stats["skipped_no_6level"]
        summary[out_stem] = {
            "rows": kept,
            "splits": stats["per_split"],
            "skipped_no_6level": stats["skipped_no_6level"],
            "label_source_counts": stats["label_source_counts"],
        }
        src_counts = stats["label_source_counts"]
        print(f"  {out_stem}: kept={kept} "
              f"(train={stats['per_split']['train']}, dev={stats['per_split']['dev']}, "
              f"test={stats['per_split']['test']}, other={stats['per_split']['other']})"
              f"  skipped_no_6level={stats['skipped_no_6level']}"
              f"  source=rec:{src_counts['reconciliation']}/"
              f"a1:{src_counts['annotator1']}/a2:{src_counts['annotator2']}")

    report = OUT_DIR / "parlasent_prep_report.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {report.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
