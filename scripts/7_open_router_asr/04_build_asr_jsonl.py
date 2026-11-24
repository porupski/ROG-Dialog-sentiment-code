#!/usr/bin/env python3
"""04_build_asr_jsonl — merge Chirp3 ASR transcripts into a ROG-style JSONL.

Emits one row per original instance, with:
  - `text` ← ASR transcript (so Reports 2 + 4 pipelines consume ASR by default)
  - `text_gold` ← original gold text (kept for diff / scoring)
  - `asr_wer_norm`, `asr_cer_norm`, `overlap`, `overlap_frac` ← joined from enriched metrics
  - everything else carried over unchanged (instance_id, split, labels, audio_path, ...)

Rows where the ASR failed or returned empty transcript keep `text` as empty string
(downstream scripts must tolerate empty text — or filter on `asr_wer_norm == NaN`).

Usage:
    mamba run -n ssp python3 scripts/7_open_router_asr/04_build_asr_jsonl.py \\
        --predictions runs/from_remote/7_asr_run/chirp3_fullrog_2026-11-23_145526/predictions_enriched.jsonl \\
        --rog_jsonl data/rog_dialog_v2_instance.jsonl \\
        --out data/rog_dialog_v2_instance_ASR.jsonl
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--predictions", required=True,
                   help="predictions_enriched.jsonl from 03_compute_metrics.py")
    p.add_argument("--rog_jsonl", required=True,
                   help="source rog_dialog_v2_instance.jsonl")
    p.add_argument("--out", required=True,
                   help="output JSONL (text=ASR, text_gold=gold)")
    p.add_argument("--also_subset_schema",
                   help="optional: also emit an OR-pipeline flat-schema variant, "
                   "reading the schema template from this subset_rog_test_full.jsonl "
                   "and writing <stem>_ASR.jsonl next to it")
    return p.parse_args()


def main():
    args = parse_args()
    pred_path = Path(args.predictions).resolve()
    rog_path = Path(args.rog_jsonl).resolve()
    out_path = Path(args.out).resolve()

    preds = {p["instance_id"]: p
             for p in (json.loads(l) for l in open(pred_path, encoding="utf-8"))}
    print(f"Loaded {len(preds)} ASR predictions from {pred_path.name}")

    rog = [json.loads(l) for l in open(rog_path, encoding="utf-8")]
    print(f"Loaded {len(rog)} ROG rows from {rog_path.name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = n_missing = n_empty = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rog:
            iid = r["instance_id"]
            p = preds.get(iid)
            if p is None:
                n_missing += 1
                new_row = dict(r)
                new_row["text_gold"] = r.get("text", "")
                new_row["text"] = ""
                new_row["asr_status"] = "missing"
                new_row["asr_wer_norm"] = None
                new_row["asr_cer_norm"] = None
                new_row["overlap"] = "unknown"
                new_row["overlap_frac"] = None
            else:
                asr_text = (p.get("transcript") or "").strip()
                if not asr_text:
                    n_empty += 1
                new_row = dict(r)
                new_row["text_gold"] = r.get("text", "")
                new_row["text"] = asr_text
                new_row["asr_status"] = p.get("status") or ("empty" if not asr_text else "ok")
                new_row["asr_wer_norm"] = p.get("wer_norm")
                new_row["asr_cer_norm"] = p.get("cer_norm")
                new_row["overlap"] = p.get("overlap")
                new_row["overlap_frac"] = p.get("overlap_frac")
            f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\nWrote {n_written} rows to {out_path}")
    print(f"  missing from predictions: {n_missing}")
    print(f"  empty ASR text:           {n_empty}")

    scored = [p for p in preds.values()
              if p.get("wer_norm") is not None
              and p.get("overlap") in ("no", "won")]
    if scored:
        wn = [p["wer_norm"] for p in scored if p.get("wer_norm") == p.get("wer_norm")]
        print(f"  priority WER_norm (reference, from enriched): "
              f"mean {sum(wn)/len(wn):.3f} over n={len(wn)}")

    if args.also_subset_schema:
        tmpl_path = Path(args.also_subset_schema).resolve()
        if not tmpl_path.exists():
            print(f"\nWARNING: --also_subset_schema path {tmpl_path} not found, skipping")
            return
        tmpl = [json.loads(l) for l in open(tmpl_path, encoding="utf-8")]
        asr_sub = tmpl_path.parent / f"{tmpl_path.stem}_ASR.jsonl"
        n_sub_written = n_sub_empty = n_sub_missing = 0
        with open(asr_sub, "w", encoding="utf-8") as f:
            for row in tmpl:
                iid = row["instance_id"]
                p = preds.get(iid)
                new_row = dict(row)
                new_row["text_gold"] = row.get("text", "")
                if p is None:
                    new_row["text"] = ""
                    n_sub_missing += 1
                else:
                    asr = (p.get("transcript") or "").strip()
                    new_row["text"] = asr
                    if not asr:
                        n_sub_empty += 1
                new_row["asr_wer_norm"] = p.get("wer_norm") if p else None
                new_row["overlap"] = p.get("overlap") if p else "unknown"
                f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                n_sub_written += 1
        print(f"\nAlso wrote subset-schema ASR variant to {asr_sub}")
        print(f"  rows: {n_sub_written} · missing: {n_sub_missing} · empty ASR: {n_sub_empty}")


if __name__ == "__main__":
    main()
