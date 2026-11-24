#!/usr/bin/env python3
"""04_build_asr_jsonl — merge Chirp3 ASR transcripts into a ROG-style JSONL.

Emits one row per original instance with THREE text fields, all explicit:
  - `text`     ← ASR transcript (= `text_asr`); default for downstream pipelines
  - `text_asr` ← same as `text`, explicit alias for readability
  - `text_gold`← original gold text (kept for diff / scoring)

Plus the ASR metrics joined in:
  - `asr_wer_norm`, `asr_cer_norm`, `asr_status`
  - `overlap` ∈ {no, won, lost, unknown}, `overlap_frac` ∈ [0,1]

Everything else carried over unchanged (instance_id, split, labels, audio_path,
metadata.colloq_text, ...). `colloq_text` is unrelated — it's the dialect variant
from the ROG EXB source, not ASR.

Rows with empty ASR transcript keep `text`/`text_asr` as empty string.

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
            gold_text = r.get("text", "")
            if p is None:
                n_missing += 1
                asr_text = ""
                asr_status = "missing"
                wer_n = cer_n = None
                overlap = "unknown"
                overlap_frac = None
            else:
                asr_text = (p.get("transcript") or "").strip()
                if not asr_text:
                    n_empty += 1
                asr_status = p.get("status") or ("empty" if not asr_text else "ok")
                wer_n = p.get("wer_norm")
                cer_n = p.get("cer_norm")
                overlap = p.get("overlap")
                overlap_frac = p.get("overlap_frac")
            new_row = dict(r)
            new_row["text"] = asr_text          # downstream default
            new_row["text_asr"] = asr_text      # explicit alias
            new_row["text_gold"] = gold_text    # preserved gold
            new_row["asr_status"] = asr_status
            new_row["asr_wer_norm"] = wer_n
            new_row["asr_cer_norm"] = cer_n
            new_row["overlap"] = overlap
            new_row["overlap_frac"] = overlap_frac
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
                gold_text = row.get("text", "")
                if p is None:
                    asr = ""
                    n_sub_missing += 1
                else:
                    asr = (p.get("transcript") or "").strip()
                    if not asr:
                        n_sub_empty += 1
                new_row = dict(row)
                new_row["text"] = asr          # downstream default
                new_row["text_asr"] = asr      # explicit alias
                new_row["text_gold"] = gold_text
                new_row["asr_wer_norm"] = p.get("wer_norm") if p else None
                new_row["overlap"] = p.get("overlap") if p else "unknown"
                f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
                n_sub_written += 1
        print(f"\nAlso wrote subset-schema ASR variant to {asr_sub}")
        print(f"  rows: {n_sub_written} · missing: {n_sub_missing} · empty ASR: {n_sub_empty}")


if __name__ == "__main__":
    main()
