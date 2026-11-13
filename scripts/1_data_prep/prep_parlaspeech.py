#!/usr/bin/env python3
"""Slim down a ParlaSpeech utterance JSONL by dropping fields we never read.

ParlaSpeech ships every row with words_align / graph_align / linguistic_annots payloads
that can account for 2/3 of the file size and are useless for text-sentiment finetuning.
This script streams the source JSONL, drops those fields, and writes a slim JSONL to
data/ that the finetune_text engine (and anything else here) can read directly.

Default --src points at the ParlaSpeech-SI v4 raw JSONL under SSP's cache; override to
retarget to any ParlaSpeech language.

Usage:
  python3 scripts/1_data_prep/prep_parlaspeech.py
  python3 scripts/1_data_prep/prep_parlaspeech.py \\
      --src  /path/to/ParlaSpeech-XX.vN.0.jsonl \\
      --out  data/parlaspeech_vN_xx_utterance_instance.jsonl
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DEFAULT_SRC = Path(
    "/cache/ivanp/projects/slavic-speech-pipeline/data/unpacked"
    "/ParlaSpeech-SI-v4/ParlaSpeech-SI.v4.0/ParlaSpeech-SI.v4.0.jsonl"
)
DEFAULT_OUT = ROOT / "data" / "parlaspeech_v4_si_utterance_instance.jsonl"

# Known-heavy annotation fields that finetune_text never reads. Add here as more show up.
DROP_FIELDS = ("words_align", "graph_align", "linguistic_annots")


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help=f"source JSONL (default: {DEFAULT_SRC})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output slim JSONL (default: {DEFAULT_OUT.relative_to(ROOT)})")
    p.add_argument("--drop", nargs="*", default=list(DROP_FIELDS),
                   help=f"top-level fields to drop (default: {' '.join(DROP_FIELDS)})")
    args = p.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source JSONL not found: {args.src}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    drop = set(args.drop)

    src_bytes = args.src.stat().st_size
    n_in = 0
    n_out = 0
    n_fields_stripped = 0

    print(f"src: {args.src}  ({human_bytes(src_bytes)})")
    print(f"out: {args.out}")
    print(f"dropping fields: {sorted(drop)}")

    with open(args.src) as fin, open(args.out, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            row = json.loads(line)
            for k in drop:
                if k in row:
                    del row[k]
                    n_fields_stripped += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    out_bytes = args.out.stat().st_size
    ratio = (1 - out_bytes / src_bytes) * 100 if src_bytes else 0.0
    print(f"rows in={n_in}  out={n_out}  field-strips={n_fields_stripped}")
    print(f"size in={human_bytes(src_bytes)}  out={human_bytes(out_bytes)}  "
          f"saved {ratio:.1f} %")


if __name__ == "__main__":
    main()
