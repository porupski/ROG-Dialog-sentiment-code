#!/usr/bin/env python3
"""Slim a ParlaSpeech utterance JSONL and assign a speaker-disjoint train/dev/test split.

ParlaSpeech v4 rows carry alignment/parse payloads that account for the majority of the
file size and are useless for text-sentiment finetuning:

  words                  — per-word timing + char spans (time_s/time_e/char_s/char_e each)
  linguistic_annotation  — full UD parse per token (lemma/upos/xpos/feats/head/deprel/misc)
  chars_align            — char-level alignment (null in SI, large when populated)
  primary_stress         — primary-stress alignment (null in SI, large elsewhere)
  words_align            — alignment array (null in SI, large elsewhere)

What we keep: id, audio, audio_length, text, text_start/end, audio_start/end, speaker_info,
filled_pauses, silent_pauses (prosody-adjacent, small), sentiment (the label payload), cap.

ParlaSpeech v4 ships WITHOUT a split field, so this script also assigns a speaker-disjoint
80/10/10 train/dev/test split, grouped by speaker_info.Speaker_ID. Allocation walks speakers
biggest-first (ties broken by seeded shuffle) and drops each one into whichever split has
the largest row-count deficit vs its target — size-aware greedy packing. No sentiment-class
stratification; the resulting per-split class distribution is printed so skew is visible.

Default --src points at the ParlaSpeech-SI v4 raw JSONL under SSP's cache; override to
retarget to any ParlaSpeech language.

Usage:
  python3 scripts/1_data_prep/prep_parlaspeech.py
  python3 scripts/1_data_prep/prep_parlaspeech.py \\
      --src  /path/to/ParlaSpeech-XX.vN.0.jsonl \\
      --out  data/parlaspeech_vN_xx_utterance_instance.jsonl \\
      --train-frac 0.8 --dev-frac 0.1 --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

DEFAULT_SRC = Path(
    "/cache/ivanp/projects/slavic-speech-pipeline/data/unpacked"
    "/ParlaSpeech-SI-v4/ParlaSpeech-SI.v4.0/ParlaSpeech-SI.v4.0.jsonl"
)
DEFAULT_OUT = ROOT / "data" / "parlaspeech_v4_si_utterance_instance.jsonl"

DROP_FIELDS = (
    "words",
    "linguistic_annotation",
    "chars_align",
    "primary_stress",
    "words_align",
)

UNKNOWN_SPEAKER = "_unknown_speaker"


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024


def speaker_id_for(row: dict) -> str:
    """Prefer speaker_info.Speaker_ID; fall back to the utterance id's leading slug so
    rows without speaker metadata still group together rather than each getting a unique id."""
    sid = (row.get("speaker_info") or {}).get("Speaker_ID")
    if sid:
        return str(sid)
    # Row id looks like 'ParlaMint-SI_2011-03-23-SDZ5-Redna-26.seg31.2' — the session prefix
    # before the last two dot-segments is a usable fallback.
    rid = row.get("id")
    if isinstance(rid, str):
        return rid.rsplit(".", 2)[0]
    return UNKNOWN_SPEAKER


def assign_splits(rows: list[dict], train_frac: float, dev_frac: float, seed: int) -> dict:
    """Mutates rows in place by adding row['split']. Returns an info dict."""
    test_frac = 1.0 - train_frac - dev_frac
    if test_frac < 0:
        raise SystemExit(f"train_frac + dev_frac = {train_frac + dev_frac} > 1.0")

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_speaker[speaker_id_for(r)].append(r)

    total = len(rows)
    targets = {
        "train": train_frac * total,
        "dev":   dev_frac   * total,
        "test":  test_frac  * total,
    }
    counts = {"train": 0, "dev": 0, "test": 0}

    rng = random.Random(seed)
    items = list(by_speaker.items())
    rng.shuffle(items)                       # seeded tie-break within same-size groups
    items.sort(key=lambda kv: -len(kv[1]))   # biggest-first; sort is stable so shuffle holds

    for sid, srows in items:
        # Pick the split with the biggest row-count deficit vs its target.
        best = max(counts, key=lambda s: targets[s] - counts[s])
        for r in srows:
            r["split"] = best
        counts[best] += len(srows)

    return {
        "n_speakers": len(by_speaker),
        "counts": counts,
        "targets": {k: int(round(v)) for k, v in targets.items()},
    }


def label_bucket(row: dict) -> str:
    """Return the 3-class ParlaSent collapse string for reporting, or '∅' if absent."""
    s = (row.get("sentiment") or {}).get("ParlaSent_3")
    if s is None:
        return "∅"
    return str(s)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help=f"source JSONL (default: {DEFAULT_SRC})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output slim JSONL (default: {DEFAULT_OUT.relative_to(ROOT)})")
    p.add_argument("--drop", nargs="*", default=list(DROP_FIELDS),
                   help=f"top-level fields to drop (default: {' '.join(DROP_FIELDS)})")
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--dev-frac",   type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Source JSONL not found: {args.src}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    drop = set(args.drop)

    src_bytes = args.src.stat().st_size
    print(f"src: {args.src}  ({human_bytes(src_bytes)})")
    print(f"out: {args.out}")
    print(f"dropping fields: {sorted(drop)}")

    # Pass 1: stream + slim into memory. The slimmed file for SI fits comfortably; if a
    # much larger corpus ever outgrows RAM, rework into a two-pass variant that assigns
    # splits from a speaker→split map built in pass 1 and written in pass 2.
    rows: list[dict] = []
    n_fields_stripped = 0
    with open(args.src) as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            for k in drop:
                if k in row:
                    del row[k]
                    n_fields_stripped += 1
            rows.append(row)

    n_in = len(rows)
    print(f"read: {n_in} rows, stripped {n_fields_stripped} field occurrences")

    split_info = assign_splits(rows, args.train_frac, args.dev_frac, args.seed)
    counts = split_info["counts"]
    print(f"speakers: {split_info['n_speakers']}")
    print(f"split targets: {split_info['targets']}")
    print(f"split counts : {counts}")

    # Per-split × per-ParlaSent_3 distribution for a sanity eyeball.
    dist: dict[str, Counter] = {"train": Counter(), "dev": Counter(), "test": Counter()}
    for r in rows:
        dist[r["split"]][label_bucket(r)] += 1
    print("per-split ParlaSent_3 distribution:")
    for split in ("train", "dev", "test"):
        total = counts[split] or 1
        parts = [f"{lbl}={n} ({100 * n / total:.1f}%)" for lbl, n in sorted(dist[split].items())]
        print(f"  {split:5s}: {' '.join(parts)}")

    with open(args.out, "w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_bytes = args.out.stat().st_size
    ratio = (1 - out_bytes / src_bytes) * 100 if src_bytes else 0.0
    print(f"wrote {n_in} rows -> {args.out}")
    print(f"size in={human_bytes(src_bytes)}  out={human_bytes(out_bytes)}  "
          f"saved {ratio:.1f} %")


if __name__ == "__main__":
    main()
