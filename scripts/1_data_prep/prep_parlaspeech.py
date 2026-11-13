#!/usr/bin/env python3
"""Slim a ParlaSpeech utterance JSONL, cap + balance by 6-class label, split by speaker.

ParlaSpeech v4 rows carry alignment/parse payloads that account for the majority of the
file size and are useless for text-sentiment finetuning:

  words                  — per-word timing + char spans (time_s/time_e/char_s/char_e each)
  linguistic_annotation  — full UD parse per token (lemma/upos/xpos/feats/head/deprel/misc)
  chars_align            — char-level alignment (null in SI, large when populated)
  primary_stress         — primary-stress alignment (null in SI, large elsewhere)
  words_align            — alignment array (null in SI, large elsewhere)

What we keep: id, audio, audio_length, text, text_start/end, audio_start/end, speaker_info,
filled_pauses, silent_pauses (prosody-adjacent, small), sentiment (the label payload), cap.

Processing order:
  1. Stream rows in (keep `words` so we can count length).
  2. Filter 10 ≤ len(words) ≤ 40  — matches ROG-Dialog's utterance-length distribution
     and avoids both "ok" one-liners and multi-minute monologues that drift off-topic.
  3. Balance-sample up to N per 6-class ParlaSent_6 label (default 1000 → 6k total cap).
     If a class has fewer than N instances we take all of them. Seeded.
  4. Drop the heavy fields listed above.
  5. Assign a speaker-disjoint 80/10/10 split (speaker_info.Speaker_ID grouped, size-aware
     greedy packing).
  6. Write.

ParlaSpeech labels are inferred by XLM-R-ParlaSent itself, so fine-tuning *XLM-R-ParlaSent*
on them is self-labeled training. Note this when reading matrix results.

Default --src points at the ParlaSpeech-SI v4 raw JSONL under SSP's cache; override to
retarget to any ParlaSpeech language.

Usage:
  python3 scripts/1_data_prep/prep_parlaspeech.py
  python3 scripts/1_data_prep/prep_parlaspeech.py \\
      --src  /path/to/ParlaSpeech-XX.vN.0.jsonl \\
      --out  data/parlaspeech_vN_xx_utterance_instance.jsonl \\
      --cap-per-class 1000 --min-words 10 --max-words 40 --seed 42
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
UNKNOWN_LABEL = "_unknown_label"


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024


def word_count(row: dict) -> int:
    """Count words from the `words` array when available (authoritative word segmentation
    from ParlaSpeech's forced alignment); fall back to whitespace split on text."""
    w = row.get("words")
    if isinstance(w, list):
        return len(w)
    t = row.get("text") or ""
    return len(t.split())


def label_6(row: dict) -> str:
    s = (row.get("sentiment") or {}).get("ParlaSent_6")
    return str(s) if s is not None else UNKNOWN_LABEL


def label_3(row: dict) -> str:
    s = (row.get("sentiment") or {}).get("ParlaSent_3")
    return str(s) if s is not None else "∅"


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


def balance_sample(rows: list[dict], cap_per_class: int, seed: int) -> list[dict]:
    """Group by ParlaSent_6 label, shuffle each group deterministically, take up to
    cap_per_class from each. Rows with no label are dropped (the model needs one)."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        lbl = label_6(r)
        if lbl == UNKNOWN_LABEL:
            continue
        by_label[lbl].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for lbl, group in by_label.items():
        rng.shuffle(group)
        out.extend(group[:cap_per_class])
    rng.shuffle(out)                 # mix classes so split allocation doesn't see blocks
    return out


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
        best = max(counts, key=lambda s: targets[s] - counts[s])
        for r in srows:
            r["split"] = best
        counts[best] += len(srows)

    return {
        "n_speakers": len(by_speaker),
        "counts": counts,
        "targets": {k: int(round(v)) for k, v in targets.items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC,
                   help=f"source JSONL (default: {DEFAULT_SRC})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"output slim JSONL (default: {DEFAULT_OUT.relative_to(ROOT)})")
    p.add_argument("--drop", nargs="*", default=list(DROP_FIELDS),
                   help=f"top-level fields to drop (default: {' '.join(DROP_FIELDS)})")
    p.add_argument("--min-words", type=int, default=10,
                   help="drop utterances with fewer than N words (default 10)")
    p.add_argument("--max-words", type=int, default=40,
                   help="drop utterances with more than N words (default 40)")
    p.add_argument("--cap-per-class", type=int, default=1000,
                   help="max rows to keep per ParlaSent_6 label (default 1000 → 6k total)")
    p.add_argument("--no-cap", action="store_true",
                   help="disable class cap entirely (writes everything that passes the word filter)")
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
    print(f"word-count filter: {args.min_words} .. {args.max_words}")
    print(f"cap per ParlaSent_6 class: {'∞ (no-cap)' if args.no_cap else args.cap_per_class}")

    # Pass 1: stream in. Keep the whole row (words included) so we can count length;
    # drop heavy fields later, once we've decided which rows survive.
    rows: list[dict] = []
    n_in = 0
    with open(args.src) as fin:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            rows.append(json.loads(line))
    print(f"read: {n_in} rows")

    # Word-length filter.
    kept = [r for r in rows if args.min_words <= word_count(r) <= args.max_words]
    print(f"word-filter kept: {len(kept)} of {n_in} ({100 * len(kept) / max(1, n_in):.1f}%)")

    # Per-6cls counts BEFORE cap (so we know how much headroom each class has).
    pre_cap_counts = Counter(label_6(r) for r in kept)
    print("pre-cap counts (ParlaSent_6):")
    for lbl, c in sorted(pre_cap_counts.items()):
        print(f"  {lbl:20s}: {c}")

    # Class cap + balance.
    if args.no_cap:
        capped = [r for r in kept if label_6(r) != UNKNOWN_LABEL]
    else:
        capped = balance_sample(kept, args.cap_per_class, args.seed)
    post_cap_counts = Counter(label_6(r) for r in capped)
    print(f"post-cap total: {len(capped)}")
    print("post-cap counts (ParlaSent_6):")
    for lbl, c in sorted(post_cap_counts.items()):
        print(f"  {lbl:20s}: {c}")

    # Drop heavy fields on the surviving rows.
    n_fields_stripped = 0
    for r in capped:
        for k in drop:
            if k in r:
                del r[k]
                n_fields_stripped += 1
    print(f"stripped {n_fields_stripped} heavy-field occurrences from {len(capped)} rows")

    # Speaker-disjoint split.
    split_info = assign_splits(capped, args.train_frac, args.dev_frac, args.seed)
    counts = split_info["counts"]
    print(f"speakers: {split_info['n_speakers']}")
    print(f"split targets: {split_info['targets']}")
    print(f"split counts : {counts}")

    # Per-split × per-ParlaSent_3 distribution for a sanity eyeball.
    dist: dict[str, Counter] = {"train": Counter(), "dev": Counter(), "test": Counter()}
    for r in capped:
        dist[r["split"]][label_3(r)] += 1
    print("per-split ParlaSent_3 distribution:")
    for split in ("train", "dev", "test"):
        total = counts[split] or 1
        parts = [f"{lbl}={n} ({100 * n / total:.1f}%)" for lbl, n in sorted(dist[split].items())]
        print(f"  {split:5s}: {' '.join(parts)}")

    with open(args.out, "w") as fout:
        for row in capped:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_bytes = args.out.stat().st_size
    ratio = (1 - out_bytes / src_bytes) * 100 if src_bytes else 0.0
    print(f"wrote {len(capped)} rows -> {args.out}")
    print(f"size in={human_bytes(src_bytes)}  out={human_bytes(out_bytes)}  "
          f"saved {ratio:.1f} %")


if __name__ == "__main__":
    main()
