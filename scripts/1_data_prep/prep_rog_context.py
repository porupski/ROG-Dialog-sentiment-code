#!/usr/bin/env python3
"""Pre-render context-windowed ROG training data for the finetune matrix.

For each ROG row, build a window of W words per side (accumulator, identical to the
OR context runner at scripts/2_probe_text/03_run_openrouter_text_context.py), collapse
speakers to SPK1/SPK2/..., and REPLACE the row's `text` field with the rendered
multi-turn window (so downstream readers see one big string per instance). All other
fields are preserved so the existing `_reader_rog_dia_unfiltered` still works.

Window string example (W=20, target marker "eol"):

    [SPK1] prev utterance one
    [SPK2] prev utterance two
    [SPK1] target text  [TARGET]
    [SPK2] next utterance one

Non-target rows are kept out of the output's `text` field; the label is the TARGET
row's label (unchanged from the corpus).

Output path: data/rog_dialog_v2_instance_ctxw{W}.jsonl

Usage:
  python3 scripts/1_data_prep/prep_rog_context.py --W 10
  python3 scripts/1_data_prep/prep_rog_context.py --W 20
  python3 scripts/1_data_prep/prep_rog_context.py --W 30
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CORPUS = ROOT / "data" / "rog_dialog_v2_instance.jsonl"


def _load_ctx_module():
    """Import 03_run_openrouter_text_context as a module despite its leading digit
    (regular `import` won't work with a filename starting with a number).
    Shares the SINGLE source of truth for build_window / render_window / CorpusIndex."""
    src = ROOT / "scripts" / "2_probe_text" / "03_run_openrouter_text_context.py"
    spec = importlib.util.spec_from_file_location("ctx_runner", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--W", type=int, required=True,
                    help="per-side word threshold (10, 20, 30, ...). "
                         "W=0 reproduces the OR runner's legacy ±1-utterance window.")
    ap.add_argument("--target-marker", default="eol", choices=["eol", "after_spk", "own_line"],
                    help="Where to place the [TARGET] token in the rendered window.")
    args = ap.parse_args()

    ctx = _load_ctx_module()

    if not CORPUS.exists():
        raise SystemExit(f"missing {CORPUS}")
    rows_in = []
    with open(CORPUS) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("labels") or {}).get("sentiment") is None:
                continue
            rows_in.append(r)
    print(f"Loaded {len(rows_in)} ROG rows with sentiment")

    corpus = ctx.CorpusIndex(rows_in)

    out_path = ROOT / "data" / f"rog_dialog_v2_instance_ctxw{args.W}.jsonl"
    n_hit_prev = n_hit_next = 0
    n_no_prev = n_no_next = 0
    out_rows = []
    for r in rows_in:
        # The OR runner's build_window takes a row with top-level `text` and `instance_id`;
        # ROG corpus rows carry those at top level too, so we pass the row directly.
        try:
            win = ctx.build_window(r, corpus, args.W)
        except SystemExit as e:
            # Should not happen — all corpus rows are in corpus.by_id by construction.
            raise
        rendered = ctx.render_window(win, args.target_marker)
        if win["prev_hit_threshold"]:
            n_hit_prev += 1
        if win["next_hit_threshold"]:
            n_hit_next += 1
        if win["boundary_flag"] in ("no_prev", "no_both"):
            n_no_prev += 1
        if win["boundary_flag"] in ("no_next", "no_both"):
            n_no_next += 1
        out = dict(r)  # shallow copy
        out["text_iso"] = r["text"]            # preserve bare utterance for diagnostics
        out["text"] = rendered                 # REPLACE text with the rendered window
        out["context_words_target"] = args.W
        out["prev_word_count"] = win["prev_word_count"]
        out["next_word_count"] = win["next_word_count"]
        out["prev_hit_threshold"] = win["prev_hit_threshold"]
        out["next_hit_threshold"] = win["next_hit_threshold"]
        out["n_prev_items"] = win["n_prev_items"]
        out["n_next_items"] = win["n_next_items"]
        out["window_distinct_speakers_count"] = win["window_distinct_speakers_count"]
        out["boundary_flag"] = win["boundary_flag"]
        out_rows.append(out)

    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out_rows)
    print(f"wrote {out_path.relative_to(ROOT)}  n={n}")
    print(f"  prev-side hit W: {n_hit_prev}/{n} ({100*n_hit_prev/n:.1f}%)")
    print(f"  next-side hit W: {n_hit_next}/{n} ({100*n_hit_next/n:.1f}%)")
    print(f"  no_prev boundary: {n_no_prev} · no_next boundary: {n_no_next}")

    # Quick sanity sample
    print("\nFirst rendered window:")
    print("---")
    print(out_rows[0]["text"])
    print("---")


if __name__ == "__main__":
    main()
