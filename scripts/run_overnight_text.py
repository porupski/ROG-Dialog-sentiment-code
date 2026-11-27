#!/usr/bin/env python3
"""Overnight text-sentiment task list. One-shot runner — no SSP, no hand-edits.

Usage (from repo root, on remote):
    python3 scripts/run_overnight_text.py

What it does, sequentially:

  BLOCK B (FIRST — fail fast on model download)
    Slovenian backbone sanity: R_full iso, 4 ep, seed 42.
    EMBEDDIA/sloberta  (if it loads)
    cjvt/SloBERTa-2.0  (fallback if EMBEDDIA errors)
    ~15 min each. If both fail, we log and move on to Block A.

  BLOCK A (SECOND — the deliverable push)
    3-seed × 3-epoch sweep on champion (R_full × xlm-r-parlasent).
    seeds {42, 1337, 2024} × epochs {8, 15, 20} = 9 cells.
    each ~25–55 min · total ~5 h.
    Patches config.json's shared.seed + mixture_overrides.R_full.num_epochs
    per cell, resets config at end of block.

Why Block B first: HuggingFace downloads can stall/fail. If the SloBERTa slug
is wrong or the model is gated, the script errors out within a minute, you wake
up and fix it. With Block A first, a Block B failure at 05:00 wastes the slot.

Outputs: one dir per cell under `runs/finetune_sloberta_*` and
`runs/finetune_deliv_*`. Combined log at `runs/overnight_text_<stamp>.log`.

All compute stays in this repo (scripts/4_finetune_text/run_matrix.py).
GPU is the project's reserved GPU 3.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "scripts" / "4_finetune_text" / "config.json"
RUNNER = ROOT / "scripts" / "4_finetune_text" / "run_matrix.py"
RUNS_DIR = ROOT / "runs"
STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = RUNS_DIR / f"overnight_text_{STAMP}.log"

GPU = "3"
BACKBONE_CHAMP = "classla/xlm-r-parlasent"
MIX_CHAMP = "R_full"
SEEDS = [42, 1337, 2024]
EPOCHS = [8, 15, 20]
# SloBERTa deferred: `EMBEDDIA/sloberta` triggers a Camembert tokenizer bug
# ('too many values to unpack') in the current transformers version; the slow
# path hits it too so use_fast=True doesn't help. `cjvt/SloBERTa-2.0` does
# not exist as an HF repo (401). Resolve with a monkey-patch or a pinned
# transformers version before re-enabling. For now, leave empty so Block B
# is a no-op.
SLOBERTA_CANDIDATES: list[str] = []


class Tee:
    """stdout → both terminal and log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def tee_to(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("a")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_out, f)
    sys.stderr = Tee(orig_err, f)
    try:
        yield
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err
        f.close()


def patch_config(seed: int | None = None,
                 r_full_epochs: int | None = None) -> dict:
    """Edit config.json's shared.seed and mixture_overrides.R_full.num_epochs
    in-place. Returns the previous values so we can restore later."""
    c = json.loads(CONFIG_PATH.read_text())
    prev = {
        "seed": c["shared"].get("seed"),
        "R_full_overrides": dict(c["mixture_overrides"].get("R_full", {})),
    }
    if seed is not None:
        c["shared"]["seed"] = seed
    if r_full_epochs is not None:
        c["mixture_overrides"].setdefault("R_full", {})["num_epochs"] = r_full_epochs
    CONFIG_PATH.write_text(json.dumps(c, indent=2) + "\n")
    return prev


def restore_config(prev: dict) -> None:
    c = json.loads(CONFIG_PATH.read_text())
    c["shared"]["seed"] = prev["seed"]
    if prev["R_full_overrides"]:
        c["mixture_overrides"]["R_full"] = prev["R_full_overrides"]
    else:
        c["mixture_overrides"].pop("R_full", None)
    CONFIG_PATH.write_text(json.dumps(c, indent=2) + "\n")


def run_cell(backbones: list[str], mixtures: list[str], out_tag: str) -> tuple[int, Path]:
    """Run run_matrix.py for a set of (backbone × mixture) cells. Returns
    (exit code, output directory). The caller checks matrix_summary.json to
    see whether individual cells actually succeeded (run_matrix.py exits 0
    even when child cells crash)."""
    out_dir = RUNS_DIR / f"{out_tag}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        sys.executable, str(RUNNER),
        "--backbones", *backbones,
        "--mixtures", *mixtures,
        "--gpu", GPU,
        "--mode", "full",
        "--out", str(out_dir),
    ]
    print(f"\n$ {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    print(f"-> exit {rc}, out: {out_dir}")
    return rc, out_dir


def all_cells_ok(out_dir: Path) -> bool:
    """True iff every cell in the batch's matrix_summary.json has status 'ok'."""
    summary_path = out_dir / "matrix_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return False
    cells = summary.get("cells", [])
    if not cells:
        return False
    return all(c.get("status") == "ok" for c in cells)


def block_a_deliverable() -> None:
    print("\n" + "=" * 72)
    print("BLOCK A — 3-seed × 3-epoch sweep on R_full × xlm-r-parlasent")
    print("=" * 72)
    original = None
    try:
        original = patch_config()  # snapshot
        for seed in SEEDS:
            for ep in EPOCHS:
                print(f"\n--- seed={seed} epochs={ep} ---")
                patch_config(seed=seed, r_full_epochs=ep)
                rc, out = run_cell([BACKBONE_CHAMP], [MIX_CHAMP],
                                   out_tag=f"finetune_deliv_s{seed}_e{ep}")
                if rc != 0 or not all_cells_ok(out):
                    print(f"  WARN: cell failed (rc={rc}, cells_ok={all_cells_ok(out)}) — continuing.")
    finally:
        if original is not None:
            restore_config(original)
            print("\n[config restored to pre-Block-A values]")


def block_b_sloberta() -> None:
    print("\n" + "=" * 72)
    print("BLOCK B — Slovenian backbone sanity on R_full (4 ep, seed 42)")
    print("=" * 72)
    if not SLOBERTA_CANDIDATES:
        print("  SLOBERTA_CANDIDATES is empty — block B skipped by configuration.")
        return
    for slug in SLOBERTA_CANDIDATES:
        print(f"\n--- backbone={slug} ---")
        rc, out = run_cell([slug], [MIX_CHAMP],
                           out_tag=f"finetune_sloberta_{slug.split('/')[-1]}")
        if rc == 0 and all_cells_ok(out):
            print(f"  OK — {slug} trained + evaluated successfully. Stopping Block B here.")
            return
        print(f"  {slug} FAILED (rc={rc}, cells_ok={all_cells_ok(out)}); trying next candidate.")
    print("  Block B exhausted all candidates — no SloBERTa cell landed.")


def main() -> int:
    if not CONFIG_PATH.exists() or not RUNNER.exists():
        sys.exit("error: this script expects to live under <repo>/scripts/ and "
                 "see scripts/4_finetune_text/{config.json,run_matrix.py}.")
    RUNS_DIR.mkdir(exist_ok=True)
    with tee_to(LOG_PATH):
        t0 = dt.datetime.now()
        print(f"Overnight text task list — start {t0.isoformat()}  (log: {LOG_PATH})")
        block_b_sloberta()
        block_a_deliverable()
        t1 = dt.datetime.now()
        print(f"\nAll done. Total wall: {(t1 - t0)}")
        print(f"Logs: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
