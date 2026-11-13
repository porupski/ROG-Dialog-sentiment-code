#!/usr/bin/env python3
"""run_06_matrix — launch the full backbone × mixture fine-tune matrix.

- 3 backbones × 7 mixtures = 21 training runs; each evaluated on 3 test sets = 63 eval cells.
- Each cell is run as a SUBPROCESS of 06_finetune_text.py so one OOM doesn't kill the
  weekend. Logs stream through this parent and through the subprocess (printed to stdout).
- GPU guard: CUDA_VISIBLE_DEVICES is set BEFORE this script imports torch (torch is only
  imported indirectly, via the subprocess). We NEVER touch another GPU.

RECOMMENDED LAUNCH (remote, reserved GPU 2):
  tmux new -s six
  python3 scripts/run_06_matrix.py --mode full --use_gpu 2>&1 \\
      | tee -a runs/finetune_$(date +%F_%H%M).log

Useful subsets:
  --backbones FacebookAI/xlm-roberta-large classla/xlm-r-parla
  --mixtures  S R SPR
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# NOTE: do NOT import torch here. GPU guard below must precede any torch import in
# child processes too — we propagate CUDA_VISIBLE_DEVICES via env.
from utils_text_finetune import BACKBONES, MIXTURES, backbone_short


def load_reserved_gpu(config_json: Path) -> str:
    if config_json.exists():
        j = json.loads(config_json.read_text())
        return str(j.get("shared", {}).get("reserved_gpu", "2"))
    return "2"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", "-m", default="full", choices=["test", "demo", "full", "custom"])
    p.add_argument("--config", "-c", default=str(HERE / "6_config.json"))
    p.add_argument("--backbones", nargs="*", default=None, help="subset of BACKBONES")
    p.add_argument("--mixtures", nargs="*", default=None, choices=list(MIXTURES.keys()),
                   help="subset of MIXTURES (default: all)")
    p.add_argument("--use_gpu", action="store_true",
                   help="arm reserved GPU from 6_config.json (shared.reserved_gpu)")
    p.add_argument("--out", "-o", default=None, help="override runs output dir")
    p.add_argument("--dry-run", action="store_true", help="print the planned command list only")
    args = p.parse_args()

    config_json = Path(args.config)

    env = os.environ.copy()
    if args.use_gpu:
        gpu = load_reserved_gpu(config_json)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"GPU guard: CUDA_VISIBLE_DEVICES={gpu} (reserved GPU from 6_config.json)")
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""   # force CPU in child processes
        print("No --use_gpu: child processes forced to CPU (CUDA_VISIBLE_DEVICES='').")

    env["TOKENIZERS_PARALLELISM"] = "false"
    env["MPLBACKEND"] = "Agg"

    backbones = args.backbones or BACKBONES
    mixtures = args.mixtures or list(MIXTURES.keys())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (ROOT / "runs" / f"finetune_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Matrix output root: {out_dir.relative_to(ROOT)}")

    engine = HERE / "06_finetune_text.py"

    cells = [(b, x) for b in backbones for x in mixtures]
    print(f"Running {len(cells)} cells = {len(backbones)} backbones × {len(mixtures)} mixtures\n")

    matrix_summary = {"ts_start": stamp, "mode": args.mode,
                      "cells": [], "config": args.config, "env_CUDA": env["CUDA_VISIBLE_DEVICES"]}
    t_matrix = time.time()
    for i, (backbone, mix) in enumerate(cells, 1):
        tag = f"{backbone_short(backbone)}__{mix}"
        cmd = [sys.executable, str(engine),
               "--backbone", backbone,
               "--mixture", mix,
               "--mode", args.mode,
               "--config", str(config_json),
               "--out", str(out_dir)]
        if args.use_gpu:
            cmd.append("--use_gpu")

        print(f"\n[{i:>2}/{len(cells)}] {tag}")
        print(f"     cmd: {' '.join(cmd)}")
        if args.dry_run:
            matrix_summary["cells"].append({"cell": tag, "status": "dry_run"})
            continue

        t0 = time.time()
        proc = subprocess.run(cmd, env=env)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"fail(code={proc.returncode})"
        print(f"     -> {status} in {elapsed:.0f}s")
        matrix_summary["cells"].append({
            "cell": tag, "backbone": backbone, "mixture": mix,
            "status": status, "elapsed_sec": round(elapsed, 1),
        })
        # Incremental save so a crash in cell 18/21 doesn't lose the first 17 rows.
        (out_dir / "matrix_summary.json").write_text(
            json.dumps(matrix_summary, indent=2, ensure_ascii=False)
        )

    matrix_summary["total_elapsed_sec"] = round(time.time() - t_matrix, 1)
    (out_dir / "matrix_summary.json").write_text(
        json.dumps(matrix_summary, indent=2, ensure_ascii=False)
    )
    print(f"\nMatrix done. Summary: {out_dir / 'matrix_summary.json'}")


if __name__ == "__main__":
    main()
