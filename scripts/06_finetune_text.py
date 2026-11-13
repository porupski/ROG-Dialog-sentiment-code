#!/usr/bin/env python3
"""06_finetune_text — single (backbone × mixture) regression fine-tune for Slovenian sentiment.

Head: scalar regression (num_labels=1, problem_type='regression'), MSE loss, z-score label
normalization fit on train. Test eval rounds + clips to {0..5} for 6cls metrics and
collapses to {neg,neu,pos} for 3cls (user's requested round-up-to-category mapping).

This is the single-run engine. For the full backbone × mixture matrix use
run_06_matrix.py which calls this file via subprocess, one process per cell, so one
blow-up doesn't kill the whole weekend.

RECOMMENDED LAUNCH (remote, reserved GPU 2):
  tmux new -s six
  python3 scripts/run_06_matrix.py --mode full --use_gpu 2>&1 \\
      | tee -a runs/finetune_$(date +%F_%H%M).log

The CUDA_VISIBLE_DEVICES=2 env var is set BEFORE torch import inside run_06_matrix.py.
Never call this engine directly on the remote without --use_gpu or you'll run on CPU
and finish the weekend run sometime next month.

Usage (direct, single cell):
  python3 scripts/06_finetune_text.py \\
      --backbone FacebookAI/xlm-roberta-large --mixture R --mode demo
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from utils_text_finetune import Config, BACKBONES, MIXTURES, run_training, backbone_short


def apply_mode(cfg: Config, mode: str, config_json: Path | None):
    """Override cfg with values from 6_config.json modes[mode] block."""
    if config_json and config_json.exists():
        j = json.loads(config_json.read_text())
        for k, v in j.get("shared", {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        tb = j.get("regression", {})
        for k, v in tb.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        mb = j.get("modes", {}).get(mode, {})
        for k, v in mb.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backbone", "-b", required=True, choices=BACKBONES + [b for b in BACKBONES])
    p.add_argument("--mixture", "-x", required=True, choices=list(MIXTURES.keys()))
    p.add_argument("--mode", "-m", default="full", choices=["test", "demo", "full", "custom"])
    p.add_argument("--config", "-c", default=str(HERE / "6_config.json"))
    p.add_argument("--out", "-o", default=None, help="override runs output dir")
    p.add_argument("--use_gpu", action="store_true",
                   help="informational — GPU is armed by run_06_matrix.py via CUDA_VISIBLE_DEVICES")
    args = p.parse_args()

    cfg = Config(backbone=args.backbone, mixture=args.mixture, use_gpu=args.use_gpu)
    cfg = apply_mode(cfg, args.mode, Path(args.config))

    out_dir = Path(args.out) if args.out else (ROOT / cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_training(cfg, ROOT, out_dir)
    print(f"\nWrote {out_dir / summary['run_tag']}/summary.json")


if __name__ == "__main__":
    main()
