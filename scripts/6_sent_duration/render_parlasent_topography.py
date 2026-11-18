"""Headless render of the Phase-2 domain-gap topography PNGs.

Two plots:
  docs/img/topo_parlasent_zs_on_rog_full.png — 3-D (duration × WPS × F1_3) over
    stock classla/xlm-r-parlasent predictions on ROG v2 test (n=2233). The
    "parliament-trained head on dialogue test" picture. Drawn from the batch
    emitted by 04_run_parlasent.py --eval-set data/subset_rog_test_full.jsonl
    --eval-tag FULLROG --protocols reg --schemes 3cls.

  docs/img/topo_parlasent_zs_on_parlasent.png — 1-D (word-count × F1_3) over
    stock parlasent predictions on ParlaSent-SL test (home-domain reference).
    No duration on ParlaSent → the 3-D plot has nothing to bin on, so this is
    a bar envelope across word-count quantile bins. Drawn from the batch
    emitted by 04_run_parlasent.py --eval-set data/subset_parlasent_sl_test.jsonl
    --eval-tag PARLASENT --protocols reg --schemes 3cls.

Usage: pass the two batch dirs via CLI. Falls back to the latest
`batch_LOCAL_FULLROG_*` / `batch_LOCAL_PARLASENT_*` under runs/ if no args.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import utils_rog_explorer as u

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)


def latest_batch(prefix):
    cands = [p for p in (ROOT / "runs").iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not cands:
        return None
    cands.sort(key=lambda p: p.name)
    return cands[-1]


def find_parlasent_reg_3cls_run(batch_dir):
    """The reg protocol run_id is 'LOCAL-text-parlasent-reg-0shot-iso-3cls'."""
    if batch_dir is None:
        return None
    cand = batch_dir / "LOCAL-text-parlasent-reg-0shot-iso-3cls"
    return cand if cand.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rog-batch", default=None, help="Batch dir with parlasent × ROG-full-test run")
    ap.add_argument("--parlasent-batch", default=None, help="Batch dir with parlasent × ParlaSent-SL-test run")
    args = ap.parse_args()

    rog_batch = Path(args.rog_batch) if args.rog_batch else latest_batch("batch_LOCAL_FULLROG_")
    par_batch = Path(args.parlasent_batch) if args.parlasent_batch else latest_batch("batch_LOCAL_PARLASENT_")

    # ----- A: parlasent on full-ROG test (3-D over duration × WPS) -----
    if rog_batch is None:
        print("WARN: no batch_LOCAL_FULLROG_* dir found; skipping plot A.")
    else:
        run = find_parlasent_reg_3cls_run(rog_batch)
        if run is None:
            print(f"WARN: no parlasent reg 3cls run under {rog_batch}; skipping plot A.")
        else:
            df = u.load_or_run_predictions(run)
            fig, (dur_edges, wps_edges, f1, cnt, overall) = u.plot_predictions_topography_3d(
                df, title="ParlaSent-ZS on ROG v2 test (full, n=2233)")
            png = OUT / "topo_parlasent_zs_on_rog_full.png"
            fig.savefig(png, dpi=150, bbox_inches="tight")
            print(f"wrote {png.relative_to(ROOT)}  overall={overall:.3f}  "
                  f"peak={float(f1[~__import__('numpy').isnan(f1)].max()) if (f1==f1).any() else float('nan'):.3f}")

    # ----- B: parlasent on ParlaSent-SL test (1-D over word count) -----
    if par_batch is None:
        print("WARN: no batch_LOCAL_PARLASENT_* dir found; skipping plot B.")
    else:
        run = find_parlasent_reg_3cls_run(par_batch)
        if run is None:
            print(f"WARN: no parlasent reg 3cls run under {par_batch}; skipping plot B.")
        else:
            df = u.load_or_run_predictions(run)
            fig, (edges, f1, cnt, overall) = u.plot_predictions_envelope_1d(
                df, feature="words", title="ParlaSent-ZS on ParlaSent-SL test (home-domain)")
            png = OUT / "topo_parlasent_zs_on_parlasent.png"
            fig.savefig(png, dpi=150, bbox_inches="tight")
            print(f"wrote {png.relative_to(ROOT)}  overall={overall:.3f}")


if __name__ == "__main__":
    main()
