#!/usr/bin/env python3
"""Sync lightweight artifacts from runs/ into the committed results/ folder.

Pass a batch/matrix dir under runs/ and a destination slug under results/; this
script copies the keep-forever files (batch_meta.json, batch_results.csv,
predictions.jsonl → gzipped if >1MB, matrix_summary.json, matrix_long.csv,
matrix_report.md, heatmap_*.png, per-cell summary.json + predictions_*.jsonl.gz).

Weights/checkpoints/large logs stay in runs/ only.

Usage:
  python3 scripts/sync_results.py runs/batch_FULLROG_<stamp>          or_phase0/iso
  python3 scripts/sync_results.py runs/batch_FULLROG_CTXW30_<stamp>   or_phase0/ctxw30
  python3 scripts/sync_results.py runs/batch_LOCAL_FULLROG_<stamp>    parlasent_zs/on_rog_full
  python3 scripts/sync_results.py runs/batch_LOCAL_PARLASENT_<stamp>  parlasent_zs/on_parlasent
  python3 scripts/sync_results.py runs/finetune_<stamp>               finetune/<stamp>
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
GZIP_ABOVE_BYTES = 1_000_000  # 1 MB


KEEP_TOP = {
    "batch_meta.json", "batch_results.csv", "batch_results.json",
    "matrix_summary.json", "matrix_long.csv", "matrix_report.md",
    "overview_macro_f1.png", "overview_spearman.png",
    "overview_accuracy.png", "overview_ordinal_mae.png",
    "agreement_alpha_3cls.csv", "agreement_alpha_6cls.csv",
    "agreement_alpha_3cls.png", "agreement_alpha_6cls.png",
}
KEEP_SUFFIX = {".csv", ".md", ".png", ".json"}  # also keep these recursively
KEEP_PREDS = ("predictions.jsonl", "predictions_")  # name-starts-with


def copy_file(src: Path, dst: Path, gzip_large: bool = True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if gzip_large and src.suffix == ".jsonl" and src.stat().st_size > GZIP_ABOVE_BYTES:
        dst = dst.with_suffix(".jsonl.gz")
        with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo)
        print(f"  gz  {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}  "
              f"({src.stat().st_size / 1e6:.1f} MB → {dst.stat().st_size / 1e6:.2f} MB)")
    else:
        shutil.copy2(src, dst)
        print(f"  cp  {src.relative_to(ROOT)} → {dst.relative_to(ROOT)}")


def _is_keeper(p: Path) -> bool:
    if p.name in KEEP_TOP:
        return True
    if p.name.startswith(KEEP_PREDS):
        return True
    if p.suffix in KEEP_SUFFIX and p.parent.name != "logs":
        return True
    return False


def sync(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        raise SystemExit(f"source does not exist: {src_dir}")
    n = 0
    for p in src_dir.rglob("*"):
        if not p.is_file():
            continue
        if not _is_keeper(p):
            continue
        rel = p.relative_to(src_dir)
        copy_file(p, dst_dir / rel)
        n += 1
    print(f"\nSynced {n} files from {src_dir.relative_to(ROOT)} → {dst_dir.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="source dir under runs/ (e.g. runs/batch_FULLROG_20261118_...)")
    ap.add_argument("dst_slug", help="destination slug under results/ (e.g. or_phase0/iso)")
    args = ap.parse_args()
    sync(Path(args.src).resolve(), (RESULTS / args.dst_slug).resolve())


if __name__ == "__main__":
    main()
