"""Headless render of the context-envelope 3D plots into docs/img/.

Rebuild after editing utils_rog_explorer.plot_context_envelope_3d (or the W-batch
list) without opening Jupyter. Prints the (W × target-feature) grid's iso and peak
F1 so you can eyeball regressions.

Mirrors the three cells in 06_rog_dia_explorer.(ipynb|py) under the context-envelope
section — single source of PNG truth.
"""
import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import utils_rog_explorer as u

ISO_BATCH = ROOT / "runs" / "batch_FULL_20261111_115520"
CTX_BATCHES = {
    0:  ROOT / "runs" / "batch_CTX_20261117_124601",
    10: ROOT / "runs" / "batch_CTXW10_20261117_140930",
    20: ROOT / "runs" / "batch_CTXW20_20261117_145015",
    30: ROOT / "runs" / "batch_CTXW30_20261117_152331",
    50: ROOT / "runs" / "batch_CTXW50_20261117_155630",
}
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

for model in ("flash", "haiku"):
    sweep = u.load_context_sweep(ISO_BATCH, CTX_BATCHES, model=model, shot="0shot", scheme="3cls")
    for feature in ("duration", "words", "wps"):
        fig, (Ws, edges, F1, cnt, iso) = u.plot_context_envelope_3d(
            sweep, feature=feature, n_bins=5, title_suffix=f"{model} 0-shot 3-cls")
        png = OUT / f"ctx_envelope_{model}_{feature}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"wrote {png.relative_to(ROOT)}  iso={iso:.3f}  max={F1.max():.3f}")
