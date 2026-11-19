"""Headless render of the context-envelope 3D plots on **full ROG test** (n=2233).

Sibling of render_ctx_envelopes.py. Same code path (utils_rog_explorer.plot_context_envelope_3d),
different batch pool (FULLROG iso + W={10,20,30,40}) and output filenames
(ctx_envelope_flash_{words,wps}_FULLROG.png — does NOT overwrite the 170-subset PNGs).

Duration axis is skipped (words = absolute length, wps = relative delivery; duration
is deferred to the audio story in Report 5).
"""
import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import utils_rog_explorer as u

BATCH_ROOT = ROOT / "runs" / "from_remote" / "Full_ROG_dia-v2_OR_test"
ISO_BATCH = BATCH_ROOT / "batch_FULLROG_20261118_115058"
CTX_BATCHES = {
    10: BATCH_ROOT / "batch_FULLROG_CTXW10_20261118_220839",
    20: BATCH_ROOT / "batch_FULLROG_CTXW20_20261119_000624",
    30: BATCH_ROOT / "batch_FULLROG_CTXW30_20261118_165459",
    40: BATCH_ROOT / "batch_FULLROG_CTXW40_20261119_020205",
}
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "flash"
SHOT = "0shot"
SCHEME = "3cls"
FEATURES = ("words", "wps")

sweep = u.load_context_sweep(ISO_BATCH, CTX_BATCHES, model=MODEL, shot=SHOT, scheme=SCHEME)
print(f"loaded sweep: Ws={sorted(sweep.keys())}  n_per_W={[len(v) for v in sweep.values()]}")

for feature in FEATURES:
    fig, (Ws, edges, F1, cnt, iso) = u.plot_context_envelope_3d(
        sweep, feature=feature, n_bins=5,
        title_suffix=f"{MODEL} {SHOT} {SCHEME} FULLROG")
    png = OUT / f"ctx_envelope_{MODEL}_{feature}_FULLROG.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png.relative_to(ROOT)}  iso={iso:.3f}  max={F1.max():.3f}  "
          f"min_cnt_per_bin={int(cnt[cnt>0].min()) if (cnt>0).any() else 0}")
