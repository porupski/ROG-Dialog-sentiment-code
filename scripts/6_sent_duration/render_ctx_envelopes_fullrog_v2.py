"""Headless v2 renderer — context envelopes on FULL ROG test, 3cls + 6cls.

One run emits four PNGs:
  ctx_envelope_flash_words_FULLROG_v2.png
  ctx_envelope_flash_wps_FULLROG_v2.png
  ctx_envelope_flash_words_6cls_FULLROG_v2.png
  ctx_envelope_flash_wps_6cls_FULLROG_v2.png

Layout per PNG: 2x3 grid, main 3-D on the left 2x2, two 1-D side profiles on
the right column (F1 vs feature, F1 vs context-W). Signal surface uses the
RdYlGn colormap; baseline water plane drawn at the iso macro-F1; optional IAA
ceiling plane drawn at `iaa_ceiling` with dual-layer compositing.

Per-feature bin edges are set manually in EDGES; switching a feature to
BIN_MODE="auto_p" delegates to the chi-square greedy auto-binning in
utils_rog_explorer._auto_bin_edges_p (ignores EDGES for that feature).
"""
import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import utils_rog_explorer as u

BATCH_ROOT = ROOT / "runs" / "from_remote" / "Full_ROG_dia-v2_OR_test"
ISO_BATCH = BATCH_ROOT / "batch_FULLROG_20261118_115058"

# Per-scheme context batches. W=30 shares a mixed batch; the dir picker
# inside load_context_sweep selects the correct scheme's run.
SCHEMES = {
    "3cls": {
        "scheme_util": "3",
        "ctx_batches": {
            10: BATCH_ROOT / "batch_FULLROG_CTXW10_20261118_220839",
            20: BATCH_ROOT / "batch_FULLROG_CTXW20_20261119_000624",
            30: BATCH_ROOT / "batch_FULLROG_CTXW30_20261118_165459",
            40: BATCH_ROOT / "batch_FULLROG_CTXW40_20261119_020205",
        },
        "iaa_ceiling": 0.699,
    },
    "6cls": {
        "scheme_util": "6",
        "ctx_batches": {
            10: BATCH_ROOT / "batch_FULLROG_CTXW10_6cls_20261119_035451",
            20: BATCH_ROOT / "batch_FULLROG_CTXW20_6cls_20261119_063238",
            30: BATCH_ROOT / "batch_FULLROG_CTXW30_20261118_165459",
            40: BATCH_ROOT / "batch_FULLROG_CTXW40_6cls_20261119_091517",
        },
        "iaa_ceiling": 0.625,
    },
}

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "flash"
SHOT = "0shot"
FEATURES = ("words", "wps")

# Manual bin edges. Quantile edges collapse on `words` (46 % of ROG is 1-word)
# so explicit integer edges give better mid-range resolution.
EDGES = {
    "words": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18],
    "wps":   [1, 2, 3, 4, 5, 6, 7, 8, 10],
}

# Per-feature binning mode: "manual" uses EDGES, "auto_p" runs the chi-square
# greedy auto-selector in utils_rog_explorer._auto_bin_edges_p.
BIN_MODE = {
    "words": "auto_p",#"manual",
    "wps":   "manual",#"manual",
}

# Rendering knobs.
DRAW_INTERSECTION  = False
WATER_COLOR        = "#5dade2"   # baseline plane tint
CEILING_COLOR      = "#d7a4b3"   # IAA ceiling plane tint
INTERSECTION_COLOR = "#00e5ff"   # cyan contour at baseline crossing
CEILING_INTERSECTION_COLOR = "#ff2ea6"   # magenta contour at ceiling crossing


def _mode_and_edges(feature):
    bm = BIN_MODE.get(feature, "manual")
    if bm == "auto_p":
        return "auto_p", None
    return "quantile", EDGES.get(feature)


for scheme_tag, scheme_cfg in SCHEMES.items():
    scheme_util = scheme_cfg["scheme_util"]
    ctx_batches = scheme_cfg["ctx_batches"]
    iaa_ceiling = scheme_cfg["iaa_ceiling"]

    sweep = u.load_context_sweep(ISO_BATCH, ctx_batches,
                                 model=MODEL, shot=SHOT, scheme=scheme_tag)
    print(f"[{scheme_tag}] loaded sweep: Ws={sorted(sweep.keys())}  "
          f"n_per_W={[len(v) for v in sweep.values()]}")

    grids = {}
    for feature in FEATURES:
        mode, edges_override = _mode_and_edges(feature)
        Ws, edges, F1, cnt = u.context_envelope_grid(
            sweep, feature=feature, n_bins=5, mode=mode,
            scheme=scheme_util, edges_override=edges_override)
        grids[feature] = (Ws, edges, F1, cnt)
        iso_mean = float(np.nanmean(F1[Ws.index(-1)]))
        print(f"  {feature}: mode={BIN_MODE.get(feature, 'manual')}  "
              f"edges={np.round(edges, 2).tolist()}  "
              f"iso_mean={iso_mean:.3f}  raw_F1_max={np.nanmax(F1):.3f}  "
              f"min_cnt_per_bin={int(cnt[cnt>0].min()) if (cnt>0).any() else 0}")

    iso_means = [float(np.nanmean(F1[Ws.index(-1)])) for Ws, _, F1, _ in grids.values()]
    all_max   = max(float(np.nanmax(F1)) for _, _, F1, _ in grids.values())
    zlo = max(0.0, min(iso_means) - 0.15)
    zhi = min(1.0, all_max + 0.03)
    if iaa_ceiling is not None:
        zhi = max(zhi, iaa_ceiling + 0.03)
    print(f"  joint z-axis: ({zlo:.3f}, {zhi:.3f})")

    for feature in FEATURES:
        mode, edges_override = _mode_and_edges(feature)
        fig, (Ws, edges, F1, cnt, iso) = u.plot_context_envelope_3d_v2(
            sweep, feature=feature, n_bins=5, mode=mode, scheme=scheme_util,
            title_suffix=f"{MODEL} {SHOT} {scheme_tag} FULLROG",
            zlim=(zlo, zhi), edges_override=edges_override,
            iaa_ceiling=iaa_ceiling,
            water_color=WATER_COLOR,
            ceiling_color=CEILING_COLOR,
            intersection_color=INTERSECTION_COLOR,
            ceiling_intersection_color=CEILING_INTERSECTION_COLOR,
            draw_intersection=DRAW_INTERSECTION)
        tag = "_6cls" if scheme_tag == "6cls" else ""
        png = OUT / f"ctx_envelope_{MODEL}_{feature}{tag}_FULLROG_v2.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"  wrote {png.relative_to(ROOT)}  iso={iso:.3f}  "
              f"raw_max={np.nanmax(F1):.3f}  "
              f"min_cnt_per_bin={int(cnt[cnt>0].min()) if (cnt>0).any() else 0}")
