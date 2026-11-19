"""Headless render of the 6-class context-envelope 3D plots on **full ROG test**.

Sibling of render_ctx_envelopes_fullrog.py (which plots the 3-cls envelope).
The utility context_envelope_grid in utils_rog_explorer is 3-cls-only (sparse
6-cls bins on the 170-subset); on n=2233 FULLROG we have ~446 rows / bin, so
6-cls macro-F1 is usable. Grid+plot are inlined here to keep that call explicit.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import utils_rog_explorer as u

BATCH_ROOT = ROOT / "runs" / "from_remote" / "Full_ROG_dia-v2_OR_test"
ISO_BATCH = BATCH_ROOT / "batch_FULLROG_20261118_115058"
CTX_BATCHES = {
    10: BATCH_ROOT / "batch_FULLROG_CTXW10_6cls_20261119_035451",
    20: BATCH_ROOT / "batch_FULLROG_CTXW20_6cls_20261119_063238",
    30: BATCH_ROOT / "batch_FULLROG_CTXW30_20261118_165459",  # has both schemes; picker grabs 6cls
    40: BATCH_ROOT / "batch_FULLROG_CTXW40_6cls_20261119_091517",
}
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "flash"
SHOT = "0shot"
SCHEME = "6cls"
FEATURES = ("words", "wps")

LABELS_6 = [
    "predominantlyNegative", "mixedNegative", "neutralNegative",
    "neutralPositive", "mixedPositive", "predominantlyPositive",
]


def macro_f1_6(y_true, y_pred):
    f1s = []
    for L in LABELS_6:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == L and p == L)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != L and p == L)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == L and p != L)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def envelope_grid_6cls(sweep, feature, n_bins=5, wps_clip=10.0, words_edges=None):
    def _feat(df):
        if feature == "words":
            return df["words"].values.astype(float)
        if feature == "wps":
            v = df["wps"].values.astype(float)
            return np.where(np.isnan(v), np.nan, np.minimum(v, wps_clip))
        raise ValueError(feature)

    if feature == "words" and words_edges is not None:
        edges = np.asarray(words_edges, dtype=float)
        n_bins = len(edges) - 1
    else:
        all_feat = []
        for df in sweep.values():
            v = _feat(df)
            v = v[~np.isnan(v)] if feature == "wps" else v
            all_feat.extend(list(v))
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(all_feat, qs)
        edges = np.unique(edges)
        while len(edges) < n_bins + 1:
            edges = np.concatenate([edges, [edges[-1] + 1e-6]])

    Ws = sorted(sweep.keys())
    F1 = np.full((len(Ws), n_bins), np.nan)
    cnt = np.zeros((len(Ws), n_bins), dtype=int)
    for wi, W in enumerate(Ws):
        df = sweep[W]
        feat = _feat(df)
        gold = df["gold_6"].tolist()
        pred = df["pred_6"].tolist()
        valid = ~np.isnan(feat) if feature == "wps" else np.ones(len(feat), dtype=bool)
        for j in range(n_bins):
            lo, hi = edges[j], edges[j + 1] + (1e-6 if j == n_bins - 1 else 0)
            mask = valid & (feat >= lo) & (feat < hi)
            if not mask.any():
                continue
            g = [gold[i] for i in range(len(gold)) if mask[i] and gold[i] is not None and pred[i] is not None]
            p = [pred[i] for i in range(len(pred)) if mask[i] and gold[i] is not None and pred[i] is not None]
            if not g:
                continue
            cnt[wi, j] = len(g)
            F1[wi, j] = macro_f1_6(g, p)
    return Ws, edges, F1, cnt


def plot_envelope_6(sweep, feature, iso_scalar, title_suffix, outpath,
                    n_bins=5, wps_clip=10.0, smooth_sigma=1.2, finer_grid=60,
                    words_edges=None):
    Ws, edges, F1, cnt = envelope_grid_6cls(
        sweep, feature=feature, n_bins=n_bins, wps_clip=wps_clip, words_edges=words_edges)

    # Fit an "iso row" into the sweep for plotter parity. Plotter uses Ws.index(-1).
    # If the iso row is in the sweep (key -1), use its row; otherwise use iso_scalar flat.
    if -1 not in Ws:
        Ws = [-1] + Ws
        F1 = np.vstack([np.full(F1.shape[1], iso_scalar), F1])
        cnt = np.vstack([np.full(F1.shape[1], 0, dtype=int), cnt])

    x_raw = np.arange(len(Ws), dtype=float)
    y_raw = 0.5 * (edges[:-1] + edges[1:])
    W_labels = ["iso" if w == -1 else f"W={w}" for w in Ws]
    iso_mean = float(np.nanmean(F1[Ws.index(-1)]))

    F1_filled = np.where(np.isnan(F1), iso_mean, F1)
    interp = RegularGridInterpolator((x_raw, y_raw), F1_filled, method="linear",
                                     bounds_error=False, fill_value=iso_mean)
    xf = np.linspace(x_raw.min(), x_raw.max(), finer_grid)
    yf = np.linspace(y_raw.min(), y_raw.max(), finer_grid)
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    Zf = interp(np.stack([Xf.ravel(), Yf.ravel()], axis=-1)).reshape(Xf.shape)
    Zf = gaussian_filter(Zf, sigma=smooth_sigma)

    vmin = max(0.0, iso_mean - 0.10)
    vmax = min(1.0, max(float(np.nanmax(F1)) + 0.05, iso_mean + 0.25))
    zlo = max(0.0, iso_mean - 0.15)
    zhi = min(1.0, vmax + 0.02)

    fig = plt.figure(figsize=(11.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xf, Yf, Zf, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                           edgecolor="none", linewidth=0, alpha=0.92,
                           rstride=1, cstride=1)
    Zb = np.full_like(Xf, iso_mean)
    ax.plot_surface(Xf, Yf, Zb, color="#6a6a6a", alpha=0.20,
                    edgecolor="none", linewidth=0)
    for i in range(len(Ws)):
        for j in range(F1.shape[1]):
            if cnt[i, j] == 0 and Ws[i] != -1:
                continue
            x, y, z = x_raw[i], y_raw[j], F1[i, j]
            if np.isnan(z):
                continue
            ax.plot([x, x], [y, y], [iso_mean, z],
                    color=("#3a7a3a" if z >= iso_mean else "#a04040"),
                    lw=0.8, alpha=0.75)
            ax.scatter([x], [y], [z], s=26,
                       c=[z], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       edgecolor="black", linewidth=0.4, depthshade=False)

    ax.set_xticks(x_raw)
    ax.set_xticklabels(W_labels, fontsize=8.5)
    y_unit = {"words": "words", "wps": f"words/s, clip={wps_clip:g}"}[feature]
    ax.set_xlabel("context width"); ax.set_ylabel(f"target {feature} ({y_unit})")
    ax.set_zlabel("macro-F1 (6-cls)")
    ax.set_zlim(zlo, zhi)
    ax.view_init(elev=28, azim=-120)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, label="macro-F1")
    _ = cbar
    ax.set_title(f"Context envelope (6-cls) — F1 vs (W, target {feature})  "
                 f"iso={iso_mean:.3f}{('  ·  ' + title_suffix) if title_suffix else ''}")
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Ws, edges, F1, cnt


def main():
    sweep = u.load_context_sweep(ISO_BATCH, CTX_BATCHES, model=MODEL, shot=SHOT, scheme=SCHEME)
    print(f"loaded sweep: Ws={sorted(sweep.keys())}  n_per_W={[len(v) for v in sweep.values()]}")

    # Pre-compute an iso scalar from the iso row (W=-1 in the sweep)
    # by averaging per-bin F1 at any feature. Easier: compute whole-eval macro F1 on iso.
    iso_df = sweep[-1]
    iso_f1 = macro_f1_6([g for g in iso_df["gold_6"] if g is not None],
                        [p for g, p in zip(iso_df["gold_6"], iso_df["pred_6"]) if g is not None])
    print(f"iso whole-eval 6cls F1 = {iso_f1:.3f}")

    # Manual words edges (same fix as the parlasent plot: the lone 25-word utterance
    # collapses the top quantile bin to n=1).
    words_edges = [1.0, 2.0, 4.0, 6.0, 16.0]

    for feature in FEATURES:
        outpath = OUT / f"ctx_envelope_flash_{feature}_6cls_FULLROG.png"
        we = words_edges if feature == "words" else None
        Ws, edges, F1, cnt = plot_envelope_6(
            sweep, feature=feature, iso_scalar=iso_f1,
            title_suffix=f"{MODEL} {SHOT} {SCHEME} FULLROG",
            outpath=outpath, words_edges=we)
        print(f"wrote {outpath.relative_to(ROOT)}  iso_scalar={iso_f1:.3f}  "
              f"max={np.nanmax(F1):.3f}  min_cnt={int(cnt[cnt>0].min()) if (cnt>0).any() else 0}")
        for wi, W in enumerate(Ws):
            tag = "iso" if W == -1 else f"W={W}"
            print(f"    {tag:>5}: " + " ".join(f"{v:.3f}" if not np.isnan(v) else "  nan" for v in F1[wi]))


if __name__ == "__main__":
    main()
