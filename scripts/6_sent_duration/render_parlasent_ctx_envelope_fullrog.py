"""Headless render of the local-finetune context-envelope 3D plots on **full ROG test**.

Target: xlm-r-parlasent × R_full × W ∈ {10, 20, 30}, eval on rog_dia_full (n=2233).
This is the fine-tune analogue of render_ctx_envelopes_fullrog.py (which plots Flash).

W=0 (iso) predictions are not in this batch (they live in finetune_20261118_132942/),
so the baseline plane uses the published iso scalar F1_3 = 0.469 from Report 4.
If you sync the iso cell later, extend CTX_BATCHES with 0: <path> and the plotter
will pick it up.

Duration axis skipped (words = absolute length, wps = relative rate — those two
cover what text-only can say; duration becomes relevant in Report 5).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

BATCH = ROOT / "runs" / "from_remote" / "Full_ROG_dia-v2_XLM_test" / "finetune_20261118_160746"
CTX_CELLS = {
    10: BATCH / "xlm-r-parlasent__R_full_ctxw10" / "predictions_rog_dia_full.jsonl",
    20: BATCH / "xlm-r-parlasent__R_full_ctxw20" / "predictions_rog_dia_full.jsonl",
    30: BATCH / "xlm-r-parlasent__R_full_ctxw30" / "predictions_rog_dia_full.jsonl",
}
CORPUS = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

ISO_F1 = 0.469   # xlm-r-parlasent × R_full iso on rog_dia_full — from Report 4 table
LABELS3 = ["negative", "neutral", "positive"]
IDX3 = {l: i for i, l in enumerate(LABELS3)}
NORM3 = {"neg": "negative", "neu": "neutral", "pos": "positive"}


def _load_corpus_durations(corpus_path):
    """instance_id -> duration (seconds), text, word count."""
    out = {}
    with open(corpus_path) as f:
        for ln in f:
            r = json.loads(ln)
            iid = r["instance_id"]
            dur = (r.get("end_t") or 0.0) - (r.get("start_t") or 0.0)
            txt = r.get("text") or ""
            out[iid] = (dur, txt, len(txt.split()))
    return out


def load_predictions(path, corpus_meta):
    """Return DataFrame with instance_id, gold_3, pred_3, duration, text, words, wps."""
    rows = []
    with open(path) as f:
        for ln in f:
            r = json.loads(ln)
            g = NORM3.get(r.get("gold_3cls"), r.get("gold_3cls"))
            p = NORM3.get(r.get("pred_3cls"), r.get("pred_3cls"))
            if g not in IDX3 or p not in IDX3:
                continue
            iid = r["instance_id"]
            dur, txt, words = corpus_meta.get(iid, (None, r.get("text", ""), 0))
            if not words:
                words = len((r.get("text") or "").split())
            if not txt:
                txt = r.get("text") or ""
            wps = (words / dur) if (dur and dur > 0) else float("nan")
            rows.append({
                "instance_id": iid, "gold_3": g, "pred_3": p,
                "duration": dur, "text": txt, "words": words, "wps": wps,
            })
    return pd.DataFrame(rows)


def macro_f1(y_true, y_pred):
    f1s = []
    for L in LABELS3:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == L and p == L)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != L and p == L)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == L and p != L)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


# Manual bin edges for words: quantile collapses on this skewed distribution (46 %
# single-word utterances + a lone 25-word outlier). These bins give balanced counts
# and avoid the degenerate top bin.
WORDS_EDGES_MANUAL = np.array([1.0, 2.0, 4.0, 6.0, 16.0])

def envelope_grid(sweep, feature, n_bins=5, wps_clip=10.0):
    """Build (W, feature-bin) F1 grid. Quantile bin edges over the UNION of all W rows,
    except `words` uses the manual edges above (see WORDS_EDGES_MANUAL)."""
    def _feat(df):
        if feature == "words":
            return df["words"].values.astype(float)
        if feature == "wps":
            v = df["wps"].values.astype(float)
            return np.where(np.isnan(v), np.nan, np.minimum(v, wps_clip))
        raise ValueError(feature)

    if feature == "words":
        edges = WORDS_EDGES_MANUAL.copy()
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
        gold = df["gold_3"].tolist()
        pred = df["pred_3"].tolist()
        valid = ~np.isnan(feat) if feature == "wps" else np.ones(len(feat), dtype=bool)
        for j in range(n_bins):
            lo, hi = edges[j], edges[j + 1] + (1e-6 if j == n_bins - 1 else 0)
            mask = valid & (feat >= lo) & (feat < hi)
            if not mask.any():
                continue
            g = [gold[i] for i in range(len(gold)) if mask[i]]
            p = [pred[i] for i in range(len(pred)) if mask[i]]
            cnt[wi, j] = int(mask.sum())
            F1[wi, j] = macro_f1(g, p)
    return Ws, edges, F1, cnt


def plot_envelope(sweep, feature, iso_scalar, title_suffix, outpath,
                  n_bins=5, wps_clip=10.0, smooth_sigma=1.2, finer_grid=60):
    Ws, edges, F1, cnt = envelope_grid(sweep, feature, n_bins=n_bins, wps_clip=wps_clip)

    x_raw = np.arange(len(Ws), dtype=float)
    y_raw = 0.5 * (edges[:-1] + edges[1:])
    W_labels = [f"W={w}" for w in Ws]

    F1_filled = np.where(np.isnan(F1), iso_scalar, F1)
    interp = RegularGridInterpolator((x_raw, y_raw), F1_filled, method="linear",
                                     bounds_error=False, fill_value=iso_scalar)
    xf = np.linspace(x_raw.min(), x_raw.max(), finer_grid)
    yf = np.linspace(y_raw.min(), y_raw.max(), finer_grid)
    Xf, Yf = np.meshgrid(xf, yf, indexing="ij")
    Zf = interp(np.stack([Xf.ravel(), Yf.ravel()], axis=-1)).reshape(Xf.shape)
    Zf = gaussian_filter(Zf, sigma=smooth_sigma)

    vmin = max(0.0, iso_scalar - 0.10)
    vmax = min(1.0, max(float(np.nanmax(F1)) + 0.05, iso_scalar + 0.20))
    zlo = max(0.0, iso_scalar - 0.15)
    zhi = min(1.0, vmax + 0.02)

    fig = plt.figure(figsize=(11.5, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xf, Yf, Zf, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                           edgecolor="none", linewidth=0, alpha=0.92,
                           rstride=1, cstride=1)
    Zb = np.full_like(Xf, iso_scalar)
    ax.plot_surface(Xf, Yf, Zb, color="#6a6a6a", alpha=0.20,
                    edgecolor="none", linewidth=0)

    for i in range(len(Ws)):
        for j in range(n_bins):
            if cnt[i, j] == 0:
                continue
            x, y, z = x_raw[i], y_raw[j], F1[i, j]
            ax.plot([x, x], [y, y], [iso_scalar, z],
                    color=("#3a7a3a" if z >= iso_scalar else "#a04040"),
                    lw=0.8, alpha=0.75)
            ax.scatter([x], [y], [z], s=26,
                       c=[z], cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       edgecolor="black", linewidth=0.4, depthshade=False)

    ax.set_xticks(x_raw)
    ax.set_xticklabels(W_labels, fontsize=8.5)
    y_unit = {"words": "words", "wps": f"words/s, clip={wps_clip:g}"}[feature]
    ax.set_xlabel("context width"); ax.set_ylabel(f"target {feature} ({y_unit})")
    ax.set_zlabel("macro-F1 (3-cls)")
    ax.set_zlim(zlo, zhi)
    ax.view_init(elev=28, azim=-120)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, label="macro-F1")
    _ = cbar
    ax.set_title(f"Local-finetune envelope — F1 vs (W, target {feature})  "
                 f"iso baseline={iso_scalar:.3f}{('  ·  ' + title_suffix) if title_suffix else ''}")
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Ws, edges, F1, cnt


def main():
    corpus_meta = _load_corpus_durations(CORPUS)
    sweep = {W: load_predictions(p, corpus_meta) for W, p in CTX_CELLS.items()}
    print(f"loaded sweep: Ws={sorted(sweep.keys())}  n_per_W={[len(v) for v in sweep.values()]}")
    # words distribution has a 1-row outlier at the max (25-word bin), so 5 quantile
    # bins collapse the last one to n=1 — use 4 bins for words, 5 for wps.
    bins_per_feature = {"words": 4, "wps": 5}
    for feature in ("words", "wps"):
        outpath = OUT / f"ctx_envelope_parlasent_R_full_{feature}_FULLROG.png"
        Ws, edges, F1, cnt = plot_envelope(
            sweep, feature=feature, iso_scalar=ISO_F1,
            title_suffix="xlm-r-parlasent × R_full · FULLROG",
            outpath=outpath, n_bins=bins_per_feature[feature])
        print(f"wrote {outpath.relative_to(ROOT)}  iso_scalar={ISO_F1:.3f}  "
              f"max={np.nanmax(F1):.3f}  min_cnt={int(cnt[cnt>0].min()) if (cnt>0).any() else 0}")
        for wi, W in enumerate(Ws):
            print(f"    W={W}: " + " ".join(f"{v:.3f}" if not np.isnan(v) else "  nan" for v in F1[wi]))


if __name__ == "__main__":
    main()
