#!/usr/bin/env python3
"""Compute metrics and plots for a BATCH of OpenRouter / local-model PROBE runs.

Consumes the per-run predictions.jsonl emitted by scripts/2_probe_text/*,
scripts/3_probe_audio/*, and scripts/2_probe_text/04_run_parlasent.py. For the
4_finetune_text matrix runner use `scripts/analyze_finetune.py` instead — that
format is per-cell summary.json, not predictions.jsonl.

Per-run metrics: accuracy_all, accuracy_parsed, parse_rate, macro_f1, spearman_rho, ordinal_mae.
Per-run artefacts: predictions.jsonl (input), confusion_matrix.png.
Batch-level artefacts: batch_results.csv, batch_results.json,
  overview_macro_f1.png, overview_spearman.png, overview_accuracy.png,
  agreement_alpha_6cls.{png,csv}, agreement_alpha_3cls.{png,csv}.

Metric choices:
- macro_f1 and accuracy_parsed computed on parsed predictions only (parse failures are a
  separate failure mode and reported via parse_rate).
- spearman_rho uses scipy.stats.spearmanr on ordinal label indices.
- ordinal_mae is the mean |gold_ord - pred_ord| on parsed predictions.
- agreement: Krippendorff's alpha (ordinal), pairwise between all runs in the same class scheme.
  Falls back to a minimal interval-level alpha if the `krippendorff` package is missing.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS_DIR = ROOT / "runs"

LABELS_6 = [
    "predominantlyNegative",
    "mixedNegative",
    "neutralNegative",
    "neutralPositive",
    "mixedPositive",
    "predominantlyPositive",
]
LABELS_3 = ["negative", "neutral", "positive"]
SENT_6_ORD = {l: i for i, l in enumerate(LABELS_6)}
SENT_3_ORD = {l: i for i, l in enumerate(LABELS_3)}


def labels_and_ord(scheme):
    return (LABELS_6, SENT_6_ORD) if scheme == "6cls" else (LABELS_3, SENT_3_ORD)


def parse_label(output, class_scheme):
    """Kept in sync with 02_run_openrouter_text.py. Analyzer re-parses output_text to fix
    cases where the model wrote ambiguous or structured output and the old parser accepted
    the first in-list label (e.g. "neutral or negative?" → was "negative", now ambiguous)."""
    if not output:
        return None, False
    labels = LABELS_6 if class_scheme == "6cls" else LABELS_3
    labels_lc = [l.lower() for l in labels]
    first_line = output.strip().split("\n")[0]
    s_strict = first_line.strip().strip("`\"' \n.,:;!?*->→ ").lower()
    for i, lbl in enumerate(labels_lc):
        if s_strict == lbl:
            return labels[i], True
    sl = output.lower()
    found = []
    for i, lbl in enumerate(labels_lc):
        pos = sl.find(lbl)
        if pos >= 0:
            found.append((pos, labels[i]))
    if found:
        found.sort()
        distinct = {x[1] for x in found}
        if len(distinct) > 1:
            return found[0][1], False
        return found[0][1], True
    return None, False


def load_predictions(run_dir):
    return [json.loads(line) for line in open(run_dir / "predictions.jsonl")]


def per_class_f1(y_true, y_pred, labels):
    out = {}
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def spearman_rho(x, y):
    if len(x) < 2:
        return None
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y).correlation
        return float(r) if r == r else None
    except ImportError:
        return None


def confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def metrics_for_run(run_dir, config):
    scheme = config["class_scheme"]
    labels, ord_map = labels_and_ord(scheme)
    records = load_predictions(run_dir)
    # Re-parse output_text with the current (stricter) parser. For LOCAL runs the
    # output_text is a raw regression score and the parsed_label is set directly by 04,
    # so skip re-parsing there.
    n_reparsed_changed = 0
    for r in records:
        if config.get("infra") == "LocalHF":
            continue
        new_label, new_ok = parse_label(r.get("output_text"), scheme)
        if new_label != r.get("parsed_label") or new_ok != r.get("parse_ok"):
            n_reparsed_changed += 1
            r["parsed_label"] = new_label
            r["parse_ok"] = new_ok
    n = len(records)
    parsed = [r for r in records if r.get("parse_ok")]
    n_parsed = len(parsed)
    parse_rate = n_parsed / n if n else 0.0

    n_correct_all = sum(1 for r in records if r.get("parse_ok") and r["parsed_label"] == r["gold_for_scheme"])
    acc_all = n_correct_all / n if n else 0.0

    out = {
        "n": n, "n_parsed": n_parsed, "parse_rate": parse_rate,
        "accuracy_all": acc_all, "accuracy_parsed": None,
        "macro_f1": None, "spearman_rho": None, "ordinal_mae": None,
        "per_class": {}, "confusion_matrix": None, "labels_order": labels,
        "gold_ord": None, "pred_ord": None,
    }
    if n_parsed == 0:
        return out

    y_true = [r["gold_for_scheme"] for r in parsed]
    y_pred = [r["parsed_label"] for r in parsed]
    out["accuracy_parsed"] = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n_parsed
    pc = per_class_f1(y_true, y_pred, labels)
    out["per_class"] = pc
    out["macro_f1"] = sum(v["f1"] for v in pc.values()) / len(labels)
    g_ord = [ord_map[t] for t in y_true]
    p_ord = [ord_map[p] for p in y_pred]
    out["spearman_rho"] = spearman_rho(g_ord, p_ord)
    out["ordinal_mae"] = sum(abs(a - b) for a, b in zip(g_ord, p_ord)) / n_parsed
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels)
    out["n_reparsed_changed"] = n_reparsed_changed

    # Also store per-instance ord values for agreement analysis (keyed by instance_id)
    per_inst = {}
    for r in parsed:
        per_inst[r["instance_id"]] = ord_map[r["parsed_label"]]
    out["pred_ord_by_instance"] = per_inst
    return out


_TS_RE = __import__("re").compile(r"_(\d{8})_(\d{6})$")


def _batch_ts_key(path):
    """Sort by the trailing _YYYYMMDD_HHMMSS in the folder name, not the alphabetical prefix.
    `batch_AUDIO_FULL_20261112_112856` should outrank `batch_LOCAL_FULL_20261111_162030`."""
    m = _TS_RE.search(path.name)
    if not m:
        return ("", "", path.name)
    return (m.group(1), m.group(2), path.name)


def latest_batch(runs_dir):
    cands = [p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("batch_")]
    if not cands:
        raise SystemExit(f"No batch_* folders under {runs_dir}")
    cands.sort(key=_batch_ts_key)
    return cands[-1]


def krippendorff_alpha_ordinal(rater_values, n_categories):
    """Krippendorff's alpha with the ordinal metric (Krippendorff 2011 formulation).
    rater_values: list of lists, one per rater, same length; use None for missing values.
    n_categories: number of ordinal categories (values are 0..n_categories-1).
    """
    n_raters = len(rater_values)
    if n_raters < 2:
        return None
    N = len(rater_values[0])
    if any(len(r) != N for r in rater_values):
        return None

    O = [[0.0] * n_categories for _ in range(n_categories)]
    for u in range(N):
        vals = [r[u] for r in rater_values if r[u] is not None]
        m = len(vals)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                O[vals[i]][vals[j]] += w

    n_c = [sum(O[c]) for c in range(n_categories)]
    n = sum(n_c)
    if n < 2:
        return None

    def delta2(c, k):
        if c == k:
            return 0.0
        a, b = (c, k) if c < k else (k, c)
        s = sum(n_c[g] for g in range(a, b + 1)) - (n_c[a] + n_c[b]) / 2.0
        return s * s

    Do = sum(O[c][k] * delta2(c, k) for c in range(n_categories) for k in range(n_categories))
    De_sum = sum(n_c[c] * n_c[k] * delta2(c, k) for c in range(n_categories) for k in range(n_categories))
    if De_sum == 0:
        return None
    return 1.0 - (n - 1) * Do / De_sum


def plot_cm(cm, labels, out_path, title):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(labels) + 2), max(4, 0.7 * len(labels) + 2)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title(title)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_overview_heatmap(rows, metric_key, out_path, title, cmap="RdYlGn", vmin=0.0, vmax=1.0, lower_better=False):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    # Prefix model with modality when the batch contains more than one — otherwise the
    # text "flash" and audio "flash" rows would collapse into the same y-axis slot.
    modalities = {r.get("modality", "text") for r in rows}
    if len(modalities) > 1:
        row_label = lambda r: f"{r.get('modality','text')}:{r['model']}"
    else:
        row_label = lambda r: r["model"]
    models = sorted({row_label(r) for r in rows})
    col_keys = []
    seen = set()
    for r in rows:
        k = (r["shot"], r["classes"])
        if k not in seen:
            seen.add(k)
            col_keys.append(k)
    col_keys.sort(key=lambda x: (x[0], x[1]))
    mat = np.full((len(models), len(col_keys)), np.nan)
    for r in rows:
        if r[metric_key] is None:
            continue
        i = models.index(row_label(r))
        j = col_keys.index((r["shot"], r["classes"]))
        mat[i, j] = r[metric_key]

    fig, ax = plt.subplots(figsize=(1.5 * len(col_keys) + 2, 0.7 * len(models) + 2))
    if lower_better:
        cmap = "RdYlGn_r"
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_keys)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([f"{s}-{c}" for s, c in col_keys], rotation=30, ha="right")
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(col_keys)):
            v = mat[i, j]
            txt = "—" if np.isnan(v) else f"{v:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=10)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_alpha_matrix(alpha_mat, run_names, out_path, title):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return False
    n = len(run_names)
    mat = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            v = alpha_mat[i][j]
            if v is not None:
                mat[i, j] = v
    fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.0 * n + 2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(run_names, rotation=45, ha="right")
    ax.set_yticklabels(run_names)
    for i in range(n):
        for j in range(n):
            v = alpha_mat[i][j]
            txt = "—" if v is None else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def build_alpha_matrix(details, scheme):
    runs = [d for d in details if d["classes"] == scheme]
    if len(runs) < 2:
        return [], [[]]
    all_inst = set()
    for d in runs:
        all_inst.update(d["pred_ord_by_instance"].keys())
    all_inst = sorted(all_inst)
    n_cats = len(LABELS_6 if scheme == "6cls" else LABELS_3)
    names = [d["run_id"] for d in runs]
    mat = [[None] * len(runs) for _ in runs]
    for i in range(len(runs)):
        for j in range(len(runs)):
            if i == j:
                mat[i][j] = 1.0
                continue
            r1 = [runs[i]["pred_ord_by_instance"].get(k) for k in all_inst]
            r2 = [runs[j]["pred_ord_by_instance"].get(k) for k in all_inst]
            mat[i][j] = krippendorff_alpha_ordinal([r1, r2], n_cats)
    return names, mat


def fmt(v, n=3):
    if v is None:
        return "   -"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", nargs="?", default=None)
    ap.add_argument("--include-batch", action="append", default=[],
                    help="Extra batch folder(s) contributing runs for the α matrix. "
                         "Can be passed multiple times. Primary batch still owns the output files.")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    batch_dir = (Path(args.batch).resolve() if args.batch else latest_batch(RUNS_DIR))
    try:
        rel = batch_dir.relative_to(ROOT)
    except ValueError:
        rel = batch_dir
    print(f"Analyzing batch: {rel}")
    include_dirs = [Path(p).resolve() for p in args.include_batch]
    for p in include_dirs:
        try:
            print(f"  + include: {p.relative_to(ROOT)}")
        except ValueError:
            print(f"  + include: {p}")
    print()

    rows = []
    details = []
    RUN_GLOBS = ("OR-text-*", "OR-audio-*", "LOCAL-text-*", "LOCAL-audio-*")

    def collect_runs(dir_):
        out = []
        for g in RUN_GLOBS:
            out.extend(sorted(dir_.glob(g)))
        return out

    primary_runs = collect_runs(batch_dir)
    extra_runs = []
    for p in include_dirs:
        extra_runs.extend(collect_runs(p))
    for is_primary, run_dir in [(True, r) for r in primary_runs] + [(False, r) for r in extra_runs]:
        meta_p = run_dir / "meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        cfg = meta["config"]
        m = metrics_for_run(run_dir, cfg)
        row = {
            "run_id": cfg["run_id"],
            "modality": cfg.get("modality", "text"),
            "model": cfg.get("model_name", cfg.get("model", "?")),
            "protocol": cfg.get("protocol", "api"),
            "shot": cfg["shot_mode"],
            "classes": cfg["class_scheme"],
            "n": m["n"],
            "n_parsed": m["n_parsed"],
            "parse_rate": m["parse_rate"],
            "accuracy_all": m["accuracy_all"],
            "accuracy_parsed": m["accuracy_parsed"],
            "macro_f1": m["macro_f1"],
            "spearman_rho": m["spearman_rho"],
            "ordinal_mae": m["ordinal_mae"],
            "cost_usd": meta.get("total_cost_usd", 0.0),
            "elapsed_sec": meta.get("elapsed_sec", 0.0),
            "n_reparsed_changed": m.get("n_reparsed_changed", 0),
            "is_primary": is_primary,
            "batch_src": run_dir.parent.name,
        }
        if is_primary:
            rows.append(row)
        details.append({
            **row,
            "per_class": m["per_class"],
            "confusion_matrix": m["confusion_matrix"],
            "labels_order": m["labels_order"],
            "pred_ord_by_instance": m.get("pred_ord_by_instance", {}),
        })

        if is_primary and not args.no_plots and m["confusion_matrix"] is not None:
            plot_cm(m["confusion_matrix"], m["labels_order"],
                    run_dir / "confusion_matrix.png", cfg["run_id"])

    rows.sort(key=lambda r: (r["classes"], r["modality"], r["model"], r["protocol"], r["shot"]))

    if not rows:
        print(f"No runs found in {batch_dir}. Expected subfolders matching one of: {', '.join(RUN_GLOBS)}.")
        return

    cols = [("modality", 8), ("model", 11), ("protocol", 9), ("shot", 6), ("classes", 7),
            ("n", 4), ("n_parsed", 8),
            ("parse_rate", 10), ("accuracy_all", 12), ("accuracy_parsed", 14),
            ("macro_f1", 9), ("spearman_rho", 12), ("ordinal_mae", 11),
            ("cost_usd", 10), ("elapsed_sec", 11)]
    header = " | ".join(f"{c:<{w}}" for c, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        parts = []
        for c, w in cols:
            v = r[c]
            if c in ("parse_rate", "accuracy_all", "accuracy_parsed", "macro_f1", "spearman_rho", "ordinal_mae"):
                s = fmt(v, 3)
            elif c == "cost_usd":
                s = fmt(v, 5)
            elif c == "elapsed_sec":
                s = fmt(v, 2)
            else:
                s = str(v)
            parts.append(f"{s:<{w}}")
        print(" | ".join(parts))

    csv_path = batch_dir / "batch_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (batch_dir / "batch_results.json").write_text(
        json.dumps([{k: v for k, v in d.items() if k != "pred_ord_by_instance"} for d in details],
                   indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {csv_path.relative_to(ROOT)}")

    if not args.no_plots:
        plot_overview_heatmap(rows, "macro_f1", batch_dir / "overview_macro_f1.png",
                              f"macro-F1 — {batch_dir.name}", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        plot_overview_heatmap(rows, "spearman_rho", batch_dir / "overview_spearman.png",
                              f"Spearman ρ — {batch_dir.name}", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        plot_overview_heatmap(rows, "accuracy_parsed", batch_dir / "overview_accuracy.png",
                              f"accuracy — {batch_dir.name}", cmap="RdYlGn", vmin=0.0, vmax=1.0)
        max_mae = max((r["ordinal_mae"] or 0) for r in rows)
        plot_overview_heatmap(rows, "ordinal_mae", batch_dir / "overview_ordinal_mae.png",
                              f"ordinal MAE (lower better) — {batch_dir.name}",
                              vmin=0.0, vmax=max(1.0, max_mae), lower_better=True)
        print(f"Wrote overview_*.png")

        alpha_summary_rows = []
        for scheme in ("6cls", "3cls"):
            names, mat = build_alpha_matrix(details, scheme)
            if not names:
                continue
            with open(batch_dir / f"agreement_alpha_{scheme}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([""] + names)
                for i, n in enumerate(names):
                    w.writerow([n] + ["" if v is None else f"{v:.4f}" for v in mat[i]])
            plot_alpha_matrix(mat, names, batch_dir / f"agreement_alpha_{scheme}.png",
                              f"Krippendorff ordinal α — {scheme} — {batch_dir.name}")
            print(f"Wrote agreement_alpha_{scheme}.(csv|png)")

            idx = {n: i for i, n in enumerate(names)}

            def model_of(name):
                return name.split("-")[2]

            def shot_of(name):
                return name.split("-")[3]

            # Compute cross-model and within-model-shot-flip α for each (infra, modality)
            # family separately — flash-text vs flash-audio is apples-to-oranges here.
            for prefix in ("OR-text-", "OR-audio-"):
                fam_runs = [n for n in names if n.startswith(prefix)]
                if not fam_runs:
                    continue
                modality = prefix.strip("-").split("-")[1]  # text / audio
                models_seen = sorted({model_of(n) for n in fam_runs})
                for shot in ("0shot", "fshot"):
                    pair_names = [n for n in fam_runs if shot_of(n) == shot]
                    for i in range(len(pair_names)):
                        for j in range(i + 1, len(pair_names)):
                            a, b = pair_names[i], pair_names[j]
                            val = mat[idx[a]][idx[b]]
                            if val is not None:
                                alpha_summary_rows.append((f"cross-model ({modality})", scheme, shot,
                                                          f"{model_of(a)} vs {model_of(b)}", val))
                for m in models_seen:
                    a = f"{prefix}{m}-0shot-iso-{scheme}"
                    b = f"{prefix}{m}-fshot-iso-{scheme}"
                    if a in idx and b in idx:
                        val = mat[idx[a]][idx[b]]
                        if val is not None:
                            alpha_summary_rows.append((f"within-model shot-flip ({modality})",
                                                      scheme, m, "0shot vs fshot", val))

        if alpha_summary_rows:
            print("\n=== Krippendorff ordinal α — focused pairs ===")
            print(f"{'kind':<32} {'scheme':<6} {'axis':<7} {'pair':<18} {'alpha':>7}")
            for kind, scheme, axis, pair, val in alpha_summary_rows:
                print(f"{kind:<32} {scheme:<6} {axis:<7} {pair:<18} {val:>7.3f}")


if __name__ == "__main__":
    main()
