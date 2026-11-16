#!/usr/bin/env python3
"""Baseline classifiers for the ROG-Dialog sentiment task.

Runs three dummy strategies on every available eval set and writes a markdown + json
report that plugs directly into the finding paragraphs of report_sentiment_signal_hunt.html.
Every table row in the report should be read as "beat this".

Strategies (per class scheme, seeded):
  - uniform             : pick a class uniformly at random
  - stratified_train    : sample from the TRAIN-split label prior of each source
                          (standard scikit "stratified" dummy)
  - majority            : always predict the single most-frequent train class

Eval sets:
  - subset_170          : data/subset_174_seed42_v2.jsonl   (the OR text/audio eval set)
  - rog_test_2s         : full ROG test split with the 2.0 s duration filter applied
                          (matches the 434 rows in matrix predictions_rog_dia.jsonl)
  - rog_test_full       : full ROG test split, NO duration filter
                          (populated once the full-ROG rerun is in; present here so
                           the user can re-run this script after that train)

Metrics (per scheme):
  - macro_f1            : mean of per-class F1, unweighted
  - accuracy            : fraction correct
  - spearman_rho        : rank correlation of predicted ordinal vs gold ordinal
  - mae_ord             : mean |pred_ord - gold_ord|
  - r2_ord              : coefficient of determination on ordinal scalars
                          (so random -> ~0 by construction; majority -> negative
                           when the mean is not the majority class's ordinal)
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SUBSET_PATH = ROOT / "data" / "subset_174_seed42_v2.jsonl"
ROG_FULL_PATH = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
OUT_DIR = ROOT / "runs" / "baselines"

LABELS_6 = [
    "predominantlyNegative",
    "mixedNegative",
    "neutralNegative",
    "neutralPositive",
    "mixedPositive",
    "predominantlyPositive",
]
LABELS_3 = ["negative", "neutral", "positive"]

SENT_6_TO_3 = {
    "predominantlyNegative": "negative",
    "mixedNegative": "negative",
    "neutralNegative": "neutral",
    "neutralPositive": "neutral",
    "mixedPositive": "positive",
    "predominantlyPositive": "positive",
}
SENT_6_ORD = {lbl: i for i, lbl in enumerate(LABELS_6)}
SENT_3_ORD = {lbl: i for i, lbl in enumerate(LABELS_3)}

SEED = 42
MIN_DUR = 2.0
MAX_DUR = 20.0


# ----------------------------------------------------------------------------
# Loaders — each returns a list of {sentiment, sentiment_3} dicts
# ----------------------------------------------------------------------------

def load_subset170():
    rows = []
    with open(SUBSET_PATH) as f:
        for line in f:
            r = json.loads(line)
            rows.append({"sentiment": r["sentiment"], "sentiment_3": r["sentiment_3"]})
    return rows


def _iter_rog(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def load_rog_test(duration_filter):
    """duration_filter: '2s' or 'full'. Filters on split=='test' always."""
    rows = []
    for r in _iter_rog(ROG_FULL_PATH):
        if r.get("split") != "test":
            continue
        s = (r.get("labels") or {}).get("sentiment")
        if s is None:
            continue
        if duration_filter == "2s":
            dur = r["end_t"] - r["start_t"]
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
        s3 = SENT_6_TO_3[s]
        rows.append({"sentiment": s, "sentiment_3": s3})
    return rows


def rog_train_prior(duration_filter):
    """Return (prior_6cls, prior_3cls, majority_6, majority_3) computed on ROG train."""
    c6 = Counter()
    for r in _iter_rog(ROG_FULL_PATH):
        if r.get("split") != "train":
            continue
        s = (r.get("labels") or {}).get("sentiment")
        if s is None:
            continue
        if duration_filter == "2s":
            dur = r["end_t"] - r["start_t"]
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
        c6[s] += 1
    total = sum(c6.values())
    prior_6 = np.array([c6[l] / total for l in LABELS_6])
    c3 = Counter()
    for l, n in c6.items():
        c3[SENT_6_TO_3[l]] += n
    prior_3 = np.array([c3[l] / total for l in LABELS_3])
    maj_6 = LABELS_6[int(np.argmax(prior_6))]
    maj_3 = LABELS_3[int(np.argmax(prior_3))]
    return prior_6, prior_3, maj_6, maj_3


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def macro_f1(gold, pred, labels):
    f1s = []
    for lbl in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lbl and p == lbl)
        fp = sum(1 for g, p in zip(gold, pred) if g != lbl and p == lbl)
        fn = sum(1 for g, p in zip(gold, pred) if g == lbl and p != lbl)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def accuracy(gold, pred):
    return float(sum(1 for g, p in zip(gold, pred) if g == p) / len(gold))


def spearman_rho(gold_ord, pred_ord):
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(gold_ord, pred_ord)
        return float(r) if r == r else 0.0
    except ImportError:
        g = np.asarray(gold_ord, dtype=float)
        p = np.asarray(pred_ord, dtype=float)
        gr = g.argsort().argsort()
        pr = p.argsort().argsort()
        return float(np.corrcoef(gr, pr)[0, 1])


def mae_ord(gold_ord, pred_ord):
    return float(np.mean(np.abs(np.asarray(gold_ord) - np.asarray(pred_ord))))


def r2_ord(gold_ord, pred_ord):
    g = np.asarray(gold_ord, dtype=float)
    p = np.asarray(pred_ord, dtype=float)
    ss_res = float(np.sum((g - p) ** 2))
    ss_tot = float(np.sum((g - np.mean(g)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def score(gold6, pred6, gold3, pred3):
    g6o = [SENT_6_ORD[l] for l in gold6]
    p6o = [SENT_6_ORD[l] for l in pred6]
    return {
        "n": len(gold6),
        "macro_f1_6cls": macro_f1(gold6, pred6, LABELS_6),
        "macro_f1_3cls": macro_f1(gold3, pred3, LABELS_3),
        "accuracy_6cls": accuracy(gold6, pred6),
        "accuracy_3cls": accuracy(gold3, pred3),
        "spearman_rho_6cls": spearman_rho(g6o, p6o),
        "mae_ord_6cls": mae_ord(g6o, p6o),
        "r2_ord_6cls": r2_ord(g6o, p6o),
    }


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------

def run_uniform(gold_rows, seed):
    rng = random.Random(seed)
    g6 = [r["sentiment"] for r in gold_rows]
    g3 = [r["sentiment_3"] for r in gold_rows]
    p6 = [LABELS_6[rng.randrange(6)] for _ in gold_rows]
    p3 = [LABELS_3[rng.randrange(3)] for _ in gold_rows]
    return score(g6, p6, g3, p3)


def run_stratified(gold_rows, prior_6, prior_3, seed):
    rng = np.random.default_rng(seed)
    g6 = [r["sentiment"] for r in gold_rows]
    g3 = [r["sentiment_3"] for r in gold_rows]
    p6 = list(rng.choice(LABELS_6, size=len(gold_rows), p=prior_6))
    p3 = list(rng.choice(LABELS_3, size=len(gold_rows), p=prior_3))
    return score(g6, p6, g3, p3)


def run_majority(gold_rows, maj_6, maj_3):
    g6 = [r["sentiment"] for r in gold_rows]
    g3 = [r["sentiment_3"] for r in gold_rows]
    p6 = [maj_6] * len(gold_rows)
    p3 = [maj_3] * len(gold_rows)
    return score(g6, p6, g3, p3)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def fmt_metrics(m):
    return (f"F1_3 {m['macro_f1_3cls']:.3f} · F1_6 {m['macro_f1_6cls']:.3f} · "
            f"acc_3 {m['accuracy_3cls']:.3f} · acc_6 {m['accuracy_6cls']:.3f} · "
            f"ρ {m['spearman_rho_6cls']:+.3f} · MAE {m['mae_ord_6cls']:.3f} · "
            f"R² {m['r2_ord_6cls']:+.3f}")


def write_report(out_md, eval_set_name, n_rows, prior_src, maj_6, maj_3, results):
    lines = []
    lines.append(f"# Baselines — {eval_set_name}")
    lines.append(f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"n_rows: {n_rows}")
    lines.append(f"prior source: {prior_src} train split")
    lines.append(f"majority class: 6cls={maj_6}, 3cls={maj_3}")
    lines.append("")
    lines.append("| strategy | F1_3 | F1_6 | acc_3 | acc_6 | ρ_6 | MAE_6 | R²_6 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in results.items():
        lines.append(
            f"| {name} "
            f"| {m['macro_f1_3cls']:.3f} "
            f"| {m['macro_f1_6cls']:.3f} "
            f"| {m['accuracy_3cls']:.3f} "
            f"| {m['accuracy_6cls']:.3f} "
            f"| {m['spearman_rho_6cls']:+.3f} "
            f"| {m['mae_ord_6cls']:.3f} "
            f"| {m['r2_ord_6cls']:+.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--include-rog-full", action="store_true",
                    help="Also run on ROG test split WITHOUT the 2s duration filter. "
                         "Useful only after the user's full-ROG rerun has reused the same test set.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # --- 170-instance subset -------------------------------------------------
    subset = load_subset170()
    prior_6_2s, prior_3_2s, maj_6_2s, maj_3_2s = rog_train_prior("2s")
    r_subset = {
        "uniform": run_uniform(subset, args.seed),
        "stratified_train_2s": run_stratified(subset, prior_6_2s, prior_3_2s, args.seed),
        "majority_train_2s": run_majority(subset, maj_6_2s, maj_3_2s),
    }
    write_report(out / "baselines_subset170.md", "subset_174_seed42_v2 (170 rows)",
                 len(subset), "ROG (2s-filtered)", maj_6_2s, maj_3_2s, r_subset)
    all_results["subset_170"] = {"n": len(subset), "strategies": r_subset}

    # --- ROG test, 2s filter -------------------------------------------------
    rog_test_2s = load_rog_test("2s")
    r_rog_2s = {
        "uniform": run_uniform(rog_test_2s, args.seed),
        "stratified_train_2s": run_stratified(rog_test_2s, prior_6_2s, prior_3_2s, args.seed),
        "majority_train_2s": run_majority(rog_test_2s, maj_6_2s, maj_3_2s),
    }
    write_report(out / "baselines_rog_test_2s.md", "ROG test split, 2s duration filter",
                 len(rog_test_2s), "ROG (2s-filtered)", maj_6_2s, maj_3_2s, r_rog_2s)
    all_results["rog_test_2s"] = {"n": len(rog_test_2s), "strategies": r_rog_2s}

    # --- ROG test, full (no filter) -----------------------------------------
    if args.include_rog_full:
        prior_6_full, prior_3_full, maj_6_full, maj_3_full = rog_train_prior("full")
        rog_test_full = load_rog_test("full")
        r_rog_full = {
            "uniform": run_uniform(rog_test_full, args.seed),
            "stratified_train_full": run_stratified(rog_test_full, prior_6_full, prior_3_full, args.seed),
            "majority_train_full": run_majority(rog_test_full, maj_6_full, maj_3_full),
        }
        write_report(out / "baselines_rog_test_full.md", "ROG test split, NO duration filter",
                     len(rog_test_full), "ROG (full, unfiltered)",
                     maj_6_full, maj_3_full, r_rog_full)
        all_results["rog_test_full"] = {"n": len(rog_test_full), "strategies": r_rog_full}

    # Aggregated JSON for HTML consumption
    (out / "baselines.json").write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "eval_sets": all_results,
    }, indent=2, ensure_ascii=False))

    # Console summary
    print("\n=== Baselines ===")
    for es, body in all_results.items():
        print(f"\n[{es}]  n={body['n']}")
        for name, m in body["strategies"].items():
            print(f"  {name:<28} {fmt_metrics(m)}")


if __name__ == "__main__":
    main()
