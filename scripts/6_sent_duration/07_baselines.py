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
import os
import random
import sys
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
    g3o = [SENT_3_ORD[l] for l in gold3]
    p3o = [SENT_3_ORD[l] for l in pred3]
    return {
        "n": len(gold6),
        "macro_f1_6cls": macro_f1(gold6, pred6, LABELS_6),
        "macro_f1_3cls": macro_f1(gold3, pred3, LABELS_3),
        "accuracy_6cls": accuracy(gold6, pred6),
        "accuracy_3cls": accuracy(gold3, pred3),
        "spearman_rho_6cls": spearman_rho(g6o, p6o),
        "spearman_rho_3cls": spearman_rho(g3o, p3o),
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
# Ceiling — ROG curator (labels.sentiment) vs ROG annotator (labels.sentiment_annotated).
# No model can beat this without overfitting to the curator's disagreements with the
# annotator. Treat it as the "beyond this is noise" line in the comparison tables.
# ----------------------------------------------------------------------------

def load_rog_iaa_rows(split_filter, duration_filter):
    """Return [{'sentiment' (curator), 'sentiment_annotated' (annotator), 'sentiment_3', ...}]
    for rows where BOTH labels are present. split_filter: 'all' | 'train' | 'dev' | 'test'."""
    rows = []
    for r in _iter_rog(ROG_FULL_PATH):
        if split_filter != "all" and r.get("split") != split_filter:
            continue
        labs = r.get("labels") or {}
        s = labs.get("sentiment")
        sa = labs.get("sentiment_annotated")
        if s is None or sa is None:
            continue
        if sa not in SENT_6_ORD or s not in SENT_6_ORD:
            continue
        if duration_filter == "2s":
            dur = r["end_t"] - r["start_t"]
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue
        rows.append({
            "sentiment":            s,
            "sentiment_annotated":  sa,
            "sentiment_3":          SENT_6_TO_3[s],
            "sentiment_annotated_3":SENT_6_TO_3[sa],
        })
    return rows


def run_iaa_ceiling(split_filter, duration_filter):
    """Use annotator labels as 'pred', curator labels as 'gold'. Produces the ceiling
    that any model will struggle to beat (training-set human noise)."""
    rows = load_rog_iaa_rows(split_filter, duration_filter)
    if not rows:
        return None
    g6 = [r["sentiment"] for r in rows]
    p6 = [r["sentiment_annotated"] for r in rows]
    g3 = [r["sentiment_3"] for r in rows]
    p3 = [r["sentiment_annotated_3"] for r in rows]
    return score(g6, p6, g3, p3)


# ----------------------------------------------------------------------------
# Stock ParlaSent zero-shot — no fine-tune, just the released classla/xlm-r-parlasent
# regression head over whatever eval set. torch + transformers must be in the env.
# ----------------------------------------------------------------------------

def _load_parlasent():
    """Returns (model, tokenizer, device).

    GPU-safety: honour the CUDA_VISIBLE_DEVICES env var set by the caller BEFORE torch
    was imported in this process. If CVD is unset or empty, force CPU — do NOT grab
    whatever GPU happens to be visible on a shared machine. Callers wanting GPU must
    pass --gpu N on the CLI, which sets CVD=N before this function imports torch.
    """
    import os
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None or cvd == "":
        # Explicit opt-out of CUDA. torch may still have been compiled with CUDA, but
        # setting CVD='' before the import prevents any device from being seen.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # Only use cuda if the caller explicitly reserved a specific device index (CVD set to
    # a non-empty value) AND torch sees at least one device after the CVD mask.
    use_cuda = (cvd not in (None, "")) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"  [parlasent-zs] device={device}  CUDA_VISIBLE_DEVICES={cvd!r}")
    name = "classla/xlm-r-parlasent"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    return model, tok, device


def _parlasent_predict(texts, model, tok, device, batch_size=32, max_length=256):
    """Return list of (pred_6cls_label, pred_3cls_label) for each input text.
    The ParlaSent head outputs a scalar on ~[0,5]; we round-and-clip to int 0..5."""
    import torch
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, truncation=True, padding=True,
                  max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1).detach().cpu().numpy()
        for v in logits:
            idx = int(max(0, min(5, round(float(v)))))
            s6 = LABELS_6[idx]
            s3 = SENT_6_TO_3[s6]
            out.append((s6, s3))
    return out


def run_parlasent_zs(rows_with_text):
    """rows_with_text: list of {'sentiment','sentiment_3','text'}. Returns a score dict or None."""
    try:
        model, tok, device = _load_parlasent()
    except Exception as e:
        print(f"  [parlasent-zs] skipped — {type(e).__name__}: {e}")
        return None
    texts = [r["text"] for r in rows_with_text]
    preds = _parlasent_predict(texts, model, tok, device)
    g6 = [r["sentiment"] for r in rows_with_text]
    g3 = [r["sentiment_3"] for r in rows_with_text]
    p6 = [p[0] for p in preds]
    p3 = [p[1] for p in preds]
    return score(g6, p6, g3, p3)


def load_subset170_with_text():
    rows = []
    with open(SUBSET_PATH) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "sentiment":   r["sentiment"],
                "sentiment_3": r["sentiment_3"],
                "text":        r["text"],
            })
    return rows


def load_rog_test_with_text(duration_filter):
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
        rows.append({
            "sentiment":   s,
            "sentiment_3": SENT_6_TO_3[s],
            "text":        r["text"],
        })
    return rows


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def fmt_metrics(m):
    return (f"F1_3 {m['macro_f1_3cls']:.3f} · F1_6 {m['macro_f1_6cls']:.3f} · "
            f"acc_3 {m['accuracy_3cls']:.3f} · acc_6 {m['accuracy_6cls']:.3f} · "
            f"ρ_3 {m['spearman_rho_3cls']:+.3f} · ρ_6 {m['spearman_rho_6cls']:+.3f} · "
            f"MAE {m['mae_ord_6cls']:.3f} · R² {m['r2_ord_6cls']:+.3f}")


def write_report(out_md, eval_set_name, n_rows, prior_src, maj_6, maj_3, results):
    lines = []
    lines.append(f"# Baselines — {eval_set_name}")
    lines.append(f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"n_rows: {n_rows}")
    lines.append(f"prior source: {prior_src} train split")
    lines.append(f"majority class: 6cls={maj_6}, 3cls={maj_3}")
    lines.append("")
    lines.append("| strategy | n | F1_3 | F1_6 | acc_3 | acc_6 | ρ_3 | ρ_6 | MAE_6 | R²_6 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m in results.items():
        if m is None:
            lines.append(f"| {name} | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} "
            f"| {m['n']} "
            f"| {m['macro_f1_3cls']:.3f} "
            f"| {m['macro_f1_6cls']:.3f} "
            f"| {m['accuracy_3cls']:.3f} "
            f"| {m['accuracy_6cls']:.3f} "
            f"| {m['spearman_rho_3cls']:+.3f} "
            f"| {m['spearman_rho_6cls']:+.3f} "
            f"| {m['mae_ord_6cls']:.3f} "
            f"| {m['r2_ord_6cls']:+.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def main():
    # GPU guard — sets CUDA_VISIBLE_DEVICES *before* any torch import happens in this
    # process. Parse argv first for --gpu, then set the env var. No --gpu flag → force CPU
    # by setting CVD="". This mirrors the strict guard used by scripts/4_finetune_text/run_matrix.py.
    import os
    # Lightweight pre-parse (don't use argparse yet — we need CVD set before anything else).
    _gpu_val = None
    _argv = sys.argv[1:]
    for i, a in enumerate(_argv):
        if a == "--gpu" and i + 1 < len(_argv):
            _gpu_val = _argv[i + 1]
            break
        if a.startswith("--gpu="):
            _gpu_val = a.split("=", 1)[1]
            break
    if _gpu_val is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_val)
    # NOTE: do not `import torch` anywhere at module scope. Only _load_parlasent() imports
    # torch, and that only happens if --parlasent-zs is passed.

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--include-rog-full", action="store_true",
                    help="Also run on ROG test split WITHOUT the 2s duration filter.")
    ap.add_argument("--parlasent-zs", action="store_true",
                    help="Also run stock classla/xlm-r-parlasent (no fine-tune) over each eval set. "
                         "Needs torch + transformers; slow on CPU for large sets.")
    ap.add_argument("--gpu", type=str, default=None,
                    help="Reserve a specific GPU index for --parlasent-zs (e.g. --gpu 2). "
                         "Omit to force CPU. CUDA_VISIBLE_DEVICES is set BEFORE torch import.")
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
    if args.parlasent_zs:
        print("[parlasent-zs] subset_170 ...")
        r_subset["parlasent_zero_shot"] = run_parlasent_zs(load_subset170_with_text())

    write_report(out / "baselines_subset170.md", "subset_174_seed42_v2 (170 rows)",
                 len(subset), "ROG (2s-filtered)", maj_6_2s, maj_3_2s, r_subset)
    all_results["subset_170"] = {"n": len(subset), "strategies": r_subset}

    # --- ROG test, 2s filter -------------------------------------------------
    rog_test_2s = load_rog_test("2s")
    r_rog_2s = {
        "uniform": run_uniform(rog_test_2s, args.seed),
        "stratified_train_2s": run_stratified(rog_test_2s, prior_6_2s, prior_3_2s, args.seed),
        "majority_train_2s": run_majority(rog_test_2s, maj_6_2s, maj_3_2s),
        # Ceiling computed on the test split only (what we actually evaluate against).
        "iaa_ceiling_test_2s": run_iaa_ceiling(split_filter="test", duration_filter="2s"),
    }
    if args.parlasent_zs:
        print("[parlasent-zs] rog_test_2s ...")
        r_rog_2s["parlasent_zero_shot"] = run_parlasent_zs(load_rog_test_with_text("2s"))

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
            "iaa_ceiling_test_full": run_iaa_ceiling(split_filter="test", duration_filter="full"),
        }
        if args.parlasent_zs:
            print("[parlasent-zs] rog_test_full ...")
            r_rog_full["parlasent_zero_shot"] = run_parlasent_zs(load_rog_test_with_text("full"))

        write_report(out / "baselines_rog_test_full.md", "ROG test split, NO duration filter",
                     len(rog_test_full), "ROG (full, unfiltered)",
                     maj_6_full, maj_3_full, r_rog_full)
        all_results["rog_test_full"] = {"n": len(rog_test_full), "strategies": r_rog_full}

    # --- ROG whole-corpus IAA (not an eval set per se; the structural ceiling) ---
    iaa_all = {
        "iaa_ceiling_rog_all_2s":   run_iaa_ceiling(split_filter="all", duration_filter="2s"),
        "iaa_ceiling_rog_all_full": run_iaa_ceiling(split_filter="all", duration_filter="full"),
    }
    write_report(out / "ceiling_iaa_rog.md", "IAA ceiling — ROG curator vs annotator",
                 0, "ROG", "n/a", "n/a", iaa_all)
    all_results["iaa_ceiling"] = {
        "n_2s":   (iaa_all["iaa_ceiling_rog_all_2s"] or {}).get("n", 0),
        "n_full": (iaa_all["iaa_ceiling_rog_all_full"] or {}).get("n", 0),
        "strategies": iaa_all,
    }

    # Aggregated JSON for HTML consumption
    (out / "baselines.json").write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "parlasent_zs": args.parlasent_zs,
        "eval_sets": all_results,
    }, indent=2, ensure_ascii=False))

    # Console summary
    print("\n=== Baselines ===")
    for es, body in all_results.items():
        print(f"\n[{es}]")
        for name, m in body["strategies"].items():
            if m is None:
                print(f"  {name:<30} (unavailable)")
            else:
                print(f"  {name:<30} {fmt_metrics(m)}")


if __name__ == "__main__":
    main()
