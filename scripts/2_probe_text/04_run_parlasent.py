#!/usr/bin/env python3
"""Run three local protocols against three XLM-R-family models on the 170-instance subset.

Models (coarse → fine on the sentiment-finetune axis):
  xlmrlarge  = FacebookAI/xlm-roberta-large   (MLM pretrain, no sentiment head)
  xlmrparla  = classla/xlm-r-parla            (XLM-R + Slovenian parliamentary MLM continue-pretrain)
  parlasent  = classla/xlm-r-parlasent        (xlmrparla + 6-class sentiment regression head)

Protocols:
  fillmask  — zero-shot via PLL over Slovenian label phrases at a <mask> position.
              Applies to xlmrlarge, xlmrparla (parlasent's MLM head is lost in the regression wrap).
  linprobe  — fit a frozen-backbone LogisticRegression head on the ROG-Dialog v2 TRAIN split
              (~2452 instances), predict on the 170-subset. Applies to all three models.
  reg       — parlasent's native regression output rounded to LABELS_6; 3cls via collapse.

Each (model, protocol, scheme) writes predictions.jsonl in the same schema as 02 so 03 Just Works.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SUBSET = ROOT / "data" / "subset_174_seed42_v2.jsonl"
FULL_DATA = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
RUNS_DIR = ROOT / "runs"
SCRIPT_LOG = RUNS_DIR / "_script_log.jsonl"

MIN_DUR, MAX_DUR = 2.0, 20.0

MODELS = {
    "xlmrlarge": "FacebookAI/xlm-roberta-large",
    "xlmrparla": "classla/xlm-r-parla",
    "parlasent": "classla/xlm-r-parlasent",
}

# (model_key, protocol, scheme)
TASKS = [
    ("xlmrlarge", "fillmask", "6cls"),
    ("xlmrlarge", "fillmask", "3cls"),
    ("xlmrlarge", "linprobe", "6cls"),
    ("xlmrlarge", "linprobe", "3cls"),
    ("xlmrparla", "fillmask", "6cls"),
    ("xlmrparla", "fillmask", "3cls"),
    ("xlmrparla", "linprobe", "6cls"),
    ("xlmrparla", "linprobe", "3cls"),
    ("parlasent", "linprobe", "6cls"),
    ("parlasent", "linprobe", "3cls"),
    ("parlasent", "reg",      "6cls"),
    ("parlasent", "reg",      "3cls"),
]

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
    "predominantlyNegative": "negative", "mixedNegative": "negative",
    "neutralNegative": "neutral",        "neutralPositive": "neutral",
    "mixedPositive": "positive",         "predominantlyPositive": "positive",
}

# Slovenian phrases used at the <mask> position for PLL-based fill-mask scoring.
# Pretežno / mešano / nevtralno keep valence-polarity readable to XLM-R without compound rarity.
FILLMASK_PHRASES_3 = {
    "negative": "negativno",
    "neutral":  "nevtralno",
    "positive": "pozitivno",
}
FILLMASK_PHRASES_6 = {
    "predominantlyNegative": "pretežno negativno",
    "mixedNegative":          "mešano negativno",
    "neutralNegative":        "nevtralno negativno",
    "neutralPositive":        "nevtralno pozitivno",
    "mixedPositive":          "mešano pozitivno",
    "predominantlyPositive":  "pretežno pozitivno",
}
FILLMASK_TEMPLATE = "Izjava: {text}\nČustvo je {LABEL}."


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_jsonl(p):
    return [json.loads(line) for line in open(p)]


def load_train_pool():
    """Replicate 01's filter (duration 2–20 s, non-null sentiment) on the TRAIN split only."""
    rows = load_jsonl(FULL_DATA)
    keep = []
    for r in rows:
        labels = r.get("labels", {})
        s = labels.get("sentiment")
        d = r["end_t"] - r["start_t"]
        if s is None or not (MIN_DUR <= d <= MAX_DUR):
            continue
        if r["split"] != "train":
            continue
        keep.append({
            "text": r["text"],
            "sentiment": s,
            "sentiment_3": SENT_6_TO_3[s],
        })
    return keep


def write_pred_record(fout, inst, scheme, pred, run_cfg, output_text, response_raw, latency_ms, parse_ok=True):
    gold = inst["sentiment"] if scheme == "6cls" else inst["sentiment_3"]
    fout.write(json.dumps({
        "ts": now_iso(),
        "run_id": run_cfg["run_id"],
        "instance_id": inst["instance_id"],
        "speaker": inst.get("speaker"),
        "split": inst.get("split"),
        "duration": inst.get("duration"),
        "text": inst["text"],
        "gold_sent6": inst["sentiment"],
        "gold_sent3": inst["sentiment_3"],
        "gold_for_scheme": gold,
        "model": run_cfg["model"],
        "model_name": run_cfg["model_name"],
        "class_scheme": scheme,
        "shot_mode": run_cfg["shot_mode"],
        "context_mode": run_cfg["context_mode"],
        "protocol": run_cfg["protocol"],
        "request_messages": None,
        "request_params": {"max_length": 512, "truncation": True},
        "http_status": 200,
        "response_raw": response_raw,
        "output_text": output_text,
        "parsed_label": pred,
        "parse_ok": parse_ok,
        "latency_ms": latency_ms,
        "attempts": [],
        "usage": None,
        "cost_usd": 0.0,
        "error": None,
    }, ensure_ascii=False) + "\n")


# =========================== Protocols ===========================

def _build_ids_for_pll(tok, text, label_phrase):
    """Return (full_ids, label_start, label_ids). Splice prefix / label / suffix tokenized separately
    so the exact label span is known regardless of SentencePiece re-tokenization at boundaries."""
    prefix = FILLMASK_TEMPLATE.split("{LABEL}")[0].format(text=text)
    suffix = FILLMASK_TEMPLATE.split("{LABEL}")[1]
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    # Prefix ends with a space, so the label starts a new SP word -> tokenize with leading space.
    label_ids = tok(" " + label_phrase, add_special_tokens=False)["input_ids"]
    suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
    sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    full_ids = [cls] + prefix_ids + label_ids + suffix_ids + [sep]
    if len(full_ids) > 512:
        # Chop from the FRONT of the prefix so the label region is preserved.
        over = len(full_ids) - 512
        prefix_ids = prefix_ids[over:]
        full_ids = [cls] + prefix_ids + label_ids + suffix_ids + [sep]
    label_start = 1 + len(prefix_ids)
    return full_ids, label_start, label_ids


def pll_fillmask(mlm, tok, text, label_phrase, device):
    """Length-normalized pseudo-log-likelihood. Mask each label token in turn; one batched forward."""
    import torch
    full_ids, start, label_ids = _build_ids_for_pll(tok, text, label_phrase)
    m = len(label_ids)
    if m == 0:
        return float("-inf")
    full_tensor = torch.tensor(full_ids, dtype=torch.long)
    masked_batch = full_tensor.unsqueeze(0).expand(m, -1).clone()
    for k in range(m):
        masked_batch[k, start + k] = tok.mask_token_id
    with torch.no_grad():
        logits = mlm(masked_batch.to(device)).logits  # [m, T, V]
    total = 0.0
    for k in range(m):
        logp = logits[k, start + k].log_softmax(-1)
        total += logp[label_ids[k]].item()
    return total / m


def run_fillmask_for_model(model_id, scheme, instances, device):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    print(f"    loading MLM: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    mlm = AutoModelForMaskedLM.from_pretrained(model_id).to(device).eval()

    phrases = FILLMASK_PHRASES_6 if scheme == "6cls" else FILLMASK_PHRASES_3
    labels_ordered = LABELS_6 if scheme == "6cls" else LABELS_3

    per_inst = []
    t_infer = time.time()
    for i, inst in enumerate(instances):
        t0 = time.time()
        scores = {lbl: pll_fillmask(mlm, tok, inst["text"], phrases[lbl], device) for lbl in labels_ordered}
        latency = int((time.time() - t0) * 1000)
        pred = max(scores, key=scores.get)
        per_inst.append((inst, pred, scores, latency))
        if (i + 1) % 20 == 0 or i == len(instances) - 1:
            gold = inst["sentiment"] if scheme == "6cls" else inst["sentiment_3"]
            print(f"      [{i+1:>3}/{len(instances)}] pred={pred} gold={gold}")
    del mlm, tok
    return per_inst, time.time() - t_infer


def mean_pool_embeddings(backbone, tok, texts, device, batch_size=8):
    import torch
    import numpy as np
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = backbone(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(pooled.cpu().numpy())
        done = min(i + batch_size, len(texts))
        if done % (batch_size * 20) == 0 or done == len(texts):
            print(f"        embedded {done}/{len(texts)}")
    return np.vstack(embs)


def run_linprobe_for_model(model_id, instances, train_pool, device):
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print(f"    loading backbone: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    # AutoModel drops task-specific heads; for parlasent we get the raw XLM-R encoder,
    # i.e. "what frozen features did sentiment fine-tuning leave in the backbone?"
    backbone = AutoModel.from_pretrained(model_id).to(device).eval()

    t0 = time.time()
    print(f"    embedding {len(train_pool)} train instances...")
    X_train = mean_pool_embeddings(backbone, tok, [r["text"] for r in train_pool], device)
    y_train_6 = np.array([r["sentiment"] for r in train_pool])
    y_train_3 = np.array([r["sentiment_3"] for r in train_pool])
    print(f"    embedding {len(instances)} eval instances...")
    X_test = mean_pool_embeddings(backbone, tok, [r["text"] for r in instances], device)
    t_embed = time.time() - t0
    del backbone, tok

    # Standardize features: XLM-R hidden states have per-dim scales spanning 2–3 orders
    # of magnitude. L-BFGS converges much faster on a well-conditioned Hessian, which
    # fixes the earlier non-convergence warning on parlasent-3cls. Fit on train, apply
    # the same transform to test — never fit on test.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    t1 = time.time()
    clf6 = LogisticRegression(max_iter=10000, class_weight="balanced").fit(X_train, y_train_6)
    clf3 = LogisticRegression(max_iter=10000, class_weight="balanced").fit(X_train, y_train_3)
    preds_6 = clf6.predict(X_test)
    preds_3 = clf3.predict(X_test)
    probs_6 = clf6.predict_proba(X_test)
    probs_3 = clf3.predict_proba(X_test)
    t_fit = time.time() - t1

    pi6 = [(inst, preds_6[i], dict(zip(clf6.classes_, probs_6[i].tolist()))) for i, inst in enumerate(instances)]
    pi3 = [(inst, preds_3[i], dict(zip(clf3.classes_, probs_3[i].tolist()))) for i, inst in enumerate(instances)]
    return pi6, pi3, {"embed_sec": round(t_embed, 1), "fit_sec": round(t_fit, 2)}


def run_regression_for_model(model_id, instances, device):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"    loading regression head: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
    per_inst = []
    t_infer = time.time()
    with torch.no_grad():
        for i, inst in enumerate(instances):
            t_i = time.time()
            enc = tok(inst["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
            score = mdl(**enc).logits.squeeze().item()
            idx = max(0, min(5, round(float(score))))
            pred_6 = LABELS_6[idx]
            latency = int((time.time() - t_i) * 1000)
            per_inst.append((inst, pred_6, float(score), latency))
            if (i + 1) % 50 == 0 or i == len(instances) - 1:
                print(f"      [{i+1:>3}/{len(instances)}] score={score:.3f} -> {pred_6}")
    del mdl, tok
    return per_inst, time.time() - t_infer


# =========================== Emit ===========================

def _make_cfg(batch_id, mkey, model_id, proto, scheme):
    run_id = f"LOCAL-text-{mkey}-{proto}-0shot-iso-{scheme}"
    return {
        "run_id": run_id,
        "batch_id": batch_id,
        "infra": "LocalHF",
        "modality": "text",
        "model": model_id,
        "model_name": mkey,
        "protocol": proto,
        "context_mode": "iso",
        "shot_mode": "0shot",
        "class_scheme": scheme,
    }


def _emit_fillmask(batch_dir, cfg, per_inst, scheme, elapsed_sec, started_at):
    run_dir = batch_dir / cfg["run_id"]; run_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(run_dir / "predictions.jsonl", "w") as fout:
        for inst, pred, scores, latency in per_inst:
            gold = inst["sentiment"] if scheme == "6cls" else inst["sentiment_3"]
            if pred == gold:
                n_ok += 1
            write_pred_record(fout, inst, scheme, pred, cfg,
                              output_text=pred,
                              response_raw={"pll_scores": {k: round(v, 4) for k, v in scores.items()}},
                              latency_ms=latency)
    _write_meta(run_dir, cfg, started_at, elapsed_sec, len(per_inst))
    print(f"    -> {cfg['run_id']}: n_correct={n_ok}/{len(per_inst)}")
    return n_ok


def _emit_linprobe(batch_dir, cfg, per_inst, scheme, timing, n_train, started_at):
    run_dir = batch_dir / cfg["run_id"]; run_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(run_dir / "predictions.jsonl", "w") as fout:
        for inst, pred, scores in per_inst:
            gold = inst["sentiment"] if scheme == "6cls" else inst["sentiment_3"]
            if pred == gold:
                n_ok += 1
            write_pred_record(fout, inst, scheme, pred, cfg,
                              output_text=pred,
                              response_raw={"probe_probs": {k: round(float(v), 4) for k, v in scores.items()}},
                              latency_ms=0)
    extra = {"n_train_pool": n_train, "embed_sec": timing["embed_sec"], "fit_sec": timing["fit_sec"]}
    _write_meta(run_dir, cfg, started_at, timing["embed_sec"] + timing["fit_sec"], len(per_inst), extra)
    print(f"    -> {cfg['run_id']}: n_correct={n_ok}/{len(per_inst)} (train={n_train})")
    return n_ok


def _emit_regression(batch_dir, cfg, per_inst, scheme, elapsed_sec, started_at):
    run_dir = batch_dir / cfg["run_id"]; run_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(run_dir / "predictions.jsonl", "w") as fout:
        for inst, pred_6, score, latency in per_inst:
            pred = pred_6 if scheme == "6cls" else SENT_6_TO_3[pred_6]
            gold = inst["sentiment"] if scheme == "6cls" else inst["sentiment_3"]
            if pred == gold:
                n_ok += 1
            write_pred_record(fout, inst, scheme, pred, cfg,
                              output_text=f"{score:.4f}",
                              response_raw={"regression_score": score},
                              latency_ms=latency)
    _write_meta(run_dir, cfg, started_at, elapsed_sec, len(per_inst))
    print(f"    -> {cfg['run_id']}: n_correct={n_ok}/{len(per_inst)}")
    return n_ok


def _write_meta(run_dir, cfg, started_at, elapsed_sec, n_inst, extra=None):
    meta = {
        "run_id": cfg["run_id"],
        "config": cfg,
        "started_at": started_at,
        "ended_at": now_iso(),
        "elapsed_sec": round(elapsed_sec, 2),
        "n_instances": n_inst,
        "n_http_ok": n_inst,
        "n_errors": 0,
        "n_parsed_ok": n_inst,
        "total_cost_usd": 0.0,
        "fewshot_pool_size": 0,
        "subset_path": str(SUBSET),
        "fewshot_path": None,
    }
    if extra:
        meta.update(extra)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


# =========================== Driver ===========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true",
                    help="n_examples=6 per run; also caps train pool for linprobe to 60")
    ap.add_argument("--n-examples", type=int, default=None, help="Cap eval instances per run")
    ap.add_argument("--n-train-cap", type=int, default=None, help="Cap train pool size for linprobe")
    ap.add_argument("--models", nargs="*", default=None, choices=list(MODELS.keys()))
    ap.add_argument("--protocols", nargs="*", default=None, choices=["fillmask", "linprobe", "reg"])
    ap.add_argument("--schemes", nargs="*", default=None, choices=["6cls", "3cls"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = ap.parse_args()

    if args.test and args.n_examples is None:
        args.n_examples = 6
    if args.test and args.n_train_cap is None:
        args.n_train_cap = 60

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as e:
        print(f"Missing deps ({e}). Install: pip install torch transformers scikit-learn", file=sys.stderr)
        sys.exit(1)

    subset = load_jsonl(SUBSET)
    instances = subset[: args.n_examples] if args.n_examples else subset

    models_filter = set(args.models) if args.models else None
    protos_filter = set(args.protocols) if args.protocols else None
    schemes_filter = set(args.schemes) if args.schemes else None

    tag = "TEST" if args.test else "FULL"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_LOCAL_{tag}_{stamp}"
    batch_dir = RUNS_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Group tasks by (model, protocol) to avoid reloading the backbone per scheme.
    grouped = {}
    for mkey, proto, scheme in TASKS:
        if models_filter and mkey not in models_filter: continue
        if protos_filter and proto not in protos_filter: continue
        if schemes_filter and scheme not in schemes_filter: continue
        grouped.setdefault((mkey, proto), []).append(scheme)

    if not grouped:
        print("No tasks selected; exiting.")
        return

    print(f"Batch: {batch_id}")
    print(f"Device: {args.device}  | eval_n={len(instances)}  | configs: {len(grouped)}")
    for (mkey, proto), schemes in grouped.items():
        print(f"  - {mkey}/{proto}: {schemes}")

    script_start = now_iso()
    results = []
    train_pool = None
    if any(proto == "linprobe" for (_, proto) in grouped):
        print("\nLoading ROG-Dialog v2 train pool for linprobe fitting...")
        train_pool = load_train_pool()
        if args.n_train_cap:
            train_pool = train_pool[: args.n_train_cap]
        print(f"  train pool: {len(train_pool)} instances")

    for (mkey, proto), schemes in grouped.items():
        model_id = MODELS[mkey]
        print(f"\n=== {mkey} ({model_id}) × {proto} === schemes={schemes}")

        if proto == "fillmask":
            for scheme in schemes:
                print(f"  fillmask / {scheme}")
                per_inst, elapsed = run_fillmask_for_model(model_id, scheme, instances, args.device)
                cfg = _make_cfg(batch_id, mkey, model_id, proto, scheme)
                n_ok = _emit_fillmask(batch_dir, cfg, per_inst, scheme, elapsed, script_start)
                results.append((cfg["run_id"], n_ok, len(instances), elapsed))

        elif proto == "linprobe":
            pi6, pi3, timing = run_linprobe_for_model(model_id, instances, train_pool, args.device)
            for scheme in schemes:
                cfg = _make_cfg(batch_id, mkey, model_id, proto, scheme)
                per_inst = pi6 if scheme == "6cls" else pi3
                n_ok = _emit_linprobe(batch_dir, cfg, per_inst, scheme, timing, len(train_pool), script_start)
                results.append((cfg["run_id"], n_ok, len(instances), timing["embed_sec"] + timing["fit_sec"]))

        elif proto == "reg":
            per_inst, elapsed = run_regression_for_model(model_id, instances, args.device)
            for scheme in schemes:
                cfg = _make_cfg(batch_id, mkey, model_id, proto, scheme)
                n_ok = _emit_regression(batch_dir, cfg, per_inst, scheme, elapsed, script_start)
                results.append((cfg["run_id"], n_ok, len(instances), elapsed))

    batch_meta = {
        "batch_id": batch_id,
        "ts_start": script_start,
        "ts_end": now_iso(),
        "argv": sys.argv,
        "test_mode": args.test,
        "n_examples_per_run": args.n_examples,
        "n_subset": len(subset),
        "subset_path": str(SUBSET),
        "full_data_path": str(FULL_DATA),
        "models": MODELS,
        "train_pool_size": len(train_pool) if train_pool else 0,
        "device": args.device,
        "runs": [{"run_id": r[0], "n_correct": r[1], "n": r[2], "elapsed_sec": round(r[3], 2)} for r in results],
        "total_cost_usd": 0.0,
    }
    (batch_dir / "batch_meta.json").write_text(json.dumps(batch_meta, indent=2, ensure_ascii=False))

    SCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SCRIPT_LOG, "a") as f:
        f.write(json.dumps({
            "script": "04_run_parlasent.py",
            "batch_id": batch_id,
            "ts_start": script_start,
            "ts_end": now_iso(),
            "argv": sys.argv,
            "test_mode": args.test,
            "configs_run": [r[0] for r in results],
            "device": args.device,
        }, ensure_ascii=False) + "\n")

    print(f"\nDone. Batch: {batch_dir.relative_to(ROOT)}")
    print(f"Analyze with: python3 scripts/analyze_results.py {batch_dir}")


if __name__ == "__main__":
    main()
