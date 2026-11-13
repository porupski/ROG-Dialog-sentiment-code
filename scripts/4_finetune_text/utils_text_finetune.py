"""Fine-tuning utilities for 06_finetune_text.py.

Mirrors SSP's chapter-4 orchestration (Config dataclass, TARGETS/MIXTURES registries,
HF Trainer with best-epoch restore, z-score label normalizer, no-save-model pattern),
but swaps audio feature extraction for text tokenization and swaps multi-metric
classification for scalar regression with ordinal rounding at eval.

The three sources all land on the same ParlaSent 0..5 ordinal scale (ROG-Dialog via
`sentiment_6_ord`, ParlaSent via `sentiment_6_ord` emitted by prep_parlasent.py,
ParlaSpeech-SI via `sentiment.ParlaSent_logit` which empirically sits in [0, 5]
too — ParlaSent model output). A z-score normalizer (fit on train pool only) is
applied before loss; predictions are un-normalized then rounded + clipped to {0..5}
for 6cls metrics and collapsed to {neg, neu, pos} for 3cls metrics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

LABELS_6 = [
    "predominantlyNegative", "mixedNegative", "neutralNegative",
    "neutralPositive",       "mixedPositive", "predominantlyPositive",
]
LABELS_3 = ["negative", "neutral", "positive"]
SENT_6_TO_3_IDX = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
SENT_6_STR_TO_ORD = {l: i for i, l in enumerate(LABELS_6)}


# =========================================================================
# Source registry — one entry per raw-data JSONL.
# Each `reader` yields {"text": str, "label_scalar": float, "split": str, "instance_id": str}
# =========================================================================

def _reader_parlasent(row: dict) -> dict | None:
    if row.get("sentiment_6_ord") is None:
        return None
    return {
        "text": row["text"],
        "label_scalar": float(row["sentiment_6_ord"]),
        "split": row.get("split", "train"),
        "instance_id": row["instance_id"],
    }


def _reader_rog_dia_full(row: dict) -> dict | None:
    """ROG-Dialog v2 full JSONL — labels are nested under row['labels']['sentiment'] as 6-class
    strings; we compute the ordinal on the fly. Filter mirrors 01's 2–20 s duration."""
    labels = row.get("labels", {})
    s = labels.get("sentiment")
    if s is None:
        return None
    ord6 = SENT_6_STR_TO_ORD.get(s)
    if ord6 is None:
        return None
    dur = row.get("end_t", 0) - row.get("start_t", 0)
    if not (2.0 <= dur <= 20.0):
        return None
    return {
        "text": row["text"],
        "label_scalar": float(ord6),
        "split": row.get("split", "train"),
        "instance_id": row["instance_id"],
    }


def _reader_parlaspeech_si(row: dict) -> dict | None:
    """ParlaSpeech-SI `utterance_instance` (SSP 11c output). `sentiment.ParlaSent_logit`
    is the ParlaSent model's own regression output on the parliamentary audio
    transcript — sits approximately on the ParlaSent 0..5 ordinal scale
    (e.g. 3.162 → NeutralPositive). Used directly as our label."""
    sent = row.get("sentiment") or {}
    logit = sent.get("ParlaSent_logit")
    if logit is None:
        return None
    return {
        "text": row.get("text") or row.get("sentence") or "",
        "label_scalar": float(logit),
        "split": row.get("split", "train"),
        "instance_id": row.get("instance_id") or f"ParlaSpeech-SI_{row.get('file_id','?')}_{row.get('start_t','?')}",
    }


# Paths are relative to ROOT. Resolved by load_source().
SOURCES: dict[str, dict] = {
    "parlasent_sl": {
        "path": "data/parlasent_sl_instance.jsonl",
        "reader": _reader_parlasent,
        "family": "parliament_text",
        "modality": "text",
        "lang": "sl",
    },
    "parlaspeech_si": {
        "path": "data/parlaspeech_v4_si_utterance_instance.jsonl",
        "reader": _reader_parlaspeech_si,
        "family": "parliament_speech",
        "modality": "text_transcript",
        "lang": "sl",
    },
    "rog_dia": {
        "path": "data/rog_dialog_v2_instance.jsonl",
        "reader": _reader_rog_dia_full,
        "family": "dialog_speech",
        "modality": "text_transcript",
        "lang": "sl",
    },
}

# Mixtures: train on the union of listed sources. Eval is ALWAYS on all three test splits.
MIXTURES: dict[str, list[str]] = {
    "S":   ["parlasent_sl"],
    "P":   ["parlaspeech_si"],
    "R":   ["rog_dia"],
    "SP":  ["parlasent_sl", "parlaspeech_si"],
    "SR":  ["parlasent_sl", "rog_dia"],
    "PR":  ["parlaspeech_si", "rog_dia"],
    "SPR": ["parlasent_sl", "parlaspeech_si", "rog_dia"],
}

BACKBONES: list[str] = [
    "FacebookAI/xlm-roberta-large",
    "classla/xlm-r-parla",
    "classla/xlm-r-parlasent",
]


# =========================================================================
# Config
# =========================================================================

@dataclass
class Config:
    # Training
    backbone: str = "FacebookAI/xlm-roberta-large"
    mixture: str = "R"
    max_length: int = 128
    batch_size: int = 32
    grad_accum: int = 1
    learning_rate: float = 8e-6
    num_epochs: int = 4
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    # Mixing
    mix_mode: str = "concat"  # "concat" | "balanced" | "weighted"
    source_weights: dict = field(default_factory=dict)
    # Eval
    eval_sources: list = field(default_factory=lambda: ["parlasent_sl", "parlaspeech_si", "rog_dia"])
    best_metric: str = "spearman"  # dev metric for best-epoch pick
    # Caps (test/demo)
    cap_train: int | None = None
    cap_dev: int | None = None
    cap_test: int | None = None
    # Runtime
    seed: int = 42
    out_dir: str = "runs/finetune"
    run_tag: str | None = None  # defaults to <backbone_short>_<mixture>
    dataloader_num_workers: int = 2
    # Device (set by runner, not CLI)
    use_gpu: bool = False


def backbone_short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


# =========================================================================
# Data loading
# =========================================================================

def _load_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def source_available(src_name: str, root: Path) -> bool:
    return (root / SOURCES[src_name]["path"]).exists()


def missing_sources(src_names: list[str], root: Path) -> list[str]:
    return [s for s in src_names if not source_available(s, root)]


def load_source(src_name: str, root: Path) -> list[dict]:
    """Return instances for a source. If the backing JSONL is missing, warn and return [].
    Callers treat empty as 'drop this source' so a missing local dataset degrades the
    mixture gracefully instead of aborting the whole run."""
    meta = SOURCES[src_name]
    p = root / meta["path"]
    if not p.exists():
        print(f"  [warn] missing source JSONL for '{src_name}': {p} — skipping.")
        return []
    out = []
    for row in _load_jsonl(p):
        rec = meta["reader"](row)
        if rec is None:
            continue
        rec["source"] = src_name
        out.append(rec)
    return out


def split_rows(rows: list[dict], split: str) -> list[dict]:
    """ParlaSent uses train/dev/test; ParlaSpeech may use train/test only; ROG uses train/dev/test.
    For missing 'dev', the caller carves a slice from train."""
    return [r for r in rows if r.get("split") == split]


def mix_train(cfg: Config, root: Path, rng_seed: int = 42):
    """Return (train_rows, dev_rows). Dev = each source's dev split; if any source has no
    dev, carve 10 % of its own train. Sources whose JSONL is missing contribute nothing;
    if ALL sources in the mixture are missing, raises SystemExit so the runner can
    record the cell as skipped."""
    import random
    rng = random.Random(rng_seed)
    train_all, dev_all = [], []
    contributed = []
    for src in MIXTURES[cfg.mixture]:
        rows = load_source(src, root)
        if not rows:
            continue
        contributed.append(src)
        t = split_rows(rows, "train")
        d = split_rows(rows, "dev")
        if not d:
            rng.shuffle(t)
            n_dev = max(1, len(t) // 10)
            d, t = t[:n_dev], t[n_dev:]
        train_all.extend(t)
        dev_all.extend(d)
        print(f"  [{src}] train={len(t):>6} dev={len(d):>5}")
    if not contributed:
        raise SystemExit(
            f"Mixture '{cfg.mixture}' has no available sources ({MIXTURES[cfg.mixture]}). "
            f"Run the matching prep script(s) or pick a different mixture."
        )
    if cfg.mix_mode == "balanced":
        # Downsample each source to the size of the smallest contributing train split.
        by_src = {}
        for r in train_all:
            by_src.setdefault(r["source"], []).append(r)
        min_n = min(len(v) for v in by_src.values())
        bal = []
        for v in by_src.values():
            rng.shuffle(v)
            bal.extend(v[:min_n])
        print(f"  [balanced] each source downsampled to {min_n}; train={len(bal)}")
        train_all = bal
    rng.shuffle(train_all)
    if cfg.cap_train is not None:
        train_all = train_all[:cfg.cap_train]
    if cfg.cap_dev is not None:
        dev_all = dev_all[:cfg.cap_dev]
    return train_all, dev_all


def load_test_sets(cfg: Config, root: Path) -> dict[str, list[dict]]:
    """Missing eval sources (JSONL absent) are silently dropped from the returned dict so
    the engine only evaluates on sources that are actually present."""
    out = {}
    for src in cfg.eval_sources:
        if not source_available(src, root):
            print(f"  [test:{src}] unavailable — skipping eval on this source.")
            continue
        rows = load_source(src, root)
        test = split_rows(rows, "test")
        if cfg.cap_test is not None:
            test = test[:cfg.cap_test]
        out[src] = test
        print(f"  [test:{src}] n={len(test)}")
    return out


# =========================================================================
# Label normalization — z-score fit on train pool only.
# =========================================================================

class LabelNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, labels: list[float]):
        import statistics
        if len(labels) < 2:
            return self
        self.mean = statistics.fmean(labels)
        self.std = statistics.pstdev(labels) or 1.0
        return self

    def norm(self, y: float) -> float:
        return (y - self.mean) / self.std

    def denorm(self, y: float) -> float:
        return y * self.std + self.mean


# =========================================================================
# HF dataset build + tokenize
# =========================================================================

def build_hf_datasets(rows_train, rows_dev, rows_test_by_src, tokenizer, normalizer, max_length):
    from datasets import Dataset

    def enc(batch):
        enc_out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        enc_out["labels"] = [normalizer.norm(y) for y in batch["label_scalar"]]
        return enc_out

    def to_ds(rows):
        return Dataset.from_list(rows).map(enc, batched=True,
                                           remove_columns=[c for c in rows[0].keys() if c not in ("instance_id", "source", "label_scalar")])
    train_ds = to_ds(rows_train)
    dev_ds   = to_ds(rows_dev)
    test_ds_by_src = {k: to_ds(v) for k, v in rows_test_by_src.items() if v}
    return train_ds, dev_ds, test_ds_by_src


# =========================================================================
# Metrics — on denormalized predictions. 6cls = round+clip, 3cls = collapse.
# =========================================================================

def compute_metrics_from_preds(y_gold_raw: list[float], y_pred_raw: list[float]) -> dict:
    import math
    n = len(y_gold_raw)
    if n == 0:
        return {}
    mae = sum(abs(a - b) for a, b in zip(y_gold_raw, y_pred_raw)) / n
    mean_g = sum(y_gold_raw) / n
    ss_tot = sum((g - mean_g) ** 2 for g in y_gold_raw) or 1e-9
    ss_res = sum((g - p) ** 2 for g, p in zip(y_gold_raw, y_pred_raw))
    r2 = 1.0 - ss_res / ss_tot
    try:
        from scipy.stats import spearmanr
        sp = spearmanr(y_gold_raw, y_pred_raw).correlation
        if sp != sp:
            sp = None
    except ImportError:
        sp = None

    def round6(y):
        return max(0, min(5, round(y)))

    g6 = [round6(y) for y in y_gold_raw]
    p6 = [round6(y) for y in y_pred_raw]
    g3 = [SENT_6_TO_3_IDX[y] for y in g6]
    p3 = [SENT_6_TO_3_IDX[y] for y in p6]
    acc6 = sum(a == b for a, b in zip(g6, p6)) / n
    acc3 = sum(a == b for a, b in zip(g3, p3)) / n

    def per_class_prf(y_true, y_pred, k):
        """Returns (precisions, recalls, f1s) each of length k."""
        precs, recs, f1s = [], [], []
        for c in range(k):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            precs.append(prec)
            recs.append(rec)
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return precs, recs, f1s

    prec6, rec6, f1_6 = per_class_prf(g6, p6, 6)
    prec3, rec3, f1_3 = per_class_prf(g3, p3, 3)

    return {
        "n": n,
        "mae": mae, "r2": r2, "spearman": sp,
        "acc_6cls": acc6, "macro_f1_6cls": sum(f1_6) / 6,
        "acc_3cls": acc3, "macro_f1_3cls": sum(f1_3) / 3,
        # Per-class breakdown — important for the ParlaSent minority-positive calibration
        # check flagged in the lit review (Mochtak et al. 2023: only 7% positive in SL).
        "per_class_recall_6cls": rec6,
        "per_class_recall_3cls": rec3,
        "per_class_precision_6cls": prec6,
        "per_class_precision_3cls": prec3,
        "per_class_f1_6cls": f1_6,
        "per_class_f1_3cls": f1_3,
        # Convenience alias: recall for the 3-class positive bucket (index 2).
        "minority_positive_recall": rec3[2] if len(rec3) == 3 else None,
    }


# =========================================================================
# Trainer wrapper: MSE loss, scalar head.
# =========================================================================

def build_model_tokenizer(backbone: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(backbone)
    # Regression head = 1-logit classifier with MSELoss (HF convention).
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone, num_labels=1, problem_type="regression",
    )
    return model, tok


def run_training(cfg: Config, root: Path, out_dir: Path):
    """End-to-end: load data, fit model, best-epoch restore, predict on each eval source,
    emit predictions + metrics to out_dir. Returns the collated summary dict."""
    import os, random
    import numpy as np
    import torch
    from transformers import (
        Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tag = cfg.run_tag or f"{backbone_short(cfg.backbone)}__{cfg.mixture}"
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== RUN {tag} ===")
    print(f"  backbone = {cfg.backbone}")
    print(f"  mixture  = {cfg.mixture} ({MIXTURES[cfg.mixture]})")
    print(f"  lr={cfg.learning_rate}  batch={cfg.batch_size}x{cfg.grad_accum}  epochs={cfg.num_epochs}")

    print("  loading train + dev...")
    train_rows, dev_rows = mix_train(cfg, root)
    print(f"  total train={len(train_rows)}  dev={len(dev_rows)}")

    print("  loading test sets...")
    test_rows_by_src = load_test_sets(cfg, root)

    normalizer = LabelNormalizer().fit([r["label_scalar"] for r in train_rows])
    print(f"  label norm: mean={normalizer.mean:.3f} std={normalizer.std:.3f}")

    model, tok = build_model_tokenizer(cfg.backbone)
    train_ds, dev_ds, test_ds_by_src = build_hf_datasets(
        train_rows, dev_rows, test_rows_by_src, tok, normalizer, cfg.max_length
    )

    def compute_metrics_hf(eval_pred):
        preds = eval_pred.predictions.reshape(-1).tolist()
        labels = eval_pred.label_ids.reshape(-1).tolist()
        preds_raw = [normalizer.denorm(p) for p in preds]
        labels_raw = [normalizer.denorm(l) for l in labels]
        return compute_metrics_from_preds(labels_raw, preds_raw)

    args = TrainingArguments(
        output_dir=str(run_dir / "_hf"),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,                         # the "best epoch restore" guard
        metric_for_best_model=f"eval_{cfg.best_metric}",
        greater_is_better=True,
        report_to=[],                                        # no wandb
        logging_steps=50,
        dataloader_num_workers=cfg.dataloader_num_workers,
        save_total_limit=1,                                  # keep only the best; discard after predict
        remove_unused_columns=True,
        seed=cfg.seed,
    )
    # `tokenizer=` was deprecated in transformers 4.46+ in favour of `processing_class=`.
    # We try the new keyword first and fall back for older installs.
    import inspect
    trainer_kwargs = dict(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics_hf,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = Trainer(**trainer_kwargs)

    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    print(f"  training done in {train_sec:.1f}s")

    # Per-test-source predictions
    per_src_metrics = {}
    for src_name, ds in test_ds_by_src.items():
        print(f"  predicting on test:{src_name}...")
        preds_obj = trainer.predict(ds)
        preds_norm = preds_obj.predictions.reshape(-1).tolist()
        labels_norm = preds_obj.label_ids.reshape(-1).tolist()
        preds_raw = [normalizer.denorm(p) for p in preds_norm]
        labels_raw = [normalizer.denorm(l) for l in labels_norm]
        metrics = compute_metrics_from_preds(labels_raw, preds_raw)
        per_src_metrics[src_name] = metrics

        # Emit per-instance predictions
        src_rows = test_rows_by_src[src_name]
        pred_path = run_dir / f"predictions_{src_name}.jsonl"
        with open(pred_path, "w") as f:
            for row, p_raw in zip(src_rows, preds_raw):
                def round6(y): return max(0, min(5, round(y)))
                g6 = round6(row["label_scalar"]); p6 = round6(p_raw)
                g3 = SENT_6_TO_3_IDX[g6]; p3 = SENT_6_TO_3_IDX[p6]
                f.write(json.dumps({
                    "instance_id": row["instance_id"],
                    "source": src_name,
                    "backbone": cfg.backbone,
                    "mixture": cfg.mixture,
                    "text": row["text"],
                    "gold_scalar": row["label_scalar"],
                    "pred_scalar": p_raw,
                    "gold_6cls": LABELS_6[g6], "pred_6cls": LABELS_6[p6],
                    "gold_3cls": LABELS_3[g3], "pred_3cls": LABELS_3[p3],
                }, ensure_ascii=False) + "\n")
        rho_s = "n/a" if metrics["spearman"] is None else f"{metrics['spearman']:.3f}"
        rec3 = metrics["per_class_recall_3cls"]
        print(f"    -> {pred_path.relative_to(root)}: "
              f"R²={metrics['r2']:.3f} ρ={rho_s} MAE={metrics['mae']:.3f} "
              f"F1_6cls={metrics['macro_f1_6cls']:.3f} F1_3cls={metrics['macro_f1_3cls']:.3f} "
              f"| rec_3cls neg/neu/pos = {rec3[0]:.2f}/{rec3[1]:.2f}/{rec3[2]:.2f}")

    summary = {
        "run_tag": tag,
        "config": asdict(cfg),
        "mixture_sources": MIXTURES[cfg.mixture],
        "train_n": len(train_rows),
        "dev_n": len(dev_rows),
        "test_n_by_src": {k: len(v) for k, v in test_rows_by_src.items()},
        "label_norm": {"mean": normalizer.mean, "std": normalizer.std},
        "train_sec": round(train_sec, 1),
        "metrics_by_test": per_src_metrics,
        "state": trainer.state.log_history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Clean up HF checkpoint dir — we don't want to keep model weights for 21 runs.
    import shutil
    hf_dir = run_dir / "_hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)

    return summary
