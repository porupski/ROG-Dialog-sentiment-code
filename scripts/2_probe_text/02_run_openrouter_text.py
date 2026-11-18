#!/usr/bin/env python3
"""Run text-sentiment configs via OpenRouter on the fixed 174-instance subset.

Configs: {haiku, flash, gpt4o, gpt5, luna} x {0shot, fshot} x {6cls, 3cls}, all iso context.
Models:
  anthropic/claude-haiku-5.5   — cheap current-gen Anthropic.
  google/gemini-3.8-flash      — cheap current-gen Google; tops single-utterance so far.
  openai/gpt-4o                — expensive sliding OpenAI slug. Was pinned to the 2024-11-20
      snapshot to match ParlaSpeech 3.0 (Ljubešić et al., LREC 2026; GPT-4o parliamentary
      mean Spearman 0.807 vs ParlaSent 0.786). Unpinned now — the single-utterance ceiling
      question matters more than one published literature comparison. Record whatever OR
      resolves it to in the response for provenance.
  openai/gpt-5                 — medium-cost current OpenAI frontier.
  openai/gpt-5.6-luna          — cheap/light/latest OpenAI; direct comparator for Flash
      and Haiku at the cheap tier.

Prompts are identical across the two models for a given (shot_mode, class_scheme) pair.
For few-shot, the same 18 fixed examples (3/class from train) are prepended as user/assistant turns;
for 3-class runs, those examples are relabelled with their collapsed label — the example *texts*
and order don't change.

Every API call writes one JSONL line to runs/<run_id>/predictions.jsonl with the full raw response,
usage, cost, latency, and parsed label. A meta.json per run captures config + totals. A single line
per script invocation is appended to runs/_script_log.jsonl.

--test runs n_examples=3 per config (24 calls total), for cheap pipeline smoke-testing.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# DEFAULT_SUBSET is the 170-instance class-balanced OR subset. Pass --eval-set to
# point at any other JSONL in the same flat schema (e.g. data/subset_rog_test_full.jsonl
# for the full-ROG natural-distribution test, built by scripts/1_data_prep/prep_rog_test_full.py).
DEFAULT_SUBSET = ROOT / "data" / "subset_174_seed42_v2.jsonl"
FEWSHOT = ROOT / "data" / "fewshot_pool_18_seed42_v2.jsonl"
KEY_FILE = ROOT / "misc" / "key.txt"
RUNS_DIR = ROOT / "runs"
SCRIPT_LOG = RUNS_DIR / "_script_log.jsonl"

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

# =========================================================================
# EDIT HERE to change what gets run.
#
# MODELS: short key -> OpenRouter model slug. Add new entries to try more models.
# TASKS:  (model_key, shot_mode, class_scheme). Each tuple is one run.
# All tasks in this file use context_mode="iso"; ctx1 is a future script.
# CLI flags --models / --shots / --schemes further FILTER this list at runtime.
# =========================================================================
MODELS = {
    "haiku": "anthropic/claude-haiku-5.5",
    "flash": "google/gemini-3.8-flash",
    # All three OpenAI slugs below are SLIDING — OR routes to OpenAI's current default for
    # each tag and the resolved snapshot can drift between runs. Record whatever OR reports
    # in the response for provenance; don't rely on across-batch reproducibility.
    "gpt4o": "openai/gpt-4o",
    "gpt5":  "openai/gpt-5",
    "luna":  "openai/gpt-5.6-luna",
}

TASKS = [
    # (model_key, shot_mode, class_scheme)
    ("haiku", "0shot", "6cls"),
    ("haiku", "0shot", "3cls"),
    ("haiku", "fshot", "6cls"),
    ("haiku", "fshot", "3cls"),
    ("flash", "0shot", "6cls"),
    ("flash", "0shot", "3cls"),
    ("flash", "fshot", "6cls"),
    ("flash", "fshot", "3cls"),
    ("gpt4o", "0shot", "6cls"),
    ("gpt4o", "0shot", "3cls"),
    ("gpt4o", "fshot", "6cls"),
    ("gpt4o", "fshot", "3cls"),
    ("gpt5",  "0shot", "6cls"),
    ("gpt5",  "0shot", "3cls"),
    ("gpt5",  "fshot", "6cls"),
    ("gpt5",  "fshot", "3cls"),
    ("luna",  "0shot", "6cls"),
    ("luna",  "0shot", "3cls"),
    ("luna",  "fshot", "6cls"),
    ("luna",  "fshot", "3cls"),
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
    "predominantlyNegative": "negative",
    "mixedNegative": "negative",
    "neutralNegative": "neutral",
    "neutralPositive": "neutral",
    "mixedPositive": "positive",
    "predominantlyPositive": "positive",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_key():
    if not KEY_FILE.exists():
        raise SystemExit(f"missing {KEY_FILE}")
    key = KEY_FILE.read_text().strip()
    if not key or key.startswith("<") or key == "PLACEHOLDER":
        raise SystemExit(f"{KEY_FILE} is empty / placeholder; put the OpenRouter API key in it")
    return key


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f]


def system_prompt(class_scheme):
    if class_scheme == "6cls":
        return (
            "You classify Slovenian spoken-dialogue utterances by sentiment.\n"
            "Choose exactly one label from this list:\n"
            + "\n".join(f"- {lbl}" for lbl in LABELS_6)
            + "\n\nReply with only the label. No explanation."
        )
    return (
        "You classify Slovenian spoken-dialogue utterances by sentiment.\n"
        "Choose exactly one label from this list:\n"
        + "\n".join(f"- {lbl}" for lbl in LABELS_3)
        + "\n\nReply with only the label. No explanation."
    )


def label_for_scheme(inst, class_scheme):
    return inst["sentiment"] if class_scheme == "6cls" else inst["sentiment_3"]


def user_msg(text):
    return f"Utterance: {text}"


def build_messages(target_text, fewshot_examples, class_scheme):
    msgs = [{"role": "system", "content": system_prompt(class_scheme)}]
    for ex in fewshot_examples:
        msgs.append({"role": "user", "content": user_msg(ex["text"])})
        msgs.append({"role": "assistant", "content": label_for_scheme(ex, class_scheme)})
    msgs.append({"role": "user", "content": user_msg(target_text)})
    return msgs


def parse_label(output, class_scheme):
    """Three-tier parser:
    1. Exact match on cleaned first line (strictest).
    2. First label by position in the full text; if the text names two or more distinct
       labels, flag ambiguous (parse_ok=False). Catches model-echoing-few-shot output
       like "neutral or negative?".
    3. Nothing recognizable → parse_ok=False.
    """
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


def _backoff_sec(status, attempt, resp=None):
    """attempt is 0-indexed. Returns seconds to sleep before the NEXT attempt."""
    if status == 429:
        if resp is not None:
            ra = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
            if ra:
                try:
                    return max(float(ra), 1.0)
                except ValueError:
                    pass
        return min(10 * (2 ** attempt), 60.0)
    return min(2 * (2 ** attempt), 30.0)


def call_openrouter(key, model, messages, timeout=90, max_retries=6):
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ijs.si/rog-dialog-sentiment-probe",
        "X-Title": "rog-dialog-sentiment-probe",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        # Max tokens generous enough to cover any provider-mandated reasoning *plus* the
        # one-word label. Cost is bounded by actual usage, not this ceiling.
        "max_tokens": 500,
        "usage": {"include": True},
    }
    # Only Anthropic lets us skip extended thinking. Gemini 3.8-flash's endpoint on OR is
    # "reasoning mandatory" — sending reasoning=disabled there yields HTTP 400. For Gemini
    # we just give the model room to think and parse the final content normally.
    if model.startswith("anthropic/"):
        body["reasoning"] = {"enabled": False, "exclude": True, "max_tokens": 0}
    attempts = []
    for i in range(max_retries):
        t0 = time.time()
        try:
            resp = requests.post(OR_URL, headers=headers, json=body, timeout=timeout)
            latency_ms = int((time.time() - t0) * 1000)
            try:
                j = resp.json()
            except Exception as e:
                j = {"_parse_error": str(e), "_text": resp.text[:4000]}
            attempts.append({"attempt": i + 1, "status": resp.status_code, "latency_ms": latency_ms})
            if resp.status_code < 500 and resp.status_code != 429:
                return resp.status_code, j, latency_ms, attempts
            if i < max_retries - 1:
                time.sleep(_backoff_sec(resp.status_code, i, resp))
        except (requests.Timeout, requests.ConnectionError) as e:
            latency_ms = int((time.time() - t0) * 1000)
            attempts.append({"attempt": i + 1, "status": None, "latency_ms": latency_ms, "exception": repr(e)})
            if i < max_retries - 1:
                time.sleep(_backoff_sec(None, i))
        except Exception as e:
            latency_ms = int((time.time() - t0) * 1000)
            attempts.append({"attempt": i + 1, "status": None, "latency_ms": latency_ms, "exception": repr(e)})
            if i < max_retries - 1:
                time.sleep(_backoff_sec(None, i))
    return None, {"_exhausted": True, "_attempts": attempts}, 0, attempts


def extract_output_text(body):
    try:
        return body["choices"][0]["message"]["content"]
    except Exception:
        return None


def extract_cost(usage):
    if not isinstance(usage, dict):
        return None
    for key in ("cost", "total_cost"):
        if key in usage:
            return usage[key]
    return None


def run_one(key, cfg, subset, fewshot_pool, out_dir, subset_path, n_examples=None, dry_run=False):
    run_id = cfg["run_id"]
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "predictions.jsonl"
    meta_path = run_dir / "meta.json"

    fewshot_examples = fewshot_pool if cfg["shot_mode"] == "fshot" else []
    instances = subset[:n_examples] if n_examples else subset

    started = now_iso()
    t_start = time.time()
    total_cost = 0.0
    n_ok = 0
    n_err = 0
    n_parsed = 0

    with open(preds_path, "w") as fout:
        for i, inst in enumerate(instances):
            text = inst["text"]
            gold = label_for_scheme(inst, cfg["class_scheme"])
            messages = build_messages(text, fewshot_examples, cfg["class_scheme"])

            if dry_run:
                status, body, latency_ms, attempts = 200, {"_dry_run": True, "choices": [{"message": {"content": gold}}]}, 0, []
            else:
                status, body, latency_ms, attempts = call_openrouter(key, cfg["model"], messages)

            output_text = extract_output_text(body)
            parsed, parse_ok = parse_label(output_text, cfg["class_scheme"])
            usage = body.get("usage") if isinstance(body, dict) else None
            cost = extract_cost(usage)
            if cost:
                total_cost += cost

            err = None
            if status is None or not (200 <= status < 300):
                err = f"http_status={status}"
                n_err += 1
            else:
                n_ok += 1
                if parse_ok:
                    n_parsed += 1

            fout.write(json.dumps({
                "ts": now_iso(),
                "run_id": run_id,
                "idx": i,
                "instance_id": inst["instance_id"],
                "speaker": inst.get("speaker"),
                "split": inst.get("split"),
                "duration": inst.get("duration"),
                "text": text,
                "gold_sent6": inst["sentiment"],
                "gold_sent3": inst["sentiment_3"],
                "gold_for_scheme": gold,
                "model": cfg["model"],
                "model_name": cfg["model_name"],
                "class_scheme": cfg["class_scheme"],
                "shot_mode": cfg["shot_mode"],
                "context_mode": cfg["context_mode"],
                "request_messages": messages,
                "request_params": {"temperature": 0, "max_tokens": 30},
                "http_status": status,
                "response_raw": body,
                "output_text": output_text,
                "parsed_label": parsed,
                "parse_ok": parse_ok,
                "latency_ms": latency_ms,
                "attempts": attempts,
                "usage": usage,
                "cost_usd": cost,
                "error": err,
            }, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"  [{i+1:>3}/{len(instances)}] gold={gold!s:<24} out={str(output_text)[:40]!r:<42} parse_ok={parse_ok} status={status} cost={cost}")

    elapsed = time.time() - t_start
    meta = {
        "run_id": run_id,
        "config": cfg,
        "started_at": started,
        "ended_at": now_iso(),
        "elapsed_sec": round(elapsed, 2),
        "n_instances": len(instances),
        "n_http_ok": n_ok,
        "n_errors": n_err,
        "n_parsed_ok": n_parsed,
        "total_cost_usd": total_cost,
        "fewshot_pool_size": len(fewshot_examples),
        "subset_path": str(subset_path),
        "fewshot_path": str(FEWSHOT),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="n_examples=3 per run (24 calls total)")
    ap.add_argument("--n-examples", type=int, default=None, help="Cap instances per run (overrides --test cap if higher)")
    ap.add_argument("--models", nargs="*", default=None, choices=list(MODELS.keys()), help="Subset of models to run")
    ap.add_argument("--shots", nargs="*", default=None, choices=["0shot", "fshot"])
    ap.add_argument("--schemes", nargs="*", default=None, choices=["6cls", "3cls"])
    ap.add_argument("--eval-set", default=None,
                    help="Override the eval JSONL (default: data/subset_174_seed42_v2.jsonl). "
                         "Any file in the same flat subset_*_v2 schema is accepted — e.g. "
                         "data/subset_rog_test_full.jsonl for the full-ROG natural-distribution test.")
    ap.add_argument("--eval-tag", default=None,
                    help="Prefix the batch_id with this tag (e.g. FULLROG) so full-ROG batches "
                         "sort separately from 170-subset batches under runs/. "
                         "No effect on run_id or JSONL schema. Defaults to empty.")
    ap.add_argument("--out", default=str(RUNS_DIR))
    ap.add_argument("--dry-run", action="store_true", help="Do not hit OpenRouter; echo gold label as output")
    args = ap.parse_args()

    if args.test and args.n_examples is None:
        args.n_examples = 3

    key = "DRYRUN" if args.dry_run else read_key()
    subset_path = Path(args.eval_set).resolve() if args.eval_set else DEFAULT_SUBSET
    if not subset_path.exists():
        raise SystemExit(f"missing eval set: {subset_path}")
    subset = load_jsonl(subset_path)
    fewshot_pool = load_jsonl(FEWSHOT)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_filter = set(args.models) if args.models else None
    shots_filter = set(args.shots) if args.shots else None
    schemes_filter = set(args.schemes) if args.schemes else None

    # batch-id tag:
    #   --eval-tag FULLROG                   → batch_FULLROG_<stamp>
    #   --eval-tag FULLROG --test            → batch_FULLROG_TEST_<stamp>
    #   (default)                            → batch_FULL_<stamp>   (legacy)
    #   --test (default eval)                → batch_TEST_<stamp>   (legacy)
    pieces = []
    if args.eval_tag: pieces.append(args.eval_tag)
    if args.test: pieces.append("TEST")
    elif args.dry_run: pieces.append("DRY")
    elif not args.eval_tag: pieces.append("FULL")
    tag = "_".join(pieces)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_{tag}_{stamp}"
    batch_dir = out_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    skipped = []
    for mname, shot, scheme in TASKS:
        if mname not in MODELS:
            skipped.append((mname, shot, scheme, f"unknown model key: {mname}"))
            continue
        if models_filter and mname not in models_filter: continue
        if shots_filter and shot not in shots_filter: continue
        if schemes_filter and scheme not in schemes_filter: continue
        run_id = f"OR-text-{mname}-{shot}-iso-{scheme}"
        configs.append({
            "run_id": run_id,
            "batch_id": batch_id,
            "infra": "OpenRouter",
            "modality": "text",
            "model": MODELS[mname],
            "model_name": mname,
            "context_mode": "iso",
            "shot_mode": shot,
            "class_scheme": scheme,
        })
    if skipped:
        print(f"Skipped {len(skipped)} TASKS rows: {skipped}")

    print(f"Batch: {batch_id}")
    print(f"Running {len(configs)} configs; n_examples per run = {args.n_examples or len(subset)}")
    if args.dry_run:
        print("DRY RUN — no API calls.")

    script_start = now_iso()
    results = []
    try:
        for cfg in configs:
            print(f"\n=== RUN {cfg['run_id']} ===")
            meta = run_one(key, cfg, subset, fewshot_pool, batch_dir, subset_path,
                           n_examples=args.n_examples, dry_run=args.dry_run)
            results.append(meta)
            print(f"  -> parsed_ok={meta['n_parsed_ok']}/{meta['n_instances']} "
                  f"errors={meta['n_errors']} cost=${meta['total_cost_usd']:.5f} "
                  f"elapsed={meta['elapsed_sec']}s")
    finally:
        batch_meta = {
            "batch_id": batch_id,
            "ts_start": script_start,
            "ts_end": now_iso(),
            "argv": sys.argv,
            "test_mode": args.test,
            "dry_run": args.dry_run,
            "n_examples_per_run": args.n_examples,
            "n_subset": len(subset),
            "n_fewshot_pool": len(fewshot_pool),
            "subset_path": str(subset_path),
            "eval_tag": args.eval_tag,
            "fewshot_path": str(FEWSHOT),
            "models": {c["model_name"]: c["model"] for c in configs},
            "tasks_declared": [list(t) for t in TASKS],
            "runs": results,
            "total_cost_usd": sum(r.get("total_cost_usd", 0) or 0 for r in results),
        }
        (batch_dir / "batch_meta.json").write_text(json.dumps(batch_meta, indent=2, ensure_ascii=False))

        SCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SCRIPT_LOG, "a") as f:
            f.write(json.dumps({
                "script": "02_run_openrouter_text.py",
                "batch_id": batch_id,
                "ts_start": script_start,
                "ts_end": now_iso(),
                "argv": sys.argv,
                "test_mode": args.test,
                "dry_run": args.dry_run,
                "n_examples": args.n_examples,
                "configs_run": [r["run_id"] for r in results],
                "total_cost_usd": sum(r.get("total_cost_usd", 0) or 0 for r in results),
            }, ensure_ascii=False) + "\n")

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['run_id']}: parsed={r['n_parsed_ok']}/{r['n_instances']} "
              f"err={r['n_errors']} cost=${r['total_cost_usd']:.5f}")
    try:
        rel = batch_dir.relative_to(ROOT)
    except ValueError:
        rel = batch_dir
    print(f"\nBatch: {rel}")
    print(f"Analyze with: python3 scripts/analyze_probe.py {rel}")


if __name__ == "__main__":
    main()
