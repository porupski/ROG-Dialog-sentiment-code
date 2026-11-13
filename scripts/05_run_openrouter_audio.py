#!/usr/bin/env python3
"""Run audio-sentiment configs via OpenRouter on the fixed 170-instance subset.

Mirrors 02's logging schema and parse strategy; swaps the text input for a base64-encoded WAV
audio part alongside the same label-only instruction. Haiku is excluded: Anthropic models on
OR don't accept audio input today — Gemini-family only.

Default configs: {flash, carbon} × 0shot × {6cls, 3cls} = 4 runs × 170 calls.

Few-shot audio (uploading 18 reference WAVs per call) is gated by INCLUDE_FEWSHOT below; OFF
by default because each fshot call ships ~18× the audio bytes versus 0shot and audio tokens
are not cheap. Flip it to True (or pass --enable-fshot) once 0shot costs are known.

Safety rails:
  --test runs n_examples=3 per config.
  --dry-run NEVER touches OR; it fabricates a 200 response echoing the gold label and still
  exercises WAV discovery + base64 encoding so missing-file issues surface locally.
"""

import argparse
import base64
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SUBSET = ROOT / "data" / "subset_174_seed42_v2.jsonl"
FEWSHOT = ROOT / "data" / "fewshot_pool_18_seed42_v2.jsonl"
KEY_FILE = ROOT / "key.txt"
RUNS_DIR = ROOT / "runs"
SCRIPT_LOG = RUNS_DIR / "_script_log.jsonl"

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

# =========================================================================
# EDIT HERE to change what gets run.
# =========================================================================
MODELS = {
    "flash":  "google/gemini-3.8-flash",
    "carbon": "google/gemini-4-carbon",
}

# Toggle few-shot audio runs on/off. OFF by default: fshot multiplies request bytes ~19x
# and audio tokens are billed; we want 0shot cost observed first. CLI --enable-fshot flips.
INCLUDE_FEWSHOT = False

BASE_TASKS = [
    # (model_key, shot_mode, class_scheme)
    # ("flash",  "0shot", "6cls"),
    # ("flash",  "0shot", "3cls"),
    # ("carbon", "0shot", "6cls"),
    # ("carbon", "0shot", "3cls"),
]
# TODO: add a locally fine-tuned Slovenian-audio sentiment model as a LOCAL-audio-* run here
# (parallel to 04's LOCAL-text-* protocols). Candidates to try first: whisper-slovene +
# classifier head, or a ROG-Dialog-fine-tuned wav2vec2 variant. 03 already globs
# LOCAL-audio-* so dropping new runs in will Just Work.
FSHOT_TASKS = [
    ("flash",  "fshot", "6cls"),
    ("flash",  "fshot", "3cls"),
    ("carbon", "fshot", "6cls"),
    ("carbon", "fshot", "3cls"),
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
    body = (
        "You classify Slovenian spoken-dialogue utterances by sentiment. You will receive an "
        "audio clip of a short utterance. Listen to the audio (tone, prosody, emphasis, and the "
        "words themselves) and choose exactly one label from this list:\n"
    )
    labels = LABELS_6 if class_scheme == "6cls" else LABELS_3
    return body + "\n".join(f"- {lbl}" for lbl in labels) + "\n\nReply with only the label. No explanation."


def label_for_scheme(inst, class_scheme):
    return inst["sentiment"] if class_scheme == "6cls" else inst["sentiment_3"]


def wav_to_b64(wav_path):
    with open(wav_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii")


def audio_part(b64_data, fmt="wav"):
    # OpenRouter normalizes this to the provider-native audio part. Gemini models accept
    # inline_data audio via this content-type on OR's /v1/chat/completions.
    return {"type": "input_audio", "input_audio": {"data": b64_data, "format": fmt}}


def build_messages(target_wav_b64, few_shots_b64, class_scheme):
    """few_shots_b64: list of (b64, label_for_scheme) pairs, in order."""
    msgs = [{"role": "system", "content": system_prompt(class_scheme)}]
    for b64, lbl in few_shots_b64:
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": "Utterance:"},
            audio_part(b64),
        ]})
        msgs.append({"role": "assistant", "content": lbl})
    msgs.append({"role": "user", "content": [
        {"type": "text", "text": "Utterance:"},
        audio_part(target_wav_b64),
    ]})
    return msgs


def redact_messages_for_log(messages):
    """Strip base64 blobs from message log so predictions.jsonl stays small and greppable."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            new_c = []
            for part in c:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    data = part.get("input_audio", {}).get("data", "")
                    new_c.append({
                        "type": "input_audio",
                        "input_audio": {
                            "format": part["input_audio"].get("format", "wav"),
                            "data_len_b64": len(data),
                            "data": "<elided>",
                        },
                    })
                else:
                    new_c.append(part)
            out.append({"role": m["role"], "content": new_c})
        else:
            out.append(m)
    return out


def parse_label(output, class_scheme):
    """Same three-tier parser as 02_run_openrouter_text.py."""
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


def call_openrouter(key, model, messages, timeout=180, max_retries=6):
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
        # Same reasoning rationale as 02: give Gemini room for mandatory thinking.
        "max_tokens": 500,
        "usage": {"include": True},
    }
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


def resolve_audio_path(rel_path):
    """subset entries store paths as 'data/processed_audio/...'; resolve to absolute under ROOT."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return ROOT / p


def preload_fewshot_b64(fewshot_pool):
    pre = []
    for ex in fewshot_pool:
        wp = resolve_audio_path(ex["audio_path"])
        if not wp.exists():
            raise SystemExit(f"fewshot WAV missing: {wp}")
        pre.append({"inst": ex, "b64": wav_to_b64(wp)})
    return pre


def run_one(key, cfg, subset, fewshot_pre, out_dir, n_examples=None, dry_run=False):
    run_id = cfg["run_id"]
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "predictions.jsonl"
    meta_path = run_dir / "meta.json"

    use_fshot = cfg["shot_mode"] == "fshot"
    instances = subset[:n_examples] if n_examples else subset

    started = now_iso()
    t_start = time.time()
    total_cost = 0.0
    total_bytes_sent = 0
    n_ok = 0
    n_err = 0
    n_parsed = 0

    # Prepare fewshot content once per run (same across instances).
    few_shots_b64 = []
    if use_fshot:
        for pre in fewshot_pre:
            few_shots_b64.append((pre["b64"], label_for_scheme(pre["inst"], cfg["class_scheme"])))

    with open(preds_path, "w") as fout:
        for i, inst in enumerate(instances):
            wav_path = resolve_audio_path(inst["audio_path"])
            if not wav_path.exists():
                err = f"missing_wav:{wav_path}"
                n_err += 1
                fout.write(json.dumps({
                    "ts": now_iso(), "run_id": run_id, "idx": i,
                    "instance_id": inst["instance_id"], "error": err,
                    "class_scheme": cfg["class_scheme"], "parse_ok": False,
                    "parsed_label": None, "model": cfg["model"],
                }, ensure_ascii=False) + "\n")
                print(f"  [{i+1:>3}/{len(instances)}] MISSING WAV -> {wav_path}")
                continue

            target_b64 = wav_to_b64(wav_path)
            messages = build_messages(target_b64, few_shots_b64, cfg["class_scheme"])
            gold = label_for_scheme(inst, cfg["class_scheme"])

            if dry_run:
                # Simulate a 200 echoing the gold label; still measures WAV loading / b64 cost.
                status = 200
                body = {"_dry_run": True, "choices": [{"message": {"content": gold}}], "usage": {}}
                latency_ms = 0
                attempts = []
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

            # Rough bytes-sent telemetry: target + fewshot bytes.
            req_bytes = len(target_b64) + sum(len(b64) for b64, _ in few_shots_b64)
            total_bytes_sent += req_bytes

            fout.write(json.dumps({
                "ts": now_iso(),
                "run_id": run_id,
                "idx": i,
                "instance_id": inst["instance_id"],
                "speaker": inst.get("speaker"),
                "split": inst.get("split"),
                "duration": inst.get("duration"),
                "text": inst["text"],
                "audio_path": inst["audio_path"],
                "gold_sent6": inst["sentiment"],
                "gold_sent3": inst["sentiment_3"],
                "gold_for_scheme": gold,
                "model": cfg["model"],
                "model_name": cfg["model_name"],
                "class_scheme": cfg["class_scheme"],
                "shot_mode": cfg["shot_mode"],
                "context_mode": cfg["context_mode"],
                "modality": "audio",
                "request_messages": redact_messages_for_log(messages),
                "request_bytes_b64": req_bytes,
                "request_params": {"temperature": 0, "max_tokens": 500},
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

            print(f"  [{i+1:>3}/{len(instances)}] gold={gold!s:<24} out={str(output_text)[:40]!r:<42} "
                  f"parse_ok={parse_ok} status={status} cost={cost}")

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
        "fewshot_pool_size": len(few_shots_b64),
        "total_request_bytes_b64": total_bytes_sent,
        "subset_path": str(SUBSET),
        "fewshot_path": str(FEWSHOT) if use_fshot else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="n_examples=3 per run")
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--models", nargs="*", default=None, choices=list(MODELS.keys()))
    ap.add_argument("--shots", nargs="*", default=None, choices=["0shot", "fshot"])
    ap.add_argument("--schemes", nargs="*", default=None, choices=["6cls", "3cls"])
    ap.add_argument("--enable-fshot", action="store_true",
                    help="Include fshot tasks even though INCLUDE_FEWSHOT is False")
    ap.add_argument("--out", default=str(RUNS_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="Do NOT hit OpenRouter; still exercises WAV discovery + base64 encoding")
    args = ap.parse_args()

    if args.test and args.n_examples is None:
        args.n_examples = 3

    tasks = list(BASE_TASKS)
    if INCLUDE_FEWSHOT or args.enable_fshot:
        tasks += FSHOT_TASKS

    key = "DRYRUN" if args.dry_run else read_key()
    subset = load_jsonl(SUBSET)
    fewshot_pool = load_jsonl(FEWSHOT)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_filter = set(args.models) if args.models else None
    shots_filter = set(args.shots) if args.shots else None
    schemes_filter = set(args.schemes) if args.schemes else None

    tag = "TEST" if args.test else ("DRY" if args.dry_run else "FULL")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_AUDIO_{tag}_{stamp}"
    batch_dir = out_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    skipped = []
    for mname, shot, scheme in tasks:
        if mname not in MODELS:
            skipped.append((mname, shot, scheme, f"unknown model key: {mname}"))
            continue
        if models_filter and mname not in models_filter: continue
        if shots_filter and shot not in shots_filter: continue
        if schemes_filter and scheme not in schemes_filter: continue
        run_id = f"OR-audio-{mname}-{shot}-iso-{scheme}"
        configs.append({
            "run_id": run_id,
            "batch_id": batch_id,
            "infra": "OpenRouter",
            "modality": "audio",
            "model": MODELS[mname],
            "model_name": mname,
            "context_mode": "iso",
            "shot_mode": shot,
            "class_scheme": scheme,
        })
    if skipped:
        print(f"Skipped {len(skipped)} tasks: {skipped}")

    need_fshot = any(c["shot_mode"] == "fshot" for c in configs)
    print(f"Batch: {batch_id}")
    print(f"Running {len(configs)} configs; n_examples per run = {args.n_examples or len(subset)}")
    print(f"Fewshot audio enabled: {need_fshot}  (INCLUDE_FEWSHOT={INCLUDE_FEWSHOT}, --enable-fshot={args.enable_fshot})")
    if args.dry_run:
        print("DRY RUN — no API calls. WAV loading + b64 still exercised.")

    # Preload fewshot WAVs only if some config needs them.
    fewshot_pre = preload_fewshot_b64(fewshot_pool) if need_fshot else []

    script_start = now_iso()
    results = []
    try:
        for cfg in configs:
            print(f"\n=== RUN {cfg['run_id']} ===")
            meta = run_one(key, cfg, subset, fewshot_pre, batch_dir,
                           n_examples=args.n_examples, dry_run=args.dry_run)
            results.append(meta)
            print(f"  -> parsed_ok={meta['n_parsed_ok']}/{meta['n_instances']} "
                  f"errors={meta['n_errors']} cost=${meta['total_cost_usd']:.5f} "
                  f"bytes_b64={meta['total_request_bytes_b64']:,} elapsed={meta['elapsed_sec']}s")
    finally:
        batch_meta = {
            "batch_id": batch_id,
            "ts_start": script_start,
            "ts_end": now_iso(),
            "argv": sys.argv,
            "test_mode": args.test,
            "dry_run": args.dry_run,
            "include_fewshot": bool(need_fshot),
            "n_examples_per_run": args.n_examples,
            "n_subset": len(subset),
            "n_fewshot_pool": len(fewshot_pool),
            "subset_path": str(SUBSET),
            "fewshot_path": str(FEWSHOT),
            "models": {c["model_name"]: c["model"] for c in configs},
            "tasks_declared": [list(t) for t in tasks],
            "runs": results,
            "total_cost_usd": sum(r.get("total_cost_usd", 0) or 0 for r in results),
            "total_request_bytes_b64": sum(r.get("total_request_bytes_b64", 0) for r in results),
        }
        (batch_dir / "batch_meta.json").write_text(json.dumps(batch_meta, indent=2, ensure_ascii=False))

        SCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SCRIPT_LOG, "a") as f:
            f.write(json.dumps({
                "script": "05_run_openrouter_audio.py",
                "batch_id": batch_id,
                "ts_start": script_start,
                "ts_end": now_iso(),
                "argv": sys.argv,
                "test_mode": args.test,
                "dry_run": args.dry_run,
                "include_fewshot": bool(need_fshot),
                "n_examples": args.n_examples,
                "configs_run": [r["run_id"] for r in results],
                "total_cost_usd": sum(r.get("total_cost_usd", 0) or 0 for r in results),
            }, ensure_ascii=False) + "\n")

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['run_id']}: parsed={r['n_parsed_ok']}/{r['n_instances']} "
              f"err={r['n_errors']} cost=${r['total_cost_usd']:.5f} "
              f"bytes_b64={r['total_request_bytes_b64']:,}")


if __name__ == "__main__":
    main()
