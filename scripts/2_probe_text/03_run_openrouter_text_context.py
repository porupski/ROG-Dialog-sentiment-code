#!/usr/bin/env python3
"""Run text-sentiment configs via OpenRouter with a 3-turn natural-flow context window.

Sibling of 02_run_openrouter_text.py. Same subset, same MODELS, same parser, same
retry/backoff, same JSONL schema (plus window fields). The only experimental change is
the user message: instead of one isolated utterance, the model sees

    [SPK1] <prev utterance>
    [SPK2] <target utterance>  [TARGET]
    [SPK1] <next utterance>

and is told to classify ONLY the [TARGET] turn.

Window construction:
  - Each target in data/subset_174_seed42_v2.jsonl is joined by instance_id to the FULL
    ROG v2 corpus (data/rog_dialog_v2_instance.jsonl). A target missing from the corpus
    is a hard error.
  - Neighbours = previous / next utterance in the same file_id, ordered by
    (start_t, end_t, instance_id). Neighbours are taken from the full corpus, not the subset.
  - Speakers are collapsed per window: first-seen speaker is SPK1, next distinct is SPK2, ...
  - Boundary policy: a missing neighbour slot is rendered as a bare "[NONE]" line and the row
    gets boundary_flag in {"ok", "no_prev", "no_next", "no_both"}. Nothing is dropped.

Pass 1: {haiku, flash} x 0shot x {6cls, 3cls} = 4 runs x 170 = 680 calls.
The three OpenAI models and fshot are supported by the code; add rows to TASKS to run them.
(fshot renders each few-shot example as its own ctx3 window from the corpus, so prompts
are ~18x longer.)

--target-marker picks where [TARGET] goes (eol | after_spk | own_line). Use it with
--test --n-examples 5 to compare variants before fixing DEFAULT_TARGET_MARKER.

--test runs n_examples=3 per config (12 calls total). Works with --dry-run.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

SUBSET = ROOT / "data" / "subset_174_seed42_v2.jsonl"
FEWSHOT = ROOT / "data" / "fewshot_pool_18_seed42_v2.jsonl"
CORPUS = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
KEY_FILE = ROOT / "misc" / "key.txt"
RUNS_DIR = ROOT / "runs"
SCRIPT_LOG = RUNS_DIR / "_script_log.jsonl"

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

CONTEXT_MODE = "ctx3"
MAX_TOKENS = 500
TEMPERATURE = 0
NONE_TOKEN = "[NONE]"
TARGET_TOKEN = "[TARGET]"
TARGET_MARKERS = ["eol", "after_spk", "own_line"]
DEFAULT_TARGET_MARKER = "eol"   # provisional until the 5-call placement test is run

# =========================================================================
# EDIT HERE to change what gets run.
#
# MODELS: identical to 02_run_openrouter_text.py — keep them in sync.
# TASKS:  (model_key, shot_mode, class_scheme). Each tuple is one run.
# All tasks in this file use context_mode="ctx3".
# CLI flags --models / --shots / --schemes further FILTER this list at runtime.
# =========================================================================
MODELS = {
    "haiku": "anthropic/claude-haiku-5.5",
    "flash": "google/gemini-3.8-flash",
    # All three OpenAI slugs below are SLIDING — OR routes to OpenAI's current default for
    # each tag and the resolved snapshot can drift between runs. Record whatever OR reports
    # in the response for provenance; don't rely on across-batch reproducibility.
    "gpt4o": "openai/gpt-4o-2024-11-20", #"openai/gpt-4o",
    "gpt5":  "openai/gpt-5",
    "luna":  "openai/gpt-5.6-luna",
}

TASKS = [
    # (model_key, shot_mode, class_scheme)
    # Pass 1: the two models with single-utterance numbers to compare against, 0shot only.
    ("haiku", "0shot", "6cls"),
    ("haiku", "0shot", "3cls"),
    ("flash", "0shot", "6cls"),
    ("flash", "0shot", "3cls"),
    # Pass 2 (uncomment when pass 1 shows signal):
    # ("gpt4o", "0shot", "6cls"),
    # ("gpt4o", "0shot", "3cls"),
    # ("gpt5",  "0shot", "6cls"),
    # ("gpt5",  "0shot", "3cls"),
    # ("luna",  "0shot", "6cls"),
    # ("luna",  "0shot", "3cls"),
    # ("haiku", "fshot", "6cls"),
    # ("haiku", "fshot", "3cls"),
    # ("flash", "fshot", "6cls"),
    # ("flash", "fshot", "3cls"),
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
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- windows

class CorpusIndex:
    """file_id-ordered view of the full ROG corpus for neighbour lookup."""

    def __init__(self, rows):
        self.by_id = {}
        by_file = defaultdict(list)
        for r in rows:
            iid = r["instance_id"]
            if iid in self.by_id:
                raise SystemExit(f"duplicate instance_id in corpus: {iid}")
            self.by_id[iid] = r
            by_file[r["file_id"]].append(r)
        self.pos = {}
        self.by_file = {}
        for fid, lst in by_file.items():
            lst.sort(key=lambda r: (r["start_t"], r["end_t"], r["instance_id"]))
            self.by_file[fid] = lst
            for i, r in enumerate(lst):
                self.pos[r["instance_id"]] = i

    def neighbours(self, instance_id):
        if instance_id not in self.by_id:
            raise KeyError(instance_id)
        row = self.by_id[instance_id]
        lst = self.by_file[row["file_id"]]
        i = self.pos[instance_id]
        prev = lst[i - 1] if i > 0 else None
        nxt = lst[i + 1] if i < len(lst) - 1 else None
        return row, prev, nxt


def _dur(row):
    return row["end_t"] - row["start_t"]


def build_window(inst, corpus):
    """inst: a subset / few-shot row. Returns the window dict that goes into the JSONL row.

    Target text comes from `inst` (identical to what 02_ sent), neighbours from the corpus.
    """
    iid = inst["instance_id"]
    try:
        crow, prev, nxt = corpus.neighbours(iid)
    except KeyError:
        raise SystemExit(
            f"FATAL: target instance {iid} is not in {CORPUS}. The subset was built from this "
            f"corpus, so this means the wrong corpus file (e.g. a slim sample) or a changed "
            f"subset. Refusing to pad silently."
        )
    if crow["file_id"] != inst["file_id"]:
        raise SystemExit(f"FATAL: file_id mismatch for {iid}: subset={inst['file_id']} corpus={crow['file_id']}")

    flag = {(True, True): "ok", (False, True): "no_prev",
            (True, False): "no_next", (False, False): "no_both"}[(prev is not None, nxt is not None)]

    spk_map = {}
    window_speakers = []
    for r in (prev, crow, nxt):
        if r is None:
            window_speakers.append(None)
            continue
        spk_map.setdefault(r["speaker"], f"SPK{len(spk_map) + 1}")
        window_speakers.append(spk_map[r["speaker"]])

    return {
        "target_instance_id": iid,
        "prev_instance_id": prev["instance_id"] if prev else None,
        "next_instance_id": nxt["instance_id"] if nxt else None,
        "prev_text": prev["text"] if prev else None,
        "target_text": inst["text"],
        "next_text": nxt["text"] if nxt else None,
        "prev_speaker": prev["speaker"] if prev else None,
        "target_speaker": crow["speaker"],
        "next_speaker": nxt["speaker"] if nxt else None,
        "window_speakers": window_speakers,
        "boundary_flag": flag,
        "window_duration_sec": (round(_dur(prev) + _dur(crow) + _dur(nxt), 3) if flag == "ok" else None),
        "target_text_matches_corpus": inst["text"] == crow["text"],
    }


def render_window(win, target_marker):
    def line(spk, text, is_target):
        if spk is None:
            return NONE_TOKEN
        tok = f"[{spk}]"
        if not is_target:
            return f"{tok} {text}"
        if target_marker == "eol":
            return f"{tok} {text}  {TARGET_TOKEN}"
        if target_marker == "after_spk":
            return f"{tok} {TARGET_TOKEN} {text}"
        if target_marker == "own_line":
            return f"{TARGET_TOKEN}\n{tok} {text}"
        raise ValueError(target_marker)

    s = win["window_speakers"]
    return "\n".join([
        line(s[0], win["prev_text"], False),
        line(s[1], win["target_text"], True),
        line(s[2], win["next_text"], False),
    ])


# ---------------------------------------------------------------- prompts

def system_prompt(class_scheme):
    labels = LABELS_6 if class_scheme == "6cls" else LABELS_3
    return (
        "You classify Slovenian spoken-dialogue utterances by sentiment.\n"
        "You are shown three consecutive turns from one conversation "
        f"({NONE_TOKEN} = no turn in that position); classify ONLY the turn marked "
        f"{TARGET_TOKEN}, using the other turns as context.\n"
        "Choose exactly one label from this list:\n"
        + "\n".join(f"- {lbl}" for lbl in labels)
        + "\n\nReply with only the label. No explanation."
    )


def label_for_scheme(inst, class_scheme):
    return inst["sentiment"] if class_scheme == "6cls" else inst["sentiment_3"]


def user_msg(win, target_marker):
    return "Conversation:\n" + render_window(win, target_marker)


def build_messages(win, fewshot, class_scheme, target_marker):
    """fewshot: list of (example_row, example_window)."""
    msgs = [{"role": "system", "content": system_prompt(class_scheme)}]
    for ex, ex_win in fewshot:
        msgs.append({"role": "user", "content": user_msg(ex_win, target_marker)})
        msgs.append({"role": "assistant", "content": label_for_scheme(ex, class_scheme)})
    msgs.append({"role": "user", "content": user_msg(win, target_marker)})
    return msgs


def parse_label(output, class_scheme):
    """Identical to 02_: exact first-line match, else first label by position
    (ambiguous if >1 distinct label named), else fail."""
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


# ---------------------------------------------------------------- OpenRouter

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
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }
    # Only Anthropic lets us skip extended thinking; Gemini's OR endpoint is "reasoning
    # mandatory" and 400s on reasoning=disabled. See 02_.
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


# ---------------------------------------------------------------- run loop

def run_one(key, cfg, subset, windows, fewshot, out_dir, n_examples=None, dry_run=False):
    run_id = cfg["run_id"]
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "predictions.jsonl"
    meta_path = run_dir / "meta.json"

    fewshot_examples = fewshot if cfg["shot_mode"] == "fshot" else []
    instances = subset[:n_examples] if n_examples else subset

    started = now_iso()
    t_start = time.time()
    total_cost = 0.0
    n_ok = n_err = n_parsed = 0
    flags = Counter()

    with open(preds_path, "w") as fout:
        for i, inst in enumerate(instances):
            win = windows[inst["instance_id"]]
            flags[win["boundary_flag"]] += 1
            text = inst["text"]
            gold = label_for_scheme(inst, cfg["class_scheme"])
            messages = build_messages(win, fewshot_examples, cfg["class_scheme"], cfg["target_marker"])

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

            row = {
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
                "target_marker": cfg["target_marker"],
                # --- window fields (new vs 02_) ---
                "target_instance_id": win["target_instance_id"],
                "prev_instance_id": win["prev_instance_id"],
                "next_instance_id": win["next_instance_id"],
                "prev_text": win["prev_text"],
                "next_text": win["next_text"],
                "prev_speaker": win["prev_speaker"],
                "target_speaker": win["target_speaker"],
                "next_speaker": win["next_speaker"],
                "window_speakers": win["window_speakers"],
                "boundary_flag": win["boundary_flag"],
                "window_duration_sec": win["window_duration_sec"],
                # --- back to 02_ schema ---
                "request_messages": messages,
                "request_params": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
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
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            print(f"  [{i+1:>3}/{len(instances)}] gold={gold!s:<24} out={str(output_text)[:40]!r:<42} "
                  f"parse_ok={parse_ok} status={status} cost={cost} bnd={win['boundary_flag']}")

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
        "boundary_flags": dict(flags),
        "subset_path": str(SUBSET),
        "fewshot_path": str(FEWSHOT),
        "corpus_path": str(CORPUS),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="n_examples=3 per run (12 calls total)")
    ap.add_argument("--n-examples", type=int, default=None, help="Cap instances per run (overrides --test cap)")
    ap.add_argument("--models", nargs="*", default=None, choices=list(MODELS.keys()), help="Subset of models to run")
    ap.add_argument("--shots", nargs="*", default=None, choices=["0shot", "fshot"])
    ap.add_argument("--schemes", nargs="*", default=None, choices=["6cls", "3cls"])
    ap.add_argument("--target-marker", default=DEFAULT_TARGET_MARKER, choices=TARGET_MARKERS,
                    help="Where [TARGET] goes: end of target line / after speaker token / own line above")
    ap.add_argument("--out", default=str(RUNS_DIR))
    ap.add_argument("--dry-run", action="store_true", help="Do not hit OpenRouter; echo gold label as output")
    ap.add_argument("--show-prompt", action="store_true", help="Print the first instance's messages per run config and exit")
    args = ap.parse_args()

    if args.test and args.n_examples is None:
        args.n_examples = 3

    subset = load_jsonl(SUBSET)
    if not CORPUS.exists():
        raise SystemExit(f"missing {CORPUS}")
    corpus = CorpusIndex(load_jsonl(CORPUS))

    # Build ALL subset windows up front, so a missing target fails before any API spend,
    # and so the boundary distribution is reported for the full subset even under --test.
    windows = {inst["instance_id"]: build_window(inst, corpus) for inst in subset}
    boundary_dist = dict(Counter(w["boundary_flag"] for w in windows.values()))
    n_text_mismatch = sum(not w["target_text_matches_corpus"] for w in windows.values())
    n_same_spk = sum(1 for w in windows.values()
                     if w["boundary_flag"] == "ok" and len(set(w["window_speakers"])) == 1)
    print(f"Windows: {len(windows)} targets | boundary_flag: {boundary_dist} | "
          f"single-speaker windows: {n_same_spk}")
    if n_text_mismatch:
        print(f"WARNING: {n_text_mismatch} targets have subset text != corpus text "
              f"(prompt uses SUBSET text, as 02_ did). Check the corpus file version.")

    models_filter = set(args.models) if args.models else None
    shots_filter = set(args.shots) if args.shots else None
    schemes_filter = set(args.schemes) if args.schemes else None

    tag = "CTX-TEST" if args.test else ("CTX-DRY" if args.dry_run else "CTX")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"batch_{tag}_{stamp}"

    configs = []
    skipped = []
    for mname, shot, scheme in TASKS:
        if mname not in MODELS:
            skipped.append((mname, shot, scheme, f"unknown model key: {mname}"))
            continue
        if models_filter and mname not in models_filter: continue
        if shots_filter and shot not in shots_filter: continue
        if schemes_filter and scheme not in schemes_filter: continue
        configs.append({
            "run_id": f"OR-text-ctx-{mname}-{shot}-{scheme}",
            "batch_id": batch_id,
            "infra": "OpenRouter",
            "modality": "text",
            "model": MODELS[mname],
            "model_name": mname,
            "context_mode": CONTEXT_MODE,
            "shot_mode": shot,
            "class_scheme": scheme,
            "target_marker": args.target_marker,
        })
    if skipped:
        print(f"Skipped {len(skipped)} TASKS rows: {skipped}")

    # Few-shot windows only if some selected config needs them (pass 1 never does).
    fewshot_pool = load_jsonl(FEWSHOT)
    fewshot = []
    if any(c["shot_mode"] == "fshot" for c in configs):
        fewshot = [(ex, build_window(ex, corpus)) for ex in fewshot_pool]

    if args.show_prompt:
        for cfg in configs:
            msgs = build_messages(windows[subset[0]["instance_id"]],
                                  fewshot if cfg["shot_mode"] == "fshot" else [],
                                  cfg["class_scheme"], cfg["target_marker"])
            print(f"\n=== {cfg['run_id']} (marker={cfg['target_marker']}) ===")
            for m in msgs:
                print(f"--- {m['role']} ---\n{m['content']}")
        return

    key = "DRYRUN" if args.dry_run else read_key()
    out_dir = Path(args.out)
    batch_dir = out_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch: {batch_id}")
    print(f"Running {len(configs)} configs; n_examples per run = {args.n_examples or len(subset)}; "
          f"target_marker = {args.target_marker}")
    if args.dry_run:
        print("DRY RUN — no API calls.")

    script_start = now_iso()
    results = []
    try:
        for cfg in configs:
            print(f"\n=== RUN {cfg['run_id']} ===")
            meta = run_one(key, cfg, subset, windows, fewshot, batch_dir,
                           n_examples=args.n_examples, dry_run=args.dry_run)
            results.append(meta)
            print(f"  -> parsed_ok={meta['n_parsed_ok']}/{meta['n_instances']} "
                  f"errors={meta['n_errors']} cost=${meta['total_cost_usd']:.5f} "
                  f"elapsed={meta['elapsed_sec']}s")
    finally:
        total_cost = sum(r.get("total_cost_usd", 0) or 0 for r in results)
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
            "subset_path": str(SUBSET),
            "fewshot_path": str(FEWSHOT),
            "corpus_path": str(CORPUS),
            "context_mode": CONTEXT_MODE,
            "target_marker": args.target_marker,
            "boundary_flags_full_subset": boundary_dist,
            "n_single_speaker_windows": n_same_spk,
            "n_target_text_mismatch": n_text_mismatch,
            "models": {c["model_name"]: c["model"] for c in configs},
            "tasks_declared": [list(t) for t in TASKS],
            "runs": results,
            "total_cost_usd": total_cost,
        }
        (batch_dir / "batch_meta.json").write_text(json.dumps(batch_meta, indent=2, ensure_ascii=False))

        SCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SCRIPT_LOG, "a") as f:
            f.write(json.dumps({
                "script": "03_run_openrouter_text_context.py",
                "batch_id": batch_id,
                "ts_start": script_start,
                "ts_end": now_iso(),
                "argv": sys.argv,
                "test_mode": args.test,
                "dry_run": args.dry_run,
                "n_examples": args.n_examples,
                "target_marker": args.target_marker,
                "configs_run": [r["run_id"] for r in results],
                "total_cost_usd": total_cost,
            }, ensure_ascii=False) + "\n")

    print("\n=== summary ===")
    for r in results:
        print(f"  {r['run_id']}: parsed={r['n_parsed_ok']}/{r['n_instances']} "
              f"err={r['n_errors']} cost=${r['total_cost_usd']:.5f}")


if __name__ == "__main__":
    main()