#!/usr/bin/env python3
"""Run text-sentiment configs via OpenRouter with a natural-flow context window.

Sibling of 02_run_openrouter_text.py. Same subset, same MODELS, same parser, same
retry/backoff, same JSONL schema (plus window fields). The experimental change is
the user message: instead of one isolated utterance, the model sees

    [SPK1] <prev utterance>
    [SPK1] <prev utterance>
    [SPK2] <target utterance>  [TARGET]
    [SPK1] <next utterance>
    [SPK2] <next utterance>

and is told to classify ONLY the [TARGET] turn.

Window construction — WORD-THRESHOLD ACCUMULATOR:
  - Each target in data/subset_174_seed42_v2.jsonl is joined by instance_id to the FULL
    ROG v2 corpus (data/rog_dialog_v2_instance.jsonl). A target missing from the corpus
    is a hard error.
  - For each side (prev / next), accumulate utterances in the same file_id, ordered by
    (start_t, end_t, instance_id), stepping outward from the target, until the cumulative
    word count reaches --context-words. The side closes at the utterance that CROSSES the
    threshold (so a short-utterance file gets a couple extra words, a long-utterance file
    gets one neighbour and done).
  - --context-words 0 reproduces the LEGACY ±1-utterance behaviour (the 2026-11-17 committed
    batch batch_CTX_20261117_124601 was W=0), so the first shipped batch is reproducible.
  - Speakers collapse per window: chronological first-seen → SPK1, next distinct → SPK2, ...
  - Boundary policy: a side with zero available utterances is rendered as "[NONE]" and the
    row gets boundary_flag in {"ok", "no_prev", "no_next", "no_both"}. Nothing is dropped.
    "ok" does NOT imply the word threshold was hit — only that at least one utterance was
    available on that side. per_side_hit_threshold flags record whether W was actually reached.

Pass 1 (committed): {haiku, flash} x 0shot x {6cls, 3cls} x W=0 = 4 runs x 170 = 680 calls.
Current sweep: run the same 4 configs across W in {10, 20, 30, 50} words to map the
context-length envelope. Each W is a separate batch (one --context-words per invocation).

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

# DEFAULT_SUBSET is the 170-instance class-balanced OR subset. Pass --eval-set to
# point at any other JSONL in the same flat subset_*_v2 schema (e.g.
# data/subset_rog_test_full.jsonl for the full-ROG natural-distribution test, built
# by scripts/1_data_prep/prep_rog_test_full.py). The corpus (CORPUS) is NOT swapped
# — context windows are always resolved against the full ROG v2 corpus.
DEFAULT_SUBSET = ROOT / "data" / "subset_174_seed42_v2.jsonl"
FEWSHOT = ROOT / "data" / "fewshot_pool_18_seed42_v2.jsonl"
CORPUS = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
KEY_FILE = ROOT / "misc" / "key.txt"
RUNS_DIR = ROOT / "runs"
SCRIPT_LOG = RUNS_DIR / "_script_log.jsonl"

OR_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_TOKENS = 500
TEMPERATURE = 0
NONE_TOKEN = "[NONE]"
TARGET_TOKEN = "[TARGET]"
TARGET_MARKERS = ["eol", "after_spk", "own_line"]
DEFAULT_TARGET_MARKER = "eol"      # confirmed cleanest parse in the 2026-11-17 placement test
DEFAULT_CONTEXT_WORDS = 20         # 0 = legacy ±1-utterance; otherwise accumulate until W words per side


def context_mode_tag(words):
    """Canonical context_mode string. W=0 preserves the legacy 'ctx3' label."""
    return "ctx3" if words == 0 else f"ctx_w{words}"

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


def _word_count(text):
    return len((text or "").split())


def _accumulate_side(lst, start_idx, direction, target_words):
    """Walk lst from start_idx in direction (+1 or -1), collecting utterances until the
    cumulative word count reaches target_words. The utterance that CROSSES the threshold is
    included. target_words=0 → take exactly one utterance (legacy ±1 behaviour). Returns
    (items in chronological order, cumulative word count, hit_threshold flag).
    """
    items = []
    total = 0
    i = start_idx + direction
    while 0 <= i < len(lst):
        items.append(lst[i])
        total += _word_count(lst[i]["text"])
        if target_words == 0 or total >= target_words:
            break
        i += direction
    if direction < 0:
        items.reverse()
    hit = (target_words == 0 and len(items) > 0) or (target_words > 0 and total >= target_words)
    return items, total, hit


def build_window(inst, corpus, context_words):
    """Build the window dict for one target. Target text comes from `inst` (matches what 02_
    sent), neighbours are pulled from the full corpus. context_words=0 → legacy ±1 window.
    """
    iid = inst["instance_id"]
    if iid not in corpus.by_id:
        raise SystemExit(
            f"FATAL: target instance {iid} is not in {CORPUS}. The subset was built from this "
            f"corpus, so this means the wrong corpus file (e.g. a slim sample) or a changed "
            f"subset. Refusing to pad silently."
        )
    crow = corpus.by_id[iid]
    if crow["file_id"] != inst["file_id"]:
        raise SystemExit(f"FATAL: file_id mismatch for {iid}: subset={inst['file_id']} corpus={crow['file_id']}")

    lst = corpus.by_file[crow["file_id"]]
    i = corpus.pos[iid]
    prev_items, prev_wc, prev_hit = _accumulate_side(lst, i, -1, context_words)
    next_items, next_wc, next_hit = _accumulate_side(lst, i, +1, context_words)

    have_prev, have_next = len(prev_items) > 0, len(next_items) > 0
    flag = {(True, True): "ok", (False, True): "no_prev",
            (True, False): "no_next", (False, False): "no_both"}[(have_prev, have_next)]

    # Collapse speakers in chronological order: prev oldest → target → next newest.
    spk_map = {}
    def collapse(spk):
        spk_map.setdefault(spk, f"SPK{len(spk_map) + 1}")
        return spk_map[spk]
    prev_collapsed = [collapse(r["speaker"]) for r in prev_items]
    target_collapsed = collapse(crow["speaker"])
    next_collapsed = [collapse(r["speaker"]) for r in next_items]

    def _item_d(r, cspk):
        return {"instance_id": r["instance_id"], "text": r["text"], "speaker": r["speaker"],
                "collapsed_speaker": cspk, "word_count": _word_count(r["text"]),
                "duration_sec": round(_dur(r), 3)}
    prev_items_d = [_item_d(r, c) for r, c in zip(prev_items, prev_collapsed)]
    next_items_d = [_item_d(r, c) for r, c in zip(next_items, next_collapsed)]

    nearest_prev = prev_items[-1] if prev_items else None   # closest-to-target on prev side
    nearest_next = next_items[0] if next_items else None    # closest-to-target on next side
    distinct_spk = {crow["speaker"], *(r["speaker"] for r in prev_items + next_items)}
    all_rows = prev_items + [crow] + next_items
    window_duration = round(sum(_dur(r) for r in all_rows), 3)

    return {
        "target_instance_id": iid,
        "target_text": inst["text"],
        "target_speaker": crow["speaker"],
        "target_collapsed_speaker": target_collapsed,
        # Accumulated arrays (chronological on each side)
        "prev_items": prev_items_d,
        "next_items": next_items_d,
        "n_prev_items": len(prev_items_d),
        "n_next_items": len(next_items_d),
        "prev_word_count": prev_wc,
        "next_word_count": next_wc,
        "prev_hit_threshold": prev_hit,
        "next_hit_threshold": next_hit,
        "context_words_target": context_words,
        # Nearest-neighbour fields (preserved from the legacy W=0 schema for inspection)
        "prev_instance_id": nearest_prev["instance_id"] if nearest_prev else None,
        "next_instance_id": nearest_next["instance_id"] if nearest_next else None,
        "prev_text": nearest_prev["text"] if nearest_prev else None,
        "next_text": nearest_next["text"] if nearest_next else None,
        "prev_speaker": nearest_prev["speaker"] if nearest_prev else None,
        "next_speaker": nearest_next["speaker"] if nearest_next else None,
        # 3-slot speaker summary — [nearest_prev, target, nearest_next] collapsed tokens.
        # Meaningful only when W=0; kept for backward-compat with the committed batch.
        "window_speakers": [
            prev_collapsed[-1] if prev_collapsed else None,
            target_collapsed,
            next_collapsed[0] if next_collapsed else None,
        ],
        "window_distinct_speakers_count": len(distinct_spk),
        "boundary_flag": flag,
        "window_duration_sec": window_duration,
        "target_text_matches_corpus": inst["text"] == crow["text"],
    }


def render_window(win, target_marker):
    """Render variable-width window as [SPK*] / [NONE] / [TARGET] text lines.
    Prev side chronological; then target; then next side chronological."""
    lines = []
    if win["prev_items"]:
        for it in win["prev_items"]:
            lines.append(f"[{it['collapsed_speaker']}] {it['text']}")
    else:
        lines.append(NONE_TOKEN)

    tok = f"[{win['target_collapsed_speaker']}]"
    text = win["target_text"]
    if target_marker == "eol":
        lines.append(f"{tok} {text}  {TARGET_TOKEN}")
    elif target_marker == "after_spk":
        lines.append(f"{tok} {TARGET_TOKEN} {text}")
    elif target_marker == "own_line":
        lines.append(f"{TARGET_TOKEN}\n{tok} {text}")
    else:
        raise ValueError(target_marker)

    if win["next_items"]:
        for it in win["next_items"]:
            lines.append(f"[{it['collapsed_speaker']}] {it['text']}")
    else:
        lines.append(NONE_TOKEN)

    return "\n".join(lines)


# ---------------------------------------------------------------- prompts

def system_prompt(class_scheme):
    labels = LABELS_6 if class_scheme == "6cls" else LABELS_3
    return (
        "You classify Slovenian spoken-dialogue utterances by sentiment.\n"
        "You are shown a series of consecutive turns from one conversation "
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

def run_one(key, cfg, subset, windows, fewshot, out_dir, subset_path, n_examples=None, dry_run=False):
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
                "context_words_target": cfg["context_words"],
                "target_marker": cfg["target_marker"],
                # --- window fields (new vs 02_) ---
                "target_instance_id": win["target_instance_id"],
                "prev_items": win["prev_items"],
                "next_items": win["next_items"],
                "n_prev_items": win["n_prev_items"],
                "n_next_items": win["n_next_items"],
                "prev_word_count": win["prev_word_count"],
                "next_word_count": win["next_word_count"],
                "prev_hit_threshold": win["prev_hit_threshold"],
                "next_hit_threshold": win["next_hit_threshold"],
                "prev_instance_id": win["prev_instance_id"],
                "next_instance_id": win["next_instance_id"],
                "prev_text": win["prev_text"],
                "next_text": win["next_text"],
                "prev_speaker": win["prev_speaker"],
                "target_speaker": win["target_speaker"],
                "next_speaker": win["next_speaker"],
                "window_speakers": win["window_speakers"],
                "window_distinct_speakers_count": win["window_distinct_speakers_count"],
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
        "subset_path": str(subset_path),
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
    ap.add_argument("--context-words", type=int, default=DEFAULT_CONTEXT_WORDS,
                    help="Words accumulated per side before closing. 0 = legacy ±1-utterance window "
                         "(reproduces batch_CTX_20261117_124601).")
    ap.add_argument("--eval-set", default=None,
                    help="Override the eval JSONL (default: data/subset_174_seed42_v2.jsonl). "
                         "Any file in the same flat subset_*_v2 schema is accepted — e.g. "
                         "data/subset_rog_test_full.jsonl for the full-ROG natural-distribution test. "
                         "Every row's instance_id must exist in CORPUS (so neighbours can be resolved).")
    ap.add_argument("--eval-tag", default=None,
                    help="Prefix the batch_id with this tag (e.g. FULLROG) so full-ROG batches "
                         "sort separately from 170-subset batches under runs/. "
                         "No effect on run_id or JSONL schema. Defaults to empty.")
    ap.add_argument("--out", default=str(RUNS_DIR))
    ap.add_argument("--dry-run", action="store_true", help="Do not hit OpenRouter; echo gold label as output")
    ap.add_argument("--show-prompt", action="store_true", help="Print the first instance's messages per run config and exit")
    args = ap.parse_args()

    if args.context_words < 0:
        raise SystemExit("--context-words must be >= 0 (0 = legacy ±1-utterance)")

    if args.test and args.n_examples is None:
        args.n_examples = 3

    subset_path = Path(args.eval_set).resolve() if args.eval_set else DEFAULT_SUBSET
    if not subset_path.exists():
        raise SystemExit(f"missing eval set: {subset_path}")
    subset = load_jsonl(subset_path)
    if not CORPUS.exists():
        raise SystemExit(f"missing {CORPUS}")
    corpus = CorpusIndex(load_jsonl(CORPUS))

    # Build ALL subset windows up front, so a missing target fails before any API spend,
    # and so the boundary / word-count distributions are reported for the full subset
    # even under --test.
    W = args.context_words
    ctx_mode = context_mode_tag(W)
    windows = {inst["instance_id"]: build_window(inst, corpus, W) for inst in subset}
    boundary_dist = dict(Counter(w["boundary_flag"] for w in windows.values()))
    n_text_mismatch = sum(not w["target_text_matches_corpus"] for w in windows.values())
    n_same_spk = sum(1 for w in windows.values() if w["window_distinct_speakers_count"] == 1)
    prev_wc_list = [w["prev_word_count"] for w in windows.values()]
    next_wc_list = [w["next_word_count"] for w in windows.values()]
    n_prev_hit = sum(1 for w in windows.values() if w["prev_hit_threshold"])
    n_next_hit = sum(1 for w in windows.values() if w["next_hit_threshold"])
    avg_prev_wc = sum(prev_wc_list) / len(prev_wc_list) if prev_wc_list else 0
    avg_next_wc = sum(next_wc_list) / len(next_wc_list) if next_wc_list else 0
    print(f"Windows: {len(windows)} targets | context_mode={ctx_mode} (W={W}) | "
          f"boundary_flag: {boundary_dist} | single-speaker: {n_same_spk}")
    print(f"  per-side word counts — prev avg={avg_prev_wc:.1f} (hit W: {n_prev_hit}/{len(windows)}); "
          f"next avg={avg_next_wc:.1f} (hit W: {n_next_hit}/{len(windows)})")
    if n_text_mismatch:
        print(f"WARNING: {n_text_mismatch} targets have subset text != corpus text "
              f"(prompt uses SUBSET text, as 02_ did). Check the corpus file version.")

    models_filter = set(args.models) if args.models else None
    shots_filter = set(args.shots) if args.shots else None
    schemes_filter = set(args.schemes) if args.schemes else None

    # batch-id tag:
    #   --context-words 30                                 → batch_CTXW30_<stamp>
    #   --context-words 30 --eval-tag FULLROG              → batch_FULLROG_CTXW30_<stamp>
    #   --context-words 30 --test                          → batch_CTXW30_TEST_<stamp>
    #   --context-words 30 --eval-tag FULLROG --test       → batch_FULLROG_CTXW30_TEST_<stamp>
    w_tag = "W0" if W == 0 else f"W{W}"
    pieces = []
    if args.eval_tag: pieces.append(args.eval_tag)
    pieces.append(f"CTX{w_tag}")
    if args.test: pieces.append("TEST")
    elif args.dry_run: pieces.append("DRY")
    tag = "_".join(pieces)
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
        run_ctx_slug = "ctx" if W == 0 else f"ctxw{W}"
        configs.append({
            "run_id": f"OR-text-{run_ctx_slug}-{mname}-{shot}-{scheme}",
            "batch_id": batch_id,
            "infra": "OpenRouter",
            "modality": "text",
            "model": MODELS[mname],
            "model_name": mname,
            "context_mode": ctx_mode,
            "context_words": W,
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
        fewshot = [(ex, build_window(ex, corpus, W)) for ex in fewshot_pool]

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
            meta = run_one(key, cfg, subset, windows, fewshot, batch_dir, subset_path,
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
            "subset_path": str(subset_path),
            "eval_tag": args.eval_tag,
            "fewshot_path": str(FEWSHOT),
            "corpus_path": str(CORPUS),
            "context_mode": ctx_mode,
            "context_words_target": W,
            "target_marker": args.target_marker,
            "boundary_flags_full_subset": boundary_dist,
            "n_single_speaker_windows": n_same_spk,
            "n_target_text_mismatch": n_text_mismatch,
            "word_count_stats": {
                "prev_avg": round(avg_prev_wc, 2), "next_avg": round(avg_next_wc, 2),
                "n_prev_hit_threshold": n_prev_hit, "n_next_hit_threshold": n_next_hit,
                "prev_min": min(prev_wc_list, default=0), "prev_max": max(prev_wc_list, default=0),
                "next_min": min(next_wc_list, default=0), "next_max": max(next_wc_list, default=0),
            },
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
                "context_mode": ctx_mode,
                "context_words_target": W,
                "configs_run": [r["run_id"] for r in results],
                "total_cost_usd": total_cost,
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