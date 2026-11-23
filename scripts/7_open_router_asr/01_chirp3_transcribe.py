#!/usr/bin/env python3
"""01_chirp3_transcribe — OpenRouter → google/chirp-3 ASR over ROG-Dialog v2.

Posts base64-encoded WAVs to https://openrouter.ai/api/v1/audio/transcriptions
and writes one JSONL line per instance with the Chirp 3 transcript plus full
raw response.

Three modes:
  (default)   live OR calls on the full input JSONL
  --dry-run   no OR calls; uses text_gold as the transcript. Still loads the
              WAV + base64-encodes it so missing-audio issues surface locally.
              Purpose: exercise the pipeline plumbing (schemas, resume,
              meta.json, 02 loader) without burning OR credits.
  --test-run  picks 5 rows (3 random + 2 shortest with duration > 5 s),
              posts them for real, computes WER/CER vs text_gold via jiwer,
              prints wall-time breakdown and writes a test_report.md to the
              run dir. Purpose: cheap sanity check before the 170-subset run.

Pattern mirrors scripts/2_probe_text/02_run_openrouter_text.py and
scripts/3_probe_audio/05_run_openrouter_audio.py (OR_URL, misc/key.txt,
retry/backoff).

Usage:
  # 1. plumbing-only dry run on the 170-subset (no OR calls)
  python 01_chirp3_transcribe.py \\
      --subset_jsonl data/subset_174_seed42_v2.jsonl --dry-run

  # 2. 5-instance smoke on real OR (spends a few cents, prints WER/CER)
  python 01_chirp3_transcribe.py \\
      --subset_jsonl data/subset_174_seed42_v2.jsonl --test-run

  # 3. full 170-subset validation run (real OR, ~170 calls)
  python 01_chirp3_transcribe.py \\
      --subset_jsonl data/subset_174_seed42_v2.jsonl \\
      --run_id chirp3_subset170_$(date +%Y-%m-%d_%H%M%S)

  # 4. full ROG v2 (run on remote after subset validates)
  python 01_chirp3_transcribe.py \\
      --jsonl data/processed_jsonl/rog_dialog_v2_instance.jsonl \\
      --run_id chirp3_fullrog_$(date +%Y-%m-%d_%H%M%S)

Auth: reads the OR key from misc/key.txt (same file the text/audio probes use).

Output schema per instance (one line in predictions.jsonl):
  {
    "instance_id":   "ROG-Dialog_...",
    "audio_path":    "data/processed_audio/ROG-Dialog-v2/....wav",
    "text_gold":     "Kaj zdaj,",
    "duration_s":    0.578,
    "transcript":    "kaj zdaj",
    "model":         "google/chirp-3",
    "language":      "sl",
    "status":        200,
    "latency_s":     1.42,
    "attempts":      [...],
    "mode":          "live" | "dry" | "test",
    "error":         null | "<repr>",
    "raw_response":  {... full OR response ...}
  }
"""

import argparse
import base64
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

OR_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
KEY_FILE = ROOT / "misc" / "key.txt"

DEFAULT_MODEL = "google/chirp-3"
DEFAULT_LANGUAGE = "sl"
DEFAULT_TIMEOUT = 120
DEFAULT_SEED = 42


def load_api_key(path: Path | None = None) -> str:
    kf = Path(path) if path else KEY_FILE
    if not kf.exists():
        print(f"ERROR: OR key file not found: {kf}", file=sys.stderr)
        sys.exit(2)
    return kf.read_text().strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=str,
                     help="full ROG JSONL to transcribe")
    src.add_argument("--subset_jsonl", type=str,
                     help="pre-filtered subset JSONL (e.g. subset_174_seed42_v2.jsonl)")

    p.add_argument("--run_id", default=None,
                   help="run subdir name under data/7_asr_runs/ "
                        "(default: chirp3_<timestamp>, or chirp3_test_<timestamp> under --test-run)")
    p.add_argument("--audio_root", default=None,
                   help="base dir to resolve row['audio_path'] under "
                        "(default: project ROOT, i.e. two dirs above this script)")
    p.add_argument("--resume", action="store_true",
                   help="if predictions.jsonl exists in run dir, skip already-transcribed instance_ids")
    p.add_argument("--limit", type=int, default=None,
                   help="transcribe only the first N instances (smoke test)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="OR model slug")
    p.add_argument("--language", default=DEFAULT_LANGUAGE,
                   help="ISO language code passed to OR (e.g. sl, hr, en)")
    p.add_argument("--sleep_between_calls_s", type=float, default=0.0,
                   help="throttle (seconds per call); raise if hitting quotas")
    p.add_argument("--max_attempts", type=int, default=6,
                   help="retry each call this many times on transient errors")
    p.add_argument("--timeout_s", type=int, default=DEFAULT_TIMEOUT,
                   help="per-request timeout")

    p.add_argument("--key_file", default=None,
                   help="path to OR API key file (default: <ROOT>/misc/key.txt)")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="no OR calls; copy text_gold into transcript. Still loads WAV + b64.")
    p.add_argument("--test-run", action="store_true", dest="test_run",
                   help="5-instance real-OR smoke: 3 random + 2 shortest with duration>5s; "
                        "computes WER/CER, writes test_report.md.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="RNG seed for --test-run row sampling")
    return p.parse_args()


def load_input_rows(jsonl_path: Path) -> list[dict]:
    """Load the ROG JSONL as a list of row dicts. Filters to non-null audio paths only."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r.get("audio_path"):
                continue
            rows.append(r)
    return rows


def load_already_done(run_dir: Path) -> set[str]:
    """When --resume, read predictions.jsonl and skip instance_ids already transcribed."""
    pred = run_dir / "predictions.jsonl"
    if not pred.exists():
        return set()
    done = set()
    with open(pred) as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("instance_id") and not rec.get("error"):
                    done.add(rec["instance_id"])
            except json.JSONDecodeError:
                continue
    return done


def _backoff_sec(status, attempt, resp=None) -> float:
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


def call_openrouter_transcribe(key: str, model: str, language: str, b64_wav: str,
                               timeout: int, max_retries: int):
    """POST base64 WAV to OR audio/transcriptions. Returns (status, json, latency_s, attempts)."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ijs.si/rog-dialog-chirp3-asr",
        "X-Title": "rog-dialog-chirp3-asr",
    }
    body = {
        "model": model,
        "language": language,
        "input_audio": {"data": b64_wav, "format": "wav"},
    }
    attempts = []
    for i in range(max_retries):
        t0 = time.time()
        try:
            resp = requests.post(OR_URL, headers=headers, json=body, timeout=timeout)
            latency_s = round(time.time() - t0, 3)
            try:
                j = resp.json()
            except Exception as e:
                j = {"_parse_error": str(e), "_text": resp.text[:4000]}
            attempts.append({"attempt": i + 1, "status": resp.status_code, "latency_s": latency_s})
            if resp.status_code < 500 and resp.status_code != 429:
                return resp.status_code, j, latency_s, attempts
            if i < max_retries - 1:
                time.sleep(_backoff_sec(resp.status_code, i, resp))
        except (requests.Timeout, requests.ConnectionError) as e:
            latency_s = round(time.time() - t0, 3)
            attempts.append({"attempt": i + 1, "status": None, "latency_s": latency_s, "exception": repr(e)})
            if i < max_retries - 1:
                time.sleep(_backoff_sec(None, i))
        except Exception as e:
            latency_s = round(time.time() - t0, 3)
            attempts.append({"attempt": i + 1, "status": None, "latency_s": latency_s, "exception": repr(e)})
            if i < max_retries - 1:
                time.sleep(_backoff_sec(None, i))
    return None, {"_exhausted": True, "_attempts": attempts}, 0.0, attempts


def resolve_audio_path(row: dict, audio_root: Path) -> Path:
    p = Path(row["audio_path"])
    return p if p.is_absolute() else (audio_root / p)


def row_duration(row: dict) -> float:
    if row.get("duration") is not None:
        return float(row["duration"])
    if row.get("start_t") is not None and row.get("end_t") is not None:
        return float(row["end_t"]) - float(row["start_t"])
    return float(row.get("metadata", {}).get("audio_length") or 0.0)


def pick_test_rows(rows: list[dict], seed: int) -> list[dict]:
    """3 random + 2 shortest-with-duration>5s, de-duplicated by instance_id, order preserved."""
    rng = random.Random(seed)
    rand3 = rng.sample(rows, min(3, len(rows)))
    over5 = sorted((r for r in rows if row_duration(r) > 5.0), key=row_duration)[:2]
    picked, seen = [], set()
    for r in list(rand3) + list(over5):
        if r["instance_id"] in seen:
            continue
        seen.add(r["instance_id"])
        picked.append(r)
    return picked


def run_live_call(audio_path: Path, key: str, args) -> dict:
    """Load WAV, base64-encode, post to OR, return a dict of result fields."""
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    status, body, latency_s, attempts = call_openrouter_transcribe(
        key=key, model=args.model, language=args.language,
        b64_wav=b64, timeout=args.timeout_s, max_retries=args.max_attempts,
    )
    transcript = None
    if isinstance(body, dict):
        transcript = body.get("text")
    return {
        "transcript": transcript,
        "status": status,
        "latency_s": latency_s,
        "attempts": attempts,
        "raw_response": body,
    }


def run_dry(audio_path: Path, row: dict) -> dict:
    with open(audio_path, "rb") as f:
        _ = base64.b64encode(f.read())
    return {
        "transcript": row.get("text"),
        "status": 200,
        "latency_s": 0.0,
        "attempts": [{"attempt": 1, "status": 200, "latency_s": 0.0, "dry_run": True}],
        "raw_response": {"_dry_run": True, "echoed_from": "text_gold"},
    }


def process_row(row: dict, key: str, args, audio_root: Path, mode: str) -> dict:
    inst = row["instance_id"]
    audio_path = resolve_audio_path(row, audio_root)
    t0 = time.time()
    rec = {
        "instance_id": inst,
        "audio_path":  row["audio_path"],
        "text_gold":   row.get("text"),
        "duration_s":  round(row_duration(row), 3),
        "model":       args.model,
        "language":    args.language,
        "mode":        mode,
        "error":       None,
    }
    try:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if args.dry_run:
            out = run_dry(audio_path, row)
        else:
            out = run_live_call(audio_path, key, args)
        rec.update(out)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec.setdefault("latency_s", round(time.time() - t0, 3))
    return rec


def _try_jiwer():
    try:
        import jiwer  # noqa: F401
        return jiwer
    except Exception:
        return None


def test_report(records: list[dict], run_dir: Path, args, wall_time_s: float) -> str:
    jiwer = _try_jiwer()
    lines = [
        f"# Chirp 3 test-run — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"- model: `{args.model}`",
        f"- language: `{args.language}`",
        f"- rows: {len(records)}",
        f"- wall time: {wall_time_s:.2f} s total, "
        f"{(wall_time_s / max(len(records),1)):.2f} s/row avg",
        f"- OR latency: sum {sum(r.get('latency_s',0) or 0 for r in records):.2f} s, "
        f"max {max((r.get('latency_s',0) or 0) for r in records):.2f} s",
        "",
    ]
    oks = [r for r in records if not r.get("error") and r.get("transcript") is not None]
    if jiwer and oks:
        golds = [r["text_gold"] or "" for r in oks]
        asrs  = [r["transcript"] or "" for r in oks]
        try:
            wer_micro = jiwer.wer(golds, asrs)
            cer_micro = jiwer.cer(golds, asrs)
            lines += [f"- WER (micro, n={len(oks)}): **{wer_micro:.3f}**",
                      f"- CER (micro, n={len(oks)}): **{cer_micro:.3f}**", ""]
        except Exception as e:
            lines += [f"- jiwer aggregate failed: {e}", ""]
    elif not jiwer:
        lines += ["- jiwer not installed → WER/CER skipped. `pip install jiwer` to enable.", ""]

    lines += ["| idx | instance_id | dur(s) | status | latency(s) | wer | cer | gold → asr |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for i, r in enumerate(records):
        wer = cer = ""
        if jiwer and not r.get("error") and r.get("transcript") is not None:
            try:
                wer = f"{jiwer.wer(r['text_gold'] or '', r['transcript'] or ''):.3f}"
                cer = f"{jiwer.cer(r['text_gold'] or '', r['transcript'] or ''):.3f}"
            except Exception:
                pass
        gold = (r.get("text_gold") or "").replace("|", "\\|")
        asr  = (r.get("transcript") or r.get("error") or "").replace("|", "\\|")
        lines.append(
            f"| {i+1} | `{r['instance_id']}` | {r.get('duration_s','')} | "
            f"{r.get('status','')} | {r.get('latency_s','')} | {wer} | {cer} | "
            f"`{gold}` → `{asr}` |"
        )
    report = "\n".join(lines) + "\n"
    (run_dir / "test_report.md").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()

    jsonl_path = Path(args.jsonl or args.subset_jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = (ROOT / jsonl_path).resolve()
    if not jsonl_path.exists():
        print(f"ERROR: input JSONL not found: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    audio_root = Path(args.audio_root).resolve() if args.audio_root else ROOT

    if args.test_run and args.dry_run:
        print("ERROR: --dry-run and --test-run are mutually exclusive.", file=sys.stderr)
        sys.exit(2)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    if args.run_id:
        run_id = args.run_id
    elif args.test_run:
        run_id = f"chirp3_test_{stamp}"
    elif args.dry_run:
        run_id = f"chirp3_dry_{stamp}"
    else:
        run_id = f"chirp3_{stamp}"
    run_dir = ROOT / "data" / "7_asr_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    key = "" if args.dry_run else load_api_key(args.key_file)

    rows = load_input_rows(jsonl_path)
    print(f"loaded {len(rows)} rows from {jsonl_path}")

    if args.test_run:
        rows = pick_test_rows(rows, seed=args.seed)
        print(f"test-run → {len(rows)} rows (3 random seed={args.seed} + 2 shortest >5s)")
    elif args.limit:
        rows = rows[: args.limit]
        print(f"limit → {len(rows)} rows")

    already = load_already_done(run_dir) if args.resume else set()
    if already:
        print(f"resume: skipping {len(already)} already-transcribed instance_ids")

    (run_dir / "meta.json").write_text(json.dumps({
        "run_id":    run_id,
        "started":   datetime.now(timezone.utc).isoformat(),
        "input":     str(jsonl_path.relative_to(ROOT) if jsonl_path.is_relative_to(ROOT) else jsonl_path),
        "audio_root": str(audio_root),
        "model":     args.model,
        "language":  args.language,
        "dry_run":   args.dry_run,
        "test_run":  args.test_run,
        "seed":      args.seed,
        "limit":     args.limit,
        "resume":    args.resume,
        "max_attempts": args.max_attempts,
        "timeout_s":    args.timeout_s,
        "n_rows":    len(rows),
        "n_already_done": len(already),
    }, indent=2))

    mode = "dry" if args.dry_run else ("test" if args.test_run else "live")
    pred_path = run_dir / "predictions.jsonl"
    n_ok = n_err = 0
    records = []
    wall_t0 = time.time()
    with open(pred_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            inst = row["instance_id"]
            if inst in already:
                continue
            rec = process_row(row, key, args, audio_root, mode)
            if rec.get("error"):
                n_err += 1
            else:
                n_ok += 1
            records.append(rec)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if args.sleep_between_calls_s > 0 and not args.dry_run:
                time.sleep(args.sleep_between_calls_s)
            if args.test_run or i % 25 == 0:
                print(f"  {i}/{len(rows)}  ok={n_ok} err={n_err}  "
                      f"last_latency={rec.get('latency_s','?')}s")
    wall_time_s = time.time() - wall_t0

    print(f"done: ok={n_ok} err={n_err}  wall={wall_time_s:.2f}s  →  {pred_path.relative_to(ROOT)}")

    if args.test_run:
        report = test_report(records, run_dir, args, wall_time_s)
        print()
        print(report)


if __name__ == "__main__":
    main()
