#!/usr/bin/env python3
"""01_chirp3_transcribe — Google Cloud Speech-to-Text v2 (Chirp 3) over ROG-Dialog v2.

SKELETON — next session fleshes out. This file holds the shape of the driver, the
argparse surface, the resume logic, and the output schema; the actual
Google Cloud Speech client calls are marked TODO and left as stubs.

Pattern mirrors scripts/2_probe_text/02_run_openrouter_text.py:
  - argparse for mode / subset / limit / resume flags
  - append-only JSONL output with full raw response + metadata per instance
  - meta.json per run
  - stdout+stderr teed to runs/<run_id>/run.log via the caller

Usage (once fleshed out):
  # smoke-test on 10 instances from the 170-subset
  python 01_chirp3_transcribe.py --subset_jsonl data/processed_jsonl/subset_170.jsonl --limit 10

  # full 170-subset validation run
  python 01_chirp3_transcribe.py --subset_jsonl data/processed_jsonl/subset_170.jsonl \\
      --run_id chirp3_subset170_$(date +%Y-%m-%d_%H%M%S)

  # full ROG v2 (after 170-subset validates)
  python 01_chirp3_transcribe.py --jsonl data/processed_jsonl/rog_dialog_v2_instance.jsonl \\
      --run_id chirp3_fullrog_$(date +%Y-%m-%d_%H%M%S) \\
      2>&1 | tee runs/_chirp3_fullrog_$(date +%Y-%m-%d_%H%M%S).log

Environment:
  GOOGLE_CLOUD_PROJECT      — GCP project id (required)
  GOOGLE_APPLICATION_CREDENTIALS — service-account JSON path (required)

Output schema per instance (one line in predictions.jsonl):
  {
    "instance_id": "ROG-Dialog_...",
    "audio_path":  "data/processed_audio/ROG-Dialog-v2/....wav",
    "text_gold":   "Kaj zdaj,",
    "duration_s":  0.578,
    "transcript":  "kaj zdaj",
    "language_code": "sl-SI",
    "confidence":  0.9123,
    "words":       [ {word, start_offset_s, end_offset_s, speaker_label}, ... ],
    "raw_response": {<full RecognizeResponse as dict>},
    "model":       "chirp_3",
    "region":      "us",
    "latency_s":   1.42,
    "attempt":     1,
    "error":       null | "<repr>"
  }
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


# TODO (next session): import and instantiate once in main(), reuse across calls.
# from google.cloud.speech_v2 import SpeechClient
# from google.cloud.speech_v2.types import cloud_speech
# from google.api_core.client_options import ClientOptions


DEFAULT_MODEL = "chirp_3"
DEFAULT_REGION = "us"
DEFAULT_LANGS = ["sl-SI", "auto"]   # try sl-SI first, fall back to auto if empty


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=str,
                     help="full ROG JSONL to transcribe")
    src.add_argument("--subset_jsonl", type=str,
                     help="pre-filtered subset JSONL (e.g. the 170-subset)")

    p.add_argument("--run_id", default=None,
                   help="run subdir name under data/7_asr_runs/ "
                        "(default: chirp3_<timestamp>)")
    p.add_argument("--resume", action="store_true",
                   help="if predictions.jsonl exists in run dir, skip already-transcribed instance_ids")
    p.add_argument("--limit", type=int, default=None,
                   help="transcribe only the first N instances (smoke test)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Google Cloud Speech model slug")
    p.add_argument("--region", default=DEFAULT_REGION,
                   help="GCP region for the speech endpoint")
    p.add_argument("--language_codes", nargs="+", default=DEFAULT_LANGS,
                   help="language codes; use 'auto' for language-ID")
    p.add_argument("--diarize", action="store_true",
                   help="enable speaker diarization in the response")
    p.add_argument("--sleep_between_calls_s", type=float, default=0.0,
                   help="throttle (seconds per call); raise if hitting quotas")
    p.add_argument("--max_attempts", type=int, default=3,
                   help="retry each call this many times on transient errors")
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


def transcribe_one(audio_path: Path, args, client=None) -> dict:
    """Submit one WAV to Chirp 3, return the structured response dict.

    TODO (next session): wire the google.cloud.speech_v2 client here per the
    sync pattern in the Chirp 3 docs. Pseudocode:

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=args.language_codes,
            model=args.model,
            features=(cloud_speech.RecognitionFeatures(
                diarization_config=cloud_speech.SpeakerDiarizationConfig(),
            ) if args.diarize else None),
        )
        with open(audio_path, "rb") as f:
            content = f.read()
        req = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{project_id}/locations/{args.region}/recognizers/_",
            config=config,
            content=content,
        )
        resp = client.recognize(request=req)
        # Flatten resp.results[*].alternatives[0] into the output schema.
    """
    raise NotImplementedError("Chirp 3 client call is a stub — flesh out next session.")


def main() -> None:
    args = parse_args()

    jsonl_path = Path(args.jsonl or args.subset_jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = (ROOT / jsonl_path).resolve()
    if not jsonl_path.exists():
        print(f"ERROR: input JSONL not found: {jsonl_path}", file=sys.stderr)
        sys.exit(2)

    run_id = args.run_id or f"chirp3_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
    run_dir = ROOT / "data" / "7_asr_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "meta.json").write_text(json.dumps({
        "run_id":    run_id,
        "started":   datetime.now(timezone.utc).isoformat(),
        "input":     str(jsonl_path.relative_to(ROOT) if jsonl_path.is_relative_to(ROOT) else jsonl_path),
        "model":     args.model,
        "region":    args.region,
        "language_codes": args.language_codes,
        "diarize":   args.diarize,
        "limit":     args.limit,
        "resume":    args.resume,
    }, indent=2))

    rows = load_input_rows(jsonl_path)
    print(f"loaded {len(rows)} rows from {jsonl_path}")
    if args.limit:
        rows = rows[: args.limit]
        print(f"limit → {len(rows)} rows")

    already = load_already_done(run_dir) if args.resume else set()
    if already:
        print(f"resume: skipping {len(already)} already-transcribed instance_ids")

    # TODO (next session): construct the SpeechClient once, pass to transcribe_one.
    # client = SpeechClient(client_options=ClientOptions(
    #     api_endpoint=f"{args.region}-speech.googleapis.com"))
    client = None

    pred_path = run_dir / "predictions.jsonl"
    n_ok = n_err = 0
    with open(pred_path, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            inst = row["instance_id"]
            if inst in already:
                continue
            audio_path = ROOT / row["audio_path"]
            t0 = time.time()
            rec = {
                "instance_id": inst,
                "audio_path":  row["audio_path"],
                "text_gold":   row.get("text"),
                "duration_s":  (row.get("end_t", 0.0) - row.get("start_t", 0.0)) or
                               row.get("metadata", {}).get("audio_length"),
                "model":       args.model,
                "region":      args.region,
                "error":       None,
            }
            try:
                # resp = transcribe_one(audio_path, args, client=client)
                # rec.update(resp)
                raise NotImplementedError(
                    "Chirp3 stub — see transcribe_one(). Fleshed out next session."
                )
            except NotImplementedError as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                n_err += 1
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                n_err += 1
            else:
                n_ok += 1
            finally:
                rec["latency_s"] = round(time.time() - t0, 3)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            if args.sleep_between_calls_s > 0:
                time.sleep(args.sleep_between_calls_s)
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}  ok={n_ok} err={n_err}")

    print(f"done: ok={n_ok} err={n_err}  →  {pred_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
