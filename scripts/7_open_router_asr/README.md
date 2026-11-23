# 7 · Google Chirp 3 ASR of ROG-Dialog v2

Transcribe all of ROG-Dialog v2 with Chirp 3 via OpenRouter
(`https://openrouter.ai/api/v1/audio/transcriptions`, model `google/chirp-3`) and attach the ASR
transcripts to each instance alongside the human gold transcripts. Goal: a realistic "what the model sees in
production" variant of ROG, which the text probing matrices (Reports 2 and 4) can then be rerun on top of,
giving a gold-vs-ASR text performance delta.

OR path chosen over the native `google.cloud.speech_v2` SDK because the rest of the project already runs on
OR (same `misc/key.txt`, same retry pattern) — no new auth, no GCP project to provision.

## Why Chirp 3

Peter Rupnik (IJS)'s Chirp 3 result on SloBench ASR v2 (leaderboard
<https://slobench.cjvt.si/leaderboard/view/20>) is currently the best Slovenian ASR: WER1 = 0.0675 on the
original test set, WER2 = 0.3027 on the new (harder) extended set, with 0.4288 NORM1 bias toward the
normalised tier. Those are the headline numbers we're trying to match / beat on ROG dialogue, which is a
colloquial-dialect register the stock Chirp3 may not have been trained on.

## Plan of record

1. **Validate on the 170-subset first** (`data/subset_174_seed42_v2.jsonl` — same subset Report 3 audio probing used; 170 rows post-filter).
   Transcribe, compute WER + CER against both the colloquial and normalised ROG tiers (ROG only has the
   colloquial tier; "normalised" here means the orthographic cleanup already in `rog_dialog_v2_instance.jsonl`).
   If WER < ~0.25 on the 170-subset, proceed to full ROG. If not, fall back to turn-level or file-level
   ASR + timestamp-based re-alignment to instance boundaries.
2. **Histograms** (via `02_explore_asr_quality.py`): duration + word-count distributions for the subset vs
   the full test set vs the full ROG v2. Confirm the subset is representative before scaling.
3. **Full ROG v2** transcription (~13.6 k instances, ~15 h audio). Expect ~2 h on Google Cloud at the
   non-batch quota.
4. **Attach ASR** to a new JSONL variant: `rog_dialog_v2_instance_asr.jsonl` with an extra
   `labels.text_asr_chirp3` field (plus the full raw response in
   `metadata.asr_response_chirp3` for provenance).
5. **Rerun Reports 2 and 4** on the ASR variant — the deliverable headline is the gold-vs-ASR F1 delta.

## Preprocessing decision — instance-cut, not turn/file

ROG-Dialog v2 is already segmented per sentiment instance. Each instance has a dedicated WAV at
`data/processed_audio/ROG-Dialog-v2/<file_id>_<instance_id>.wav` (typically 1–12 s, often ~1 s). The three
options for Chirp3 input were:

- **A. Instance-cut (chosen)**: one Chirp3 call per instance. Alignment to the per-instance label is
  trivial — same `instance_id`, same WAV, one transcript per call. Downside: short clips
  (sub-500 ms) may degrade WER. Validate on the 170-subset first.
- **B. Turn-bundled**: concatenate consecutive utterances from the same speaker into turns. Fewer calls,
  more context, but requires re-splitting Chirp3's returned words back to instance boundaries via word
  timestamps. Net effort: much higher, saved only if A's WER is unacceptable.
- **C. Full-file ASR with diarization**: send the full EXB source audio. Get diarized transcript with
  speaker labels. Align back to instance boundaries via `start_t / end_t`. Richest context, but alignment
  errors compound and diarization quality on ROG's close-mic conversational audio is unproven.

A is the default. B and C are backup plans if WER on the 170-subset is bad.

## Data layout

```
data/
├── processed_audio/
│   └── ROG-Dialog-v2/
│       └── <file_id>_<instance_id>.wav         # one per instance
├── processed_jsonl/
│   ├── rog_dialog_v2_instance.jsonl            # gold text
│   ├── rog_dialog_v2_instance_asr.jsonl        # (output) gold + ASR
│   └── subset_170.jsonl                        # 170-instance validation subset
└── 7_asr_runs/
    └── chirp3_<run_id>/
        ├── predictions.jsonl                   # one line per instance, full raw response
        ├── meta.json                           # config + totals
        └── run.log                             # tee'd stdout+stderr
```

## Expected artifacts next session

- `scripts/7_open_router_asr/01_chirp3_transcribe.py` — the batch transcription driver. Resumable,
  supports `--subset_jsonl` for the 170-subset path, `--limit N` for smoke-testing, writes
  `predictions.jsonl` with the full Google Cloud response per instance.
- `scripts/7_open_router_asr/02_explore_asr_quality.py` — jupytext percent-format notebook twin.
  Loads `predictions.jsonl`, plots duration + word-count histograms, computes WER/CER (via jiwer or
  the SloBench evaluator script), produces a transcript diff view for manual inspection.
- `scripts/7_open_router_asr/03_attach_asr_to_rog.py` — merges the predictions into
  `rog_dialog_v2_instance_asr.jsonl`.

See the accompanying handoff: `misc/2026-11-23_prompt_chirp3_or_asr.md`.
