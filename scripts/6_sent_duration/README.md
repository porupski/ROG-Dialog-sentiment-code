# sent_duration — sentiment-signal vs utterance-duration envelope study

Answer the question the matrix run can't: *does sentiment classification signal grow,
shrink, or stay flat with utterance duration* — on text (word count) and on audio
(seconds) — and does the answer differ between parliament and colloquial domains?

Phase A is **eval-side only**: it reads existing `predictions_<src>.jsonl` artefacts from
a `4_finetune_text` matrix run, bins each test item by duration or word count, and
computes per-bin metrics. No retraining.

Phase B (cross-lang replication across ParlaSpeech 3.0) and Phase C (train-side duration
specialists) are deferred pending Phase A's envelope shape.

## Prerequisites

1. A completed matrix run — e.g. `misc/run_results/finetune_20261113_162724/` or
   `runs/finetune_<stamp>/`. Must contain `<cell>/predictions_<src>.jsonl` for each
   `(backbone, mixture, test_source)` cell.
2. The source JSONLs used for the matrix must be present under `data/` so each
   prediction row can be joined to its duration/word-count metadata by `instance_id`.
   Fallbacks (word-count from the predictions row's `text` field) kick in when a
   source JSONL is missing — the script will warn and continue.

## Phase A — launch

```bash
# Primary run (fixed bins)
python3 scripts/6_sent_duration/01_bin_predictions.py \
    misc/run_results/finetune_20261113_162724 \
    --bin_mode fixed

# Sanity cross-check (quantile bins, same matrix)
python3 scripts/6_sent_duration/01_bin_predictions.py \
    misc/run_results/finetune_20261113_162724 \
    --bin_mode quantile

# Report
python3 scripts/6_sent_duration/02_envelope_report.py \
    misc/run_results/finetune_20261113_162724
```

## Outputs

`<matrix_dir>/sent_duration/`
- `matrix_duration_long_<bin_mode>.csv` — one row per
  `(backbone × mixture × test_source × axis × bin)` with all metrics plus per-bin n.
- `envelope_<test_src>_<axis>_<bin_mode>.png` — primary curve, one line per backbone,
  best-mixture-avg per bin. Self-labeled cells excluded.
- `envelope_facet_<test_src>_<axis>_<bin_mode>.png` — audit panel: per-backbone
  subplots, one line per mixture.
- `report_phaseA.md` — per-test envelope tables, structure classification
  (monotone±/flat/U-shape) via per-backbone Spearman and linear slope, plus the
  fixed-vs-quantile sanity cross-check.

## Design notes

- **Axis per test source:** declared in `config.json → test_source_axis`. ROG uses
  audio-sec AND word-count (we have `start_t`/`end_t`). ParlaSent-SL uses word-count
  only (no audio exists). ParlaSpeech-SI uses word-count as a Phase-A fallback
  until its slim JSONL is re-exposed; then it will switch to audio-sec.
- **Fixed vs quantile:** fixed boundaries match prior literature (Kumawat & Routray
  1–4 s SER; Zhang et al. tweet-length bins). Quantile bins hold per-bin n constant
  so apparent structure can be sanity-checked against boundary artefacts.
- **Self-labeled flag:** cells where `xlm-r-parlasent` trained on any mixture
  containing `P` reproduce the ParlaSent-model's own labels on the ParlaSpeech test
  set — self-consistency, not transfer. Marked ⚠ in the report; excluded from
  the primary envelope line but kept in the long CSV.
- **Min bin n:** bins with fewer than `min_bin_n` (default 25) items are kept in the
  long CSV with a warning flag but omitted from the primary envelope plot.

## What this folder is NOT

- Not an audio-modality probe (that's the future `7_sent_duration_audio/`).
- Not a trainer. Zero model weights are touched. If the envelope reveals structure
  worth training against, `05_train_duration_cell.py` + `06_run_duration_matrix.py`
  land as Phase C.
- Not language-agnostic yet in the Phase-A scripts. The `languages_registry` block
  in `config.json` is a Phase-B placeholder; current scripts default to SI.
