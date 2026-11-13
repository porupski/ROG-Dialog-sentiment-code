# finetune_text — matrix fine-tune

Fine-tune XLM-R-family backbones on mixtures of {ParlaSent-SL, ParlaSpeech-SI transcripts,
ROG-Dialog}, evaluate each on all three test sets. Regression head (scalar 0–5) with
MSE + z-score normalization, matching the ParlaSent paper's protocol.

## Prerequisites

1. `python3 scripts/1_data_prep/prep_parlasent.py` — downloads ParlaSent 1.0 from CLARIN.SI
   into `misc/parlasent_raw/` and emits `data/parlasent_<lang>_instance.jsonl`. ~8 MB.
2. ROG-Dialog v2 (`data/rog_dialog_v2_instance.jsonl`) — symlink or copy from SSP.
3. ParlaSpeech-SI slim JSONL — produce with `prep_parlaspeech.py`, see root README. If it
   is missing on first run the matrix will skip the mixtures that need it instead of
   crashing (`skipped(missing=parlaspeech_si)` / `degraded(dropped=parlaspeech_si)`).

## Launch (reserved GPU 2, tee logs)

```bash
tmux new -s six
python3 scripts/4_finetune_text/run_matrix.py --mode full --use_gpu 2>&1 \
    | tee -a runs/finetune_$(date +%F_%H%M).log
```

Smoke-test first (CPU, 1 min/cell, tiny caps):

```bash
python3 scripts/4_finetune_text/run_matrix.py --mode test
```

## Running subsets

Only two backbones × three mixtures:

```bash
python3 scripts/4_finetune_text/run_matrix.py --mode full --use_gpu \
    --backbones FacebookAI/xlm-roberta-large classla/xlm-r-parla \
    --mixtures S R SPR
```

## Outputs

`runs/finetune_<timestamp>/`
- `matrix_summary.json` — one row per cell: tag, backbone, mixture, status, elapsed
- `<backbone_short>__<mixture>/`
  - `summary.json`     — resolved config, train/dev/test sizes, label-norm stats,
                         per-eval-source metrics (R², MAE, ρ, F1_6cls, F1_3cls),
                         HF state log history (per-step loss and eval metrics)
  - `predictions_<src>.jsonl` — one row per test instance with gold + pred scalar
                                and rounded 6cls/3cls labels

Model weights are **NOT** saved. If a cell wins and you want to ship it, rerun that
single `--backbone X --mixture Y` cell with the save knob flipped (TODO: add a
`--save-model` escape hatch to the engine when we need it).

## GPU guard

`run_matrix.py` sets `CUDA_VISIBLE_DEVICES` **before** any torch import, and propagates
it to every child subprocess. Reserved GPU index lives in `config.json` at
`shared.reserved_gpu` (default `"2"`). Without `--use_gpu` the children get
`CUDA_VISIBLE_DEVICES=""` — forced to CPU — so an accidental launch can't land on
someone else's GPU.
