# Sentiment Search

Probing where sentiment signal actually lives — **text vs audio, across models and domains** — for short Slovenian spoken utterances. Target deliverable: a Slovenian speech-sentiment model for 2–5 s dialogue clips.

Three data sources, same ParlaSent 0–5 ordinal schema:
- **ParlaSent-SL** — parliamentary text, 1,024 Slovenian sentences
- **ParlaSpeech-SI v4** — parliamentary speech (audio + transcripts), ParlaSent-labeled
- **ROG-Dialog v2** — colloquial Slovenian dialogue (audio + transcripts), human-labeled

See [`collective_results.md`](collective_results.md) for numbers and [`development_log.md`](development_log.md) for the how/why trail.

---

## Layout

```
data/                   source JSONLs + derived subsets  (gitignored)
misc/                   keys, cached raw downloads, notes (gitignored)
runs/                   all script outputs                (gitignored)
scripts/
  analyze_results.py           shared probe analyzer (reads text + audio runs)
  0_env_setup/                 conda env recipes (cloned from SSP)
  1_data_prep/
    01_build_subset.py         ROG eval subset + few-shot pool
    prep_parlasent.py          CLARIN.SI download + canonical JSONL
    prep_parlaspeech.py        slim a ParlaSpeech JSONL (drops heavy align fields)
  2_probe_text/
    02_run_openrouter_text.py  Claude / Gemini / GPT text zero/few-shot
    04_run_parlasent.py        local XLM-R family (fillmask / linear probe / regression head)
  3_probe_audio/
    05_run_openrouter_audio.py Gemini / GPT audio zero/few-shot
  4_finetune_text/
    finetune_text.py           single (backbone × mixture) regression fine-tune
    run_matrix.py              the full 3 × 7 matrix runner (subprocess per cell)
    utils_text_finetune.py
    config.json, README.md
  5_finetune_audio/            placeholder — same shape as 4_finetune_text/
```

---

## Setup

```bash
bash scripts/0_env_setup/setup_env_cuda.sh   # or setup_env_cpu.sh for data prep
mamba activate ssp-cuda              # or ssp
```

Keys:

```bash
echo "<openrouter-key>" > misc/key.txt
```

## Data

**ROG-Dialog v2** — symlink from an existing checkout (text + audio):

```bash
ln -s /abs/path/to/slavic-speech-pipeline/data/processed_jsonl/rog_dialog_v2_instance.jsonl \
      data/rog_dialog_v2_instance.jsonl
mkdir -p data/processed_audio
ln -s /abs/path/to/slavic-speech-pipeline/data/processed_audio/ROG-Dialog-v2 \
      data/processed_audio/ROG-Dialog-v2
```

**ParlaSent** — downloads from CLARIN.SI:

```bash
python3 scripts/1_data_prep/prep_parlasent.py
```

**ParlaSpeech-SI v4** — slim the raw JSONL (drops `words_align` / `graph_align` / `linguistic_annots`, ~2/3 size reduction):

```bash
python3 scripts/1_data_prep/prep_parlaspeech.py
# default --src is /cache/ivanp/projects/slavic-speech-pipeline/data/unpacked/
#                  ParlaSpeech-SI-v4/ParlaSpeech-SI.v4.0/ParlaSpeech-SI.v4.0.jsonl
# default --out is data/parlaspeech_v4_si_utterance_instance.jsonl
```

If a dataset is missing, downstream scripts degrade gracefully — the finetune matrix tags affected cells `skipped(missing=...)` or `degraded(dropped=...)` instead of crashing.

## Pipeline

**Probe phase** — measure how far off-the-shelf systems get:

```bash
python3 scripts/1_data_prep/01_build_subset.py       # ROG 170-instance eval + 18 few-shots
python3 scripts/2_probe_text/02_run_openrouter_text.py
python3 scripts/2_probe_text/04_run_parlasent.py
python3 scripts/3_probe_audio/05_run_openrouter_audio.py
python3 scripts/analyze_results.py runs/<batch_dir>
```

Each run writes `predictions.jsonl` in the same schema so `analyze_results.py` works on any batch.

**Fine-tune phase** — build the deliverable:

```bash
# Smoke-test (CPU, ~1 min/cell, 128 train caps):
python3 scripts/4_finetune_text/run_matrix.py --mode test

# Full matrix (reserved GPU 2, tmux):
tmux new -s six
python3 scripts/4_finetune_text/run_matrix.py --mode full --use_gpu 2>&1 \
    | tee -a runs/finetune_$(date +%F_%H%M).log
```

See [`scripts/4_finetune_text/README.md`](scripts/4_finetune_text/README.md) for subset flags, output schema, and GPU-guard details.
