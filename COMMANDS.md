# Commands log

Running log of commands executed on the remote `ssp-cuda` env, with timestamps,
result paths, and status markers. Point of this file: reproducibility — anyone
can replay the pipeline by running the commands in order and pointing scripts at
the recorded result dirs.

Status legend: `TODO` · `RUNNING` · `DONE` · `FAILED` · `SKIPPED`

Conventions:
- One block per distinct command. The block title is the thing being done.
- Fill in `When`, `Status`, `Result dir`, `Notes` as each command lands.
- Never delete past blocks — leave them with status + results.

Mixture alphabet (recurring glossary — same as `scripts/4_finetune_text/utils_text_finetune.py`):

| key | sources | gloss |
|---|---|---|
| `S` | `parlasent_sl` | parliament text, published labels |
| `P` | `parlaspeech_si` | parliament speech transcripts, XLM-R-ParlaSent self-labels |
| `R` | `rog_dia` (2 s-filtered) | ROG dialogue, duration ≥ 2 s |
| `SP` | `S + P` | parliament text + speech |
| `SR` | `S + R` | parliament text + ROG 2 s |
| `PR` | `P + R` | parliament speech + ROG 2 s |
| `SPR` | `S + P + R` | all three, ROG 2 s-filtered |
| `R_full` | `rog_dia_full` (no 2 s filter) | ROG full train split |
| `SR_full` | `S + rog_dia_full` | parliament text + ROG full |
| `PR_full` | `P + rog_dia_full` | parliament speech + ROG full |
| `SPR_full` | `S + P + rog_dia_full` | all three, ROG full-train |

Backbone short names (`BACKBONES` in `utils_text_finetune.py`):
- `xlm-roberta-large` → `FacebookAI/xlm-roberta-large` (baseline, no sentiment pretrain)
- `xlm-r-parla` → `classla/xlm-r-parla` (parliamentary MLM continue-pretrain, no sentiment head)
- `xlm-r-parlasent` → `classla/xlm-r-parlasent` (parla + ParlaSent sentiment regression head)

---

## Phase 0 — full-ROG OpenRouter sanity (Flash 0-shot)

### 0.1 — iso baseline on full ROG test
- **When**: 2026-11-18 (launched)
- **Status**: RUNNING
- **Command**:
  ```
  python scripts/2_probe_text/02_run_openrouter_text.py \
      --models flash --shots 0shot \
      --eval-set data/subset_rog_test_full.jsonl --eval-tag FULLROG
  ```
- **Result dir**: `runs/batch_FULLROG_<stamp>/` (fill in)
- **Analyze**: `python3 scripts/analyze_probe.py runs/batch_FULLROG_<stamp>`
- **Lean copy**: `results/or_phase0/iso/` (see §Results-save)

### 0.2 — W=30 context on full ROG test (queue after 0.1 finishes)
- **When**: 2026-11-18 (queued)
- **Status**: TODO
- **Command** (real — strip `--test --dry-run` from the smoke variant you ran earlier):
  ```
  python scripts/2_probe_text/03_run_openrouter_text_context.py \
      --models flash --shots 0shot --context-words 30 \
      --eval-set data/subset_rog_test_full.jsonl --eval-tag FULLROG
  ```
- **Result dir**: `runs/batch_FULLROG_CTXW30_<stamp>/` (fill in)
- **Analyze**: `python3 scripts/analyze_probe.py runs/batch_FULLROG_CTXW30_<stamp>`
- **Lean copy**: `python3 scripts/sync_results.py runs/batch_FULLROG_CTXW30_<stamp> or_phase0/ctxw30`
- **Post-land**: fill in Report 2 (`docs/2_open_router_probing_text.html` → the 2026-11-18 Progress section I added) with the full-ROG iso vs W=30 F1 numbers.

---

## Phase 2 — stock XLM-R-parlasent inference (no fine-tune)

### 2.1 — ParlaSent-ZS on full ROG test
- **When**: 2026-11-18 11:52 UTC
- **Status**: DONE
- **Command**:
  ```
  python3 scripts/2_probe_text/04_run_parlasent.py \
      --models parlasent --protocols reg \
      --eval-set data/subset_rog_test_full.jsonl --eval-tag FULLROG --gpu 3
  ```
- **Result dir**: `runs/batch_LOCAL_FULLROG_20261118_115203/`
- **Numbers**: 3cls F1 = 0.421, 6cls F1 = 0.212, acc_3 = 0.573 (matches 07_baselines.py baseline)
- **Lean copy**: `results/parlasent_zs/on_rog_full/`

### 2.2 — ParlaSent-ZS on ParlaSent-SL test (home-domain reference)
- **When**: 2026-11-18 11:57 UTC
- **Status**: DONE
- **Command**:
  ```
  python3 scripts/2_probe_text/04_run_parlasent.py \
      --models parlasent --protocols reg \
      --eval-set data/subset_parlasent_sl_test.jsonl --eval-tag PARLASENT --gpu 3
  ```
- **Result dir**: `runs/batch_LOCAL_PARLASENT_20261118_115717/`
- **Numbers**: 3cls F1 = 0.666, 6cls F1 = 0.351, acc_3 = 0.716
- **Lean copy**: `results/parlasent_zs/on_parlasent/`

### 2.R — Render topography PNGs
- **Status**: TODO
- **Command**:
  ```
  python3 scripts/6_sent_duration/render_parlasent_topography.py
  ```
- **Outputs**:
  - `docs/img/topo_parlasent_zs_on_rog_full.png` (3-D duration × WPS × F1_3)
  - `docs/img/topo_parlasent_zs_on_parlasent.png` (1-D word count × F1_3)

---

## Phase 3 — fine-tune matrix on remote GPU 3

### 3.1 — first cells (S, R_full, SR_full × 3 backbones) [DONE]
- **When**: 2026-11-18 13:29 UTC
- **Status**: DONE
- **Command**:
  ```
  python3 scripts/4_finetune_text/run_matrix.py --gpu 3 --mode full \
      --backbones FacebookAI/xlm-roberta-large classla/xlm-r-parla classla/xlm-r-parlasent \
      --mixtures S R_full SR_full 2>&1 | tee -a runs/finetune_$(date +%F_%H%M).log
  ```
- **Result dir**: `runs/finetune_20261118_132942/`
- **Headlines** (ROG-full test, macro-F1_3):
  - `xlm-r-parlasent × R_full` → **0.469** (best ROG)
  - `xlm-r-parlasent × SR_full` → 0.417
  - `xlm-r-parla × SR_full` → 0.373
  - `xlm-roberta-large × R_full` → 0.276 (collapsed to always-neutral)
  - `xlm-r-parla × R_full` → 0.276 (collapsed)
- **Headlines** (ParlaSent-SL test, macro-F1_3):
  - `xlm-r-parlasent × SR_full` → **0.672**
  - `xlm-r-parla × S` → 0.626
  - `xlm-roberta-large × S` → 0.605
- **Lean copy**: `results/finetune/20261118_132942/`

### 3.2 — SKIPPED (user override)
User chose to move to context fine-tune instead of PR_full / SPR_full — "comparable
to the flash ctx runs" is the more interesting next step. The 4-mixtures-× 3
matrix (R SR PR_full SPR_full) is parked; can be re-queued later.

### 3.3 — context-windowed fine-tune (overnight) ← THE OVERNIGHT RUN
- **When**: 2026-11-18 (launching tonight)
- **Status**: TODO
- **Prep** (one-shot — rebuilds the three JSONLs with pre-rendered windows):
  ```
  python3 scripts/1_data_prep/prep_rog_context.py --W 10
  python3 scripts/1_data_prep/prep_rog_context.py --W 20
  python3 scripts/1_data_prep/prep_rog_context.py --W 30
  ```
  → emits `data/rog_dialog_v2_instance_ctxw{10,20,30}.jsonl`, each 13 658 rows.
- **Launch**:
  ```
  tmux new -s ft
  python3 scripts/4_finetune_text/run_matrix.py --gpu 3 --mode full \
      --backbones FacebookAI/xlm-roberta-large classla/xlm-r-parla classla/xlm-r-parlasent \
      --mixtures R_full_ctxw10 R_full_ctxw20 R_full_ctxw30 \
      2>&1 | tee -a runs/finetune_ctx_$(date +%F_%H%M).log
  ```
- **Cells**: 3 backbones × 3 W's = **9 cells**. Window-rendered text is ~2–3× longer → `max_length` bumped to 256 (W10/20) / 320 (W30) via `mixture_overrides` in `config.json`.
- **Expected elapsed**: ~2 h total (~12–15 min per cell).
- **Analysis**: auto-runs via `run_matrix.py`.
- **Result dir**: `runs/finetune_<stamp>/` (fill in)
- **Lean copy**: `python3 scripts/sync_results.py runs/finetune_<stamp> finetune/<stamp>`
- **Post-land**: table in `docs/4_local_text_probing_finetuning.html` (2026-11-18 Progress section) for the ctx-W sweep, then compare to Flash W=30 lift (+0.093 F1_3 iso→W=30 on 170-subset).

---

## Analysis & docs

### A.1 — results/ sync (after each remote run)
- Copy `matrix_long.csv`, `matrix_report.md`, `heatmap_*.png` from each finished `runs/finetune_*` into a mirror under `results/finetune/<stamp>/`.
- Copy `batch_results.csv`, `batch_meta.json`, `predictions.jsonl` (gzipped if > 1 MB) from OR / parlasent batches into `results/or_phase0/` and `results/parlasent_zs/`.
- Commit the `results/` folder — it's the permanent record, the `runs/` folder is scratch.

### A.2 — Dev-log entry
- **Status**: TODO (defer until Phase 0 OR lands + Phase 3.2 matrix finishes)
- Target: `results_docs/development_log.md`, 2026-11-18 entry.
