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
- **Status**: DONE
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
- **Status**: DONE (per Ivan 2026-11-18 evening)
- **Command** (real — strip `--test --dry-run` from the smoke variant you ran earlier):
  ```
  python scripts/2_probe_text/03_run_openrouter_text_context.py \
      --models flash --shots 0shot --context-words 30 \
      --eval-set data/subset_rog_test_full.jsonl --eval-tag FULLROG
  ```
- **Result dir**: `runs/batch_FULLROG_CTXW30_<stamp>/` (remote; sync pending — numbers not in doc 2 yet)
- **Analyze**: `python3 scripts/analyze_probe.py runs/batch_FULLROG_CTXW30_<stamp>`
- **Lean copy**: `python3 scripts/sync_results.py runs/batch_FULLROG_CTXW30_<stamp> or_phase0/ctxw30`
- **Post-land**: fill in Report 2 (`docs/2_open_router_probing_text.html` → the 2026-11-18 Progress section I added) with the full-ROG iso vs W=30 F1 numbers.

### 0.3 — overnight context sweep (7 cells)
- **When**: 2026-11-18 evening (queued)
- **Status**: TODO (paste inside an open tmux pane at the repo root)
- **Command** (one liner, paste into tmux on `ssp-cuda`):
  ```
  bash scripts/2_probe_text/run_fullrog_ctx_sweep.sh
  ```
  The wrapper runs seven cells sequentially (safer than parallel re: Flash rate limits):
  - Flash 0shot **3cls** W=10 / W=20 / W=40  (W=30 already in `batch_FULLROG_CTXW30_20261118_165459`)
  - Flash 0shot **6cls** W=10 / W=20 / W=30 / W=40  (full-sweep + prompt-sensitivity leg)
- **Est. cost**: ~$17.2 (3cls leg ~$5.3, 6cls leg ~$11.9). Within $20 budget.
- **Est. time**: ~15–16 h total sequential.
- **Logs**: `runs/_sweep_logs/fullrog_ctx_<stamp>/{_master.log, <cell>.log}` — per-cell logs captured even on failure; master log for quick sanity check.
- **Post-land**: fill in Report 2's queued rows + a 6cls F1 vs W curve; decide whether to run §0.4 (fshot) based on the 0shot peak.

### 0.4 — fshot leg: 3cls+6cls iso + 3cls+6cls W=20 (decision after 0.3)
- **When**: 2026-11-19 (pending 0.3 results)
- **Status**: TODO
- **Plan**: four fshot cells — `{3cls, 6cls} × {iso, W=20}` — to tell us whether few-shot stacks on context in the natural distribution. Est. cost ~$10 (one batch per cell; fshot roughly 1.5× the 0shot cost).
- **Decision gate**: only run if 0.3 shows 0shot F1_3 > 0.60 at some W (so fshot has room to lift) AND no scheme sensitivity (so 6cls add is worth probing).

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

---

## Phase 4 — audio fine-tune matrix (SSP chapter 4)

Audio side of the matrix. Engine lives in the sibling repo
`slavic-speech-pipeline/4_instance_models/` and is invoked through
`43_run_task_list.py` (sequential queue; per-task output dirs under
`runs/<list_dir>/<task_label>/`; aggregate CSV + PNG at the list root).

**Two SSP files changed for Phase 4.1** (both need to land on the remote
machine before launch):

- `slavic-speech-pipeline/4_instance_models/utils_instance_train.py` — the four
  `rog_dia_*` TARGETS entries now point at `data/processed_jsonl/rog_dialog_v2_instance.jsonl`
  (the v2 file that's already present in SSP; the former `rog_dialog_instance.jsonl`
  path was stale and did not exist on disk).
- `slavic-speech-pipeline/4_instance_models/43_run_task_list.py` — TASKS list
  swapped for three iso audio-sentiment benchmarks on `rog_dia_sentiment_3`.
  Previous TASKS list (sentiment3 + DA dimension/function cells) is preserved
  as a commented block for future rerun.

### 4.1 — iso benchmarks on `rog_dia_full` (three backbones) — v1 landed

Three backbones × ROG-Dialog v2 full train/dev/test × no context window.
Train recipe: SSP default (`freeze_feature_encoder=true`, encoder trainable,
head from scratch), 10 epochs, backbone-matched LR, `class_weighting=none`.

- **When**: 2026-11-20
- **Status**: DONE (2 cells completed, 1 OOM)
- **Result dir**: `slavic-speech-pipeline/runs/senthunt_audio_iso_v1/`
- **Headline (ROG-full TEST, macro-F1_3)**:
  - `iso_w2v2-base`  → **0.276** (collapsed to majority-neutral; majority baseline = 0.278)
  - `iso_xls-r-300m` → **0.292** (barely above majority baseline)
  - `iso_w2v-bert-2` → OOM at `batch_size=32` on 40 GB GPU after 34 s
- **Wall time**: 970 s (base) + 1706 s (xls-r) + 34 s (OOM) ≈ 45 min
- **Read**: with `class_weighting=none`, the 72 % neutral prior dominates; the
  3-class head converges on all-neutral. Any audio modality lift has to come
  from recipes that break this collapse. Phase 4.2 moves to sqrt-inverse
  weighting, higher LR, and longer training.
- **Prep** (on the remote host, in the `ssp-cuda` conda env, from the SSP
  repo root):
  ```
  mamba activate ssp-cuda
  cd 4_instance_models

  # Sanity-check target wiring + split sizes before committing to a 2–4 h run:
  python -c "
  import utils_instance_train as uit
  from utils_instance_train import TARGETS, load_split
  t = TARGETS['rog_dia_sentiment_3']
  print(t['jsonl_path'])
  for sp in ('train', 'dev', 'test'):
      n = len(load_split(t['jsonl_path'], sp, t['label_key'], 15.0,
                        label_remap=t.get('label_remap')))
      print(f'  {sp}: {n}')
  "
  # Expected: ~11.1k train, ~0.6k dev, 2233 test (ROG-Dialog v2 unfiltered).
  ```
- **Launch** (in tmux on the remote host; GPU id read from
  `4_instance_models/4_config.json → shared.reserved_gpu`, edit there if the
  reserved GPU on the target cluster is not 2):
  ```
  tmux new -s audio_iso
  cd 4_instance_models
  python 43_run_task_list.py \
      --mode full --use_gpu \
      --list_dir senthunt_audio_iso_v1 \
      2>&1 | tee ../runs/43_senthunt_audio_iso_$(date +%Y-%m-%d_%H%M%S).log
  ```
  Dry-run first if you want to confirm the planned tasks:
  ```
  python 43_run_task_list.py --mode full --dry_run
  ```
- **Tasks** (3 cells, sequential):
  | label                           | model                          | lr    |
  |---------------------------------|--------------------------------|-------|
  | `iso_w2v2-base_rog_dia_full`    | `facebook/wav2vec2-base`       | 1e-5  |
  | `iso_xls-r-300m_rog_dia_full`   | `facebook/wav2vec2-xls-r-300m` | 3e-5  |
  | `iso_w2v-bert-2_rog_dia_full`   | `facebook/w2v-bert-2.0`        | 3e-5  |
- **Est. time**: ~2–4 h total (base ~30 min, XLS-R-300M ~1 h, w2v-BERT 2.0 ~1.5 h,
  single reserved GPU). Base config has `max_duration_s=15.0 s` which covers
  ROG-Dialog v2's 12.3 s max.
- **Result dir**: `slavic-speech-pipeline/runs/senthunt_audio_iso_v1/`
  with `task_list_summary.csv` + per-task subdirs.
- **Lean copy**: `python3 scripts/sync_results.py <ssp_runs_dir> audio_iso/v1`
  (adapt or write a thin wrapper if `sync_results.py` is Sentiment_Search-specific).
- **Post-land**: Render a 3-row table + a 2-cell panel (3-class confusion
  heatmaps) at the top of Report 5. Target metric: macro-F1_3 on TEST.
  Target to beat: **0.469** (XLM-R-parlasent × R_full iso text-side lift,
  Report 4). If any audio backbone exceeds this, we've shown a modality lift.

### 4.2 — iso recipe sweep: sqrt-inverse × LR × epochs (9 cells)

Phase 4.1 collapse → 3-recipe grid to break the majority-neutral attractor.
All 9 cells train on `rog_dia_full` with `class_weighting=sqrt_inverse`.

- **When**: 2026-11-20 (queued)
- **Status**: TODO
- **TASKS now in** `slavic-speech-pipeline/4_instance_models/43_run_task_list.py`
  (replaces the Phase 4.1 cw=none block; previous iso cw=none block preserved
  below as history).
- **Recipes**:
  - **R1** — sqrt, default LR, 10 epochs (sanity check: does weighting alone break collapse?)
  - **R2** — sqrt, 3× default LR, 10 epochs (`wav2vec2-base`→3e-5, `xls-r-300m`→1e-4, `w2v-bert-2.0`→1e-4)
  - **R3** — sqrt, default LR, 20 epochs (longer anneal)
- **OOM fix for `w2v-bert-2.0`** (all three recipes): `batch_size=8`,
  `grad_accum=4` (effective batch 32). The 40 GB cluster GPU can't hold
  `w2v-bert-2.0` at BS=32; this halves resident batch and compensates with
  gradient accumulation.
- **Launch**:
  ```
  tmux new -s audio_iso_v2
  cd 4_instance_models
  python 43_run_task_list.py \
      --mode full --use_gpu \
      --list_dir senthunt_audio_iso_v2 \
      2>&1 | tee ../runs/43_senthunt_audio_iso_v2_$(date +%Y-%m-%d_%H%M%S).log
  ```
- **Est. time**: base cells ~16 min, xls-r cells ~28 min, w2v-bert-2 cells
  ~45 min (BS=8 ↑ time), R3 (20 epochs) ~2× the 10-epoch cells.
  Rough total ~6 h.
- **Result dir**: `slavic-speech-pipeline/runs/senthunt_audio_iso_v2/`
- **Read**: compare R1 vs R2 vs R3 per backbone to decide which knob matters.
  Target: beat the 0.469 local-text ceiling on any (backbone, recipe) cell.
  If sqrt alone lifts F1_3 above ~0.35 for all three backbones, context-window
  Phase 4.3 is next; if all three stay stuck near the majority baseline, we
  need to look at freezing/unfreezing the feature extractor, warmup length,
  or dropping the regression-head recipe in favour of a MSE-on-ordinal variant.

### 4.3 — mixture matrix, freeze probes, regression axis (27 cells, three stages)

**When**: 2026-11-21 (queued for weekend)
**Status**: QUEUED — task list in `slavic-speech-pipeline/4_instance_models/task_list.json`
**Launch**:
```
tmux new -s audio_p43
cd 4_instance_models
python -u 43_run_task_list.py \
    --mode full --use_gpu \
    --list_dir senthunt_audio_p43 \
    2>&1 | tee ../runs/43_senthunt_audio_p43_$(date +%Y-%m-%d_%H%M%S).log
```

**Prereqs (run once, local or remote, before launching)**:

1. **w2v-BERT collator patch** has landed in `utils_instance_train.py`
   (`DataCollatorForInstance.__call__` now branches on `ndim == 2` and pads
   log-Mel features manually — HF base `SequenceFeatureExtractor._pad` was
   1-D-only). Smoke-test locally on CPU first:
   ```
   python -u 43_run_task_list.py --limit 1 --mode test --dry_run
   ```
2. **ParlaSpeech-SI subset JSONLs** — 11c's full
   `parlaspeech_v4_si_utterance_instance.jsonl` needs to be subset to 200h
   and 500h caps, plus two ROG-merged files. Launch from the SSP repo root
   (or 1_data_prep/ — relative paths resolve against both):
   ```
   cd slavic-speech-pipeline
   python 1_data_prep/11c_prep_subset_parlaspeech.py --target_hours 200 --name 200h \
       --merge_with data/processed_jsonl/rog_dialog_v2_instance.jsonl \
       --merge_tag rog_plus
   python 1_data_prep/11c_prep_subset_parlaspeech.py --target_hours 500 --name 500h \
       --merge_with data/processed_jsonl/rog_dialog_v2_instance.jsonl \
       --merge_tag rog_plus
   ```
   Emits `parlaspeech_v4_si_subset_{200h,500h}.jsonl` +
   `rog_plus_merged_{200h,500h}.jsonl` under
   `data/processed_jsonl/`. The merged files carry a unified `sentiment_f`
   float on [0, 5] (ROG strings mapped via ordinal, PS via its native
   `sentiment_6`) — consumed by the four `*_sentiment_regr_*` TARGETS
   entries added to `utils_instance_train.py`.

**Matrix (27 cells)**:

- **Phase 0** (3 cells, w2v-BERT classification on ROG-only, R1/R2/R3
  recipes from REF-2026-11-20) — patch-verification runs; if any still
  OOM/ValueError, halt and retry the collator fix before the rest.
- **Stage A** (12 cells, regression on `sentiment_f`): 2 regression-capable
  backbones (wav2vec2-base, xls-r-300m) × 3 mixtures (R_full, P_200h,
  PR_200h) × 2 freeze (F0=conv-only, F1=head-only via `freeze_encoder=true`).
- **Stage B** (6 cells, mixed): 3 longer-training variants on expected
  Stage-A leaders (xls-r/w2v2 × PR_200h × 20 epochs, xls-r PR_200h at
  high-LR) + 3 classification probes (w2v-BERT F1 + xls-r / w2v2 F0 at
  20 epochs on R_full for direct Phase-0 comparison).
- **Stage C** (6 cells, regression): xls-r × {P_500h, PR_500h × {F0, F1,
  20e}, PR_200h/PR_500h at seed=5678 for variance check}.

Stage B's picks are currently generic (longest-LR / longest-anneal probes
against the expected leaders); after Stage A completes, open
`task_list.json` and swap the Stage B / C backbone picks if a different
backbone tops the board.

**Est. time**: ~45–55 h cumulative (overnight × 2–3).

**Guardrails recap**:
- w2v-BERT runs classification only (SSP's `Wav2Vec2ForRegression` hard-fails
  on non-wav2vec2 backbones; writing a `Wav2Vec2BertForRegression` mirror is
  deferred — see handoff for rationale).
- All regression cells report Spearman on the raw float and (post-hoc, in
  analysis) on 3-cls bins via thresholds 1.5 and 3.5.
- Task list is the live `task_list.json`; the pre-v2 queues live on disk as
  `task_list_REF-2026-11-20.json` (audio iso v2) and
  `task_list_REF-2026-11-09.json` (pre-v2 ROG-Dialog).

### 4.4 — context-window sweep (W ∈ {10, 20, 30, 40})

**Status**: DESIGN PENDING. The handover notes two plausible implementations:
- (A) Concatenate neighbor-clip audio into the input waveform.
- (B) Pass the window transcript as a text side-channel.

SSP's chapter-4 trainer accepts audio only, so (A) is the native path but
raises its own questions (concat only trailing context? bidirectional? gap
handling?). (B) would need a new model class (dual encoder + fusion head).

Defer until 4.1 lands and we have a reference F1 to lift above.

### 4.R — Render audio envelope plots
- **Status**: TODO (post-4.3)
- Mirror of Report 2's Flash envelopes:
  - `scripts/6_sent_duration/render_audio_ctx_envelopes_fullrog.py` (new;
    copy from `render_ctx_envelopes_fullrog.py`, swap data source to the audio
    matrix outputs).
  - Red-green diverging cmap anchored at the iso baseline (per memory
    `feedback-plotting-conventions`). Duration axis is in scope for Report 5
    (unlike text-side reports).
  - Output: `docs/img/ctx_envelope_<backbone>_<train>_<axis>_FULLROG.png`.
