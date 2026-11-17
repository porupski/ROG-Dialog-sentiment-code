# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ssp
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ROG-Dialog v2 — explorer
#
# Interactive look at ROG-Dialog v2: length / WPS distributions, **F1 topography** (how well does the matrix *perform* across the (duration × WPS) plane, not what label is there), and random listens across duration bands.
#
# **Kernel:** select the `ssp` mamba env in VS Code.
#
# Plot and loader code lives in `utils_rog_explorer.py` so this notebook stays scannable. Nothing is written to disk except the histogram PNG; nothing is retrained.

# %%
import json
import random
import sys
from pathlib import Path

from IPython.display import Audio, Markdown, display

ROOT = Path.cwd()
while ROOT.name and not (ROOT / 'data' / 'rog_dialog_v2_instance.jsonl').exists():
    if ROOT == ROOT.parent:
        raise FileNotFoundError('Could not locate repo root with data/rog_dialog_v2_instance.jsonl')
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'scripts' / '6_sent_duration'))
import utils_rog_explorer as u
print('ROOT =', ROOT)

ROG_PATH   = ROOT / 'data' / 'rog_dialog_v2_instance.jsonl'
MATRIX_DIR = ROOT / 'misc' / 'run_results' / 'finetune_20261113_162724'

random.seed(42)

# %%
df = u.load_rog(ROG_PATH)
cell_preds = u.load_matrix_predictions(MATRIX_DIR)
df.head(3)

# %% [markdown]
# ## Distributions — duration, words, WPS
#
# Also writes `docs/img/ROG-dia-v2_histograms.png` so the dataset-overview page stays in sync.

# %%
fig = u.plot_histograms(df, duration_floor=2.0)
out_png = ROOT / 'docs' / 'img' / 'ROG-dia-v2_histograms.png'
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"<2s: {(df['duration'] < 2).sum()} ({100*(df['duration']<2).mean():.1f}%)")
print(f"2-5s: {((df['duration'] >= 2) & (df['duration'] < 5)).sum()}")
print(f"5-10s: {((df['duration'] >= 5) & (df['duration'] < 10)).sum()}")
print(f">10s: {(df['duration'] >= 10).sum()}")

# %% [markdown]
# ## F1 topography — model performance across duration × WPS
#
# **This is the main view.** Each cell shows the macro-F1 (3-cls) the matrix achieves on ROG test instances falling in that `(duration bin, WPS bin)`. Bins are quantiles on the scored instances (= the matrix's 2 s-filtered test split, n=434). The default below is the **best-cell envelope** — per bin, the single matrix cell with the highest F1 is picked, and its name is shown. This mirrors what the duration-signal study does.
#
# The alternative — pick ONE fixed cell and plot its F1 across bins — is one cell below.

# %%
# Best-cell envelope — per bin, pick the matrix cell with the highest F1_3.
fig = u.plot_f1_topography_2d(df, cell_preds, cell_key=None, scheme='3',
                               n_dur_bins=5, n_wps_bins=5, wps_clip=10.0, mode='quantile')

# %%
# 3-D surface of the same envelope.
fig = u.plot_f1_topography_3d(df, cell_preds, cell_key=None, scheme='3')

# %% [markdown]
# ### F1 topography for a SINGLE cell
#
# Change `cell_key` below to look at any specific (backbone, mixture). Cells in the current matrix follow `<backbone_short>__<mixture>` — e.g. `xlm-r-parla__SPR`, `xlm-r-parlasent__R`.
#
# After the full-ROG rerun, `R_full / SR_full / PR_full / SPR_full` cells (and `rog_dia_full` test rows) will show up here too — just point `MATRIX_DIR` at the new matrix dir and re-run the loader cell.

# %%
CELL = 'xlm-r-parla__SPR'   # change me
fig = u.plot_f1_topography_2d(df, cell_preds, cell_key=CELL, scheme='3')


# %% [markdown]
# ## Context-envelope — F1 vs (context W × target feature)
#
# Does giving Flash/Haiku more conversational context lift macro-F1 above the single-utterance baseline, and does the lift depend on target duration / target word count?
#
# Baseline plane = iso (no-context) F1 averaged across bins, drawn as a transparent gray floor. The signal surface sits above it where context helped; stems from the raw grid points to the floor make the "pierce through" explicit. One plot per target feature (seconds / words) — audio equivalent is a stub until the audio context probe runs.

# %%
ISO_BATCH  = ROOT / "runs" / "batch_FULL_20261111_115520"   # single-utterance Haiku/Flash
CTX_BATCHES = {
    0:  ROOT / "runs" / "batch_CTX_20261117_124601",
    10: ROOT / "runs" / "batch_CTXW10_20261117_140930",
    20: ROOT / "runs" / "batch_CTXW20_20261117_145015",
    30: ROOT / "runs" / "batch_CTXW30_20261117_152331",
    50: ROOT / "runs" / "batch_CTXW50_20261117_155630",
}

# %%
# Flash 0-shot 3-cls envelope — the config with the highest numbers in the sweep.
sweep_flash = u.load_context_sweep(ISO_BATCH, CTX_BATCHES, model="flash", shot="0shot", scheme="3cls")
fig, (_Ws, _edges, _F1, _cnt, _iso) = u.plot_context_envelope_3d(
    sweep_flash, feature="duration", n_bins=5, title_suffix="Flash 0-shot 3-cls")
out = ROOT / "docs" / "img" / "ctx_envelope_flash_duration.png"
out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# %%
fig, _ = u.plot_context_envelope_3d(
    sweep_flash, feature="words", n_bins=5, title_suffix="Flash 0-shot 3-cls")
out = ROOT / "docs" / "img" / "ctx_envelope_flash_words.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# %%
# Haiku 0-shot 3-cls — same view, cheaper model, confirms direction isn't Flash-specific.
sweep_haiku = u.load_context_sweep(ISO_BATCH, CTX_BATCHES, model="haiku", shot="0shot", scheme="3cls")
fig, _ = u.plot_context_envelope_3d(
    sweep_haiku, feature="duration", n_bins=5, title_suffix="Haiku 0-shot 3-cls")
out = ROOT / "docs" / "img" / "ctx_envelope_haiku_duration.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

# %% [markdown]
# **Audio equivalent — pending.** Same shape: x = context W, y = target duration (seconds), z = macro-F1. Blocked on the audio context probe (`scripts/3_probe_audio/`) being wired for ctx windows the way text now is.

# %% [markdown]
# ## Listen to random utterances per duration band
#
# Re-run a cell to draw a new sample. Model-performance grid only populates for instances in the 2 s-filtered ROG test split (n=434).

# %%
def listen(sub_df, n=5, seed=None):
    sampled = u.random_sample(sub_df, n=n, seed=seed)
    print(f"seed = {sampled.attrs.get('seed')}  · pool size = {len(sub_df)}")
    for i, (_, row) in enumerate(sampled.iterrows(), 1):
        display(Markdown(
            f"### {i}. `{row['instance_id']}`  —  **{row['sentiment']}**  (3-cls: {row['sentiment_3']})\n"
            f"- split: `{row['split']}`  · speaker: `{row['speaker']}`\n"
            f"- duration: **{row['duration']:.2f} s**  ·  words: **{row['words']}**  ·  WPS: **{row['wps']:.2f}**\n"
            f"- transcript: *{row['text']}*"
        ))
        wav = ROOT / row['audio_path'] if row['audio_path'] else None
        if wav is not None and wav.exists():
            display(Audio(str(wav)))
        else:
            display(Markdown(f'_audio missing at {wav}_'))
        display(Markdown(u.matrix_grid_md(row['instance_id'], cell_preds)))


# %% [markdown]
# ### `< 2 s`

# %%
listen(df[df['duration'] < 2].reset_index(drop=True), n=5)

# %% [markdown]
# ### `2 – 5 s`

# %%
listen(df[(df['duration'] >= 2) & (df['duration'] < 5)].reset_index(drop=True), n=5)

# %% [markdown]
# ### `5 – 10 s`

# %%
listen(df[(df['duration'] >= 5) & (df['duration'] < 10)].reset_index(drop=True), n=5)

# %% [markdown]
# ### `> 10 s`

# %%
listen(df[df['duration'] >= 10].reset_index(drop=True), n=5)
