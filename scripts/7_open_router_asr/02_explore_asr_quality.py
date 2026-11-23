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
#     display_name: ssh-sent
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 — Explore Chirp 3 ASR quality on ROG
#
# SKELETON. Next session flesh out. Loads a `predictions.jsonl` from
# `data/7_asr_runs/<run_id>/` and plots the histograms + metrics needed to
# validate whether instance-cut ASR is accurate enough to attach to ROG v2.
#
# Target plots:
# - duration histogram (subset vs full ROG test)
# - word count histogram (gold) vs word count histogram (ASR)
# - WER / CER against gold transcripts (jiwer or SloBench evaluator)
# - WER by duration quartile — does WER explode on sub-500 ms clips?
# - transcript diff view: top-20 worst-WER instances, side-by-side
#
# Decision: if WER_full_subset ≤ 0.25, green-light the full ROG transcription.

# %%
import json
from pathlib import Path

HERE = Path.cwd()
if HERE.name != "7_open_router_asr":
    candidate = HERE / "scripts/7_open_router_asr"
    if candidate.exists():
        HERE = candidate
ROOT = HERE.parent.parent

# TODO (next session): point this at the actual 170-subset run once it exists.
RUN_ID = "chirp3_subset170_PLACEHOLDER"
PRED_PATH = ROOT / "data" / "7_asr_runs" / RUN_ID / "predictions.jsonl"


# %% [markdown]
# ## Load predictions

# %%
def load_predictions(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        print(f"  (not yet: {path})")
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


preds = load_predictions(PRED_PATH)
print(f"{len(preds)} predictions loaded")


# %% [markdown]
# ## Duration + word-count histograms

# %%
# TODO: matplotlib histograms. Compare subset to full-ROG (loaded from
# data/processed_jsonl/rog_dialog_v2_instance.jsonl) to confirm the subset is
# representative in distribution shape before green-lighting a full run.

# %% [markdown]
# ## WER / CER per instance

# %%
# TODO: install jiwer; compute WER / CER per row; aggregate overall + by
# duration quartile. SloBench ASR v2 evaluator at
# https://slobench.cjvt.si/ (search for the ASR v2 scorer) may be the
# canonical reference — use it over jiwer if available on PyPI.

# %% [markdown]
# ## Transcript diff view — worst-WER slices

# %%
# TODO: sort by per-instance WER descending; print top-20 as gold/asr/duration
# triples. Look for systematic failure modes (silence-only clips,
# cross-talk, laughter, filler-only clips).

# %% [markdown]
# ## Green-light decision box
#
# WER on 170-subset ≤ 0.25 → proceed to full ROG (~13.6 k instances).
# WER > 0.25 → fall back plan: turn-level ASR (bundle consecutive same-speaker
# utterances, re-split via word timestamps) or file-level ASR with diarization.
