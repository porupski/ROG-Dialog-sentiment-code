# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ssh-sent
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 — Explore Chirp 3 ASR quality on ROG
#
# Loads a `predictions.jsonl` from `data/7_asr_runs/<RUN_ID>/`, computes
# WER/CER (raw and normalised) per instance, and renders three triaging
# sections — best N / worst N / random N — each row stacked as
# audio ▸ gold ▸ ASR ▸ metrics for an easy ear-plus-eye scan.
#
# Green-light rule: WER_norm on the 170-subset ≤ ~0.25 → proceed to full ROG.

# %%
import json
import random
import re
import string
from pathlib import Path

import jiwer
from IPython.display import Audio, Markdown, display

# %%
HERE = Path.cwd()
if HERE.name != "7_open_router_asr":
    candidate = HERE / "scripts/7_open_router_asr"
    if candidate.exists():
        HERE = candidate
ROOT = HERE.parent.parent

RUN_ID = "chirp3_subset170_2026-11-23_130456"
PRED_PATH = ROOT / "data" / "7_asr_runs" / RUN_ID / "predictions.jsonl"

N_SHOW = 10      # per-bucket count (best / worst / random)
RANDOM_SEED = 42

# %% [markdown]
# ## Load predictions

# %%
def load_predictions(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        print(f"  (not found: {path})")
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


preds = load_predictions(PRED_PATH)
print(f"{len(preds)} predictions loaded from {PRED_PATH.relative_to(ROOT)}")
n_err = sum(1 for p in preds if p.get("error"))
n_empty = sum(1 for p in preds if not p.get("transcript"))
print(f"  errors: {n_err}   empty transcripts: {n_empty}")

# %% [markdown]
# ## Overlap labelling (annotator-slop fix)
#
# ROG-Dialog is a mono mix. Annotators were imprecise with speaker boundaries,
# so neighbouring speakers' sentiment intervals overlap in time — and the two
# per-interval WAVs end up as the same slice of the mix with *both* voices
# present but *only one* speaker's gold text each. ASR latches onto the louder
# voice, so WER on the shorter / quieter sibling blows up artificially.
#
# Pairwise fix:
# - `no`   — no other-speaker instance intersects this one's time window
# - `won`  — overlaps someone, but this is the longest in its direct-neighbour
#            set (tiebreak: earlier start_t, then speaker lex) → keep in WER
# - `lost` — a longer (or earlier-tie) neighbour exists → drop from WER so we
#            don't double-count the same audio with mismatched gold text
#
# Aggregate WER/CER are reported twice: naive (all) and priority (no+won).

# %%
from collections import Counter

ROG_JSONL = ROOT / "data" / "rog_dialog_v2_instance.jsonl"

def load_full_rog(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (not found; overlap labels will be 'unknown'): {path}")
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def compute_overlap_labels(rog: list[dict]):
    """Pairwise overlap label + overlap_frac per instance_id. See markdown above."""
    by_file: dict[str, list[dict]] = {}
    for r in rog:
        by_file.setdefault(r["file_id"], []).append(r)
    labels, fracs = {}, {}
    for rows in by_file.values():
        rows_sorted = sorted(rows, key=lambda r: (r["start_t"], r.get("speaker", "")))
        for r in rows_sorted:
            st, et, spk = r["start_t"], r["end_t"], r.get("speaker")
            dur = max(et - st, 1e-9)
            neigh = []
            total_overlap = 0.0
            for s in rows_sorted:
                if s is r or s.get("speaker") == spk:
                    continue
                if s["end_t"] <= st or s["start_t"] >= et:
                    continue
                o = min(et, s["end_t"]) - max(st, s["start_t"])
                if o > 0:
                    neigh.append(s)
                    total_overlap += o
            fracs[r["instance_id"]] = min(total_overlap / dur, 1.0)
            if not neigh:
                labels[r["instance_id"]] = "no"
                continue
            def _key(x):  # longest dur first, then earliest start, then speaker
                return (-(x["end_t"] - x["start_t"]), x["start_t"], x.get("speaker", ""))
            contenders = sorted([r] + neigh, key=_key)
            labels[r["instance_id"]] = "won" if contenders[0] is r else "lost"
    return labels, fracs


rog = load_full_rog(ROG_JSONL)
overlap_labels, overlap_fracs = compute_overlap_labels(rog)
print(f"overlap labels over full ROG v2 ({len(rog)} instances):")
full_counts = Counter(overlap_labels.values())
for k in ("no", "won", "lost"):
    n = full_counts.get(k, 0)
    pct = 100 * n / max(len(rog), 1)
    print(f"  {k:4s}: {n:6d}  ({pct:4.1f}%)")

for p in preds:
    p["overlap"] = overlap_labels.get(p["instance_id"], "unknown")
    p["overlap_frac"] = overlap_fracs.get(p["instance_id"], float("nan"))

sub_counts = Counter(p["overlap"] for p in preds)
print("\noverlap labels in the subset:")
for k in ("no", "won", "lost", "unknown"):
    if sub_counts.get(k):
        print(f"  {k:7s}: {sub_counts[k]:3d}  ({100*sub_counts[k]/len(preds):4.1f}%)")

# %% [markdown]
# ## Text normalisation + per-row WER/CER
#
# Raw metrics compare strings as returned. Normalised metrics lowercase,
# strip punctuation (including Slovenian quotes and the ellipsis char),
# collapse whitespace. The normalised number is the one to judge against
# the 0.25 green-light threshold; raw stays close to it for clean inputs
# but inflates on comma/period/case-only differences.

# %%
_PUNCT = string.punctuation + "„“”‚‘’«»…–—"
_PUNCT_RE = re.compile(f"[{re.escape(_PUNCT)}]")

def normalise(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_wer_cer(gold: str, asr: str):
    gold = gold or ""
    asr = asr or ""
    if not gold.strip() and not asr.strip():
        return 0.0, 0.0
    if not gold.strip():
        return float("nan"), float("nan")
    try:
        return jiwer.wer(gold, asr), jiwer.cer(gold, asr)
    except Exception:
        return float("nan"), float("nan")


for p in preds:
    g, a = p.get("text_gold") or "", p.get("transcript") or ""
    p["gold_norm"] = normalise(g)
    p["asr_norm"]  = normalise(a)
    p["wer_raw"], p["cer_raw"]   = safe_wer_cer(g, a)
    p["wer_norm"], p["cer_norm"] = safe_wer_cer(p["gold_norm"], p["asr_norm"])

# %% [markdown]
# ## Aggregate numbers

# %%
scored = [p for p in preds if not p.get("error") and (p.get("text_gold") or "").strip()]
golds       = [p["text_gold"] for p in scored]
asrs        = [p["transcript"] or "" for p in scored]
golds_norm  = [p["gold_norm"] for p in scored]
asrs_norm   = [p["asr_norm"]  for p in scored]

wer_raw_micro  = jiwer.wer(golds, asrs)
cer_raw_micro  = jiwer.cer(golds, asrs)
wer_norm_micro = jiwer.wer(golds_norm, asrs_norm)
cer_norm_micro = jiwer.cer(golds_norm, asrs_norm)

total_cost = sum(((p.get("raw_response") or {}).get("usage") or {}).get("cost", 0) or 0
                 for p in preds)
total_latency = sum(p.get("latency_s") or 0 for p in preds)

print(f"n scored         : {len(scored)} / {len(preds)}")
print(f"WER  raw   (all) : {wer_raw_micro:.3f}")
print(f"WER  norm  (all) : {wer_norm_micro:.3f}")
print(f"CER  raw   (all) : {cer_raw_micro:.3f}")
print(f"CER  norm  (all) : {cer_norm_micro:.3f}")

# priority aggregate: drop "lost" to avoid double-counting overlap-sibling audio
prio = [p for p in scored if p.get("overlap") in ("no", "won")]
if prio and len(prio) < len(scored):
    pg  = [p["text_gold"] for p in prio]
    pa  = [p["transcript"] or "" for p in prio]
    pgn = [p["gold_norm"] for p in prio]
    pan = [p["asr_norm"]  for p in prio]
    print(f"\nn priority       : {len(prio)}  ({len(scored)-len(prio)} 'lost' dropped)")
    print(f"WER  raw   (prio): {jiwer.wer(pg, pa):.3f}")
    print(f"WER  norm  (prio): {jiwer.wer(pgn, pan):.3f}   <-- honest number (judge against SloBench WER2=0.303)")
    print(f"CER  raw   (prio): {jiwer.cer(pg, pa):.3f}")
    print(f"CER  norm  (prio): {jiwer.cer(pgn, pan):.3f}")

print(f"\ntotal OR cost    : ${total_cost:.4f}")
print(f"total latency    : {total_latency:.1f} s   avg {total_latency/max(len(preds),1):.2f} s/row")

# %% [markdown]
# ## Overlap rollup — hours / turns / words by bucket
#
# Hours = sum of instance durations. Turns = instance count. Words = gold-text
# word count. Per-bucket shares let us see how much data each label actually
# accounts for — the categorical can look innocent (small % of turns) while
# still eating a large chunk of hours if lost rows tend to be long.
# `overlap_frac_time` = sum(overlap_frac·dur) / sum(dur); the time-weighted
# mean fraction of audio that is actually crosstalk.

# %%
def _wc(s): return len((s or "").split())

rollup_scope = preds  # same scope as the WER aggregates above
total_hours = sum((p.get("duration_s") or 0) for p in rollup_scope) / 3600
total_turns = len(rollup_scope)
total_words = sum(_wc(p.get("text_gold")) for p in rollup_scope)
oframe_num = sum((p.get("overlap_frac") or 0) * (p.get("duration_s") or 0) for p in rollup_scope)
oframe_den = sum((p.get("duration_s") or 0) for p in rollup_scope) or 1.0
overlap_frac_time_mean = oframe_num / oframe_den

print(f"TOTAL: {total_hours:.3f} h   {total_turns} turns   {total_words} words")
print(f"       time-weighted mean overlap_frac = {overlap_frac_time_mean:.3f}")
print()
print(f"{'bucket':8s} {'turns':>6s} {'%t':>6s} {'hours':>8s} {'%h':>6s} "
      f"{'words':>7s} {'%w':>6s} {'ovfr̄':>7s}")
for k in ("no", "won", "lost", "unknown"):
    rows = [p for p in rollup_scope if p.get("overlap") == k]
    if not rows: continue
    h = sum((p.get("duration_s") or 0) for p in rows) / 3600
    w = sum(_wc(p.get("text_gold")) for p in rows)
    ovfr = [p["overlap_frac"] for p in rows if p.get("overlap_frac") == p.get("overlap_frac")]
    mean_of = sum(ovfr)/len(ovfr) if ovfr else float("nan")
    print(f"{k:8s} {len(rows):>6d} {100*len(rows)/total_turns:>5.1f}% "
          f"{h:>8.3f} {100*h/max(total_hours,1e-9):>5.1f}% "
          f"{w:>7d} {100*w/max(total_words,1):>5.1f}% "
          f"{mean_of:>7.3f}")

# %% [markdown]
# ## Histograms — duration, latency, cost, per-row WER

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
durations = [p.get("duration_s") or 0 for p in preds]
latencies = [p.get("latency_s") or 0 for p in preds]
costs = [((p.get("raw_response") or {}).get("usage") or {}).get("cost", 0) or 0 for p in preds]
wers_norm = [p["wer_norm"] for p in scored if p["wer_norm"] == p["wer_norm"]]  # drop NaN

axes[0, 0].hist(durations, bins=30); axes[0, 0].set_title("duration (s)")
axes[0, 1].hist(latencies, bins=30); axes[0, 1].set_title("OR latency (s)")
axes[1, 0].hist(costs, bins=30); axes[1, 0].set_title("OR cost ($)")
axes[1, 1].hist(wers_norm, bins=30); axes[1, 1].set_title("WER norm per row")
axes[1, 1].axvline(0.25, color="red", linestyle="--", label="green-light 0.25")
axes[1, 1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Triage render — best / worst / random
#
# Each row: **audio widget → gold → ASR → metrics**, stacked vertically for
# easy ear-plus-eye comparison. Click play, read the two lines, judge where
# Chirp 3 goes wrong.

# %%
def render_row(idx: int, p: dict, bucket: str) -> None:
    audio_fs = ROOT / p["audio_path"]
    ov = p.get("overlap", "?")
    ovf = p.get("overlap_frac", float("nan"))
    ovf_str = f" ({ovf:.2f})" if ovf == ovf else ""
    header = (
        f"**[{bucket} #{idx+1}]** `{p['instance_id']}` · "
        f"dur={p['duration_s']}s · overlap=**{ov}**{ovf_str} · "
        f"WER={p['wer_norm']:.3f} (raw {p['wer_raw']:.3f}) · "
        f"CER={p['cer_norm']:.3f} (raw {p['cer_raw']:.3f}) · "
        f"lat={p.get('latency_s','?')}s"
    )
    display(Markdown("---"))
    display(Markdown(header))
    if audio_fs.exists():
        display(Audio(filename=str(audio_fs)))
    else:
        display(Markdown(f"_audio missing on disk: `{audio_fs}`_"))
    display(Markdown(f"**gold :** {p.get('text_gold') or '_∅_'}"))
    display(Markdown(f"**asr  :** {p.get('transcript') or '_∅_'}"))


def render_bucket(rows: list[dict], bucket: str) -> None:
    display(Markdown(f"## {bucket} — {len(rows)} rows"))
    for i, p in enumerate(rows):
        render_row(i, p, bucket)


# order the scored predictions by normalised WER; NaN → +inf so they sink
scored_sorted = sorted(scored, key=lambda p: (float("inf") if p["wer_norm"] != p["wer_norm"]
                                              else p["wer_norm"]))
best_rows   = scored_sorted[:N_SHOW]
worst_rows  = [p for p in reversed(scored_sorted) if p["wer_norm"] == p["wer_norm"]][:N_SHOW]
rng = random.Random(RANDOM_SEED)
random_rows = rng.sample(scored, min(N_SHOW, len(scored)))

# priority-only view: WORST among the rows that actually count toward the honest WER
scored_prio = [p for p in scored_sorted if p.get("overlap") in ("no", "won")
               and p["wer_norm"] == p["wer_norm"]]
worst_prio_rows = list(reversed(scored_prio))[:N_SHOW]

# %% [markdown]
# ### Best N (lowest WER_norm)

# %%
render_bucket(best_rows, "BEST")

# %% [markdown]
# ### Worst N (highest WER_norm, all instances)

# %%
render_bucket(worst_rows, "WORST")

# %% [markdown]
# ### Worst N — priority only (no + won)
#
# These are the worst-WER instances **after** dropping overlap-lost rows —
# i.e. the ones that actually hurt the honest number. If the WORST section
# above is all `overlap=lost`, this section tells you what's left to fix.

# %%
render_bucket(worst_prio_rows, "WORST_PRIO")

# %% [markdown]
# ### Random N (seed-fixed)

# %%
render_bucket(random_rows, "RANDOM")

# %% [markdown]
# ## Green-light decision box
#
# Judge WER_norm **(prio)** against SloBench WER2=0.303 (hard-set register
# closest to ROG). WER1=0.067 is the clean-set number — too easy to compare.
#
# - WER_norm (prio) ≲ 0.30 → proceed to full ROG (~13.6 k instances) on remote.
# - Otherwise → regroup. Options: BSS (sepformer) on the mix, pyannote-gated
#   masking of non-target speakers, or accept and report.
