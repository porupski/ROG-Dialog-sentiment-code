"""Render sentiment-class composition bars for the three corpora.

Outputs under docs/img/:
  ROG-dia-v2_label_dist_by_duration.png          (6-cls, bands <2s / 2-5s / 5-10s / ≥10s)
  ROG-dia-v2_label_dist_by_duration_3cls.png     (3-cls, same bands)
  ParlaSpeech-SI_label_dist_by_duration.png      (6-cls, bands 1-5 / 5-10 / 10-20 / ≥20s)
  ParlaSpeech-SI_label_dist_by_duration_3cls.png (3-cls, same bands)
  ParlaSent-SL_label_dist_by_words.png           (6-cls, bands <10 / 10-20 / 20-30 / ≥30w)
  ParlaSent-SL_label_dist_by_words_3cls.png      (3-cls, same bands)

ParlaSpeech labels are normalized from "Neutral Positive" → "neutralPositive",
"Neutral" → "neutral", to match the ROG / ParlaSent naming. The shared plotter
lives in utils_rog_explorer.plot_label_dist_by_duration_band, extended here
with a per-dataset feature column override.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "6_sent_duration"))
import utils_rog_explorer as u

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

ROG_PATH = ROOT / "data" / "rog_dialog_v2_instance.jsonl"
PS_PATH  = ROOT / "data" / "parlaspeech_v4_si_utterance_instance_full.jsonl"
PSNT_PATH = ROOT / "data" / "parlasent_sl_instance.jsonl"

SENT_6_TO_3 = {
    "predominantlyNegative": "negative", "mixedNegative": "negative",
    "neutralNegative":       "neutral",  "neutralPositive": "neutral",
    "mixedPositive":         "positive", "predominantlyPositive": "positive",
}
# ParlaSpeech labels — "Neutral Positive" style → camelCase.
PS_6_NORM = {
    "Predominantly Negative": "predominantlyNegative",
    "Mixed Negative":          "mixedNegative",
    "Neutral Negative":        "neutralNegative",
    "Neutral Positive":        "neutralPositive",
    "Mixed Positive":          "mixedPositive",
    "Predominantly Positive":  "predominantlyPositive",
}
PS_3_NORM = {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}


def load_rog_df() -> pd.DataFrame:
    return u.load_rog(ROG_PATH)


def load_parlaspeech_df() -> pd.DataFrame:
    rows = []
    with open(PS_PATH) as f:
        for line in f:
            r = json.loads(line)
            s6_raw = (r.get("sentiment") or {}).get("ParlaSent_6")
            s3_raw = (r.get("sentiment") or {}).get("ParlaSent_3")
            if s6_raw is None or s3_raw is None:
                continue
            s6 = PS_6_NORM.get(s6_raw)
            s3 = PS_3_NORM.get(s3_raw)
            if s6 is None or s3 is None:
                continue
            rows.append({
                "duration":    r.get("audio_length") or 0.0,
                "sentiment":   s6,
                "sentiment_3": s3,
            })
    return pd.DataFrame(rows)


def load_parlasent_df() -> pd.DataFrame:
    rows = []
    with open(PSNT_PATH) as f:
        for line in f:
            r = json.loads(line)
            s = r.get("sentiment")
            s3 = r.get("sentiment_3")
            text = r.get("text") or ""
            if s is None or s3 is None:
                continue
            rows.append({
                "words":       len(text.split()),
                "sentiment":   s,
                "sentiment_3": s3,
            })
    df = pd.DataFrame(rows)
    # duration column so plot_label_dist_by_duration_band can reuse (treat words as duration).
    df["duration"] = df["words"].astype(float)
    return df


def render(df: pd.DataFrame, slug: str, edges, scheme: str, xlabel: str, title: str):
    fig, _dist = u.plot_label_dist_by_duration_band(df, edges=edges, scheme=scheme)
    ax = fig.axes[0]
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.set_size_inches(7.5, 4.0)  # a bit smaller than the 8.5 × 4.8 default
    out_path = OUT / f"{slug}_label_dist_by_{'duration' if 'duration' in xlabel else 'words'}{'_3cls' if scheme == '3' else ''}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(ROOT)}")


def main():
    rog = load_rog_df()
    print(f"ROG rows: {len(rog)}")
    for scheme in ("6", "3"):
        render(rog, "ROG-dia-v2",
               edges=(0, 2, 5, 10, float("inf")),
               scheme=scheme,
               xlabel="duration (s)",
               title=f"ROG v2 — sentiment class composition by duration band ({scheme}-cls)")

    ps = load_parlaspeech_df()
    print(f"ParlaSpeech rows: {len(ps)}")
    for scheme in ("6", "3"):
        render(ps, "ParlaSpeech-SI",
               edges=(0, 5, 10, 20, float("inf")),
               scheme=scheme,
               xlabel="duration (s)",
               title=f"ParlaSpeech-SI — sentiment class composition by duration band ({scheme}-cls, auto labels)")

    psnt = load_parlasent_df()
    print(f"ParlaSent rows: {len(psnt)}")
    for scheme in ("6", "3"):
        render(psnt, "ParlaSent-SL",
               edges=(0, 10, 20, 30, float("inf")),
               scheme=scheme,
               xlabel="word count",
               title=f"ParlaSent-SL — sentiment class composition by word-count band ({scheme}-cls, manual labels)")


if __name__ == "__main__":
    main()
