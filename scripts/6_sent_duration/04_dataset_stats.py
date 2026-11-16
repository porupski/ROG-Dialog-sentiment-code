#!/usr/bin/env python3
"""Compute dataset stats: instance counts, label distributions, and duration /
word / WPS min/max/mean/median/std per split, for each test source.

Why: we want the dataset-descriptor header of the report to be real numbers,
not hand-waved. Also exposes clip-stats mismatch between sources — e.g. ROG
2-13 s vs ParlaSpeech 2-30 s — which motivates matched-distribution testing.

Sources handled (if present under data/):
  - rog_dialog_v2_instance.jsonl           ROG-Dialog v2 (dialogue)
  - parlaspeech_v4_si_utterance_instance_full.jsonl   ParlaSpeech-SI (parliament speech)
  - parlasent_sl_instance.jsonl            ParlaSent-SL (parliament text)

Output: misc/dataset_stats.{md,json}
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


LABELS_6 = [
    "predominantlyNegative", "mixedNegative", "neutralNegative",
    "neutralPositive",       "mixedPositive", "predominantlyPositive",
]
LABEL_6_TO_3 = {
    "predominantlyNegative": "negative", "mixedNegative": "negative",
    "neutralNegative": "neutral",        "neutralPositive": "neutral",
    "mixedPositive": "positive",         "predominantlyPositive": "positive",
}
# ParlaSent's own 6-label names (strings) for ParlaSpeech sentiment payload.
# Verified against data/parlaspeech_v4_si_utterance_instance_full.jsonl labels.
PARLASENT6_TO_OURS = {
    "Negative":         "predominantlyNegative",
    "Mixed Negative":   "mixedNegative",
    "Neutral Negative": "neutralNegative",
    "Neutral Positive": "neutralPositive",
    "Mixed Positive":   "mixedPositive",
    "Positive":         "predominantlyPositive",
}

SOURCES: dict[str, dict] = {
    "rog_dia": {
        "path": "data/rog_dialog_v2_instance.jsonl",
        "has_audio": True,
        "family": "dialogue",
    },
    "parlaspeech_si": {
        "path": "data/parlaspeech_v4_si_utterance_instance_full.jsonl",
        "has_audio": True,
        "family": "parliament_speech",
    },
    "parlasent_sl": {
        "path": "data/parlasent_sl_instance.jsonl",
        "has_audio": False,
        "family": "parliament_text",
    },
}


def _summ(values: list[float]) -> dict:
    """Return min/max/mean/median/std/n for a list of values. None-safe."""
    v = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    if not v:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "n": len(v),
        "min": min(v),
        "max": max(v),
        "mean": statistics.fmean(v),
        "median": statistics.median(v),
        "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
    }


def _extract_row(src_name: str, row: dict) -> dict | None:
    """Pull a canonical record {split, label6, duration_sec, word_count}
    from a source-specific row. Returns None if the row has no usable label."""
    out = {"split": row.get("split", "unknown")}
    text = row.get("text") or row.get("sentence") or ""
    out["word_count"] = len(text.split()) if text else 0

    if src_name == "rog_dia":
        labels = row.get("labels") or {}
        out["label6"] = labels.get("sentiment")
        s = row.get("start_t")
        e = row.get("end_t")
        out["duration_sec"] = (float(e) - float(s)) if (s is not None and e is not None) else None
    elif src_name == "parlaspeech_si":
        sent = row.get("sentiment") or {}
        raw = sent.get("ParlaSent_6")
        out["label6"] = PARLASENT6_TO_OURS.get(raw, raw)
        if row.get("audio_length") is not None:
            out["duration_sec"] = float(row["audio_length"])
        else:
            s = row.get("audio_start"); e = row.get("audio_end")
            out["duration_sec"] = (float(e) - float(s)) if (s is not None and e is not None) else None
    elif src_name == "parlasent_sl":
        out["label6"] = row.get("sentiment")
        out["duration_sec"] = None   # text source
    else:
        return None

    if not out["label6"] and src_name != "parlasent_sl":
        return None
    return out


def _wps(rec: dict) -> float | None:
    if not rec.get("duration_sec") or rec["duration_sec"] <= 0:
        return None
    return rec["word_count"] / rec["duration_sec"]


def process_source(src_name: str) -> dict:
    cfg = SOURCES[src_name]
    path = ROOT / cfg["path"]
    out = {
        "name": src_name, "family": cfg["family"], "has_audio": cfg["has_audio"],
        "path": str(path), "exists": path.exists(),
        "per_split": {}, "overall": {},
    }
    if not path.exists():
        return out

    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rec = _extract_row(src_name, r)
            if rec is not None:
                rows.append(rec)

    def _one(group: list[dict]) -> dict:
        label6_counts = Counter(x["label6"] for x in group if x["label6"])
        label3_counts = Counter(LABEL_6_TO_3.get(x["label6"]) for x in group
                                if x["label6"] in LABEL_6_TO_3)
        return {
            "n_instances": len(group),
            "label_dist_6": dict(label6_counts),
            "label_dist_3": dict(label3_counts),
            "word_count": _summ([x["word_count"] for x in group]),
            "duration_sec": _summ([x["duration_sec"] for x in group])
                             if cfg["has_audio"] else None,
            "wps": _summ([_wps(x) for x in group]) if cfg["has_audio"] else None,
        }

    out["overall"] = _one(rows)
    for split in ("train", "dev", "test", "unknown"):
        grp = [x for x in rows if x["split"] == split]
        if grp:
            out["per_split"][split] = _one(grp)
    return out


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _fmt_summ(s: dict, unit: str, nd=2) -> str:
    if not s or s.get("n", 0) == 0:
        return "—"
    return (f"min={_fmt(s['min'], nd)}{unit} max={_fmt(s['max'], nd)}{unit} "
            f"mean={_fmt(s['mean'], nd)}{unit} median={_fmt(s['median'], nd)}{unit} "
            f"std={_fmt(s['std'], nd)}{unit}  (n={s['n']})")


def render_markdown(all_stats: list[dict]) -> str:
    lines = ["# Dataset stats (auto-generated by 04_dataset_stats.py)", ""]
    for s in all_stats:
        lines.append(f"## {s['name']} ({s['family']})")
        if not s["exists"]:
            lines.append(f"- MISSING: `{s['path']}`")
            lines.append("")
            continue
        lines.append(f"- Source file: `{s['path']}`")
        lines.append(f"- Has audio: {s['has_audio']}")
        overall = s["overall"]
        lines.append(f"- Total usable instances: **{overall['n_instances']}**")
        splits = {k: v["n_instances"] for k, v in s["per_split"].items()}
        lines.append(f"- Per split: " + ", ".join(f"{k}={v}" for k, v in splits.items()))
        # Label dist, 6-cls
        lines.append("")
        lines.append("### Label distribution (6-class), overall")
        lines.append("")
        total = overall["n_instances"] or 1
        for lbl in LABELS_6:
            c = overall["label_dist_6"].get(lbl, 0)
            lines.append(f"- `{lbl}`: {c} ({100*c/total:.1f}%)")
        unknown_sum = total - sum(overall["label_dist_6"].get(l, 0) for l in LABELS_6)
        if unknown_sum:
            lines.append(f"- other / unmapped: {unknown_sum} ({100*unknown_sum/total:.1f}%)")
        # Label dist, 3-cls
        lines.append("")
        lines.append("### Label distribution (3-class collapsed), overall")
        lines.append("")
        total3 = sum(overall["label_dist_3"].values()) or 1
        for lbl in ("negative", "neutral", "positive"):
            c = overall["label_dist_3"].get(lbl, 0)
            lines.append(f"- `{lbl}`: {c} ({100*c/total3:.1f}%)")
        # Numeric stats
        lines.append("")
        lines.append("### Numeric stats, overall")
        lines.append("")
        lines.append(f"- word count: {_fmt_summ(overall['word_count'], ' w', 1)}")
        if s["has_audio"]:
            lines.append(f"- duration:   {_fmt_summ(overall['duration_sec'], ' s', 2)}")
            lines.append(f"- WPS:        {_fmt_summ(overall['wps'], ' wps', 2)}")
        # Per-split numeric stats
        lines.append("")
        lines.append("### Numeric stats, per split")
        lines.append("")
        lines.append("| split | n | word count (min/median/mean/max) | duration s (min/median/mean/max) | WPS (min/median/mean/max) |")
        lines.append("|---|---:|---|---|---|")
        for split, ss in s["per_split"].items():
            wc = ss["word_count"]
            dur = ss.get("duration_sec") or {}
            wps = ss.get("wps") or {}
            wc_s = f"{_fmt(wc['min'],0)}/{_fmt(wc['median'],0)}/{_fmt(wc['mean'],1)}/{_fmt(wc['max'],0)}"
            dur_s = (f"{_fmt(dur.get('min'),2)}/{_fmt(dur.get('median'),2)}/{_fmt(dur.get('mean'),2)}/{_fmt(dur.get('max'),2)}"
                     if dur else "—")
            wps_s = (f"{_fmt(wps.get('min'),2)}/{_fmt(wps.get('median'),2)}/{_fmt(wps.get('mean'),2)}/{_fmt(wps.get('max'),2)}"
                     if wps else "—")
            lines.append(f"| {split} | {ss['n_instances']} | {wc_s} | {dur_s} | {wps_s} |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out_dir", type=Path, default=ROOT / "misc",
                   help="where to write dataset_stats.{md,json}")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    for src in SOURCES:
        print(f"[{src}] processing…", flush=True)
        s = process_source(src)
        all_stats.append(s)
        if s["exists"]:
            print(f"  -> {s['overall']['n_instances']} usable rows, "
                  f"splits={list(s['per_split'].keys())}")
        else:
            print(f"  -> MISSING file: {s['path']}")

    md_path = args.out_dir / "dataset_stats.md"
    js_path = args.out_dir / "dataset_stats.json"
    md_path.write_text(render_markdown(all_stats))
    js_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
    print(f"\nwrote {md_path}")
    print(f"wrote {js_path}")


if __name__ == "__main__":
    main()
