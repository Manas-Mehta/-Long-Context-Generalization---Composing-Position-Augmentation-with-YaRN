#!/usr/bin/env python
"""Build the 100-sample selection for the needle-position experiment.

Two-step process:

Step 1 — Rank the 305 multi-entry samples by difficulty using per-sample
predictions from prior evals (y2_base_me + y2_rpe_cur_L16k_me). Tiers:
  Tier 1 HARDEST    — RPE+YaRN failed at any of 32k/64k/128k
  Tier 2 HARD       — YaRN-only failed at 32k+ where RPE got it right
  Tier 3 EASY-LONG  — both models correct at 64k+128k

Step 2 — Build final 100-sample selection:
  80 hard          = all 74 Tier 1 + top 6 Tier 2 (by severity)
  20 single-entry  = random sample from the 694 single-entry indices (seed=42)

Outputs:
  analysis/multi_entry_eval/difficulty_ranking.json
  data/eval_needle/selected_100_indices.json

Run from project root:
  python experiments/babilong/scripts/build_needle_sample_selection.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BINS = ["0k","1k","2k","4k","8k","16k","32k","64k","128k"]
LONG_BINS = ["32k","64k","128k"]

BABILONG_DIR = Path(__file__).parent.parent            # experiments/babilong
EVAL_DIR     = BABILONG_DIR / "data/eval_multi_entry"
BASE_DIR     = BABILONG_DIR / "results/y2_base_me"
RPE_DIR      = BABILONG_DIR / "results/y2_rpe_cur_L16k_me"
TAGS_PATH    = BABILONG_DIR / "data/eval/sample_difficulty_tags.json"
ANALYSIS_DIR = BABILONG_DIR / "analysis/multi_entry_eval"
OUT_NEEDLE   = BABILONG_DIR / "data/eval_needle/selected_100_indices.json"


def rank_difficulty():
    records = defaultdict(lambda: {"per_bin": {}, "meta": {}})

    for b in BINS:
        eval_data  = json.loads((EVAL_DIR / f"{b}.json").read_text())
        base_preds = json.loads((BASE_DIR / f"predictions_{b}.json").read_text())
        rpe_preds  = json.loads((RPE_DIR  / f"predictions_{b}.json").read_text())
        assert len(eval_data) == len(base_preds) == len(rpe_preds), f"Bin {b} length mismatch"
        for i, e in enumerate(eval_data):
            idx = e["original_idx"]
            records[idx]["per_bin"][b] = (base_preds[i]["correct"], rpe_preds[i]["correct"])
            records[idx]["meta"]["object"] = e["object"]
            records[idx]["meta"]["target_entries"] = e["target_entries"]
            records[idx]["meta"]["answer"] = e["answer"]

    tier_hardest, tier_hard, tier_easy_long = [], [], []

    for idx, rec in sorted(records.items()):
        pb = rec["per_bin"]
        missing = [b for b in LONG_BINS if b not in pb]
        if missing:
            continue

        rpe_long_fails    = [b for b in LONG_BINS if not pb[b][1]]
        base_long_fails   = [b for b in LONG_BINS if not pb[b][0]]
        rpe_base_gap_bins = [b for b in LONG_BINS if not pb[b][0] and pb[b][1]]
        total_rpe_fails   = sum(1 for b in BINS if b in pb and not pb[b][1])
        total_base_fails  = sum(1 for b in BINS if b in pb and not pb[b][0])

        entry = {
            "idx": idx,
            "object": rec["meta"]["object"],
            "answer": rec["meta"]["answer"],
            "target_entries": rec["meta"]["target_entries"],
            "rpe_long_fails": rpe_long_fails,
            "base_long_fails": base_long_fails,
            "rpe_base_gap_bins": rpe_base_gap_bins,
            "total_rpe_fails": total_rpe_fails,
            "total_base_fails": total_base_fails,
        }

        if rpe_long_fails:
            tier_hardest.append(entry)
        elif rpe_base_gap_bins:
            tier_hard.append(entry)
        elif pb["128k"][0] and pb["128k"][1] and pb["64k"][0] and pb["64k"][1]:
            tier_easy_long.append(entry)

    tier_hardest.sort(key=lambda x: (-len(x["rpe_long_fails"]), -x["total_rpe_fails"]))
    tier_hard.sort(key=lambda x: (-len(x["rpe_base_gap_bins"]), -x["total_base_fails"]))
    return tier_hardest, tier_hard, tier_easy_long


def build_selection(tier_hardest, tier_hard):
    tags = json.loads(TAGS_PATH.read_text())
    single_entry_indices = sorted(tags["single_entry_indices"])
    per_sample_tags = {t["idx"]: t for t in tags["per_sample"]}

    # 80 hard = all 74 Tier 1 + top 6 Tier 2
    selected = []
    for e in tier_hardest:
        selected.append({**e, "tier": "tier1_hardest"})
    for e in tier_hard[:6]:
        selected.append({**e, "tier": "tier2_hard"})
    assert len([e for e in selected if e["tier"].startswith("tier")]) == 80

    # 20 single-entry reference (deterministic)
    rng = random.Random(42)
    single_sampled = sorted(rng.sample(single_entry_indices, 20))
    for idx in single_sampled:
        t = per_sample_tags[idx]
        selected.append({
            "idx": idx,
            "object": t["obj"],
            "target_entries": t["target_entries"],
            "tier": "single_entry_ref",
        })
    assert len(selected) == 100
    return selected


def main():
    print("=" * 70)
    print("STEP 1: Rank multi-entry samples by difficulty")
    print("=" * 70)
    tier_hardest, tier_hard, tier_easy_long = rank_difficulty()
    print(f"  Tier 1 HARDEST:  {len(tier_hardest):>3} (RPE fails at 32k/64k/128k)")
    print(f"  Tier 2 HARD:     {len(tier_hard):>3} (YaRN-only fails where RPE wins)")
    print(f"  Tier 3 EASY-LONG:{len(tier_easy_long):>3} (both correct at 64k+128k)")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ranking_path = ANALYSIS_DIR / "difficulty_ranking.json"
    ranking_path.write_text(json.dumps({
        "tier1_hardest": tier_hardest,
        "tier2_hard": tier_hard,
        "tier3_easy_long": tier_easy_long,
        "summary": {
            "n_hardest": len(tier_hardest),
            "n_hard": len(tier_hard),
            "n_easy_long": len(tier_easy_long),
        }
    }, indent=2))
    print(f"  -> {ranking_path}")

    print()
    print("=" * 70)
    print("STEP 2: Build 100-sample selection for needle eval")
    print("=" * 70)
    selected = build_selection(tier_hardest, tier_hard)
    obj_counts = Counter(e["object"] for e in selected)
    tier_counts = Counter(e["tier"] for e in selected)
    entries_counts = Counter(e["target_entries"] for e in selected)

    print(f"  Tiers:   {dict(tier_counts)}")
    print(f"  Objects: {dict(obj_counts)}")
    print(f"  target_entries: {dict(entries_counts)}")

    OUT_NEEDLE.parent.mkdir(parents=True, exist_ok=True)
    selection_doc = {
        "n_total": 100,
        "n_hard": 80,
        "n_single_entry_ref": 20,
        "description": (
            "80 hardest multi-entry samples (tier 1 + top tier 2) + "
            "20 random single-entry samples (seed=42) — for recency-bias "
            "needle-position experiment."
        ),
        "selected_indices": sorted([e["idx"] for e in selected]),
        "per_sample": selected,
    }
    OUT_NEEDLE.write_text(json.dumps(selection_doc, indent=2))
    print(f"  -> {OUT_NEEDLE}")


if __name__ == "__main__":
    main()
