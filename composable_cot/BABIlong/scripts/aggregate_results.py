#!/usr/bin/env python3
"""Aggregate eval results across all conditions into a single table.

Usage:
    python composable_cot/BABIlong/scripts/aggregate_results.py \
        --results-dir composable_cot/BABIlong/results/

Reads summary.json from each condition subdirectory and prints:
  1. Accuracy table (conditions x bins)
  2. Retention scores (acc at long bins / acc at short bin)
  3. Missing/incomplete conditions
"""

import argparse
import json
import os
from pathlib import Path

BINS_ORDERED = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]

# Conditions in display order, with labels
CONDITIONS = [
    ("lora_base",        "LoRA base",      False),
    ("rpe_only",         "RPE only",       False),
    ("pose_only",        "PoSE only",      False),
    ("y2_base",          "YaRN base",      True),
    ("y2_pose_32k",      "YaRN+PoSE",      True),
    ("y2_rpe_cur_L16k",  "YaRN+RPE-cur",   True),
]

RANDOM_CHANCE = 1 / 6  # 6-class QA3


def load_summary(results_dir: str, condition: str) -> dict | None:
    path = os.path.join(results_dir, condition, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def retention(accs: dict, ref_bin: str = "0k") -> float | None:
    """Avg accuracy at 32k/64k/128k relative to ref_bin."""
    ref = accs.get(ref_bin)
    if not ref or ref < 0.01:
        return None
    long_bins = ["32k", "64k", "128k"]
    vals = [accs[b] for b in long_bins if b in accs and accs[b] is not None]
    if not vals:
        return None
    return sum(vals) / len(vals) / ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="composable_cot/BABIlong/results/")
    args = p.parse_args()

    # Load all summaries
    data = {}
    for cond_key, _, _ in CONDITIONS:
        summary = load_summary(args.results_dir, cond_key)
        if summary:
            # summary.json format: {"condition": ..., "bins": {"0k": {"accuracy": 0.xx, "n": N}, ...}}
            # OR flat: {"0k": 0.xx, ...}
            # Handle both
            if "bins" in summary:
                data[cond_key] = {b: v["accuracy"] for b, v in summary["bins"].items()}
            else:
                # flat format — filter to known bins
                data[cond_key] = {b: summary[b] for b in BINS_ORDERED if b in summary}

    if not data:
        print(f"No results found in: {args.results_dir}")
        print("Conditions expected:")
        for cond_key, label, _ in CONDITIONS:
            path = os.path.join(args.results_dir, cond_key, "summary.json")
            print(f"  {path}")
        return

    # --- Accuracy Table ---
    print("=" * 90)
    print("BABILong QA3 — EVAL RESULTS (epoch-1 checkpoints, 100 samples/bin)")
    print("=" * 90)

    # Header
    header = f"{'Condition':<18}" + "".join(f"{b:>7}" for b in BINS_ORDERED) + f"{'Retain':>9}"
    print(header)
    print("-" * len(header))

    for cond_key, label, uses_yarn in CONDITIONS:
        if cond_key not in data:
            print(f"  {label:<16}  (not ready)")
            continue

        accs = data[cond_key]
        row = f"{label:<18}"
        for b in BINS_ORDERED:
            v = accs.get(b)
            if v is None:
                row += f"{'—':>7}"
            elif v <= RANDOM_CHANCE + 0.03:
                row += f"{'~rnd':>7}"   # near random
            else:
                row += f"{v:>7.1%}"

        ret = retention(accs)
        row += f"  {ret:.0%}" if ret is not None else f"{'—':>7}"
        print(row)

    print()

    # --- Which conditions are still running ---
    missing = [label for cond_key, label, _ in CONDITIONS if cond_key not in data]
    if missing:
        print(f"Still running / not ready: {', '.join(missing)}")
        print()

    # --- Best per bin ---
    print("Best condition per bin:")
    for b in BINS_ORDERED:
        best_val = -1
        best_label = "—"
        for cond_key, label, _ in CONDITIONS:
            v = data.get(cond_key, {}).get(b)
            if v is not None and v > best_val:
                best_val = v
                best_label = label
        if best_val > RANDOM_CHANCE:
            print(f"  {b:>5}: {best_label:<18} ({best_val:.1%})")

    print()

    # --- Collapse detection ---
    print("Collapse check (any bin near random chance):")
    for cond_key, label, _ in CONDITIONS:
        if cond_key not in data:
            continue
        accs = data[cond_key]
        collapsed_bins = [b for b in ["0k", "4k", "8k"] if accs.get(b, 1.0) <= RANDOM_CHANCE + 0.05]
        if collapsed_bins:
            print(f"  {label}: COLLAPSED at {collapsed_bins}")
        else:
            print(f"  {label}: OK (short-context bins look healthy)")

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
