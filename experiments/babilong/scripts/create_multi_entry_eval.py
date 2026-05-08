#!/usr/bin/env python
"""Create multi-entry eval subset from full BABILong eval data.

Filters to 305 samples where the object visits the target room 2+ times.
These are harder because the model must track which visit the question refers to.

Uses pre-computed sample_difficulty_tags.json (indices are consistent across bins
since the underlying bAbI stories are identical — only PG19 noise changes).

Run this on HPC before eval:
    python composable_cot/BABIlong/scripts/create_multi_entry_eval.py

Input:  composable_cot/BABIlong/data/eval/*.json (full 999 samples per bin)
Output: composable_cot/BABIlong/data/eval_multi_entry/*.json (305 samples per bin)
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BABILONG_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.join(BABILONG_DIR, "data", "eval")
OUT_DIR = os.path.join(BABILONG_DIR, "data", "eval_multi_entry")
TAGS_PATH = os.path.join(EVAL_DIR, "sample_difficulty_tags.json")

BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(TAGS_PATH) as f:
        tags = json.load(f)

    multi_indices = sorted(tags["multi_entry_indices"])
    idx_to_tag = {t["idx"]: t for t in tags["per_sample"] if t["multi"]}

    print(f"Multi-entry indices: {len(multi_indices)}")
    print(f"Output dir: {OUT_DIR}")
    print()

    for b in BINS:
        in_path = os.path.join(EVAL_DIR, f"{b}.json")
        out_path = os.path.join(OUT_DIR, f"{b}.json")

        with open(in_path) as f:
            data = json.load(f)

        filtered = []
        for idx in multi_indices:
            if idx >= len(data):
                continue
            sample = data[idx]
            tag = idx_to_tag[idx]
            sample["original_idx"] = idx
            sample["target_entries"] = tag["target_entries"]
            sample["object"] = tag["obj"]
            filtered.append(sample)

        with open(out_path, "w") as f:
            json.dump(filtered, f)

        print(f"  {b}: {len(data)} -> {len(filtered)} samples")

    # Metadata
    metadata = {
        "description": "Multi-entry subset of BABILong QA3 eval (target room visited 2+ times)",
        "source": "Filtered from eval/ using sample_difficulty_tags.json",
        "total_samples": len(multi_indices),
        "entry_distribution": {
            f"{e}_entries": sum(1 for t in tags["per_sample"] if t["multi"] and t["target_entries"] == e)
            for e in [2, 3, 4, 5]
        },
        "object_distribution": {
            obj: sum(1 for t in tags["per_sample"] if t["multi"] and t["obj"] == obj)
            for obj in ["football", "apple", "milk"]
        },
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Metadata saved to {OUT_DIR}/metadata.json")


if __name__ == "__main__":
    main()
