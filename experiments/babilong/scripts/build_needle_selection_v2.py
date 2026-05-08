#!/usr/bin/env python
"""Build the 227-sample selection for needle-position eval v2.

Sources:
  - 207 hard multi-entry samples from the differentiating subset
    (all-bins × all-models failure score definition)
  - 20 single-entry reference samples (same 20 as v1, seed=42)

Output: data/eval_needle_v2/selected_227_indices.json
"""
import json
from pathlib import Path

BABILONG = Path(__file__).resolve().parent.parent

HARD_SUBSET = BABILONG / "analysis/multi_entry_eval/differentiating_subset_207.json"
V1_SELECTION = BABILONG / "data/eval_needle/selected_100_indices.json"
EVAL_DIR = BABILONG / "data/eval_multi_entry"
RESULTS = {
    "lora": BABILONG / "results/lora_base_me",
    "y2_base": BABILONG / "results/y2_base_me",
    "y2_rpe": BABILONG / "results/y2_rpe_cur_L16k_me",
}
BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
OUT_PATH = BABILONG / "data/eval_needle_v2/selected_227_indices.json"


def load_eval_meta():
    """Load per-sample metadata from the multi-entry eval data."""
    meta = {}
    for b in BINS:
        path = EVAL_DIR / f"{b}.json"
        if not path.exists():
            continue
        for d in json.loads(path.read_text()):
            idx = d["original_idx"]
            if idx not in meta:
                meta[idx] = {
                    "object": d["object"],
                    "answer": d["answer"],
                    "target_entries": d["target_entries"],
                }
    return meta


def load_per_model_failures():
    """Load per-sample failure bins from multi-entry eval predictions."""
    failures = {}
    for model_key, result_dir in RESULTS.items():
        for b in BINS:
            pred_path = result_dir / f"predictions_{b}.json"
            eval_path = EVAL_DIR / f"{b}.json"
            if not pred_path.exists() or not eval_path.exists():
                continue
            preds = json.loads(pred_path.read_text())
            evals = json.loads(eval_path.read_text())
            for p, e in zip(preds, evals):
                idx = e["original_idx"]
                if idx not in failures:
                    failures[idx] = {}
                if model_key not in failures[idx]:
                    failures[idx][model_key] = {"fails": [], "total": 0}
                failures[idx][model_key]["total"] += 1
                if not p["correct"]:
                    failures[idx][model_key]["fails"].append(b)
    return failures


def main():
    hard_subset = json.loads(HARD_SUBSET.read_text())
    hard_indices = set(s["original_idx"] for s in hard_subset["samples"])
    print(f"Hard subset: {len(hard_indices)} multi-entry samples")

    v1 = json.loads(V1_SELECTION.read_text())
    single_entry_indices = set()
    for ps in v1["per_sample"]:
        if ps["tier"] == "single_entry_ref":
            single_entry_indices.add(ps["idx"])
    print(f"Single-entry reference: {len(single_entry_indices)} samples (from v1)")

    assert not (hard_indices & single_entry_indices), "Overlap between hard and single-entry!"

    all_indices = sorted(hard_indices | single_entry_indices)
    print(f"Total: {len(all_indices)} samples")

    eval_meta = load_eval_meta()
    failures = load_per_model_failures()

    per_sample = []
    for idx in all_indices:
        m = eval_meta.get(idx, {})
        f = failures.get(idx, {})

        if idx in single_entry_indices:
            tier = "single_entry_ref"
        else:
            tier = "hard_multi_entry"

        failure_score = 0
        for model_key in RESULTS:
            if model_key in f:
                failure_score += len(f[model_key]["fails"])

        entry = {
            "idx": idx,
            "object": m.get("object", "unknown"),
            "answer": m.get("answer", "unknown"),
            "target_entries": m.get("target_entries", 0),
            "tier": tier,
            "failure_score": failure_score,
        }
        per_sample.append(entry)

    per_sample.sort(key=lambda x: -x["failure_score"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "n_total": len(all_indices),
        "n_hard_multi_entry": len(hard_indices),
        "n_single_entry_ref": len(single_entry_indices),
        "description": (
            "207 hard multi-entry samples (differentiating subset: "
            "excluded samples correct by every model at every bin) + "
            "20 single-entry reference. For needle-position eval v2."
        ),
        "hard_subset_definition": hard_subset["definition"],
        "selected_indices": all_indices,
        "per_sample": per_sample,
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {OUT_PATH}")
    print(f"  {len(hard_indices)} hard multi-entry + {len(single_entry_indices)} single-entry = {len(all_indices)} total")


if __name__ == "__main__":
    main()
