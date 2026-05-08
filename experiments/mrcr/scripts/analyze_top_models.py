#!/usr/bin/env python3
"""Compare top models on recency bias and error types.

Outputs:
  analysis/top_models_position_bias.png  - "Lost in the Middle" style: avg score by needle position bucket
  analysis/top_models_bin4_breakdown.png - bar: perfect vs wrong-needle vs fail at bin 4
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("composable_cot/mrcr_context_extension")
DATA_DIR = BASE_DIR / "data"

MODELS = {
    "LoRA baseline": {
        "prefix": "lora_baseline",
        "base": BASE_DIR / "outputs",
        "color": "#777777", "marker": "s",
    },
    "RPE cur L=16K": {
        "prefix": "rpe_curriculum_lora_L16k",
        "base": BASE_DIR / "outputs",
        "color": "#1565C0", "marker": "D",
    },
    "Y2-Rc16": {
        "prefix": "y2_rpe_cur_L16k_yarn4",
        "base": BASE_DIR / "phase6/outputs_new",
        "color": "#42A5F5", "marker": "D",
    },
    "PoSE fixed": {
        "prefix": "pose_lora",
        "base": BASE_DIR / "outputs",
        "color": "#C62828", "marker": "^",
    },
    "Y2-P32": {
        "prefix": "y2_pose_32k_yarn4",
        "base": BASE_DIR / "phase6/outputs_new",
        "color": "#EF5350", "marker": "^",
    },
}

BINS = [
    ("bin0_4K-8K", "4K-8K"),
    ("bin1_8K-16K", "8K-16K"),
    ("bin2_16K-32K", "16K-32K"),
    ("bin3_32K-64K", "32K-64K"),
    ("bin4_64K-128K", "64K-128K"),
]


def load_preds(model_info, bin_suffix):
    path = model_info["base"] / f"{model_info['prefix']}_{bin_suffix}" / "predictions.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_test_data(bin_suffix):
    path = DATA_DIR / bin_suffix / "test.json"
    with open(path) as f:
        return json.load(f)


def extract_needle_positions(sample):
    messages = json.loads(sample["prompt"])
    last_msg = messages[-1]["content"]
    random_str = sample["random_string_to_prepend"]
    needle_content = sample["answer"][len(random_str):]

    topic_match = re.search(r"(?:st|nd|rd|th) \(1 indexed\) (.+?)\. Do not", last_msg)
    topic = topic_match.group(1).lower() if topic_match else ""
    idx_match = re.search(r"the (\d+)(?:st|nd|rd|th)", last_msg)
    requested_idx = int(idx_match.group(1)) if idx_match else 1

    char_offsets, offset = [], 0
    for m in messages:
        char_offsets.append(offset)
        offset += len(m["content"])
    total_chars = offset

    needle_infos = []
    for i, m in enumerate(messages[:-1]):
        if (m["role"] == "user" and topic in m["content"].lower()
                and "Prepend" not in m["content"]
                and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant"):
            rel_pos = char_offsets[i + 1] / total_chars if total_chars > 0 else 0
            needle_infos.append({"msg_idx": i + 1, "relative_pos": rel_pos,
                                 "content_preview": messages[i + 1]["content"][:200]})

    result = {"target_pos": None, "distractor_pos": None,
              "target_content": needle_content[:200], "distractor_content": None}
    if len(needle_infos) >= 2:
        target = needle_infos[requested_idx - 1]
        distractor = needle_infos[1 - (requested_idx - 1)]
        result["target_pos"] = target["relative_pos"]
        result["distractor_pos"] = distractor["relative_pos"]
        result["distractor_content"] = distractor["content_preview"]
    elif len(needle_infos) == 1:
        result["target_pos"] = needle_infos[0]["relative_pos"]
    return result


def classify_error(pred, needle_info):
    score = pred["score"]
    if score >= 0.95:
        return "Correct"
    response = pred.get("response_preview", "")
    dist_content = needle_info.get("distractor_content")
    if dist_content and len(response) > 20:
        resp_body = response[10:] if len(response) > 10 else response
        target_body = needle_info["target_content"]
        dist_sim = SequenceMatcher(None, resp_body[:300], dist_content[:300]).ratio()
        target_sim = SequenceMatcher(None, resp_body[:300], target_body[:300]).ratio()
        if dist_sim > 0.5 and dist_sim > target_sim + 0.1:
            return "Wrong Needle"
    if score >= 0.15:
        return "Other Error"
    return "Other Error"


def main():
    print("Loading data...")

    # Pre-load test metadata
    test_meta = {}
    for bin_suffix, _ in BINS:
        test_data = load_test_data(bin_suffix)
        for i, sample in enumerate(test_data):
            test_meta[(bin_suffix, i)] = extract_needle_positions(sample)

    # Load all predictions
    all_rows = []
    for model_name, model_info in MODELS.items():
        for bin_suffix, bin_label in BINS:
            preds = load_preds(model_info, bin_suffix)
            if preds is None:
                continue
            for pred in preds:
                idx = pred["index"]
                ni = test_meta.get((bin_suffix, idx), {})
                all_rows.append({
                    "model": model_name, "bin_label": bin_label,
                    "bin_idx": [b[1] for b in BINS].index(bin_label),
                    "index": idx, "score": pred["score"],
                    "error_type": classify_error(pred, ni),
                    "target_pos": ni.get("target_pos"),
                    "color": model_info["color"],
                })

    print(f"Loaded {len(all_rows)} predictions")
    ANALYSIS_DIR = BASE_DIR / "analysis"
    ANALYSIS_DIR.mkdir(exist_ok=True)

    # =====================================================================
    # Plot 1: "Lost in the Middle" style — avg score by needle position
    # Bins 3+4 combined (32K-128K), positions bucketed into 5 equal bins
    # =====================================================================
    N_BUCKETS = 5
    bucket_labels = ["0-20%\n(early)", "20-40%", "40-60%\n(middle)", "60-80%", "80-100%\n(late)"]
    model_names = list(MODELS.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(N_BUCKETS)

    for mn in model_names:
        rows = [r for r in all_rows if r["model"] == mn
                and r["bin_idx"] >= 3 and r["target_pos"] is not None]
        bucket_scores = [[] for _ in range(N_BUCKETS)]
        for r in rows:
            b = min(int(r["target_pos"] * N_BUCKETS), N_BUCKETS - 1)
            bucket_scores[b].append(r["score"])

        means = [np.mean(bs) if bs else 0 for bs in bucket_scores]
        ax.plot(x, means, marker=MODELS[mn]["marker"], markersize=8, linewidth=2.2,
                color=MODELS[mn]["color"], label=mn)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=10)
    ax.set_ylabel("Average Score", fontsize=12)
    ax.set_xlabel("Needle Position in Context", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Recency Bias: Score by Needle Position (32K–128K)", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "top_models_position_bias.png", dpi=150)
    print("Saved top_models_position_bias.png")

    # =====================================================================
    # Plot 2: Bin 4 breakdown — simple grouped bar: Correct / Wrong Needle / Other
    # =====================================================================
    categories = ["Correct", "Wrong Needle", "Other Error"]
    cat_colors = ["#4CAF50", "#E91E63", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(model_names))
    width = 0.25

    for i, (cat, cc) in enumerate(zip(categories, cat_colors)):
        counts = []
        for mn in model_names:
            matching = [r for r in all_rows if r["model"] == mn and r["bin_label"] == "64K-128K"]
            total = len(matching)
            n = sum(1 for r in matching if r["error_type"] == cat)
            counts.append(n)
        offset = (i - 1) * width
        ax.bar(x + offset, counts, width, label=cat, color=cc)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Count (out of 30)", fontsize=11)
    ax.set_title("Bin 4 (64K–128K): What Happens When the Model Responds?", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "top_models_bin4_breakdown.png", dpi=150)
    print("Saved top_models_bin4_breakdown.png")

    # =====================================================================
    # Print summary stats
    # =====================================================================
    print("\n--- Recency bias (avg score per position bucket, bins 3-4) ---")
    for mn in model_names:
        rows = [r for r in all_rows if r["model"] == mn
                and r["bin_idx"] >= 3 and r["target_pos"] is not None]
        bucket_scores = [[] for _ in range(N_BUCKETS)]
        for r in rows:
            b = min(int(r["target_pos"] * N_BUCKETS), N_BUCKETS - 1)
            bucket_scores[b].append(r["score"])
        means = [f"{np.mean(bs):.2f}" if bs else "N/A" for bs in bucket_scores]
        early = np.mean(bucket_scores[0]) if bucket_scores[0] else 0
        late = np.mean(bucket_scores[-1]) if bucket_scores[-1] else 0
        print(f"  {mn:20s}  buckets: {means}  gap(late-early): {late-early:+.2f}")

    print("\n--- Bin 4 breakdown ---")
    for mn in model_names:
        matching = [r for r in all_rows if r["model"] == mn and r["bin_label"] == "64K-128K"]
        correct = sum(1 for r in matching if r["error_type"] == "Correct")
        wn = sum(1 for r in matching if r["error_type"] == "Wrong Needle")
        other = sum(1 for r in matching if r["error_type"] == "Other Error")
        print(f"  {mn:20s}  Correct: {correct}  Wrong Needle: {wn}  Other: {other}")


if __name__ == "__main__":
    main()
