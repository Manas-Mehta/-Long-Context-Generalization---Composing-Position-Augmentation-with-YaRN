#!/usr/bin/env python
"""Download and prepare BABILong QA3 dataset for RPE/YaRN/PoSE experiments.

BABILong is structured as HuggingFace configs (by length) and splits (by task).
To load QA3 at 4K: load_dataset("RMT-team/babilong-train-5k-samples", "4k", split="qa3")

Train dataset: RMT-team/babilong-train-5k-samples
  - Configs: 0k, 2k, 4k, 8k, 16k, 32k (no 1k)
  - Splits: qa1-qa10
  - 5000 samples per config+split combination
  - Fields: input, question, target

Eval dataset: RMT-team/babilong-1k-samples
  - Configs: 0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k
  - Splits: qa1-qa5 (qa1-qa20 for 0k)
  - ~1000 samples per config+split combination
  - Fields: input, question, target, split

We only use QA3 (three supporting facts) — hardest multi-hop task.

Output format per sample (saved as JSON):
{
    "messages": [
        {"role": "user",      "content": "<input>\\nQuestion: <question>\\nAnswer with only one word."},
        {"role": "assistant", "content": "<target>"}
    ],
    "answer":      "kitchen",
    "bin":         "4k",
    "token_count": 3712
}

Usage:
    python prepare_babilong.py --output-dir composable_cot/BABIlong/data
    python prepare_babilong.py --output-dir composable_cot/BABIlong/data --inspect-only
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

TRAIN_DATASET_ID = "RMT-team/babilong-train-5k-samples"
EVAL_DATASET_ID  = "RMT-team/babilong-1k-samples"
TASK             = "qa3"

# Bins we use for training (available in train set, no 1k bin)
TRAIN_BINS = ["0k", "2k", "4k", "8k"]

# All eval bins (0k through 128k)
EVAL_BINS  = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]

# QA3 closed-vocabulary answer labels
QA3_LABELS = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]


# -----------------------------------------------------------------------
# Prompt builder
# -----------------------------------------------------------------------

def build_prompt(sample: dict) -> str:
    """Combine input + question into the user message.

    The input field already contains the full context (bAbI facts interleaved
    with PG19 book text). The question field is the query. We append the
    question and a one-word instruction to the input.
    """
    return f"{sample['input'].strip()}\nQuestion: {sample['question'].strip()}\nAnswer with only one word."


def sample_to_training_format(sample: dict, bin_label: str, tokenizer) -> dict:
    """Convert a raw BABILong sample to our training JSON format."""
    user_content = build_prompt(sample)
    assistant_content = sample["target"].strip().lower()

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # Token count: full conversation as model will see it
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    token_count = len(tokenizer.encode(full_text, add_special_tokens=False))

    return {
        "messages":    messages,
        "answer":      assistant_content,
        "question":    sample["question"].strip(),  # raw question for grading
        "bin":         bin_label,
        "token_count": token_count,
    }


# -----------------------------------------------------------------------
# Inspect mode — print dataset structure without saving
# -----------------------------------------------------------------------

def inspect_dataset():
    """Print detailed dataset structure, field names, and sample examples."""
    print("=" * 70)
    print("BABILONG DATASET INSPECTION")
    print("=" * 70)

    print(f"\n--- TRAIN: {TRAIN_DATASET_ID} ---")
    for bin_label in TRAIN_BINS:
        try:
            ds = load_dataset(TRAIN_DATASET_ID, bin_label, split=TASK)
            print(f"\n  Config={bin_label}, split={TASK}")
            print(f"    Samples:    {len(ds)}")
            print(f"    Features:   {list(ds.features.keys())}")
            ex = ds[0]
            print(f"    target:     {ex['target']!r}")
            print(f"    question:   {ex['question']!r}")
            print(f"    input[:200]: {ex['input'][:200]!r}")
        except Exception as e:
            print(f"  Config={bin_label}: ERROR — {e}")

    print(f"\n--- EVAL: {EVAL_DATASET_ID} ---")
    for bin_label in EVAL_BINS:
        try:
            ds = load_dataset(EVAL_DATASET_ID, bin_label, split=TASK)
            print(f"\n  Config={bin_label}, split={TASK}")
            print(f"    Samples:    {len(ds)}")
            print(f"    Features:   {list(ds.features.keys())}")
            ex = ds[0]
            print(f"    target:     {ex['target']!r}")
            print(f"    question:   {ex['question']!r}")
            print(f"    input[:200]: {ex['input'][:200]!r}")
        except Exception as e:
            print(f"  Config={bin_label}: ERROR — {e}")


# -----------------------------------------------------------------------
# Main preparation
# -----------------------------------------------------------------------

def prepare(output_dir: str, tokenizer_name: str):
    print("=" * 70)
    print("BABILONG QA3 DATA PREPARATION")
    print("=" * 70)
    print(f"  Task:       {TASK}")
    print(f"  Train bins: {TRAIN_BINS}")
    print(f"  Eval bins:  {EVAL_BINS}")
    print(f"  Output:     {output_dir}")
    print(f"  Tokenizer:  {tokenizer_name}")

    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dir = os.path.join(output_dir, "train")
    eval_dir  = os.path.join(output_dir, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir,  exist_ok=True)

    # ----------------------------------------------------------------
    # Training data
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("TRAINING DATA")
    print(f"{'='*70}")

    all_train = []
    train_stats = {}

    for bin_label in TRAIN_BINS:
        print(f"\n  Loading train config={bin_label}, split={TASK}...")
        ds = load_dataset(TRAIN_DATASET_ID, bin_label, split=TASK)
        print(f"    Raw samples: {len(ds)}")

        # Verify answer labels
        unique_targets = set(ds["target"])
        unexpected = unique_targets - set(QA3_LABELS)
        if unexpected:
            print(f"    WARNING: unexpected targets: {unexpected}")

        # Convert
        converted = []
        token_counts = []
        for i, sample in enumerate(ds):
            if (i + 1) % 1000 == 0:
                print(f"    Tokenizing {i+1}/{len(ds)}...", flush=True)
            rec = sample_to_training_format(sample, bin_label, tokenizer)
            converted.append(rec)
            token_counts.append(rec["token_count"])

        # Per-bin stats
        stats = {
            "n":       len(converted),
            "tok_min": min(token_counts),
            "tok_max": max(token_counts),
            "tok_avg": round(sum(token_counts) / len(token_counts)),
        }
        train_stats[bin_label] = stats
        print(f"    Converted: {stats['n']} samples | tokens: {stats['tok_min']}-{stats['tok_max']} (avg {stats['tok_avg']})")

        # Save per-bin file
        bin_path = os.path.join(train_dir, f"{bin_label}.json")
        with open(bin_path, "w") as f:
            json.dump(converted, f)
        print(f"    Saved: {bin_path}")

        all_train.extend(converted)

    # Save combined
    combined_path = os.path.join(train_dir, "all_train.json")
    with open(combined_path, "w") as f:
        json.dump(all_train, f)
    print(f"\n  Combined train: {len(all_train)} samples -> {combined_path}")

    # ----------------------------------------------------------------
    # Eval data
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("EVAL DATA")
    print(f"{'='*70}")

    eval_stats = {}

    for bin_label in EVAL_BINS:
        print(f"\n  Loading eval config={bin_label}, split={TASK}...")
        try:
            ds = load_dataset(EVAL_DATASET_ID, bin_label, split=TASK)
        except Exception as e:
            print(f"    SKIP — {e}")
            continue
        print(f"    Raw samples: {len(ds)}")

        converted = []
        token_counts = []
        for i, sample in enumerate(ds):
            if (i + 1) % 500 == 0:
                print(f"    Tokenizing {i+1}/{len(ds)}...", flush=True)
            rec = sample_to_training_format(sample, bin_label, tokenizer)
            converted.append(rec)
            token_counts.append(rec["token_count"])

        stats = {
            "n":       len(converted),
            "tok_min": min(token_counts),
            "tok_max": max(token_counts),
            "tok_avg": round(sum(token_counts) / len(token_counts)),
        }
        eval_stats[bin_label] = stats
        print(f"    Converted: {stats['n']} samples | tokens: {stats['tok_min']}-{stats['tok_max']} (avg {stats['tok_avg']})")

        bin_path = os.path.join(eval_dir, f"{bin_label}.json")
        with open(bin_path, "w") as f:
            json.dump(converted, f)
        print(f"    Saved: {bin_path}")

    # ----------------------------------------------------------------
    # Save metadata
    # ----------------------------------------------------------------
    metadata = {
        "task":             TASK,
        "train_dataset":    TRAIN_DATASET_ID,
        "eval_dataset":     EVAL_DATASET_ID,
        "tokenizer":        tokenizer_name,
        "train_bins":       TRAIN_BINS,
        "eval_bins":        EVAL_BINS,
        "qa3_labels":       QA3_LABELS,
        "train_stats":      train_stats,
        "eval_stats":       eval_stats,
        "total_train":      len(all_train),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Training data ({TRAIN_DATASET_ID}, {TASK}):")
    print(f"  {'Bin':<8} {'Samples':>8} {'Tok Min':>8} {'Tok Max':>8} {'Tok Avg':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for b, s in train_stats.items():
        print(f"  {b:<8} {s['n']:>8} {s['tok_min']:>8} {s['tok_max']:>8} {s['tok_avg']:>8}")
    print(f"  {'TOTAL':<8} {len(all_train):>8}")

    print(f"\n  Eval data ({EVAL_DATASET_ID}, {TASK}):")
    print(f"  {'Bin':<8} {'Samples':>8} {'Tok Min':>8} {'Tok Max':>8} {'Tok Avg':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for b, s in eval_stats.items():
        print(f"  {b:<8} {s['n']:>8} {s['tok_min']:>8} {s['tok_max']:>8} {s['tok_avg']:>8}")

    print(f"\n  Metadata: {meta_path}")
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",    type=str, default="composable_cot/BABIlong/data")
    parser.add_argument("--tokenizer",     type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--inspect-only",  action="store_true",
                        help="Print dataset structure only, do not save files")
    args = parser.parse_args()

    if args.inspect_only:
        inspect_dataset()
    else:
        prepare(args.output_dir, args.tokenizer)


if __name__ == "__main__":
    main()
