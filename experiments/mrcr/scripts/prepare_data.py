#!/usr/bin/env python
"""Download MRCR dataset, bin by Qwen tokenizer, and create train/test splits.

The MRCR (Multi-Round Coreference Resolution) dataset is a harder version of
needle-in-a-haystack: identify the value for a specific query buried in a
very long irrelevant multi-turn conversation.

Official bins (by token count):
  [4096, 8192], (8192, 16384], (16384, 32768], (32768, 65536],
  (65536, 131072], (131072, 262144], (262144, 524288], (524288, 1048576]

We re-bin using Qwen2.5's tokenizer (not OpenAI's o200k_base) since token
counts differ across tokenizers.

Usage:
    python experiments/mrcr/scripts/prepare_data.py \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --output-dir experiments/mrcr/data \
        --train-ratio 0.7
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


# Official MRCR bin boundaries (inclusive lower, exclusive upper for internal bins)
BIN_BOUNDARIES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
BIN_LABELS = [
    "4K-8K", "8K-16K", "16K-32K", "32K-64K",
    "64K-128K", "128K-256K", "256K-512K", "512K-1M",
]


def get_bin_index(token_count: int) -> int:
    """Assign a token count to a bin index. Returns -1 if below minimum."""
    if token_count < BIN_BOUNDARIES[0]:
        return -1  # Below smallest bin
    for i in range(len(BIN_BOUNDARIES) - 1):
        if token_count <= BIN_BOUNDARIES[i + 1]:
            return i
    return len(BIN_BOUNDARIES) - 1  # Above largest bin


def get_bin_label(bin_index: int) -> str:
    """Get human-readable label for a bin index."""
    if bin_index < 0:
        return f"<{BIN_BOUNDARIES[0]}"
    if bin_index >= len(BIN_LABELS):
        return f">{BIN_BOUNDARIES[-1]}"
    return BIN_LABELS[bin_index]


def tokenize_prompt(prompt_json: str, tokenizer) -> int:
    """Count tokens in an MRCR prompt using the given tokenizer.

    The prompt field is a JSON string containing a list of message dicts.
    We apply the chat template to get the actual token count the model will see.
    """
    messages = json.loads(prompt_json)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(text)
    return len(token_ids)


def main():
    parser = argparse.ArgumentParser(description="Prepare MRCR data with Qwen tokenizer binning")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Tokenizer to use for binning")
    parser.add_argument("--output-dir", type=str, default="experiments/mrcr/data",
                        help="Output directory for processed data")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of data to use for training (rest is test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--n-needles", type=int, nargs="+", default=[2],
                        help="Needle counts to include (default: 2 only)")
    parser.add_argument("--max-bin", type=int, default=2,
                        help="Maximum bin index to process (0=4K-8K, 1=8K-16K, 2=16K-32K)")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    print("=" * 70)
    print("MRCR Data Preparation")
    print("=" * 70)
    print(f"  Tokenizer:    {args.tokenizer}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Train ratio:  {args.train_ratio}")
    print(f"  Needle counts: {args.n_needles}")
    print(f"  Max bin:      {args.max_bin} ({get_bin_label(args.max_bin)})")

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load MRCR dataset
    print("Loading MRCR dataset from HuggingFace...")
    ds = load_dataset("openai/mrcr", split="train")
    print(f"  Total samples: {len(ds)}")

    # Filter by needle count
    ds_filtered = ds.filter(lambda x: x["n_needles"] in args.n_needles)
    print(f"  After filtering n_needles={args.n_needles}: {len(ds_filtered)} samples")

    # Tokenize and bin each sample
    print("\nTokenizing and binning samples (this may take a while for long sequences)...")
    binned_data = defaultdict(list)
    skipped_below = 0
    skipped_above = 0

    for i, example in enumerate(ds_filtered):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(ds_filtered)}...", flush=True)

        token_count = tokenize_prompt(example["prompt"], tokenizer)
        bin_idx = get_bin_index(token_count)

        if bin_idx < 0:
            skipped_below += 1
            continue
        if bin_idx > args.max_bin:
            skipped_above += 1
            continue

        sample = {
            "prompt": example["prompt"],  # JSON string of messages
            "answer": example["answer"],
            "random_string_to_prepend": example["random_string_to_prepend"],
            "n_needles": example["n_needles"],
            "n_chars": example["n_chars"],
            "token_count_qwen": token_count,
            "bin_index": bin_idx,
            "bin_label": get_bin_label(bin_idx),
        }
        binned_data[bin_idx].append(sample)

    # Report bin distribution
    print(f"\n{'=' * 50}")
    print("Bin distribution (Qwen tokenizer):")
    print(f"  {'Bin':<12} {'Count':>6}")
    print(f"  {'-'*12} {'-'*6}")
    for bin_idx in sorted(binned_data.keys()):
        label = get_bin_label(bin_idx)
        print(f"  {label:<12} {len(binned_data[bin_idx]):>6}")
    print(f"  {'Skipped (<4K)':<12} {skipped_below:>6}")
    print(f"  {'Skipped (>max)':<12} {skipped_above:>6}")

    # Create train/test splits per bin
    os.makedirs(args.output_dir, exist_ok=True)
    split_info = {}

    for bin_idx in sorted(binned_data.keys()):
        samples = binned_data[bin_idx]
        random.shuffle(samples)

        split_point = int(len(samples) * args.train_ratio)
        train_samples = samples[:split_point]
        test_samples = samples[split_point:]

        label = get_bin_label(bin_idx)
        bin_dir = os.path.join(args.output_dir, f"bin{bin_idx}_{label}")
        os.makedirs(bin_dir, exist_ok=True)

        train_path = os.path.join(bin_dir, "train.json")
        test_path = os.path.join(bin_dir, "test.json")

        with open(train_path, "w") as f:
            json.dump(train_samples, f, indent=2)
        with open(test_path, "w") as f:
            json.dump(test_samples, f, indent=2)

        split_info[label] = {
            "bin_index": bin_idx,
            "total": len(samples),
            "train": len(train_samples),
            "test": len(test_samples),
            "train_path": train_path,
            "test_path": test_path,
        }
        print(f"\n  {label}: {len(train_samples)} train / {len(test_samples)} test -> {bin_dir}")

    # Save metadata
    metadata = {
        "tokenizer": args.tokenizer,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "n_needles": args.n_needles,
        "max_bin": args.max_bin,
        "bin_boundaries": BIN_BOUNDARIES,
        "splits": split_info,
    }
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to {meta_path}")

    print(f"\n{'=' * 70}")
    print("Data preparation complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
