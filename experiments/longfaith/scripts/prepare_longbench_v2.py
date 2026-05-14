#!/usr/bin/env python
"""Prepare LongBench v2 evaluation: filter to QA subsets + bucket by Qwen tokens.

LongBench v2 (THUDM/LongBench-v2) has 503 examples across 6 domains. We use
only Single-Document QA + Multi-Document QA (300 examples), then bucket by
Qwen2.5 tokenizer length into 16K / 32K / 64K / 128K bins.

Output:
    <output-dir>/longbench_v2_qa.json          — 300 filtered QA examples
                                                 (with computed `n_tokens` field)
    <output-dir>/eval_v2_bin_indices.json      — {bin_label: [list of indices
                                                 into longbench_v2_qa.json]}

Bin assignment (Qwen token count of `context`):
    16k  : context_tokens <= 16384
    32k  : 16384 < context_tokens <= 32768
    64k  : 32768 < context_tokens <= 65536
    128k : 65536 < context_tokens <= 131072
    >128k: dropped (cannot fit even with YaRN f=4)

Run on a node with internet + Qwen tokenizer cache (HPC login node, or local
with HF cache populated). CPU-only, ~5 minutes.

Usage:
    python prepare_longbench_v2.py --output-dir experiments/longfaith/data
    # or from a local copy if HF unreachable:
    python prepare_longbench_v2.py --output-dir ... --local-path /tmp/longbench_v2.json
"""

import argparse
import json
import os
import sys

QA_DOMAINS = {"Single-Document QA", "Multi-Document QA"}

# Bin upper bounds in Qwen tokens
BIN_BOUNDS = [
    ("16k", 16384),
    ("32k", 32768),
    ("64k", 65536),
    ("128k", 131072),
]


def load_v2(local_path: str | None) -> list[dict]:
    if local_path:
        with open(local_path) as f:
            return json.load(f)
    # Pull from HF using hf_hub_download (raw file, no Arrow conversion).
    # The datasets library load_dataset() builds an in-memory Arrow table
    # which OOMs login nodes on the 465MB v2 file. Direct download avoids
    # this and is faster.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.", file=sys.stderr)
        sys.exit(1)
    print("  Downloading THUDM/LongBench-v2/data.json via hf_hub_download...")
    path = hf_hub_download(
        repo_id="THUDM/LongBench-v2",
        filename="data.json",
        repo_type="dataset",
    )
    print(f"  Downloaded -> {path}")
    with open(path) as f:
        return json.load(f)


def filter_to_qa(examples: list[dict]) -> list[dict]:
    return [ex for ex in examples if ex.get("domain") in QA_DOMAINS]


def compute_token_counts(examples: list[dict], tokenizer) -> list[dict]:
    print(f"  Tokenizing {len(examples)} contexts with Qwen tokenizer...")
    for i, ex in enumerate(examples):
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(examples)}...")
        n = len(tokenizer.encode(ex["context"], add_special_tokens=False))
        ex["n_tokens"] = n
    return examples


def bucket(examples: list[dict]) -> dict:
    buckets: dict[str, list[int]] = {b: [] for b, _ in BIN_BOUNDS}
    dropped = 0
    for idx, ex in enumerate(examples):
        n = ex["n_tokens"]
        placed = False
        for (bin_label, upper) in BIN_BOUNDS:
            if n <= upper:
                buckets[bin_label].append(idx)
                placed = True
                break
        if not placed:
            dropped += 1

    print(f"\n  Bin distribution (Qwen tokens):")
    for (bin_label, upper) in BIN_BOUNDS:
        print(f"    {bin_label:>5} (≤{upper:>6}): {len(buckets[bin_label]):>3} examples")
    print(f"    >128k (dropped): {dropped} examples")
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--local-path", default=None,
                    help="Use a local longbench_v2.json instead of downloading from HF.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Tokenizer to use for token counting.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed.", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    raw = load_v2(args.local_path)
    print(f"  Total v2 examples: {len(raw)}")

    qa = filter_to_qa(raw)
    print(f"  After QA filter (Single-Doc + Multi-Doc): {len(qa)}")

    qa = compute_token_counts(qa, tok)

    qa_path = os.path.join(args.output_dir, "longbench_v2_qa.json")
    with open(qa_path, "w") as f:
        json.dump(qa, f, indent=2)
    print(f"\n  Saved filtered QA -> {qa_path}")

    buckets = bucket(qa)
    bucket_path = os.path.join(args.output_dir, "eval_v2_bin_indices.json")
    with open(bucket_path, "w") as f:
        json.dump(buckets, f, indent=2)
    print(f"  Saved bin indices -> {bucket_path}")

    print(f"\n  DONE.")


if __name__ == "__main__":
    main()
