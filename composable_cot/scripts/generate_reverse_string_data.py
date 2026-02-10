#!/usr/bin/env python
"""Generate reverse_string datasets in Composable CoT format.

Creates training/validation/test data for the reverse_string task following
the CCoT data format (instruction/input/output JSON), matching DeepMind's
RPE paper parameters (binary strings, lengths 1-40 for training, 41-100 OOD).

The CoT trace follows the composable format with <prefix> markers, providing
step-by-step reversal reasoning that the model learns to generate.

Usage:
    python composable_cot/scripts/generate_reverse_string_data.py
    python composable_cot/scripts/generate_reverse_string_data.py --dry-run
"""

import argparse
import json
import os
import random
from pathlib import Path


def ordinal(n: int) -> str:
    """Return ordinal string for integer (1st, 2nd, 3rd, ...)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def generate_example(length: int, rng: random.Random) -> dict:
    """Generate a single reverse_string example in CCoT format.

    Args:
        length: Number of binary digits in the string.
        rng: Random number generator for reproducibility.

    Returns:
        Dict with instruction, input, output, answer_label fields.
    """
    # Generate random binary string (matching DeepMind's vocab_size=2)
    digits = [rng.choice(["0", "1"]) for _ in range(length)]
    input_str = " ".join(digits)  # Space-separated for clean BPE tokenization
    reversed_str = "".join(reversed(digits))  # Concatenated answer

    # Build CoT trace: step-by-step reversal from end to start
    steps = []
    for i in range(length):
        pos = ordinal(i + 1)
        char = digits[length - 1 - i]
        steps.append(f"The {pos} character from the end is {char}.")

    cot_trace = " ".join(steps)
    cot_trace += f" So the answer is {reversed_str}."

    return {
        "instruction": f"Reverse the following binary string: {input_str} answer: ",
        "input": "",
        "output": f"<prefix> {cot_trace}</prefix><|endoftext|>",
        "answer_label": f"{reversed_str}<|endoftext|>",
    }


def generate_answer_only_example(length: int, rng: random.Random) -> dict:
    """Generate an answer-only example (for evaluation)."""
    digits = [rng.choice(["0", "1"]) for _ in range(length)]
    input_str = " ".join(digits)
    reversed_str = "".join(reversed(digits))

    return {
        "instruction": f"Reverse the following binary string: {input_str} answer: ",
        "input": "",
        "output": f"{reversed_str}<|endoftext|>",
        "string_length": length,
    }


def generate_split(
    num_examples: int,
    min_length: int,
    max_length: int,
    seed: int,
    answer_only: bool = False,
) -> list[dict]:
    """Generate a dataset split with uniform length distribution.

    Args:
        num_examples: Total examples to generate.
        min_length: Minimum string length (inclusive).
        max_length: Maximum string length (inclusive).
        seed: Random seed.
        answer_only: If True, generate answer-only format (no CoT trace).

    Returns:
        List of example dicts.
    """
    rng = random.Random(seed)
    examples = []

    for _ in range(num_examples):
        length = rng.randint(min_length, max_length)
        if answer_only:
            examples.append(generate_answer_only_example(length, rng))
        else:
            examples.append(generate_example(length, rng))

    return examples


def generate_length_stratified_split(
    samples_per_length: int,
    min_length: int,
    max_length: int,
    seed: int,
    answer_only: bool = True,
) -> list[dict]:
    """Generate a test split with equal samples per length (for fair eval).

    Args:
        samples_per_length: Number of examples at each length.
        min_length: Minimum string length (inclusive).
        max_length: Maximum string length (inclusive).
        seed: Random seed.
        answer_only: If True, generate answer-only format.

    Returns:
        List of example dicts, sorted by string_length.
    """
    rng = random.Random(seed)
    examples = []

    for length in range(min_length, max_length + 1):
        for _ in range(samples_per_length):
            if answer_only:
                examples.append(generate_answer_only_example(length, rng))
            else:
                examples.append(generate_example(length, rng))

    return examples


def save_json(data: list[dict], path: str) -> None:
    """Save data to JSON file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} examples to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reverse_string datasets in CCoT format"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sample examples without saving")
    parser.add_argument("--train-count", type=int, default=5000,
                        help="Number of training examples")
    parser.add_argument("--val-count", type=int, default=500,
                        help="Number of validation examples")
    parser.add_argument("--test-samples-per-length", type=int, default=10,
                        help="Test examples per length (length-stratified)")
    parser.add_argument("--min-train-length", type=int, default=1,
                        help="Min training string length (DeepMind: 1)")
    parser.add_argument("--max-train-length", type=int, default=40,
                        help="Max training string length (DeepMind: 40)")
    parser.add_argument("--max-eval-length", type=int, default=100,
                        help="Max evaluation string length (DeepMind: 100)")
    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    ccot_root = script_dir.parent
    atomic_dir = ccot_root / "data" / "atomic" / "reverse_string_composable_cot"
    eval_dir = ccot_root / "data" / "reverse_string_eval"

    print("=" * 70)
    print("Generating Reverse String Datasets (CCoT Format)")
    print("=" * 70)
    print(f"  Training lengths:   [{args.min_train_length}, {args.max_train_length}]")
    print(f"  OOD eval lengths:   [{args.max_train_length + 1}, {args.max_eval_length}]")
    print(f"  Train examples:     {args.train_count}")
    print(f"  Val examples:       {args.val_count}")
    print(f"  Test samples/length:{args.test_samples_per_length}")
    print(f"  Seed:               {args.seed}")

    # --- Generate examples ---
    print("\nGenerating...")

    # Training data (Composable CoT format with traces)
    train_data = generate_split(
        args.train_count, args.min_train_length, args.max_train_length,
        seed=args.seed, answer_only=False,
    )

    # Validation data (Composable CoT format)
    val_data = generate_split(
        args.val_count, args.min_train_length, args.max_train_length,
        seed=args.seed + 1000, answer_only=False,
    )

    # Test data: length-stratified for fair per-length evaluation
    # In-distribution test (answer-only for evaluation)
    test_in_dist = generate_length_stratified_split(
        args.test_samples_per_length,
        args.min_train_length, args.max_train_length,
        seed=args.seed + 2000, answer_only=True,
    )

    # OOD test (answer-only for evaluation)
    test_ood = generate_length_stratified_split(
        args.test_samples_per_length,
        args.max_train_length + 1, args.max_eval_length,
        seed=args.seed + 3000, answer_only=True,
    )

    # --- Show samples ---
    print("\n--- Sample training example (length=5) ---")
    sample = generate_example(5, random.Random(0))
    for k, v in sample.items():
        print(f"  {k}: {v[:120]}{'...' if len(v) > 120 else ''}")

    print("\n--- Sample training example (length=15) ---")
    sample = generate_example(15, random.Random(1))
    for k, v in sample.items():
        print(f"  {k}: {v[:120]}{'...' if len(v) > 120 else ''}")

    print("\n--- Sample answer-only eval example ---")
    sample = generate_answer_only_example(8, random.Random(2))
    for k, v in sample.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n[DRY RUN] No files saved.")
        return

    # --- Save datasets ---
    print("\nSaving datasets...")
    save_json(train_data, str(atomic_dir / "train.json"))
    save_json(val_data, str(atomic_dir / "val.json"))
    save_json(test_in_dist, str(eval_dir / "test_in_dist.json"))
    save_json(test_ood, str(eval_dir / "test_ood.json"))

    # Also save a combined test set for convenience
    test_all = test_in_dist + test_ood
    save_json(test_all, str(eval_dir / "test_all.json"))

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("Dataset Generation Complete")
    print(f"{'=' * 70}")
    print(f"  Atomic CoT data:    {atomic_dir}/")
    print(f"    train.json:       {len(train_data)} examples")
    print(f"    val.json:         {len(val_data)} examples")
    print(f"  Evaluation data:    {eval_dir}/")
    print(f"    test_in_dist.json:{len(test_in_dist)} examples (lengths {args.min_train_length}-{args.max_train_length})")
    print(f"    test_ood.json:    {len(test_ood)} examples (lengths {args.max_train_length + 1}-{args.max_eval_length})")
    print(f"    test_all.json:    {len(test_all)} examples (lengths {args.min_train_length}-{args.max_eval_length})")


if __name__ == "__main__":
    main()
