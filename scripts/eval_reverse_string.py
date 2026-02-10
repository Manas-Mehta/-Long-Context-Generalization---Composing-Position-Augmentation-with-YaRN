#!/usr/bin/env python
"""Evaluate reverse-string task: baseline vs RPE-patched Qwen 2.5 1.5B.

This is pipeline validation — we expect poor accuracy from both modes since
the model has not been fine-tuned on string reversal. The goals are:
  1. Verify the evaluation harness runs end-to-end
  2. Confirm RPE-patched model produces different outputs from baseline
  3. Validate the infrastructure before training experiments

Usage:
    python scripts/eval_reverse_string.py
    python scripts/eval_reverse_string.py --lengths 5 10 20 --examples-per-length 3
"""

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpe.patching import RPEPatcher
from rpe.tasks.reverse_string import ReverseStringTask


def print_table(baseline_results: dict, rpe_results: dict, lengths: list[int]) -> None:
    """Print a comparison table of baseline vs RPE accuracy per length."""
    header = f"{'Length':>6}  {'Baseline':>10}  {'RPE':>10}  {'Diff':>8}  {'B samples':>20}  {'R samples':>20}"
    print(header)
    print("-" * len(header))

    for length in lengths:
        b = baseline_results["per_length"].get(length, {"accuracy": 0.0, "correct": 0, "total": 0})
        r = rpe_results["per_length"].get(length, {"accuracy": 0.0, "correct": 0, "total": 0})

        b_acc = b["accuracy"]
        r_acc = r["accuracy"]
        diff = r_acc - b_acc

        # Find a sample for each
        b_sample = ""
        r_sample = ""
        for d in baseline_results["details"]:
            if d["length"] == length:
                b_sample = f"{d['expected'][:8]}→{d['generated'][:8]}"
                break
        for d in rpe_results["details"]:
            if d["length"] == length:
                r_sample = f"{d['expected'][:8]}→{d['generated'][:8]}"
                break

        print(f"{length:>6}  {b_acc:>9.0%}  {r_acc:>9.0%}  {diff:>+7.0%}  {b_sample:>20}  {r_sample:>20}")

    b_all = baseline_results["overall_accuracy"]
    r_all = rpe_results["overall_accuracy"]
    print("-" * len(header))
    print(f"{'Total':>6}  {b_all:>9.0%}  {r_all:>9.0%}  {r_all - b_all:>+7.0%}")


def count_output_differences(baseline_results: dict, rpe_results: dict) -> tuple[int, int]:
    """Count how many examples produced different outputs between baseline and RPE."""
    differ = 0
    total = len(baseline_results["details"])
    for b_detail, r_detail in zip(baseline_results["details"], rpe_results["details"]):
        if b_detail["generated"] != r_detail["generated"]:
            differ += 1
    return differ, total


def main():
    parser = argparse.ArgumentParser(description="Reverse string: baseline vs RPE")
    parser.add_argument("--lengths", type=int, nargs="+", default=[5, 10, 20, 40],
                        help="String lengths to evaluate")
    parser.add_argument("--examples-per-length", type=int, default=5,
                        help="Number of examples per length")
    parser.add_argument("--max-sim-length", type=int, default=8192,
                        help="RPE max simulation length L")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("Reverse String Evaluation: Baseline vs RPE")
    print("=" * 70)

    # --- Load model ---
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.eval()
    print(f"Loaded. Vocab={model.config.vocab_size}, Layers={model.config.num_hidden_layers}")

    # --- Generate examples ---
    task = ReverseStringTask(seed=args.seed)
    examples = []
    for length in args.lengths:
        for _ in range(args.examples_per_length):
            examples.append(task.generate_example(length))
    print(f"\nGenerated {len(examples)} examples across lengths {args.lengths}")
    print(f"Sample: \"{examples[0].prompt.strip()}\" → \"{examples[0].expected}\"")

    # --- Baseline evaluation ---
    print("\n" + "-" * 70)
    print("Running BASELINE evaluation (standard positions)...")
    print("-" * 70)
    baseline_results = task.evaluate(model, tokenizer, examples)

    # --- RPE evaluation ---
    print("\n" + "-" * 70)
    print(f"Running RPE evaluation (L={args.max_sim_length})...")
    print("-" * 70)
    rpe_config = {"max_simulation_length": args.max_sim_length, "seed": args.seed}
    patcher = RPEPatcher(model, rpe_config)
    patcher.patch()
    # model is in eval mode, so RPE will pass through standard positions.
    # For RPE evaluation, switch to train mode so positions are randomized.
    model.train()
    rpe_results = task.evaluate(model, tokenizer, examples)
    model.eval()
    patcher.unpatch()

    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print_table(baseline_results, rpe_results, args.lengths)

    # --- Difference check ---
    differ, total = count_output_differences(baseline_results, rpe_results)
    print(f"\nOutput difference: {differ}/{total} examples produced different text "
          f"({differ/total:.0%})")

    # --- Show some example outputs ---
    print("\n" + "-" * 70)
    print("Sample outputs (first 2 per length):")
    print("-" * 70)
    shown = {}
    for b_d, r_d in zip(baseline_results["details"], rpe_results["details"]):
        length = b_d["length"]
        shown.setdefault(length, 0)
        if shown[length] >= 2:
            continue
        shown[length] += 1
        marker = "==" if b_d["generated"] == r_d["generated"] else "!="
        print(f"  len={length:>3} | input=\"{b_d['input'][:15]}\" expect=\"{b_d['expected'][:15]}\"")
        print(f"         | baseline: \"{b_d['generated'][:30]}\"")
        print(f"         | rpe:      \"{r_d['generated'][:30]}\"  [{marker}]")

    # --- Pipeline verdict ---
    print("\n" + "=" * 70)
    pipeline_ok = differ > 0
    print(f"Pipeline validation: {'PASS' if pipeline_ok else 'FAIL'}")
    if pipeline_ok:
        print("  RPE patching produces different model outputs — pipeline is working.")
    else:
        print("  WARNING: All outputs identical — RPE may not be affecting generation.")
    print("  (Low accuracy is expected — model is not fine-tuned on reversal.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
