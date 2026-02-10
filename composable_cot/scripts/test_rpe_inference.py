#!/usr/bin/env python
"""Test RPE integration with Composable CoT's data format.

Pipeline validation: loads real Composable CoT examples and runs
inference with and without RPE patching. We use Qwen2.5-1.5B (not 7B)
since this is just validating the pipeline on a Mac M3.

We expect:
  - Both modes produce poor accuracy (model is not fine-tuned)
  - Outputs DIFFER between baseline and RPE modes
  - No crashes, correct shapes, no NaN/Inf

Usage:
    python composable_cot/scripts/test_rpe_inference.py
    python composable_cot/scripts/test_rpe_inference.py --num-examples 3
"""

import argparse
import json
import os
import sys

# Add project root so we can import rpe/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpe.patching import RPEPatcher

# Path to composable_cot data relative to this script
CCOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Task data files (answer-only format — simplest for evaluation)
TASK_FILES = {
    "letter_concat_ascii_multiply": "data/composition/answer_only/letter_concat_ascii_multiply/test.json",
    "letter_concat_next_last_letter": "data/composition/answer_only/letter_concat_next_last_letter/test.json",
    "next_last_letter_ascii_multiply": "data/composition/answer_only/next_last_letter_ascii_multiply/test.json",
}


def load_examples(task_name: str, num_examples: int) -> list[dict]:
    """Load a few examples from a Composable CoT answer-only test set."""
    path = os.path.join(CCOT_ROOT, TASK_FILES[task_name])
    with open(path) as f:
        data = json.load(f)
    return data[:num_examples]


def run_inference(
    model,
    tokenizer,
    examples: list[dict],
    max_new_tokens: int = 32,
) -> list[dict]:
    """Run greedy inference on Composable CoT examples.

    Each example has an 'instruction' field that ends with " answer: ".
    We generate from that prompt and compare to the 'output' field.
    """
    results = []
    for ex in examples:
        prompt = ex["instruction"]
        expected = ex["output"].replace("<|endoftext|>", "").strip()

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(next(model.parameters()).device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, prompt_len:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        results.append({
            "prompt_tail": prompt[-60:],
            "expected": expected,
            "generated": generated[:60],
            "correct": generated.strip() == expected,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Test RPE with Composable CoT data")
    parser.add_argument("--num-examples", type=int, default=4,
                        help="Examples per task to evaluate")
    parser.add_argument("--max-sim-length", type=int, default=8192,
                        help="RPE max simulation length")
    args = parser.parse_args()

    print("=" * 72)
    print("RPE × Composable CoT Pipeline Validation")
    print("=" * 72)

    # --- Load model ---
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"\nLoading {model_name} (pipeline test — not the 7B used in paper)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.eval()
    print(f"Loaded {type(model).__name__}")

    # --- Pick one task to test ---
    task_name = "letter_concat_ascii_multiply"
    examples = load_examples(task_name, args.num_examples)
    print(f"\nTask: {task_name}")
    print(f"Loaded {len(examples)} examples")
    print(f"Sample instruction (tail): ...{examples[0]['instruction'][-80:]}")
    print(f"Sample expected output:    {examples[0]['output'][:40]}")

    # --- Baseline ---
    print("\n" + "-" * 72)
    print("BASELINE (standard position IDs)")
    print("-" * 72)
    baseline = run_inference(model, tokenizer, examples)
    for i, r in enumerate(baseline):
        tag = "OK" if r["correct"] else "  "
        print(f"  [{tag}] expected={r['expected']:>12s}  got=\"{r['generated'][:40]}\"")

    # --- RPE ---
    print("\n" + "-" * 72)
    print(f"RPE (max_simulation_length={args.max_sim_length})")
    print("-" * 72)
    rpe_config = {"max_simulation_length": args.max_sim_length, "seed": 42}
    patcher = RPEPatcher(model, rpe_config)
    patcher.patch()

    rpe_results = run_inference(model, tokenizer, examples)
    for i, r in enumerate(rpe_results):
        tag = "OK" if r["correct"] else "  "
        print(f"  [{tag}] expected={r['expected']:>12s}  got=\"{r['generated'][:40]}\"")

    patcher.unpatch()

    # --- Comparison ---
    print("\n" + "=" * 72)
    print("COMPARISON")
    print("=" * 72)

    differ_count = 0
    for i, (b, r) in enumerate(zip(baseline, rpe_results)):
        same = b["generated"] == r["generated"]
        if not same:
            differ_count += 1
        marker = "SAME" if same else "DIFF"
        print(f"  Example {i}: [{marker}]")
        print(f"    baseline: \"{b['generated'][:50]}\"")
        print(f"    rpe:      \"{r['generated'][:50]}\"")

    print(f"\nOutputs differ: {differ_count}/{len(baseline)}")

    # --- Multi-task spot check ---
    print("\n" + "-" * 72)
    print("Multi-task spot check (1 example each)")
    print("-" * 72)
    for t_name in TASK_FILES:
        exs = load_examples(t_name, 1)
        b = run_inference(model, tokenizer, exs)

        patcher_t = RPEPatcher(model, rpe_config)
        patcher_t.patch()
        r = run_inference(model, tokenizer, exs)
        patcher_t.unpatch()

        same = b[0]["generated"] == r[0]["generated"]
        marker = "SAME" if same else "DIFF"
        print(f"  {t_name:<40s} [{marker}]")
        print(f"    baseline: \"{b[0]['generated'][:50]}\"")
        print(f"    rpe:      \"{r[0]['generated'][:50]}\"")

    # --- Verdict ---
    print("\n" + "=" * 72)
    pipeline_ok = differ_count > 0
    print(f"Pipeline validation: {'PASS' if pipeline_ok else 'FAIL'}")
    if pipeline_ok:
        print("  RPE produces different outputs on Composable CoT data.")
    else:
        print("  WARNING: outputs identical — RPE may not be affecting generation.")
    print("  (Low accuracy expected — model not fine-tuned on these tasks.)")
    print("=" * 72)


if __name__ == "__main__":
    main()
