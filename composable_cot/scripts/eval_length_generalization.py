#!/usr/bin/env python
"""Evaluate length generalization for reverse_string with CCoT + RPE.

Loads a Qwen2.5-7B base model + LoRA checkpoint, merges the adapter,
then evaluates per-length accuracy on in-distribution and OOD test sets.

Follows the CCoT inference pattern: prompt ends with "answer: ", model
generates the full CCoT trace including the final answer. We extract
the answer and compare to the ground truth.

Usage:
    # Evaluate a single checkpoint
    python composable_cot/scripts/eval_length_generalization.py \
        --base-model Qwen/Qwen2.5-7B \
        --lora-ckpt composable_cot/model_ckpt/reverse_string_rpe \
        --test-file composable_cot/data/reverse_string_eval/test_all.json \
        --output-dir composable_cot/outputs/reverse_string_rpe_eval

    # Evaluate on existing CCoT tasks
    python composable_cot/scripts/eval_length_generalization.py \
        --base-model Qwen/Qwen2.5-7B \
        --lora-ckpt composable_cot/model_ckpt/reverse_string_rpe \
        --test-file composable_cot/data/composition/answer_only/letter_concat_next_last_letter/test.json \
        --output-dir composable_cot/outputs/reverse_string_rpe_ccot_eval \
        --task-type ccot_standard
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_with_lora(
    base_model_name: str,
    lora_ckpt_dir: str,
    device: str = "cuda",
    torch_dtype=torch.bfloat16,
):
    """Load base model, apply LoRA adapter, merge and unload.

    Returns the merged model ready for inference (no adapter overhead).
    """
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    if lora_ckpt_dir and os.path.exists(lora_ckpt_dir):
        print(f"Loading LoRA adapter: {lora_ckpt_dir}")
        model = PeftModel.from_pretrained(model, lora_ckpt_dir)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
    else:
        print(f"No LoRA checkpoint found at {lora_ckpt_dir} — using base model only")

    model.eval()
    return model, tokenizer


def extract_answer_reverse_string(generated_text: str) -> str:
    """Extract the reversed string answer from CCoT-style generation.

    Looks for patterns like:
    - "So the answer is 10110."
    - "the answer is 10110"
    - Falls back to last word if no pattern found.
    """
    # Try: "the answer is <answer>"
    match = re.search(r"the answer is\s+([01]+)", generated_text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try: "answer: <answer>" (if model just outputs the answer directly)
    match = re.search(r"answer:\s*([01]+)", generated_text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Fallback: extract any contiguous binary string from the end
    matches = re.findall(r"[01]+", generated_text)
    if matches:
        return matches[-1]

    return generated_text.strip()


def extract_answer_ccot_standard(generated_text: str) -> str:
    """Extract answer from standard CCoT generation (non-reverse-string tasks).

    Looks for "the answer is <answer>" or "gives us <answer>".
    """
    for indicator in ["the answer is", "gives us"]:
        parts = generated_text.lower().split(indicator)
        if len(parts) > 1:
            answer = parts[-1].strip().rstrip(".").strip()
            return answer
    # Fallback: last word
    return generated_text.strip().split()[-1] if generated_text.strip() else ""


def evaluate_test_set(
    model,
    tokenizer,
    test_data: list[dict],
    task_type: str = "reverse_string",
    max_new_tokens: int = 512,
    batch_size: int = 1,
) -> dict:
    """Evaluate model on a test set with per-length accuracy tracking.

    Args:
        model: Merged model ready for inference.
        tokenizer: Tokenizer matching the model.
        test_data: List of dicts with instruction, output, and optionally string_length.
        task_type: "reverse_string" or "ccot_standard" — determines answer extraction.
        max_new_tokens: Maximum tokens to generate per example.
        batch_size: Batch size for generation (1 for greedy is fine).

    Returns:
        Dict with per-length results, overall metrics, and raw predictions.
    """
    tokenizer.padding_side = "left"

    results_by_length = defaultdict(lambda: {"correct": 0, "total": 0, "examples": []})
    all_predictions = []

    extract_fn = (
        extract_answer_reverse_string
        if task_type == "reverse_string"
        else extract_answer_ccot_standard
    )

    total_correct = 0
    total_examples = 0

    for i, example in enumerate(test_data):
        prompt = example["instruction"]

        # Determine expected answer
        raw_output = example["output"].replace("<|endoftext|>", "").strip()
        if task_type == "reverse_string":
            # For answer-only format, output is just the binary string
            # For CoT format, extract from "the answer is ..."
            if raw_output.startswith("<prefix>"):
                expected = extract_answer_reverse_string(raw_output)
            else:
                expected = raw_output
        else:
            expected = raw_output

        # Determine string length for grouping
        if "string_length" in example:
            length = example["string_length"]
        else:
            # Try to infer from instruction
            digits = re.findall(r"\b[01]\b", prompt)
            length = len(digits) if digits else 0

        # Generate
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract and compare answer
        predicted = extract_fn(generated_text)
        is_correct = predicted.strip().lower() == expected.strip().lower()

        if is_correct:
            total_correct += 1
        total_examples += 1

        results_by_length[length]["total"] += 1
        if is_correct:
            results_by_length[length]["correct"] += 1

        prediction_record = {
            "index": i,
            "length": length,
            "prompt_tail": prompt[-60:],
            "expected": expected,
            "predicted": predicted,
            "generated_text": generated_text[:200],
            "correct": is_correct,
        }
        all_predictions.append(prediction_record)
        results_by_length[length]["examples"].append(prediction_record)

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_data)}] acc so far: {total_correct}/{total_examples} "
                  f"= {total_correct/total_examples:.3f}")

    # Compute per-length accuracy
    per_length = {}
    for length in sorted(results_by_length.keys()):
        info = results_by_length[length]
        acc = info["correct"] / info["total"] if info["total"] > 0 else 0.0
        per_length[length] = {
            "accuracy": acc,
            "correct": info["correct"],
            "total": info["total"],
        }

    overall_acc = total_correct / total_examples if total_examples > 0 else 0.0

    return {
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_examples": total_examples,
        "per_length": per_length,
        "predictions": all_predictions,
    }


def compute_summary_metrics(
    per_length: dict,
    train_max_length: int = 40,
) -> dict:
    """Compute in-dist vs OOD summary metrics.

    Args:
        per_length: Dict mapping length -> {accuracy, correct, total}.
        train_max_length: Maximum training length (OOD starts at train_max_length+1).

    Returns:
        Dict with in_dist_accuracy, ood_accuracy, dm_score.
    """
    in_dist_accs = []
    ood_accs = []

    for length, info in sorted(per_length.items()):
        if length <= train_max_length:
            in_dist_accs.append(info["accuracy"])
        else:
            ood_accs.append(info["accuracy"])

    in_dist_acc = sum(in_dist_accs) / len(in_dist_accs) if in_dist_accs else 0.0
    ood_acc = sum(ood_accs) / len(ood_accs) if ood_accs else 0.0

    return {
        "in_dist_accuracy": in_dist_acc,
        "ood_accuracy": ood_acc,
        "dm_score": ood_acc,  # DeepMind's score = mean OOD accuracy
        "num_in_dist_lengths": len(in_dist_accs),
        "num_ood_lengths": len(ood_accs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate length generalization for CCoT + RPE"
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Base model name or path")
    parser.add_argument("--lora-ckpt", type=str, required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to test JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--task-type", type=str, default="reverse_string",
                        choices=["reverse_string", "ccot_standard"],
                        help="Task type for answer extraction")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate per example")
    parser.add_argument("--train-max-length", type=int, default=40,
                        help="Max training length (for in-dist/OOD split)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda, cpu, mps)")
    parser.add_argument("--torch-dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Model precision")
    parser.add_argument("--min-length", type=int, default=1,
                        help="Minimum string length to evaluate (skip shorter examples)")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Limit number of examples (0 = all)")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    print("=" * 70)
    print("CCoT + RPE Length Generalization Evaluation")
    print("=" * 70)
    print(f"  Base model:       {args.base_model}")
    print(f"  LoRA checkpoint:  {args.lora_ckpt}")
    print(f"  Test file:        {args.test_file}")
    print(f"  Task type:        {args.task_type}")
    print(f"  Train max length: {args.train_max_length}")
    print(f"  Min eval length:  {args.min_length}")
    print(f"  Max new tokens:   {args.max_new_tokens}")
    print(f"  Device:           {args.device}")
    print(f"  Precision:        {args.torch_dtype}")

    # Load model
    model, tokenizer = load_model_with_lora(
        args.base_model, args.lora_ckpt,
        device=args.device, torch_dtype=torch_dtype,
    )

    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    with open(args.test_file) as f:
        test_data = json.load(f)

    if args.min_length > 1:
        before = len(test_data)
        test_data = [ex for ex in test_data if ex.get("string_length", 0) >= args.min_length]
        print(f"  Filtered to length >= {args.min_length}: {before} -> {len(test_data)} examples")

    if args.max_examples > 0:
        test_data = test_data[:args.max_examples]
    print(f"  {len(test_data)} examples loaded")

    # Evaluate
    print(f"\nEvaluating...")
    results = evaluate_test_set(
        model, tokenizer, test_data,
        task_type=args.task_type,
        max_new_tokens=args.max_new_tokens,
    )

    # Compute summary
    summary = compute_summary_metrics(
        results["per_length"],
        train_max_length=args.train_max_length,
    )

    # Report
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Overall accuracy:     {results['overall_accuracy']:.4f} "
          f"({results['total_correct']}/{results['total_examples']})")
    print(f"  In-distribution:      {summary['in_dist_accuracy']:.4f} "
          f"(lengths 1-{args.train_max_length})")
    print(f"  Out-of-distribution:  {summary['ood_accuracy']:.4f} "
          f"(lengths {args.train_max_length + 1}+)")
    print(f"  DeepMind score:       {summary['dm_score']:.4f}")

    # Per-length breakdown (selected lengths)
    print(f"\n  Per-length accuracy:")
    print(f"  {'Length':>6}  {'Accuracy':>8}  {'Correct':>7}  {'Total':>5}  {'Region'}")
    for length in sorted(results["per_length"].keys()):
        info = results["per_length"][length]
        region = "ID" if length <= args.train_max_length else "OOD"
        print(f"  {length:>6}  {info['accuracy']:>8.3f}  {info['correct']:>7}  "
              f"{info['total']:>5}  {region}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Full results (without raw predictions for space)
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": args.base_model,
            "lora_ckpt": args.lora_ckpt,
            "test_file": args.test_file,
            "task_type": args.task_type,
            "train_max_length": args.train_max_length,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": len(test_data),
        },
        "overall_accuracy": results["overall_accuracy"],
        "in_dist_accuracy": summary["in_dist_accuracy"],
        "ood_accuracy": summary["ood_accuracy"],
        "dm_score": summary["dm_score"],
        "per_length": results["per_length"],
    }

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Save raw predictions for debugging
    preds_path = os.path.join(args.output_dir, "predictions.json")
    with open(preds_path, "w") as f:
        json.dump(results["predictions"], f, indent=2)
    print(f"  Predictions saved to {preds_path}")

    print(f"\n{'=' * 70}")
    print("Evaluation complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
