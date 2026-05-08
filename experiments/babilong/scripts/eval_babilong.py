#!/usr/bin/env python
"""Evaluate a trained BABILong LoRA checkpoint across all 9 context-length bins.

Loads a LoRA adapter on top of Qwen2.5-7B-Instruct, runs greedy decoding on
each eval bin, grades with the official BABILong metric (preprocess_output +
compare_answers from babilong/metrics.py), and saves full per-sample
predictions + an accuracy summary.

Eval YaRN settings per condition (all conditions trained with f=2 if YaRN):
  lora_base     : no YaRN  (vanilla RoPE at eval)
  y2_base       : --enable-yarn --yarn-factor 4.0  (f=2 train → f=4 eval)
  y2_rpe_cur    : --enable-yarn --yarn-factor 4.0
  y2_pose_32k   : --enable-yarn --yarn-factor 4.0
  rpe_only      : no YaRN  (ablation — RPE only)
  pose_only     : no YaRN  (ablation — PoSE only)

Output layout:
  <output-dir>/
    predictions_0k.json      # list of {question, target, prediction, correct, n_tokens}
    predictions_1k.json
    ...
    predictions_128k.json
    summary.json             # {bin: {n, n_correct, accuracy}, ..., "overall": {...}}

Usage:
  python eval_babilong.py \\
      --checkpoint-dir experiments/babilong/checkpoints/lora_base \\
      --eval-dir       experiments/babilong/data/eval \\
      --output-dir     experiments/babilong/results/lora_base \\
      --condition      lora_base

  # Zero-shot baseline (no LoRA):
  python eval_babilong.py \\
      --no-lora \\
      --eval-dir   experiments/babilong/data/eval \\
      --output-dir experiments/babilong/results/zero_shot \\
      --condition  zero_shot
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Prompt + grading — must match train_babilong_lora.py exactly
# ---------------------------------------------------------------------------

QA3_INSTRUCTION = (
    "I give you context with the facts about locations and actions of different persons "
    "hidden in some random text and a question. "
    "You need to answer the question based only on the information from the facts.\n"
    "If a person got an item in the first location and travelled to the second location "
    "the item is also in the second location. "
    "If a person dropped an item in the first location and moved to the second location "
    "the item remains in the first location."
)

QA3_POST_PROMPT = (
    "Your answer must be exactly one word — one of: "
    "bathroom, bedroom, garden, hallway, kitchen, office. "
    "Do not write anything else."
)

QA3_LABELS = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]

EVAL_BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


def build_prompt(sample: dict) -> str:
    """Build the user prompt. Mirrors build_messages() in train script."""
    if "messages" in sample:
        user_content  = sample["messages"][0]["content"]
        question_text = sample.get("question", "").strip()
    else:
        user_content  = sample["input"].strip()
        question_text = sample["question"].strip()

    # Extract context (everything before the first "\nQuestion:")
    parts   = user_content.rsplit("\nQuestion:", 1)
    context = parts[0].strip() if len(parts) > 1 else user_content.strip()

    if not question_text and len(parts) > 1:
        q_line = parts[1].replace("\nAnswer with only one word.", "").strip()
        question_text = q_line.split("\n")[0].strip()

    return (
        f"{QA3_INSTRUCTION}\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question_text}\n"
        f"{QA3_POST_PROMPT}"
    ), question_text


def grade(response: str, target: str, question: str) -> bool:
    """Official BABILong grading (mirrors preprocess_output + compare_answers).

    Strips the response down to its first meaningful word, then checks:
    - target is in the remaining labels
    - no other room label is also present (unless it also appears in the question)
    """
    response = response.lower()
    response = response.split(".")[0]
    response = response.split("<context>")[0]
    response = response.split("<example>")[0]
    response = response.split("Question")[0]
    response = response.strip()

    question = question.lower()
    labels   = set(QA3_LABELS)

    labels_in_question  = {l for l in labels if l in question}
    labels_in_response  = {l for l in labels if l in response}
    labels_in_response -= labels_in_question

    return target in labels_in_response and len(labels_in_response) == 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    base_model:  str,
    checkpoint:  str | None,
    enable_yarn: bool,
    yarn_factor: float,
    no_cuda:     bool,
):
    print(f"\n{'='*70}")
    print("MODEL LOADING")
    print(f"{'='*70}")
    print(f"  Base model:  {base_model}")
    print(f"  Checkpoint:  {checkpoint or '(none — zero-shot)'}")
    print(f"  YaRN:        {enable_yarn} (factor={yarn_factor})")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Left-pad for decoder-only generation
    tokenizer.padding_side = "left"
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s")

    config = AutoConfig.from_pretrained(base_model)
    model_kwargs = {"torch_dtype": torch.bfloat16}

    if enable_yarn:
        print(f"\n  Enabling YaRN: factor={yarn_factor}")
        rope_theta = None
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            rope_theta = config.rope_parameters.get("rope_theta")
        if rope_theta is None:
            rope_theta = getattr(config, "rope_theta", 1000000.0)

        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            config.rope_parameters.update({
                "type": "yarn", "rope_type": "yarn", "factor": yarn_factor,
            })
            if config.rope_parameters.get("rope_theta") is None:
                config.rope_parameters["rope_theta"] = rope_theta
        else:
            config.rope_scaling = {"type": "yarn", "factor": yarn_factor}

        model_kwargs["config"] = config

    device_map = None if no_cuda else "auto"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        **model_kwargs,
    )
    print(f"  Base model loaded in {time.time()-t0:.1f}s")

    if checkpoint is not None:
        t0 = time.time()
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        print(f"  LoRA merged in {time.time()-t0:.1f}s")

    model.eval()
    print(f"  Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Eval one bin
# ---------------------------------------------------------------------------

def eval_bin(
    model,
    tokenizer,
    bin_path:      str,
    bin_label:     str,
    max_samples:   int,
    max_seq_len:   int,
    max_new_tokens: int,
    no_cuda:       bool,
) -> tuple[list[dict], dict]:
    """Run inference on one bin. Returns (predictions, stats)."""
    with open(bin_path) as f:
        data = json.load(f)

    if max_samples > 0:
        data = data[:max_samples]

    device  = "cpu" if no_cuda else next(model.parameters()).device
    results = []
    n_skip  = 0

    t_bin = time.time()
    for i, sample in enumerate(data):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_bin
            rate    = (i + 1) / elapsed if elapsed > 0 else 0
            eta     = (len(data) - i - 1) / rate if rate > 0 else 0
            print(
                f"    [{bin_label}] {i+1}/{len(data)} | "
                f"{rate:.1f} samples/min | ETA {eta/60:.1f}min",
                flush=True,
            )

        target        = sample.get("answer", sample.get("target", "")).strip().lower()
        user_content, question_text = build_prompt(sample)

        # Tokenize prompt only
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
        n_tokens  = input_ids.shape[1]

        if n_tokens > max_seq_len:
            # Left-truncate prompt to fit
            input_ids = input_ids[:, -max_seq_len:]
            n_tokens  = input_ids.shape[1]

        if not no_cuda:
            input_ids = input_ids.to(device)

        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(
                out[0][input_ids.shape[1]:], skip_special_tokens=True
            ).strip().lower()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            prediction = "__OOM__"
            n_skip    += 1
        except Exception as e:
            prediction = f"__ERROR__: {e}"
            n_skip    += 1

        correct = grade(prediction, target, question_text)

        results.append({
            "question":   question_text,
            "target":     target,
            "prediction": prediction,
            "correct":    correct,
            "n_tokens":   n_tokens,
        })

    n_correct = sum(r["correct"] for r in results)
    n_gradable = sum(1 for r in results if not r["prediction"].startswith("__"))
    accuracy  = n_correct / len(results) if results else 0.0

    stats = {
        "n":          len(results),
        "n_correct":  n_correct,
        "accuracy":   round(accuracy, 4),
        "n_skipped":  n_skip,
        "elapsed_s":  round(time.time() - t_bin, 1),
    }
    print(
        f"    [{bin_label}] DONE: {n_correct}/{len(results)} correct "
        f"({accuracy:.1%}) | skipped={n_skip} | "
        f"{stats['elapsed_s']:.0f}s",
        flush=True,
    )
    return results, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Eval BABILong LoRA checkpoint")

    p.add_argument("--checkpoint-dir", default=None,
                   help="Path to LoRA adapter dir. Omit or --no-lora for zero-shot.")
    p.add_argument("--no-lora",        action="store_true",
                   help="Skip LoRA loading (zero-shot baseline eval).")
    p.add_argument("--base-model",     default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--enable-yarn",    action="store_true")
    p.add_argument("--yarn-factor",    type=float, default=4.0)
    p.add_argument("--no-cuda",        action="store_true")

    p.add_argument("--eval-dir",       required=True,
                   help="Dir with 0k.json … 128k.json")
    p.add_argument("--output-dir",     required=True)
    p.add_argument("--condition",      default="unknown",
                   help="Label for this condition (for summary file)")

    p.add_argument("--bins",           nargs="+", default=EVAL_BINS,
                   help="Which bins to eval (default: all 9)")
    p.add_argument("--max-samples",    type=int,  default=0,
                   help="Max samples per bin (0 = all)")
    p.add_argument("--max-seq-len",    type=int,  default=131072,
                   help="Max prompt tokens (left-truncate if over)")
    p.add_argument("--max-new-tokens", type=int,  default=10)

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("BABILong QA3 — Evaluation")
    print("=" * 70)
    print(f"  Condition:    {args.condition}")
    print(f"  Checkpoint:   {args.checkpoint_dir or '(zero-shot)'}")
    print(f"  YaRN:         {args.enable_yarn} (factor={args.yarn_factor})")
    print(f"  Eval dir:     {args.eval_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Bins:         {args.bins}")
    print(f"  Max samples:  {args.max_samples or 'all'}")
    print(f"  Max seq len:  {args.max_seq_len}")
    print(f"  Timestamp:    {datetime.now().isoformat()}")

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = None if args.no_lora else args.checkpoint_dir

    model, tokenizer = load_model(
        base_model  = args.base_model,
        checkpoint  = checkpoint,
        enable_yarn = args.enable_yarn,
        yarn_factor = args.yarn_factor,
        no_cuda     = args.no_cuda,
    )

    all_stats  = {}
    t_total    = time.time()

    for bin_label in args.bins:
        bin_path = os.path.join(args.eval_dir, f"{bin_label}.json")
        if not os.path.exists(bin_path):
            print(f"\n  [{bin_label}] SKIP — file not found: {bin_path}", flush=True)
            continue

        print(f"\n{'='*70}")
        print(f"BIN: {bin_label}")
        print(f"{'='*70}")

        predictions, stats = eval_bin(
            model         = model,
            tokenizer     = tokenizer,
            bin_path      = bin_path,
            bin_label     = bin_label,
            max_samples   = args.max_samples,
            max_seq_len   = args.max_seq_len,
            max_new_tokens= args.max_new_tokens,
            no_cuda       = args.no_cuda,
        )

        # Save per-bin predictions
        pred_path = os.path.join(args.output_dir, f"predictions_{bin_label}.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)

        all_stats[bin_label] = stats

        # Save running summary after each bin (so we have results even if later bins OOM)
        _save_summary(args, all_stats, t_total)

    _save_summary(args, all_stats, t_total)
    _print_summary(args.condition, all_stats)


def _save_summary(args, all_stats: dict, t_total: float):
    total_n       = sum(s["n"]        for s in all_stats.values())
    total_correct = sum(s["n_correct"] for s in all_stats.values())

    summary = {
        "condition":   args.condition,
        "checkpoint":  args.checkpoint_dir,
        "enable_yarn": args.enable_yarn,
        "yarn_factor": args.yarn_factor,
        "timestamp":   datetime.now().isoformat(),
        "elapsed_s":   round(time.time() - t_total, 1),
        "bins":        all_stats,
        "overall": {
            "n":         total_n,
            "n_correct": total_correct,
            "accuracy":  round(total_correct / total_n, 4) if total_n > 0 else 0.0,
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def _print_summary(condition: str, all_stats: dict):
    print(f"\n{'='*70}")
    print(f"SUMMARY — {condition}")
    print(f"{'='*70}")
    print(f"  {'Bin':<8} {'N':>6} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*10}")
    for bin_label, s in all_stats.items():
        print(f"  {bin_label:<8} {s['n']:>6} {s['n_correct']:>8} {s['accuracy']:>10.1%}")
    total_n       = sum(s["n"]        for s in all_stats.values())
    total_correct = sum(s["n_correct"] for s in all_stats.values())
    if total_n > 0:
        print(f"  {'OVERALL':<8} {total_n:>6} {total_correct:>8} {total_correct/total_n:>10.1%}")
    print(f"{'='*70}")
    print("EVAL COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
