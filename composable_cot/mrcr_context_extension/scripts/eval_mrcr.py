#!/usr/bin/env python
"""Evaluate Qwen2.5-7B on MRCR (Multi-Round Coreference Resolution).

Supports three modes:
  1. Vanilla:  Base model, no modifications
  2. YaRN:    Base model with YaRN RoPE scaling (no training needed)
  3. LoRA:    Base model + LoRA checkpoint (with or without RPE training)

Grading uses the official MRCR metric: SequenceMatcher ratio between
the model's response (after stripping the prepended random string) and
the expected answer.

Usage:
    # Vanilla baseline
    python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --test-file composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json \
        --output-dir composable_cot/mrcr_context_extension/outputs/vanilla_4K-8K

    # YaRN baseline
    python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --enable-yarn --yarn-factor 4.0 \
        --test-file composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json \
        --output-dir composable_cot/mrcr_context_extension/outputs/yarn_4K-8K

    # RPE+LoRA
    python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --lora-ckpt composable_cot/mrcr_context_extension/checkpoints/rpe_rank16 \
        --test-file composable_cot/mrcr_context_extension/data/bin1_8K-16K/test.json \
        --output-dir composable_cot/mrcr_context_extension/outputs/rpe_rank16_8K-16K
"""

import argparse
import json
import os
import time
from datetime import datetime
from difflib import SequenceMatcher

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def grade_mrcr(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Official MRCR grading function.

    The model must prepend the random string before the answer.
    Score is SequenceMatcher ratio between response and answer
    (after stripping the random prefix from both).

    Returns:
        Float between 0.0 and 1.0.
    """
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def _diagnose_rope(model, config, enable_yarn, yarn_factor):
    """Print detailed RoPE diagnostics to confirm YaRN is actually changing things."""
    print("\n" + "=" * 70, flush=True)
    print("ROPE DIAGNOSTIC", flush=True)
    print("=" * 70, flush=True)

    # Get rotary embedding from layer 0
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        rotary = model.model.layers[0].self_attn.rotary_emb

    # Print all relevant attributes
    print(f"  Rotary class:           {type(rotary).__name__}", flush=True)
    print(f"  rope_type attr:         {getattr(rotary, 'rope_type', 'N/A')}", flush=True)
    print(f"  scaling_factor attr:    {getattr(rotary, 'scaling_factor', 'N/A')}", flush=True)
    print(f"  attention_scaling attr: {getattr(rotary, 'attention_scaling', 'N/A')}", flush=True)
    print(f"  max_seq_len_cached:     {getattr(rotary, 'max_seq_len_cached', 'N/A')}", flush=True)

    # Check config values
    base = getattr(config, "rope_theta", 1000000.0)
    max_pos = getattr(config, "max_position_embeddings", "N/A")
    print(f"  config.rope_theta:      {base}", flush=True)
    print(f"  config.max_position_embeddings: {max_pos}", flush=True)
    print(f"  config.rope_scaling:    {getattr(config, 'rope_scaling', 'N/A')}", flush=True)

    # Check inv_freq
    if hasattr(rotary, "inv_freq") and rotary.inv_freq is not None:
        inv_freq = rotary.inv_freq.float().cpu()
        dim = inv_freq.shape[0] * 2
        print(f"\n  inv_freq shape: {inv_freq.shape} (head_dim={dim})", flush=True)
        print(f"  inv_freq[0:5]:  {inv_freq[:5].tolist()}", flush=True)
        print(f"  inv_freq[-5:]:  {inv_freq[-5:].tolist()}", flush=True)
        print(f"  inv_freq mean:  {inv_freq.mean().item():.8e}", flush=True)
        print(f"  inv_freq sum:   {inv_freq.sum().item():.8e}", flush=True)

        # Compute expected VANILLA inv_freq for comparison
        vanilla_inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        print(f"\n  Expected vanilla inv_freq[0:5]:  {vanilla_inv[:5].tolist()}", flush=True)
        print(f"  Expected vanilla inv_freq[-5:]:  {vanilla_inv[-5:].tolist()}", flush=True)
        print(f"  Expected vanilla inv_freq mean:  {vanilla_inv.mean().item():.8e}", flush=True)
        print(f"  Expected vanilla inv_freq sum:   {vanilla_inv.sum().item():.8e}", flush=True)

        # Compare
        diff = (inv_freq - vanilla_inv).abs()
        max_diff = diff.max().item()
        n_different = (diff > 1e-8).sum().item()
        print(f"\n  Diff from vanilla: max={max_diff:.6e}, n_different={n_different}/{len(diff)}", flush=True)

        if n_different == 0:
            print("  *** inv_freq IDENTICAL to vanilla -- YaRN is NOT changing frequencies! ***", flush=True)
        else:
            print(f"  inv_freq IS different from vanilla -- YaRN appears to modify frequencies", flush=True)
    else:
        print("\n  inv_freq: NOT FOUND as buffer (may be computed dynamically in forward())", flush=True)

    # Run a short forward pass to get a logit fingerprint
    print(f"\n  Running 5-token forward pass for logit fingerprint...", flush=True)
    test_input = torch.tensor([[1, 2, 3, 4, 5]], device=model.device)
    with torch.no_grad():
        output = model(test_input)
    logits = output.logits[0, -1, :].float().cpu()
    print(f"  Logits shape:  {logits.shape}", flush=True)
    print(f"  Logits mean:   {logits.mean().item():.6f}", flush=True)
    print(f"  Logits std:    {logits.std().item():.6f}", flush=True)
    print(f"  Logits[0:5]:   {logits[:5].tolist()}", flush=True)
    top5 = logits.topk(5)
    print(f"  Top-5 tokens:  {top5.indices.tolist()}", flush=True)
    print(f"  Top-5 values:  {[f'{v:.4f}' for v in top5.values.tolist()]}", flush=True)
    print("=" * 70 + "\n", flush=True)


def _apply_yarn_manual(model, yarn_factor, config):
    """Manually patch rotary embedding inv_freq with YaRN scaling.

    Fallback when config-based YaRN is silently ignored.
    Implements the NTK-aware interpolation from the YaRN paper (simplified).
    """
    import math

    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        rotary = model.model.layers[0].self_attn.rotary_emb

    dim = rotary.inv_freq.shape[0] * 2  # inv_freq has dim/2 elements
    base = getattr(config, "rope_theta", 1000000.0)
    original_max_pos = config.max_position_embeddings

    # NTK-aware scaling: scale the base theta
    # This is the simplified YaRN approach (NTK-by-parts is more complex)
    scaled_base = base * (yarn_factor ** (dim / (dim - 2)))
    inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    inv_freq = inv_freq.to(rotary.inv_freq.device)

    old_inv_freq = rotary.inv_freq.float().cpu()
    rotary.inv_freq = torch.nn.Parameter(inv_freq, requires_grad=False)
    new_inv_freq = rotary.inv_freq.float().cpu()

    diff = (old_inv_freq - new_inv_freq).abs()
    n_changed = (diff > 1e-8).sum().item()
    print(f"  Manual patch: {n_changed}/{len(diff)} inv_freq dims changed", flush=True)
    print(f"  Max diff: {diff.max().item():.6e}", flush=True)


def load_model(
    base_model_name: str,
    lora_ckpt_dir: str = None,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    device: str = "cuda",
    torch_dtype=torch.bfloat16,
):
    """Load model with optional YaRN or LoRA.

    Returns (model, tokenizer, config_description).
    """
    print(f"Loading tokenizer: {base_model_name}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s", flush=True)

    config_desc = "vanilla"
    config = AutoConfig.from_pretrained(base_model_name)
    model_kwargs = {
        "device_map": device,
        "torch_dtype": torch_dtype,
    }

    # Apply YaRN if requested (follows official Qwen docs exactly)
    # Ref: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts
    if enable_yarn:
        print(f"Enabling YaRN with factor={yarn_factor}", flush=True)
        config.rope_scaling = {
            "type": "yarn",
            "factor": yarn_factor,
            "original_max_position_embeddings": config.max_position_embeddings,
        }
        print(f"  rope_scaling = {config.rope_scaling}", flush=True)
        model_kwargs["config"] = config
        config_desc = f"yarn_factor{yarn_factor}"

    print(f"Loading base model: {base_model_name}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    # Verify YaRN was actually applied by checking rope_type
    if enable_yarn:
        rotary = getattr(model.model, "rotary_emb", None)
        if rotary is None:
            rotary = model.model.layers[0].self_attn.rotary_emb
        actual_rope_type = getattr(rotary, "rope_type", "unknown")
        print(f"  Actual rope_type after loading: {actual_rope_type}", flush=True)

        if actual_rope_type not in ("yarn",):
            print(f"  WARNING: YaRN not applied via config! Patching inv_freq manually...", flush=True)
            _apply_yarn_manual(model, yarn_factor, config)
            print(f"  Manual YaRN patch applied.", flush=True)

    # Apply LoRA if provided
    if lora_ckpt_dir and os.path.exists(lora_ckpt_dir):
        from peft import PeftModel
        print(f"Loading LoRA adapter: {lora_ckpt_dir}", flush=True)
        t0 = time.time()
        model = PeftModel.from_pretrained(model, lora_ckpt_dir)
        print(f"  LoRA loaded in {time.time()-t0:.1f}s", flush=True)
        print("Merging LoRA weights...", flush=True)
        t0 = time.time()
        model = model.merge_and_unload()
        print(f"  Merge done in {time.time()-t0:.1f}s", flush=True)
        config_desc = f"lora_{os.path.basename(lora_ckpt_dir)}"

    model.eval()

    # Always run RoPE diagnostics so we can compare vanilla vs YaRN
    _diagnose_rope(model, config, enable_yarn, yarn_factor)

    return model, tokenizer, config_desc


def evaluate_mrcr(
    model,
    tokenizer,
    test_data: list[dict],
    max_new_tokens: int = 2048,
) -> dict:
    """Run inference and grade each MRCR example.

    Args:
        model: Model ready for inference.
        tokenizer: Tokenizer matching the model.
        test_data: List of MRCR samples (with prompt, answer, random_string_to_prepend).
        max_new_tokens: Max tokens to generate.

    Returns:
        Dict with per-bin results, overall score, and raw predictions.
    """
    tokenizer.padding_side = "left"

    all_predictions = []
    total_score = 0.0
    total_examples = 0
    bin_scores = {}  # bin_label -> list of scores

    for i, sample in enumerate(test_data):
        # Parse the message list from the prompt
        messages = json.loads(sample["prompt"])

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(text, return_tensors="pt")
        input_ids = input_ids.to(model.device)
        prompt_len = input_ids.shape[1]

        # Generate
        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0

        generated_ids = output_ids[0, prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Grade
        score = grade_mrcr(
            response,
            sample["answer"],
            sample["random_string_to_prepend"],
        )

        total_score += score
        total_examples += 1

        # Track per-bin
        bin_label = sample.get("bin_label", "unknown")
        if bin_label not in bin_scores:
            bin_scores[bin_label] = []
        bin_scores[bin_label].append(score)

        # Record prediction
        prediction = {
            "index": i,
            "bin_label": bin_label,
            "token_count": sample.get("token_count_qwen", prompt_len),
            "n_needles": sample.get("n_needles", -1),
            "score": score,
            "gen_time_s": round(gen_time, 1),
            "prompt_tokens": prompt_len,
            "generated_tokens": len(generated_ids),
            "response_preview": response[:300],
            "answer_preview": sample["answer"][:300],
        }
        all_predictions.append(prediction)

        # Progress
        if (i + 1) % 5 == 0 or i == 0:
            avg_so_far = total_score / total_examples
            print(
                f"  [{i+1}/{len(test_data)}] score={score:.3f}  "
                f"avg={avg_so_far:.3f}  "
                f"tokens={prompt_len}  "
                f"gen_time={gen_time:.1f}s",
                flush=True,
            )

    # Aggregate per-bin
    per_bin = {}
    for bin_label, scores in sorted(bin_scores.items()):
        per_bin[bin_label] = {
            "mean_score": sum(scores) / len(scores),
            "num_samples": len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "num_perfect": sum(1 for s in scores if s == 1.0),
            "num_zero": sum(1 for s in scores if s == 0.0),
        }

    overall_score = total_score / total_examples if total_examples > 0 else 0.0

    return {
        "overall_score": overall_score,
        "total_examples": total_examples,
        "per_bin": per_bin,
        "predictions": all_predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MRCR on Qwen2.5")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--lora-ckpt", type=str, default=None,
                        help="Path to LoRA checkpoint (optional)")
    parser.add_argument("--enable-yarn", action="store_true",
                        help="Enable YaRN RoPE scaling")
    parser.add_argument("--yarn-factor", type=float, default=4.0,
                        help="YaRN scaling factor (default: 4.0 = 4x context)")
    parser.add_argument("--test-file", type=str, required=True,
                        help="Path to test JSON file (from prepare_mrcr_data.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens to generate per example")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Limit number of examples (0 = all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda, cpu)")
    parser.add_argument("--torch-dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Model precision")
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.torch_dtype]

    print("=" * 70)
    print("MRCR Evaluation")
    print("=" * 70)
    print(f"  Base model:      {args.base_model}")
    print(f"  LoRA checkpoint: {args.lora_ckpt or '(none)'}")
    print(f"  YaRN enabled:    {args.enable_yarn} (factor={args.yarn_factor})")
    print(f"  Test file:       {args.test_file}")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print(f"  Device:          {args.device}")
    print(f"  Precision:       {args.torch_dtype}")
    print(f"  Timestamp:       {datetime.now().isoformat()}")

    # Load model
    model, tokenizer, config_desc = load_model(
        args.base_model,
        lora_ckpt_dir=args.lora_ckpt,
        enable_yarn=args.enable_yarn,
        yarn_factor=args.yarn_factor,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    with open(args.test_file) as f:
        test_data = json.load(f)

    if args.max_examples > 0:
        test_data = test_data[:args.max_examples]
    print(f"  {len(test_data)} examples loaded")

    # Evaluate
    print(f"\nRunning evaluation ({config_desc})...")
    results = evaluate_mrcr(model, tokenizer, test_data, max_new_tokens=args.max_new_tokens)

    # Report
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Configuration:    {config_desc}")
    print(f"  Overall score:    {results['overall_score']:.4f}")
    print(f"  Total examples:   {results['total_examples']}")

    print(f"\n  Per-bin breakdown:")
    print(f"  {'Bin':<12} {'Score':>7} {'Perfect':>8} {'Zero':>6} {'Count':>6}")
    print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*6} {'-'*6}")
    for bin_label, info in results["per_bin"].items():
        print(
            f"  {bin_label:<12} {info['mean_score']:>7.3f} "
            f"{info['num_perfect']:>8} {info['num_zero']:>6} {info['num_samples']:>6}"
        )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "base_model": args.base_model,
            "lora_ckpt": args.lora_ckpt,
            "enable_yarn": args.enable_yarn,
            "yarn_factor": args.yarn_factor,
            "config_desc": config_desc,
            "test_file": args.test_file,
            "max_new_tokens": args.max_new_tokens,
            "num_examples": results["total_examples"],
        },
        "overall_score": results["overall_score"],
        "per_bin": results["per_bin"],
    }

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    preds_path = os.path.join(args.output_dir, "predictions.json")
    with open(preds_path, "w") as f:
        json.dump(results["predictions"], f, indent=2)
    print(f"  Predictions saved to {preds_path}")

    print(f"\n{'=' * 70}")
    print("Evaluation complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
