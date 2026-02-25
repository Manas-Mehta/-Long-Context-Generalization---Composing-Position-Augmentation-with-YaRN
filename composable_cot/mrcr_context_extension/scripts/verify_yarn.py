#!/usr/bin/env python
"""Diagnostic script to verify YaRN is actually applied to Qwen2.5.

Loads model once as vanilla, extracts inv_freq, deletes it.
Then loads with YaRN, extracts inv_freq, compares.

Usage:
    # Quick check (CPU, no generation):
    python composable_cot/mrcr_context_extension/scripts/verify_yarn.py

    # Full check with generation (GPU):
    python composable_cot/mrcr_context_extension/scripts/verify_yarn.py --device cuda --generate
"""

import argparse
import gc
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def get_inv_freq(model):
    """Extract inv_freq from model's rotary embedding (handles both locations)."""
    # Try shared rotary_emb first (v5+, some late v4), then per-layer (older v4)
    if hasattr(model.model, "rotary_emb"):
        rotary = model.model.rotary_emb
        print(f"  Found rotary_emb at model.model.rotary_emb")
    else:
        rotary = model.model.layers[0].self_attn.rotary_emb
        print(f"  Found rotary_emb at model.model.layers[0].self_attn.rotary_emb")
    rope_type = getattr(rotary, "rope_type", "unknown")
    inv_freq = rotary.inv_freq.float().cpu().clone()
    return inv_freq, rope_type


def apply_yarn_config(base_model_name, yarn_factor):
    """Create YaRN config following official Qwen docs."""
    config = AutoConfig.from_pretrained(base_model_name)
    # Exactly as specified by Qwen:
    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts
    config.rope_scaling = {
        "type": "yarn",
        "factor": yarn_factor,
        "original_max_position_embeddings": config.max_position_embeddings,
    }
    print(f"  rope_scaling = {config.rope_scaling}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Verify YaRN is applied")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--yarn-factor", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--generate", action="store_true",
                        help="Also compare generated text (slower, needs GPU)")
    args = parser.parse_args()

    dtype = torch.bfloat16

    # === Step 1: Load vanilla, get inv_freq ===
    print("\n" + "=" * 60)
    print("STEP 1: Loading VANILLA model")
    print("=" * 60)
    vanilla_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map=args.device, dtype=dtype,
    )
    vanilla_inv_freq, vanilla_rope_type = get_inv_freq(vanilla_model)
    print(f"  rope_type: {vanilla_rope_type}")
    print(f"  inv_freq shape: {vanilla_inv_freq.shape}")
    print(f"  inv_freq[:5]: {[f'{x:.6f}' for x in vanilla_inv_freq[:5].tolist()]}")
    print(f"  inv_freq[-5:]: {[f'{x:.10f}' for x in vanilla_inv_freq[-5:].tolist()]}")

    # Optionally get logits
    vanilla_logits = None
    if args.generate:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        test_text = "The capital of France is"
        input_ids = tokenizer.encode(test_text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            vanilla_logits = vanilla_model(input_ids).logits[0, -1, :].float().cpu()
        vanilla_top = vanilla_logits.argmax().item()
        print(f"  Top predicted token: {tokenizer.decode([vanilla_top])!r}")

    # Free memory
    del vanilla_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model deleted, memory freed")

    # === Step 2: Load YaRN, get inv_freq ===
    print("\n" + "=" * 60)
    print(f"STEP 2: Loading YaRN model (factor={args.yarn_factor})")
    print("=" * 60)
    yarn_config = apply_yarn_config(args.base_model, args.yarn_factor)
    yarn_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, config=yarn_config, device_map=args.device, dtype=dtype,
    )
    yarn_inv_freq, yarn_rope_type = get_inv_freq(yarn_model)
    print(f"  rope_type: {yarn_rope_type}")
    print(f"  inv_freq shape: {yarn_inv_freq.shape}")
    print(f"  inv_freq[:5]: {[f'{x:.6f}' for x in yarn_inv_freq[:5].tolist()]}")
    print(f"  inv_freq[-5:]: {[f'{x:.10f}' for x in yarn_inv_freq[-5:].tolist()]}")

    yarn_logits = None
    if args.generate and vanilla_logits is not None:
        input_ids = tokenizer.encode(test_text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            yarn_logits = yarn_model(input_ids).logits[0, -1, :].float().cpu()
        yarn_top = yarn_logits.argmax().item()
        print(f"  Top predicted token: {tokenizer.decode([yarn_top])!r}")

    del yarn_model
    gc.collect()

    # === Step 3: Compare ===
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    print(f"\n  Vanilla rope_type: {vanilla_rope_type}")
    print(f"  YaRN rope_type:    {yarn_rope_type}")
    type_match = vanilla_rope_type == yarn_rope_type
    print(f"  rope_type changed:  {'NO [FAIL]' if type_match else 'YES [PASS]'}")

    inv_freq_identical = torch.allclose(vanilla_inv_freq, yarn_inv_freq, atol=1e-8)
    if inv_freq_identical:
        print(f"\n  inv_freq:  IDENTICAL [FAIL] — YaRN NOT applied!")
    else:
        diff = (vanilla_inv_freq - yarn_inv_freq).abs()
        n_changed = (diff > 1e-8).sum().item()
        print(f"\n  inv_freq:  DIFFER [PASS] — YaRN IS applied!")
        print(f"  Dimensions changed: {n_changed}/{len(diff)}")
        print(f"  Max absolute diff:  {diff.max().item():.6e}")
        print(f"  Mean absolute diff: {diff.mean().item():.6e}")

    if vanilla_logits is not None and yarn_logits is not None:
        logits_identical = torch.allclose(vanilla_logits, yarn_logits, atol=1e-4)
        if logits_identical:
            print(f"\n  Logits:    IDENTICAL [FAIL]")
        else:
            logit_diff = (vanilla_logits - yarn_logits).abs()
            print(f"\n  Logits:    DIFFER [PASS]")
            print(f"  Max logit diff:  {logit_diff.max().item():.4f}")
            print(f"  Mean logit diff: {logit_diff.mean().item():.6f}")

    # === Verdict ===
    print("\n" + "=" * 60)
    if not inv_freq_identical and not type_match:
        print("VERDICT: YaRN is WORKING correctly!")
    elif inv_freq_identical:
        print("VERDICT: YaRN is BROKEN — inv_freq unchanged!")
    else:
        print("VERDICT: PARTIALLY working — check details above")
    print("=" * 60)


if __name__ == "__main__":
    main()
