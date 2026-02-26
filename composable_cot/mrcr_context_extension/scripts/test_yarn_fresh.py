"""
Fresh YaRN verification script for Qwen2.5-7B-Instruct.
Tests whether HuggingFace transformers YaRN actually changes RoPE frequencies.

Two-phase test:
  Phase 1: Pure math — call transformers' ROPE_INIT_FUNCTIONS directly (no model load)
  Phase 2: Full model — load Qwen with & without YaRN, compare inv_freq + logits

CRITICAL BUG FOUND (transformers 5.0.0):
  In v5.0.0, rope_theta is stored INSIDE rope_parameters dict (not as standalone attr).
  Setting config.rope_scaling = {"type": "yarn", "factor": 4.0} REPLACES the entire
  rope_parameters dict, LOSING rope_theta. This causes rope_theta=None -> crash or
  silently wrong frequencies.

  FIX: Use config.rope_parameters.update() instead of assignment, OR include rope_theta:
    config.rope_scaling = {"type": "yarn", "factor": 4.0, "rope_theta": 1000000.0}

  For older transformers (<5.0), rope_theta is a standalone attribute, so the simple
  assignment works fine.

References:
  - https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/discussions/5
  - https://github.com/huggingface/transformers/issues/33783
  - transformers/modeling_rope_utils.py :: _compute_yarn_parameters
"""

import sys
import math
import torch
import transformers
from transformers import AutoConfig

MODEL = "Qwen/Qwen2.5-7B-Instruct"
YARN_FACTOR = 4.0

print("=" * 70)
print("YaRN VERIFICATION SCRIPT")
print(f"transformers version: {transformers.__version__}")
print(f"Model: {MODEL}")
print(f"YaRN factor: {YARN_FACTOR}")
print("=" * 70)


def set_yarn_config(config, factor):
    """Set YaRN config in a way that works across transformers versions.

    In transformers 5.0.0+, rope_theta lives inside rope_parameters dict.
    Using config.rope_scaling = {...} replaces the whole dict and loses rope_theta.
    Instead, we update the existing dict to preserve rope_theta.
    """
    # Read rope_theta before it might get lost
    rope_theta = None
    if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
        rope_theta = config.rope_parameters.get("rope_theta")
    if rope_theta is None:
        rope_theta = getattr(config, "rope_theta", 10000.0)

    # Update instead of replace to preserve existing keys
    if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
        config.rope_parameters.update({
            "type": "yarn",
            "rope_type": "yarn",
            "factor": factor,
        })
        # Ensure rope_theta is still there
        if "rope_theta" not in config.rope_parameters or config.rope_parameters["rope_theta"] is None:
            config.rope_parameters["rope_theta"] = rope_theta
    else:
        # Fallback for older transformers
        config.rope_scaling = {"type": "yarn", "factor": factor}

    return config


# ─────────────────────────────────────────────────────────────────────
# PHASE 1: Pure math test — no model loading needed
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 1: Pure math test (ROPE_INIT_FUNCTIONS)")
print("=" * 70)

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Load config
config_vanilla = AutoConfig.from_pretrained(MODEL)
print(f"\nOriginal config:")
print(f"  max_position_embeddings = {config_vanilla.max_position_embeddings}")
print(f"  rope_parameters = {config_vanilla.rope_parameters}")
head_dim = getattr(config_vanilla, "head_dim", config_vanilla.hidden_size // config_vanilla.num_attention_heads)
print(f"  head_dim = {head_dim}")

# --- Vanilla inv_freq (manual computation, since "default" not in ROPE_INIT_FUNCTIONS) ---
rope_theta = config_vanilla.rope_parameters.get("rope_theta", 10000.0)
dim = head_dim
inv_freq_vanilla = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
print(f"\nVanilla RoPE (manual):")
print(f"  rope_theta = {rope_theta}")
print(f"  inv_freq shape: {inv_freq_vanilla.shape}")
print(f"  inv_freq first 5: {inv_freq_vanilla[:5].tolist()}")
print(f"  inv_freq last 5:  {inv_freq_vanilla[-5:].tolist()}")

# --- YaRN inv_freq ---
config_yarn = AutoConfig.from_pretrained(MODEL)
set_yarn_config(config_yarn, YARN_FACTOR)
print(f"\nYaRN config (after set_yarn_config):")
print(f"  rope_parameters = {config_yarn.rope_parameters}")

config_yarn.standardize_rope_params()
print(f"  After standardize_rope_params():")
print(f"  rope_parameters = {config_yarn.rope_parameters}")

yarn_fn = ROPE_INIT_FUNCTIONS["yarn"]
inv_freq_yarn, attn_factor_yarn = yarn_fn(config_yarn, device="cpu")
print(f"\nYaRN RoPE:")
print(f"  inv_freq shape: {inv_freq_yarn.shape}")
print(f"  attention_factor: {attn_factor_yarn}")
print(f"  inv_freq first 5: {inv_freq_yarn[:5].tolist()}")
print(f"  inv_freq last 5:  {inv_freq_yarn[-5:].tolist()}")

# --- Compare ---
diff = (inv_freq_vanilla - inv_freq_yarn).abs()
n_total = len(diff)
n_changed = (diff > 1e-8).sum().item()
n_same = n_total - n_changed

print(f"\n{'─' * 40}")
print(f"  PHASE 1 COMPARISON:")
print(f"  Total dims: {n_total}")
print(f"  Dims changed by YaRN: {n_changed}")
print(f"  Dims unchanged: {n_same}")
print(f"  Max absolute diff: {diff.max().item():.6e}")
print(f"  Mean absolute diff: {diff.mean().item():.6e}")
print(f"  Attention factor: 1.0 (vanilla) -> {attn_factor_yarn:.6f} (YaRN)")

expected_attn = 0.1 * math.log(YARN_FACTOR) + 1.0
print(f"  Expected YaRN attention_factor (0.1*ln({YARN_FACTOR})+1): {expected_attn:.6f}")

if n_changed > 0:
    print(f"\n  PHASE 1 PASSED: YaRN changes {n_changed}/{n_total} frequency dimensions")
    changed_mask = diff > 1e-8
    changed_idx = torch.where(changed_mask)[0].tolist()
    print(f"  Changed dim indices: {changed_idx[:20]}{'...' if n_changed > 20 else ''}")
    ratio = inv_freq_vanilla[changed_mask] / inv_freq_yarn[changed_mask]
    print(f"  Vanilla/YaRN ratio for changed dims (first 10): {[f'{r:.4f}' for r in ratio[:10].tolist()]}")
    print(f"  (ratio > 1 means YaRN has lower freq = longer wavelength = handles longer context)")
else:
    print(f"\n  PHASE 1 FAILED: YaRN produced IDENTICAL frequencies to vanilla!")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# PHASE 2: Full model test — load actual model
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2: Full model test (load Qwen with & without YaRN)")
print("=" * 70)

from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

tokenizer = AutoTokenizer.from_pretrained(MODEL)
test_text = "The capital of France is"
input_ids = tokenizer(test_text, return_tensors="pt")["input_ids"]
print(f"\nTest input: '{test_text}'")
print(f"Token IDs: {input_ids[0].tolist()}")


def get_model_inv_freq(model):
    """Extract inv_freq from the model's rotary embedding, version-agnostic."""
    # Try shared rotary embedding (transformers v5+)
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        # Try per-layer (older transformers)
        layer0 = model.model.layers[0].self_attn
        rotary = getattr(layer0, "rotary_emb", None) or getattr(layer0, "rotary_fn", None)
    if rotary is None:
        return None, "unknown", {}, "None"

    inv_freq = getattr(rotary, "inv_freq", None)
    rope_type = getattr(rotary, "rope_type", "unknown")

    extra = {}
    for attr in ["scaling_factor", "attention_scaling", "max_seq_len_cached",
                  "original_max_position_embeddings", "config"]:
        val = getattr(rotary, attr, None)
        if val is not None and attr != "config":
            extra[attr] = val

    return inv_freq, rope_type, extra, rotary.__class__.__name__


# --- Load vanilla model ---
print("\n--- Loading VANILLA model ---")
model_v = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map="cpu")
model_v.eval()
inv_v, rope_type_v, extra_v, class_v = get_model_inv_freq(model_v)
print(f"  Rotary class: {class_v}")
print(f"  rope_type: {rope_type_v}")
print(f"  Extra attrs: {extra_v}")
if inv_v is not None:
    print(f"  inv_freq shape: {inv_v.shape}")
    print(f"  inv_freq first 5: {inv_v[:5].tolist()}")
    print(f"  inv_freq last 5: {inv_v[-5:].tolist()}")
else:
    print(f"  inv_freq: None (not stored as buffer)")

with torch.no_grad():
    out_v = model_v(input_ids)
    logits_v = out_v.logits[0, -1, :].clone()
    top5_v = torch.topk(logits_v, 5)
    print(f"  Logits shape: {out_v.logits.shape}")
    print(f"  Last token top-5 values: {top5_v.values.tolist()}")
    print(f"  Last token top-5 tokens: {[tokenizer.decode(t) for t in top5_v.indices.tolist()]}")

del model_v, out_v
gc.collect()

# --- Load YaRN model ---
print("\n--- Loading YaRN model ---")
config_y = AutoConfig.from_pretrained(MODEL)
set_yarn_config(config_y, YARN_FACTOR)
print(f"  Config rope_parameters: {config_y.rope_parameters}")

model_y = AutoModelForCausalLM.from_pretrained(MODEL, config=config_y, torch_dtype=torch.float32, device_map="cpu")
model_y.eval()
inv_y, rope_type_y, extra_y, class_y = get_model_inv_freq(model_y)
print(f"  Rotary class: {class_y}")
print(f"  rope_type: {rope_type_y}")
print(f"  Extra attrs: {extra_y}")
if inv_y is not None:
    print(f"  inv_freq shape: {inv_y.shape}")
    print(f"  inv_freq first 5: {inv_y[:5].tolist()}")
    print(f"  inv_freq last 5: {inv_y[-5:].tolist()}")
else:
    print(f"  inv_freq: None (not stored as buffer)")

with torch.no_grad():
    out_y = model_y(input_ids)
    logits_y = out_y.logits[0, -1, :].clone()
    top5_y = torch.topk(logits_y, 5)
    print(f"  Logits shape: {out_y.logits.shape}")
    print(f"  Last token top-5 values: {top5_y.values.tolist()}")
    print(f"  Last token top-5 tokens: {[tokenizer.decode(t) for t in top5_y.indices.tolist()]}")

# ─────────────────────────────────────────────────────────────────────
# PHASE 2 RESULTS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 2 RESULTS: Model comparison")
print("=" * 70)

inv_freq_pass = None
if inv_v is not None and inv_y is not None:
    model_diff = (inv_v.float().cpu() - inv_y.float().cpu()).abs()
    model_n_changed = (model_diff > 1e-8).sum().item()
    print(f"\n  inv_freq comparison:")
    print(f"    Dims changed: {model_n_changed}/{len(model_diff)}")
    print(f"    Max diff: {model_diff.max().item():.6e}")
    inv_freq_pass = model_n_changed > 0
    if inv_freq_pass:
        print(f"    PASS: Model inv_freq DIFFERS between vanilla and YaRN")
    else:
        print(f"    FAIL: Model inv_freq IDENTICAL — YaRN not applied to model!")
else:
    print(f"\n  Could not compare inv_freq (one or both are None)")
    print(f"    Vanilla inv_freq: {'exists' if inv_v is not None else 'None'}")
    print(f"    YaRN inv_freq: {'exists' if inv_y is not None else 'None'}")

logit_diff = (logits_v - logits_y).abs()
logit_pass = logit_diff.max().item() > 0.01
print(f"\n  Logit comparison (last token):")
print(f"    Max logit diff: {logit_diff.max().item():.6f}")
print(f"    Mean logit diff: {logit_diff.mean().item():.6f}")
print(f"    Same top-1 prediction: {top5_v.indices[0].item() == top5_y.indices[0].item()}")
if logit_pass:
    print(f"    PASS: Logits DIFFER between vanilla and YaRN")
else:
    print(f"    FAIL: Logits nearly IDENTICAL — YaRN may not be working!")

# ─────────────────────────────────────────────────────────────────────
# OVERALL VERDICT
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("OVERALL VERDICT")
print("=" * 70)
phase1_pass = n_changed > 0

print(f"  Phase 1 (math):       {'PASS' if phase1_pass else 'FAIL'} ({n_changed}/{n_total} dims changed)")
if inv_freq_pass is not None:
    print(f"  Phase 2 (inv_freq):   {'PASS' if inv_freq_pass else 'FAIL'}")
else:
    print(f"  Phase 2 (inv_freq):   SKIPPED (inv_freq not stored as buffer)")
print(f"  Phase 2 (logits):     {'PASS' if logit_pass else 'FAIL'} (max diff={logit_diff.max().item():.6f})")

all_pass = phase1_pass and logit_pass and (inv_freq_pass is None or inv_freq_pass)
if all_pass:
    print(f"\n  YaRN IS WORKING in transformers {transformers.__version__}")
    print(f"  Config method: update rope_parameters dict (preserves rope_theta)")
    print(f"  Key: Do NOT use config.rope_scaling = {{...}} in transformers 5.0+")
    print(f"       Use config.rope_parameters.update({{...}}) instead")
else:
    print(f"\n  YaRN may not be working correctly. Check details above.")

del model_y, out_y
gc.collect()
