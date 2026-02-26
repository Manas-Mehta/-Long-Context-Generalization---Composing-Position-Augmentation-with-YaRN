# YaRN Verification for Qwen2.5-7B-Instruct

**Date:** 2026-02-25
**transformers version:** 5.0.0
**Model:** Qwen/Qwen2.5-7B-Instruct
**Script:** `composable_cot/mrcr_context_extension/scripts/test_yarn_fresh.py`

## Summary

YaRN (Yet another RoPE extensioN) **is working** in HuggingFace transformers 5.0.0 when configured correctly. The root cause of our previous failures was a **config assignment bug** specific to transformers 5.0.0+.

## The Bug

In transformers 5.0.0, the architecture for RoPE config changed:
- **Old (< 5.0):** `rope_theta` is a standalone config attribute (`config.rope_theta = 1000000.0`)
- **New (5.0+):** `rope_theta` lives **inside** the `rope_parameters` dict (`config.rope_parameters = {"rope_theta": 1000000.0, "rope_type": "default"}`)
- `config.rope_scaling` is now a **property alias** for `config.rope_parameters`

When we did:
```python
config.rope_scaling = {"type": "yarn", "factor": 4.0}
```

This **replaced the entire `rope_parameters` dict**, losing `rope_theta`. After `standardize_rope_params()`, `rope_theta` became `None`, causing either:
- A crash (`TypeError: NoneType ** Tensor`)
- Silent fallback to wrong frequencies

## The Fix

**Use `update()` instead of assignment** to preserve existing keys:

```python
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
config.rope_parameters.update({
    "type": "yarn",
    "rope_type": "yarn",
    "factor": 4.0,
})
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, ...)
```

**Alternative** (also works): Include `rope_theta` explicitly:
```python
config.rope_scaling = {"type": "yarn", "factor": 4.0, "rope_theta": 1000000.0}
```

## What `standardize_rope_params()` Does

When `rope_type` is "yarn", it automatically fills in:
- `original_max_position_embeddings` ← `max_position_embeddings` (32768 for Qwen2.5-7B)
- `rope_theta` ← tries `getattr(self, "rope_theta", None)` (which is `None` in v5.0+ if the dict was replaced)

Final standardized config:
```python
{
    "rope_theta": 1000000.0,
    "rope_type": "yarn",
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
}
```

Note: `original_max_position_embeddings` is NOT the vLLM-only parameter here — it's auto-filled by `standardize_rope_params()` from the model's `max_position_embeddings`. You do NOT need to set it manually.

## Test Results

### Phase 1: Pure Math (ROPE_INIT_FUNCTIONS)
- **40/64 frequency dimensions changed** by YaRN
- First 24 dims (high frequency) unchanged — these handle local/short-range patterns
- Last 40 dims (low frequency) modified — these handle long-range dependencies
- `attention_factor`: 1.0 → 1.1386 (matches formula: `0.1 * ln(4.0) + 1.0`)

### Phase 2: Full Model Load
- **inv_freq:** 40/64 dims differ (max diff: 5.23e-04) — **PASS**
- **Logits:** max diff = 1.115, mean diff = 0.183 — **PASS**
- Both predict "Paris" for "The capital of France is" but with different confidence
  - Vanilla: 15.13 (top logit)
  - YaRN: 14.62 (top logit)
- This is expected — YaRN slightly changes short-context behavior but is designed to improve beyond 32K

## How YaRN Works (from transformers source)

From `transformers/modeling_rope_utils.py::_compute_yarn_parameters`:

1. Compute vanilla frequencies: `pos_freqs = base ** (arange(0, dim, 2) / dim)`
2. Compute interpolated frequencies: `inv_freq_interpolation = 1 / (factor * pos_freqs)`
3. Find correction range using `beta_fast=32`, `beta_slow=1`, and `original_max_position_embeddings`
4. Create linear ramp between low and high dims
5. Blend: `inv_freq = interpolation * (1 - ramp) + extrapolation * ramp`

Result:
- **High-frequency dims** (short wavelength, local patterns): kept as-is (extrapolation)
- **Low-frequency dims** (long wavelength, long-range): scaled down by factor (interpolation)
- **Middle dims**: smooth blend

## Version Compatibility

| transformers | `config.rope_scaling = {...}` | `config.rope_parameters.update({...})` |
|-------------|-------------------------------|----------------------------------------|
| 4.45-4.52   | Works (rope_theta is standalone) | N/A (no rope_parameters dict)       |
| 5.0+        | **BROKEN** (loses rope_theta)    | **Works** (preserves rope_theta)     |

## For HPC (transformers 4.52.4)

The HPC runs transformers 4.52.4 where `rope_theta` is a standalone attribute. The simple assignment should work there. But to be safe, the eval script should use the version-agnostic `set_yarn_config()` helper that:
1. Reads `rope_theta` from wherever it lives
2. Uses `update()` if `rope_parameters` is a dict, otherwise uses assignment
3. Ensures `rope_theta` is preserved

## References

- [Qwen2.5-32B YaRN discussion](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/discussions/5) — confirms `original_max_position_embeddings` is for vLLM only
- [transformers #33783](https://github.com/huggingface/transformers/issues/33783) — config format incompatibility
- [transformers PR #36877](https://github.com/huggingface/transformers/pull/36877) — added `original_max_position_embeddings` to accepted keys
- YaRN paper: [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
