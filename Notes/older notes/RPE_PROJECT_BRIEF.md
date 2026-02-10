# PROJECT RPE: Randomized Positional Encodings for Decoder-Only Models

## Context
Research project at TAUR Labs NYU. Implementing Randomized Positional Encodings (RPE) from DeepMind's paper (arXiv:2305.16843) for decoder-only models, then integrating with Composable CoT framework.

## Key References
- RPE Paper: https://arxiv.org/abs/2305.16843
- RPE Code (JAX, encoder-only): https://github.com/google-deepmind/randomized_positional_encodings
- Composable CoT: https://github.com/fc2869/composable_cot

## Core RPE Algorithm
```python
# Instead of position_ids = [0, 1, 2, ..., N-1]
# Sample N random integers from [0, L) where L >> N (e.g., L=8192)
# SORT them (critical for causality)
# Use as position_ids: e.g., [4, 15, 89, 203, ...]
```

This teaches the model to rely on relative order, not absolute positions, enabling length generalization.

## Design Decision: Encoding-Agnostic
RPE operates at the **position_ids level** BEFORE any encoding (RoPE, ALiBi, etc.):
```
Tokens → Standard IDs [0,1,2,...] → RPE Transform → Randomized IDs → RoPE/ALiBi → Attention
```
This makes it work with any model that accepts position_ids.

## Current Phase: Phase 1 - Standalone RPE & Validation

### Milestone 1.1: Environment Setup
```bash
# Clone repos
git clone https://github.com/fc2869/composable_cot.git
git clone https://github.com/google-deepmind/randomized_positional_encodings.git

# Create environment
conda create -n rpe python=3.11
conda activate rpe
pip install torch transformers accelerate datasets pytest

# Download Qwen 2.5 1.5B
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B'); AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')"
```

### Milestone 1.2: Core RPE Module
Create `rpe/core.py` with:
- `RandomizedPositionalEncoding` class
- `get_randomized_positions(seq_length, max_sim_length=8192)` → sorted random tensor
- `transform_position_ids(position_ids, training=True)` → randomized or passthrough

### Milestone 1.3: Model Patching
Create `rpe/patching.py` with:
- `RPEPatcher` class that monkey-patches HuggingFace models
- Intercepts forward pass, replaces position_ids with randomized ones
- `patch()` and `unpatch()` methods

### Milestone 1.4: Sanity Checks
Create `rpe/tests/test_rpe.py`:
- Position IDs: sorted, unique, in range [0, L), correct length
- Tensors: shapes unchanged, no NaN/Inf, reasonable logit range
- Determinism: same seed → same output
- Behavior: training mode randomizes, inference mode sequential (configurable)

### Milestone 1.5: Reverse String Task
Reproduce DeepMind's evaluation:
- Task: "reverse: abcde" → "edcba"
- Train lengths: 1-40, Test lengths: 41-100+
- Compare RPE vs baseline on length generalization

## Target File Structure
```
project_root/
├── rpe/
│   ├── __init__.py
│   ├── core.py           # Core RPE algorithm
│   ├── patching.py       # Model patching
│   ├── config.py         # Config dataclasses
│   └── tests/
│       └── test_rpe.py   # All sanity checks
├── scripts/
│   └── eval_reverse_string.py
└── requirements.txt
```

## Hardware
- Development: MacBook Pro M3 Pro (inference only)
- Training: NYU HPC via SLURM

## Model
- Qwen 2.5 1.5B (uses RoPE internally)
- HuggingFace: `Qwen/Qwen2.5-1.5B`

## Current Task
Start with Milestone 1.1 (environment setup), then proceed to 1.2 (core RPE implementation).

## Important Notes
1. We're implementing INFERENCE FIRST to validate the pipeline works
2. Training integration comes in Phase 2
3. Make RPE encoding-agnostic (work with any position_ids-based model)
4. All position IDs MUST be sorted after randomization (causality requirement)
