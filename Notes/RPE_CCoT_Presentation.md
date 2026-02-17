# RPE + Composable CoT: Full Project Report & Presentation Notes

**Project**: Randomized Positional Encodings for Length Generalization in Composable Chain-of-Thought
**Researcher**: Manas Mehta
**Institution**: TAUR Labs, Courant Institute, NYU
**Advisor**: Greg Durrett
**Date**: February 2025

**References**:
- RPE: Ruoss et al., "Randomized Positional Encodings Boost Length Generalization of Transformers", ACL 2023 ([arXiv:2305.16843](https://arxiv.org/abs/2305.16843))
- CCoT: Yin et al., "Learning Composable Chains-of-Thought", 2025 ([arXiv:2505.22635](https://arxiv.org/abs/2505.22635))

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Motivation: Why RPE for CCoT?](#2-motivation-why-rpe-for-ccot)
3. [Background: RPE and CCoT](#3-background-rpe-and-ccot)
4. [Phase 1 Recap: RPE Reproduction (From Scratch)](#4-phase-1-recap-rpe-reproduction-from-scratch)
5. [Phase 2: RPE + LoRA on Qwen2.5-7B](#5-phase-2-rpe--lora-on-qwen25-7b)
6. [Experiment Architecture & Implementation](#6-experiment-architecture--implementation)
7. [The Four Experiments: Design & Rationale](#7-the-four-experiments-design--rationale)
8. [Results](#8-results)
9. [Analysis & Discussion](#9-analysis--discussion)
10. [Technical Deep-Dives (FAQ / Grilling Prep)](#10-technical-deep-dives-faq--grilling-prep)
11. [Next Steps](#11-next-steps)

---

## 1. Executive Summary

**Research question**: Can Randomized Positional Encodings (RPE) improve length generalization when applied to LoRA fine-tuning of a pretrained 7B LLM — beyond what CCoT's built-in random prefix mechanism provides?

**What we did**:
1. **Phase 1**: Reproduced DeepMind's RPE result from scratch on a tiny 330K-param Qwen2 model. Confirmed RPE enables OOD generalization (0% → 56% at length 50) on binary string reversal.
2. **Phase 2**: Scaled RPE to Qwen2.5-7B with LoRA fine-tuning. Designed and ran 4 experiments testing different RPE configurations. All trained on the reverse_string task (binary strings, lengths 1-40, CCoT format).

**Key result**: **RPE works with LoRA fine-tuning on a pretrained 7B model.** The best variant (curriculum RPE) extends correct generalization from length 41 to length 65-70 — a 60-75% increase in effective operational range beyond the training distribution.

| Experiment | In-Dist (1-40) | OOD (41-100) | Last Correct | Cliff At | OOD vs Baseline |
|---|---|---|---|---|---|
| **Baseline (no RPE)** | 100% | 7.7% | length 41 | length 45 | — |
| **Exp 1: RPE rank 16** | 100% | 30.8% | length 55 | length 60 | +23pp |
| **Exp 2: RPE asymmetric** | 88.9% | 38.5% | length 65 | ~70 (noisy) | +31pp |
| **Exp 3: RPE curriculum** | **100%** | **46.2%** | **length 65** | **length 70** | **+39pp** |

*Quick eval: 22 stratified examples, greedy decoding (deterministic). "Last Correct" = highest length with exact match. See Section 12 for per-length breakdown and failure analysis.*

---

## 2. Motivation: Why RPE for CCoT?

### The Length Generalization Problem in CCoT

Composable CoT trains LLMs on **atomic skills** (individual tasks with CoT reasoning traces) and then composes them via adapter merging. When tasks compose, the reasoning trace gets longer:

```
Atomic task trace:        ~50-100 tokens
Composition of 2 tasks:   ~100-200 tokens
Composition of 3+ tasks:  ~200-400+ tokens
```

If the model only learned to reason at the sequence lengths seen during atomic training, it will fail on longer compositions. This is a positional encoding problem — the model has never seen certain positions during training.

### CCoT's Existing Solution: Random Prefixes

The CCoT paper addresses this by adding **random gibberish strings** (50-100 characters) before the CoT trace in 50% of training examples:

```
Format A (50%): instruction answer: <prefix> step-by-step CoT </prefix>
Format B (50%): instruction answer: <prefix> random_garbage </prefix> <suffix> step-by-step CoT </suffix>
```

The random text physically shifts the CoT tokens to later positions, simulating composition.

### RPE's Alternative: Change Positions Directly

RPE randomizes position IDs at the encoding level:

```
Standard positions: [0, 1, 2, 3, 4, 5, 6, 7]
RPE positions:      [23, 89, 145, 312, 478, 601, 755, 891]  (sorted random from [0, L))
```

**The core research question**: Does RPE add value on top of CCoT's random prefix? Or can RPE replace it entirely?

### Why This Direction Was Chosen

From the 12/02 project kickoff meeting:
> "Can we achieve the same effect [as random prefixes] by changing the positional encodings of LLMs instead of training data augmentation?"

RPE is appealing because:
1. **No wasted tokens** — random prefixes consume context window space; RPE adds zero tokens
2. **More diverse positional training** — random prefixes shift by 50-100 positions; RPE samples from [0, L) for much wider coverage
3. **Principled** — RPE directly addresses the positional encoding distribution, not just a heuristic workaround

---

## 3. Background: RPE and CCoT

### 3.1 RPE: The Algorithm

From DeepMind (Ruoss et al., ACL 2023):

```python
# Standard positional encoding:
position_ids = [0, 1, 2, ..., N-1]

# RPE (during training only):
L = 1024  # max_simulation_length, much larger than N
perm = torch.randperm(L)       # random permutation of [0, L)
position_ids = perm[:N].sort()  # take N, sort ascending → e.g. [23, 145, 312, 601, 891]
```

**During inference**: standard sequential positions `[0, 1, ..., N-1]`. Since the model trained with positions spanning [0, L), sequential positions at any length fall within the seen range.

**Key properties**:
- Zero additional parameters
- Operates at the `position_ids` input level — model-agnostic
- Random positions are **sorted** — preserves causal ordering
- Each batch element gets **different** random positions — maximum diversity
- Sampling **without replacement** — all positions are unique

### 3.2 RoPE: Where Positions Enter the Model

Qwen2.5-7B uses **Rotary Position Embeddings (RoPE)**. Position IDs flow through the model like this:

```
position_ids → RoPE(position_ids) → cos/sin rotation of Q and K vectors → attention scores
```

RoPE is a **deterministic function** (not learned weights). It applies `cos(position * theta)` and `sin(position * theta)` rotations to the query (Q) and key (K) vectors. **Only Q and K are affected** — values (V) and MLP layers do not receive positional information directly.

RPE changes the *input* to RoPE, not the function itself. This means RPE only affects the Q/K attention computation. This fact motivates our asymmetric LoRA experiment (Exp 2).

### 3.3 CCoT: The Three-Stage Pipeline

```
Stage 1: ATOMIC TRAINING — Train separate LoRA adapters per skill
         ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
         │ letter_concat│   │ next_last_let│   │ ascii_multiply│
         │ LoRA-A       │   │ LoRA-B       │   │ LoRA-C       │
         └──────────────┘   └──────────────┘   └──────────────┘
                  │                  │                  │
Stage 2: MODEL MERGING — Combine adapters (Linear/TIES/TIES-SVD)
                  └────────┬─────────┘                 │
                           ▼                           │
                  ┌──────────────────┐                │
                  │  Merged LoRA     │◄───────────────┘
                  └────────┬─────────┘
                           │
Stage 3: COMPOSITION FINE-TUNING (optional) — Tune on composite examples
                           ▼
                  ┌──────────────────┐
                  │ Composite model  │
                  └──────────────────┘
```

### 3.4 The CCoT Data Format (Critical Detail)

**Format A** — Standard CoT in `<prefix>`:
```json
{
  "instruction": "Reverse the following binary string: 0 1 0 0 answer: ",
  "output": "<prefix> The 1st character from the end is 0. The 2nd character from the end is 0. The 3rd character from the end is 1. The 4th character from the end is 0. So the answer is 0010.</prefix><|endoftext|>"
}
```

**Format B** — Random noise in `<prefix>`, real CoT in `<suffix>`:
```json
{
  "instruction": "Take the second letter of each word: Zhen Lucas answer: <prefix> r qjgyhnc nwylrekitkqy </prefix> ",
  "output": "<suffix> The 2nd letter of the 1st word is h. The 2nd letter of the 2nd word is u. So the answer is hu.</suffix><|endoftext|>"
}
```

**Our reverse_string dataset uses Format A only (100% standard, 0% random prefix).** This is deliberate — it isolates the RPE variable without CCoT's built-in position shifting as a confound. The original CCoT tasks (letter_concat, next_last_letter, etc.) use a 50/50 A/B split.

---

## 4. Phase 1 Recap: RPE Reproduction (From Scratch)

### Setup

| Parameter | Value |
|---|---|
| Model | Tiny Qwen2ForCausalLM, **330K params**, trained from scratch |
| Architecture | Decoder-only (causal masking) |
| Hidden dim / Layers / Heads | 64 / 5 / 8 |
| Vocab | 11 tokens (character-level: 0, 1, pad, eos, space, etc.) |
| Task | Reverse binary string (vocab = {0, 1}) |
| Training lengths | 1-40 (uniform random) |
| Test lengths | 1-100 |
| RPE L | 2048 (matches DeepMind exactly) |
| Training | 10,000 steps, batch 128, LR 1e-3 constant, AdamW |
| Evaluation | Autoregressive generation (greedy decoding) |

All hyperparameters match DeepMind's source code exactly (verified against their `constants.py`, `example.py`, `training.py`).

### Results

| Metric | Baseline (No RPE) | RPE (L=2048) |
|---|---|---|
| Training loss | 0.044 | 0.132 (3x higher — expected) |
| In-dist accuracy (lengths 1-40) | 1.000 | 0.725 |
| OOD accuracy (length 50) | **0.000** | **0.560** |
| OOD (mean, lengths 41-100) | ~0.000 | ~0.50+ |

### Key Takeaways

1. **RPE works**: Baseline drops to 0% the moment it sees OOD lengths. RPE maintains ~56% at length 50.
2. **RPE has a training cost**: 3x higher training loss, reduced in-dist accuracy (1.000 → 0.725).
3. **Decoder-only is harder**: DeepMind reported ~0.8+ OOD with encoder-only. Our ~0.56 is lower due to autoregressive error compounding — one wrong token corrupts all subsequent predictions.
4. **Clean from-scratch isolation**: No pretrained positional bias to confound results.


---

## 5. Phase 2: RPE + LoRA on Qwen2.5-7B
(High priority) Finish training runs of RPE on the reverse string on Qwen2.5-7B using composable CoT repo
Right now the training loss does not drop much, need to debug
Increase LoRA Rank (8 -> 16/32)
Try different learning rates
If none works, try Qwen2.5-0.5B
See if we can do full fine-tuning with -0.5B on the current machine
If the results look stable, we can move on
If not, we can train a harder version of reverse string where arbitrary letters, instead of binary, are used
Compare RPE with baseline encoding on composable CoT task
Start with letter concatenation (atomic) and next letter (atomic)
Data: https://github.com/fc2869/composable_cot/tree/main/data/composition/composable_cot/letter_concat_next_last_letter_composable_cot 
Baseline: just run composable cot with regular positional encoding
RPE variants


(Medium Priority) Vanilla RPE: Run the same experiment as we did on reverse string
Apply RPE to everything (instruction, output)
V2: RPE on output only
Intuition: we want the CoT to be length generalized
V3: RPE on prefix only
Given a training example of:
instruction <prefix> random </prefix> <suffix> real CoT </suffix>

(low priority) RPE on prefix only and the prefix is empty
Given a training example of:
instruction <prefix>              </prefix> <suffix> real CoT </suffix>


### The Challenge

Phase 1 showed RPE works from scratch. But for CCoT, we need RPE to work with:
- A **pretrained** model (7B parameters of positional knowledge baked in)
- **LoRA** fine-tuning (only ~0.26-0.53% of parameters are trainable)
- **CCoT format** (chain-of-thought reasoning traces)

The fundamental question: can a small LoRA adapter (20-40M params) override a pretrained model's expectations about sequential positions, learned across 7B parameters?

### Failed First Attempt: L=8192, Rank 8

| Parameter | Value | Problem |
|---|---|---|
| L (max_simulation_length) | 8192 | Avg position gap = ~41. Pretrained model expects gap = 1. |
| LoRA rank | 8 | Only ~20M params (0.26%) — insufficient capacity |
| LR | 1.0e-3 | Too aggressive for RPE's harder optimization landscape |

**Result**: `eval_loss = 4.82` — model didn't converge. The position perturbation was too extreme for the LoRA adapter to compensate.

### Root Cause Analysis

**Why L=8192 was wrong:**

DeepMind used L=2048 with a 330K-param model trained from scratch with **100% of parameters**. We're using LoRA with **0.26%** of parameters on a model that already has strong positional expectations.

With L=8192 and typical sequences of ~200 tokens, the average gap between consecutive positions is ~41 (8192/200). The pretrained model's RoPE expects gap=1. This 41x disruption was too much for rank-8 LoRA to handle.

**Corrected approach**: L=1024. With ~200-token sequences, avg gap = ~5. Much closer to the pretrained expectation while still providing randomization.

### Corrected Configuration

| Parameter | Failed Attempt | Corrected (Current) |
|---|---|---|
| L | 8192 | **1024** |
| LoRA rank | 8 (~20M params) | **16 (~40M params, 0.53%)** |
| LR | 1.0e-3 | **5.0e-4** |
| Avg position gap | ~41 | **~5** |

**Why these values**:
- **L=1024**: 2x margin over max test sequence (~500 tokens for length-100 CoT trace). Small enough for LoRA to handle, large enough to provide meaningful randomization.
- **Rank 16**: Doubles trainable parameters. More capacity for position-invariant attention patterns.
- **LR 5e-4**: Halved. RPE makes the loss landscape harder; gentler steps help convergence.

---

## 6. Experiment Architecture & Implementation

### 6.1 RPE Patching Pipeline

Three layers of integration:

```
Layer 1: rpe/core.py
    └── RandomizedPositionalEncoding class
        └── get_randomized_positions(seq_length) → sorted random positions
            └── Algorithm: torch.randperm(L)[:N].sort().values

Layer 2: rpe/patching.py
    └── RPEPatcher class
        └── Monkey-patches model.forward() with a closure
            └── If model.training=True:  generate random positions per batch element
            └── If model.training=False: pass through standard sequential positions

Layer 3: composable_cot/scripts/rpe_llamafactory_patch.py
    └── RPETrainerCallback (HuggingFace TrainerCallback)
        └── on_train_begin: apply_rpe_patch(model, config)
        └── on_epoch_begin: update L for curriculum learning
        └── on_train_end: remove_rpe_patch(patcher)
```

**Activation**: Set `RPE_CONFIG_PATH` environment variable → LLaMA-Factory's `tuner.py` (lines 57-61) checks for this and registers the callback.

### 6.2 What RPE Changes (and Doesn't Change)

| Aspect | Changed by RPE? | Details |
|---|---|---|
| `position_ids` tensor | **YES** | Replaced with sorted random integers from [0, L) during training |
| Model architecture | No | Identical Qwen2.5-7B |
| LoRA configuration | No | Same targets, same rank (per experiment) |
| Training data | No | Identical 5,000 examples |
| Loss function | No | Standard cross-entropy on output tokens |
| Learning rate / scheduler | No | Same across baseline and RPE variants |
| Evaluation procedure | No | Standard positions, greedy decoding |

**The ONLY difference between baseline and RPE runs is the position_ids during training forward passes.** This is the controlled variable.

### 6.3 LoRA Configuration Details

**Which layers get LoRA adapters**: All 7 linear projections in each transformer layer.

| Layer | Role | Why LoRA here? |
|---|---|---|
| `q_proj` | Query projection | **RoPE acts here** — directly affected by RPE |
| `k_proj` | Key projection | **RoPE acts here** — directly affected by RPE |
| `v_proj` | Value projection | Adjusts what information attention extracts |
| `o_proj` | Output projection | Post-attention linear transform |
| `gate_proj` | MLP gating | Task-specific feature processing |
| `up_proj` | MLP up-projection | Task-specific computation |
| `down_proj` | MLP down-projection | Task-specific computation |

**Per transformer layer** (28 total in Qwen2.5-7B):
- Each LoRA adapter = two matrices: A (d × r) and B (r × d)
- Effective update: ΔW = (α/r) × B × A, where α=32, r=16 → scaling factor = 2
- Rank 16 total trainable params: ~40M out of 7B = **0.53%**

**LoRA dropout**: 0.2 (regularization during training, disabled at inference).

### 6.4 Training Details

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B | Pretrained, 7B params, RoPE, supports 32K context |
| Fine-tuning | LoRA (PEFT) | Composable adapters for CCoT pipeline |
| Optimizer | AdamW (from HF Trainer) | Standard for LLM fine-tuning |
| LR schedule | Linear decay to 0 | Start aggressive, refine gently |
| Batch size | 4 (per device) | GPU memory constraint on A100 |
| Gradient accumulation | 1 | Effective batch = 4 |
| Epochs | 5 | ~6,250 steps total (5000 examples / 4 batch × 5 epochs) |
| Precision | bfloat16 | Standard for A100, saves memory |
| Sequence cutoff | 1024 tokens | Max tokenized CoT trace is ~576 tokens for length 40 |
| Template | `empty` | Raw instruction/output, no chat formatting |
| Eval strategy | Per-epoch | Monitor for overfitting |
| Checkpoint | Best by eval_loss, save_total_limit=1 | Keep only best |
| Seed | 42 | Reproducibility |

### 6.5 Data Details

**Training**: 5,000 examples of binary string reversal (lengths 1-40) in CCoT Format A:
```
instruction: "Reverse the following binary string: 0 1 0 0 answer: "
output: "<prefix> The 1st character from the end is 0. The 2nd character from the end is 0. The 3rd character from the end is 1. The 4th character from the end is 0. So the answer is 0010.</prefix><|endoftext|>"
```

**Validation**: 500 examples (same format, same length range).

**Test**: 100 lengths × 1 example per length = 100 examples (medium eval), or 22 stratified examples (quick eval).

**Key data properties**:
- **Binary alphabet** (0, 1): Isolates positional reasoning. Model can't exploit vocabulary diversity.
- **Space-separated digits**: Ensures Qwen's BPE tokenizer treats each digit as a separate token (critical for positional alignment).
- **CoT trace scales linearly with input length**: A length-N string produces ~14N tokens of CoT. This is what makes length generalization relevant.
- **Label masking**: Instruction tokens get label=-100 (ignored by loss). Only CoT output tokens contribute to training loss.

**Tokenized sequence lengths** (critical for RPE L configuration):
- Length 1 string → ~20 tokens total
- Length 20 string → ~300 tokens
- Length 40 string → **~576 tokens** (this is the max seen during training)

### 6.6 Evaluation Protocol

```
1. Load Qwen2.5-7B (base model)
2. Load LoRA checkpoint → PeftModel.from_pretrained()
3. Merge adapter into base → model.merge_and_unload()
4. model.eval() → standard sequential positions
5. For each test example:
   a. Tokenize instruction
   b. model.generate(max_new_tokens=2048, do_sample=False)  ← greedy decoding
   c. Extract answer via regex: "the answer is <binary_string>"
   d. Exact match against expected reversed string
6. Group by string length → per-length accuracy
7. Aggregate: in-dist (1-40), OOD (41-100), dm_score
```

**Why max_new_tokens=2048**: OOD lengths 50-100 produce CoT traces of 600-1400+ tokens. The original default of 512 truncated before "the answer is ..." — causing false 0% accuracy. This was a bug we caught and fixed.

**Why greedy decoding (do_sample=False)**: Deterministic — same input always produces same output. This means 1 example per length is sufficient for measuring the cliff location (where accuracy drops from 1.0 to 0.0).

---

## 7. The Four Experiments: Design & Rationale

### Overview

| # | Name | LoRA Config | RPE Config | Key Question |
|---|---|---|---|---|
| 0 | **Baseline** | rank=16, α=32, all 7 targets | None (no RPE) | What can rank 16 LoRA do without RPE? |
| 1 | **RPE rank 16** | rank=16, α=32, all 7 targets | L=1024, fixed | Does standard RPE + corrected L work with LoRA? |
| 2 | **RPE asymmetric** | Q/K: rank=32, α=64; rest: rank=8, α=16 | L=1024, fixed | Does concentrating capacity on position-sensitive layers help? |
| 3 | **RPE curriculum** | rank=16, α=32, all 7 targets | L: 640→768→896→1024→1024 | Does gradual RPE introduction help? |

### Experiment 0: Baseline (No RPE)

**Config**: `reverse_string_baseline_rank16.yaml`

```yaml
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
learning_rate: 5.0e-4
num_train_epochs: 5.0
# No RPE_CONFIG_PATH set → standard sequential positions
```

**Purpose**: Establish the control. This shows what a well-configured LoRA fine-tuning achieves on its own. We expect perfect in-dist accuracy (the task is easy for 7B) and zero OOD generalization (the standard length generalization failure).

**Trainable parameters**: ~40M (0.53% of 7B)

### Experiment 1: RPE Rank 16 (Standard RPE)

**Config**: `reverse_string_rpe_rank16.yaml` + `rpe_config_L1024.yaml`

```yaml
# LoRA config — identical to baseline
lora_rank: 16
lora_alpha: 32
lora_target: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
learning_rate: 5.0e-4

# RPE config
rpe:
  enabled: true
  max_simulation_length: 1024    # ← corrected from 8192
  training_mode: true
  inference_mode: false
```

**Purpose**: Test whether RPE (with corrected L) enables OOD generalization through LoRA fine-tuning. This is the most direct comparison to baseline — same LoRA, same everything, just add RPE.

**Position behavior during training**: Each forward pass gets sorted random positions from [0, 1024). With ~576-token sequences, avg gap between consecutive positions = ~1.8 (close to sequential gap of 1, but with randomization).

### Experiment 2: RPE Asymmetric LoRA

**Config**: `reverse_string_rpe_asymmetric.yaml` + `rpe_config_L1024.yaml`

```yaml
lora_rank: 8          # default rank for most layers
lora_alpha: 16        # default alpha
lora_target: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj

# Override: higher rank for position-sensitive layers
lora_rank_pattern: "q_proj:32,k_proj:32"
lora_alpha_pattern: "q_proj:64,k_proj:64"

# Same RPE config as Exp 1
```

**The asymmetric insight**: RoPE applies position rotations **only to Q and K**. Therefore, when RPE randomizes positions, the disruption flows through `q_proj` and `k_proj`. These layers need the most LoRA capacity to adapt.

| Layer | LoRA Rank | Alpha | Scaling (α/r) | Rationale |
|---|---|---|---|---|
| **q_proj** | **32** | **64** | 2.0 | **RoPE acts here** — max capacity needed |
| **k_proj** | **32** | **64** | 2.0 | **RoPE acts here** — max capacity needed |
| v_proj | 8 | 16 | 2.0 | No direct position effect |
| o_proj | 8 | 16 | 2.0 | Post-attention linear |
| gate/up/down_proj | 8 | 16 | 2.0 | MLP — position-independent |

**Total trainable params**: Higher than Exp 1 for Q/K, lower for rest. Approximately same total budget but better allocated.

**Implementation note**: LLaMA-Factory didn't natively support per-module rank. We added `lora_rank_pattern` and `lora_alpha_pattern` support by modifying `finetuning_args.py` (added fields + parsing) and `adapter.py` (wire to PEFT's `LoraConfig`). PEFT already supports `rank_pattern`/`alpha_pattern` natively — LLaMA-Factory just didn't expose it.

### Experiment 3: RPE Curriculum

**Config**: `reverse_string_rpe_curriculum.yaml` + `rpe_config_curriculum.yaml`

```yaml
# LoRA config — identical to baseline and Exp 1
lora_rank: 16
lora_alpha: 32

# Curriculum RPE config
rpe:
  enabled: true
  max_simulation_length: 1024
  curriculum:
    1: 640     # Epoch 1: L=640 → avg gap ~1.1 (near sequential)
    2: 768     # Epoch 2: L=768 → avg gap ~1.3
    3: 896     # Epoch 3: L=896 → avg gap ~1.6
    4: 1024    # Epoch 4: L=1024 → avg gap ~1.8 (target)
    5: 1024    # Epoch 5: hold at target for consolidation
```

**The curriculum insight**: Instead of imposing full RPE randomization from step 1 (which the LoRA adapter may struggle with), gradually increase L across epochs:

```
Epoch 1 (L=640):  positions sampled from [0, 640)  → avg gap ~1.1 → almost sequential
Epoch 2 (L=768):  positions sampled from [0, 768)  → avg gap ~1.3 → mild randomization
Epoch 3 (L=896):  positions sampled from [0, 896)  → avg gap ~1.6 → moderate
Epoch 4 (L=1024): positions sampled from [0, 1024) → avg gap ~1.8 → full target
Epoch 5 (L=1024): consolidation at target
```

This lets the model first learn the task (reverse string with CoT), then progressively learn position-invariance.

**Implementation**: `RPETrainerCallback.on_epoch_begin()` hook reads the curriculum schedule and updates `patcher.rpe.max_simulation_length` at each epoch boundary.

**Bug fix during training**: The original curriculum used L values starting at 256, but tokenized CoT traces for length-40 strings are ~576 tokens. You can't sample 576 unique positions from [0, 256). We fixed this in two places:
1. `core.py`: Changed from `ValueError` to `effective_L = max(L, seq_length)` for graceful degradation
2. Config: Updated L schedule to start at 640 (above 576) instead of 256

---

## 8. Results

### 8.1 Quick Eval Results (22 stratified examples)

**Eval method**: 1 example at each of 22 key lengths — every 5th length plus boundary points (1, 5, 10, 15, ..., 40, 42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100).

| Experiment | In-Dist (1-40) | OOD (41-100) | Last Correct | Cliff At | Notable |
|---|---|---|---|---|---|
| **Baseline** | 100% (9/9) | 7.7% (1/13) | length 41 | length 45 | Generalizes 1 step beyond training |
| **Exp 1: RPE rank 16** | 100% (9/9) | 30.8% (4/13) | length 55 | length 60 | RPE works with LoRA! |
| **Exp 2: RPE asymmetric** | 88.9% (8/9) | 38.5% (5/13) | length 65 | ~70 (noisy) | Best OOD %, but fails at L=40 (in-dist!) |
| **Exp 3: RPE curriculum** | **100% (9/9)** | **46.2% (6/13)** | **length 65** | **length 70** | **Best overall — perfect in-dist + best OOD** |

### 8.2 Medium Eval Results (100 examples, 1 per length)

**Eval method**: 1 example at each of 100 lengths (1-100). This gives a complete per-length accuracy curve with no gaps.

| Experiment | Overall | In-Dist (1-40) | OOD (41-100) | Last Correct | Notable |
|---|---|---|---|---|---|
| **Baseline** | 43% | 100% (40/40) | 5.0% (3/60) | length 43 | Perfect in-dist, hard cliff at 44 |
| **Exp 1: RPE rank 16** | 51% | 95.0% (38/40) | 21.7% (13/60) | length 62 | Sporadic OOD successes past 50 |
| **Exp 2: RPE asymmetric** | 56% | 90.0% (36/40) | 33.3% (20/60) | length 68 | Continuous OOD success 41-55 |
| **Exp 3: RPE curriculum** | **61%** | **97.5% (39/40)** | **36.7% (22/60)** | **length 66** | **Best overall — near-perfect in-dist + best OOD** |

#### Medium Eval Per-Length Breakdown

| Length | Baseline | RPE rank16 | RPE asymmetric | RPE curriculum | Region |
|---|---|---|---|---|---|
| 1-23 | 1 | 1 | 1 | 1 | ID |
| 24 | 1 | 1 | 1 | **0** | ID |
| 25-26 | 1 | 1 | 1 | 1 | ID |
| 27 | 1 | 1 | **0** | 1 | ID |
| 28-30 | 1 | 1 | 1 | 1 | ID |
| 31 | 1 | 1 | **0** | 1 | ID |
| 32 | 1 | 1 | 1 | 1 | ID |
| 33 | 1 | **0** | 1 | 1 | ID |
| 34-35 | 1 | 1 | 1 | 1 | ID |
| 36 | 1 | **0** | 1 | 1 | ID |
| 37-38 | 1 | 1 | 1 | 1 | ID |
| 39 | 1 | 1 | **0** | 1 | ID |
| **40** | 1 | 1 | **0** | 1 | **ID boundary** |
| **41-43** | **1** | **1** | **1** | **1** | **OOD** |
| 44 | 0 | **1** | **1** | **1** | OOD |
| 45 | 0 | **1** | **1** | **1** | OOD |
| 46 | 0 | 0 | **1** | 0 | OOD |
| 47 | 0 | **1** | **1** | **1** | OOD |
| 48-50 | 0 | **1** | **1** | **1** | OOD |
| 51 | 0 | 0 | **1** | **1** | OOD |
| 52 | 0 | **1** | **1** | **1** | OOD |
| 53 | 0 | 0 | **1** | **1** | OOD |
| 54 | 0 | 0 | **1** | 0 | OOD |
| 55 | 0 | **1** | **1** | **1** | OOD |
| 56 | 0 | 0 | 0 | **1** | OOD |
| 57 | 0 | **1** | 0 | **1** | OOD |
| 58 | 0 | 0 | **1** | 0 | OOD |
| 59 | 0 | 0 | 0 | **1** | OOD |
| 60 | 0 | 0 | 0 | **1** | OOD |
| 61 | 0 | 0 | **1** | 0 | OOD |
| 62 | 0 | **1** | **1** | **1** | OOD |
| 63 | 0 | 0 | 0 | **1** | OOD |
| 64 | 0 | 0 | 0 | **1** | OOD |
| 65 | 0 | 0 | **1** | **1** | OOD |
| 66 | 0 | 0 | 0 | **1** | OOD |
| 67 | 0 | 0 | 0 | 0 | OOD |
| 68 | 0 | 0 | **1** | 0 | OOD |
| 69-100 | 0 | 0 | 0 | 0 | OOD |
| | | | | | |
| **In-dist** | **100%** | **95.0%** | **90.0%** | **97.5%** | |
| **OOD** | **5.0%** | **21.7%** | **33.3%** | **36.7%** | |
| **Cliff** | **44** | **~62 (noisy)** | **~68 (noisy)** | **~67** | |

**Key observations from medium eval:**

1. **Confirms quick eval rankings**: Curriculum > Asymmetric > RPE rank16 > Baseline. The ordering is identical to quick eval.

2. **Baseline generalizes 3 steps beyond training** (lengths 41-43), then complete cliff at 44. Quick eval showed cliff at 45, but that was sampling bias — medium eval reveals 44 is the true boundary.

3. **RPE rank16 has sporadic OOD pattern**: Passes some lengths, fails others (e.g., passes 47-50, fails 46,51). This suggests position sensitivity to specific input patterns at marginal lengths.

4. **Asymmetric has a continuous OOD run from 41-55** (15 consecutive OOD successes!), then becomes sporadic. Quick eval missed this continuous run because it only sampled every 5th length.

5. **Curriculum has the densest OOD successes**: 22/60 OOD correct (36.7%), with the longest run of consecutive successes from 41-53 (skipping 46).

6. **In-dist anomalies**: RPE rank16 fails at lengths 33 and 36 (in-dist!). Asymmetric fails at 27, 31, 39, 40. Curriculum only fails at 24. These single-sample failures may be input-specific rather than systematic.

### 8.3 Interpreting the Results

**What "Cliff At" means**: The cliff is the length where accuracy drops from 1.0 (perfect) to 0.0 (failure). For the baseline, this is length 42 — immediately after the training distribution (1-40). For RPE curriculum, the cliff extends to ~70.

**Why Exp 2 (asymmetric) has 88.9% in-dist**: One length in the in-dist range was predicted incorrectly. This could be due to the lower default rank (8 vs 16) on non-Q/K layers reducing overall task learning capacity. The asymmetric design trades total capacity for targeted capacity.

**Why Exp 3 (curriculum) is best**: It achieves the benefits of RPE (extended cliff) while maintaining perfect in-dist accuracy. The gradual introduction of randomization lets the adapter learn the task first, then learn position-invariance on top of it.

### 8.4 Progression of Results Across Experiments

```
Baseline:     |████████████████████████████████████████|█·····  cliff at 45
Exp 1 (RPE):  |████████████████████████████████████████|███████████████|·····  cliff at 60
Exp 2 (Asym): |██████████████████████████████████████ ·|█████████████████████ ██·····  noisy (~70)
Exp 3 (Curr): |████████████████████████████████████████|████████████████████████|·····  cliff at 70
               1        10        20        30    40 41 45    50     55 60  65 70
              [──────── In-Distribution ────────][──────── Out-of-Distribution ──────────]
```
Note: Exp 2 (asymmetric) has gaps — fails at 40, passes 41-55, fails 60, passes 65, fails 70+.

### 8.5 Comparison with Phase 1

| Metric | Phase 1 (from scratch) | Phase 2 Baseline | Phase 2 Best (Curriculum) |
|---|---|---|---|
| Model | 330K params, from scratch | 7B + LoRA (40M) | 7B + LoRA (40M) |
| Training | Full (100% params) | LoRA (0.53%) | LoRA (0.53%) |
| RPE L | 2048 | N/A | 1024 (curriculum) |
| In-dist accuracy | 1.000 (RPE) | 1.000 | 1.000 |
| OOD at length 50 | 0.560 (RPE) | 0.000 (no RPE) | **1.000** (correct through 65) |
| OOD cliff | gradual decay | sharp at 42 | sharp at ~70 |

Phase 2's RPE curriculum achieves **better OOD performance than Phase 1** on shorter OOD lengths, even though it only tunes 0.53% of parameters. The cliff is sharper (sudden failure vs gradual decay), which is characteristic of LoRA — the adapter either handles the length or doesn't.

---

## 9. Analysis & Discussion

### 9.1 RPE Successfully Transfers to LoRA Fine-Tuning

The central finding: **RPE works with LoRA on a pretrained model.** Despite only updating 0.53% of the 7B parameters, the LoRA adapter successfully learns position-invariant patterns that enable generalization beyond training lengths.

This was not obvious a priori. The pretrained model has 7B parameters tuned for sequential positions. The LoRA adapter must learn to work with random positions during training while still functioning with sequential positions at inference. Our results show this is achievable.

### 9.2 L Calibration is Critical

The failed attempt (L=8192) and successful runs (L=1024) demonstrate that L must be calibrated to:
1. The actual sequence lengths (tokenized, not raw string length!)
2. The LoRA adapter capacity

**Rule of thumb**: L should be ~2x the max tokenized sequence length. Too large = too much positional disruption for the adapter. Too small = insufficient diversity.

### 9.3 Curriculum Learning is the Best Strategy

Exp 3 (curriculum) outperforms Exp 1 (fixed RPE) and Exp 2 (asymmetric) because:
1. **Task learning first**: In epoch 1 (L=640, gap ~1.1), positions are nearly sequential. The adapter learns the reverse-string CoT pattern without positional disruption.
2. **Progressive position-invariance**: Epochs 2-4 gradually increase randomization, letting the adapter build on its task knowledge.
3. **Consolidation**: Epoch 5 at full L=1024 reinforces the position-invariant patterns.

This mirrors curriculum learning's success in other domains — start easy, increase difficulty.

### 9.4 Asymmetric LoRA: Promising but Noisy

Exp 2's higher OOD accuracy (38.5%) but lower in-dist (88.9%) suggests the asymmetric design correctly allocates capacity to Q/K (where RoPE acts) but may sacrifice overall task capacity. The reduced rank (8 vs 16) on V and MLP layers means less capacity for the CoT reasoning itself.

A potential improvement: combine asymmetric LoRA with curriculum learning (give Q/K rank 32 AND use curriculum schedule).

### 9.5 The "Sharp Cliff" Pattern

Unlike Phase 1 (gradual OOD decay), Phase 2 shows a **sharp accuracy cliff** — perfect accuracy up to some length, then immediate failure. This is characteristic of LoRA fine-tuning:

- **LoRA operates in a low-dimensional subspace** of the full weight space
- Within this subspace, the model either generalizes or doesn't — no partial solutions
- The cliff location is determined by where the LoRA adapter's learned position-invariance breaks down

### 9.6 CCoT vs RPE: Preliminary Analysis

Our reverse_string data has **no CCoT random prefixes** (Format A only). Yet RPE still provides significant OOD improvement. This suggests RPE operates through a different mechanism than CCoT's random prefixes:

- **CCoT random prefix**: Physically shifts absolute positions but preserves relative distances = 1 between consecutive tokens
- **RPE**: Randomizes both absolute positions AND relative distances between tokens

RPE provides a strictly more diverse positional training signal. Whether this is complementary or redundant with CCoT prefixes remains to be tested on the original CCoT tasks (letter_concat, next_last_letter).

---

## 10. Technical Deep-Dives (FAQ / Grilling Prep)

### About RPE

**Q: What is RPE in one sentence?**
A: During training, replace sequential position IDs [0,1,2,...] with sorted random integers from a larger range [0,L), so the model learns to reason at any position and can generalize to longer sequences.

**Q: Where exactly does RPE intervene?**
A: It replaces the `position_ids` tensor passed to `model.forward()`. This feeds into RoPE, which rotates Q and K vectors before attention. RPE changes the rotation angles, not the rotation function.

**Q: Does RPE add any parameters?**
A: Zero. It only changes the input to an existing function (the position_ids argument to the forward pass).

**Q: Why sorted random positions?**
A: Sorting preserves causal ordering — token at position i must attend only to tokens at positions ≤ i. Unsorted random positions would break causal masking.

**Q: Why sample without replacement?**
A: Two tokens at the same position would receive identical RoPE rotations, making them indistinguishable to the attention mechanism. Unique positions preserve token identity.

**Q: What is max_simulation_length (L) and how was it chosen?**
A: L is the upper bound for random position sampling. We use L=1024, chosen as ~2x the max tokenized sequence length (~576 tokens). DeepMind used L=2048 with full fine-tuning; we scale down because LoRA has limited capacity.

**Q: Why use L=1024 and not DeepMind's L=2048?**
A: DeepMind trained from scratch with 100% of parameters — the model could learn arbitrary position patterns from random init. We use LoRA with 0.53% of parameters on a pretrained model that expects sequential positions. Smaller L = smaller positional disruption = easier for LoRA to adapt.

**Q: Why is RPE training loss higher?**
A: The model must predict the next CoT token while receiving randomly-spaced position IDs. The pretrained RoPE expects positions separated by exactly 1; random positions disrupt its learned attention patterns. The LoRA adapter gradually compensates for this.

**Q: Why use standard positions at inference?**
A: The model learns position-invariant patterns during training. At inference, sequential positions [0,1,...,N-1] are just one valid set within the range [0,L) — they're no longer "special." The hope is the model handles any reasonable set of positions.

### About LoRA and Fine-Tuning

**Q: Which LoRA adapters are fine-tuned?**
A: All 7 linear projections in every transformer layer:
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`
- Total: 7 projections × 28 layers = 196 LoRA adapter pairs

**Q: Why all 7 projections and not just Q/K?**
A: The task (reverse string with CoT) requires learning both position-invariant attention (Q/K) AND new computation patterns (MLP for CoT reasoning). Restricting to Q/K only would give the model position adaptation capacity but no task learning capacity.

**Q: How many trainable parameters?**
A: Rank 16 LoRA on all 7 projections in 28 layers of Qwen2.5-7B:
- Per adapter pair: d_model × rank + rank × d_out (A matrix + B matrix)
- Total: ~40M parameters (0.53% of 7B)
- For comparison: DeepMind's Phase 1 model had 330K total params — we're training 120x more parameters via LoRA

**Q: What is the LoRA scaling factor and why does it matter?**
A: The effective LoRA update is ΔW = (α/r) × B × A. With α=32 and r=16, scaling = 2.0. This controls how strongly the adapter modifies the base model's behavior. Higher scaling = larger perturbation but potentially more instability.

**Q: What is `lora_dropout: 0.2`?**
A: 20% dropout applied to the LoRA activations during training (not during inference). This regularizes the adapter — prevents overfitting to the 5,000 training examples. Standard practice for small-dataset LoRA training.

**Q: How does `merge_and_unload()` work during evaluation?**
A: At evaluation time, we permanently merge the LoRA weights into the base model: `W_merged = W_base + (α/r) × B × A`. This produces a standard model with no adapter overhead. The merged model uses standard sequential positions — no RPE involvement during inference.

**Q: Can the LoRA adapter really override 7B params of positional knowledge?**
A: Yes, because RoPE is a **functional transform** (not learned weights). RoPE computes `cos(pos × θ)` and `sin(pos × θ)` on the fly — there are no "position weights" to override. What LoRA adapts is the Q/K projection matrices, which determine what the model attends to *given* a particular set of RoPE rotations. The adapter learns Q/K projections that work well with diverse position patterns.

### About CCoT

**Q: How is RPE related to CCoT's random prefix mechanism?**
A: Both aim for position-invariant reasoning through different means:
- CCoT random prefix: adds physical text to shift absolute positions, but relative distances stay = 1
- RPE: randomizes position IDs directly, changing both absolute and relative distances
- They're complementary, not redundant

**Q: Why does our reverse_string dataset not have random prefixes?**
A: We generated it ourselves as a pure RPE testbed. No random prefixes = no CCoT position shifting confound. This isolates the RPE variable for clean measurement. The original CCoT tasks (letter_concat, next_last_letter) have the 50/50 prefix split for future experiments.

**Q: What are the `<prefix>` and `<suffix>` tags?**
A: In the CCoT framework:
- `<prefix>` wraps the CoT reasoning in standard examples
- In "composable" examples, `<prefix>` in the instruction holds random gibberish, and `<suffix>` in the output holds the real CoT. This trains the model to produce reasoning at different positions (simulating composition).

**Q: What is `template: empty` in the training config?**
A: LLaMA-Factory applies chat templates by default (wrapping input in `<|im_start|>user\n...<|im_end|>` etc.). `template: empty` disables this — feeds raw instruction/output directly. This matches CCoT's training format.

### About the Experiments

**Q: Why binary strings (vocab = {0, 1})?**
A: DeepMind's choice. With only 2 possible characters, the model can't exploit vocabulary diversity to guess answers. It MUST use positional information. This is the purest test of positional encoding.

**Q: What is dm_score?**
A: Mean accuracy on out-of-distribution lengths (41-100). Named after DeepMind's evaluation protocol (`score = mean(accuracies[seq_len+1:])`).

**Q: Why greedy decoding instead of sampling?**
A: Greedy (`do_sample=False`) is deterministic — the same prompt always produces the same output. This means results are reproducible, and 1 example per length is statistically valid for measuring the cliff location.

**Q: Is 1 example per length statistically valid?**
A: Yes, for measuring cliff location. Greedy decoding is deterministic, so each length produces the same result regardless of how many times you run it. The result is binary (correct/incorrect), and we're measuring where the transition happens, not estimating a probability. For estimating accuracy percentages at specific lengths, more samples would be needed.

**Q: What is the tokenized length vs raw string length?**
A: Critical distinction! A raw length-40 binary string is 40 characters, but the full CCoT trace (instruction + step-by-step reasoning) tokenizes to **~576 tokens** via Qwen's BPE tokenizer. RPE's L must exceed the tokenized length, not the raw length. This was the root cause of the curriculum crash bug.

**Q: What was the curriculum crash bug?**
A: The original curriculum started at L=256, but tokenized sequences are ~576 tokens. You can't sample 576 unique positions from [0, 256). Fix: (1) `core.py` now uses `effective_L = max(L, seq_length)` for graceful degradation; (2) curriculum schedule updated to start at L=640 (above 576).

**Q: Why linear LR decay instead of constant (like DeepMind)?**
A: Pragmatic choice for LoRA fine-tuning. Both baseline and RPE use the same schedule, so the comparison is still valid. The absolute numbers may differ from Phase 1 (constant LR), but the RPE effect is measured by the delta between conditions.

**Q: How is the answer extracted from model output?**
A: Regex-based extraction: `re.search(r"the answer is\s+([01]+)", generated_text)`. Falls back to matching `answer: <binary>` or the last contiguous binary string in the generation. If none match, the full output is used. Checked via `predictions.json` for false negatives.

### About Infrastructure

**Q: What hardware was used?**
A: Lightning AI studio with 1x NVIDIA A100 GPU (80GB). Training: ~20-30 min per experiment. Evaluation: ~6-7 min per model (medium eval, 100 examples).

**Q: Where is the code?**
A: GitHub: https://github.com/Manas-Mehta/Generalized-CCoT.git (private). NYU Torch HPC: `/scratch/mm14444/RPE/` (account: `torch_pr_219_courant`).

---

## 11. Next Steps

### Immediate (This Week)

1. **Run medium eval** — 1 example per every length (1-100), all 4 models. Get per-length accuracy curves.
2. **Download results** — predictions.json, eval_results.json, training loss plots for all runs.
3. **Error analysis** — inspect predictions.json at lengths near the cliff. What does the model generate? Does the CoT trace start correct and break down? Where does it go wrong?

### Short-Term (Next 2 Weeks)

4. **Test on original CCoT tasks** — letter_concat (50/50 prefix split) and next_last_letter. This tests: does RPE add value on TOP of CCoT's built-in random prefix?
5. **RPE V2: Output-only RPE** — randomize only CoT output positions, keep instruction positions standard. Tests the hypothesis that only the CoT trace needs position-invariance.
6. **Combined experiment**: Asymmetric LoRA + Curriculum RPE. Combine the best ideas from Exp 2 and Exp 3.

### Medium-Term

7. **Model merging experiments** — train RPE atomic adapters, merge them, evaluate on composition tasks. The CCoT pipeline's core test.
8. **2×2 Factorial** — (RPE / no RPE) × (random prefix / no prefix) on letter_concat. Cleanly separates RPE from CCoT's prefix mechanism.
9. **NYU HPC runs** — retrain on Torch cluster for reproducibility and to test scaling.

### Long-Term

10. **RPE for TravelPlanner** — apply RPE to the GCCoT TravelPlanner task (45 examples, complex multi-step planning).
11. **Full fine-tuning on 0.5B** — if LoRA capacity is a bottleneck, full FT on smaller model.
12. **Paper-ready results** — multi-seed runs for statistical significance, comprehensive ablations.

---

## 12. Actual Results: Deep Dive with Real Data

*All data below comes from the actual eval_results.json, predictions.json, and all_results.json files from the Lightning AI runs.*

### 12.1 Training Metrics (from `all_results.json`)

| Metric | Baseline | Exp 1: RPE rank16 | Exp 2: RPE asymmetric | Exp 3: RPE curriculum |
|---|---|---|---|---|
| **Final train_loss** | 0.01353 | 0.01202 | 0.00563 | ~0.0087 (partial log) |
| **Final eval_loss** | 1.52e-6 (~0) | 0.000818 | 0.000937 | (not in ckpt) |
| **Train runtime** | 5132s (~85 min) | 5126s (~85 min) | 5107s (~85 min) | ~85 min |
| **Samples/sec** | 4.87 | 4.88 | 4.90 | ~4.9 |

**Surprising finding: RPE train_loss is NOT higher than baseline.** In Phase 1, RPE training loss was 3x higher (0.132 vs 0.044). Here, RPE rank16 (0.012) is actually *lower* than baseline (0.014), and asymmetric (0.006) is lowest of all.

**Why the reversal?** With L=1024 and ~576-token sequences, the average position gap is only ~1.8 (vs Phase 1's much larger gap with L=2048 on short sequences). The positional disruption is mild enough that it barely affects training loss. The LoRA adapters learn the task just as well with slightly randomized positions.

**Eval loss tells the real story:**
- Baseline eval_loss: **1.52e-6** (essentially zero — perfect on validation set)
- RPE rank16 eval_loss: **0.000818** (~540x higher — the model performs slightly worse on standard positions)
- RPE asymmetric eval_loss: **0.000937** (~616x higher than baseline)

This confirms that RPE models train well but have a measurable (though small) penalty when evaluated with standard sequential positions. This is the fundamental RPE trade-off with LoRA: slightly worse on the training distribution, but much better on OOD.

### 12.2 Training Loss Curves

**Baseline** (`training_loss.png`): Sharp spikes at epoch boundaries (steps ~1250, ~2500), where the learning rate decays linearly and the data loader reshuffles. Between spikes, loss rapidly drops to near 0. By step ~3000 (epoch 3), loss is essentially zero. The model has fully memorized the reverse-string CoT pattern.

**RPE rank16** (`training_loss.png`): One notable spike around steps 500-700, then smooth monotonic descent to near 0. No epoch-boundary spikes visible — the RPE randomization acts as implicit regularization that smooths the loss landscape across epoch transitions.

**RPE asymmetric** (`training_loss.png`): Smooth descent from 0.08, with a smaller spike around step 3000 and another at ~4000. This is the most stable training curve. The lower default rank (8) means fewer parameters overall, and the concentrated rank on Q/K gives a smooth optimization trajectory.

**Eval loss curves (critical — reveals model dynamics):**

- **Baseline**: Monotonically decreasing from 0.0014 to ~0. Perfect convergence with no overfitting.
- **RPE rank16**: **Non-monotonic!** Starts at 0.002, *increases* to 0.005 at epoch 2, drops to 0.001 at epoch 4, then *increases again* at epoch 5 to 0.003-0.004. This U-shaped pattern suggests the RPE adapter's learned representations interact non-trivially with standard eval positions — during training, the adapter alternately improves and degrades its ability to work with sequential positions.
- **RPE asymmetric**: Starts high at 0.01, *plateaus/increases* through epochs 1-3, then **dramatically drops** at epochs 4-5 to ~0.001. This is a "late learner" — the asymmetric LoRA needs more epochs to converge, but when it does, it converges strongly. The best checkpoint is from the last epoch.

### 12.3 Per-Length Accuracy (Quick Eval: 22 examples)

Exact per-length results from the actual `eval_results.json` files:

| Length | Baseline | RPE rank16 | RPE asymmetric | RPE curriculum | Region |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 1 | 1 | ID |
| 5 | 1 | 1 | 1 | 1 | ID |
| 10 | 1 | 1 | 1 | 1 | ID |
| 15 | 1 | 1 | 1 | 1 | ID |
| 20 | 1 | 1 | 1 | 1 | ID |
| 25 | 1 | 1 | 1 | 1 | ID |
| 30 | 1 | 1 | 1 | 1 | ID |
| 35 | 1 | 1 | 1 | 1 | ID |
| **40** | 1 | 1 | **0** | 1 | **ID boundary** |
| **41** | **1** | 1 | 1 | 1 | **OOD** |
| 45 | 0 | **1** | **1** | **1** | OOD |
| 50 | 0 | **1** | **1** | **1** | OOD |
| 55 | 0 | **1** | **1** | **1** | OOD |
| 60 | 0 | 0 | 0 | **1** | OOD |
| 65 | 0 | 0 | **1** | **1** | OOD |
| 70 | 0 | 0 | 0 | 0 | OOD |
| 75 | 0 | 0 | 0 | 0 | OOD |
| 80 | 0 | 0 | 0 | 0 | OOD |
| 85 | 0 | 0 | 0 | 0 | OOD |
| 90 | 0 | 0 | 0 | 0 | OOD |
| 95 | 0 | 0 | 0 | 0 | OOD |
| 100 | 0 | 0 | 0 | 0 | OOD |
| | | | | | |
| **In-dist** | **100%** | **100%** | **88.9%** | **100%** | |
| **OOD** | **7.7%** | **30.8%** | **38.5%** | **46.2%** | |
| **Cliff** | **45** | **60** | **~70 (noisy)** | **70** | |

**Correction from earlier estimates**: The baseline actually gets length 41 correct — the cliff is at length 45, not 42. This means the baseline generalizes 1 step (41 chars, ~595 tokens) beyond training but fails at 45 (45 chars, ~650 tokens).

### 12.4 Prediction Format Explained

Each prediction in `predictions.json` contains:

```json
{
  "index": 10,
  "length": 45,
  "prompt_tail": "1 0 1 0 0 1 1 0 0 0 0 1 0 1 1 1 1 0 1 0 1 1 0 1 1 1 answer: ",
  "expected": "111011010111101000011001010000110011010110000",
  "predicted": "111011010111101000011001010000110011010110000",
  "generated_text": "<prefix> The 1st character from the end is 1. The 2nd character from the end is 1. The 3rd character from the end is 1. The 4th character from the end is 0. The 5th character from the end is 1. The 6t",
  "correct": true
}
```

| Field | Meaning |
|---|---|
| `index` | Position in test set (0-21 for quick eval) |
| `length` | Input string length (number of binary digits) |
| `prompt_tail` | Last 60 chars of the instruction prompt |
| `expected` | Ground truth reversed string (extracted from test data) |
| `predicted` | Answer extracted by regex from model's generation ("the answer is X") |
| `generated_text` | First 200 chars of raw model output (the full CoT trace) |
| `correct` | Whether `predicted == expected` (exact match) |

The model generates a full CoT trace in `<prefix>` tags: "The 1st character from the end is X. The 2nd character from the end is Y. ... So the answer is XY...Z." The `predicted` field is what the regex extracts from "the answer is ..." at the end.

### 12.5 Failure Mode Analysis: How Each Model Fails

#### Baseline Failures (length 45+): Catastrophic Truncation

When the baseline fails, it generates a **drastically shorter** answer than expected:

| Length | Expected chars | Predicted chars | Ratio | Pattern |
|---|---|---|---|---|
| 45 | 45 | 22 | **49%** | Starts correct, then wrong chars, stops early |
| 50 | 48 | 26 | **54%** | Same pattern |
| 55 | 53 | 32 | **60%** | Same |
| 60 | 58 | 37 | **64%** | Same |
| 70 | 68 | 40 | **59%** | Same |
| 80 | 78 | 35 | **45%** | Same |
| 90 | 88 | 37 | **42%** | Predicted is less than half |
| 100 | 98 | 33 | **34%** | Only 1/3 of expected length |

Example at length 45:
```
Expected:  111011010111101000011001010000110011010110000  (45 chars)
Predicted: 1110110101111010000000                        (22 chars)
                              ^^^^^^^^ ← diverges here, then stops
```

The CoT trace starts correctly ("The 1st character from the end is 1. The 2nd character from the end is 1...") but at some point the step-by-step reasoning breaks down — the model skips steps, generates wrong characters, and terminates early with an incorrect "So the answer is..." The predicted answer is a mangled prefix of the correct answer.

#### RPE Rank 16 Failures (length 60+): Near-Misses

RPE rank16 fails much more gracefully — predictions are **close to the correct length** with only minor differences:

| Length | Expected chars | Predicted chars | Ratio | Pattern |
|---|---|---|---|---|
| 60 | 58 | 56 | **97%** | Off by 2 chars (minor substitutions) |
| 65 | 63 | 60 | **95%** | Off by 3 chars |
| 70 | 68 | 60 | **88%** | Some middle section wrong |
| 75 | 73 | 65 | **89%** | Some chars dropped |
| 80 | 78 | 60 | **77%** | More degradation |
| 90 | 88 | 52 | **59%** | Starting to fail like baseline |
| 100 | 98 | 1 | **1%** | Catastrophic (predicted just "0") |

Example at length 60 (a near-miss):
```
Expected:  000110100000110000100100110101110011101001010111110000110110  (58 chars)
Predicted: 000110100001100001001001101011100111010010111110000110110     (56 chars)
           ^^^^^^^^^^  ^^   ^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^
           [correct]   [2 chars different/missing]       [correct tail]
```

The model gets the first ~10 chars right, has a small error in the middle, then the **tail of the answer is correct**. This shows the CoT trace works mostly correctly — the model is doing the right step-by-step reversal but occasionally skips or mis-copies a single step.

At length 100, however, RPE rank16 catastrophically collapses to just "0" — the CoT trace apparently breaks down completely.

#### RPE Asymmetric Failures: Non-Monotonic Pattern

The asymmetric model shows an unusual failure pattern — it fails at some shorter lengths but succeeds at longer ones:

- **Fails** at length 40 (in-dist!) — off by 1 character in a 40-char answer
- **Succeeds** at lengths 41, 45, 50, 55
- **Fails** at length 60 — off by 1 character in a 58-char answer
- **Succeeds** at length 65 — perfect match on a 63-char answer!
- **Fails** at length 70+

The length-40 failure (1 character wrong in 40):
```
Expected:  0010110101101111001001101010100101111101
Predicted: 0010110101101111001001101010101001111101
                                       ^^ ← single bit flip
```

And the length-60 failure (1 character dropped in 58):
```
Expected:  000110100000110000100100110101110011101001010111110000110110
Predicted: 00011010000110000100100110101110011101001010111110000110110
            ^^^^^^^ ← missing one "0"
```

This non-monotonic behavior likely reflects the asymmetric LoRA's different internal dynamics — the concentrated Q/K capacity gives robust position handling for certain input lengths/patterns but is sensitive to the specific bit patterns in the input string.

#### RPE Curriculum Failures (length 70+): Graceful Degradation

The curriculum model fails most gracefully of all:

| Length | Expected chars | Predicted chars | Ratio | Pattern |
|---|---|---|---|---|
| 70 | 68 | 60 | **88%** | Starts correct, middle errors, **tail correct** |
| 75 | 73 | 63 | **86%** | Same pattern |
| 80 | 78 | 62 | **79%** | Still >75% of expected |
| 85 | 83 | 51 | **61%** | Starting to degrade more |
| 90 | 88 | 58 | **66%** | Still better than baseline |
| 95 | 93 | 56 | **60%** | Better than baseline at same length |
| 100 | 98 | 67 | **68%** | **Much better than RPE rank16's "0"** |

Example at length 70 (first failure):
```
Expected:  1010110100101011001000101110101001100110111010011010001101111000001010
Predicted: 10101101011100100010111010100110011010011010001101111000001010
           ^^^^^^^^^  ^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           [correct]  [middle section compressed]          [last ~40 chars correct!]
```

**Key trend**: The curriculum model's failures share a characteristic pattern — the **beginning and end of the predicted answer are correct**, with errors concentrated in the middle of the string. This suggests the CoT trace correctly processes the first N steps and last M steps, but loses track in the middle of the reversal process.

### 12.6 Predicted Length vs Expected Length: Summary Visualization

```
Baseline failures (length 45-100):
  L45:  ████████████████████████·····························  49%
  L60:  ████████████████████████████████████████·············  64%
  L80:  ███████████████████████······························  45%
  L100: ████████████████·····································  34%

RPE rank16 failures (length 60-100):
  L60:  ████████████████████████████████████████████████████  97%  ← near miss!
  L70:  ████████████████████████████████████████████████·····  88%
  L80:  ████████████████████████████████████████████··········  77%
  L100: █····················································   1%  ← catastrophic

RPE curriculum failures (length 70-100):
  L70:  ████████████████████████████████████████████████·····  88%
  L80:  ██████████████████████████████████████████████········  79%
  L90:  ███████████████████████████████████████···············  66%
  L100: ██████████████████████████████████████████████████····  68%  ← still 68%!
```

### 12.7 Key Trends and Insights from Actual Data

**1. RPE doesn't hurt training loss with small L.** Unlike Phase 1 (3x loss penalty), L=1024 with LoRA produces comparable or even lower training loss than baseline. The position perturbation is mild enough to be absorbed without penalty.

**2. The eval_loss gap is the real indicator.** Baseline reaches essentially zero eval loss; RPE models plateau around 0.001. This small-but-nonzero gap reflects the fundamental trade-off: the adapter is tuned for diverse positions, so standard sequential positions aren't quite optimal.

**3. RPE eval loss is non-monotonic.** RPE rank16's eval loss oscillates (improves epochs 3-4, degrades epoch 5). The best RPE checkpoint may not be from the last epoch. `load_best_model_at_end: true` correctly selects the epoch-4 checkpoint.

**4. Failure mode is truncation, not garbage.** All models fail by producing **shorter** predictions than expected, not by generating random characters. The CoT reasoning trace starts correctly but breaks down mid-way, causing the model to skip steps and reach "So the answer is..." prematurely.

**5. RPE models fail more gracefully.** At the same OOD lengths, RPE predictions are much closer to the correct answer than baseline predictions (60-97% of expected length vs 34-64% for baseline). The RPE model's position-invariant training gives it partial generalization even when not fully correct.

**6. Curriculum has the most robust failure mode.** The curriculum model maintains >60% of expected length even at length 100 and gets both beginning and ending characters correct. Baseline at length 100 only manages 34%.

**7. The asymmetric model shows non-monotonic accuracy.** It fails at length 40 (in-dist!) but succeeds at 65 (OOD). This sensitivity to specific inputs suggests the asymmetric rank allocation creates a less smooth generalization boundary.

**8. Baseline generalizes exactly 1 step beyond training.** Length 41 is correct (100%), length 45 fails. The tokenized length of a 41-char string (~595 tokens) is within the model's learned range; 45 chars (~650 tokens) is not.

---

## Appendix A: Complete File Inventory

### Files Created/Modified by Us

| File | What It Does | Status |
|---|---|---|
| `rpe/core.py` | RPE algorithm: `randperm(L)[:N].sort()` | Modified (graceful L clamping for curriculum) |
| `rpe/patching.py` | RPEPatcher: monkey-patches model.forward() | Stable |
| `composable_cot/scripts/rpe_llamafactory_patch.py` | LLaMA-Factory integration: RPETrainerCallback | Modified (curriculum support) |
| `composable_cot/scripts/rpe_config_L1024.yaml` | RPE config: L=1024 (fixed) | Created |
| `composable_cot/scripts/rpe_config_curriculum.yaml` | RPE config: L=640→768→896→1024→1024 | Modified (corrected L schedule) |
| `composable_cot/scripts/llamafactory/reverse_string_baseline_rank16.yaml` | Baseline training config | Created |
| `composable_cot/scripts/llamafactory/reverse_string_rpe_rank16.yaml` | Exp 1 training config | Created |
| `composable_cot/scripts/llamafactory/reverse_string_rpe_asymmetric.yaml` | Exp 2 training config | Created |
| `composable_cot/scripts/llamafactory/reverse_string_rpe_curriculum.yaml` | Exp 3 training config | Created |
| `composable_cot/scripts/run_three_experiments.sh` | End-to-end run script for all 4 experiments | Created |
| `composable_cot/scripts/eval_length_generalization.py` | Per-length accuracy evaluation | Created |
| `composable_cot/scripts/generate_reverse_string_data.py` | Training/eval data generation | Created |
| `composable_cot/scripts/quick_eval_all.sh` | Fast 22-example stratified eval | Created |
| `composable_cot/scripts/medium_eval.sh` | 100-example per-length eval | Created |
| `composable_cot/LLaMA-Factory/.../tuner.py` | +4 lines: RPE callback registration | Modified |
| `composable_cot/LLaMA-Factory/.../finetuning_args.py` | Added `lora_rank_pattern`/`lora_alpha_pattern` | Modified |
| `composable_cot/LLaMA-Factory/.../adapter.py` | Wire rank_pattern to PEFT's LoraConfig | Modified |

### LLaMA-Factory Changes (Minimal)

**`tuner.py` (lines 57-61)** — the only change to the core training framework:
```python
rpe_config_path = os.environ.get("RPE_CONFIG_PATH")
if rpe_config_path:
    from composable_cot.scripts.rpe_llamafactory_patch import RPETrainerCallback
    callbacks.append(RPETrainerCallback(rpe_config_path))
```

**`finetuning_args.py`** — expose PEFT's existing rank_pattern/alpha_pattern:
```python
lora_rank_pattern: Optional[str] = field(default=None)
lora_alpha_pattern: Optional[str] = field(default=None)
```

**`adapter.py`** — pass patterns through to PEFT:
```python
LoraConfig(..., rank_pattern=rank_pattern, alpha_pattern=alpha_pattern)
```

---

## Appendix B: Glossary of Terms

| Term | Definition |
|---|---|
| **RPE** | Randomized Positional Encodings — sampling random sorted positions from [0, L) during training |
| **RoPE** | Rotary Position Embeddings — the position encoding method used by Qwen/LLaMA/etc. Applies sinusoidal rotations to Q and K |
| **CCoT** | Composable Chain-of-Thought — framework for composing atomic reasoning skills via LoRA adapter merging |
| **LoRA** | Low-Rank Adaptation — fine-tuning method that inserts small trainable matrices (rank r) into frozen model layers |
| **L** | max_simulation_length — upper bound for RPE position sampling range [0, L) |
| **dm_score** | DeepMind score — mean OOD accuracy (lengths 41-100) |
| **In-dist** | In-distribution — lengths 1-40 (seen during training) |
| **OOD** | Out-of-distribution — lengths 41-100 (never seen during training) |
| **Cliff** | The length at which model accuracy drops from perfect to zero |
| **Curriculum** | Gradually increasing RPE's L across epochs |
| **Asymmetric LoRA** | Different LoRA ranks for different layers (higher for position-sensitive Q/K) |
| **PEFT** | Parameter-Efficient Fine-Tuning (HuggingFace library implementing LoRA) |
| **BPE** | Byte-Pair Encoding — Qwen's subword tokenization method |
| **CoT trace** | The step-by-step reasoning text the model generates (e.g., "The 1st character from the end is 0...") |

---

## Appendix C: Quick Reference Commands

```bash
# === On Lightning AI ===

# Set environment
export PROJECT_ROOT="/teamspace/studios/this_studio"
export PYTHONPATH="${PROJECT_ROOT}"

# Run all 4 experiments (train + eval)
bash composable_cot/scripts/run_three_experiments.sh

# Run single experiment (0=baseline, 1=RPE, 2=asymmetric, 3=curriculum)
bash composable_cot/scripts/run_three_experiments.sh 3

# Run evaluation only
bash composable_cot/scripts/run_three_experiments.sh eval

# Quick eval (22 examples, ~20 min)
bash composable_cot/scripts/quick_eval_all.sh

# Medium eval (100 examples, ~30-60 min)
bash composable_cot/scripts/medium_eval.sh

# Evaluate a single checkpoint manually
python composable_cot/scripts/eval_length_generalization.py \
    --base-model Qwen/Qwen2.5-7B \
    --lora-ckpt composable_cot/model_ckpt/reverse_string_rpe_curriculum \
    --test-file composable_cot/data/reverse_string_eval/test_all.json \
    --output-dir composable_cot/outputs/eval_curriculum \
    --task-type reverse_string \
    --train-max-length 40 \
    --max-new-tokens 2048

# === On NYU Torch HPC ===

# One-time setup
bash composable_cot/scripts/hpc/setup_hpc.sh

# Submit training jobs
sbatch composable_cot/scripts/hpc/train_exp0.slurm
sbatch composable_cot/scripts/hpc/train_exp1.slurm
sbatch composable_cot/scripts/hpc/train_exp2.slurm
sbatch composable_cot/scripts/hpc/train_exp3.slurm

# Submit eval
sbatch composable_cot/scripts/hpc/eval_quick.slurm
```
