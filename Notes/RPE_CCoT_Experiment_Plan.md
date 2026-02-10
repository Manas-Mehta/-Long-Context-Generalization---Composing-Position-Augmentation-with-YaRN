# RPE + Composable CoT: Experiment Plan

**Project**: Randomized Positional Encodings for Length Generalization in Composable Chain-of-Thought
**Institution**: TAUR Labs, NYU
**Date**: February 2025
**References**:
- RPE: Ruoss et al., "Randomized Positional Encodings Boost Length Generalization of Transformers", ACL 2023 ([arXiv:2305.16843](https://arxiv.org/abs/2305.16843))
- CCoT: Yin et al., "Learning Composable Chains-of-Thought", 2025 ([arXiv:2505.22635](https://arxiv.org/abs/2505.22635))

---

## Table of Contents

1. [Recap: What Phase 1 (RPE-only) Showed](#1-recap-what-phase-1-rpe-only-showed)
2. [What is Composable CoT (CCoT)?](#2-what-is-composable-cot-ccot)
3. [Research Questions and Hypotheses](#3-research-questions-and-hypotheses)
4. [Experiment Plan: Step by Step](#4-experiment-plan-step-by-step)
5. [Dataset Details](#5-dataset-details)
6. [Training Configuration](#6-training-configuration)
7. [What to Expect: Loss Curves and Metrics](#7-what-to-expect-loss-curves-and-metrics)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [RPE Variants to Test](#9-rpe-variants-to-test)
10. [Code Changes Required](#10-code-changes-required)
11. [Logging, Metrics, and Research Procedures](#11-logging-metrics-and-research-procedures)
12. [FAQ: Questions You Should Be Able to Answer](#12-faq-questions-you-should-be-able-to-answer)
13. [Timeline and Priorities](#13-timeline-and-priorities)

---

## 1. Recap: What Phase 1 (RPE-only) Showed

### What We Did

We reproduced DeepMind's RPE experiment on the reverse string task (binary string reversal), adapted from their encoder-only transformer to a decoder-only Qwen2 model trained from scratch (~330K parameters).

### Setup

| Parameter | Value |
|-----------|-------|
| Model | Tiny Qwen2ForCausalLM, 330K params, trained from scratch |
| Task | Reverse binary string (vocab = {0, 1}) |
| Training lengths | 1-40 (uniform random) |
| Test lengths | 1-100 (10 = in-distribution, 41-100 = OOD) |
| RPE max simulation length (L) | 2048 |
| Training steps | 10,000 |
| Learning rate | 1e-3, constant (no warmup, no decay) |
| Batch size | 128 |
| Evaluation | Autoregressive generation (greedy decoding) |

### Results

| Metric | Baseline (No RPE) | RPE (L=2048) |
|--------|-------------------|--------------|
| Training loss | 0.044 | 0.132 |
| In-dist accuracy (lengths 1-40) | 1.000 | 0.725 (at length 20) |
| OOD accuracy (length 50) | **0.000** | **0.560** |
| OOD accuracy (lengths 41-100) | ~0.000 | ~0.50+ |

### Key Takeaways

1. **RPE works.** Baseline drops to 0% accuracy the moment it sees a length beyond 40. RPE maintains ~56% accuracy at length 50. The model genuinely learned to reverse strings at lengths it never trained on.

2. **RPE has a cost.** Training loss is 3x higher (0.132 vs 0.044). In-distribution accuracy drops from 1.000 to 0.725. This is the fundamental RPE trade-off: sacrifice in-dist performance for OOD generalization.

3. **Decoder-only is harder than encoder-only.** DeepMind reported ~0.8+ OOD accuracy with an encoder-only model. Our decoder-only model gets ~0.56. The gap is due to autoregressive error compounding — one wrong token corrupts all subsequent predictions.

4. **From-scratch training works cleanly with RPE.** The tiny model had no prior positional knowledge, so RPE could freely teach it position-invariant patterns.

### What Phase 1 Did NOT Answer

- Does RPE work with **fine-tuning** a pretrained model (LoRA)?
- Does RPE work with the **CCoT format** (chain-of-thought reasoning traces)?
- Does RPE help with **multi-task** composition (combining atomic skills)?
- Does RPE help with tasks **other than reverse string**?

These are the questions Phase 2 addresses.

---

## 2. What is Composable CoT (CCoT)?

### The Big Idea

CCoT (Composable Chain-of-Thought) is a framework for teaching LLMs to solve **composite tasks** by breaking them into **atomic skills** that can be learned independently and then combined.

### The Three Stages

```
Stage 1: ATOMIC TRAINING
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│   letter_concat      │    │   next_last_letter   │    │   ascii_multiply     │
│   LoRA Adapter A     │    │   LoRA Adapter B     │    │   LoRA Adapter C     │
│                      │    │                      │    │                      │
│ "Take the 2nd letter │    │ "Find the next       │    │ "ASCII value of 'n'  │
│  of each word..."    │    │  letter after the    │    │  is 110, multiply    │
│                      │    │  last letter..."     │    │  by 8 = 880"         │
└──────────────────────┘    └──────────────────────┘    └──────────────────────┘
         │                           │                           │
         └───────────────┬───────────┘                           │
                         │                                       │
Stage 2: MODEL MERGING   │                                       │
                         ▼                                       │
              ┌─────────────────────┐                            │
              │  MERGE Adapters     │                            │
              │  (Linear / TIES)    │◄───────────────────────────┘
              │                     │
              │  Combined LoRA      │
              └─────────┬───────────┘
                        │
Stage 3: COMPOSITION FINE-TUNING (optional)
                        ▼
              ┌─────────────────────┐
              │  Fine-tune on       │
              │  composite task     │
              │  (e.g., concat +    │
              │   multiply)         │
              └─────────────────────┘
```

### The CCoT Data Format: The Random Prefix Mechanism (Critical to Understand)

The key insight of the CCoT paper is that **random prefixes simulate positional shifts**. Every original CCoT atomic dataset uses a **50/50 split** of two formats:

**Format A (50% of examples) — Standard CoT in `<prefix>`:**
```
Instruction: "Take the second letter of each word and concatenate: Zhen Lucas Mayia answer: "
Output:      "<prefix> The 2nd letter of the 1st word is h. The 2nd letter of the 2nd word
              is u. The 3rd is a. So the answer is hua.</prefix><|endoftext|>"
```
Here the CoT reasoning goes inside `<prefix>` tags. This is the "normal" format.

**Format B (50% of examples) — Random noise in `<prefix>`, real CoT in `<suffix>`:**
```
Instruction: "Take the second letter of each word: Zhen Lucas answer: <prefix> r qjgyhnc nwylrekitkqy iqog tkpxl </prefix> "
Output:      "<suffix> The 2nd letter of the 1st word is h. The 2nd letter of the 2nd word
              is u. So the answer is hu.</suffix><|endoftext|>"
```
Here the `<prefix>` in the instruction is filled with **random gibberish characters** (length 50-100, roughly 0.5x to 1.5x the length of the real CoT trace). The real reasoning moves to `<suffix>` in the output.

**Why this 50/50 split matters:**

The random noise in `<prefix>` serves as a **positional shift simulator**. When the model encounters `<prefix> random garbage </prefix>` in the instruction, the subsequent `<suffix>` CoT trace starts at a different (later) position in the sequence. This mimics what happens during composition — one task's output appears before another task's reasoning, pushing the second task's CoT to different positions.

By training on both formats, the model learns:
1. How to do the CoT reasoning itself (Format A)
2. How to do the same reasoning when it starts at an arbitrary position, after arbitrary preceding content (Format B)

**This is conceptually similar to RPE.** Both aim for position-invariant reasoning:
- **CCoT's approach**: Physically add random text before the CoT → shifts token positions in the sequence
- **RPE's approach**: Randomize position IDs directly → mathematically changes positions without adding tokens

The research question is: **does RPE add anything on top of CCoT's built-in random prefix mechanism?**

**Actual format distribution across original CCoT datasets:**

| Atomic Task | Total | Format A (prefix CoT) | Format B (random prefix + suffix CoT) |
|-------------|-------|----------------------|--------------------------------------|
| letter_concat | 4,000 | 2,000 (50%) | 2,000 (50%) |
| next_last_letter | 2,000 | 1,000 (50%) | 1,000 (50%) |
| ascii_multiply | 2,000 | 1,000 (50%) | 1,000 (50%) |
| skillmix_literary | 2,118 | 1,059 (50%) | 1,059 (50%) |
| skillmix_rhetorical | 1,774 | 887 (50%) | 887 (50%) |
| **reverse_string (OURS)** | **5,000** | **5,000 (100%)** | **0 (0%)** |

**Important:** Our reverse_string dataset does NOT follow the CCoT format — it's missing the Format B examples entirely. This is because we generated it ourselves. The original CCoT repo does not have a reverse_string task.

### Why Length Generalization Matters for CCoT

When tasks compose, the chain-of-thought trace gets longer:
- Atomic task trace: ~50-100 tokens
- Composition of 2 tasks: ~100-200 tokens
- Composition of 3+ tasks: even longer

If the model can only handle the sequence lengths it saw during training, it will fail on more complex compositions. The CCoT paper partially addresses this with random prefixes (shifting positions), but RPE could provide a more principled solution at the positional encoding level.

### The 5 Original CCoT Atomic Tasks + Our Addition

The original CCoT paper has 5 atomic tasks. We added `reverse_string` ourselves to test RPE.

| Task | Description | Training Examples | Source | Has Random Prefix (Format B)? |
|------|-------------|-------------------|--------|------------------------------|
| `letter_concat` | Take the N-th letter of each word and concatenate | 4,000 | Original CCoT | Yes (50%) |
| `next_last_letter` | Find the next letter in alphabet after the last letter | 2,000 | Original CCoT | Yes (50%) |
| `ascii_multiply` | Convert letter to ASCII value and multiply by a factor | 2,000 | Original CCoT | Yes (50%) |
| `skillmix_literary` | Multi-skill mixing for literary text | 2,118 | Original CCoT | Yes (50%) |
| `skillmix_rhetorical` | Multi-skill mixing for rhetorical text | 1,774 | Original CCoT | Yes (50%) |
| `reverse_string` | Reverse a binary string digit-by-digit | 5,000 | **Ours** | **No (0%)** |

### The 4 Composition Tasks

| Composition | Atomic Tasks Combined | Train Examples |
|-------------|----------------------|----------------|
| `letter_concat_ascii_multiply` | letter_concat + ascii_multiply | 600 |
| `letter_concat_next_last_letter` | letter_concat + next_last_letter | 600 |
| `next_last_letter_ascii_multiply` | next_last_letter + ascii_multiply | 200 |
| `literary_rhetorical` | skillmix_literary + skillmix_rhetorical | 200 |

---

## 3. Research Questions and Hypotheses

### Primary Research Question

> **Does RPE improve length generalization when applied to Composable Chain-of-Thought fine-tuning of a pretrained LLM, beyond what CCoT's built-in random prefix mechanism already provides?**

### The CCoT vs RPE Relationship

CCoT already includes a positional shift mechanism: random gibberish strings in `<prefix>` push CoT tokens to later positions. RPE does something similar but at the positional encoding level. The key question is whether RPE is redundant, complementary, or superior.

From the 1/13 meeting notes:
> "we modify the positional encodings of the atomic CoT tokens by randomly shifting them to simulate their positions in a longer, compositional CoT"
> "For CCoT, we use random letters of length 50-100 (roughly 0.5n to 1.5n, where n is the length of the atomic CoT ground truth trace)"

This tells us the random prefix lengths are deliberately chosen to match plausible composition lengths.

### Specific Hypotheses

**H1: RPE adds value on top of CCoT's random prefixes.**
- CCoT's random prefixes shift positions by a fixed amount (50-100 tokens). RPE randomizes positions across a much larger range (0-8192). RPE should provide more diverse positional training, leading to better generalization to very long compositions.

**H2: RPE helps with compositional generalization.**
- When merging atomic LoRA adapters (each trained with RPE), the composed model should handle longer composite reasoning chains better than baseline-trained merged adapters.

**H3: RPE works with LoRA fine-tuning on pretrained models.**
- Phase 1 showed RPE works from scratch. Phase 2 tests whether the same benefit holds when only ~0.1% of parameters are updated via LoRA on a pretrained Qwen2.5-7B. The pretrained RoPE knowledge in the frozen base model is a potential obstacle.

**H4: RPE can replace or enhance random prefixes.**
- If RPE works, we might not need the random gibberish in `<prefix>` at all — the positional shift would be handled at the encoding level. Or RPE + random prefixes together could be stronger than either alone.

**H5: RPE benefits vary across tasks.**
- Tasks with variable-length outputs (like reverse_string where output length = input length) should benefit more from RPE than fixed-output tasks (like ascii_multiply which always outputs a number).

### Controlled Variables

For every experiment, we run two conditions with a single variable changed:

| Variable | Baseline | RPE |
|----------|----------|-----|
| Position IDs during training | Sequential [0, 1, 2, ...] | Random sorted from [0, L) |
| Position IDs during inference | Sequential [0, 1, 2, ...] | Sequential [0, 1, 2, ...] |
| Model architecture | Qwen2.5-7B | Qwen2.5-7B |
| LoRA config | rank=8, same targets | rank=8, same targets |
| Training data | Identical | Identical |
| Learning rate, epochs, etc. | Identical | Identical |

---

## 4. Experiment Plan: Step by Step

### What the Meeting Notes Want Us to Do

From the latest meeting notes and the older 1/13 and 12/22 notes, the plan is:

1. **Finish reverse string RPE on Qwen2.5-7B** — debug training loss, try rank 16/32, try Qwen2.5-0.5B if needed
2. **Move to letter_concat and next_last_letter** — these are the original CCoT tasks with proper Format A/B split
3. **Test RPE variants** (vanilla, output-only, prefix-only) on those tasks
4. **Compare RPE with CCoT's built-in random prefix mechanism** — the 12/02 and 1/13 notes make clear the lab sees RPE as a potential replacement/enhancement for the random prefix approach

The original project idea (12/02 notes): "can we achieve the same effect [as random prefixes] by changing the positional encodings of LLMs instead of training data augmentation?"

### Phase 2A: Atomic Task Training (Start Here)

**Experiment 2A.1: Reverse String (already run on Lightning AI)**
- Baseline: Qwen2.5-7B + LoRA, standard positions, CCoT format
- RPE: Same but with randomized positions (L=8192)
- Eval: Per-length accuracy on lengths 1-100
- NOTE: Our reverse_string data is Format A only (no random prefixes). This is a simpler test — pure RPE vs no RPE, no CCoT random prefix confound.

**Experiment 2A.2: Letter Concatenation (next priority — this is the real CCoT test)**
- The letter_concat dataset already has the 50/50 Format A/B split (random prefixes built in)
- Baseline: Standard positions, CCoT data with random prefixes
- RPE: Randomized positions, same CCoT data
- This tests: does RPE help ON TOP of CCoT's existing random prefix mechanism?

**Experiment 2A.3: Next Last Letter**
- Same setup on next_last_letter task (also has 50/50 Format A/B split)

### Phase 2B: RPE Variants

From the meeting notes, these are the RPE variants to test:

| Variant | RPE Applied To | Rationale |
|---------|---------------|-----------|
| **Vanilla RPE** | Entire sequence (instruction + output) | Same as DeepMind — the default |
| **V2: Output-only RPE** | Only the CoT output tokens | The CoT trace is what needs to generalize |
| **V3: Prefix-only RPE** | Only the `<prefix>` section | Target the reasoning section specifically |
| **V4: Empty prefix RPE** | `<prefix>` is empty, RPE on `<suffix>` | Test if `<prefix>` can be noise and `<suffix>` does the work |

### Phase 2C: Model Merging + Composition

After atomic adapters are trained (both baseline and RPE versions):

1. Train atomic LoRA adapters for letter_concat and ascii_multiply (baseline + RPE)
2. Train atomic LoRA adapters for next_last_letter (baseline + RPE)
3. Merge adapters using Linear/TIES/TIES-SVD strategies
4. Evaluate merged models on composition tasks (e.g., letter_concat + ascii_multiply combined)
5. Compare: do RPE-trained adapters compose better than baseline adapters?

### Phase 2D: Composition Fine-Tuning

After merging, optionally fine-tune the merged model on a small number of composition examples:
- 200-600 composition training examples
- Compare baseline-merged vs RPE-merged after composition fine-tuning

### Phase 2E: Smaller Model Fallback

From meeting notes:
- If Qwen2.5-7B + LoRA rank 8 doesn't converge well:
  - Try LoRA rank 16 or 32
  - Try different learning rates
  - Try Qwen2.5-0.5B with full fine-tuning

---

## 5. Dataset Details

### Reverse String (Atomic)

**Format:**
```json
{
  "instruction": "Reverse the following binary string: 0 1 0 0 answer: ",
  "output": "<prefix> The 1st character from the end is 0. The 2nd character from the end is 0. The 3rd character from the end is 1. The 4th character from the end is 0. So the answer is 0010.</prefix><|endoftext|>"
}
```

**Key properties:**
- Binary alphabet: only "0" and "1" (matches DeepMind vocab_size=2)
- Space-separated digits in input (clean tokenization)
- CoT trace: one step per digit, listing from end to start
- Trace length scales linearly with input length (this is what makes length generalization relevant)
- Training: lengths 1-40, Test: lengths 1-100
- 5,000 training examples, 500 validation

**Why binary?** Isolates the positional problem. With only 2 possible characters, the model can't cheat by memorizing rare tokens — it must rely purely on positional information.

### Letter Concatenation (Atomic) — Original CCoT

This dataset has the proper CCoT 50/50 format split:

**Format A (50% — standard, CoT in `<prefix>`):**
```json
{
  "instruction": "Take the first letter of each word and concatenate in lower case: Callie-may Amedeo Toryn Jaide answer: ",
  "output": "<prefix> The first letter of the 1st word is c. The first letter of the 2nd word is a. The first letter of the 3rd word is t. The first letter of the 4th word is j. So the answer is catj.</prefix><|endoftext|>",
  "answer_label": "catj<|endoftext|>"
}
```

**Format B (50% — random gibberish prefix, CoT in `<suffix>`):**
```json
{
  "instruction": "Take the second letter of each word and concatenate in lower case: Zhen Lucas-james Mayia Quintavius answer: <prefix>   r qjgyhnc   nwylrekitkqy iqog tkpxl ywe siz bvvw w g hvlka    kvd g iobu  yqej esik ghh  bu kvywrwfvxj  xud  ztp  ioylzn kumvrjm </prefix> ",
  "output": "<suffix> The second letter of the 1st word is h. The second letter of the 2nd word is u. The second letter of the 3rd word is a. The second letter of the 4th word is u. So the answer is huau.</suffix><|endoftext|>"
}
```

**Key properties:**
- Input: sequence of names/words (typically 3-5 words)
- CoT: one step per word
- The random strings in Format B are ~50-100 random characters (roughly 0.5x-1.5x the CoT trace length, as per 1/13 meeting notes)
- Length generalization = more words in the sequence
- 4,000 training examples (2,000 Format A + 2,000 Format B)

### Composition Task (letter_concat + next_last_letter)

The composition dataset mixes examples from both atomic tasks with the same format logic:

**Example with random prefix (one atomic task):**
```json
{
  "instruction": "Find the next letter in alphabet following the last letter: gjqgntntq answer: <prefix> h oy cshr agg bjg b ck l ivyteb fypedtolec q </prefix>",
  "output": " <suffix> The last letter is w, and the letter following it in alphabet is x. So the answer is x.</suffix><|endoftext|>"
}
```

**Example without random prefix (other atomic task):**
```json
{
  "instruction": "Take the second letter of each word and concatenate: Tamarion Himal Praise Corinne answer: ",
  "output": "<prefix> The second letter of the 1st word is a. The second letter of the 2nd word is i. The 3rd is r. The 4th is o. So the answer is airo.</prefix><|endoftext|>"
}
```

**Key properties:**
- Mixes examples from both constituent atomic tasks
- Same 50/50 Format A / Format B split within the composition dataset
- 600 training examples (much less than atomic — intentional, tests generalization from atomic training)
- Has both `composable_cot` version (with CoT traces) and `answer_only` version (for evaluation)

### Label Masking

During training, the instruction tokens have labels set to -100 (ignored by cross-entropy loss). The model only learns to predict the output tokens (the CoT trace). The `template: empty` setting in the YAML config means LLaMA-Factory uses the raw instruction/output without adding chat templates.

---

## 6. Training Configuration

### Current Config (Reverse String)

```yaml
# Model
model_name_or_path: Qwen/Qwen2.5-7B    # 7 billion parameters, pretrained
finetuning_type: lora                    # Only train small adapter matrices

# LoRA Settings
lora_rank: 8           # Low-rank dimension (small = fewer params, less capacity)
lora_alpha: 16         # Scaling factor (alpha/rank = 2, standard ratio)
lora_dropout: 0.2      # Regularization during training
lora_target: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj  # All linear layers

# Training
per_device_train_batch_size: 4    # Small batch (GPU memory constraint)
learning_rate: 1.0e-3             # Relatively high for LoRA
num_train_epochs: 5.0             # 5 passes through the data
lr_scheduler_type: linear         # LR decays linearly to 0 (NOTE: different from Phase 1's constant LR)
bf16: true                        # bfloat16 precision (saves memory)

# Data
cutoff_len: 1024                  # Maximum sequence length in tokens
max_samples: 5000                 # Use all training examples

# Evaluation
eval_strategy: epoch              # Evaluate once per epoch
load_best_model_at_end: true      # Keep the checkpoint with lowest eval loss
metric_for_best_model: eval_loss
save_total_limit: 1               # Only keep the best checkpoint (saves disk)
```

### What Each Hyperparameter Does (Beginner Guide)

**lora_rank: 8**
Think of this as the "capacity" of the adapter. Higher rank = more parameters = more learning capacity. Rank 8 means each LoRA matrix pair has dimensions (original_dim x 8) and (8 x original_dim). Total trainable parameters: ~6.8M out of 7B (~0.1%).

**lora_alpha: 16**
Scaling factor. The effective LoRA update is `(alpha/rank) * BA @ x = 2 * BA @ x`. Higher alpha = stronger LoRA influence relative to the frozen base model. The ratio alpha/rank is what matters.

**learning_rate: 1.0e-3**
How big each optimization step is. 1e-3 is high for full fine-tuning but standard for LoRA (since we're only updating a small fraction of parameters, we can afford bigger steps).

**lr_scheduler_type: linear**
Learning rate starts at 1e-3 and linearly decays to 0 over training. This means early training takes big steps (explore), late training takes small steps (refine). **Note**: Phase 1 used constant LR (no decay). This is a difference but affects both baseline and RPE equally.

**cutoff_len: 1024**
Maximum number of tokens per training example. Examples longer than this get truncated. For reverse string length 40, the full CoT trace is ~250 tokens, well under 1024. For longer tasks this may need increasing.

### Differences: Phase 1 vs Phase 2

| Parameter | Phase 1 (from scratch) | Phase 2 (fine-tuning) |
|-----------|----------------------|----------------------|
| Model | Tiny Qwen2, 330K params | Qwen2.5-7B, 7B params |
| Training | Full (all params) | LoRA (0.1% of params) |
| Init | Random weights | Pretrained weights |
| LR schedule | Constant | Linear decay to 0 |
| RPE L | 2048 | 8192 |
| Steps | 10,000 | ~6,250 (5 epochs) |
| Batch size | 128 | 4 |
| Data format | "reverse: 01101\n10110" | CCoT format with `<prefix>` traces |
| Evaluation | Per-token autoregressive | Per-example exact match |

### What RPE Changes (and Doesn't Change)

**RPE changes:** The `position_ids` tensor passed to the model during training. Instead of [0, 1, 2, ..., N-1], random sorted integers from [0, 8192) are used.

**RPE does NOT change:** The model architecture, LoRA configuration, training data, learning rate, loss function, or evaluation procedure. The ONLY difference is position IDs during the forward pass when model.training=True.

**How it's activated:** An environment variable `RPE_CONFIG_PATH` points to `rpe_config.yaml`. When set, LLaMA-Factory's `tuner.py` registers an `RPETrainerCallback` that patches model.forward() at training start and unpatches at training end.

---

## 7. What to Expect: Loss Curves and Metrics

### Training Loss

The training loss measures how well the model predicts the next token in the CoT trace. Lower = better.

**Expected baseline curve:**
```
Loss
1.6 |█
    |  █
0.4 |    ██
0.2 |      ████
0.1 |          ████████████████
0.08|                          ████████
    +----------------------------------------→ Steps
    0    500  1000  2000  3000  4000  5000  6250
```

**Expected RPE curve:**
```
Loss
3.4 |█
    |  █
0.8 |   █
0.4 |    ██
0.2 |      ████
0.15|          ████████
0.12|                  ████████████████
    +----------------------------------------→ Steps
    0    500  1000  2000  3000  4000  5000  6250
```

**Why RPE loss is higher:**
1. The model must predict CoT tokens while receiving random position IDs
2. Each batch gets different random positions — the "difficulty" varies batch-to-batch
3. The pretrained RoPE in Qwen2.5 expects sequential positions; random positions disrupt its learned attention patterns
4. This is expected and matches Phase 1 (RPE loss was 3x baseline's)

**What's healthy:**
- Loss decreases over time (model is learning)
- No NaN or Inf values
- Smooth descent without wild oscillations
- Loss plateaus eventually (convergence)

**What's concerning:**
- Loss doesn't decrease at all (learning rate issue, data issue)
- Loss increases (learning rate too high, catastrophic forgetting)
- Huge spikes that don't recover (numerical instability)
- Eval loss going UP while train loss going DOWN (overfitting)

### Eval Loss

Eval loss is computed on a held-out validation set with **standard sequential positions** (model.eval() mode, no RPE active).

**For baseline:** Eval loss should track training loss closely. If train loss is 0.08, eval loss should be ~0.065-0.10.

**For RPE:** Eval loss will be MUCH higher than training loss. This is a key diagnostic:
- Training loss: ~0.12 (model performs OK with random positions)
- Eval loss: could be 2.0-5.0+ (model struggles with sequential positions)

This happens because the LoRA adapter was optimized for random positions. When switched to sequential positions at eval time, the attention patterns change. **This is the core challenge of RPE + LoRA on pretrained models.**

### What the Reverse String Eval Numbers Mean

The evaluation tests 1,000 examples: 10 per length, lengths 1-100.

**In-distribution (lengths 1-40):** These are lengths the model saw during training. High accuracy here means training worked.

**OOD (lengths 41-100):** Lengths the model NEVER saw. This is the RPE test.

**dm_score:** Mean accuracy on lengths 41-100. This is the primary metric from the DeepMind paper.

Expected outcomes:

| Scenario | Baseline in-dist | Baseline OOD | RPE in-dist | RPE OOD | Interpretation |
|----------|-----------------|-------------|-------------|---------|----------------|
| Ideal | ~1.0 | ~0.0 | 0.85-0.95 | 0.3-0.8 | RPE works as expected |
| Good | ~0.9 | ~0.0 | 0.7-0.9 | 0.2-0.5 | RPE helps but less than Phase 1 |
| Neutral | ~0.9 | ~0.0 | 0.7-0.9 | ~0.0 | RPE doesn't help (LoRA can't override pretrained PE) |
| Concerning | ~0.3 | ~0.0 | ~0.3 | ~0.0 | Neither condition works (training issue) |

---

## 8. Evaluation Protocol

### For Reverse String (Length Generalization)

1. Load base Qwen2.5-7B
2. Load LoRA adapter checkpoint
3. Merge adapter into base model (`merge_and_unload()`)
4. Set model to eval mode (standard positions)
5. For each test example (1000 total, 10 per length 1-100):
   - Feed instruction: `"Reverse the following binary string: 0 1 0 answer: "`
   - Generate autoregressively (greedy, do_sample=False, max_new_tokens=512)
   - Extract answer from generation using regex ("the answer is <binary>")
   - Compare to expected answer (exact match)
6. Report per-length accuracy and summary metrics

### For Composition Tasks

1. Train atomic LoRA adapters for each sub-task (with and without RPE)
2. Merge adapters using Linear/TIES merging
3. Evaluate merged model on the composition test set
4. Compare accuracy with and without RPE training

### Metrics to Track

| Metric | Description | Where |
|--------|-------------|-------|
| Training loss | Per-step cross-entropy on CoT tokens | Training logs |
| Eval loss | Cross-entropy on validation set | Epoch boundaries |
| Grad norm | Gradient magnitude (stability indicator) | Training logs |
| Learning rate | Current LR value (should decay linearly) | Training logs |
| Per-length accuracy | Exact match at each input length | eval_results.json |
| In-dist accuracy | Mean accuracy on lengths 1-40 | eval_results.json |
| OOD accuracy (dm_score) | Mean accuracy on lengths 41-100 | eval_results.json |

---

## 9. RPE Variants to Test

The meeting notes outline 4 RPE strategies, from broadest to most targeted:

### Variant 1: Vanilla RPE (Default)

**RPE applied to:** Entire sequence (instruction + CoT output)

```
[Reverse] [the] [following] [...] [answer:] [<prefix>] [The] [1st] [...] [answer] [is] [010] [</prefix>]
  ↑ random positions applied to ALL of these tokens ↑
```

**Rationale:** Same as DeepMind's approach. The model learns to handle arbitrary positions everywhere.

**Status:** Already implemented. This is what `rpe_config.yaml` with `enabled: true` does.

### Variant 2: Output-Only RPE

**RPE applied to:** Only the CoT output tokens (after "answer: ")

```
[Reverse] [the] [following] [...] [answer:]  [<prefix>] [The] [1st] [...] [010] [</prefix>]
  ↑ standard sequential positions ↑            ↑ random positions here only ↑
```

**Rationale:** The instruction format is always the same — it doesn't need length generalization. The CoT trace is what grows longer. Focus RPE where it matters.

**Status:** NOT YET IMPLEMENTED. Requires modifying RPEPatcher to accept a boundary index.

### Variant 3: RPE on Prefix Only

**RPE applied to:** Only the `<prefix>` tokens (which may be random noise or real CoT)

```
instruction... <prefix>  [random positions here only]  </prefix> <suffix> [standard positions] </suffix>
```

**Rationale:** From the meeting notes: "we were thinking about only shifting the encodings of the atomic CoT, and keeping the instructions unchanged." If we only randomize positions for the `<prefix>` section, the instruction stays stable while the CoT reasoning gets position-invariant training.

**Status:** NOT YET IMPLEMENTED. Requires token-level position assignment.

### Variant 4: Empty Prefix + RPE on Suffix

**Training format:**
```
instruction <prefix> [empty / minimal content] </prefix> <suffix> [real CoT, with RPE positions] </suffix>
```

**Rationale:** Test whether we can replace CCoT's random gibberish strings entirely with RPE. Instead of physically adding noise text, we keep `<prefix>` empty and use RPE to simulate the positional shift. If this works, it's a cleaner approach — no random tokens wasted in the context window.

**Status:** NOT YET IMPLEMENTED. Requires data generation changes and selective RPE.

### Priority Order (from meeting)

1. **Vanilla RPE** (already done for reverse string) — HIGH
2. **V2: Output-only RPE** — MEDIUM
3. **V3: Prefix-only RPE** — MEDIUM
4. **V4: Empty prefix** — LOW

---

## 10. Code Changes Required

### Already Changed (Phase 2 Current)

**File: `composable_cot/LLaMA-Factory/src/llamafactory/train/tuner.py`**
- 4 lines added (lines 58-61)
- Checks for `RPE_CONFIG_PATH` env var
- If set, imports and registers `RPETrainerCallback`
- This is the only change to LLaMA-Factory itself

**File: `composable_cot/scripts/rpe_llamafactory_patch.py`** (NEW)
- Bridge between LLaMA-Factory and RPE module
- `RPETrainerCallback` class with `on_train_begin()` and `on_train_end()`
- `load_rpe_config()` reads rpe_config.yaml
- `apply_rpe_patch()` / `remove_rpe_patch()` manage patching lifecycle

**File: `composable_cot/scripts/rpe_config.yaml`** (NEW)
```yaml
rpe:
  enabled: true
  max_simulation_length: 8192
  seed: null              # null = non-deterministic (different each batch)
  training_mode: true     # Randomize during training
  inference_mode: false   # Standard positions at eval
```

**File: `composable_cot/scripts/run_experiment.sh`** (MODIFIED)
- Added separate YAML configs for baseline and RPE
- Added RPE_CONFIG_PATH env var for RPE training step
- Added PYTHONPATH export for rpe module imports

**Files: `composable_cot/scripts/llamafactory/reverse_string_baseline.yaml`** (NEW)
**Files: `composable_cot/scripts/llamafactory/reverse_string_rpe.yaml`** (NEW)
- Separate training configs with output_dir baked in

**File: `composable_cot/scripts/generate_reverse_string_data.py`** (NEW)
- Generates binary string reversal data in CCoT format
- Training set: 5000 examples, lengths 1-40
- Test set: 1000 examples, 10 per length 1-100

**File: `composable_cot/scripts/eval_length_generalization.py`** (NEW)
- Loads base model + LoRA, merges, evaluates per-length accuracy
- Reports in-dist, OOD, and dm_score metrics

**File: `composable_cot/scripts/plot_results.py`** (NEW)
- Generates comparison plots (baseline vs RPE)

### Changes Needed for Next Experiments

**For letter_concat + next_last_letter atomic training:**
1. Create `reverse_string_baseline.yaml`-style configs for each task:
   - `letter_concat_baseline.yaml`
   - `letter_concat_rpe.yaml`
   - `next_last_letter_baseline.yaml`
   - `next_last_letter_rpe.yaml`
2. Modify `run_experiment.sh` or create a new script for multi-task experiments

**For RPE V2 (output-only RPE):**
1. Modify `rpe/patching.py` — add parameter for "RPE start offset" (skip instruction tokens)
2. Modify `rpe_llamafactory_patch.py` — detect instruction boundary and pass offset
3. Add `rpe_target: output_only` to rpe_config.yaml

**For RPE V3 (prefix-only RPE):**
1. Modify `rpe/patching.py` — add ability to specify token ranges for RPE
2. Need to detect `<prefix>` and `</prefix>` token positions

**For LoRA rank experiments:**
1. Create new YAML configs with `lora_rank: 16` and `lora_rank: 32`
2. No code changes needed — just config changes

**For Qwen2.5-0.5B fallback:**
1. Change `model_name_or_path: Qwen/Qwen2.5-0.5B` in YAML
2. Possibly switch to `finetuning_type: full` (no LoRA) if 0.5B is small enough
3. May need to adjust batch size (0.5B uses less memory → can increase batch)

---

## 11. Logging, Metrics, and Research Procedures

### During Training: What Gets Logged

Every 50 steps (configurable via `logging_steps`), the training loop logs:

```python
{
  "loss": 0.1242,           # Cross-entropy loss on this batch
  "grad_norm": 0.385,       # L2 norm of gradients (stability indicator)
  "learning_rate": 1.76e-4, # Current LR (decays linearly)
  "epoch": 4.12             # Progress through the data
}
```

At each epoch boundary, evaluation runs and logs:
```python
{
  "eval_loss": 0.065,       # Loss on validation set
  "eval_runtime": 28.6,     # Seconds for eval
  "epoch": 3.0
}
```

### Plots Generated Automatically

LLaMA-Factory generates (when `plot_loss: true`):
- `training_loss.png` — Loss over steps (original + smoothed)
- `training_eval_loss.png` — Eval loss at each epoch

Our eval script generates:
- `eval_results.json` — Full per-length accuracy breakdown
- `predictions.json` — Every prediction (for debugging failures)

Our plot script generates:
- Length vs accuracy comparison (baseline + RPE on same axes)
- Bar chart of summary metrics (in-dist, OOD, dm_score)

### Research Log Checklist

For each experiment run, record:

- [ ] Date and time of run
- [ ] Hardware (GPU type, memory)
- [ ] Exact config YAML used
- [ ] Whether RPE was enabled (and which variant)
- [ ] Git commit hash (for reproducibility)
- [ ] Final training loss (last logged value)
- [ ] Final eval loss (best epoch)
- [ ] Total training time
- [ ] eval_results.json path
- [ ] Any errors or anomalies observed
- [ ] Screenshots of loss curves

### Best Practices

1. **Never change two variables at once.** If testing RPE vs baseline, everything else must be identical.
2. **Save all checkpoints.** Even if results look bad, keep them for analysis.
3. **Check predictions.json for failures.** Don't just look at numbers — look at what the model actually generates.
4. **Compare loss curves.** RPE should have higher training loss but still decrease over time.
5. **Watch eval loss during training.** If it goes up while train loss goes down → overfitting.
6. **Use seeds.** Set `seed: 42` in config and `rpe.seed: null` (different random positions each batch, but reproducible data ordering).

---

## 12. FAQ: Questions You Should Be Able to Answer

### About RPE

**Q: What is RPE in one sentence?**
A: Instead of giving tokens sequential positions [0,1,2,...], during training we give them random sorted positions from a much larger range, so the model learns to work with any positions and can generalize to longer sequences.

**Q: Where exactly does RPE intervene in the model?**
A: It replaces the `position_ids` tensor passed to `model.forward()`. This tensor feeds into RoPE (Rotary Position Embedding), which rotates the Q and K vectors before attention. RPE changes what rotations are applied, but doesn't touch the rotation function itself.

**Q: Does RPE add any parameters?**
A: No. Zero additional parameters. It only changes the input to an existing function.

**Q: Why is RPE training loss higher?**
A: Because the model must learn to predict the next token despite receiving random position IDs. It's solving a harder version of the same task. The pretrained RoPE expects sequential positions, so random positions disrupt the attention patterns.

**Q: Why does RPE use standard positions at inference?**
A: The goal is that the model learns position-invariant patterns during training. At inference, sequential positions are just one valid set of positions within the range the model has seen — they're no longer "special."

**Q: What is max_simulation_length (L)?**
A: The upper bound for random position sampling. Positions are sampled from [0, L). We now use L=1024 (revised down from initial L=8192 based on DeepMind paper analysis — they used L=2048 with 100% params from scratch; with LoRA's 0.26% params, we need smaller L). During training on sequences of length ~200, the model sees positions anywhere in [0, 1024). At inference, even a sequence of length 500 only uses positions [0, 499], well within the range.

### About CCoT

**Q: What's the difference between atomic and composition tasks?**
A: Atomic tasks are single skills (e.g., letter concatenation). Composition tasks combine multiple atomic skills (e.g., concatenate letters AND then find the next letter). CCoT trains on atomics first, then merges or fine-tunes for composition.

**Q: What do `<prefix>` and `<suffix>` mean?**
A: They serve two purposes. In Format A (standard), `<prefix>` wraps the CoT reasoning. In Format B (composable), `<prefix>` in the instruction is filled with random noise, and `<suffix>` in the output wraps the real CoT. This trains the model to reason at arbitrary positions in a sequence.

**Q: What are the random strings in the CCoT data?**
A: 50% of CCoT atomic training examples have random gibberish characters (length 50-100) stuffed into the `<prefix>` section of the instruction. This is the CCoT paper's key trick — it simulates what happens during composition, where one task's output appears before the next task's reasoning. The random strings push CoT tokens to different positions, teaching position-invariant reasoning.

**Q: How is the random prefix mechanism related to RPE?**
A: Both try to achieve position-invariant reasoning through different means. CCoT adds random text to physically shift token positions. RPE randomizes position IDs directly. Our research question: does RPE provide additional benefit on top of CCoT's built-in mechanism?

**Q: How does model merging work?**
A: After training separate LoRA adapters for each atomic task, you combine them. Linear merging = weighted sum of adapter weights. TIES merging = smarter combination that handles parameter conflicts via magnitude filtering and sign agreement. The merged model can (ideally) do both tasks without further training.

**Q: Why use LoRA instead of full fine-tuning?**
A: Memory efficiency (only ~0.1% of 7B params trained), faster training, and composability (LoRA adapters can be merged/swapped).

**Q: Does our reverse_string follow the CCoT format?**
A: Partially. Our reverse_string has all examples in Format A only (CoT in `<prefix>`, no random noise). The original CCoT tasks use a 50/50 split. We need to decide whether to add Format B examples to reverse_string or focus on the original CCoT tasks (letter_concat, next_last_letter) which already have the proper format.

### About the Experiment

**Q: Why binary strings for reverse string?**
A: DeepMind's choice. With only 2 possible characters (0,1), the model can't use vocabulary diversity to cheat. It MUST use positional information to know which digit goes where. This is the purest test of positional encoding.

**Q: What's the evaluation metric?**
A: Exact match on the extracted answer. The model generates a full CoT trace, we extract the answer using regex ("the answer is X"), and check if X exactly matches the expected reversed string.

**Q: What is dm_score?**
A: Mean accuracy on out-of-distribution lengths (41-100). Named after DeepMind's evaluation protocol. This is the primary metric — it measures whether the model generalizes beyond training lengths.

**Q: Why might RPE not work as well with LoRA fine-tuning?**
A: The pretrained Qwen2.5-7B has 7 billion parameters deeply tuned for sequential RoPE positions. LoRA only updates ~6.8M parameters (0.1%). Those small adapter matrices may not have enough capacity to override the base model's positional expectations when positions are randomized. This is an open research question.

**Q: How do we know if the experiment worked?**
A: If RPE's dm_score (OOD accuracy) is significantly higher than baseline's dm_score. Even 0.1 vs 0.0 is meaningful — it means RPE enables *some* generalization that is completely absent without it.

**Q: What if both conditions show low accuracy?**
A: Then we debug. Check predictions.json to see what the model generates. Check if the CoT format is correct. Try higher LoRA rank. Try lower learning rate. Try Qwen2.5-0.5B. The meeting notes have a clear escalation path.

---

## 13. Timeline and Priorities

### From Meeting Notes (Ordered)

**HIGH PRIORITY:**
1. Finish reverse string RPE experiment on Qwen2.5-7B (in progress on Lightning AI)
   - Debug if training loss isn't dropping
   - Try LoRA rank 16/32
   - Try different learning rates
2. If Qwen2.5-7B doesn't work, try Qwen2.5-0.5B (possibly with full fine-tuning)
3. Compare RPE vs baseline on CCoT task: letter_concat (atomic)
4. Compare RPE vs baseline on CCoT task: next_last_letter (atomic)

**MEDIUM PRIORITY:**
5. RPE V2: output-only RPE
6. RPE V3: prefix-only RPE
7. Model merging experiments (letter_concat + ascii_multiply, letter_concat + next_last_letter)
8. If results are stable, try harder version of reverse string with full alphabet (26 letters)

**LOW PRIORITY:**
9. RPE V4: empty prefix experiment
10. Multi-seed runs for statistical significance
11. Composition fine-tuning experiments

### Decision Tree

```
Start: Reverse string on Qwen2.5-7B + LoRA rank 8
  │
  ├── Training loss drops, eval shows signal?
  │     ├── YES → Run letter_concat and next_last_letter
  │     │          → Proceed to model merging
  │     │          → Try RPE variants V2/V3
  │     │
  │     └── NO → Try LoRA rank 16
  │               ├── Works? → Continue with rank 16
  │               └── Still no? → Try LoRA rank 32
  │                                ├── Works? → Continue
  │                                └── Still no? → Try Qwen2.5-0.5B
  │                                                 ├── LoRA on 0.5B
  │                                                 └── Full fine-tuning on 0.5B
```

---

## Appendix A: File Structure

```
RPE/
├── rpe/                                    # Core RPE module
│   ├── core.py                             # RPE algorithm: randperm(L)[:N].sort()
│   ├── patching.py                         # RPEPatcher: monkey-patches model.forward()
│   ├── config.py                           # Configuration dataclasses
│   └── tasks/
│       ├── reverse_string.py               # Phase 1 eval harness
│       └── reverse_string_dataset.py       # Phase 1 training data
│
├── composable_cot/                         # Phase 2: CCoT integration
│   ├── LLaMA-Factory/                      # Forked training framework
│   │   ├── src/llamafactory/train/
│   │   │   └── tuner.py                    # 4 lines added for RPE callback
│   │   └── data/
│   │       └── dataset_info.json           # 30 registered datasets
│   │
│   ├── scripts/
│   │   ├── rpe_llamafactory_patch.py       # RPETrainerCallback bridge (+ curriculum support)
│   │   ├── rpe_config.yaml                 # RPE settings (L=8192, original)
│   │   ├── rpe_config_L1024.yaml           # RPE settings (L=1024, current)
│   │   ├── rpe_config_curriculum.yaml      # RPE curriculum (L: 256→1024)
│   │   ├── run_experiment.sh               # Old: baseline vs RPE pipeline
│   │   ├── run_three_experiments.sh        # NEW: 4-experiment suite
│   │   ├── generate_reverse_string_data.py # Data generation
│   │   ├── eval_length_generalization.py   # Per-length evaluation
│   │   ├── plot_results.py                 # Comparison plots
│   │   └── llamafactory/                   # Training YAML configs
│   │       ├── reverse_string_baseline.yaml       # Original baseline (rank 8)
│   │       ├── reverse_string_baseline_rank16.yaml # Baseline (rank 16, no RPE)
│   │       ├── reverse_string_rpe.yaml            # Original RPE (rank 8)
│   │       ├── reverse_string_rpe_rank16.yaml     # Exp 1: RPE rank 16
│   │       ├── reverse_string_rpe_asymmetric.yaml # Exp 2: RPE asymmetric
│   │       ├── reverse_string_rpe_curriculum.yaml # Exp 3: RPE curriculum
│   │       ├── letter_concat_ascii_multiply_composable_cot.yaml
│   │       ├── letter_concat_next_last_letter_composable_cot.yaml
│   │       ├── next_last_letter_ascii_multiply_composable_cot.yaml
│   │       └── skillmix_literary_rhetorical_composable_cot.yaml
│   │
│   ├── data/
│   │   ├── atomic/                         # 6 atomic task datasets
│   │   ├── composition/                    # 4 composition task datasets (2 formats each)
│   │   └── reverse_string_eval/            # Length-stratified test sets
│   │
│   ├── inference/
│   │   ├── inference.py                    # Full inference pipeline
│   │   └── utils.py                        # Model merging (Linear/TIES/TIES-SVD)
│   │
│   ├── model_ckpt/                         # Saved LoRA checkpoints
│   └── outputs/                            # Evaluation results and plots
│
├── Notes/                                  # Documentation
│   ├── Meeting_notes.md
│   ├── EXPERIMENT_REPORT.md                # Phase 1 detailed report
│   ├── RPE_CCoT_Experiment_Plan.md         # THIS FILE
│   └── older notes/
│
└── Phase2.md                               # Phase 2 audit document
```

## Appendix B: Quick Command Reference

```bash
# On Lightning AI:

# Set environment
export PROJECT_ROOT="/teamspace/studios/this_studio"
export PYTHONPATH="${PROJECT_ROOT}"

# Run full experiment (baseline + RPE + eval + plots)
bash composable_cot/scripts/run_experiment.sh

# Train just baseline
cd composable_cot/LLaMA-Factory
unset RPE_CONFIG_PATH
llamafactory-cli train ../scripts/llamafactory/reverse_string_baseline.yaml

# Train just RPE
cd composable_cot/LLaMA-Factory
RPE_CONFIG_PATH="../scripts/rpe_config.yaml" llamafactory-cli train ../scripts/llamafactory/reverse_string_rpe.yaml

# Evaluate a checkpoint
python composable_cot/scripts/eval_length_generalization.py \
    --base-model Qwen/Qwen2.5-7B \
    --lora-ckpt composable_cot/model_ckpt/reverse_string_rpe \
    --test-file composable_cot/data/reverse_string_eval/test_all.json \
    --output-dir composable_cot/outputs/reverse_string_rpe_eval \
    --train-max-length 40
```

---

## 14. Evaluation Protocol (Reverse String)

### What We Measure

For each experiment, we measure **3 key metrics**:

| Metric | What it means | How computed |
|--------|---------------|--------------|
| **Overall accuracy** | Exact match across all test lengths | correct / total |
| **In-distribution accuracy** | Accuracy on lengths 1-40 (seen during training) | Mean per-length accuracy, lengths 1-40 |
| **OOD accuracy (dm_score)** | Accuracy on lengths 41-100 (never seen) | Mean per-length accuracy, lengths 41-100 |

The **primary metric is OOD accuracy** (dm_score). This directly measures length generalization.

### Evaluation Data

| File | Contents | Count |
|------|----------|-------|
| `data/reverse_string_eval/test_all.json` | Lengths 1-100, ~5 examples per length | ~500 |
| `data/reverse_string_eval/test_in_dist.json` | Lengths 1-40 only | ~200 |
| `data/reverse_string_eval/test_ood.json` | Lengths 41-100 only | ~300 |

### How Evaluation Works

1. **Load model**: Base Qwen2.5-7B + LoRA adapter, merged via `model.merge_and_unload()`
2. **Prompt**: Each test example's `instruction` field (ends with "answer: ")
3. **Generate**: Greedy decoding, `max_new_tokens=512`, `do_sample=False`
4. **Extract answer**: Regex looks for "the answer is <binary_string>" in generated text
5. **Compare**: Exact string match against expected reversed string
6. **Group by length**: Compute per-length accuracy, then aggregate to in-dist/OOD

### Evaluation Script

```bash
python composable_cot/scripts/eval_length_generalization.py \
    --base-model Qwen/Qwen2.5-7B \
    --lora-ckpt <checkpoint_dir> \
    --test-file composable_cot/data/reverse_string_eval/test_all.json \
    --output-dir <output_dir> \
    --task-type reverse_string \
    --train-max-length 40
```

### Output Files

Each eval run produces:
- `eval_results.json` — Overall/in-dist/OOD accuracy + per-length breakdown
- `predictions.json` — Every prediction for debugging (prompt, expected, predicted, correct)

### What to Look For

- **Baseline should have high in-dist accuracy** (~0.8+) and **near-zero OOD** (~0.0). If baseline in-dist is low too, the training itself failed.
- **RPE should show non-zero OOD accuracy**. Even 0.05-0.10 is meaningful vs 0.0 baseline.
- **Check predictions.json** if accuracy is unexpectedly low — the model might be generating valid CoT but with a formatting issue the regex misses.
- **Loss during training**: eval_loss should decrease across epochs. If it spikes up → overfitting.

---

## 15. Current Experiment Suite (Feb 2025)

### The 3 RPE Variations + Baseline

| # | Name | LoRA Config | RPE Config | Key Idea |
|---|------|------------|------------|----------|
| 0 | **Baseline** | rank=16, alpha=32, all 7 targets | None (no RPE) | Control — does rank 16 alone help? |
| 1 | **RPE Rank 16** | rank=16, alpha=32, all 7 targets | L=1024 | Standard RPE with better capacity & smaller L |
| 2 | **RPE Asymmetric** | rank=8 default + rank=32 on Q/K | L=1024 | Concentrate capacity on position-sensitive layers |
| 3 | **RPE Curriculum** | rank=16, alpha=32, all 7 targets | L: 256→512→768→1024→1024 | Gradual L increase across epochs |

### Key Changes from Previous Run (L=8192, rank=8)

| Parameter | Previous (failed) | Current |
|-----------|-------------------|---------|
| `max_simulation_length` (L) | 8192 | 1024 |
| `lora_rank` | 8 | 16 (or 8+32 asymmetric) |
| `lora_alpha` | 16 | 32 (or 16+64 asymmetric) |
| `learning_rate` | 1.0e-3 | 5.0e-4 |
| Avg position gap | ~41 | ~5 |
| Trainable params | ~20M | ~40M (rank 16) |

### Why These Changes

- **L=1024**: DeepMind used L=2048 with 100% params from scratch. We use LoRA with 0.26% params, so we need smaller L. L=1024 gives 2x margin over max test sequence (~500 tokens) with avg gap ~5 (close to pretrained gap of 1).
- **Rank 16**: Previous rank 8 = 20M params. Rank 16 = 40M. More capacity to learn position-invariant patterns.
- **LR 5e-4**: Previous 1e-3 may have been too aggressive with RPE's harder optimization landscape.
- **Asymmetric (Exp 2)**: RoPE only acts on Q/K. LoRA on q_proj/k_proj needs the most capacity for position adaptation.
- **Curriculum (Exp 3)**: Instead of shocking the model with random positions from epoch 1, we warm up gradually.

### Run Commands

```bash
# On Lightning AI — run all 4 experiments end-to-end:
bash composable_cot/scripts/run_three_experiments.sh

# Run a specific experiment (0=baseline, 1=rank16, 2=asymmetric, 3=curriculum):
bash composable_cot/scripts/run_three_experiments.sh 1

# Run evaluation only (after all training is done):
bash composable_cot/scripts/run_three_experiments.sh eval
```

---

## Appendix C: File Change Log

### Files Created (Feb 2025 Experiment Suite)

| File | Purpose |
|------|---------|
| `composable_cot/scripts/rpe_config_L1024.yaml` | RPE config with L=1024 (down from 8192) |
| `composable_cot/scripts/rpe_config_curriculum.yaml` | RPE config with epoch-dependent L schedule |
| `composable_cot/scripts/llamafactory/reverse_string_baseline_rank16.yaml` | Baseline: rank 16, no RPE |
| `composable_cot/scripts/llamafactory/reverse_string_rpe_rank16.yaml` | Exp 1: RPE rank 16, L=1024 |
| `composable_cot/scripts/llamafactory/reverse_string_rpe_asymmetric.yaml` | Exp 2: RPE asymmetric LoRA (Q/K=32, rest=8) |
| `composable_cot/scripts/llamafactory/reverse_string_rpe_curriculum.yaml` | Exp 3: RPE curriculum learning |
| `composable_cot/scripts/run_three_experiments.sh` | End-to-end run script for all 4 experiments |

### Files Modified (Feb 2025 Experiment Suite)

| File | What Changed | Why |
|------|-------------|-----|
| `composable_cot/LLaMA-Factory/src/llamafactory/hparams/finetuning_args.py` | Added `lora_rank_pattern` and `lora_alpha_pattern` fields to `FinetuningArguments` dataclass, plus parsing in `__post_init__` | Enable asymmetric LoRA (different ranks per module) |
| `composable_cot/LLaMA-Factory/src/llamafactory/model/adapter.py` | Added `rank_pattern`/`alpha_pattern` passthrough to PEFT's `LoraConfig` in `_setup_lora_tuning()` | Wire asymmetric rank config to PEFT |
| `composable_cot/scripts/rpe_llamafactory_patch.py` | Added curriculum learning support: `load_rpe_config()` now parses `curriculum` dict, `RPETrainerCallback` has `on_epoch_begin()` hook that updates L at epoch boundaries | Curriculum RPE: gradually increase L across epochs |
| `composable_cot/LLaMA-Factory/data/dataset_info.json` | Added atomic `letter_concat` and `next_last_letter` dataset entries | Prepare for future letter_concat experiments |

### Lightning AI Upload Checklist

Upload the entire `RPE/` folder. The critical directories are:

```
RPE/
├── rpe/                          # Core RPE module (REQUIRED)
├── composable_cot/
│   ├── LLaMA-Factory/            # Training framework (REQUIRED)
│   ├── scripts/                  # Configs + run scripts (REQUIRED)
│   └── data/
│       ├── atomic/reverse_string_composable_cot/  # Training data (REQUIRED)
│       └── reverse_string_eval/  # Eval data (REQUIRED)
```

After uploading, run:
```bash
export PROJECT_ROOT="/teamspace/studios/this_studio"
cd "${PROJECT_ROOT}"
bash composable_cot/scripts/run_three_experiments.sh
```
