# MRCR Context Extension Experiment — Complete Guide

This document explains everything about the MRCR experiment: what the data looks like, how it flows through the pipeline, what the baselines do, and how evaluation works. Written so you can confidently explain any part of it.

---

## Table of Contents
1. [What is MRCR?](#1-what-is-mrcr)
2. [What Does the Data Look Like?](#2-what-does-the-data-look-like)
3. [Data Pipeline Walkthrough](#3-data-pipeline-walkthrough)
4. [How the Vanilla Baseline Works](#4-how-the-vanilla-baseline-works)
5. [How YaRN Works](#5-how-yarn-works)
6. [The Evaluation Pipeline End-to-End](#6-the-evaluation-pipeline-end-to-end)
7. [Folder Structure](#7-folder-structure)
8. [Code Reference (Functions)](#8-code-reference)
9. [HPC Submission Workflow](#9-hpc-submission-workflow)
10. [Phase 3+: RPE and PoSE (to be added)](#10-phase-3-rpe-and-pose)

---

## 1. What is MRCR?

MRCR stands for **Multi-Round Coreference Resolution**. It's a benchmark created by OpenAI to test whether language models can retrieve specific information buried in very long conversations.

### The classic "Needle in a Haystack" (NIAH)

The original NIAH test works like this:
- Stuff a long document with random text (the "haystack")
- Insert one specific fact somewhere in the middle (the "needle"), e.g., "The secret code is 7492"
- Ask the model: "What is the secret code?"
- If the model can find and return "7492", it passes

**Problem:** Modern models have basically solved basic NIAH. They can find a single needle easily.

### MRCR: The harder version

MRCR makes it harder by:
1. **Multiple needles**: Instead of 1 needle, there are 2, 4, or 8 similar-looking entities scattered through the conversation
2. **Coreference**: The needles share the same topic/format but have different content. For example, there might be 3 different poems about tapirs, each in a different style. The model must identify the *correct* one
3. **Multi-turn conversation**: The haystack is a realistic multi-turn chat conversation, not just random text

**Example scenario (simplified):**
> A 6000-token conversation where 2 different writing samples appear:
> - At position 1500: A poem about ocean waves in haiku format
> - At position 4200: A poem about ocean waves in sonnet format
>
> Question: "Please reproduce the poem about ocean waves that was written in sonnet format"
>
> The model must find and reproduce the sonnet (not the haiku).

### Why we picked MRCR

- It tests **long-context retrieval** — the model must attend across thousands of tokens to find the right needle
- It has **built-in length binning** — samples range from 4K to 1M tokens, so we can test at different context lengths
- It's **harder than NIAH** — so even good models will show degradation at longer contexts, giving us room to measure improvement from RPE

---

## 2. What Does the Data Look Like?

Each MRCR sample has these fields:

| Field | What it is | Example |
|-------|-----------|---------|
| `prompt` | A JSON string containing a list of chat messages. This is the full multi-turn conversation with the needle(s) buried inside, ending with the question. | `'[{"role":"user","content":"Hi!"}, {"role":"assistant","content":"Hello!..."}, ... {"role":"user","content":"Please reproduce the poem..."}]'` |
| `answer` | The expected response. Starts with a random string prefix, followed by the actual needle content. | `"xK7mQ9z... O Ocean, thy waves crash upon the shore / With thunderous..."` |
| `random_string_to_prepend` | A unique random string the model must output before the answer. Prevents guessing. | `"xK7mQ9z"` |
| `n_needles` | How many confounding entities are in the conversation (2, 4, or 8) | `2` |
| `n_chars` | Length of the needle content | `150` |

### Concrete example of a prompt (abbreviated)

```json
[
  {"role": "user", "content": "Hello, I'd like to have a conversation about various topics."},
  {"role": "assistant", "content": "Sure! I'd love to chat about anything..."},

  ... hundreds of conversation turns about random topics ...

  {"role": "assistant", "content": "Here's a poem I wrote about tapirs:\n\nIn the jungle deep and green\nThe tapir walks, a gentle scene\nWith snout so long and eyes so bright\nA creature of the fading light"},

  ... more conversation turns ...

  {"role": "assistant", "content": "Here's another poem about tapirs:\n\nO noble tapir, beast of ancient days\nThrough emerald forests thou dost gently graze\nThy prehensile snout, a trunk divine\nBeneath the canopy of twisted vine"},

  ... more conversation turns ...

  {"role": "user", "content": "Can you please repeat the second poem about tapirs, the one in a more formal style? Please start your response with: xK7mQ9z"}
]
```

**Expected answer:**
```
xK7mQ9zO noble tapir, beast of ancient days
Through emerald forests thou dost gently graze
Thy prehensile snout, a trunk divine
Beneath the canopy of twisted vine
```

The model must:
1. Find the correct poem among 2+ similar poems in a ~6000 token conversation
2. Prepend the exact random string `xK7mQ9z`
3. Reproduce the poem verbatim

### The bins

Samples are grouped by total token count:

| Bin | Token range | Context length | Our role |
|-----|------------|---------------|----------|
| Bin 0 | 4,096 - 8,192 | ~2-4 pages | **Train RPE here** (Phase 3), Eval here (Phase 2) |
| Bin 1 | 8,192 - 16,384 | ~4-8 pages | **OOD eval** — LoRA hasn't seen these lengths |
| Bin 2 | 16,384 - 32,768 | ~8-16 pages | Extended eval if needed |
| Bin 3+ | 32K - 1M | Very long | Beyond Qwen's native 32K context window |

---

## 3. Data Pipeline Walkthrough

### What `prepare_data.py` does, step by step

**Step 1: Load the tokenizer**
```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```
We need Qwen's tokenizer because different tokenizers produce different token counts for the same text. GPT-4's tokenizer might say a sample is 5000 tokens, but Qwen's might say 6200. We bin by what our model actually sees.

**Step 2: Download MRCR from HuggingFace**
```python
ds = load_dataset("openai/mrcr", split="train")  # ~2400 samples total
```

**Step 3: Filter by needle count**
We start with 2-needle only (simplest). This gives us ~800 samples.

**Step 4: Tokenize and bin each sample**

For each sample, we take the `prompt` field (which is a JSON string of messages), apply Qwen's chat template (which adds special tokens like `<|im_start|>user\n...<|im_end|>`), and count the tokens:

```python
# What the raw prompt looks like:
prompt_json = '[{"role":"user","content":"Hi"}, {"role":"assistant","content":"Hello!"}]'

# Parse it:
messages = json.loads(prompt_json)
# Result: [{"role":"user","content":"Hi"}, {"role":"assistant","content":"Hello!"}]

# Apply Qwen's chat template:
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Result: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>assistant\n"

# Count tokens:
token_ids = tokenizer.encode(text)
token_count = len(token_ids)  # e.g., 5832
```

Then we assign this sample to a bin based on `token_count`:
- 5832 tokens → falls in [4096, 8192] → Bin 0 ("4K-8K")

**Step 5: Create train/test splits**

Within each bin, we shuffle and split 70/30:
- Bin 0 (4K-8K): say 100 samples → 70 train, 30 test
- Bin 1 (8K-16K): say 100 samples → 70 train, 30 test

The **train** set is reserved for Phase 3 (RPE+LoRA fine-tuning).
The **test** set is used for all evaluations (Phase 2 baselines + Phase 3 RPE).

**Step 6: Save to disk**

```
data/
├── bin0_4K-8K/
│   ├── train.json    # 70 samples for RPE training (Phase 3)
│   └── test.json     # 30 samples for evaluation
├── bin1_8K-16K/
│   ├── train.json
│   └── test.json
└── metadata.json     # Records tokenizer, split ratio, bin info
```

---

## 4. How the Vanilla Baseline Works

The vanilla baseline is the simplest experiment: load Qwen2.5-7B-Instruct as-is (no modifications) and see how well it handles MRCR at different context lengths.

### What happens under the hood

**Step 1: Load the model**
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="cuda",           # Put on GPU
    torch_dtype=torch.bfloat16,  # 16-bit precision to fit in memory
)
model.eval()  # Inference mode — no gradient computation
```

This loads Qwen2.5-7B with its pretrained weights. The model has 7.6 billion parameters, uses **RoPE** (Rotary Position Embeddings), and was pretrained with a 32K context window.

**Step 2: For each test sample, format the input**

```python
# Parse the MRCR prompt into a message list
messages = json.loads(sample["prompt"])
# [{"role":"user","content":"Hi!"}, {"role":"assistant","content":"..."},  ...]

# Apply Qwen's chat template (adds special tokens)
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize into IDs the model understands
input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
# Shape: [1, ~6000]  (batch_size=1, seq_len=~6000 for a 4K-8K sample)
```

**Step 3: Generate a response**

```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=2048,    # Allow up to 2048 new tokens
    do_sample=False,        # Greedy decoding: always pick the most likely next token
    pad_token_id=tokenizer.eos_token_id,
)
```

The model processes all ~6000 input tokens through its 28 transformer layers, then autoregressively generates new tokens one at a time until it hits `max_new_tokens` or produces an end-of-sequence token.

**What the model "sees" internally:**

At each layer, the model uses **self-attention** to let each token attend to all previous tokens. For a 6000-token input, this means:
- Token at position 5999 can attend to tokens at positions 0, 1, 2, ..., 5998
- The attention weights determine how much each previous token contributes to the current token's representation
- To find the needle (say at position 4200), the model needs the attention at the final position to "look back" at position 4200 and weight it highly

**Where position encoding comes in:**

Each position gets a **RoPE** (Rotary Position Embedding) encoding:
```
Position 0:    cos(0 × θ), sin(0 × θ)     applied to Q and K vectors
Position 1:    cos(1 × θ), sin(1 × θ)
Position 2:    cos(2 × θ), sin(2 × θ)
...
Position 5999: cos(5999 × θ), sin(5999 × θ)
```

These rotations encode **relative distance** between tokens in the attention computation. Two tokens 5 positions apart will have a consistent rotational relationship regardless of where they appear in the sequence. This is why RoPE-based models can handle variable-length inputs (up to their trained context window).

**Step 4: Decode and grade**

```python
# Extract only the newly generated tokens
response = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

# Grade using official metric
score = grade_mrcr(response, sample["answer"], sample["random_string_to_prepend"])
```

### What we expect from vanilla baseline

- **4K-8K bin**: Should perform reasonably well. 6000 tokens is well within Qwen's 32K context window. The model has been pretrained on contexts this long.
- **8K-16K bin**: Might start degrading. Not because of position encoding limits (still within 32K), but because the task gets harder — more text to search through, more distractors, harder to maintain attention on the needle.

This baseline tells us: **how hard is MRCR at each context length for this model?**

---

## 5. How YaRN Works

YaRN (Yet another RoPE extensioN) is a method to extend a model's effective context window **without any training**. It works by modifying how RoPE computes position encodings.

### The problem YaRN solves

Qwen2.5-7B was pretrained with `max_position_embeddings = 32768`. This means RoPE's rotary frequencies were calibrated for positions 0 to 32767. If you feed the model a 50,000-token input, positions 32768-49999 get RoPE values the model has never seen during pretraining. The attention mechanism breaks down — it produces garbage.

### How RoPE normally works

RoPE assigns each dimension pair `d` of the Q and K vectors a rotation frequency:

```
θ_d = base^(-2d / dim)
```

For Qwen2.5-7B: `base = 1,000,000`, `dim = 128`.

At position `pos`, the rotation applied is:
```
rotation_angle = pos × θ_d
```

**Low-frequency dimensions** (large d): slow rotation. One full rotation (2π) takes many positions. These capture long-range patterns.

**High-frequency dimensions** (small d): fast rotation. These capture local, nearby-token patterns.

### What YaRN changes

Instead of naively compressing all positions (which breaks high-frequency dimensions), YaRN treats different dimensions differently:

**High-frequency dimensions** (fast rotation, local patterns):
→ Fully **interpolated** (divide frequency by scale factor)
→ These are most sensitive to compression, so we compress them gently

**Low-frequency dimensions** (slow rotation, global patterns):
→ Left **unchanged** (extrapolated)
→ These can naturally handle larger positions without modification

**Middle dimensions:**
→ **Blended** between interpolation and extrapolation using a smooth ramp function

Additionally, YaRN applies **attention temperature scaling** to prevent attention distributions from becoming too sharp after the frequency modification:
```
attention = softmax(QK^T / (t × sqrt(d)))
```
where `t` adjusts based on the scaling factor.

### How we enable it (zero training)

```python
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Inject YaRN configuration
config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,                            # 4x extension
    "original_max_position_embeddings": 32768, # Original training context
}

# Load model with modified config
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", config=config, ...)
```

That's it. The model weights are completely unchanged. Only the RoPE frequency computation is modified at model load time. HuggingFace's transformers library handles this internally — it reads the `rope_scaling` config and adjusts the `inv_freq` tensor used in RoPE computations.

### What changes at inference

With `factor=4.0`:
- Original context window: 32K
- New effective context: ~131K (32K × 4)
- The model can now process inputs up to ~131K tokens

**For our 4K-16K experiments:** YaRN is actually overkill — Qwen already handles 32K natively. But we test it anyway because:
1. It might improve attention quality even within the native range
2. It establishes a baseline for longer bins (32K+) that we might test later
3. **It might slightly hurt 4K-8K performance** — since YaRN compresses all positions even when the input is short (static scaling). This is worth measuring.

### YaRN vs RPE — The key conceptual difference

| | What it changes | When it's applied | Training needed? |
|---|---|---|---|
| **YaRN** | The RoPE frequency basis (`inv_freq`) — how the model *interprets* positions | Always (inference time) | No |
| **RPE** | The position IDs fed to RoPE — *which* positions the model sees | Only during LoRA training | Yes (LoRA fine-tuning) |

YaRN says: "Let me recalibrate the ruler so it can measure longer distances."
RPE says: "Let me train you with a scrambled ruler so you learn to handle any ruler."

They operate at different levels, which is why combining them is interesting — they're not mutually exclusive.

---

## 6. The Evaluation Pipeline End-to-End

Here's exactly what happens when you run `eval_mrcr.py`:

### Step 1: Load the model (one of three modes)

```
Mode A (Vanilla):   Load Qwen2.5-7B-Instruct as-is
Mode B (YaRN):      Load Qwen2.5-7B-Instruct with rope_scaling config injected
Mode C (LoRA):      Load Qwen2.5-7B-Instruct + LoRA adapter, merge weights
```

All three produce a single merged model in eval mode. At inference time, there's no behavioral difference in how generation works — only the internal weights/config differ.

### Step 2: Load test data

```python
with open("data/bin0_4K-8K/test.json") as f:
    test_data = json.load(f)
# test_data = [{"prompt": "...", "answer": "...", "random_string_to_prepend": "...", ...}, ...]
```

### Step 3: For each test sample

```
┌─────────────────────────────────────────────────────────────────┐
│ Sample: {"prompt": "[{...messages...}]", "answer": "xK7...",   │
│          "random_string_to_prepend": "xK7"}                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │ Parse JSON  │  json.loads(sample["prompt"])
                    │ messages    │  → [{"role":"user","content":"..."},
                    └──────┬──────┘     {"role":"assistant","content":"..."}]
                           │
                    ┌──────▼──────┐
                    │ Apply chat  │  tokenizer.apply_chat_template(messages)
                    │ template    │  → "<|im_start|>user\n...<|im_end|>\n..."
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Tokenize    │  tokenizer.encode(text)
                    │             │  → tensor([151644, 872, 198, ...])
                    └──────┬──────┘    shape: [1, ~6000]
                           │
                    ┌──────▼──────┐
                    │ model       │  Forward pass through 28 transformer layers
                    │ .generate() │  RoPE applied at each layer to Q and K
                    │             │  Greedy decoding: pick argmax at each step
                    └──────┬──────┘  Generates until EOS or max_new_tokens
                           │
                    ┌──────▼──────┐
                    │ Decode      │  tokenizer.decode(generated_ids)
                    │ response    │  → "xK7O noble tapir, beast of ancient..."
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Grade       │  grade_mrcr(response, answer, prefix)
                    │             │
                    │ 1. Starts   │  response.startswith("xK7")? → Yes
                    │    with     │
                    │    prefix?  │
                    │             │
                    │ 2. Strip    │  response = "O noble tapir..."
                    │    prefix   │  answer   = "O noble tapir..."
                    │             │
                    │ 3. Compute  │  SequenceMatcher(response, answer)
                    │    ratio    │  → 0.95  (95% similar)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Record:     │  {"score": 0.95, "bin": "4K-8K",
                    │ prediction  │   "gen_time": 12.3, "tokens": 5832}
                    └─────────────┘
```

### Step 4: Aggregate results

After processing all test samples:
```python
per_bin = {
    "4K-8K": {"mean_score": 0.82, "num_samples": 30, "num_perfect": 18, "num_zero": 3},
    "8K-16K": {"mean_score": 0.65, "num_samples": 30, "num_perfect": 10, "num_zero": 8},
}
overall_score = 0.735  # Mean across all samples
```

### Step 5: Save results

Two files:
- `eval_results.json` — Summary: config, overall score, per-bin breakdown
- `predictions.json` — Per-sample: score, response preview, token count, generation time

### Grading deep-dive

The `grade_mrcr` function uses Python's `difflib.SequenceMatcher`:

```python
from difflib import SequenceMatcher

# Example 1: Perfect match
response = "xK7O noble tapir, beast of ancient days"
answer   = "xK7O noble tapir, beast of ancient days"
# Strip prefix "xK7" from both → identical → score = 1.0

# Example 2: Close but not perfect
response = "xK7O noble tapir, beast of ancient day"   # Missing final "s"
answer   = "xK7O noble tapir, beast of ancient days"
# After stripping → SequenceMatcher ratio ≈ 0.97

# Example 3: Wrong needle retrieved
response = "xK7In the jungle deep and green, the tapir walks..."  # Wrong poem!
answer   = "xK7O noble tapir, beast of ancient days..."
# After stripping → very different text → score ≈ 0.3

# Example 4: Missing prefix
response = "O noble tapir, beast of ancient days"  # Forgot "xK7"!
answer   = "xK7O noble tapir, beast of ancient days"
# response.startswith("xK7") → False → score = 0.0
```

Why SequenceMatcher? The expected answer can be a full paragraph. Exact match would be too harsh — a single missing space would score 0. SequenceMatcher gives partial credit proportional to how much of the text was reproduced correctly.

---

## 7. Folder Structure

```
composable_cot/mrcr_context_extension/
├── scripts/                    # All Python scripts
│   ├── prepare_data.py         # Phase 1: Download, tokenize, bin, split MRCR data
│   └── eval_mrcr.py            # Phase 2+3: Inference + grading (vanilla/YaRN/LoRA)
├── hpc/                        # SLURM job scripts for NYU Torch HPC
│   ├── prepare_data.slurm      # Phase 1: Data prep job
│   ├── eval_vanilla.slurm      # Phase 2: Vanilla baseline eval
│   └── eval_yarn.slurm         # Phase 2: YaRN baseline eval
├── data/                       # Generated by prepare_data.py
│   ├── bin0_4K-8K/
│   │   ├── train.json          # Reserved for Phase 3 RPE+LoRA training
│   │   └── test.json           # Used for all evaluations
│   ├── bin1_8K-16K/
│   │   ├── train.json
│   │   └── test.json
│   ├── bin2_16K-32K/
│   │   ├── train.json
│   │   └── test.json
│   └── metadata.json
├── outputs/                    # Generated by eval_mrcr.py
│   ├── vanilla_4K-8K/
│   │   ├── eval_results.json   # Summary metrics
│   │   └── predictions.json    # Per-sample details
│   ├── vanilla_8K-16K/
│   ├── yarn_4K-8K/
│   ├── yarn_8K-16K/
│   └── ...
├── configs/                    # Phase 3: RPE/training configs (to be added)
└── checkpoints/                # Phase 3: LoRA checkpoints (to be added)
```

---

## 8. Code Reference

### `scripts/prepare_data.py`

| Function | What it does |
|----------|-------------|
| `get_bin_index(token_count)` | Maps a token count to a bin index. E.g., 5832 → 0 (4K-8K). Returns -1 if below 4096. |
| `get_bin_label(bin_index)` | Converts bin index to label: 0 → "4K-8K", 1 → "8K-16K", etc. |
| `tokenize_prompt(prompt_json, tokenizer)` | Takes the raw `prompt` JSON string → parses into messages → applies chat template → tokenizes → returns token count. |
| `main()` | Orchestrates everything: load tokenizer → download dataset → filter by needle count → tokenize + bin each sample → split train/test per bin → save. |

### `scripts/eval_mrcr.py`

| Function | What it does |
|----------|-------------|
| `grade_mrcr(response, answer, random_string_to_prepend)` | Official MRCR grading. Checks random prefix → strips it from both strings → returns SequenceMatcher ratio (0.0-1.0). |
| `load_model(base_model_name, lora_ckpt_dir, enable_yarn, yarn_factor, ...)` | Loads model in one of three modes: vanilla (as-is), YaRN (inject rope_scaling config), or LoRA (load + merge adapter). Returns (model, tokenizer, description). |
| `evaluate_mrcr(model, tokenizer, test_data, max_new_tokens)` | Main loop: for each sample → parse messages → chat template → tokenize → generate → decode → grade → aggregate per-bin scores. |
| `main()` | CLI entrypoint: parse args → load model → load test data → run eval → print summary → save results. |

---

## 9. HPC Submission Workflow

```bash
# 1. SSH into HPC
ssh mm14444@login.torch.hpc.nyu.edu

# 2. Go to project and pull latest code
cd /scratch/mm14444/RPE
git pull

# 3. Ensure slurm_logs directory exists
mkdir -p slurm_logs

# 4. Phase 1: Prepare data (downloads MRCR — needs internet)
sbatch composable_cot/mrcr_context_extension/hpc/prepare_data.slurm

# 5. Monitor job
squeue -u mm14444
# When it finishes, check output:
cat slurm_logs/mrcr_prep_*.out

# 6. Verify data was created
ls composable_cot/mrcr_context_extension/data/
# Should see: bin0_4K-8K/  bin1_8K-16K/  bin2_16K-32K/  metadata.json

# 7. Phase 2: Submit both baselines (can run in parallel)
sbatch composable_cot/mrcr_context_extension/hpc/eval_vanilla.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_yarn.slurm

# 8. Monitor
squeue -u mm14444

# 9. Check results when done
cat composable_cot/mrcr_context_extension/outputs/vanilla_4K-8K/eval_results.json
cat composable_cot/mrcr_context_extension/outputs/yarn_4K-8K/eval_results.json
cat composable_cot/mrcr_context_extension/outputs/vanilla_8K-16K/eval_results.json
cat composable_cot/mrcr_context_extension/outputs/yarn_8K-16K/eval_results.json
```

---

## 10. Phase 3+: RPE and PoSE

*Scripts to be added after Phase 2 baselines are collected.*

### RPE (Phase 3)

Uses the existing RPE infrastructure from Phase 2:
- `rpe/core.py` — Generates sorted random positions from [0, L)
- `rpe/patching.py` — Monkey-patches model.forward() to replace position_ids
- `composable_cot/scripts/rpe_llamafactory_patch.py` — TrainerCallback for LLaMA-Factory
- Triggered via `RPE_CONFIG_PATH` environment variable

**What we'll do:** LoRA fine-tune Qwen2.5-7B-Instruct on the 4K-8K train set with RPE enabled (L=16384). Then evaluate on 4K-8K test (in-distribution) and 8K-16K test (out-of-distribution). Compare against vanilla and YaRN baselines.

### PoSE (Phase 4)

Will mirror RPE's structure but use structured chunk+skip-bias position manipulation instead of fully random positions.
