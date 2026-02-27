# MRCR Context Extension: RPE vs YaRN vs PoSE

**Date:** February 27, 2026
**Model:** Qwen2.5-7B-Instruct | **Hardware:** NYU Torch HPC (H200 80GB / L40S 48GB)

---

## 1. What is MRCR?

**MRCR (Multi-Round Coreference Resolution)** is a retrieval benchmark designed to test a model's ability to find and recall specific information buried in long multi-turn conversations.

- Each sample is a **multi-turn chat** (10+ messages, 4K-128K tokens) with 2 "needles" (target entities) scattered among similar-looking distractors
- The model must find the correct needle and reproduce it **verbatim**, prepended with a **random string** (prevents memorization cheating)
- **Grading:** `SequenceMatcher.ratio()` between model output and expected answer (0.0-1.0). Returns 0.0 if the random prefix is missing.

### Why MRCR?

We wanted to test whether position manipulation strategies (RPE, PoSE, YaRN) help models **generalize to longer contexts** on a real NLP task. MRCR is ideal because:
1. It requires **precise retrieval** at specific positions (not just general understanding)
2. Performance degrades predictably with context length (clear signal)
3. The multi-turn format maps naturally to instruction-tuned models

### Data Bins

We binned MRCR samples by token count into 5 context-length ranges:

| Bin | Token Range | Train | Test | Purpose |
|-----|-------------|-------|------|---------|
| 0 | 4K-8K | 60 | 26 | **Training data** (within Qwen's native window) |
| 1 | 8K-16K | — | 30 | Near-transfer generalization |
| 2 | 16K-32K | — | 30 | Within native 32K window |
| 3 | 32K-64K | — | 30 | Beyond native window |
| 4 | 64K-128K | — | 30 | Far extrapolation |

**Key experimental setup:** Train on bin 0 only, evaluate on ALL bins. Tests whether training strategies help the adapter generalize to 2x-16x longer contexts than it was trained on.

---

## 2. Three Position Manipulation Strategies

### 2a. YaRN (Yet another RoPE extensioN)

**What it does:** Modifies the RoPE **frequency basis** (`inv_freq`) to extend context.
- High-frequency dims (local patterns): interpolated (divided by factor)
- Low-frequency dims (global patterns): left unchanged
- Middle dims: smooth blend via linear ramp
- Attention temperature scaled by `0.1 * ln(factor) + 1.0`

**When applied:** Can be used at both training and inference time.
- For Qwen2.5-7B with factor=4.0: 40/64 dims modified, attention_factor=1.139
- Effective context: 32K native -> ~131K extended

**Analogy:** "Recalibrate the ruler to measure longer distances"

### 2b. RPE (Randomized Positional Encodings)

**Paper:** Ruoss et al. (DeepMind), arXiv:2305.16843

**What it does:** Replaces sequential position IDs `[0, 1, 2, ...]` with **sorted random integers** from `[0, L)` where `L >> sequence length`.

```
Standard: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
RPE:      [12, 45, 89, 156, 203, 891, 1204, 3401, 5502, 7891]
          ↑ random gaps between EVERY token
```

**When applied:** Training-only. At eval, standard sequential positions are used.
**Analogy:** "Train with a scrambled ruler so you handle any ruler"

### 2c. PoSE (Positional Skip-wisE)

**Paper:** Zhu et al. (ICLR 2024), arXiv:2309.10400

**What it does:** Splits the training sequence into **2 contiguous chunks** with a **random skip** between them.

```
PoSE: [0, 1, 2, 3, 4,  6560, 6561, 6562, 6563, 6564]
       └─ chunk 1 ─┘    └──── chunk 2 (after skip) ────┘
       contiguous        contiguous
```

**When applied:** Training-only. At eval, standard sequential positions are used.

**Key difference from RPE:**
- RPE: O(N) discontinuities — gaps between every token
- PoSE: O(1) discontinuities — just 1 gap between 2 sequential blocks
- PoSE preserves within-chunk local structure (which the model learned during pretraining)

**Algorithm (2-chunk, per paper):**
1. Pick random split: `rt1 = randint(1, seq_length // 2)`
2. Chunk 1: positions `[0, 1, ..., rt1-1]` (always starts at 0)
3. Pick random skip: `rt = randint(0, target_length - seq_length)`
4. Chunk 2: positions `[rt, rt+1, ..., rt + remaining - 1]`

### Comparison Table

| | What it changes | When applied | Preserves local structure? | Training needed? |
|---|---|---|---|---|
| **YaRN** | RoPE frequencies (`inv_freq`) | Train and/or eval | Yes | No (but can help) |
| **RPE** | Position IDs (input to RoPE) | Training only | No (fully random) | Yes |
| **PoSE** | Position IDs (input to RoPE) | Training only | Yes (within chunks) | Yes |

YaRN and RPE/PoSE are **orthogonal** — they can be combined (and this turns out to be important).

---

## 3. Experiments Run

### Phase 2: No-Training Baselines

| # | Condition | Description |
|---|-----------|-------------|
| 1 | Vanilla | Base Qwen2.5-7B-Instruct, no modifications |
| 2 | YaRN inference-only | YaRN (factor=4.0) applied at inference, no training |

### Phase 3: LoRA Training (4 conditions)

All trained on **bin 0 (4K-8K, 60 samples)**. Identical hyperparameters except position strategy.

| # | Condition | Position strategy during training | Position strategy during eval |
|---|-----------|-----------------------------------|-------------------------------|
| 3 | LoRA baseline | Sequential `[0,1,2,...]` | Sequential |
| 4 | YaRN+LoRA | YaRN modifies `inv_freq` at model load | YaRN + sequential positions |
| 5 | RPE+LoRA (fixed L=32768) | Random sorted from `[0, 32768)` | Sequential (RPE off) |
| 6 | RPE+LoRA (curriculum) | RPE L: 10240 -> 16384 -> 24576 -> 32768 -> 32768 | Sequential (RPE off) |

### Phase 3b: Combination Experiments

| # | Condition | What's different |
|---|-----------|-----------------|
| 7 | RPE fixed + YaRN at eval | RPE-trained LoRA, YaRN applied at inference only |
| 8 | RPE curriculum + YaRN at eval | RPE curriculum LoRA, YaRN at inference only |

### Phase 4: PoSE Comparison (in progress)

| # | Condition | Position strategy during training |
|---|-----------|-----------------------------------|
| 9 | PoSE fixed (L=32768) | 2 chunks with random skip in `[0, 32768)` |
| 10 | PoSE curriculum | PoSE L: 10240 -> 16384 -> 24576 -> 32768 -> 32768 |

---

## 4. Results

### 4a. Main Results Table (SequenceMatcher score, higher = better)

| Condition | 4K-8K (n=26) | 8K-16K (n=30) | 16K-32K (n=30) | 32K-64K (n=30) | 64K-128K (n=30) | Retention |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla (no training) | 0.389 | 0.365 | 0.465 | 0.165 | 0.056 | 14.4% |
| YaRN inf-only (no training) | 0.346 | 0.302 | 0.319 | 0.242 | 0.114 | 32.9% |
| **LoRA baseline** | **0.998** | **0.966** | 0.746 | 0.545 | 0.317 | 31.8% |
| **YaRN+LoRA** | 0.891 | 0.692 | 0.619 | 0.619 | 0.441 | 49.5% |
| **RPE fixed** | 0.636 | 0.504 | 0.503 | 0.473 | 0.265 | 41.7% |
| **RPE curriculum** | 0.922 | 0.691 | **0.749** | 0.563 | 0.255 | 27.7% |
| **RPE fixed + YaRN eval** | 0.531 | 0.499 | 0.641 | 0.590 | 0.551 | **103.8%** |
| **RPE curriculum + YaRN eval** | 0.817 | 0.592 | 0.677 | **0.646** | **0.528** | **64.6%** |
| PoSE fixed | — | — | — | — | — | — |
| PoSE curriculum | — | — | — | — | — | — |

**Retention** = score at 64K-128K / score at 4K-8K (how well performance holds at extreme lengths).

### 4b. Key Findings

**1. LoRA fine-tuning is the dominant factor.**
All LoRA conditions massively outperform vanilla. Even the worst LoRA (RPE fixed at bin 4: 0.265) beats vanilla (0.056) by 373%. Fine-tuning teaches the model the MRCR *task format* — how to find needles and prepend the random string.

**2. RPE + YaRN is the best combination for length generalization.**
- RPE fixed + YaRN: **103.8% retention** — performance at 128K *exceeds* 4K-8K (near-flat curve!)
- RPE curriculum + YaRN: **64.6% retention** — highest absolute scores at bins 3-4

**3. Why RPE + YaRN works:**
RPE trains the adapter to be **position-invariant** (doesn't rely on absolute positions). YaRN at eval extends the **frequency basis** so attention can physically reach longer distances. Together: the adapter handles any position arrangement, and YaRN ensures RoPE frequencies don't degrade at long range.

**4. YaRN+LoRA is the best single-strategy condition.**
Best at bins 3-4 (0.619, 0.441) and best retention (49.5%) among non-combo conditions. But RPE+YaRN beats it at every bin beyond 16K.

**5. RPE alone hurts short-context accuracy.**
RPE fixed: 0.636 at bin 0 (vs baseline's 0.998). The random positions during training disrupt the task-learning signal. Only recovers when combined with YaRN at eval.

**6. Curriculum > Fixed for RPE.**
RPE curriculum (0.922 bin 0) >> RPE fixed (0.636 bin 0). Gradual L ramp lets the adapter first learn the task, then learn position-invariance.

### 4c. Degradation Slope

| Condition | Bin 0 Score | Bin 4 Score | Retention | Slope/bin |
|-----------|:-----------:|:-----------:|:---------:|:---------:|
| Vanilla | 0.389 | 0.056 | 14.4% | -0.083 |
| LoRA baseline | 0.998 | 0.317 | 31.8% | -0.170 |
| YaRN+LoRA | 0.891 | 0.441 | 49.5% | -0.113 |
| RPE cur + YaRN eval | 0.817 | 0.528 | 64.6% | -0.072 |
| RPE fixed + YaRN eval | 0.531 | 0.551 | 103.8% | +0.005 |

RPE fixed + YaRN has an **essentially flat** performance curve across all context lengths.

### 4d. Training Loss Summary

| Condition | Avg Loss | Epoch 1 Loss | Final Loss | Max Grad Norm | Time |
|-----------|:--------:|:------------:|:----------:|:-------------:|:----:|
| LoRA baseline | 0.0039 | 0.0052 | ~0.0 | 0.37 | 21 min |
| YaRN+LoRA | 0.0123 | 0.0230 | 0.0003 | 2.56 | 21 min |
| RPE curriculum | 0.0265 | 0.0181 | 0.0010 | 0.95 | 21 min |
| RPE fixed | 0.3669 | 1.5740 | 0.0099 | **496** | 21 min |
| PoSE fixed | 0.0045 | 0.0070 | ~0.0 | 0.47 | 8 min |

PoSE trains much more smoothly than RPE — comparable to baseline loss, no gradient issues.

---

## 5. Detailed Technical Section

### 5.1 Model & Infrastructure

| Item | Value |
|------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Model params | 7.6B total |
| Trainable params (LoRA) | ~40M (0.5% of total) |
| HPC | NYU Torch HPC (`login.torch.hpc.nyu.edu`) |
| Training GPU | NVIDIA L40S (48GB VRAM) |
| Eval GPU | NVIDIA H200 (80GB VRAM) — needed for 64K+ sequences |
| Framework | transformers 4.52.4, PEFT (peft library), PyTorch |
| Precision | bf16 throughout |
| Model cache | `/scratch/mm14444/hf_cache` (offline mode, `HF_HUB_OFFLINE=1`) |

### 5.2 LoRA Hyperparameters (identical across ALL conditions)

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank (r) | 16 | Same as Phase 2 CCoT best config |
| LoRA alpha | 32 | 2x rank (standard) |
| LoRA dropout | 0.1 | Regularization |
| Target modules | `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj` | All 7 projection matrices |
| Learning rate | 2e-4 | Standard for LoRA |
| LR scheduler | cosine | Smooth decay to 0 |
| Warmup ratio | 0.1 | ~8 warmup steps out of 75 |
| Epochs | 5 | |
| Batch size | 1 per device | Memory-limited (8K token sequences) |
| Gradient accumulation | 4 | Effective batch size = 4 |
| Max sequence length | 8192 | Bin 0 upper bound |
| Max gradient norm | 1.0 | Gradient clipping |
| Gradient checkpointing | Yes | Required for 8K sequences on L40S |
| Seed | 42 | Reproducibility |
| Training data | 60 samples from bin 0 (4K-8K) | |
| Steps/epoch | 15 (= 60 samples / 4 effective batch) | |
| Total steps | 75 (= 15 x 5 epochs) | |

**Only variable: position strategy during training.** Everything else is identical.

### 5.3 YaRN Configuration

**Training (YaRN+LoRA condition):** YaRN applied at model load, before LoRA attachment. The LoRA adapter trains on top of YaRN-modified RoPE.

**Eval (YaRN conditions):** YaRN applied at model load before inference.

```python
# transformers 4.52.x (HPC):
config.rope_scaling = {"type": "yarn", "factor": 4.0}

# transformers 5.0+ (local dev):
config.rope_parameters.update({"type": "yarn", "rope_type": "yarn", "factor": 4.0})
# Must use .update() to preserve rope_theta (loses it if you assign directly)
```

**Verification at every run:** Check that `inv_freq` differs from vanilla (40/64 dims should change). Don't trust `rope_type` attribute alone — it can lie.

**Important distinction:**
- **YaRN+LoRA** (condition 4): YaRN at BOTH training and eval
- **RPE + YaRN eval** (conditions 7-8): RPE at training, YaRN at eval only (no YaRN during training)

### 5.4 RPE Configuration

**Fixed (L=32768):**
```yaml
# configs/rpe_config_mrcr.yaml
rpe:
  enabled: true
  max_simulation_length: 32768  # 4x max training seq = bin 2 upper bound
  seed: null
  training_mode: true
  inference_mode: false
```

With training sequences of ~6K tokens and L=32768:
- Average position gap: 32768/6000 ≈ 5.5 between consecutive tokens
- Positions are random sorted integers from `[0, 32768)`

**Curriculum:**
```yaml
# configs/rpe_config_mrcr_curriculum.yaml
rpe:
  enabled: true
  max_simulation_length: 32768
  seed: null
  training_mode: true
  inference_mode: false
  curriculum:
    1: 10240    # 1.25x max seq — near-sequential (avg gap ~1.3)
    2: 16384    # 2x max seq — mild randomization (avg gap ~2.0)
    3: 24576    # 3x max seq — moderate (avg gap ~3.1)
    4: 32768    # 4x max seq = target (avg gap ~4.1)
    5: 32768    # consolidation
```

**How RPE patches the model:**
1. `RPEPatcher.patch()` monkey-patches `model.forward()`
2. Wrapper intercepts `position_ids` argument
3. If `model.training == True`: generates random sorted positions, passes them in
4. If `model.training == False`: uses standard sequential positions (passthrough)
5. `RPEPatcher.unpatch()` restores original forward at end of training

### 5.5 PoSE Configuration

**Fixed (target_length=32768):**
```yaml
# configs/pose_config_mrcr.yaml
pose:
  enabled: true
  target_length: 32768
  seed: null
```

**Curriculum:**
```yaml
# configs/pose_config_mrcr_curriculum.yaml
pose:
  enabled: true
  target_length: 32768
  seed: null
  curriculum:
    1: 10240
    2: 16384
    3: 24576
    4: 32768
    5: 32768
```

**Same schedule as RPE curriculum** for fair comparison. Same patching mechanism (monkey-patch `model.forward`, gate on `model.training`).

### 5.6 Eval Configuration

All conditions evaluated identically:
- **Greedy decoding** (`do_sample=False`)
- **Max new tokens:** 2048
- **Sequential positions** for all conditions (RPE/PoSE are training-only)
- **YaRN at eval** only for YaRN+LoRA and RPE+YaRN conditions

### 5.7 Data Pipeline

`scripts/prepare_data.py`:
- Downloads MRCR from HuggingFace (`dmayhem93/MRCR`)
- Filters to 2-needle samples only
- Tokenizes with Qwen tokenizer, bins by token count
- Creates 70/30 train/test splits per bin
- Saves to `data/bin{0-4}_{range}/train.json` and `test.json`

**Key function — `tokenize_prompt()`:**
```python
messages = json.loads(sample["prompt"])
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
token_count = len(tokenizer.encode(text))
```

### 5.8 Training Pipeline

`scripts/train_mrcr_lora.py` — standalone (not LLaMA-Factory) because MRCR's multi-turn format doesn't fit Alpaca template, and we need YaRN injection at model load.

**Key class — `MRCRDataset`:**
```python
# Applies Qwen chat template to multi-turn messages
# Creates labels with -100 mask on prompt tokens
# Trains ONLY on the final assistant answer
labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
```

**Key function — `load_model_for_training()`:**
```python
# 1. Load tokenizer
# 2. Apply YaRN config if --enable-yarn
# 3. Load base model
# 4. Verify YaRN (check inv_freq differs from vanilla)
# 5. Enable gradient checkpointing
# 6. Attach LoRA via PEFT
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
))
```

**RPE/PoSE integration — via TrainerCallback:**
```python
# RPE:
from composable_cot.scripts.rpe_llamafactory_patch import RPETrainerCallback
callbacks.append(RPETrainerCallback("configs/rpe_config_mrcr.yaml"))

# PoSE:
from composable_cot.scripts.pose_patch import PoSETrainerCallback
callbacks.append(PoSETrainerCallback("configs/pose_config_mrcr.yaml"))
```

Both callbacks implement:
- `on_train_begin()` — patches model.forward with position manipulation
- `on_epoch_begin()` — updates L/target_length for curriculum
- `on_train_end()` — unpatches (clean state for model saving)

**CLI commands (exact SLURM invocations):**

```bash
# LoRA baseline
python composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/lora_baseline" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

# YaRN+LoRA (only difference: --enable-yarn --yarn-factor 4.0)
python composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/yarn_lora" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

# RPE fixed (only difference: --rpe-config)
python composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config "composable_cot/mrcr_context_extension/configs/rpe_config_mrcr.yaml" \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/rpe_lora" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

# RPE curriculum (only difference: curriculum config)
python composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config "composable_cot/mrcr_context_extension/configs/rpe_config_mrcr_curriculum.yaml" \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/rpe_curriculum_lora" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

# PoSE fixed (only difference: --pose-config)
python composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --pose-config "composable_cot/mrcr_context_extension/configs/pose_config_mrcr.yaml" \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/pose_lora" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

# PoSE curriculum (only difference: curriculum config)
python composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --pose-config "composable_cot/mrcr_context_extension/configs/pose_config_mrcr_curriculum.yaml" \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/pose_curriculum_lora" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42
```

### 5.9 Eval Pipeline

`scripts/eval_mrcr.py` — single script for all conditions.

**Key function — `grade_mrcr()`:**
```python
def grade_mrcr(response, answer, random_string_to_prepend):
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())
```

**Key function — `load_model()`:**
```python
# Handles 3 modes:
# 1. Vanilla: just load model
# 2. YaRN: modify config.rope_scaling before model load, verify inv_freq
# 3. LoRA: load base model, then PeftModel.from_pretrained() + merge_and_unload()
```

**Eval CLI commands:**

```bash
# Vanilla baseline
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/vanilla_bin0_4K-8K" \
    --max-new-tokens 2048

# YaRN baseline (add --enable-yarn)
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_bin0_4K-8K" \
    --max-new-tokens 2048

# LoRA baseline eval (add --lora-ckpt)
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --lora-ckpt "composable_cot/mrcr_context_extension/checkpoints/lora_baseline" \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/lora_baseline_bin0_4K-8K" \
    --max-new-tokens 2048

# YaRN+LoRA eval (add --enable-yarn AND --lora-ckpt)
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --lora-ckpt "composable_cot/mrcr_context_extension/checkpoints/yarn_lora" \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_lora_bin0_4K-8K" \
    --max-new-tokens 2048

# RPE fixed eval (just --lora-ckpt, no RPE at eval — RPE is training-only)
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --lora-ckpt "composable_cot/mrcr_context_extension/checkpoints/rpe_lora" \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/rpe_lora_bin0_4K-8K" \
    --max-new-tokens 2048

# RPE fixed + YaRN eval (RPE checkpoint + YaRN at inference)
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --lora-ckpt "composable_cot/mrcr_context_extension/checkpoints/rpe_lora" \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/rpe_lora_yarn_bin0_4K-8K" \
    --max-new-tokens 2048
```

### 5.10 Code Architecture

```
RPE/
├── rpe/                          # Core modules
│   ├── __init__.py               # Exports: RandomizedPositionalEncoding, RPEPatcher, PositionalSkipWise, PoSEPatcher
│   ├── core.py                   # RandomizedPositionalEncoding class (random sorted positions)
│   ├── patching.py               # RPEPatcher class (monkey-patches model.forward)
│   ├── pose.py                   # PositionalSkipWise class (2-chunk skip positions)
│   └── pose_patching.py          # PoSEPatcher class (monkey-patches model.forward)
│
├── composable_cot/
│   ├── scripts/
│   │   ├── rpe_llamafactory_patch.py  # RPETrainerCallback (loads YAML, patches/unpatches)
│   │   └── pose_patch.py              # PoSETrainerCallback (same interface as RPE)
│   │
│   └── mrcr_context_extension/
│       ├── scripts/
│       │   ├── prepare_data.py        # Data pipeline: HF download -> tokenize -> bin -> split
│       │   ├── train_mrcr_lora.py     # Training: MRCRDataset, load_model, Trainer, callbacks
│       │   └── eval_mrcr.py           # Eval: load model, generate, grade_mrcr, save results
│       ├── configs/
│       │   ├── rpe_config_mrcr.yaml               # RPE fixed L=32768
│       │   ├── rpe_config_mrcr_curriculum.yaml     # RPE curriculum L schedule
│       │   ├── pose_config_mrcr.yaml               # PoSE fixed target=32768
│       │   └── pose_config_mrcr_curriculum.yaml    # PoSE curriculum target schedule
│       ├── hpc/                       # 22 SLURM scripts (train + eval for all conditions)
│       ├── data/                      # bin0-bin4 train/test JSONs
│       ├── outputs/                   # eval_results.json + predictions.json per condition
│       └── checkpoints/               # LoRA weights per condition
```

### 5.11 Key Implementation Detail: Patching Mechanism

Both RPE and PoSE use the same patching architecture:

```python
class RPEPatcher:   # (or PoSEPatcher)
    def patch(self):
        self._original_forward = self.model.forward

        def rpe_forward(input_ids=None, **kwargs):
            if self.model.training:
                # Replace position_ids with random/PoSE positions
                kwargs["position_ids"] = self.rpe.get_randomized_positions(seq_length)
            else:
                # Use standard sequential positions (eval passthrough)
                kwargs["position_ids"] = torch.arange(seq_length)
            return self._original_forward(input_ids=input_ids, **kwargs)

        self.model.forward = rpe_forward

    def unpatch(self):
        self.model.forward = self._original_forward
```

This means:
- **Training:** model sees manipulated positions via the patched forward
- **Saving:** unpatch is called before save (clean LoRA weights)
- **Eval:** `model.training=False` -> standard sequential positions (or you load the checkpoint fresh)

### 5.12 SLURM Scripts (complete list)

| Script | Purpose | GPU | Time |
|--------|---------|-----|------|
| `hpc/prepare_data.slurm` | Data pipeline | CPU | 1hr |
| `hpc/eval_vanilla.slurm` | Vanilla baseline eval (bins 0-4) | H200 | 12hr |
| `hpc/eval_yarn.slurm` | YaRN baseline eval (bins 0-4) | H200 | 12hr |
| `hpc/train_lora_baseline.slurm` | Train LoRA baseline | L40S/H200 | 4hr |
| `hpc/train_lora_yarn.slurm` | Train YaRN+LoRA | L40S/H200 | 4hr |
| `hpc/train_lora_rpe.slurm` | Train RPE fixed | L40S/H200 | 4hr |
| `hpc/train_lora_rpe_curriculum.slurm` | Train RPE curriculum | L40S/H200 | 4hr |
| `hpc/eval_lora_baseline.slurm` | Eval LoRA baseline (bins 0-4) | H200 | 12hr |
| `hpc/eval_lora_yarn.slurm` | Eval YaRN+LoRA (bins 0-4) | H200 | 12hr |
| `hpc/eval_lora_rpe.slurm` | Eval RPE fixed (bins 0-4) | H200 | 12hr |
| `hpc/eval_lora_rpe_curriculum.slurm` | Eval RPE curriculum (bins 0-4) | H200 | 12hr |
| `hpc/eval_lora_rpe_yarn.slurm` | Eval RPE fixed + YaRN (bins 0-4) | H200 | 12hr |
| `hpc/eval_lora_rpe_curriculum_yarn.slurm` | Eval RPE cur + YaRN (bins 0-4) | H200 | 12hr |
| `hpc/train_lora_rpe_v2.slurm` | Train RPE fixed v2 (LR 1e-4, grad 0.5) | H200 | 4hr |
| `hpc/train_lora_rpe_curriculum_v2.slurm` | Train RPE cur v2 (aggressive L) | H200 | 4hr |
| `hpc/eval_lora_rpe_v2.slurm` | Eval RPE fixed v2 | H200 | 12hr |
| `hpc/eval_lora_rpe_curriculum_v2.slurm` | Eval RPE cur v2 | H200 | 12hr |
| `hpc/train_lora_pose.slurm` | Train PoSE fixed | H200 | 4hr |
| `hpc/train_lora_pose_curriculum.slurm` | Train PoSE curriculum | H200 | 4hr |
| `hpc/eval_lora_pose.slurm` | Eval PoSE fixed (bins 0-4) | H200 | 12hr |
| `hpc/eval_lora_pose_curriculum.slurm` | Eval PoSE curriculum (bins 0-4) | H200 | 12hr |

### 5.13 The YaRN Bug Saga

**Problem:** YaRN initially produced scores identical to vanilla despite being "enabled."

**Root cause (transformers 4.52.4):** We included `original_max_position_embeddings` in `rope_scaling` — this is a vLLM-only parameter that caused HF transformers to silently compute wrong frequencies.

**Root cause (transformers 5.0+):** `rope_theta` moved inside `rope_parameters` dict. Using `config.rope_scaling = {...}` replaces the entire dict, losing `rope_theta`.

**Fix:**
1. Removed vLLM-only params from config
2. Use `config.rope_parameters.update(...)` to preserve `rope_theta`
3. Added `inv_freq` verification (compare against vanilla — don't trust `rope_type` attribute)
4. Added manual fallback (`_apply_yarn_manual`) that patches `inv_freq` directly
5. Pre-flight sanity check in SLURM scripts verifies YaRN math before model load

### 5.14 v2 RPE Experiments (did not improve)

| Condition | Change from v1 | Result |
|-----------|---------------|--------|
| RPE fixed v2 | LR 1e-4, grad clip 0.5, 1-epoch warmup (L=8192 -> 32768) | Bin 0 improved (+17%), long-context unchanged |
| RPE curriculum v2 | Aggressive L: 16K -> 65K (vs v1's 10K -> 32K) | Worse everywhere, 7 zero-scores at bin 4 |

**Conclusion:** v1 remains the best RPE-only configuration. The original gentle curriculum (10K -> 32K) works best.

---

## 6. Status & Next Steps

### Completed
- Phase 2: Vanilla + YaRN baselines
- Phase 3: 4 LoRA conditions (baseline, YaRN, RPE fixed, RPE curriculum)
- Phase 3b: RPE+YaRN combination, v2 RPE experiments
- Phase 4: PoSE implementation + training (PoSE fixed complete, PoSE curriculum complete)

### In Progress
- Phase 4: PoSE evaluation (eval scripts ready, awaiting results)

### Pending
- PoSE + YaRN at eval (if PoSE results warrant it)
- Final consolidated analysis & paper-ready figures

### Key Open Question
Does PoSE (which preserves local structure) produce better length generalization than RPE (which destroys it)? The training loss suggests PoSE is much healthier (loss comparable to baseline, no gradient issues), but the real test is eval on bins 3-4.

---

## Appendix A: PoSE Paper Summary (for Q&A)

### Paper Details
**Title:** "PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wisE Training"
**Authors:** Zhu et al. (Peking University)
**Venue:** ICLR 2024 (arXiv:2309.10400)

### Models Tested
- LLaMA-7B (native 2K context)
- LLaMA2-7B (native 4K context)
- GPT-J-6B (native 2K context)
- Baichuan2-7B (native 4K context)

### Tasks & Datasets

**Language modeling (perplexity):**
- GovReport (19,402 congressional reports, avg 7,866 tokens)
- Proof-pile (13GB math dataset)
- Books3 (literary works)
- PG-19 (20 Gutenberg books >128K tokens)

**Passkey retrieval:**
- Synthetic needle-in-haystack: find a 5-digit passkey in a long context
- 50 trials per context length (2K-32K)

**Downstream tasks (Open LLM Leaderboard):**
- Zero-shot: BoolQ, PIQA, WinoGrande, TruthfulQA
- Few-shot: ARC-Challenge (25-shot), HellaSwag (10-shot)

### Why Only 2 Chunks? (Key ablation)

They tested N = 1, 2, 3, and 2048 chunks. Results on Proof-pile (2K -> 16K extension):

| Chunks (N) | 2K PPL | 4K PPL | 8K PPL | 16K PPL | Notes |
|:---:|:---:|:---:|:---:|:---:|---|
| N=1 | 2.83 | >1000 | >1000 | >1000 | = Position Interpolation, catastrophic failure |
| **N=2** | **2.95** | **2.74** | **2.61** | **2.60** | Sweet spot |
| N=3 | 2.93 | 2.72 | 2.60 | 2.59 | Marginal improvement over N=2 |
| N=2048 | 7.26 | 6.83 | 6.76 | 7.73 | **= RandPos/RPE**, 2.5x worse PPL |

**Rationale from paper:** "An increase in the number of chunks will further deviate from the position structure of pre-training, which may harm the ability acquired during pre-training." N=2 balances efficiency (exposing model to wide position range) with effectiveness (preserving contiguity).

### PoSE vs RandPos/RPE (Direct Comparison from Paper)

**Extending 2K -> 16K (perplexity, lower = better):**

| Dataset | RandPos (=RPE) | PoSE | Gap |
|---------|:-:|:-:|---|
| GovReport 2K | 11.63 | **4.84** | 2.4x worse |
| GovReport 16K | 15.16 | **4.60** | 3.3x worse |
| Proof-pile 2K | 7.26 | **2.95** | 2.5x worse |
| Proof-pile 16K | 7.73 | **2.60** | 3.0x worse |

**Extending 2K -> 32K (even more dramatic):**

| Dataset | RandPos (=RPE) | PoSE | Gap |
|---------|:-:|:-:|---|
| GovReport 32K | 97.57 | **4.66** | 21x worse |
| Proof-pile 32K | 66.47 | **2.59** | 26x worse |

RandPos/RPE perplexity **explodes** at longer extensions because fully random positions destroy local contiguity that the pretrained model depends on.

### Training Details
- **Dataset:** The Pile (split 00, ~100K samples, min 2,048 tokens each)
- **Training steps:** 1,000
- **Batch size:** 64
- **Training context window:** Fixed at model's native size (2K or 4K) — key efficiency win
- **Hardware:** 8x V100 (32GB) for training, 1x A100 for eval
- **Framework:** DeepSpeed ZeRO Stage 3
- **Interpolation strategies tested:** Linear, NTK, YaRN (YaRN was best, especially at extreme lengths)

### Key Results
- PoSE + YaRN extends LLaMA from 2K to **128K** (64x) with PPL of 11.33 on PG-19
- Passkey retrieval: >=90% accuracy across full extended range
- Downstream tasks: minimal degradation (BoolQ: 75.11 -> 73.61, PIQA: 78.67 -> 77.80)
- After just **100 training steps**, PoSE matches full-length fine-tuning quality

### How Our Work Differs
1. **Task:** We use MRCR (retrieval accuracy), not perplexity. PoSE paper didn't test multi-hop retrieval.
2. **Model:** Qwen2.5-7B-Instruct (32K native) vs LLaMA-7B (2K native). Much larger starting window.
3. **Training:** LoRA adapters (40M params) vs full fine-tuning. Different capacity.
4. **Finding:** Our RPE + YaRN combo achieves 103.8% retention — suggesting RPE's "weakness" (destroying local structure) may actually be a *strength* for retrieval tasks where position-invariance matters more than fluency.

---

## Appendix B: L Sweep Experiment (Planned)

### Motivation
We only tested L=32768 for RPE. How do we know this is optimal? L controls the average gap between consecutive position IDs during training — too small and there's not enough randomization, too large and the signal is too noisy.

### L Values to Test

| L | Multiplier | Avg Position Gap | Target Bin |
|---|:---:|:---:|---|
| 16,384 | 2x | ~2.7 | bin 1 (8K-16K) |
| 32,768 | 4x | ~5.5 | bin 2 (16K-32K) — **current** |
| 65,536 | 8x | ~10.9 | bin 3 (32K-64K) |
| 131,072 | 16x | ~21.8 | bin 4 (64K-128K) |

Each L value gets both **fixed** and **curriculum** variants (8 new conditions total).

### Curriculum Schedules

| Final L | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---------|:---:|:---:|:---:|:---:|:---:|
| 16K | 8,192 | 10,240 | 12,288 | 16,384 | 16,384 |
| 32K | 10,240 | 16,384 | 24,576 | 32,768 | 32,768 |
| 64K | 10,240 | 24,576 | 40,960 | 65,536 | 65,536 |
| 128K | 10,240 | 32,768 | 65,536 | 131,072 | 131,072 |

### Hypothesis
- **L=16K** may have best bin 0-1 scores (less disruption) but worst long-context
- **L=32K** (current) may be the sweet spot for bins 0-2
- **L=65K/128K** may trade off short-context accuracy for better retention at bins 3-4
- When combined with YaRN at eval, higher L may finally show its value

### HPC Commands

```bash
# --- Training (submit all 6 in parallel) ---
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_L16k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_L64k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_L128k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_cur_L16k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_cur_L64k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_cur_L128k.slurm

# --- Eval (after training completes) ---
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_L16k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_L64k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_L128k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_cur_L16k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_cur_L64k.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_cur_L128k.slurm
```
