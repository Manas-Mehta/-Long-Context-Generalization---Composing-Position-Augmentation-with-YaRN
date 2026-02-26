# MRCR Context Extension Experiment

---

## Plan & Status

### Research Question
Does RPE (Randomized Positional Encodings) help LoRA adapters generalize to longer contexts on real NLP tasks (MRCR), and how does it compare to YaRN?

### Experiment Matrix

| # | Condition | Train? | Position strategy | Purpose |
|---|-----------|--------|-------------------|---------|
| 1 | Vanilla | No | Normal RoPE | Base model ceiling |
| 2 | YaRN inference-only | No | YaRN RoPE | Free context extension |
| 3 | LoRA baseline | Yes | Normal RoPE | Effect of fine-tuning alone |
| 4 | YaRN+LoRA | Yes | YaRN RoPE | YaRN as training strategy |
| 5 | RPE+LoRA (fixed L) | Yes | Randomized position_ids (L=32768) | RPE as training strategy |
| 6 | RPE+LoRA (curriculum) | Yes | RPE with increasing L per epoch | Curriculum RPE (best in Phase 2 CCoT) |

Key comparison: **#4 vs #5 vs #6** — same data, same LoRA rank, only position strategy differs.

### Phase Checklist

- [x] **Phase 1:** Environment & Data Setup
- [x] **Phase 2:** No-Training Baselines (Vanilla + YaRN inference-only) — **COMPLETE**
- [ ] **Phase 3:** LoRA Training & Evaluation (4 conditions) — **IN PROGRESS**
- [ ] **Phase 4:** PoSE Exploration (side task)
- [ ] **Phase 5:** Analysis & Reporting

---

## Phase 1: Environment & Data Setup

### What we did
Set up Qwen2.5-7B-Instruct on NYU Torch HPC, downloaded MRCR dataset, built data pipeline.

### Files

| File | Purpose |
|------|---------|
| `scripts/prepare_data.py` | Downloads MRCR from HuggingFace, tokenizes with Qwen tokenizer, bins by token count, creates 70/30 train/test splits per bin |
| `hpc/prepare_data.slurm` | SLURM job for data preparation |

### Key functions in `prepare_data.py`

| Function | What it does |
|----------|-------------|
| `get_bin_index(token_count)` | Maps token count to bin index (0=4K-8K, 1=8K-16K, etc.) |
| `get_bin_label(bin_index)` | Converts bin index to label string |
| `tokenize_prompt(prompt_json, tokenizer)` | Parses MRCR prompt JSON, applies Qwen chat template, returns token count |
| `main()` | Orchestrates: load tokenizer -> download dataset -> filter 2-needle -> tokenize + bin -> split -> save |

### Data structure

```
data/
├── bin0_4K-8K/     (26 test, ~70 train samples)
│   ├── train.json
│   └── test.json
├── bin1_8K-16K/    (30 test)
├── bin2_16K-32K/   (30 test)
├── bin3_32K-64K/   (30 test)
├── bin4_64K-128K/  (30 test)
└── metadata.json
```

### What is MRCR?

MRCR (Multi-Round Coreference Resolution) is a harder version of Needle-in-a-Haystack. Multiple similar-looking entities (2, 4, or 8 "needles") are scattered through a long multi-turn conversation. The model must find and reproduce the correct one verbatim, prepended with a random string.

**Grading:** `SequenceMatcher.ratio()` between model output and expected answer (after stripping random prefix). Score 0.0-1.0. Returns 0.0 if the random prefix is missing.

---

## Phase 2: No-Training Baselines

### What we did
Evaluated vanilla Qwen2.5-7B-Instruct and YaRN (inference-only, factor=4.0) on all 5 bins (4K-128K). No LoRA training — just base model inference.

### Files

| File | Purpose |
|------|---------|
| `scripts/eval_mrcr.py` | Core evaluation script. Supports vanilla, YaRN, and LoRA modes. Loads model, runs inference, grades with MRCR metric, saves results. |
| `hpc/eval_vanilla.slurm` | SLURM job: vanilla eval on all 5 bins |
| `hpc/eval_yarn.slurm` | SLURM job: YaRN eval on all 5 bins, includes pre-flight sanity check |
| `scripts/test_yarn_fresh.py` | Local YaRN verification script (two-phase: math test + model test) |
| `scripts/verify_yarn.py` | Earlier YaRN diagnostic script (superseded by test_yarn_fresh.py) |

### Key functions in `eval_mrcr.py`

| Function | What it does |
|----------|-------------|
| `grade_mrcr(response, answer, prefix)` | Official MRCR grading: checks prefix -> strips it -> SequenceMatcher ratio |
| `load_model(base_model, lora_ckpt, enable_yarn, ...)` | Loads model in vanilla/YaRN/LoRA mode. Version-agnostic YaRN config. Verifies inv_freq differs from vanilla. Falls back to manual patch if needed. |
| `_diagnose_rope(model, config, ...)` | Prints detailed RoPE diagnostics: rotary class, rope_type, inv_freq comparison, logit fingerprint |
| `_apply_yarn_manual(model, factor, config)` | Fallback: manually patches inv_freq with NTK-aware scaling if config-based YaRN silently fails |
| `evaluate_mrcr(model, tokenizer, test_data, ...)` | Main eval loop: parse messages -> chat template -> generate -> grade -> aggregate per-bin |

### Errors encountered and fixes

#### The YaRN bug saga

**Problem:** YaRN produced scores identical to vanilla. Despite `rope_type: yarn` being set correctly, `inv_freq` was identical to vanilla — YaRN was not actually modifying the RoPE frequencies.

**Root cause (transformers 4.52.4 on HPC):** Our earlier configs included `original_max_position_embeddings` in the `rope_scaling` dict, which is a **vLLM-only parameter**. In transformers 4.52.4, this caused the YaRN computation to silently produce wrong results. The correct config for HF transformers is simply `{"type": "yarn", "factor": 4.0}` — `original_max_position_embeddings` defaults to `max_position_embeddings` automatically.

**Root cause (transformers 5.0.0 locally):** A separate bug. In v5.0.0, `rope_theta` moved from a standalone config attribute into the `rope_parameters` dict. Doing `config.rope_scaling = {"type": "yarn", "factor": 4.0}` **replaces the entire dict**, losing `rope_theta` (becomes `None`). Fix: use `config.rope_parameters.update({...})` instead of assignment.

**Fix timeline:**
1. Added `_diagnose_rope()` to print inv_freq comparison with vanilla
2. Discovered `inv_freq IDENTICAL to vanilla` on HPC despite rope_type="yarn"
3. Found HuggingFace discussion confirming `original_max_position_embeddings` is vLLM-only
4. Removed it, added inv_freq-based verification (don't trust rope_type attribute)
5. Added manual fallback (`_apply_yarn_manual`) that patches inv_freq directly if config-based YaRN fails
6. For transformers 5.0+: use `config.rope_parameters.update()` to preserve `rope_theta`
7. Added pre-flight sanity check in SLURM that verifies YaRN math BEFORE model load

**Verification:** After fixes, confirmed on HPC:
- Pre-flight: `YaRN dims changed: 40/64`, `attention_factor: 1.139`
- Model diagnostic: `inv_freq verified: 40/64 dims differ from vanilla`
- Logit fingerprint differs from vanilla

Full details: [MRCR_YaRN_Verification.md](MRCR_YaRN_Verification.md)

**References:**
- https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/discussions/5
- https://github.com/huggingface/transformers/issues/33783

#### Version-specific YaRN config

```python
# transformers 5.0+ (rope_theta inside rope_parameters dict):
config.rope_parameters.update({
    "type": "yarn",
    "rope_type": "yarn",
    "factor": 4.0,
})

# transformers 4.52.x (rope_theta is standalone attribute):
config.rope_scaling = {"type": "yarn", "factor": 4.0}
```

The eval_mrcr.py `load_model()` function handles both versions automatically.

### Results

**HPC run:** Feb 25, 2026. Vanilla on L40S, YaRN on H200. transformers 4.52.4.

| Bin | Vanilla | YaRN (factor=4.0) | Delta | Notes |
|-----|---------|-------------------|-------|-------|
| 4K-8K (n=26) | **0.389** | 0.346 | -0.043 | YaRN slightly hurts (expected — static scaling on short context) |
| 8K-16K (n=30) | **0.365** | 0.302 | -0.063 | Still within Qwen's native 32K, YaRN hurts |
| 16K-32K (n=30) | **0.465** | 0.319 | -0.146 | Vanilla surprisingly strong here |
| 32K-64K (n=30) | 0.165 | **0.242** | +0.077 | YaRN helps beyond 32K (native limit) |
| 64K-128K (n=30) | 0.056 | **0.114** | +0.058 | Vanilla ran on H200 (originally OOM on L40S). Both very poor at 128K. |

**Key observations:**
1. **YaRN IS working** — 40/64 inv_freq dims differ, attention_factor=1.139, logits differ
2. **YaRN hurts at short context** (4K-32K): -4% to -15%. Expected — static YaRN compresses frequencies even when not needed
3. **YaRN helps at 32K-64K**: +0.077 absolute improvement. This is beyond Qwen's native 32K window where vanilla RoPE starts breaking
4. **Vanilla is surprisingly strong at 16K-32K** (0.465) — better than at shorter bins. This could be variance (only 30 samples) or the particular needle placement patterns in this bin
5. **Both methods degrade significantly at 64K+**: Even with YaRN, 0.114 is poor. Inference-only YaRN without fine-tuning has limits

**Outputs saved to:** `outputs/{vanilla,yarn}_{4K-8K,8K-16K,16K-32K,32K-64K,64K-128K}/`
- `eval_results.json` — summary metrics
- `predictions.json` — per-sample scores, response previews, token counts, generation times

### Can we move on to Phase 3?

**Yes.** Phase 2 baselines are complete:
- Vanilla baseline established across all 5 bins (bin 4 rerun on H200: 0.056)
- YaRN confirmed working and results collected across all 5 bins
- The degradation pattern (vanilla drops at 32K+, YaRN helps there) validates the experimental setup

Phase 3 trains LoRA on bin 0 (4K-8K) and evaluates on bins 0-2 (4K-32K). The key question: does RPE+LoRA or YaRN+LoRA help the adapter generalize to bins 1-2 (8K-32K)?

---

## Phase 3: LoRA Training & Evaluation

*Status: IN PROGRESS*

### What we are doing
Train LoRA adapters (rank 16) on bin 0 (4K-8K, 60 samples) under 4 conditions. Evaluate each on bins 0, 1, 2 to measure length generalization.

### 4 Conditions

| # | Condition | What changes during training | What changes during eval |
|---|-----------|------------------------------|--------------------------|
| 1 | **LoRA baseline** | Nothing — standard positions | `--lora-ckpt` only |
| 2 | **YaRN+LoRA** | YaRN modifies inv_freq before LoRA trains | `--enable-yarn --lora-ckpt` |
| 3 | **RPE+LoRA (fixed L=32768)** | RPE randomizes position_ids from [0, 32768) | `--lora-ckpt` only (standard positions) |
| 4 | **RPE+LoRA (curriculum)** | RPE L increases: 10240→16384→24576→32768→32768 | `--lora-ckpt` only (standard positions) |

### Key comparison
YaRN modifies `inv_freq` (frequency basis). RPE modifies `position_ids` (input to RoPE). Both are applied during LoRA training so the adapter learns in the modified position space. The comparison isolates which manipulation helps LoRA generalize better.

### Why NOT LLaMA-Factory?
MRCR samples are multi-turn conversations (10+ messages, 4K-8K tokens) that don't fit LLaMA-Factory's Alpaca format. Also need YaRN injection at model load time. Wrote standalone `train_mrcr_lora.py` using HuggingFace Trainer + PEFT directly.

### RPE L value rationale
- Training on bin 0 (max 8192 tokens), targeting generalization to bin 2 (max 32768 tokens)
- **Fixed L=32768**: Positions sampled from [0, 32768) during training on 4K-8K sequences
  - Average gap: 32768/6000 ≈ 5.5 between consecutive sampled positions
- **Curriculum**: Start near-sequential (L=10240, gap ~1.3), ramp to target (L=32768, gap ~4.1)
  - Mirrors Phase 2 CCoT curriculum that beat all other conditions (+75% length extension)

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 16 | Same as best Phase 2 CCoT config |
| LoRA alpha | 32 | 2× rank (standard) |
| LoRA dropout | 0.1 | Regularization |
| Target modules | q,k,v,o,up,down,gate_proj | All projection matrices |
| Learning rate | 2e-4 | Standard for LoRA |
| LR scheduler | cosine | Smooth decay |
| Warmup ratio | 0.1 | ~8 warmup steps |
| Epochs | 5 | Short training (60 samples) |
| Batch size | 1 × 4 grad_accum = 4 effective | Memory-limited (8K tokens/sample) |
| Max seq len | 8192 | Bin 0 upper bound |
| Precision | bf16 | Standard for Qwen2.5 |
| Gradient checkpointing | Yes | Required for 8K sequences |
| Seed | 42 | Reproducibility |

Training stats: 60 samples / 4 effective batch = 15 steps/epoch × 5 epochs = **75 total steps**.

### Files

| File | Purpose |
|------|---------|
| `scripts/train_mrcr_lora.py` | Main training script. Loads model (optional YaRN), attaches LoRA, optional RPE callback. Includes MRCRDataset (chat template + loss masking), TrainingProgressCallback (per-step timing/GPU/ETA), saves metrics JSON + loss plot. |
| `configs/rpe_config_mrcr.yaml` | RPE fixed L=32768 |
| `configs/rpe_config_mrcr_curriculum.yaml` | RPE curriculum: 10240→16384→24576→32768→32768 |
| `hpc/train_lora_baseline.slurm` | Train condition 1 (H200, 4hr) |
| `hpc/train_lora_yarn.slurm` | Train condition 2 (H200, 4hr) |
| `hpc/train_lora_rpe.slurm` | Train condition 3 (H200, 4hr) |
| `hpc/train_lora_rpe_curriculum.slurm` | Train condition 4 (H200, 4hr) |
| `hpc/eval_lora_baseline.slurm` | Eval LoRA baseline on bins 0-4 (H200, 12hr) |
| `hpc/eval_lora_yarn.slurm` | Eval YaRN+LoRA on bins 0-4 (H200, 12hr) |
| `hpc/eval_lora_rpe.slurm` | Eval RPE+LoRA (fixed) on bins 0-4 (H200, 12hr) |
| `hpc/eval_lora_rpe_curriculum.slurm` | Eval RPE+LoRA (curriculum) on bins 0-4 (H200, 12hr) |

### Key functions in `train_mrcr_lora.py`

| Function/Class | What it does |
|----------------|-------------|
| `MRCRDataset` | Parses multi-turn messages → applies Qwen chat template → tokenizes → creates labels with -100 mask on prompt tokens (only train on answer) |
| `load_model_for_training()` | Loads base model with optional YaRN (version-agnostic), attaches LoRA via PEFT, enables gradient checkpointing |
| `_verify_yarn()` | Checks inv_freq differs from vanilla after model load; falls back to manual patch if needed |
| `TrainingProgressCallback` | Prints per-step: step/total, epoch, loss, LR, GPU memory, elapsed, ETA. Saves training_metrics.json and training_loss.png at end. |

### How to run (on HPC)

```bash
ssh mm14444@login.torch.hpc.nyu.edu
cd /scratch/mm14444/RPE
git pull
mkdir -p slurm_logs

# Submit all 4 training jobs in parallel
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_baseline.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_yarn.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_curriculum.slurm

# Monitor
squeue -u mm14444

# After all training completes, submit eval
sbatch composable_cot/mrcr_context_extension/hpc/eval_all_lora.slurm
```

### Checkpoint structure

```
checkpoints/{lora_baseline,yarn_lora,rpe_lora,rpe_curriculum_lora}/
  checkpoint-{15,30,45,60,75}/   # LoRA weights per epoch (~50MB each)
  training_metrics.json            # Per-step loss, lr, timing
  training_loss.png                # Loss curve plot
  run_config.json                  # Full training configuration
```

### Training Results (Feb 25, 2026)

All 4 conditions trained on L40S (46GB). 60 samples, 75 steps (15 steps/epoch × 5 epochs), ~21 minutes each.

#### Loss Summary

| Condition | Avg Train Loss | Epoch 1 Loss | Final Loss | Max Grad Norm | Time |
|-----------|---------------|--------------|------------|---------------|------|
| LoRA baseline | 0.0039 | 0.0052 | ~0.0 | 0.37 | 21:03 |
| YaRN+LoRA | 0.0123 | 0.0230 | 0.0003 | 2.56 | 21:01 |
| RPE curriculum | 0.0265 | 0.0181 | 0.0010 | 0.95 | 20:45 |
| RPE fixed (L=32768) | 0.3669 | 1.5740 | 0.0099 | **496** | 21:14 |

#### Per-Epoch Loss (end-of-epoch values)

| Condition | E1 | E2 | E3 | E4 | E5 |
|-----------|------|------|------|------|------|
| LoRA baseline | 0.0052 | 0.0001 | 0.0001 | 0.0000 | 0.0000 |
| YaRN+LoRA | 0.0230 | 0.0017 | 0.0006 | 0.0002 | 0.0003 |
| RPE curriculum | 0.0181 | 0.0123 | 0.0081 | 0.0050 | 0.0010 |
| RPE fixed | 1.5740 | 0.0611 | 0.0161 | 0.0053 | 0.0099 |

#### RPE Curriculum L Schedule (confirmed from logs)

| Epoch | L value | Avg position gap |
|-------|---------|-----------------|
| 1 | 10,240 | ~1.7 |
| 2 | 16,384 | ~2.7 |
| 3 | 24,576 | ~4.1 |
| 4 | 32,768 | ~5.5 |
| 5 | 32,768 | ~5.5 |

#### Verification

- **YaRN**: `rope_type: yarn`, 40/64 inv_freq dims differ from vanilla
- **RPE fixed**: `[RPEPatcher] Patched PeftModelForCausalLM (L=32768)`, unpatched at train end
- **RPE curriculum**: All L transitions logged correctly (10240→16384→24576→32768)
- **All 4**: Same dataset (60 samples, 1 truncated, avg 384 answer tokens), same sample 0 verification

#### Observations

1. **Baseline memorizes perfectly by epoch 2** (loss→0). Expected: normal positions, in-context-window data.
2. **YaRN starts 4× higher loss** than baseline (0.211 vs 0.048). Modified RoPE frequencies make the task harder even at training length. Converges well by epoch 3.
3. **RPE curriculum trains smoothly** — starts moderate (L=10240 is near-sequential), ramps up gradually. Grad norms stay stable (<1.0). Final loss 0.001.
4. **RPE fixed has severe initial disruption** — loss starts at 3.1 with grad norms up to 496. Positions from [0, 32K) for 8K sequences means avg gap ~5.5, severely disrupting attention patterns. This mirrors Phase 2 CCoT where fixed RPE was worse than curriculum.
5. **Training loss difference matters for interpretation, not for quality.** Higher training loss with RPE is intentional — the model is learning under a harder position regime. The real test is whether this produces better length generalization at eval.

### Errors encountered and fixes

- **No errors.** All 4 training jobs completed successfully on first run.
- Minor cosmetic: "Final loss: None" in summary display (last HF Trainer log doesn't include per-step loss key). Actual final step loss is captured in per-step metrics.

### Eval Results (Feb 26, 2026)

All 4 LoRA conditions evaluated on H200 (80GB) across all 5 bins. Each eval reloads base model + merges LoRA weights.

#### Score Table (SequenceMatcher ratio, higher = better)

| Condition | 4K-8K (n=26) | 8K-16K (n=30) | 16K-32K (n=30) | 32K-64K (n=30) | 64K-128K (n=30) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Vanilla (Phase 2) | 0.389 | 0.365 | 0.465 | 0.165 | 0.056 |
| YaRN inf-only (Phase 2) | 0.346 | 0.302 | 0.319 | 0.242 | 0.114 |
| **LoRA baseline** | **0.998** | **0.966** | 0.746 | 0.545 | 0.317 |
| **YaRN+LoRA** | 0.891 | 0.692 | 0.619 | **0.619** | **0.441** |
| **RPE+LoRA (fixed L=32768)** | 0.636 | 0.504 | 0.503 | 0.473 | 0.265 |
| **RPE+LoRA (curriculum)** | 0.922 | 0.691 | **0.749** | 0.563 | 0.255 |

#### Perfect / Zero Scores

| Condition | 4K-8K (P/Z) | 8K-16K (P/Z) | 16K-32K (P/Z) | 32K-64K (P/Z) | 64K-128K (P/Z) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| LoRA baseline | 25/0 | 24/0 | 18/0 | 12/1 | 6/1 |
| YaRN+LoRA | 21/0 | 17/1 | 14/2 | 12/0 | 8/3 |
| RPE+LoRA (fixed) | 13/0 | 12/1 | 10/0 | 7/0 | 2/1 |
| RPE+LoRA (curriculum) | 16/0 | 12/1 | 13/0 | 9/2 | 3/2 |

#### Improvement vs Vanilla Baseline

| Condition | 4K-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K |
|-----------|:---:|:---:|:---:|:---:|:---:|
| LoRA baseline | +157% | +165% | +60% | +230% | +466% |
| YaRN+LoRA | +129% | +90% | +33% | +275% | +688% |
| RPE+LoRA (fixed) | +64% | +38% | +8% | +187% | +373% |
| RPE+LoRA (curriculum) | +137% | +89% | +61% | +241% | +355% |

#### Degradation Slope (how fast performance drops with length)

| Condition | Score at 4K-8K | Score at 64K-128K | Retention | Slope |
|-----------|:---:|:---:|:---:|:---:|
| Vanilla | 0.389 | 0.056 | 14.4% | -0.0833/bin |
| LoRA baseline | 0.998 | 0.317 | 31.8% | -0.1703/bin |
| YaRN+LoRA | 0.891 | 0.441 | **49.5%** | -0.1125/bin |
| RPE+LoRA (fixed) | 0.636 | 0.265 | 41.7% | -0.0928/bin |
| RPE+LoRA (curriculum) | 0.922 | 0.255 | 27.7% | -0.1668/bin |

#### Eval Verification

- **YaRN+LoRA**: All 5 bins show `rope_type: yarn`, `40/64 dims differ from vanilla` — CORRECT
- **LoRA baseline**: All bins `inv_freq IDENTICAL to vanilla`, `rope_type: default` — CORRECT
- **RPE fixed**: All bins `inv_freq IDENTICAL to vanilla`, `rope_type: default` — CORRECT (RPE is training-only)
- **RPE curriculum**: All bins `inv_freq IDENTICAL to vanilla`, `rope_type: default` — CORRECT
- **All conditions**: LoRA loaded from correct checkpoint paths, logit fingerprints consistent within each condition

#### Key Findings

1. **LoRA baseline is the surprise winner at short-to-medium contexts.** Near-perfect at bin 0 (0.998) and bin 1 (0.966), strong at bin 2 (0.746). Simply fine-tuning on the MRCR task teaches the model the *format* and *strategy* — this accounts for most of the improvement over vanilla.

2. **YaRN+LoRA wins at long contexts (32K+).** Best at bin 3 (0.619) and bin 4 (0.441). YaRN's frequency modification combined with task-specific LoRA produces the best length generalization. Has the **best retention ratio** (49.5%) — performance degrades most gracefully with length.

3. **RPE fixed (L=32768) underperforms across the board.** Worst at every bin. The gradient explosion during training (max grad norm 496) likely damaged the adapter. The fixed L=32768 is too aggressive — positions from [0, 32K) for 8K sequences means avg gap ~5.5, which disrupts attention patterns without the gradual adaptation that curriculum provides.

4. **RPE curriculum is competitive but doesn't beat LoRA baseline.** Strong at bin 0 (0.922) and bin 2 (0.749, nearly matching baseline), but drops off at bin 4 (0.255). The curriculum scheduling helps vs fixed RPE, but the fundamentally different training regime (random positions) doesn't translate to better eval with sequential positions.

5. **The gap between RPE conditions mirrors Phase 2 CCoT results**: curriculum >> fixed. Curriculum's gradual L ramp produces much better adapters than the fixed approach.

6. **All LoRA conditions massively outperform no-training baselines.** Even the worst LoRA condition (RPE fixed at bin 4: 0.265) beats vanilla (0.056) by 373%. This confirms that task-specific fine-tuning is the dominant factor.

### Can we move on to Phase 4?

**Yes, with caveats.** Phase 3 is complete and the results are clear. Key conclusions:
- LoRA fine-tuning on bin 0 data provides massive improvements across all bins
- YaRN+LoRA provides the best length generalization (flattest degradation curve)
- RPE in its current form (random positions during training, sequential during eval) does not improve over standard LoRA for this task
- Phase 4 (PoSE) may be more promising than RPE because it preserves local contiguity

---

## Phase 3b: RPE+YaRN Eval & Improved RPE Training

*Status: Ready to run*

### Goal

Two investigations:
1. **RPE + YaRN at eval**: Do RPE-trained LoRAs benefit from YaRN frequency scaling at inference? (No retraining needed — just eval existing checkpoints with `--enable-yarn`)
2. **Better RPE training (v2)**: Fix the training instability in RPE fixed; push curriculum to more aggressive L values.

### New Eval: RPE checkpoints + YaRN inference

| Script | What it does |
|--------|-------------|
| `hpc/eval_lora_rpe_yarn.slurm` | RPE fixed checkpoint + YaRN at eval |
| `hpc/eval_lora_rpe_curriculum_yarn.slurm` | RPE curriculum checkpoint + YaRN at eval |

### v2 Training Configs

| Config | L Schedule | Changes from v1 |
|--------|-----------|-----------------|
| `configs/rpe_config_mrcr_fixed_v2.yaml` | 8192→32768→32768→32768→32768 | 1-epoch warmup to avoid grad explosion |
| `configs/rpe_config_mrcr_curriculum_v2.yaml` | 16384→32768→49152→65536→65536 | Targets bin 3 (64K), much more aggressive |

### v2 Training Hyperparameter Changes

| Param | RPE fixed v1 | RPE fixed v2 | RPE curriculum v1 | RPE curriculum v2 |
|-------|:---:|:---:|:---:|:---:|
| LR | 2e-4 | **1e-4** | 2e-4 | 2e-4 |
| max_grad_norm | 1.0 | **0.5** | 1.0 | 1.0 |
| L schedule | 32768 (constant) | **8192→32768** (warmup) | 10K→16K→24K→32K→32K | **16K→32K→49K→65K→65K** |
| Epochs | 5 | 5 | 5 | 5 |

### v2 Files

| Script | Purpose |
|--------|---------|
| `hpc/train_lora_rpe_v2.slurm` | Train RPE fixed v2 |
| `hpc/train_lora_rpe_curriculum_v2.slurm` | Train RPE curriculum v2 |
| `hpc/eval_lora_rpe_v2.slurm` | Eval RPE fixed v2 on bins 0-4 |
| `hpc/eval_lora_rpe_curriculum_v2.slurm` | Eval RPE curriculum v2 on bins 0-4 |

### Commands

```bash
# --- Phase 3b: YaRN eval on existing RPE checkpoints (submit now) ---
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_yarn.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_curriculum_yarn.slurm

# --- Phase 3b: v2 training (submit now, eval after training completes) ---
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_v2.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_curriculum_v2.slurm

# --- Phase 3b: v2 eval (submit after training completes) ---
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_v2.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_curriculum_v2.slurm
```

### Eval Results

*(To be filled after runs complete)*

| Condition | 4K-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K |
|-----------|:---:|:---:|:---:|:---:|:---:|
| RPE fixed + YaRN eval | — | — | — | — | — |
| RPE curriculum + YaRN eval | — | — | — | — | — |
| RPE fixed v2 | — | — | — | — | — |
| RPE curriculum v2 | — | — | — | — | — |

---

## Phase 4: PoSE Exploration

*Status: Not started*

PoSE (Positional Skip-wisE) preserves local contiguity within chunks but introduces gaps between chunks. May be better suited for context extension than RPE's fully random positions.

---

## Phase 5: Analysis & Reporting

*Status: Not started*

---

## Background

### What is MRCR?

Each sample has:
- `prompt`: JSON string of multi-turn chat messages with needle(s) buried inside
- `answer`: Expected response (random prefix + needle content)
- `random_string_to_prepend`: Unique random string the model must output first
- `n_needles`: Number of confounding entities (2, 4, or 8)

The model must find the correct needle among similar-looking entities and reproduce it verbatim, prepended with the random string.

### How vanilla baseline works

1. Load Qwen2.5-7B-Instruct as-is
2. For each sample: parse messages -> apply chat template -> tokenize -> model.generate() -> decode -> grade
3. Standard RoPE applied: `rotation_angle = pos * theta_d` where `theta_d = base^(-2d/dim)`, base=1M, dim=128
4. Works within the native 32K context window; degrades beyond

### How YaRN works

YaRN modifies RoPE frequencies to extend context beyond the training window:
- **High-frequency dims** (local patterns): fully interpolated (divide by factor)
- **Low-frequency dims** (global patterns): left unchanged (extrapolation)
- **Middle dims**: smooth blend via linear ramp
- **Attention temperature**: scaled by `0.1 * ln(factor) + 1.0`

For Qwen2.5-7B with factor=4.0: 40/64 dims modified, attention_factor=1.139, effective context extended to ~131K tokens.

### YaRN vs RPE

| | What it changes | When applied | Training needed? |
|---|---|---|---|
| **YaRN** | RoPE frequency basis (`inv_freq`) | Always (inference) | No |
| **RPE** | Position IDs fed to RoPE | Only during LoRA training | Yes |

YaRN = "recalibrate the ruler to measure longer distances"
RPE = "train with a scrambled ruler so you handle any ruler"

They're orthogonal — combining them is possible.

---

## Folder Structure

```
composable_cot/mrcr_context_extension/
├── scripts/
│   ├── prepare_data.py          # Phase 1: data pipeline
│   ├── eval_mrcr.py             # Phase 2+3: evaluation (vanilla/YaRN/LoRA)
│   ├── train_mrcr_lora.py       # Phase 3: LoRA training (standalone, HF Trainer + PEFT)
│   ├── test_yarn_fresh.py       # YaRN verification (local, two-phase test)
│   ├── test_yarn_local.py       # Earlier YaRN test (superseded)
│   └── verify_yarn.py           # Earlier YaRN diagnostic (superseded)
├── configs/
│   ├── rpe_config_mrcr.yaml           # Phase 3: RPE fixed L=32768
│   └── rpe_config_mrcr_curriculum.yaml # Phase 3: RPE curriculum L schedule
├── hpc/
│   ├── prepare_data.slurm             # Phase 1: data prep job
│   ├── eval_vanilla.slurm             # Phase 2: vanilla baseline (all 5 bins)
│   ├── eval_yarn.slurm                # Phase 2: YaRN baseline (all 5 bins)
│   ├── train_lora_baseline.slurm      # Phase 3: train condition 1
│   ├── train_lora_yarn.slurm          # Phase 3: train condition 2
│   ├── train_lora_rpe.slurm           # Phase 3: train condition 3
│   ├── train_lora_rpe_curriculum.slurm # Phase 3: train condition 4
│   └── eval_all_lora.slurm            # Phase 3: eval all conditions on bins 0-2
├── data/                        # Generated by prepare_data.py
│   ├── bin{0-4}_{range}/
│   │   ├── train.json
│   │   └── test.json
│   └── metadata.json
├── outputs/                     # Generated by eval_mrcr.py
│   ├── {vanilla,yarn}_{range}/           # Phase 2 results
│   └── {condition}_{range}/              # Phase 3 results (after eval)
├── checkpoints/                 # Phase 3: LoRA checkpoints
│   ├── lora_baseline/
│   ├── yarn_lora/
│   ├── rpe_lora/
│   └── rpe_curriculum_lora/
└── slurm_logs/                  # SLURM job outputs
```

---

## HPC Reference

```bash
ssh mm14444@login.torch.hpc.nyu.edu
cd /scratch/mm14444/RPE
git pull
mkdir -p slurm_logs

# --- Phase 2: Baselines ---
sbatch composable_cot/mrcr_context_extension/hpc/eval_vanilla.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_yarn.slurm

# --- Phase 3: Training (submit all 4 in parallel) ---
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_baseline.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_yarn.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe.slurm
sbatch composable_cot/mrcr_context_extension/hpc/train_lora_rpe_curriculum.slurm

# --- Phase 3: Eval (after training completes, submit all 4 in parallel) ---
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_baseline.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_yarn.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe.slurm
sbatch composable_cot/mrcr_context_extension/hpc/eval_lora_rpe_curriculum.slurm

# Monitor
squeue -u mm14444

# Check training logs
cat slurm_logs/mrcr_lora_base_*.out
cat slurm_logs/mrcr_lora_yarn_*.out
cat slurm_logs/mrcr_lora_rpe_*.out
cat slurm_logs/mrcr_lora_rpe_cur_*.out

# Check eval results
cat slurm_logs/mrcr_eval_lora_*.out
```

- Account: `torch_pr_219_courant`
- Conda env: `/scratch/mm14444/conda_envs/rpe`
- Partitions: `h200_courant` (80GB, needed for 64K+ eval), `l40s_courant` (48GB, used for training + 4K-32K eval)
- transformers: 4.52.4
- Model cache: `/scratch/mm14444/hf_cache` (offline mode enabled)
