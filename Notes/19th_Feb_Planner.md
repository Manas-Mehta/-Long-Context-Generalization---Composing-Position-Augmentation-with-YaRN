# 19th Feb Planner: RPE for Context Extension with MRCR

---

## Analysis: Why RPE on MRCR is a Direct Extension of Our CCoT Work

### The connection between our CCoT experiment and MRCR

**The experimental structure is the same.** The OOD problem exists at the **LoRA adapter level**, not the base model level.

Qwen2.5-7B handles positions 0-32K natively — but the LoRA adapter is only fine-tuned on sequences of a certain length. When we train LoRA on 4K-8K token examples, the adapter weights have only been optimized to work at positions 0-8192. At test time on 8K-16K inputs, the adapter must function at positions 8192-16384 — positions it was never fine-tuned on, even though the base model knows them.

This is structurally identical to Phase 2:

| | CCoT (Phase 2) | MRCR (proposed) |
|---|---|---|
| LoRA trained on positions | 0 - ~576 | 0 - ~8192 |
| LoRA tested on positions | ~576 - ~1500 | ~8192 - ~16384 |
| Base model native range | 0 - 32768 | 0 - 32768 |
| Both within base range? | Yes | Yes |
| RPE helped? | Yes (75% extension) | To be tested |

In Phase 2, the base model had no trouble with positions 576-1500, yet the LoRA cliff was at position ~595 (length 43). The bottleneck was the **LoRA adapter's inability to generalize beyond its training positions**. RPE fixed this by scattering positions across [0, L=1024) during LoRA training, so the adapter learned to function across a wider range of positions instead of overfitting to 0-576.

The same logic applies to MRCR: RPE with L=16384 would expose the LoRA adapter to positions across [0, 16384) during training, even though the training examples themselves are only 4K-8K tokens. At test time on 8K-16K inputs, the adapter has already seen those position ranges.

### What's genuinely new/different about MRCR vs CCoT

The parallel holds for the position generalization mechanism, but MRCR introduces real questions:

1. **Task type**: String reversal is algorithmic (apply a rule at each position). MRCR is retrieval (find and reproduce a needle in a haystack). Does RPE's position robustness help with content retrieval across positions, not just algorithmic computation at positions?

2. **Sequence length scale**: CCoT sequences were ~576 tokens. MRCR sequences are 4K-8K — an order of magnitude longer. RPE with L=16384 on 8K-token sequences gives avg position gap of ~2, which is mild. The disruption-to-benefit ratio may differ.

3. **Comparison with YaRN**: In Phase 2 we only compared RPE vs no-RPE. Here we compare against YaRN, which is purpose-built for context extension via RoPE frequency rescaling. YaRN operates differently (modifies `inv_freq`, not `position_ids`), so this is a real apples-to-oranges comparison of two distinct approaches to the same LoRA OOD problem.

These are open empirical questions — which is exactly why the experiment is worth running.

### Predictions

| Scenario | Likelihood | Reasoning |
|----------|-----------|-----------|
| RPE helps on 8K-16K OOD (vs LoRA baseline without RPE) | **Medium-High** | Same mechanism that worked in Phase 2; LoRA position generalization is the bottleneck |
| RPE slightly hurts 4K-8K ID performance | Medium | Same trade-off as Phase 2 (100% → 96.7%); RPE noise costs a bit in-distribution |
| YaRN is strong baseline at 8K-16K without any training | **High** | YaRN is designed for this; it rescales RoPE frequencies without needing LoRA at all |
| RPE + YaRN combined outperforms either alone | Medium | Orthogonal mechanisms — YaRN fixes base model frequencies, RPE helps LoRA generalize |
| PoSE outperforms RPE on this task | Medium | PoSE preserves local coherence while extending; may suit retrieval better |
| Curriculum RPE outperforms fixed-L RPE | **Medium-High** | Consistent with Phase 2 finding |

### Framing for the paper

The contribution is testing whether RPE's LoRA position generalization benefit — proven on synthetic algorithmic tasks — transfers to real NLP tasks (long-context retrieval). Comparing against YaRN and optionally PoSE establishes RPE within the context extension literature. Whether RPE wins or not, the controlled comparison across position manipulation methods (RPE, PoSE, YaRN) on the same model/data/eval is a clear contribution.

---

## Context Summary (for future reference)

### What happened before
We completed a full research cycle applying **Randomized Positional Encodings (RPE)** to **LoRA fine-tuning** of Qwen2.5-7B on the **binary string reversal** task from the Composable Chain-of-Thought (CCoT) framework. Key results:
- **Baseline** (no RPE): Generalizes to length 43 (training max = 40), then collapses to 0%
- **RPE + Curriculum** (best): Generalizes to length 73 — a **75% extension** of operational range
- **Ranking**: Curriculum (45.5%) > Asymmetric (39.2%) > RPE rank16 (33.9%) > Baseline (13.6%) on OOD lengths 41-100
- Full evaluation: 660 examples on NYU Torch HPC with L40S GPUs
- RPE patches `position_ids` with sorted random integers from [0, L); curriculum gradually increases L across epochs
- Core finding: RPE works for length generalization in LoRA fine-tuning, but operates differently from CCoT's random prefix (RPE randomizes relative distances; CCoT preserves them)

### What we are doing now and why
We are **pivoting from synthetic tasks to real-world context extension**, specifically:
1. **Domain shift**: From binary string reversal (synthetic, algorithmic) → MRCR (natural language, long-context retrieval+reasoning)
2. **Model**: Qwen2.5-7B-Instruct (same model family as Phase 2; switch to 3B only if compute is too expensive)
3. **Baseline shift**: From no-RPE LoRA → YaRN (established context extension method)
4. **Motivation**: The DeepMind RPE paper only tested synthetic tasks trained from scratch. We want to show RPE generalizes to real NLP tasks with pretrained models, and compare against state-of-the-art context extension methods.

The MRCR dataset (Multi-Round Coreference Resolution) is a harder successor to needle-in-a-haystack: the model must identify the correct instance of a repeated entity/format pair buried in long multi-turn conversations. It has 2,400 samples binned by token count from 4K to 1M tokens.

---

## Experiment Design: Why YaRN+LoRA Training is Required

### The problem with inference-only YaRN as a baseline

Our original plan compared RPE+LoRA (trained on bin 0) against YaRN applied only at inference time (no training). This is **not a valid comparison** for the following reason:

**The research question is:** "Which position encoding strategy helps LoRA adapters generalize to longer contexts?"

To answer this, both methods must be applied during training under identical conditions. Comparing a trained method (RPE+LoRA) against an untrained method (inference-only YaRN) conflates two variables: (1) the position encoding strategy and (2) whether fine-tuning occurred at all. Any difference in results could be attributed to the training itself rather than the position strategy.

### How YaRN training works

YaRN modifies the RoPE frequency basis (`inv_freq`) at the model config level. When applied during LoRA training:
1. Load base model with YaRN config → modified `inv_freq` is baked into the rotary embedding
2. Attach LoRA adapter on top of the YaRN-modified model
3. Train LoRA on bin 0 data — the adapter learns to work with the YaRN frequency basis
4. At eval, load base model with same YaRN config + merge trained LoRA weights

The LoRA weights are optimized for the modified frequency space, so they should generalize better to longer contexts where YaRN's scaling matters most.

This parallels RPE exactly: RPE modifies `position_ids` (not `inv_freq`) during training, so the LoRA learns to work with varied positions. Both are position encoding strategies applied during training — the only difference is *what* they modify.

### The corrected experiment matrix

| # | Condition | Train on bin 0? | Position strategy at train | Position strategy at eval | Purpose |
|---|-----------|-----------------|---------------------------|--------------------------|---------|
| 1 | Vanilla | No | — | Normal RoPE | Base model ceiling on this task |
| 2 | YaRN inference-only | No | — | YaRN RoPE | Free context extension (no training cost) |
| 3 | LoRA baseline | Yes (LoRA) | Normal RoPE | Normal RoPE | Effect of fine-tuning alone |
| 4 | **YaRN+LoRA** | Yes (LoRA) | YaRN RoPE | YaRN RoPE | **YaRN as training strategy** |
| 5 | **RPE+LoRA** | Yes (LoRA) | Randomized position_ids | Normal RoPE | **RPE as training strategy** |

**The key comparison is #4 vs #5** — same data, same LoRA rank, same compute, only the position encoding strategy differs. This is the apples-to-apples test.

Conditions #1-3 are supporting baselines:
- #1 tells us the base model's ability on this task at these lengths
- #2 tells us how much you can get for free (no training) with frequency scaling
- #3 isolates the effect of LoRA fine-tuning itself (no position tricks)

### Why this matters for the paper

Without YaRN+LoRA (#4), we could only claim "RPE+LoRA beats inference-only YaRN" — a weak claim since one method is trained and the other isn't. With #4, we can make the stronger claim: "RPE as a position encoding strategy during LoRA training outperforms/matches/underperforms YaRN as a position encoding strategy during LoRA training, under identical conditions." This is the comparison the paper needs.

### Implementation note

YaRN+LoRA training requires only a config change when loading the model in LLaMA-Factory's `tuner.py` — apply the version-aware YaRN config (see bug fix above) before LoRA is attached. No new code is needed beyond what we already have for the YaRN eval fix.

---

## Master Checklist

### Phase 1: Environment & Data Setup
- [x] **1.1** Set up Qwen2.5-7B-Instruct on HPC (download model, verify inference works)
- [x] **1.2** Download and explore MRCR dataset (`openai/mrcr` from HuggingFace)
- [x] **1.3** Implement MRCR data loading, binning (by model tokenizer), and train/test splitting
- [x] **1.4** Implement MRCR evaluation script (grading function, per-bin aggregation)

### Phase 2: No-Training Baselines
- [x] **2.1** Evaluate vanilla Qwen2.5-7B-Instruct on 4K-8K bin → **0.3887**
- [x] **2.2** Evaluate vanilla Qwen2.5-7B-Instruct on 8K-16K bin → **0.3649**
- [ ] **2.3** Evaluate YaRN (inference-only) on 4K-8K bin *(re-running with bug fix)*
- [ ] **2.4** Evaluate YaRN (inference-only) on 8K-16K bin *(re-running with bug fix)*
- [ ] **2.5** If baseline performance is high on 4K-16K, extend evaluation to 16K-32K bin

### Phase 3: LoRA Training & Evaluation [CORE EXPERIMENTS]
All conditions train LoRA on bin 0 (4K-8K) train split, eval on bin 0 test + bin 1 + bin 2.

- [ ] **3.1** Prepare MRCR 4K-8K train split for LLaMA-Factory LoRA fine-tuning
- [ ] **3.2** Train **LoRA baseline** (normal RoPE, no position tricks)
- [ ] **3.3** Evaluate LoRA baseline on bins 0/1/2
- [ ] **3.4** Train **YaRN+LoRA** (YaRN config applied at model load before LoRA)
- [ ] **3.5** Evaluate YaRN+LoRA on bins 0/1/2 (with YaRN at eval too)
- [ ] **3.6** Adapt RPE codebase for Qwen2.5-7B-Instruct
- [ ] **3.7** Train **RPE+LoRA** on 4K-8K train split
- [ ] **3.8** Evaluate RPE+LoRA on bins 0/1/2

### Phase 4: PoSE Exploration [SIDE TASK]
- [ ] **4.1** Implement PoSE position manipulation (chunk + skip bias)
- [ ] **4.2** Compare PoSE vs RPE on same training data and eval bins

### Phase 5: Analysis & Reporting
- [ ] **5.1** Compile results table (method x bin x needle_count)
- [ ] **5.2** Write up findings and next steps

---

## Recommended Task Order

```
Week 1: Setup + Baselines (tasks 1.1 → 1.4 → 2.1 → 2.2 → 2.3 → 2.4)
Week 2: RPE Training + Eval (tasks 3.1 → 3.2 → 3.3 → 3.4 → 3.5)
Week 3: Extended Eval + PoSE + Analysis (tasks 2.5 → 3.6 → 4.1 → 4.2 → 5.1 → 5.2)
```

Start with baselines because:
1. They require **zero training** — just inference — so you get results fast
2. Baseline numbers determine whether RPE training is even worth pursuing on this task
3. If vanilla Qwen2.5-7B already scores high on 4K-8K, you skip to harder bins immediately
4. YaRN is a one-line config change — fastest possible experiment

---

## Detailed Task Plans

---

### Task 1.1: Set Up Qwen2.5-7B-Instruct on HPC

**Subtasks:**
1. Download model to HPC scratch: `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /scratch/$USER/models/Qwen2.5-7B-Instruct`
2. Write a minimal inference test script:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto")
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
   messages = [{"role": "user", "content": "What is 2+2?"}]
   text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   inputs = tokenizer([text], return_tensors="pt").to(model.device)
   outputs = model.generate(**inputs, max_new_tokens=64)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
   ```
3. Submit as SLURM job, verify it completes successfully
4. Check GPU memory usage — Qwen2.5-7B in fp16/bf16 needs ~14GB for weights, plus KV cache for long contexts

**Potential issues:**
- Qwen2.5-7B in fp16/bf16 needs ~14GB for weights alone. KV cache for 16K+ tokens adds ~4-8GB. Total ~20-25GB — fits on L40S (48GB) or A100 (80GB)
- If memory is tight, use `torch_dtype=torch.bfloat16` and `attn_implementation="flash_attention_2"`
- **Fallback**: If 7B is too slow or OOMs on longer bins, switch to Qwen2.5-3B-Instruct

---

### Task 1.2: Download and Explore MRCR Dataset

**Subtasks:**
1. Download dataset:
   ```python
   from datasets import load_dataset
   ds = load_dataset("openai/mrcr")
   ```
2. Explore structure — fields: `prompt` (JSON list of messages), `answer`, `random_string_to_prepend`, `n_needles`, `n_chars`, `total_messages`
3. Compute token counts using **Qwen's tokenizer** (not `o200k_base` — we bin by our model's tokenizer)
4. Verify bin distribution: 100 samples per bin per needle count
5. Note: filter by `date_added` to use corrected data (post 12/5/2025 fix)

**Key detail:** The official bins use OpenAI's `o200k_base` tokenizer. For our purposes, re-bin using Qwen2.5's tokenizer since token counts will differ. Report both for comparability.

---

### Task 1.3: MRCR Data Loading, Binning, and Train/Test Splitting

**Subtasks:**
1. Write a data utility script (`mrcr_utils.py`) that:
   - Loads the dataset
   - Tokenizes each sample's prompt with Qwen tokenizer to get token counts
   - Assigns each sample to a bin based on token count
   - Creates train/test splits per bin (e.g., 70/30 or 80/20)
2. For the initial experiments, focus on:
   - **2-needle** subset only (simplest, 800 samples total)
   - **Bins 0 and 1** (4K-8K and 8K-16K)
3. Save splits to disk as JSON/parquet for reproducibility

**Design decision:** Start with 2-needle only. If RPE shows promise, extend to 4-needle and 8-needle.

---

### Task 1.4: Implement MRCR Evaluation Script

**Subtasks:**
1. Write `eval_mrcr.py` with the official grading function:
   ```python
   from difflib import SequenceMatcher

   def grade(response, answer, random_string_to_prepend):
       if not response.startswith(random_string_to_prepend):
           return 0.0
       response = response.removeprefix(random_string_to_prepend)
       answer = answer.removeprefix(random_string_to_prepend)
       return float(SequenceMatcher(None, response, answer).ratio())
   ```
2. Implement inference loop:
   - Load model + tokenizer
   - For each sample: parse `json.loads(row["prompt"])` into message list
   - Apply chat template, generate response (greedy decoding, temperature=0)
   - Grade and log per-sample scores
3. Aggregate: mean score per bin, per needle count
4. Output: CSV + console summary table
5. Add SLURM submission script with appropriate time/memory limits

**Important:** Set `max_new_tokens` appropriately. MRCR answers can be long (full paragraphs). Use `max_new_tokens=2048` to be safe.

---

### Task 2.1-2.2: Vanilla Qwen2.5-7B-Instruct Baseline

**Subtasks:**
1. Run `eval_mrcr.py` with vanilla model on 4K-8K bin (2-needle subset)
2. Run on 8K-16K bin
3. Record: mean score, score distribution, any failure patterns
4. Note: Qwen2.5-7B has `max_position_embeddings=32768`, so 4K-16K positions are within the base model's native range. However, these baselines (without LoRA fine-tuning) measure the base model's retrieval ability — they tell us whether the *task* is hard at these lengths, independent of any LoRA generalization question.

**Expected outcome:** The model should perform reasonably on 4K-8K, possibly degrading on 8K-16K due to task difficulty (more distractors, harder retrieval). If it's already near-perfect on both, we need harder bins (16K-32K) to see degradation. These numbers establish the ceiling for what RPE+LoRA should aim to match or exceed.

---

### Task 2.3-2.4: YaRN Inference-Only Baseline

**Purpose:** Measure how much context extension you get for free (no training) with YaRN frequency scaling. This is a supporting baseline, NOT the main comparison against RPE.

**Subtasks:**
1. Run `eval_mrcr.py` with `--enable-yarn --yarn-factor 4.0` on bins 0 and 1
2. Compare against vanilla — any improvement shows YaRN's inference-time benefit

**Note:** YaRN with factor=4.0 extends context to ~131K tokens. See bug fix section above for correct config code (version-aware for transformers v4/v5).

**Caveat:** Static YaRN may slightly degrade short-context (4K-8K) performance. Watch for this.

---

### Task 3.1: Prepare MRCR Training Data for LLaMA-Factory

**Subtasks:**
1. Convert MRCR bin 0 train split to LLaMA-Factory ShareGPT format
2. Create training config YAML (LoRA rank 16, lr 2e-4, 3-5 epochs)
3. This is shared across all three training conditions (LoRA baseline, YaRN+LoRA, RPE+LoRA)

---

### Task 3.4-3.5: YaRN+LoRA Training & Evaluation

**Purpose:** This is the apples-to-apples comparison against RPE+LoRA. Both methods train LoRA on the same data with the same hyperparameters — only the position encoding strategy differs.

**How it works:**
1. Load base model with YaRN config applied (version-aware, see bug fix section)
2. Attach LoRA adapter on top — the modified `inv_freq` is already baked into the rotary embedding
3. Train LoRA on bin 0 data — the adapter learns in YaRN's modified frequency space
4. At eval: load base model with YaRN config + merge LoRA checkpoint

**Implementation:** In LLaMA-Factory's `tuner.py`, apply the YaRN config when loading the base model (before LoRA attachment). This is a config-level change — no new training code needed.

**Key detail:** YaRN modifies `inv_freq` (the frequency basis of RoPE). RPE modifies `position_ids` (the input to RoPE). These are orthogonal approaches to the same problem: helping the model handle positions beyond its training range. By training both with LoRA under identical conditions, we isolate which manipulation strategy the LoRA adapter benefits from more.

---

### Task 3.6: Adapt RPE Codebase for Qwen2.5-7B-Instruct

**Subtasks:**
1. Verify RPE patching works on Qwen2.5-7B (our `RPEPatcher` should be model-agnostic since it patches `model.forward()` to replace `position_ids`)
2. Choose appropriate L value:
   - For 4K-8K bin training: max tokenized length ~8K, so L = 16K (2x margin, same heuristic as before)
   - Alternatively, start with L = 8192 since we're training on shorter sequences
3. Update RPE config for new model and task
4. Test: load model + apply RPE patch + run one forward pass + verify positions are randomized

**Key difference from reverse_string:** MRCR inputs are much longer (4K-8K tokens vs ~576 tokens). RPE's L must be scaled accordingly.

---

### Task 3.2: Prepare MRCR Training Data

**Subtasks:**
1. Convert MRCR train split into LLaMA-Factory format:
   - The existing CCoT setup uses ShareGPT format (list of conversation turns)
   - MRCR's `prompt` field is already a list of `{"role": ..., "content": ...}` messages — close to what we need
   - Add the expected answer as the final assistant turn
2. Create training config YAML for LLaMA-Factory:
   - Model: Qwen2.5-7B-Instruct
   - LoRA rank: Start with 16 (same as best Phase 2 config)
   - Learning rate: 2e-4
   - Epochs: 3-5
   - RPE config: L = 16384, curriculum schedule TBD

**Design decision on training objective:** The model needs to learn to reproduce the correct needle verbatim, prepended with the random string. The training data format should have the full multi-turn conversation as input and the answer (with prepended string) as the target.

---

### Task 3.3: Train RPE + LoRA

**Subtasks:**
1. Submit training job to HPC
2. Monitor training loss — expect it to be higher than baseline (RPE disruption)
3. Watch eval loss for non-monotonic behavior (our Phase 2 finding)
4. Save best checkpoint based on eval loss

**Estimated resources:**
- Qwen2.5-7B with LoRA: ~14GB model + ~1-2GB LoRA weights
- 4K-8K token sequences: KV cache ~4-8GB
- Total: ~25-30GB — fits on single L40S (48GB) or A100 (80GB)
- Training time: depends on dataset size and epochs; monitor first epoch closely
- **If too slow/expensive**: Switch to Qwen2.5-3B-Instruct (~6GB model, ~12-15GB total)

---

### Task 3.4-3.6: RPE Evaluation

**Subtasks:**
1. Load base model + merge LoRA adapter (same as Phase 2: `model.merge_and_unload()`)
2. Run `eval_mrcr.py` on test splits
3. Compare against vanilla and YaRN baselines

**Key question:** Does RPE improve OOD generalization (8K-16K) when trained only on 4K-8K? This is the same experimental structure as Phase 2 (train on 1-40, test on 41-100).

---

### Task 4.1-4.2: PoSE Exploration (Side Task)

**What is PoSE?** Positional Skip-wisE training. Instead of fully random positions (RPE), PoSE divides the context window into N chunks and applies random skip biases between chunks while keeping positions contiguous within each chunk.

**How it differs from RPE:**
| Aspect | RPE | PoSE |
|--------|-----|------|
| Position pattern | Fully random, sorted | Contiguous within chunks, gaps between |
| Relative distances | Randomized (avg gap = L/seq_len) | Preserved within chunks (=1), large between chunks |
| Local structure | Destroyed | Preserved |
| Implementation | Sample + sort random positions | Chunk + sample skip biases |

**Why PoSE might be better for context extension:**
- Preserves local token-to-token relationships (relative distance = 1 within chunks)
- Only disrupts long-range relationships (gaps between chunks)
- For MRCR, local coherence matters (reading text), while long-range retrieval needs extension
- PoSE was specifically designed for context extension; RPE was designed for length generalization generally (tested on synthetic tasks in the original DeepMind paper, but the mechanism is not task-specific)

**Implementation plan:**
1. Create `pose/core.py` mirroring `rpe/core.py`:
   ```python
   class PositionalSkipWiseEncoding:
       def get_pose_positions(seq_length, target_length, n_chunks=2):
           chunk_size = seq_length // n_chunks
           biases = sorted(random.sample(range(target_length - chunk_size), n_chunks))
           positions = []
           for i, bias in enumerate(biases):
               start = i * chunk_size
               end = start + chunk_size if i < n_chunks - 1 else seq_length
               positions.extend(range(bias + start - i*chunk_size, bias + end - i*chunk_size))
           return torch.tensor(positions)
   ```
2. Create `pose/patching.py` reusing the same monkey-patch structure as RPE
3. Train on same data and compare head-to-head

---

## Additional Suggestions

### 1. RPE + YaRN Combined Experiment
YaRN modifies the RoPE frequency basis; RPE modifies position IDs. These are orthogonal — you could apply both simultaneously. The hypothesis: YaRN smooths out the frequency response for long positions while RPE trains the model to be robust to position variation. This could be strictly better than either alone.

**Risk:** YaRN's attention temperature scaling was calibrated for sequential positions. Random positions might interact poorly.

### 2. Needle Count as Difficulty Axis
MRCR has 2-needle, 4-needle, and 8-needle variants. After establishing baselines on 2-needle, evaluate on 4/8-needle to see if RPE helps with the retrieval precision aspect (distinguishing between more confounders).

### 3. Per-Position Analysis (Where is the needle?)
MRCR buries needles at different positions in the conversation. Analyze whether RPE helps more when the needle is near the beginning, middle, or end. This would give insight into whether RPE is helping with position-invariant retrieval specifically.

### 4. Dynamic YaRN vs Static YaRN
Static YaRN applies the same scaling factor regardless of actual input length (degrades short-context). Dynamic YaRN adjusts the factor based on the actual sequence length. If HuggingFace transformers supports dynamic YaRN for Qwen, test it — it might be a stronger baseline.

### 5. Curriculum RPE for Context Extension
Our Phase 2 curriculum (gradually increasing L) was the best approach. For MRCR, consider a curriculum that starts with shorter-context examples and gradually introduces longer ones, with L scaled accordingly.

### 6. Hybrid PoSE-RPE
Take the best of both: PoSE's chunk structure (preserve local contiguity) + RPE's randomness (within-chunk position jitter). Each chunk preserves relative distance = 1 internally, but chunk boundaries have random gaps AND positions within chunks have small random perturbations.

### 7. Ablation: Number of Needles in Training
If training RPE on MRCR, consider whether training on harder variants (8-needle) transfers to easier ones (2-needle), or vice versa. This helps understand if RPE's benefit is in retrieval vs. discrimination.

---

## Quick Reference: Key Technical Details

### MRCR Grading Function
```python
from difflib import SequenceMatcher

def grade(response, answer, random_string_to_prepend):
    if not response.startswith(random_string_to_prepend):
        return 0.0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())
```

### YaRN Config for Qwen2.5-7B-Instruct

**BUG FIX (Feb 2025):** The original `config.rope_scaling = {...}` silently broke in transformers v5 (overwrites `rope_parameters`, loses `rope_theta=1M`, uses wrong key `"type"` vs `"rope_type"`). On v4 it was silently ignored — first eval produced scores identical to vanilla. Fix: version-aware config in `eval_mrcr.py` (v5 sets `config.rope_parameters` directly, v4 sets `config.rope_scaling`). Verified with `verify_yarn.py`: 40/64 inv_freq dimensions changed, rope_type changed from `"default"` → `"yarn"`.

```python
# v5+:
config.rope_parameters = {
    "rope_type": "yarn",
    "rope_theta": 1000000.0,  # MUST preserve Qwen's theta
    "factor": 4.0,
    "original_max_position_embeddings": config.max_position_embeddings,
}
# v4:
config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": config.max_position_embeddings,
}
```

### MRCR Dataset Loading
```python
from datasets import load_dataset
ds = load_dataset("openai/mrcr")
# Fields: prompt (JSON string), answer, random_string_to_prepend, n_needles, n_chars
# Bins by token count: [4K,8K], (8K,16K], (16K,32K], ..., (512K,1M]
```

### RPE Config for 4K-8K Training
```yaml
max_simulation_length: 16384  # 2x the max bin boundary
enabled: true
# Curriculum: 8192 → 10240 → 12288 → 14336 → 16384
```

### PoSE Position Generation (Pseudocode)
```python
def pose_positions(seq_len, target_len, n_chunks=2):
    chunk_len = seq_len // n_chunks
    biases = sorted(sample(range(target_len - seq_len), n_chunks))
    positions = []
    for i, bias in enumerate(biases):
        local_start = i * chunk_len
        local_end = min(local_start + chunk_len, seq_len)
        positions += [bias + j for j in range(local_end - local_start)]
    return positions
```

---

## Summary: What Happened Before → What We're Doing Now → Why

**Before:** We proved that RPE + LoRA fine-tuning achieves 75% length generalization improvement on binary string reversal (Qwen2.5-7B). Curriculum RPE was the best variant. This was a synthetic task where the model learns an algorithmic pattern.

**Now:** We're testing whether RPE's LoRA position generalization benefit transfers from synthetic algorithmic tasks to real NLP — long-context retrieval and reasoning via the MRCR dataset. We stay with Qwen2.5-7B-Instruct (same model family as Phase 2; fallback to 3B if compute is too expensive) and use YaRN as the established baseline. The experimental structure directly mirrors Phase 2: train LoRA on shorter contexts (4K-8K), test on longer ones (8K-16K, potentially 16K-32K) where the LoRA must generalize to positions it hasn't been fine-tuned on.

**Why:** The DeepMind RPE paper only showed results on synthetic tasks trained from scratch. Demonstrating RPE works on (1) pretrained models, (2) with LoRA, (3) on natural language tasks, and (4) competitively against YaRN would be a significant contribution. If RPE + YaRN combined outperforms either alone, that's even stronger.

**Side exploration:** PoSE (Positional Skip-wisE training) is a related method that preserves local contiguity while introducing position gaps. It may be better suited than RPE for context extension specifically (as opposed to synthetic length generalization). Implementing and comparing it would strengthen the paper.
