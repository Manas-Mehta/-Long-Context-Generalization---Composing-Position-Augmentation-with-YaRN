# RPE on Reverse String: Decoder-Only Reproduction of DeepMind's Experiment

**Institution**: TAUR Labs, NYU
**Paper**: Ruoss et al., "Randomized Positional Encodings Boost Length Generalization of Transformers", ACL 2023 ([arXiv:2305.16843](https://arxiv.org/abs/2305.16843))
**DeepMind Code**: [github.com/google-deepmind/randomized_positional_encodings](https://github.com/google-deepmind/randomized_positional_encodings)

---

## 1. Goal

Reproduce DeepMind's RPE experiment on the **reverse string task**, adapted from their **encoder-only** transformer (JAX/Haiku) to a **decoder-only** model (Qwen2, PyTorch/HuggingFace). The objective is to test whether RPE provides the same length generalization benefits on decoder-only architectures.

### Research Question

> Does Randomized Positional Encoding (RPE) improve length generalization for decoder-only transformers on the reverse string task, as it does for encoder-only transformers in the original paper?

---

## 2. Background: RPE Algorithm

### The Problem

Transformers use positional encodings (e.g., RoPE) to encode token positions. During training on sequences of length N, the model only sees positions [0, N-1]. At inference on longer sequences, positions beyond N are out-of-distribution, causing failure.

### The Solution (RPE)

During training, instead of standard position IDs `[0, 1, 2, ..., N-1]`:
1. Sample N unique integers from a much larger range `[0, L)` where `L >> N`
2. Sort them in ascending order (preserves causal order)
3. Use these as position IDs

```
Standard:  [0, 1, 2, 3, 4]
RPE:       [127, 456, 892, 3421, 7103]  (sorted random from [0, 2048))
```

During inference, use standard sequential positions (no randomization).

**Why it works**: The model learns to rely on relative ordering rather than absolute position values. Since positions from the full range [0, L) are seen during training, no position is truly out-of-distribution at inference time.

---

## 3. DeepMind's Original Setup

Source: `experiments/example.py`, `experiments/constants.py`, `models/transformer.py`, `tasks/dcf/reverse_string.py`

| Parameter | Value | Source File |
|-----------|-------|-------------|
| **Architecture** | Encoder-only transformer | `models/transformer.py` |
| **Layers** | 5 | `experiments/constants.py` |
| **Embedding dim** | 64 | `experiments/constants.py` |
| **Attention heads** | 8 | `experiments/constants.py` |
| **FFN size** | 64 x 4 = 256 | `widening_factor=4` |
| **Dropout** | 0.1 | `experiments/constants.py` |
| **Positional encoding** | NOISY_ROTARY (randomized RoPE) | `experiments/example.py` |
| **noise_max_length (L)** | 2048 | `experiments/example.py` |
| **Task** | Reverse string, vocab_size=2 (binary: "0","1") | `tasks/dcf/reverse_string.py` |
| **Training lengths** | Uniform [1, 40] | `UniformCurriculum(values=range(1, 41))` |
| **Batch size** | 128 | `experiments/example.py` |
| **Learning rate** | 1e-3 | `experiments/example.py` |
| **Optimizer** | Adam (no weight decay) | `experiments/training.py` |
| **Gradient clipping** | 1.0 (global norm) | `experiments/training.py` |
| **Training steps** | 10,000 | `experiments/example.py` |
| **LR schedule** | Constant | `experiments/example.py` |
| **Eval lengths** | [1, 100] | `max_range_test_length=100` |
| **Eval samples/length** | 512 | `range_test_total_batch_size=512` |
| **Eval metric** | Per-token accuracy | `accuracy_fn` |
| **Loss** | Cross-entropy on output positions only | `experiments/training.py` |
| **Data generation** | Fresh random data each step (infinite) | `experiments/training.py` |
| **Framework** | JAX / Haiku | - |

### DeepMind's Reverse String Task

- Input: binary string of length n, e.g., `"011010"`
- Output: reversed string, e.g., `"010110"`
- Alphabet: `{0, 1}` (vocab_size=2)
- The encoder processes both input and output as a single sequence
- Loss is computed only on the output positions

---

## 4. Our Decoder-Only Adaptation

### What We Changed (and Why)

| Aspect | DeepMind (Original) | Our Adaptation | Justification |
|--------|---------------------|----------------|---------------|
| **Architecture** | Encoder-only | Decoder-only (Qwen2) | Testing RPE on modern LLM architecture |
| **Attention** | Bidirectional (or optional causal) | Causal only (autoregressive) | Inherent to decoder-only |
| **Framework** | JAX/Haiku | PyTorch/HuggingFace | Ecosystem compatibility |
| **Tokenizer** | Integer IDs for binary alphabet | Character-level tokenizer | Equivalent; each char = 1 token |
| **Input format** | Raw input→output sequences | `"reverse: 01101\n10110"` | Prompt prefix masked in loss |
| **EOS token** | None (encoder) | Added at end of output | Required for generation stopping |
| **Evaluation** | Forward pass (all positions predicted simultaneously) | Autoregressive generation (greedy) | Required for decoder-only |
| **RPE injection** | Inside rotary encoding function | Intercept `position_ids` before RoPE | Equivalent: same randomized positions reach RoPE |

### What We Kept Identical

| Parameter | DeepMind | Ours | Match? |
|-----------|----------|------|--------|
| Layers | 5 | 5 | Yes |
| Embedding dim | 64 | 64 | Yes |
| Attention heads | 8 | 8 | Yes |
| FFN multiplier | 4x | 4x (intermediate_size=256) | Yes |
| Dropout | 0.1 | 0.1 (attention_dropout) | Yes |
| Positional encoding | Randomized RoPE | Qwen2 RoPE + randomized position_ids | Yes |
| L (noise_max_length) | 2048 | 2048 | Yes |
| Task | Binary string reversal | Binary string reversal | Yes |
| Training lengths | [1, 40] | [1, 40] | Yes |
| Batch size | 128 | 128 | Yes |
| LR | 1e-3 | 1e-3 | Yes |
| Optimizer | Adam | AdamW (weight_decay=0 = Adam) | Yes |
| Grad clip | 1.0 | 1.0 | Yes |
| Training steps | 10,000 | 10,000 | Yes |
| LR schedule | Constant | Constant | Yes |
| Eval lengths | [1, 100] | [1, 100] | Yes |
| Eval metric | Per-token accuracy | Per-token accuracy | Yes |
| Weight decay | 0 | 0 | Yes |
| Warmup | None | None | Yes |
| Seed | 0 | 0 | Yes |
| Training from scratch | Yes (random init) | Yes (random init) | Yes |

### Key Architectural Difference: Causal Masking

The most significant difference is causal masking. In DeepMind's encoder-only model, each output token can attend to ALL positions (including future output tokens). In our decoder-only model, each output token can only attend to PREVIOUS tokens.

**Impact**: The reverse string task is harder for decoder-only models because:
- At output position i, the model cannot "peek" at output positions i+1, i+2, ...
- It must predict each reversed character using only the input and previously generated output
- The encoder can use bidirectional context to inform all predictions simultaneously

This is an expected and documented difference. If RPE still improves length generalization under this harder setting, it strengthens the evidence for RPE's general applicability.

---

## 5. Implementation Details

### 5.1 RPE Core (`rpe/core.py`)

```python
class RandomizedPositionalEncoding:
    def __init__(self, max_simulation_length=2048):
        self.max_simulation_length = max_simulation_length

    def get_randomized_positions(self, seq_length, device=None):
        # Sample seq_length unique positions from [0, L)
        perm = torch.randperm(self.max_simulation_length)
        positions = perm[:seq_length].sort().values
        return positions
```

This exactly matches DeepMind's algorithm:
- DeepMind: `jrandom.choice(key, jnp.arange(noise_max_length), shape=(seq_len,), replace=False)` then `jnp.sort()`
- Ours: `torch.randperm(L)[:N].sort().values`

Both sample N unique integers from [0, L) without replacement, then sort ascending.

### 5.2 Model Patching (`rpe/patching.py`)

The `RPEPatcher` wraps `model.forward()` to intercept `position_ids`:
- When `model.training=True`: replaces position_ids with randomized ones (per batch element)
- When `model.training=False`: passes through standard sequential positions

This is equivalent to DeepMind's approach where `noisy=True` is passed to the rotary encoding function during training.

### 5.3 Dataset (`rpe/tasks/reverse_string_dataset.py`)

Format: `"reverse: 01101\n10110<eos>"`

- Lengths sampled uniformly from [min_length, max_length] (matches DeepMind's `UniformCurriculum`)
- Binary alphabet: "0" and "1" (matches DeepMind's `vocab_size=2`)
- Labels: prompt tokens masked with -100, loss only on output tokens (matches DeepMind)

### 5.4 Character Tokenizer (`scripts/train_reverse_string.py`)

Custom character-level tokenizer with ~12 tokens:
- `<pad>=0, <eos>=1, \n=2, space=3, :=4, 0=5, 1=6, e=7, r=8, s=9, v=10`

This replaces Qwen's 151,936-token BPE tokenizer, matching DeepMind's small-vocabulary setup. Each character maps to exactly one token ID, so there are no subword tokenization artifacts.

### 5.5 Model Configuration

```python
Qwen2Config(
    vocab_size=~12,              # Character-level (not BPE)
    hidden_size=64,              # Matches paper
    num_hidden_layers=5,         # Matches paper
    num_attention_heads=8,       # Matches paper
    num_key_value_heads=8,       # No GQA (matches paper's standard MHA)
    intermediate_size=256,       # 64 * 4 = 256 (matches paper's widening_factor=4)
    max_position_embeddings=2048,# = RPE L parameter
    attention_dropout=0.1,       # Matches paper
    tie_word_embeddings=True,    # Weight tying for small vocab
    use_sliding_window=False,    # Standard full attention
)
```

Approximate parameter count: ~2M (DeepMind's encoder-only model is similar size).

### 5.6 Evaluation

Autoregressive generation with greedy decoding:
1. Prompt: `"reverse: {binary_string}\n"`
2. Generate `length + 5` new tokens (budget for EOS)
3. Compare generated characters to expected reversed string
4. Per-token accuracy = (correct characters) / (total characters)

DeepMind uses a forward pass where all output positions are predicted simultaneously. Our autoregressive approach is the standard evaluation method for decoder-only models.

---

## 6. Running the Experiment

### Prerequisites

```bash
pip install torch transformers accelerate datasets pytest
```

### Full Training (Matches DeepMind, ~2-5 min on GPU)

```bash
# RPE training
python scripts/train_reverse_string.py

# Baseline (no RPE)
python scripts/train_reverse_string.py --no-rpe --output-dir outputs/reverse_string_baseline
```

### Pipeline Test (Quick Validation)

```bash
python scripts/train_reverse_string.py --pipeline-test
```

### Custom Configuration

All DeepMind hyperparameters are exposed as CLI args:

```bash
python scripts/train_reverse_string.py \
    --max-train-length 40 \
    --rpe-max-sim-length 2048 \
    --hidden-size 64 \
    --num-layers 5 \
    --num-heads 8 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --training-steps 10000 \
    --max-eval-length 100 \
    --eval-samples-per-length 512 \
    --output-dir outputs/reverse_string_rpe
```

### Running Tests

```bash
pytest rpe/tests/test_rpe.py -v
```

---

## 7. Expected Results

### DeepMind's Paper Results (Encoder-Only)

From Figure 3 and Table 1 of the paper:
- **Baseline (standard RoPE)**: High in-distribution accuracy, rapid degradation beyond training length 40
- **RPE (randomized RoPE)**: Maintains ~80%+ accuracy on lengths up to 100
- **DeepMind score** (mean OOD accuracy, lengths 41-100): ~0.8+ for RPE vs ~0.0-0.2 for baseline

### Our Expected Results (Decoder-Only)

- **In-distribution** (lengths 1-40): Both RPE and baseline should achieve high accuracy
- **Out-of-distribution** (lengths 41-100): RPE should show improved generalization
- **Absolute OOD accuracy may be lower** than DeepMind's due to causal masking constraint
- The key metric is the **relative improvement** of RPE over baseline

---

## 8. File Structure

```
RPE/
├── rpe/                                    # Core RPE module
│   ├── __init__.py                         # Exports: RandomizedPositionalEncoding, RPEPatcher
│   ├── core.py                             # RPE algorithm (position sampling & sorting)
│   ├── patching.py                         # Model monkey-patching for HuggingFace models
│   ├── tasks/
│   │   ├── reverse_string.py               # Evaluation harness (pretrained model)
│   │   └── reverse_string_dataset.py       # HF Dataset for training
│   └── tests/
│       └── test_rpe.py                     # 19 pytest tests
│
├── scripts/
│   ├── train_reverse_string.py             # Main training script (from-scratch)
│   └── eval_reverse_string.py              # Baseline vs RPE comparison
│
├── randomized_positional_encodings/        # DeepMind's reference code (JAX)
│
├── outputs/                                # Training results (JSON)
│   ├── reverse_string_rpe/
│   └── reverse_string_baseline/
│
├── requirements.txt
└── RPE_REVERSE_STRING_EXPERIMENT.md        # This file
```

---

## 9. Detailed Comparison with DeepMind Code

### 9.1 Position Randomization

**DeepMind** (`models/positional_encodings.py`):
```python
# Sample without replacement from [0, noise_max_length)
indexes = jrandom.choice(key, jnp.arange(noise_max_length), shape=(seq_len,), replace=False)
indexes = jnp.sort(indexes)
```

**Ours** (`rpe/core.py`):
```python
perm = torch.randperm(self.max_simulation_length)
positions = perm[:seq_length].sort().values
```

**Equivalence**: Both sample `seq_length` unique integers from `[0, L)` and sort them. `randperm` + slice is equivalent to `choice(replace=False)`.

### 9.2 Reverse String Task

**DeepMind** (`tasks/dcf/reverse_string.py`):
```python
class ReverseString(DuplicateString):
    # Inherits input generation from DuplicateString
    # Output = jnp.flip(batch['input'], axis=1)
```

**Ours** (`rpe/tasks/reverse_string_dataset.py`):
```python
input_str = "".join(self.rng.choice("01") for _ in range(length))
output_str = input_str[::-1]  # Python string reversal = flip
```

**Equivalence**: Both reverse the input sequence. DeepMind uses `jnp.flip()`, we use Python string slicing `[::-1]`.

### 9.3 Loss Computation

**DeepMind**: Loss computed on output positions only (task-level masking in `experiments/training.py`).

**Ours**: Labels set to -100 for prompt tokens, HuggingFace CrossEntropyLoss ignores -100 by default.

**Equivalence**: Same effect - loss computed only on the reversed output tokens.

### 9.4 Curriculum

**DeepMind**: `UniformCurriculum(values=list(range(1, 41)))` - samples length uniformly from {1, 2, ..., 40}.

**Ours**: `self.rng.randint(self.min_length, self.max_length)` with min=1, max=40.

**Equivalence**: Both sample uniformly from [1, 40]. Python's `random.randint(a, b)` is inclusive on both ends, matching DeepMind's `range(1, 41)`.

### 9.5 Evaluation

**DeepMind** (`experiments/range_evaluation.py`):
- Tests lengths 1 through `max_test_length` (100)
- 512 samples per length (processed in sub-batches of 64)
- Reports per-token accuracy
- "Score" = mean accuracy on lengths > training length

**Ours** (`scripts/train_reverse_string.py`, `evaluate_length_generalization()`):
- Tests lengths 1 through `max_eval_length` (100)
- Default 64 samples per length (configurable to 512 via `--eval-samples-per-length`)
- Reports per-token accuracy
- "DeepMind score" = mean accuracy on lengths > max_train_length

**Note**: For final experiments, use `--eval-samples-per-length 512` to match DeepMind exactly.

---

## 10. Known Differences and Their Impact

### 10.1 Causal Masking (Major)

**Impact**: Makes the task harder for our decoder-only model. Each output token can only attend to previous context, not future output tokens. The encoder can use full bidirectional attention.

**Mitigation**: None needed - this is the experimental variable. We are explicitly testing whether RPE helps decoder-only models.

### 10.2 Autoregressive Generation at Eval (Moderate)

**Impact**: DeepMind evaluates via forward pass (all positions predicted at once). We use autoregressive generation where errors can compound (a wrong token affects subsequent predictions).

**Mitigation**: This is inherent to decoder-only models and matches standard evaluation practice.

### 10.3 Pre-generated vs Infinite Data (Minor)

**Impact**: We pre-generate 500K examples and loop, while DeepMind generates fresh data each step. With random binary strings of length 1-40, the space of possible examples is enormous (2^1 + 2^2 + ... + 2^40 ≈ 10^12), so 500K examples provide sufficient diversity.

**Mitigation**: Can increase the cap in `train_reverse_string.py` line 316: `num_train = min(args.training_steps * args.batch_size, 500_000)`. For 10K steps x 128 batch = 1.28M examples needed.

### 10.4 Prompt Format (Negligible)

**Impact**: We wrap sequences as `"reverse: 01101\n10110"`. DeepMind uses raw input→output sequences. Since the prompt is masked in loss computation, the model only needs to learn the reversal, not the prompt format.

**Mitigation**: None needed.

### 10.5 Qwen2 vs Custom Transformer (Minor)

**Impact**: Qwen2 has slight architectural differences from a vanilla transformer (SwiGLU activation, RMSNorm, etc.). These are standard modern improvements and shouldn't affect the RPE experiment.

**Mitigation**: None needed - we are testing RPE on a modern decoder-only architecture.

---

## 11. Reproducing This Experiment

### Step 1: Clone and Setup

```bash
git clone <this-repo>
cd RPE
pip install -r requirements.txt
```

### Step 2: Run RPE Training

```bash
python scripts/train_reverse_string.py \
    --output-dir outputs/reverse_string_rpe \
    --eval-samples-per-length 512
```

### Step 3: Run Baseline Training

```bash
python scripts/train_reverse_string.py \
    --no-rpe \
    --output-dir outputs/reverse_string_baseline \
    --eval-samples-per-length 512
```

### Step 4: Compare Results

Results are saved as JSON in the output directories. Key metrics:
- `in_dist_accuracy`: Mean accuracy on lengths 1-40
- `ood_accuracy`: Mean accuracy on lengths 41-100
- `dm_score`: Same as ood_accuracy (DeepMind's scoring metric)
- `per_length`: Accuracy for each individual length

### Step 5: Multiple Seeds (Recommended)

For statistical significance, run with multiple seeds:

```bash
for seed in 0 1 2 3 4; do
    python scripts/train_reverse_string.py \
        --seed $seed \
        --output-dir outputs/rpe_seed_${seed} \
        --eval-samples-per-length 512

    python scripts/train_reverse_string.py \
        --no-rpe \
        --seed $seed \
        --output-dir outputs/baseline_seed_${seed} \
        --eval-samples-per-length 512
done
```

---

## 12. References

1. Ruoss, A., DelTredici, G., Niculae, V., Catt, E., & Hutter, M. (2023). **Randomized Positional Encodings Boost Length Generalization of Transformers**. ACL 2023. [arXiv:2305.16843](https://arxiv.org/abs/2305.16843)

2. DeepMind RPE Code: [github.com/google-deepmind/randomized_positional_encodings](https://github.com/google-deepmind/randomized_positional_encodings)

3. Qwen2 Technical Report: [arXiv:2407.10671](https://arxiv.org/abs/2407.10671)
