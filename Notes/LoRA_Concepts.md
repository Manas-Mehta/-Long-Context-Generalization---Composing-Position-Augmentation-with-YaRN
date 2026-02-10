# LoRA Concepts: What Gets Updated and Why

*A beginner-friendly guide for our RPE + CCoT experiments.*

---

## 1. What Is LoRA?

**LoRA (Low-Rank Adaptation)** is a way to fine-tune a large model without changing most of its parameters.

**The core idea:** Instead of updating a weight matrix `W` (which might be 4096 x 4096 = 16.7M parameters), LoRA freezes `W` and adds two small matrices `A` and `B` beside it:

```
Original:   output = W * input
With LoRA:  output = W * input  +  (B * A) * input
                     ^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^
                     frozen          trainable (tiny)
```

Where:
- `W` is the original weight matrix (e.g., 4096 x 4096) -- **FROZEN, never changes**
- `A` is a small matrix (e.g., 4096 x 8) -- **trainable**
- `B` is a small matrix (e.g., 8 x 4096) -- **trainable**
- `rank = 8` means the middle dimension is 8

**Parameter savings:** Instead of training 4096 x 4096 = 16.7M params, you train (4096 x 8) + (8 x 4096) = 65K params. That's **250x fewer parameters** per layer.

---

## 2. The Transformer Block: Where LoRA Gets Applied

Every transformer block in Qwen2.5-7B has two main components:

### Self-Attention (4 linear layers)

```
                    ┌──────────────┐
     input ────────>│  q_proj       │────> Q (queries)
                    │  (4096->4096) │
                    └──────────────┘
                    ┌──────────────┐
     input ────────>│  k_proj       │────> K (keys)
                    │  (4096->512)  │
                    └──────────────┘
                    ┌──────────────┐
     input ────────>│  v_proj       │────> V (values)
                    │  (4096->512)  │
                    └──────────────┘

         Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V

                    ┌──────────────┐
     attn_out ─────>│  o_proj       │────> output
                    │  (4096->4096) │
                    └──────────────┘
```

**What these do in plain English:**
- **q_proj** (Query): "What am I looking for?" -- transforms the current token into a query vector
- **k_proj** (Key): "What do I contain?" -- transforms each token into a key vector
- **v_proj** (Value): "What information do I carry?" -- transforms each token into a value vector
- **o_proj** (Output): Combines the attention results back into the right shape

**Where does positional encoding (RoPE) enter?** RoPE is applied to Q and K **after** q_proj and k_proj. It rotates the query and key vectors based on position_ids. This is where RPE changes things -- by changing position_ids, we change how Q and K interact.

```
Normal:   q_proj(input) --> apply RoPE(position=0,1,2,3,...) --> Q
RPE:      q_proj(input) --> apply RoPE(position=37,892,1204,5621,...) --> Q
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                       random sorted positions during training
```

### MLP / Feed-Forward Network (3 linear layers)

```
                    ┌──────────────┐
     input ────────>│  gate_proj    │──┐
                    │  (4096->11008)│  │
                    └──────────────┘  │
                                      ├──> element-wise multiply ──> down_proj ──> output
                    ┌──────────────┐  │
     input ────────>│  up_proj      │──┘
                    │  (4096->11008)│
                    └──────────────┘

                    ┌──────────────┐
     gated_out ────>│  down_proj    │────> output
                    │  (11008->4096)│
                    └──────────────┘
```

**What these do in plain English:**
- **gate_proj**: Decides which information to let through (like a gate)
- **up_proj**: Projects to a higher dimension (4096 -> 11008) for richer computation
- **down_proj**: Projects back down (11008 -> 4096)

Together, these form a "SwiGLU" activation -- a fancy nonlinear transformation that processes each token independently.

### What's NOT a linear layer (and therefore NOT targeted by LoRA)

- **Embedding layer**: Converts token IDs to vectors. Frozen.
- **RoPE (Rotary Position Embedding)**: A deterministic function, not a layer with weights. It's a mathematical rotation applied to Q and K based on position. **No learnable parameters at all.**
- **LayerNorm / RMSNorm**: Normalization layers. Frozen during LoRA.
- **LM head**: The final output projection (vocab prediction). Usually frozen.

---

## 3. Our LoRA Configuration (and CCoT's)

### The YAML Config

```yaml
finetuning_type: lora
lora_rank: 8           # Size of the low-rank matrices (A is d x 8, B is 8 x d)
lora_alpha: 16         # Scaling factor (explained below)
lora_dropout: 0.2      # Dropout on LoRA weights during training
lora_target: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj
```

### What each hyperparameter means

**lora_rank: 8**
- The "bottleneck" dimension. Higher = more expressive but more parameters.
- rank 8 means each LoRA adapter is an 8-dimensional bottleneck.
- Our meeting notes suggest trying rank 16 or 32 if rank 8 isn't enough.

**lora_alpha: 16**
- A scaling factor. The LoRA update is scaled by `alpha / rank`.
- With alpha=16, rank=8: scaling = 16/8 = 2.0
- This means the LoRA contribution is multiplied by 2x.
- Think of it as "how much influence should the LoRA adapter have."
- Rule of thumb: alpha = 2 * rank is a common default.

**lora_dropout: 0.2**
- During training, randomly zeros out 20% of LoRA weights each step.
- Prevents overfitting. Like randomly covering 20% of the adapter's "notes" so it can't memorize.

**lora_target: all 7 linear layers**
- We put LoRA adapters on ALL major linear layers in each transformer block.
- This is the current best practice (see Section 6 below).

### Parameter count

Qwen2.5-7B has **28 transformer blocks**. Each block gets 7 LoRA adapters:

```
Per adapter:  (d_in x 8) + (8 x d_out) parameters
Per block:    7 adapters
Total:        28 blocks x 7 adapters = 196 LoRA adapters

Rough total trainable params: ~18M out of 7.6B = ~0.24%
```

**99.76% of the model is frozen.** Only the small A and B matrices in each adapter are updated.

---

## 4. What CCoT's Original Paper Uses

We verified against the [CCoT GitHub repo](https://github.com/fc2869/composable_cot):

| Parameter | CCoT Original | Our Config | Match? |
|-----------|---------------|------------|--------|
| **Model** | Qwen/Qwen2.5-7B | Qwen/Qwen2.5-7B | Yes |
| **lora_rank** | 8 | 8 | Yes |
| **lora_alpha** | 16 | 16 | Yes |
| **lora_dropout** | 0.2 | 0.2 | Yes |
| **lora_target** | q,k,v,o,up,down,gate | q,k,v,o,up,down,gate | Yes |
| **Epochs** | 5 | 5 | Yes |
| **Batch size** | 4 | 4 | Yes |
| **Scheduler** | linear | linear | Yes |
| **Precision** | bf16 | bf16 | Yes |

**Our configs exactly match CCoT's.** The LLaMA-Factory example configs separately use `lora_target: all` (a shorthand that auto-discovers all linear layers), but CCoT's actual experiment configs explicitly list the same 7 modules we use.

### Learning rates differ by task

| Task | Learning Rate |
|------|---------------|
| letter_concat + next_last_letter (composition) | 1e-3 |
| letter_concat + ascii_multiply (composition) | 1e-4 |
| next_last_letter + ascii_multiply (composition) | 5e-4 |
| skillmix (composition) | 5e-4 |
| Our reverse_string (baseline & RPE) | 1e-3 |

---

## 5. LoRA and RPE: How They Interact (Key Insight)

This is the conceptual piece that matters for meetings:

```
                  ┌─────────────────────────────────────────────┐
                  │            Transformer Block                 │
                  │                                              │
  token_ids ─────>│ Embedding ──> hidden_states                  │
                  │                    │                          │
                  │              ┌─────┴──────┐                  │
                  │              │  q_proj     │                  │
  position_ids ──>│              │  + LoRA_q   │──> Q ──┐        │
      │           │              └────────────┘         │        │
      │           │              ┌────────────┐         │        │
      │           │              │  k_proj     │         │        │
      │           │              │  + LoRA_k   │──> K ──┤        │
      │           │              └────────────┘         │        │
      │           │                                     │        │
      │           │              Apply RoPE(pos_ids)    │        │
      │           │              to Q and K             │        │
      │           │                    │                 │        │
      └───────────│────────────────────┘                 │        │
                  │                                     │        │
                  │              Attention(Q', K', V)    │        │
                  │                    │                          │
                  │              ┌────────────┐                  │
                  │              │  o_proj     │                  │
                  │              │  + LoRA_o   │                  │
                  │              └────────────┘                  │
                  │                    │                          │
                  │              ┌────────────┐                  │
                  │              │  MLP layers │                  │
                  │              │  + LoRA_mlp │                  │
                  │              └────────────┘                  │
                  │                    │                          │
                  │                 output                        │
                  └─────────────────────────────────────────────┘
```

**The key points:**

1. **RPE changes position_ids** (input to RoPE), not any weight matrix.
2. **LoRA changes weight matrices** (q_proj, k_proj, etc.), not position_ids.
3. **They operate on different things** -- RPE is about *where* tokens appear, LoRA is about *how* tokens are processed.
4. **RoPE itself has NO learnable parameters.** It's a fixed mathematical function: `rotate(vector, angle)` where angle = f(position). RPE changes the position input, but the rotation function itself is unchanged and unlearnable.

**The interaction problem:** During training with RPE, the LoRA adapters learn to work with random positions. At inference, positions are sequential (0,1,2,...). The LoRA adapters need to generalize from random positions to sequential positions. With only ~0.24% of parameters being trainable, the LoRA adapters may not have enough capacity to learn truly position-invariant representations -- they might just memorize "how to work with random positions" rather than learning "position doesn't matter."

---

## 6. Which Modules Should You Target? (The Research Landscape)

### Historical evolution

**Original LoRA paper (2021):** Only targeted attention layers (q_proj, v_proj). Argued this was sufficient and simpler.

**Current consensus (2025-2026):** Target ALL linear layers. Research has shown:
- Attention-only LoRA underperforms MLP-only LoRA
- MLP layers contain 2-3x more parameters than attention (in Qwen2.5-7B: 11008x4096 vs 4096x4096)
- Skipping MLP means missing the majority of gradient signal
- Memory savings from skipping MLP targets are negligible

### Why all 7 layers matters for us specifically

For RPE, targeting all layers is especially important because:

1. **q_proj and k_proj** are directly affected by RPE (RoPE is applied to their outputs). If we want LoRA to learn position-invariant patterns, these adapters are critical.

2. **v_proj and o_proj** process the attention values. Even though RoPE doesn't directly touch V, the attention weights (which depend on Q and K) determine how V is aggregated. Position changes propagate through attention.

3. **MLP layers** (gate, up, down) process each token independently. They don't directly see positions, but they receive attention outputs that are affected by positions. Having LoRA adapters here gives the model more capacity to compensate for positional changes.

### What about increasing rank?

From the meeting notes, the escalation path is:

```
rank 8 (current) ──> rank 16 ──> rank 32
```

What this means in practice:

| Rank | Params per adapter | Total trainable | % of model |
|------|-------------------|-----------------|------------|
| 8 | ~65K | ~18M | 0.24% |
| 16 | ~131K | ~36M | 0.47% |
| 32 | ~262K | ~72M | 0.95% |

Higher rank = more expressive adapters = potentially better at learning position-invariance.

The tradeoff: higher rank means more memory, slightly slower training, and higher risk of overfitting on small datasets.

---

## 7. LoRA for Model Merging (Why CCoT Uses LoRA)

CCoT doesn't just use LoRA for efficiency. It uses LoRA because **adapters can be merged**:

```
Task A: Base Model + LoRA_A (trained on letter_concat)
Task B: Base Model + LoRA_B (trained on next_last_letter)

Merged: Base Model + merge(LoRA_A, LoRA_B)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Can now do BOTH tasks!
```

### Merging methods

1. **Linear merging**: `merged = w_A * LoRA_A + w_B * LoRA_B` (simple weighted average)
2. **TIES merging**: Smarter -- handles parameter conflicts by magnitude filtering and sign agreement
3. **TIES-SVD**: TIES but in SVD space for better handling of rank structure

### Why this matters for RPE

If RPE-trained LoRA adapters learn position-invariant representations, the merged model should handle longer composite reasoning chains better. Each atomic adapter would be robust to positional shifts, so when their outputs are composed (one task's output feeds into the next task's reasoning), the positions don't break the model.

---

## 8. Quick Reference Card for Meetings

**"What's being updated during LoRA training?"**
> Only the small A and B matrices attached to each of the 7 linear layers in every transformer block. That's ~18M parameters out of 7.6B (~0.24%). Everything else -- the original weights, embeddings, normalization layers, and the RoPE function -- is completely frozen.

**"Does LoRA update positional encoding?"**
> No. RoPE has zero learnable parameters. It's a deterministic rotation function. RPE changes the *input* to RoPE (the position IDs), but neither LoRA nor RPE changes RoPE itself.

**"Which layers are targeted?"**
> All 7 linear layers in each transformer block: 4 attention projections (q, k, v, o) and 3 MLP projections (gate, up, down). This matches the original CCoT paper exactly.

**"Why not just fine-tune everything?"**
> Three reasons: (1) Memory -- 7B params needs ~56GB just for optimizer states in full fine-tuning. (2) Speed -- LoRA trains much faster. (3) Composability -- LoRA adapters can be merged to combine skills, which is the whole point of CCoT.

**"Should we increase rank?"**
> Our meeting notes specify trying rank 16 or 32 if rank 8 doesn't work. Higher rank = more expressive adapters but more parameters and overfitting risk. For RPE specifically, higher rank might help because position-invariance is a complex pattern that rank 8 may struggle to capture.

---

## Sources

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [CCoT GitHub repo configs](https://github.com/fc2869/composable_cot)
- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [HuggingFace PEFT LoRA Guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [A Note on LoRA (arXiv 2404.05086)](https://arxiv.org/html/2404.05086v1)
