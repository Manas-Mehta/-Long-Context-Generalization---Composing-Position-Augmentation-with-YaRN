# Direction 2: Context Extension for Long-Context LLMs

**Researcher**: Manas Mehta
**Institution**: TAUR Labs, Courant Institute, NYU
**Date**: February 2026
**Status**: Direction selection & planning

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [How This Connects to Our RPE Work](#2-how-this-connects-to-our-rpe-work)
3. [Taxonomy of the Space](#3-taxonomy-of-the-space)
4. [Key Methods & Literature](#4-key-methods--literature)
5. [Evaluation: Benchmarks & Metrics](#5-evaluation-benchmarks--metrics)
6. [Open Challenges](#6-open-challenges)
7. [Candidate Research Directions](#7-candidate-research-directions)
8. [Proposed Plan & Next Steps](#8-proposed-plan--next-steps)
9. [References](#9-references)

---

## 1. Problem Definition

### What is Context Extension?

**Definition**: Train on short input contexts (e.g., 4K-8K tokens), generalize to much longer input contexts (e.g., 32K-128K+ tokens) at inference time.

**Nature of the task**:
- **Input**: Very long (>8K tokens, often 32K-1M+)
- **Output**: Short or medium (answer extraction, summarization, QA)

This distinguishes context extension from length generalization (Direction 3, our RPE work) where inputs and outputs scale together, and from compositional generalization (Direction 1) where short inputs produce long composed outputs.

### Why This Matters

Modern LLMs are trained with fixed context windows (e.g., Llama-3 at 8K, Qwen2.5 at 32K). Extending these windows is critical for:

1. **Document understanding**: Legal contracts, scientific papers, codebases (50K-500K tokens)
2. **Multi-turn dialogue**: Extended conversations accumulate context over time
3. **Retrieval-augmented generation**: More retrieved passages = better grounding, but requires longer context
4. **Agentic workflows**: Tool-use traces, multi-step reasoning, environment observations

The fundamental bottleneck is **self-attention's quadratic complexity**: O(n^2) in both time and memory, where n is the sequence length. A model trained at 8K context requires 16x more compute for attention at 32K, and 256x at 64K.

### The Core Technical Problem

Transformer positional encodings (particularly RoPE) are trained to represent positions within a fixed range. When sequences exceed this range:

1. **RoPE frequencies extrapolate** to unseen rotation angles, producing attention patterns the model never learned
2. **Attention distributions shift** — the model can't reliably attend to distant tokens
3. **KV-cache memory grows linearly** with sequence length, creating GPU memory pressure
4. **Information retrieval degrades** — the "lost-in-the-middle" phenomenon emerges

The research question: **How do we efficiently extend a pretrained model's effective context window while preserving short-context performance, without full retraining?**

---

## 2. How This Connects to Our RPE Work

Our Phase 1-2 RPE experiments directly inform this direction:

| RPE Insight | Context Extension Relevance |
|---|---|
| RoPE is a functional transform (not learned weights) — can be manipulated at the position_ids level | All RoPE-scaling methods (YaRN, LongRoPE, PoSE) operate on this same principle |
| LoRA can learn position-invariant patterns with only 0.53% of params | Context extension via LoRA fine-tuning (LongLoRA) is a proven approach |
| Curriculum learning (gradual L increase) outperformed fixed RPE | Curriculum context extension (progressive length training) is state-of-the-art |
| L calibration is critical (L=8192 failed, L=1024 worked) | RoPE scaling factor calibration is the central challenge in YaRN/LongRoPE |
| Sharp cliff pattern in LoRA generalization | Context extension methods also show sharp capability boundaries |
| Monkey-patching `model.forward()` for position_ids injection | Same infrastructure works for any RoPE-scaling intervention |

**Key advantage**: Our `RPEPatcher` infrastructure (monkey-patching HuggingFace models' `forward()` to modify position_ids) is directly reusable. Context extension methods like PoSE and YaRN also manipulate position_ids — we already have the scaffolding.

---

## 3. Taxonomy of the Space

Context extension methods fall into several categories:

### 3.1 Positional Encoding Methods (Inference-time / Light Fine-tuning)

Modify how RoPE encodes positions to handle longer sequences. Minimal or no training required.

| Method | Key Idea | Training Required | Context Achieved |
|---|---|---|---|
| **PI (Position Interpolation)** | Linear downscaling of positions to fit in trained range | Light FT (~1K steps) | 32K |
| **YaRN** | NTK-aware interpolation + attention temperature scaling | Light FT (~400 steps) | 128K |
| **LongRoPE** | Non-uniform interpolation via evolutionary search | Light FT | 2048K |
| **Self-Extend** | Group attention + neighbor attention, training-free | None | 16K→ |
| **PoSE** | Positional skip-wise training on fixed window | Light FT | 128K |
| **CLEX** | Learned continuous scaling via ODE | Light FT | 64K |

### 3.2 Architecture / Attention Modifications

Change how attention is computed to reduce or redistribute cost.

| Method | Key Idea | Complexity |
|---|---|---|
| **LongLoRA (S^2-Attn)** | Shifted sparse attention in groups | O(n * group_size) |
| **Ring Attention** | Distribute KV blocks across devices, overlap comm/compute | O(n^2 / num_devices) |
| **Striped Attention** | Permuted Ring Attention for balanced workload | O(n^2 / num_devices) |
| **Flash Attention 2/3** | IO-aware tiling, fused kernels | O(n^2) but fast constant |
| **Infini-Attention** | Compressive memory for infinite context | O(n) amortized |
| **StreamingLLM** | Attention sinks + sliding window | O(window_size) |

### 3.3 Data-Centric Methods

Improve what data the model trains on for long-context capability.

| Method | Key Idea |
|---|---|
| **ProLong** | Mine documents with genuine long-range dependencies (not just long documents) |
| **UltraLong** | Upsample long documents + curriculum from 128K to 4M |
| **LongAlign** | Construct long instruction-following datasets for SFT |

### 3.4 Memory-Augmented Methods

External memory to handle contexts beyond GPU capacity.

| Method | Key Idea |
|---|---|
| **InfLLM** | Store distant KV states in CPU memory, retrieve relevant blocks |
| **MemWalker** | Navigate long context via iterative memory reads |
| **Landmark Attention** | Insert landmark tokens for coarse-grained retrieval |

---

## 4. Key Methods & Literature

### 4.1 YaRN (Peng et al., 2023) — The Foundation

**Paper**: "YaRN: Efficient Context Window Extension of Large Language Models" (ICLR 2024)

YaRN is the most widely adopted RoPE-scaling method and the natural starting point for this direction.

**Core technique**: NTK-aware interpolation partitions RoPE frequency dimensions into three regions:
- **High frequencies** (local patterns): Apply gentle interpolation — these capture nearby token relationships and shouldn't be disrupted
- **Low frequencies** (global patterns): Apply stronger interpolation/extrapolation — these encode long-range position and need the most adjustment
- **Temperature scaling**: Scale attention logits by `sqrt(1/t)` where `t = 0.1 * ln(s) + 1` to compensate for the entropy change from scaled positions

**Why it works**: Standard position interpolation compresses ALL frequencies equally, destroying local patterns. NTK-aware interpolation preserves high-frequency (local) information while only modifying the low-frequency (global) components.

**Efficiency**: 10x fewer tokens and 2.5x fewer training steps than full-context training. Achieves 128K context with minimal degradation.

**Connection to RPE**: Both YaRN and RPE modify position_ids to achieve generalization. RPE randomizes positions; YaRN scales them. Our curriculum RPE schedule (L: 640 → 1024) is conceptually similar to YaRN's progressive scaling — both gradually expose the model to wider positional ranges.

### 4.2 LongRoPE (Microsoft, 2024) — Pushing the Frontier

**Paper**: "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens" (ICML 2024)

**Three innovations**:
1. **Non-uniform positional interpolation**: Uses evolutionary search to find optimal per-dimension scaling factors (not uniform like PI, not analytically derived like YaRN, but data-driven)
2. **Progressive extension**: First fine-tune to 256K, then apply second-stage interpolation to reach 2048K
3. **Short-context recovery**: Readjust RoPE on 8K sequences to recover short-context performance

**LongRoPE2 (2025)**: Extends Llama-3-8B to 128K effective context retaining >98.5% short-context performance, using only 10B tokens (80x fewer than Meta's approach).

**Key lesson**: Progressive/curriculum extension outperforms one-shot extension — consistent with our RPE curriculum finding.

### 4.3 LongLoRA (Chen et al., 2023) — LoRA for Context Extension

**Paper**: "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models" (ICLR 2024 Oral)

**Directly relevant** — uses LoRA (like us) for context extension:

1. **Shifted Sparse Attention (S^2-Attn)**: During training, splits sequence into groups and computes attention within each group. Half the attention heads shift tokens by half the group size, ensuring cross-group information flow. At inference, reverts to full attention.
2. **LoRA + embeddings + norms**: Standard LoRA alone is insufficient for context extension. Training embedding layers and normalization layers alongside LoRA is critical.

**Results**: Llama-2 7B → 100K context; 70B → 32K context. All on a single 8x A100 machine.

**Connection to our work**: We showed LoRA can learn position-invariant patterns via RPE. LongLoRA shows LoRA can learn context extension. Combining RPE-style position manipulation with LongLoRA's efficient attention could be a novel contribution.

### 4.4 PoSE (Zhu et al., 2023) — Positional Skip-Wise Training

**Paper**: "PoSE: Efficient Context Window Extension via Positional Skip-wise Training" (ICLR 2024)

**Core idea**: Simulate long sequences using a fixed short context window.

**How it works**:
1. Divide the original context window into chunks
2. Add distinct skip bias terms to position indices of each chunk, so they're spread across the target range
3. Vary bias terms and chunk lengths per training example
4. The model adapts to all positions within the target range without ever seeing a full-length sequence

**Key advantage**: Decouples training length from target length. Extended Llama to 128K using only a 2K training window. Theoretically extends to infinite context (constrained only by inference memory).

**Strong connection to RPE**: PoSE manipulates position_ids just like RPE, but with structured skips rather than random sampling. Our `RPEPatcher` infrastructure could implement PoSE with minimal modification — instead of random sampling, we'd inject skip-biased positions.

### 4.5 ProLong (Gao et al., 2024) — Data-Centric Approach

**Paper**: "How to Train Long-Context Language Models (Effectively)" (ACL 2025)

**Key insight**: Not all long documents teach long-context understanding. A 100K-token document where every paragraph is independent teaches nothing about long-range dependencies. ProLong identifies documents with genuine long-range dependencies using:

1. **Dependency strength**: Delta perplexity when conditioning on distant context vs. not
2. **Dependency distance**: Refines strength by incorporating spatial distance between segments
3. **Dependency specificity**: Filters out trivial dependencies from repetitive patterns

**Results**: ProLong-8B, trained on 40B tokens, achieves SOTA at 128K among models of similar size.

**Relevance**: If we do continued pretraining or fine-tuning for context extension, the quality of training data matters enormously. ProLong's data-mining framework could identify the most effective documents for long-context fine-tuning.

### 4.6 Infini-Attention (Google, 2024) — Infinite Context

**Paper**: "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"

**Core innovation**: Augments standard causal attention with a compressive memory:
- Local causal attention handles the current segment
- A long-term linear attention mechanism continuously updates a compressed representation of all past context
- Achieves 114x compression ratio with bounded compute

**Results**: 1B model scales to 1M sequence length; 8B model achieves SOTA on 500K book summarization.

**Relevance**: Represents the "memory augmentation" approach — fundamentally different from RoPE scaling. If RoPE methods hit a ceiling, memory-based methods offer an alternative path.

### 4.7 StreamingLLM (Xiao et al., 2023) — Attention Sinks

**Paper**: "Efficient Streaming Language Models with Attention Sinks" (ICLR 2024)

**Discovery**: LLMs allocate disproportionate attention to the first few tokens ("attention sinks"), regardless of semantic content. This is a mathematical artifact of softmax normalization — the model needs somewhere to "park" unused attention.

**Solution**: Keep the first 4 tokens permanently + sliding window for the rest. This enables stable streaming inference over 4M+ tokens with constant memory.

**Relevance**: Understanding attention sinks is crucial for any context extension work. Removing initial tokens (as in naive sliding window) causes catastrophic failure.

---

## 5. Evaluation: Benchmarks & Metrics

### 5.1 Recommended Benchmarks

#### HELMET (Yen et al., 2025) — Comprehensive Realistic Eval

**Paper**: "How to Evaluate Long-Context Models Effectively and Thoroughly" (ICLR 2025)

Seven categories, all supporting 128K+ inputs:
1. **Synthetic recall** — controlled retrieval in synthetic context
2. **Long-document QA** — NaturalQuestions, TriviaQA with long documents
3. **Summarization** — book/document summarization
4. **Many-shot ICL** — learning from 100+ in-context examples
5. **RAG** — generation with retrieved passages
6. **Re-ranking** — passage re-ranking over long lists
7. **Citation generation** — producing text with proper attribution

**Why HELMET**: Low inter-category correlation (each tests something different), model-based evaluation for reliable metrics, supports both base and instruction-tuned models. Adopted by Microsoft Phi-4, AI21 Jamba 1.6.

#### MRCR (Vodrahalli et al., 2026) — Synthetic Reasoning

**Paper**: "Michelangelo: Long Context Evaluations Beyond Haystacks"

- Multi-round coreference resolution in synthetic conversations
- Prompts up to 1M tokens
- Tests: find the i-th instance of a specific request (e.g., "return the 2nd poem about tapirs")
- Harder than NIAH: requires distinguishing between multiple similar needles
- Available: HuggingFace `openai/mrcr`

**Why MRCR**: Replacement for Needle-in-a-Haystack. Immune to pretraining contamination (synthetic). Tests reasoning, not just retrieval.

#### NIAH (Needle in a Haystack) — Classic Baseline

- Hide a fact in long context, ask model to retrieve it
- Simple but useful for ablation and position-sensitivity analysis
- Reveals lost-in-the-middle patterns clearly
- GitHub: `gkamradt/LLMTest_NeedleInAHaystack`

### 5.2 Metrics

| Metric | What It Measures | Pitfalls |
|---|---|---|
| **Perplexity (PPL)** | Language modeling quality on long text | Correlates poorly with downstream task performance on long context; averages over all tokens, masking failures on key tokens |
| **LongPPL** | PPL on key tokens identified via long-short contrast | Strong correlation (-0.96) with benchmarks; better than standard PPL |
| **Task accuracy** | Correct answers on HELMET/MRCR/NIAH tasks | Application-specific; requires per-category analysis |
| **Effective context length** | Max length where performance stays above threshold | Threshold-dependent; varies by task |
| **Short-context retention** | Performance on original-length tasks after extension | Critical — extension that breaks short context is useless |

**Recommendation**: Always report (1) task accuracy on HELMET categories, (2) short-context retention, (3) NIAH heatmap for position sensitivity analysis. Avoid relying on perplexity alone.

---

## 6. Open Challenges

### 6.1 Lost-in-the-Middle

**The problem**: LLMs perform best when relevant information is at the beginning or end of context, and significantly worse when it's in the middle — even models explicitly trained for long context.

**Root causes**:
- Causal attention: early tokens receive more cumulative attention processing
- Softmax normalization: attention scores concentrate on initial tokens (attention sinks)
- RoPE long-term decay: inherent bias toward nearby tokens

**Current solutions**: Ms-PoE (multi-scale positional encoding), position-aware training, explicit middle-biased attention. All partial fixes — the problem persists at >64K context in most models.

**Research opportunity**: Can RPE-style randomization during fine-tuning help? If the model trains with randomized positions, it can't rely on the beginning/end bias — forcing it to attend uniformly. This is a direct extension of our RPE work.

### 6.2 Perplexity vs. Downstream Performance Gap

**The problem**: A model can achieve low perplexity on 128K-token documents while failing to answer questions about information in those documents. PPL reflects local next-token prediction quality, not long-range information utilization.

**Implication**: You cannot evaluate context extension methods by perplexity alone. Must use task-based benchmarks (HELMET, MRCR).

### 6.3 Short-Context Performance Degradation

**The problem**: Many context extension methods degrade short-context performance. The model trades off precision at short range for coverage at long range.

**Quantified**: LongRoPE2 retains >98.5% short-context performance. YaRN shows minimal degradation. But aggressive methods (high scaling factors, heavy fine-tuning) can lose 5-10% on short benchmarks.

**Our RPE analogy**: We saw this — RPE curriculum had 96.7% in-dist vs. baseline's 100%. Small but real cost. Context extension must carefully balance this trade-off.

### 6.4 Memory and Compute at Scale

**The problem**: Self-attention is O(n^2) in time and memory. At 128K tokens, attention alone requires ~32GB memory for KV-cache on a 7B model. At 1M tokens, it's infeasible without distributed computing or memory offloading.

**Current approaches**:
- **Flash Attention 3**: Reduces memory via tiling/recomputation, but doesn't change asymptotic complexity
- **Ring Attention**: Distributes across devices, enabling device_count * max_length effective context
- **KV-cache compression**: Quantize, prune, or compress the cache (GQA, MQA, H2O)
- **Memory offloading**: InfLLM stores distant KV states in CPU memory

**Practical constraint**: Single-GPU methods (our NYU Torch HPC setup: 1x L40S, 48GB VRAM) limit us to ~32K-64K effective context for 7B models without compression. Multi-GPU or memory-offloading methods would be needed beyond that.

### 6.5 Evaluation Contamination

**The problem**: Synthetic benchmarks can be gamed; realistic benchmarks may appear in pretraining data.

**Mitigation**: MRCR uses synthetic generation to avoid contamination. HELMET uses natural documents but controls for overlap. Always test on multiple benchmarks.

---

## 7. Candidate Research Directions

### Direction A: RPE-Enhanced Context Extension (Builds on Our Work)

**Idea**: Combine our RPE position randomization with RoPE scaling methods (YaRN/PoSE) to improve context extension.

**Hypothesis**: RPE trains position-invariant attention patterns. If applied during the context extension fine-tuning phase, it could help the model generalize to positions it hasn't been fine-tuned on — extending the effective context beyond the fine-tuning length.

**Concrete plan**:
1. Take Qwen2.5-7B (32K native context)
2. Apply YaRN scaling to extend to 64K or 128K
3. Fine-tune with LoRA on 32K-length documents
4. During fine-tuning, apply RPE (randomize position_ids within the YaRN-scaled range)
5. Evaluate: does RPE improve generalization beyond 64K/128K vs. YaRN alone?

**Strengths**: Directly extends our existing code and results. Novel combination not explored in literature. Clear experimental design.

**Risks**: RPE's effectiveness at very long contexts is unknown. The position space is much larger (128K vs. 1024 in our experiments). LoRA capacity may be insufficient.

### Direction B: PoSE + Curriculum Extension

**Idea**: Implement PoSE (positional skip-wise training) with our curriculum learning approach.

**Hypothesis**: PoSE simulates long contexts using short windows, but uses fixed skip patterns. A curriculum approach (gradually increasing skip magnitudes) could improve stability and performance, just as curriculum RPE outperformed fixed RPE.

**Concrete plan**:
1. Implement PoSE position manipulation using our `RPEPatcher` infrastructure
2. Design curriculum: start with small skips (simulate 16K from 4K window), gradually increase to large skips (simulate 128K from 4K window)
3. Fine-tune with LoRA on document understanding tasks
4. Evaluate on HELMET + NIAH at various context lengths

**Strengths**: Reuses our infrastructure. Curriculum is our proven insight. PoSE is memory-efficient (short training window).

**Risks**: PoSE's skip patterns may not compose well with curriculum scheduling. Need to design the curriculum schedule carefully.

### Direction C: Data-Centric Long-Context Fine-Tuning

**Idea**: Apply ProLong's data-mining framework to select high-quality long-context training data, then fine-tune with LoRA + RoPE scaling.

**Hypothesis**: The quality of long-context training data matters more than quantity. Selecting documents with genuine long-range dependencies (ProLong's methodology) will produce better context extension than training on random long documents.

**Concrete plan**:
1. Implement ProLong's dependency scoring (delta-PPL based)
2. Score and rank a large corpus (e.g., RedPajama, SlimPajama)
3. Select top-k documents by long-dependency score
4. Fine-tune Qwen2.5-7B with LoRA + YaRN on selected data
5. Compare against random long-document baseline on HELMET

**Strengths**: Data-centric approach is complementary to any positional method. Could combine with Direction A or B.

**Risks**: Requires significant compute for data scoring. ProLong's methodology may not transfer directly to our setting.

### Direction D: Efficient Long-Context LoRA (LongLoRA Improvements)

**Idea**: Improve LongLoRA's shifted sparse attention with insights from our RPE work.

**Hypothesis**: LongLoRA's S^2-Attn uses fixed group boundaries. Randomizing group boundaries (RPE-style) during training could improve generalization across different context positions.

**Concrete plan**:
1. Implement S^2-Attn in our framework
2. Add random group boundary shifting (each training step, randomly offset the group boundaries)
3. Train on long-document datasets
4. Evaluate: does randomized S^2-Attn outperform fixed S^2-Attn on HELMET?

**Strengths**: Novel contribution. Directly combines LongLoRA + RPE insights. Efficient (no extra memory cost).

**Risks**: Implementation complexity of S^2-Attn. May require modifying Flash Attention kernels.

---

## 8. Proposed Plan & Next Steps

### Phase 0: Literature Deep-Dive (Week 1)

1. **Read core papers in detail**:
   - YaRN (Peng et al., 2023) — full paper, focus on NTK-aware interpolation math
   - PoSE (Zhu et al., 2023) — full paper, focus on skip-wise position construction
   - LongLoRA (Chen et al., 2023) — full paper, focus on S^2-Attn implementation
   - ProLong (Gao et al., 2024) — Sections 3-5 on dependency scoring
   - HELMET (Yen et al., 2025) — Sections 2.1, 3.1 on task categories and setup
   - MRCR (Vodrahalli et al., 2026) — Section 2 on synthetic task design

2. **Recommended reading from Fangcong** (prioritize these):
   - HELMET Section 2.1 (realistic task categories) and Section 3.1 (setup)
   - MRCR Section 2 (synthetic task design)
   - ProLong Sections 3-5 (data mining methodology)
   - NIAH baseline: `github.com/gkamradt/LLMTest_NeedleInAHaystack`

3. **Supplementary reading**:
   - LongRoPE / LongRoPE2 for frontier scaling techniques
   - StreamingLLM for attention sink understanding
   - Flash Attention 3 for implementation-level efficiency

### Phase 1: Infrastructure & Baselines (Weeks 2-3)

1. **Evaluation infrastructure**:
   - Set up HELMET evaluation pipeline (or a subset of categories: Synthetic Recall + Long-Doc QA + RAG)
   - Implement NIAH evaluation with position-sensitivity heatmap
   - Set up MRCR evaluation from HuggingFace
   - Create unified eval script with per-category reporting

2. **Baseline measurements**:
   - Measure Qwen2.5-7B native performance on HELMET at 4K, 8K, 16K, 32K
   - Identify the effective context length cliff (where does performance drop?)
   - Run NIAH heatmap at various lengths to identify position biases

3. **RoPE scaling baseline**:
   - Implement YaRN scaling for Qwen2.5-7B (may already exist in transformers library)
   - Measure YaRN-extended model (no fine-tuning) on HELMET at 32K, 64K, 128K
   - Quantify the gap between "scaling alone" and "scaling + fine-tuning"

### Phase 2: Core Experiments (Weeks 4-6)

Select **one** primary direction (A, B, C, or D) based on Phase 1 findings and advisor input. Run controlled experiments:

**If Direction A (RPE + YaRN)**:
- Exp 0: YaRN scaling only (no fine-tuning) — baseline
- Exp 1: YaRN + LoRA fine-tuning on long docs — standard approach
- Exp 2: YaRN + LoRA + RPE (fixed L) — our contribution
- Exp 3: YaRN + LoRA + RPE (curriculum L) — our best RPE variant
- Evaluate all on HELMET + NIAH at 32K, 64K, 128K

**If Direction B (PoSE + Curriculum)**:
- Exp 0: PoSE (fixed skips) — reproduce paper
- Exp 1: PoSE + curriculum (progressive skip magnitudes) — our contribution
- Exp 2: PoSE + RPE (randomized skips) — novel combination
- Evaluate on HELMET + NIAH

### Phase 3: Analysis & Paper (Weeks 7-8)

1. **Error analysis**: Where do models fail? Is it retrieval, reasoning, or generation?
2. **Position sensitivity**: NIAH heatmaps before/after our intervention
3. **Ablation studies**: What matters most — RoPE scaling, data quality, or training approach?
4. **Write up results** for lab presentation and potential paper

### Resource Requirements

| Resource | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| GPU | 1x L40S (NYU Torch) | 1-4x L40S | 1x L40S |
| VRAM | 48GB (7B inference at 32K) | 48GB+ (training at 32K+) | 48GB |
| Disk | ~50GB (model + data) | ~200GB (long-doc data + checkpoints) | ~50GB |
| Time | ~5-10 GPU-hours | ~50-100 GPU-hours | ~10 GPU-hours |

### Key Decision Points

1. **After Phase 0**: Which direction (A/B/C/D) to pursue? Based on literature gaps and feasibility.
2. **After Phase 1 baselines**: What's the actual performance cliff for Qwen2.5-7B? This determines the extension target.
3. **After Phase 2**: Is the result significant enough for a paper, or should we pivot?

---

## 9. References

### Core Papers (Must Read)

1. **YaRN**: Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models", ICLR 2024. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
2. **LongRoPE**: Ding et al., "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens", ICML 2024. [arXiv:2402.13753](https://arxiv.org/abs/2402.13753)
3. **LongLoRA**: Chen et al., "LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models", ICLR 2024 Oral. [arXiv:2309.12307](https://arxiv.org/abs/2309.12307)
4. **PoSE**: Zhu et al., "PoSE: Efficient Context Window Extension via Positional Skip-wise Training", ICLR 2024. [arXiv:2309.10400](https://arxiv.org/abs/2309.10400)
5. **ProLong**: Gao et al., "How to Train Long-Context Language Models (Effectively)", ACL 2025. [Princeton NLP](https://github.com/princeton-nlp/ProLong)
6. **RPE**: Ruoss et al., "Randomized Positional Encodings Boost Length Generalization of Transformers", ACL 2023. [arXiv:2305.16843](https://arxiv.org/abs/2305.16843)

### Evaluation Benchmarks (Must Read)

7. **HELMET**: Yen et al., "How to Evaluate Long-Context Models Effectively and Thoroughly", ICLR 2025. [arXiv:2410.02694](https://arxiv.org/abs/2410.02694)
8. **MRCR / Michelangelo**: Vodrahalli et al., "Michelangelo: Long Context Evaluations Beyond Haystacks". [arXiv:2409.12640](https://arxiv.org/abs/2409.12640)
9. **NIAH**: Kamradt, "Needle in a Haystack". [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

### Key Phenomena & Analysis

10. **Lost in the Middle**: Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", TACL 2024. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
11. **Attention Sinks / StreamingLLM**: Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
12. **LongPPL**: "What is Wrong with Perplexity for Long-context Language Modeling?", 2024. [arXiv:2410.23771](https://arxiv.org/abs/2410.23771)

### Additional Methods

13. **Self-Extend**: "LLM Maybe LongLM: SelfExtend LLM Context Window Without Tuning". [arXiv:2401.01325](https://arxiv.org/abs/2401.01325)
14. **Infini-Attention**: "Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention". [arXiv:2404.07143](https://arxiv.org/abs/2404.07143)
15. **InfLLM**: "InfLLM: Training-Free Long-Context Extrapolation", NeurIPS 2024.
16. **Ring Attention**: "Ring Attention with Blockwise Transformers for Near-Infinite Context", ICLR 2024. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
17. **Flash Attention 3**: Dao et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision". [GitHub](https://github.com/Dao-AILab/flash-attention)
18. **LongRoPE2**: "LongRoPE2: Near-Lossless LLM Context Window Scaling", 2025. [arXiv:2502.20082](https://arxiv.org/abs/2502.20082)
19. **CLEX**: "CLEX: Continuous Length Extrapolation for Large Language Models". [arXiv:2310.16450](https://arxiv.org/abs/2310.16450)
20. **Position Interpolation**: Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation". [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)

---

*This document was prepared as a research direction proposal for the TAUR Labs context extension project, building on the RPE + CCoT work (Phase 1-2).*
