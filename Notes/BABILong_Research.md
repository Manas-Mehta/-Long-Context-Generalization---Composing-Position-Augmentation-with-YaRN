# BABILong Benchmark Research

**Date:** 2026-03-23
**Purpose:** Evaluate BABILong as our next dataset for RPE/YaRN/PoSE context extension experiments.

---

## What is BABILong?

**Paper:** "BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack" (NeurIPS 2024 Datasets & Benchmarks Track, arXiv:2406.10149)

**Core idea:** Takes the classic bAbI QA tasks (Facebook Research, 2015) and embeds the reasoning facts as "needles" inside long "haystack" text from PG19 book corpus. Tests whether LLMs can find AND reason over multiple facts distributed across long documents.

**Key difference from MRCR:** MRCR tests multi-fact recall. BABILong tests multi-hop reasoning — the model must chain facts together, not just retrieve them.

**HuggingFace datasets:**
- Eval (100 samples/task/length): `RMT-team/babilong`
- Eval (1000 samples/task/length, up to 128K): `RMT-team/babilong-1k-samples`
- Train (5000 samples/task/length): `RMT-team/babilong-train-5k-samples`

**Metric:** Exact-match accuracy (%).

---

## Task Types

| Task | Name | # Facts Needed | What It Tests |
|------|------|:-:|---------------|
| **qa1** | Single Supporting Fact | 1 | "Where is Mary?" → find 1 movement statement |
| **qa2** | Two Supporting Facts | 2 | Chain 2 facts: X got apple + X went to kitchen → apple in kitchen |
| **qa3** | Three Supporting Facts | 3 | Chain 3 facts. Hardest of the main tasks. |
| qa4 | Two Arg Relations | 1 | Spatial relations with 2 arguments |
| qa5 | Three Arg Relations | 1 | Relations with 3 arguments |
| qa6-qa10 | Various (yes/no, counting, sets, negation, indefinite) | 1-10 | Various reasoning patterns |

**The benchmark primarily evaluates qa1-qa5.** Our interest is in **qa3** (three supporting facts).

---

## Context Length Bins Available

**0K, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M, 10M**

For our experiments, relevant bins would be:
- **ID:** 0K-4K (training range)
- **Slight OOD:** 8K-16K
- **Very OOD:** 32K-128K

Training set available at: 0K, 2K, 4K, 8K, 16K, 32K (5000 samples each per task).

---

## CRITICAL: Leaderboard Scores — Fine-Tuned vs Zero-Shot

### The leaderboard has TWO categories:

**1. Zero-shot/few-shot (majority of models):** GPT-4, Llama, Qwen, Mistral, etc. evaluated with 2-shot prompting. NO fine-tuning.

**2. Fine-tuned (marked with `~` prefix):** Only 3 models:
- `~ ARMT (137M)` — Associative Recurrent Memory Transformer (custom architecture, 137M params)
- `~ Mamba (130M)` — Mamba-130M, fully fine-tuned per-task with curriculum learning
- `~ RMT (137M)` — Recurrent Memory Transformer (custom architecture, 137M params)

**These fine-tuned models are tiny (130-137M params) custom architectures, NOT fine-tuned transformers.** They are not comparable to what we'd do (LoRA on Qwen2.5-7B).

### Fine-tuned model scores (avg qa1-qa5):

| Model | 0K | 4K | 8K | 16K | 32K | 64K | 128K | 512K | 1M |
|-------|:--:|:--:|:--:|:---:|:---:|:---:|:----:|:----:|:--:|
| ARMT (137M) ft | 99 | 98 | 98 | 98 | 98 | 98 | 97 | 95 | 93 |
| Mamba (130M) ft | 99 | 99 | 99 | 99 | 98 | 97 | 93 | — | — |
| RMT (137M) ft | 99 | 92 | 90 | 86 | 78 | 70 | 59 | 46 | 43 |

**Important caveat from paper:** "The current version of the dataset reuses parameters for fact generation from bAbI, this results in a low variety of names and objects within the facts. This makes BABILong tasks easier for fine-tuned models, as they can quickly learn specific tokens that differentiate facts from background text."

The tiny fine-tuned models essentially learn to spot the bAbI fact "vocabulary" (person names like Mary/John, locations like kitchen/garden) within the PG19 book text. This is pattern matching, not general reasoning.

---

## Qwen2.5-7B-Instruct (Zero-Shot) — Our Base Model

| Task | 0K | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
|------|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:---:|:----:|
| **qa1** | ~97 | ~95 | ~93 | ~90 | ~93 | ~80 | ~71 | ~79 | ~39 |
| **qa2** | ~64 | ~58 | ~56 | ~54 | ~50 | ~39 | ~37 | ~32 | ~16 |
| **qa3** | **~25** | **~28** | **~26** | **~28** | **~26** | **~30** | **~28** | **~33** | **~24** |
| avg(qa1-5) | ~62 | ~59 | ~56 | ~52 | ~50 | ~46 | ~46 | ~42 | — |

**QA3 is exactly where we want it:** ~25-30% across ALL context lengths. This is low enough that there's room to push to 0.5-0.7 with fine-tuning + position methods.

**QA1 is too easy** at short contexts (97% zero-shot) but degrades badly at 128K (39%).

**QA2 is moderately hard** — drops from 64% to 16% across length.

---

## Other Model Zero-Shot Scores (for reference)

### QA1 (Single Fact) — across models at various lengths:

| Model | 0K | 8K | 32K | 64K | 128K |
|-------|:--:|:--:|:---:|:---:|:----:|
| GPT-4-0125-preview | 100 | 100 | 93 | 83 | 64 |
| Llama-3.1-70B | 100 | 100 | 98 | 97 | 80 |
| Qwen2.5-7B | 97 | 93 | 71 | 79 | 39 |
| Llama-3.1-8B | 93 | 93 | 68 | 48 | 15 |

### QA3 (Three Facts) — the hard one:

| Model | 0K | 8K | 32K | 64K | 128K |
|-------|:--:|:--:|:---:|:---:|:----:|
| GPT-4-0125-preview | 63 | 30 | 25 | 24 | 18 |
| Llama-3.1-70B | 48 | 36 | 31 | 28 | 14 |
| Qwen2.5-7B | 25 | 26 | 28 | 33 | 24 |
| Llama-3.1-8B | 30 | 26 | 25 | 22 | 15 |
| Gemini 1.5 Pro | 68 | — | — | — | — |

**Even GPT-4 only gets 63% on QA3 at 0K (no distractors!)** and drops to 18% at 128K.

---

## What the Paper Says About YaRN

From Section 3.1: **"Yarn fails to extend to longer contexts despite showing stable results in long-context language modeling."**

They tested **YaRNv2 Mistral** — zero-shot, no fine-tuning. YaRN keeps perplexity stable at long contexts but doesn't help the model actually find and reason over scattered facts. BABILong requires active retrieval + multi-hop reasoning, not just stable next-token prediction.

**Why this is good for us:** They only tested YaRN zero-shot. We fine-tune WITH YaRN + RPE/PoSE. Their finding that YaRN alone isn't enough supports our approach of combining it with position manipulation methods.

---

## Per-Task Accuracy Details (from Table 1, 0K = no distractor text)

| Task | Total Facts | Relevant Facts | 0K Median | Satisfactory? | Notes |
|------|:---:|:---:|:---:|:---:|---|
| QA1 | 2-10 | 1 | ~93% | Yes | Most models solve it |
| QA2 | 2-68 | 2 | ~64% | No | Only GPT-4 & Gemini >85% at 0K |
| **QA3** | **4-320** | **3** | **~36%** | **No** | **Best scores below 70% even at 0K** |
| QA4 | 2 | 1 | ~33% | No | Only 5/24 models reach 70% |
| QA5 | 2-126 | 1 | ~82% | Borderline | Easy-medium |
| QA6 | 2-26 | 1 | ~81% | Borderline | Yes/no questions |
| QA7 | 2-52 | 1-10 | ~77% | No | Counting |
| QA8 | 2-50 | 1-8 | ~77% | No | Lists/sets |
| QA9 | 2-10 | 1 | ~86% | Yes | Negation |
| QA10 | 2-10 | 1 | ~65% | No | Indefinite knowledge |

**Metric:** Exact string match. Paper defines: >85% = satisfactory, <30% = complete failure.

**Why QA3 has 4-320 facts:** Each sample includes many distractor facts (other characters moving around) but only 3 are needed to answer the question. At 0K (no book text), the model still must sift through up to 317 irrelevant facts. At longer contexts, book text is added ON TOP of these distractor facts.

---

## Details on Fine-Tuned Models (Leaderboard)

### All 3 leaderboard models are custom architectures with FULL fine-tuning (no LoRA)

| Detail | RMT (137M) | ARMT (145M) | Mamba (130M) |
|--------|:---:|:---:|:---:|
| **What is it?** | GPT-2 + memory tokens passed between segments | GPT-2 + associative key-value memory per layer | State-space model (NOT a transformer) |
| **Fine-tuning** | Full (all params) | Full (all params) | Full (all params) |
| **LoRA?** | No | No | No |
| **Per-task?** | Yes (1 model per QA task) | Yes | Yes |
| **Training samples** | 10K/task | 10K/task | 10K/task |
| **Curriculum** | Yes: 1→2→4→8→16→32 segments | Yes: 2→3→5→8→16→32 segments | Yes: same schedule |
| **Segment size** | 512 tokens | 512 tokens | N/A |
| **Max train length** | 16K tokens (32 segs) | 16K tokens (32 segs) | 16K tokens |
| **Optimizer** | AdamW, LR=1e-5, WD=0.01 | AdamW, LR~1e-5 | AdamW, LR=3e-4, WD=2.0 |
| **Batch size** | 64 | Not stated | 128 |
| **Steps/stage** | 10,000 | Not stated | 10K (15K last stage) |
| **Hardware** | 1-4x A100 80GB | 1-4x A100/H100 | 4x H100, 2-3 days/task |

**Key point:** These models learn to spot the limited bAbI vocabulary (Mary, John, kitchen, garden, etc.) within book text. The paper itself notes: "low variety of names and objects makes BABILong easier for fine-tuned models."

### Standard transformer fine-tuning (from paper)

Only QA1 was tested:
- GPT-3.5-Turbo: 1K QA1 samples, 3 epochs, via OpenAI API
- Mistral-7B: 1K QA1 samples, 3 epochs, method not specified (no LoRA mentioned)
- Result: uniform ~60-70% on QA1 across lengths (better than zero-shot collapse)
- **QA2/QA3 transformer fine-tuning: NOT tested by anyone.**

**This is the gap we'd fill:** No published fine-tuned transformer results on QA3 with position extension methods.

---

## Dataset Example (QA3)

```
Input (at 4K context):
Mary went to the garden. [~500 tokens of PG19 book text]
Mary picked up the apple. [~1500 tokens of PG19 book text]
Mary went to the kitchen. [~1500 tokens of PG19 book text]

Question: Where is the apple?
Target: kitchen
```

The model must:
1. Find "Mary picked up the apple" (fact 1)
2. Find "Mary went to the kitchen" (fact 2, after fact 1)
3. Reason: apple moves with Mary → apple is in kitchen

At longer contexts (32K, 64K, 128K), the 3 facts are separated by tens of thousands of tokens of irrelevant book text.

**Answer format:** Single word (location name: kitchen, garden, bathroom, hallway, bedroom, office).

---

## Training Data Availability

| Dataset | Samples/Task/Length | Lengths | Tasks | Total Size |
|---------|:--:|---------|-------|-----------|
| `babilong-train-5k-samples` | 5,000 | 0K, 2K, 4K, 8K, 16K, 32K | qa1-qa10 | ~300K rows, 7.4 GB |
| `babilong-1k-samples` | ~1,000 | 0K-128K | qa1-qa10 | eval set |
| `babilong` | 100 | 0K-10M | qa1-qa10 | eval set |

**Note:** QA3 training set has NO 1K bin (missing from dataset). Available: 0K, 2K, 4K, 8K, 16K, 32K.

### QA3 Exact Sample Counts & Token Lengths

| Bin | Train Samples | Eval Samples (1k set) | Approx Token Length |
|:---:|:---:|:---:|:---:|
| 0K | 5,000 | 999 | ~344 (230-541) |
| 1K | — (missing) | 862 | ~709 |
| 2K | 5,000 | 998 | ~1,709 |
| 4K | 5,000 | 999 | ~3,708 |
| 8K | 5,000 | 999 | ~7,709 |
| 16K | 5,000 | 999 | ~15,709 |
| 32K | 5,000 | 999 | ~31,709 |
| 64K | — | 999 | ~63,707 |
| 128K | — | 999 | ~127,706 |

Token lengths are extremely tight (variance < 10 tokens within a bin, except 0K).

### Our Experiment Breakdown

| | ID (Train) | Slight OOD | Very OOD |
|---|---|---|---|
| **Bins** | 0K, 2K, 4K, 8K | 16K, 32K | 64K, 128K |
| **Train samples** | 5K x 4 = **20,000** | — | — |
| **Eval samples** | ~1K x 4 = **~4,000** | ~1K x 2 = **~2,000** | ~1K x 2 = **~2,000** |
| **Token range** | 344 - 7,709 | 15,709 - 31,709 | 63,707 - 127,706 |
| **Qwen2.5-7B zero-shot** | ~25-28% | ~28-30% | ~24-33% |
| **Target after fine-tuning** | 70-90% | 50-70% | 40-60% |

**Training on 0K-8K (20K samples)** is recommended. Much more than MRCR's 60 samples — no memorization concern. No need to extend to 16K for sample count.

**Comparison with MRCR:**
| | MRCR | BABILong QA3 |
|---|---|---|
| Training samples | 60 | 20,000 |
| Training token range | 4K-8K | 344-7,709 |
| Eval bins | 5 (4K-128K) | 9 (0K-128K) |
| Eval samples/bin | 26-30 | ~1,000 |
| Zero-shot baseline | 5-39% | 24-33% |
| Answer format | Free text | Single word |

---

## Feasibility Assessment for RPE/YaRN/PoSE Experiments

### Pros:
1. **QA3 is in the sweet spot (~25-30%).** Not solved, not random. Room to push to 50-70%.
2. **Plenty of training data.** 5000 samples/length vs 60 in MRCR.
3. **Clear bin structure.** 0K, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K maps perfectly to our ID/OOD framework.
4. **No published fine-tuned transformer results on QA3.** Novel contribution.
5. **Clean evaluation.** Single-word answers, exact match. No ambiguity.
6. **Multi-hop reasoning.** Tests whether position methods help with chaining facts, not just retrieval.

### Cons / Concerns:
1. **Vocabulary is limited.** Only ~10 person names and ~6 locations in bAbI facts. Fine-tuned models can learn to spot bAbI vocabulary tokens within book text (pattern matching rather than true reasoning).
2. **Mamba/ARMT dominate.** Recurrent architectures near-perfect. But they're custom architectures — not a fair comparison to LoRA fine-tuning.
3. **QA1 is too easy** for fine-tuned models. Would need QA2 or QA3 specifically.
4. **Long context eval is expensive.** 128K token inputs require significant GPU memory/time.

### Is it fair to compare RPE+YaRN vs fine-tuned YaRN baseline?

**Yes, absolutely.** The correct comparison would be:
- **Baseline 1:** LoRA fine-tuned Qwen2.5-7B (no position tricks) on QA3
- **Baseline 2:** YaRN+LoRA fine-tuned (position extension via frequency scaling only)
- **Experimental:** YaRN+RPE/PoSE+LoRA fine-tuned (our methods)

The zero-shot scores (25-30% on QA3) set the floor. LoRA baseline will be higher. The question is whether RPE/PoSE push it meaningfully above YaRN-only baseline at long contexts.

### About the Mamba concern:

The fine-tuned Mamba-130M is a **fully fine-tuned 130M parameter model** trained per-task with curriculum learning on 4x H100 for 2-3 days. This is:
- A completely different architecture (state-space model, not transformer)
- Fully fine-tuned (all parameters), not LoRA
- Per-task training (one model per QA task)
- Not competing in the same category as us

**We are studying position encoding methods for transformers.** Mamba's success actually supports our thesis — standard transformers need help with long-range position handling, and position encoding methods (RPE, PoSE, YaRN) are the transformer-native approach to this problem.

---

## Recommended Experiment Design (if we proceed)

### Task: QA3 (Three Supporting Facts)

### Training:
- Base model: Qwen2.5-7B-Instruct
- Train on: QA3, bins 0K + 2K + 4K (15,000 samples total)
- LoRA rank 16, same hyperparams as MRCR
- 2-3 epochs (much more data, won't memorize)

### Eval:
- QA3, all bins: 0K, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K
- 1000 samples per bin (use `babilong-1k-samples`)
- ID = 0K-4K, Slight OOD = 8K-16K, Very OOD = 32K-128K

### Conditions (based on MRCR winners):
1. LoRA baseline (no position tricks)
2. YaRN f=2 + LoRA (baseline for position extension)
3. YaRN f=2 + PoSE target=32K + LoRA
4. YaRN f=2 + RPE curriculum L=16K + LoRA
5. Pure YaRN f=2 + LoRA (control — how much does position manipulation add?)

### Expected outcomes:
- LoRA baseline: high ID (0K-4K), rapid drop at 16K+
- YaRN: better long-context, similar to MRCR pattern
- YaRN+PoSE / YaRN+RPE: should show flatter degradation curves
- **Target: push QA3 from ~25% (zero-shot) to 50-70% at 32K-128K**

---

## Decision Needed

**Proceed with BABILong QA3?**
- QA3 scores are genuinely low (~25-30%) — room to improve
- Training data is abundant (5000/bin vs 60 in MRCR)
- No published transformer fine-tuning results on QA3
- Clean eval metric, clear bins
- Answers the question: "Can position encoding methods help multi-hop reasoning at scale?"

**Alternative:** QA2 (Two Supporting Facts) is also viable — higher baseline (~50-64%) but still degrades badly at long contexts.
