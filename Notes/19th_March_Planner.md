# 19th March 2026 — Session Planner

## Context

Phase 3-5 of MRCR experiments are complete. Key results so far:
- **RPE cur L=16K**: Best absolute scores (0.751 avg across bins)
- **RPE fixed+YaRN (eval only)**: Flattest curve, 103.8% retention at 128K
- **YaRN+LoRA (f=4, train+eval)**: Decent (49.5% retention) but not combined with RPE/PoSE
- **PoSE fixed**: Best at bin 2 (0.937) but collapses at bin 4, uniquely solves 0 samples

**Gap in our experiments:** We've never trained with YaRN + RPE or YaRN + PoSE simultaneously. YaRN was either train-only (YaRN+LoRA) or eval-only (RPE fixed+YaRN). The hypothesis: training with both active should be strictly better — the model learns to use YaRN-scaled frequencies with randomized/skipped positions.

Professor's feedback on analysis: too complicated, too AI-written. Future analyses must be concise, clearly explained, and show only what matters.

---

## High Priority: Phase 6 — YaRN-Trained Combinations

### How YaRN + RPE works (no ordering ambiguity)

YaRN and RPE operate at **different levels** — they compose naturally:
- **YaRN** modifies the RoPE frequency basis (`inv_freq`) — changes HOW positions are encoded
- **RPE** randomizes the position IDs fed to the model — changes WHICH positions are used

So "YaRN + RPE" means: model has YaRN rope_scaling active (modified frequencies), AND RPE provides random sorted position IDs from [0, L). Both active simultaneously during training. There's no "first YaRN then RPE" vs "first RPE then YaRN" — they're orthogonal.

**One subtle design choice:** Should L be relative to the original 32K window or the YaRN-extended window? With YaRN f=4, the model "knows" up to 128K. Setting L=16K keeps positions in the original-range space; L=64K pushes into YaRN-extended space. We should try both.

Similarly for PoSE + YaRN: PoSE splits into 2 chunks with a position gap, YaRN modifies the frequency basis. Both active during training.

**Is it worth checking the "reverse" ordering?** No. They genuinely operate at different levels (position IDs vs frequency basis). There's only one way to compose them.

---

### Experiment 1: Training Matrix

All models trained on bin 0 (4K-8K, 60 samples), LoRA rank 16, 5 epochs, same hyperparameters as Phase 3. ~21 min each on L40S.

#### Understanding L relative to YaRN's extended context window

Qwen2.5-7B has a native 32K context. YaRN extends this:
- **YaRN f=2** → effective **64K** window
- **YaRN f=4** → effective **128K** window

When we set RPE's L parameter, we're choosing where in this extended space the model samples random positions from. Three regimes:

| Regime | L vs YaRN window | Example (Y4, 128K window) | What it tests |
|--------|-----------------|---------------------------|---------------|
| **L << window** | L is well inside | L=4K, 8K, 16K | RPE for position robustness only. YaRN does all the extension. |
| **L ~ half window** | L approaching | L=64K | RPE pushes into YaRN's extended range but not to the edge. |
| **L = window** | L matches exactly | L=128K | RPE samples across the full YaRN-extended space. Maximum coverage. |

We should try all three regimes.

#### Why does curriculum repeat the last L value?

Schedule example: `8K→10K→12K→16K→16K` (5 values for 5 epochs).

The final epoch repeats the target L as a **consolidation epoch**. After ramping up over 4 epochs, the model has only seen the maximum L for one epoch. Repeating gives it a second epoch to stabilize at the target. Without this, training at maximum L might be undertrained. Think of it as: 80% ramp-up, 20% consolidation.

---

#### Complete Training Table (18 runs)

##### Group A: RPE-only controls — small L (no YaRN, 2 runs)

We've done L=16K/32K/64K/128K in Phase 5 but never below 16K. Adding these as controls to compare against YaRN+RPE at the same L values.

| # | ID | YaRN | Method | L | Curriculum | Rationale |
|---|-----|------|--------|---|------------|-----------|
| 1 | **R-c4** | None | RPE cur | 4K | 2K→3K→3.5K→4K→4K | Control for Y*-Rc4 runs. Is L=4K useful without YaRN? |
| 2 | **R-c8** | None | RPE cur | 8K | 4K→5K→6K→8K→8K | Control for Y*-Rc8 runs. |

##### Group B: YaRN-only baselines (2 runs)

| # | ID | YaRN | Method | Notes |
|---|-----|------|--------|-------|
| 3 | **Y2** | 2.0 | LoRA only | Train low → eval high. We already have Y4. |
| 4 | **Y3** | 3.0 | LoRA only | Interpolation between f=2 and f=4. |

*Rationale:* If f=2 at training + f=4 at eval works well, we can train cheaply and scale at inference.

##### Group C: YaRN + RPE — small L (L << YaRN window, 6 runs)

RPE provides position robustness, YaRN handles extension. L stays well inside the native 32K window.

| # | ID | YaRN | L | Curriculum | Rationale |
|---|-----|------|---|------------|-----------|
| 5 | **Y2-Rc4** | 2.0 | 4K | 2K→3K→3.5K→4K→4K | Minimal RPE + modest YaRN. |
| 6 | **Y2-Rc8** | 2.0 | 8K | 4K→5K→6K→8K→8K | Mid-range L, modest YaRN. |
| 7 | **Y2-Rc16** | 2.0 | 16K | 8K→10K→12K→16K→16K | Our best L + modest YaRN. |
| 8 | **Y4-Rc4** | 4.0 | 4K | 2K→3K→3.5K→4K→4K | Minimal RPE + full YaRN. |
| 9 | **Y4-Rc8** | 4.0 | 8K | 4K→5K→6K→8K→8K | Mid-range L + full YaRN. |
| 10 | **Y4-Rc16** | 4.0 | 16K | 8K→10K→12K→16K→16K | Our best L + full YaRN. Top candidate. |

##### Group D: YaRN + RPE — L near/at YaRN window (4 runs)

RPE samples positions into YaRN's extended range. Tests whether the model benefits from seeing random positions at the scale YaRN was designed for.

| # | ID | YaRN | L | Curriculum | L vs YaRN window | Rationale |
|---|-----|------|---|------------|-----------------|-----------|
| 11 | **Y2-Rc32** | 2.0 | 32K | 16K→20K→24K→32K→32K | Half of 64K window | Approaching Y2's extended range. |
| 12 | **Y2-Rc64** | 2.0 | 64K | 16K→32K→48K→64K→64K | Matches 64K window | Full coverage of Y2's extended space. |
| 13 | **Y4-Rc64** | 4.0 | 64K | 16K→32K→48K→64K→64K | Half of 128K window | Approaching Y4's extended range. |
| 14 | **Y4-Rc128** | 4.0 | 128K | 16K→48K→80K→128K→128K | Matches 128K window | Full coverage of Y4's extended space. |

*Key question this answers:* Should L be small (just for robustness) or large (to explore the full YaRN-extended space)?

##### Group E: YaRN + PoSE (4 runs)

| # | ID | YaRN | PoSE target_length | n_chunks | Rationale |
|---|-----|------|-------------------|----------|-----------|
| 15 | **Y2-P16** | 2.0 | 16K | 2 | PoSE + modest YaRN, moderate skip. |
| 16 | **Y2-P32** | 2.0 | 32K | 2 | PoSE + modest YaRN, large skip. |
| 17 | **Y4-P16** | 4.0 | 16K | 2 | PoSE + full YaRN, moderate skip. |
| 18 | **Y4-P32** | 4.0 | 32K | 2 | PoSE + full YaRN, large skip. |

*PoSE design notes:*
- **target_length** (16K vs 32K): Main knob. Controls max position gap between chunks.
- **n_chunks** (2 vs 3): Currently hardcoded to 2. More chunks → more RPE-like. Skip for now — keep PoSE's identity as a "chunked" method. Try 3-chunk only if results are promising.
- **Curriculum:** Skip — PoSE curriculum didn't help in Phase 4 (27.7% vs 28.5% retention).

---

#### Total training compute

| Group | Runs | Time (est.) |
|-------|------|-------------|
| A: RPE-only controls | 2 | 42 min |
| B: YaRN-only baselines | 2 | 42 min |
| C: YaRN+RPE small L | 6 | 2h 6min |
| D: YaRN+RPE window-L | 4 | 1h 24min |
| E: YaRN+PoSE | 4 | 1h 24min |
| **Phase 6 subtotal** | **18** | **~6h 18min** |
| F: Expt 3 (compositional) | 2 | 42 min |
| **Grand total** | **20** | **~7h** |

With 3 parallel L40S jobs: **~2.5 hours wall time.**

---

### Experiment 2: Evaluation Matrix

Each model evaluated in two modes. Models trained with YaRN f=2 get an extra eval at f=4 (scaling up).

| Eval mode | YaRN at eval | What it tests |
|-----------|-------------|---------------|
| **Standard** | No (or matching train factor) | Raw learned performance. |
| **+YaRN f=4 eval** | Yes (f=4.0) | Does eval-time YaRN stack with what was trained? |
| **+YaRN f=2 eval** | Yes (f=2.0) | For Y2-trained: does matching factor help? |

| Group | Models | Eval modes | Total eval runs |
|-------|--------|------------|-----------------|
| A: RPE-only controls | 2 | 2 (no YaRN, +YaRN f=4) | 4 |
| B: YaRN-only | 2 | 3 (no YaRN, f=2, f=4) | 6 |
| C: YaRN+RPE small L | 6 | 2-3 | 14 |
| D: YaRN+RPE window-L | 4 | 2-3 | 10 |
| E: YaRN+PoSE | 4 | 2-3 | 10 |
| F: Expt 3 (compositional) | 2 | 3 (no YaRN, matching, f=4) | 6 |
| **Grand total** | **20** | | **~50** |

Each eval: ~30-60 min on H200 across 5 bins. Heavily parallelizable.

**Key comparisons:**
1. **YaRN at train+RPE vs RPE+YaRN at eval only** → Y4-Rc16 vs existing RPE cur L=16K+YaRN eval
2. **Train low, eval high** → Y2-* models + f=4 eval. Practical deployment story.
3. **YaRN+PoSE vs PoSE alone** → Y4-P32 vs existing PoSE fixed
4. **Best L for YaRN+RPE** → Is it small L (robustness only) or window-matching L?
5. **L relative to YaRN window** → Y4-Rc16 (small) vs Y4-Rc64 (half) vs Y4-Rc128 (full). Which regime wins?
6. **RPE small-L controls** → Does L=4K or 8K work without YaRN? Isolates YaRN's contribution.
7. **Pareto frontier** → Plot (bin-0 score, bin-4 score) for all conditions. Who dominates?

---

## High Priority: Experiment 3 — Compositional Context Extension

### Core Hypothesis

For the same total context extension, is **"small YaRN + RPE beyond the YaRN window"** better than **"large YaRN alone"**?

YaRN works by compressing RoPE frequencies. Larger factors compress more aggressively, which can degrade quality. The idea: use a modest YaRN factor for the "easy" part of the extension, then use RPE position training to push the model beyond the YaRN window — without further frequency compression.

```
Path A (standard):   YaRN f=4 alone          → 32K → 128K  (one big frequency compression)
Path B (ours):       YaRN f=2 + RPE L=128K   → 32K → 64K → 128K  (small compression + position training)
```

### Target: 4x extension (32K → 128K)

| # | ID | Train YaRN f | RPE L | How it reaches 128K | New? |
|---|-----|-------------|-------|---------------------|------|
| 1 | **Pure-Y4** | 4.0 | None | YaRN does all 4x | Already done |
| 2 | **Comp-Y2-R128** | 2.0 | 128K cur | YaRN 2x (→64K) + RPE 2x beyond (→128K) | **NEW** |
| 3 | **Comp-Y3-R128** | 3.0 | 128K cur | YaRN 3x (→96K) + RPE 1.3x beyond (→128K) | **NEW** |
| 4 | **Pure-R128** | None | 128K cur | RPE does all 4x | Already done (Phase 5) |

Curriculum for #2 and #3: `16K→48K→80K→128K→128K`

### Controls (isolate what helps)

| # | ID | Train YaRN f | RPE L | Purpose | New? |
|---|-----|-------------|-------|---------|------|
| 5 | **Y2-R64** | 2.0 | 64K cur | RPE stays WITHIN YaRN window (no beyond) | In Phase 6 Group D |
| 6 | **Y2-only** | 2.0 | None | Pure YaRN f=2 baseline | In Phase 6 Group B |

**Critical comparison:** If Comp-Y2-R128 >> Y2-R64, the "beyond window" push specifically matters. If Comp-Y2-R128 ≈ Y2-R64, just combining them is enough.

### Evaluation for Experiment 3

Each model gets 3 eval modes:

| Eval mode | What it tests |
|-----------|---------------|
| **Matching YaRN** (f=train factor) | Raw learned performance at trained extension |
| **No YaRN** | How much does the model retain without frequency scaling? |
| **Scaled-up YaRN f=4** | Triple extension: train YaRN 2x + train RPE 2x + eval YaRN scales further |

The **triple extension** eval is the most exciting test: Comp-Y2-R128 trained to handle 128K, then eval-time YaRN f=4 compresses frequencies further — could this push the model to 256K without any additional training?

### What "winning" looks like

If Comp-Y2-R128 beats Pure-Y4 at bins 3-4 (32K-128K), the narrative is:

> "Large YaRN factors degrade quality through aggressive frequency compression. We show that a modest YaRN factor (2x) combined with RPE position training beyond the YaRN window achieves superior context extension — RPE teaches positional robustness without further compressing the frequency space. This is a compositional approach to context extension that outperforms increasing the YaRN factor alone."

### Additional training runs for Experiment 3

| # | ID | YaRN f | RPE L | Curriculum | Time |
|---|-----|--------|-------|------------|------|
| 19 | **Comp-Y2-R128** | 2.0 | 128K | 16K→48K→80K→128K→128K | ~21 min |
| 20 | **Comp-Y3-R128** | 3.0 | 128K | 16K→48K→80K→128K→128K | ~21 min |

Eval: 2 models × 3 modes = 6 eval runs. Everything else reuses existing Phase 5/6 models.

### Experiment 3 Results (March 20, 2026)

Training: Both Comp-Y2-R128 and Comp-Y3-R128 trained on bin 0 (4K-8K), LoRA rank 16, 5 epochs.
Eval: Each model evaluated in 3 modes (no YaRN, matching YaRN, YaRN f=4). Baseline = existing RPE cur L=128K + YaRN f=4 at eval.

#### Comp-Y2-R128 (YaRN f=2 trained + RPE L=128K)

| Eval mode | 4K-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K |
|-----------|-------|--------|---------|---------|----------|
| No YaRN | 0.958 | 0.634 | 0.713 | 0.526 | 0.212 |
| YaRN f=2 (matching) | 0.993 | 0.693 | 0.618 | 0.592 | 0.427 |
| YaRN f=4 (scaled-up) | 0.916 | 0.658 | 0.808 | 0.641 | 0.499 |

#### Comp-Y3-R128 (YaRN f=3 trained + RPE L=128K)

| Eval mode | 4K-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K |
|-----------|-------|--------|---------|---------|----------|
| No YaRN | 0.854 | 0.572 | 0.547 | 0.558 | 0.233 |
| YaRN f=3 (matching) | 0.963 | 0.625 | 0.743 | 0.737 | 0.493 |
| YaRN f=4 (scaled-up) | 0.927 | 0.594 | 0.744 | 0.618 | 0.502 |

#### Baseline: Pure-R128 + YaRN f=4 at eval (no YaRN during training)

| Eval mode | 4K-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K |
|-----------|-------|--------|---------|---------|----------|
| YaRN f=4 | 0.783 | 0.599 | 0.731 | 0.585 | 0.537 |

#### Prior baselines (from Phase 3-5, for reference)

| Condition | 4K-8K | 8K-16K | 16K-32K | 32K-64K | 64K-128K |
|-----------|-------|--------|---------|---------|----------|
| Pure-Y4 (YaRN f=4 train+eval) | 0.891 | 0.692 | 0.619 | 0.619 | 0.441 |
| RPE cur L=128K (no YaRN) | 0.889 | 0.573 | 0.678 | 0.606 | 0.334 |
| RPE cur L=16K (no YaRN) | 0.924 | 0.939 | 0.874 | 0.688 | 0.330 |

#### Analysis

**Head-to-head at bin 4 (64K-128K) — the main question:**

| Condition | Bin 4 | Bin 0 |
|-----------|-------|-------|
| Pure-R128 + YaRN f=4 eval (no YaRN train) | **0.537** | 0.783 | 
| Comp-Y3-R128 + f=4 eval | 0.502 | 0.927 | 
| Comp-Y2-R128 + f=4 eval | 0.499 | 0.916 | 
| Comp-Y3-R128 + f=3 matching | 0.493 | 0.963 | 
| Pure-Y4 (train+eval) | 0.441 | 0.891 | 49.5% |
| Comp-Y2-R128 + f=2 matching | 0.427 | **0.993** | 
| RPE cur L=128K (no YaRN at all) | 0.334 | 0.889 | 

**Finding 1: Compositional models beat Pure-Y4, but NOT the eval-only baseline.**
Comp-Y2-R128 + f=4 gets 0.499 at bin 4 vs Pure-Y4's 0.441 — a clear win (+0.058). But the simplest approach — just training RPE L=128K without any YaRN, then slapping YaRN f=4 on at eval — scores 0.537, beating everything. Training with YaRN active actually hurts bin 4 by ~0.04 compared to not training with it.

**Finding 2: Training with YaRN improves short-context scores dramatically.**
The baseline (no YaRN train) gets only 0.783 at bin 0. The compositional models get 0.916-0.993. So there's a clear tradeoff: YaRN during training helps bins 0-2 but slightly hurts bin 4. The models trained with YaRN learn better short-context behavior but become slightly less flexible at extreme extrapolation.

**Finding 3: Comp-Y3 with matching f=3 has the highest bin 3 score we've ever seen (0.737).**
This beats every prior condition at the 32K-64K range. The matching-factor eval seems to be a sweet spot for Comp-Y3 specifically — better than scaling up to f=4 at bin 3 (0.737 vs 0.618). But this advantage disappears at bin 4 where f=4 takes over (0.502 > 0.493).

**Finding 4: No YaRN eval reveals what RPE actually learned.**
Without YaRN, Comp-Y2 scores 0.958/0.634/0.713/0.526/0.212. That bin 0 of 0.958 is near-perfect — the model learned the task well. But bin 4 collapses to 0.212, confirming that RPE alone (even trained with YaRN frequencies) can't reach 128K. YaRN at eval is still essential for the longest contexts.

**Finding 5: Train-low eval-high works.**
Comp-Y2 (trained f=2) + eval f=4 gets 0.808 at bin 2, beating Pure-Y4's 0.619 and even the baseline's 0.731. This validates the practical deployment story: train cheaply at f=2, scale up at inference.

**Bottom line for the compositional hypothesis:**
Mixed. The compositional approach (small YaRN + RPE beyond window) beats monolithic YaRN (Pure-Y4) as hoped. But it doesn't beat the even simpler strategy of training RPE without YaRN and adding YaRN at eval time. The "compositional" benefit comes from RPE + eval-time YaRN, not from training with YaRN.

**What this means for Phase 6:**
The Phase 6 experiments (YaRN active during training + RPE/PoSE) may show a similar pattern — YaRN during training helps short context but slightly hurts the longest extrapolation. The key comparison will be Y4-Rc16 vs existing RPE cur L=16K + YaRN eval. If the same pattern holds, eval-only YaRN remains the best strategy, and Phase 6's value shifts to understanding WHY training with YaRN hurts extrapolation.

---

### Follow-up (if Experiment 3 succeeds): 8x extension

If compositional extension works at 4x, test at 8x (32K → 256K):

| ID | Train YaRN f | RPE L | How it reaches 256K |
|-----|-------------|-------|---------------------|
| **Pure-Y8** | 8.0 | None | YaRN does all 8x |
| **Comp-Y4-R256** | 4.0 | 256K cur | YaRN 4x (→128K) + RPE 2x (→256K) |
| **Comp-Y2-R256** | 2.0 | 256K cur | YaRN 2x (→64K) + RPE 4x (→256K) |

Requires extending MRCR eval data to 256K+ bins or using BABILong (which goes to 1M). Save for follow-up.

---

## Medium Priority: BABILong QA3

Run the best-performing method from MRCR on BABILong QA3 (3-hop reasoning). This tests whether our findings generalize beyond retrieval to multi-hop reasoning.

**Plan:** After Phase 6 training + eval, take the top 2-3 conditions and run on BABILong QA1/QA2/QA3 at lengths 4K-128K.

---

## Medium Priority: LongBench v2 Length Bins

Check if LongBench v2 has proper length splits. Quick literature/dataset check. Low effort.

---

## Low Priority: OOD Detection for Positional Encodings

The question: can we detect WHEN a model treats a position as out-of-distribution?

Ideas:
- Look at attention entropy at different positions — OOD positions might have high entropy (uniform attention = "confused")
- Compare RoPE key/query dot products at in-distribution vs OOD positions
- Perplexity spike analysis per-token
- Retrieval heads (Zhang et al. 2024) — specific heads responsible for long-range retrieval. Do these heads degrade at OOD positions?

This is a research direction, not an immediate experiment. Park it.

---

## Phase 6 Status Tracker (updated March 20, 2026)

### Training Status

| # | ID | Checkpoint | Status |
|---|-----|-----------|--------|
| 1 | R-c4 | rpe_cur_L4k | DONE |
| 2 | R-c8 | rpe_cur_L8k | DONE |
| 3 | Y2 | yarn2 | QUEUED |
| 4 | Y3 | yarn3 | QUEUED |
| 5 | Y2-Rc4 | y2_rpe_cur_L4k | DONE |
| 6 | Y2-Rc8 | y2_rpe_cur_L8k | DONE |
| 7 | Y2-Rc16 | y2_rpe_cur_L16k | DONE |
| 8 | Y4-Rc4 | y4_rpe_cur_L4k | DONE |
| 9 | Y4-Rc8 | y4_rpe_cur_L8k | QUEUED |
| 10 | Y4-Rc16 | y4_rpe_cur_L16k | QUEUED |
| 11 | Y2-Rc32 | y2_rpe_cur_L32k | DONE |
| 12 | Y2-Rc64 | y2_rpe_cur_L64k | DONE |
| 13 | Y4-Rc64 | y4_rpe_cur_L64k | QUEUED |
| 14 | Y4-Rc128 | y4_rpe_cur_L128k | QUEUED |
| 15 | Y2-P16 | y2_pose_16k | DONE |
| 16 | Y2-P32 | y2_pose_32k | DONE |
| 17 | Y4-P16 | y4_pose_16k | QUEUED |
| 18 | Y4-P32 | y4_pose_32k | QUEUED |

### Eval Status

27 eval jobs submitted for the 10 completed models. Script: `phase6/hpc/submit_all_evals.sh`

| Model | no YaRN | YaRN f=2 | YaRN f=4 |
|-------|---------|----------|----------|
| rpe_cur_L4k | SUBMITTED | — | SUBMITTED |
| rpe_cur_L8k | SUBMITTED | — | SUBMITTED |
| y2_rpe_cur_L4k | SUBMITTED | SUBMITTED | SUBMITTED |
| y2_rpe_cur_L8k | SUBMITTED | SUBMITTED | SUBMITTED |
| y2_rpe_cur_L16k | SUBMITTED | SUBMITTED | SUBMITTED |
| y2_rpe_cur_L32k | SUBMITTED | SUBMITTED | SUBMITTED |
| y2_rpe_cur_L64k | SUBMITTED | SUBMITTED | SUBMITTED |
| y2_pose_16k | SUBMITTED | SUBMITTED | SUBMITTED |
| y2_pose_32k | SUBMITTED | SUBMITTED | SUBMITTED |
| y4_rpe_cur_L4k | SUBMITTED | — | SUBMITTED |

### Pending Evals (after remaining training completes)

yarn2, yarn3, y4_rpe_cur_L8k, y4_rpe_cur_L16k, y4_rpe_cur_L64k, y4_rpe_cur_L128k, y4_pose_16k, y4_pose_32k

### Key comparison to watch

**Does RPE help YaRN?** → Y4-Rc16 (YaRN f=4 + RPE L=16K) vs Pure-Y4 (0.441 at bin 4)
**Does training with YaRN help RPE?** → Y2-Rc16 + f=4 eval vs RPE cur L=16K + f=4 eval-only (no prior result for this!)
**Best L for YaRN+RPE?** → Compare L=4K vs 8K vs 16K vs 32K vs 64K (Y2 series)

---

## Claude's Thoughts

**Most promising condition (Phase 6):** Y4-Rc16 (YaRN f=4 + RPE curriculum L=16K). Combines our best RPE setting with YaRN active during training.

**Most exciting condition (Expt 3):** Comp-Y2-R128 (YaRN f=2 + RPE L=128K). If this beats Pure-Y4, we have a publishable finding: compositional extension > monolithic YaRN scaling. The triple-extension eval (train f=2 + RPE 128K, eval f=4) could push to 256K without additional training.

**Biggest risk:** YaRN + RPE at training might cause training instability — YaRN already stretches the frequency space, and RPE adds random position gaps on top. The curriculum should help (start with small L), but watch for gradient norm spikes in training logs.

**On the "train low, eval high" strategy (Y2 train → Y4 eval):** This is really interesting and could be the most practical result. If we can train with a modest YaRN factor (cheaper, more stable) and then scale up at inference, that's a strong deployment story. This is exactly how the original YaRN paper frames it — but we'd be the first to combine it with RPE.

**On PoSE chunk variations:** The most meaningful PoSE variation is target_length, not n_chunks. Changing n_chunks from 2 to 3 would make PoSE more RPE-like (more scattered position gaps), which defeats the purpose of comparing them as distinct methods. Keep n_chunks=2 to maintain PoSE's identity as a "chunked" approach.

**On analysis going forward:** Tables with 3 columns (condition, bin-0, bin-4) and one sentence per finding. No 800-line walkthrough. The professor wants to see the result and the insight, not the methodology of how you computed it.
