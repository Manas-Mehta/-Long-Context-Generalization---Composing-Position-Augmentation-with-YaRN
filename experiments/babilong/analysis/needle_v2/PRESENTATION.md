# Context Extension Needs Position Regularization

## Setup
**Task**: BABILong QA3 — multi-hop spatial reasoning ("Where was the football before the garden?"). 3 objects × 6 rooms, answer is one word. Requires chaining 3 facts buried in noise text.
**Data**: 227 selected hard samples × 3 needle zones (beginning/middle/end of document) × 8 length bins (1K–128K tokens) = 5,021 evaluation cells per model.
**Noise**: PG19 (19th-century novels). Needle position controlled: beg = first 33%, mid = middle 33%, end = last 33%.
**Models** (all evaluated with YaRN f=4 for context extension to 128K):
- **lora_base** — LoRA finetune only, no YaRN at train, no RPE
- **y2_base** — LoRA + YaRN f=2 at train (standard context-extension recipe)
- **y2_rpe** — LoRA + YaRN f=2 + RPE curriculum L=16K at train (position randomization added)

All three share: Qwen2.5-7B-Instruct base, same 60-story training set, 75 steps, checkpoint-2000, rank-16 LoRA.

## Part 1 — YaRN Fine-Tuning Alone Is Net Harmful

**Overall accuracy** (sample-weighted across all 5,021 cells):
|model|accuracy|
|---|---|
|lora_base (no YaRN train)|0.772|
|y2_base (YaRN train)|**0.727** ← worst|
|y2_rpe (YaRN + RPE train)|**0.799** ← best|

The standard context-extension recipe (fine-tune with YaRN) is **worse than not fine-tuning at all**. Both y2_base and lora_base are evaluated identically (YaRN f=4) — the only difference is what happened during the 75-step finetune.

**Cell-level evidence** — classify every (zone, bin, sample) cell by which models got it right:
|category|description|count|%|
|---|---|---|---|
|All 3 correct|no model struggles|2,833|56.4%|
|All 3 wrong|genuinely hard|321|6.4%|
|YaRN broke, RPE fixed|lora✓ y2✗ rpe✓|494|9.8%|
|YaRN broke, RPE didn't fix|lora✓ y2✗ rpe✗|254|5.1%|
|RPE uniquely solves|lora✗ y2✗ rpe✓|304|6.1%|
|RPE uniquely fails|lora✓ y2✓ rpe✗|293|5.8%|
|YaRN gained, RPE kept|lora✗ y2✓ rpe✓|382|7.6%|
|YaRN only|lora✗ y2✓ rpe✗|140|2.8%|

**748 cells** where lora gets it right but YaRN training broke it. Only **522** go the other direction. **Net: YaRN training destroyed 226 more cells than it created.**

## Part 2 — The Damage: Error-Mode Collapse

When y2_base makes errors, they cluster into two default rooms. When lora_base or y2_rpe make errors, they are spread nearly uniformly.
|model|error entropy (% of uniform)|top-2 wrong rooms (% of errors)|
|---|---|---|
|lora_base|98.5%|kitchen 24% + bedroom 19% = 43%|
|y2_base|**93.0%**|**office 28% + hallway 27% = 55%**|
|y2_rpe|97.8%|office 24% + hallway 22% = 46%|

At long bins (32K+), y2_base drops further to **89–91%** of uniform. lora_base stays at 95–99%. y2_rpe stays at 96–98%.

**Interpretation**: YaRN training with fixed position encodings taught the model a shortcut — "when confused, predict office or hallway." This is positional overfitting: the model learned associations between specific YaRN f=2 position patterns and room outputs. At eval with f=4 (different distortion), the shortcut fires on wrong inputs. The untrained model (lora_base) never learned these shortcuts. The position-randomized model (y2_rpe) couldn't learn them.

## Part 3 — Position Randomization as Regularization

RPE randomizes position IDs during training. The model sees the same content at many different positions, so it cannot learn "at position X, predict Y." It is forced toward content-based attention.

**RPE rescues 66% of YaRN damage** — verified on the exact same sample-cells:
|bin|cells broken by YaRN|RPE rescues|rate|
|---|---|---|---|
|1k|51|36|71%|
|4k|77|47|61%|
|16k|99|73|74%|
|32k|104|71|68%|
|64k|109|63|58%|
|128k|151|96|64%|

**Net ledger** (pairwise cell-level wins minus losses):
|comparison|wins|losses|net|
|---|---|---|---|
|RPE vs y2_base|798|433|**+365**|
|RPE vs lora_base|686|547|**+139**|
|y2_base vs lora_base|522|748|**−226**|

RPE is net positive against both baselines. y2_base is net negative against the untrained baseline.

**Retention** (accuracy at 128K / accuracy at 4K — how much short-context capability survives at maximum length):
|zone|lora_base|y2_base|y2_rpe|
|---|---|---|---|
|beg (hardest — early needle)|74.6%|72.5%|**89.8%**|
|end (easiest — late needle)|78.3%|74.1%|**94.9%**|

Beg-zone retention is the critical number: pure long-context capability with no recency advantage. YaRN makes it worse (72.5% < 74.6%). Adding RPE makes it dramatically better (89.8%).

## Part 4 — Position Randomization as Data Augmentation

Beyond preventing YaRN damage, RPE creates capability neither baseline has.

**304 cells** where both lora and y2_base fail but RPE succeeds. These cannot be explained by "preventing YaRN overfitting" — lora never had YaRN overfitting and still fails. These cluster at beg/mid zones (230 of 304) and long bins (63 at 128K, 48 at 64K) — exactly where long-distance content retrieval matters most.

This is analogous to data augmentation in vision: random crops don't just prevent overfitting to specific pixel positions — they teach genuinely better spatial features. Position randomization doesn't just prevent positional overfitting — it teaches attention patterns that work across positions.

**Per-sample distribution** (out of 227 samples, across all 24 cells per model):
|fraction of cells correct|lora_base|y2_base|y2_rpe|
|---|---|---|---|
|0–25% (nearly always wrong)|12|16|5|
|75–100% (nearly always right)|153|134|154|
|100% (always right)|32|32|**45**|

y2_rpe has the most always-right samples (45 vs 32) and fewest nearly-always-wrong (5 vs 12/16). The improvement is not concentrated in a few lucky cells — it shifts the whole distribution.

## Part 5 — Recency Bias

**Same-sample recency test**: for each model and bin, count samples that the model gets right at end zone but wrong at beg zone. Same question, same length, only the position of the facts changes. Higher = more recency-biased.
|bin|N|lora_base|y2_base|y2_rpe|
|---|---|---|---|---|
|2k|202|35 (17.3%)|28 (13.9%)|**19 (9.4%)**|
|4k|227|36 (15.9%)|40 (17.6%)|**32 (14.1%)**|
|8k|227|41 (18.1%)|44 (19.4%)|**37 (16.3%)**|
|32k|227|46 (20.3%)|57 (25.1%)|**40 (17.6%)**|
|64k|227|**46 (20.3%)**|56 (24.7%)|75 (33.0%)|
|128k|227|56 (24.7%)|55 (24.2%)|**42 (18.5%)**|

At 128K, RPE has 42 recency-biased samples vs 56 (lora) and 55 (y2_base) — 14 fewer than the best baseline. The 64K bin goes the other direction (75 — worst of all three). The pattern is present at most bins but not universal.

**Samples correct at BOTH end AND beg** (position-invariant success):
|bin|lora_base|y2_base|y2_rpe|
|---|---|---|---|
|32k|146|134|**165**|
|64k|130|122|126|
|128k|106|91|**145**|

At 128K, RPE gets 145 samples right at both zones vs 106 (lora) and 91 (y2_base). 39 more position-invariant successes than the next best.

**Concrete examples at 128K** — samples where both baselines get end✓ beg✗ (recency-biased) but RPE gets both right:
|sample|question|target|beg: lora / y2 / **rpe**|end: all three|
|---|---|---|---|---|
|idx 132|milk before office?|bathroom|garden / garden / **bathroom**|all correct|
|idx 192|football before bathroom?|bedroom|garden / kitchen / **bedroom**|all correct|
|idx 369|milk before garden?|kitchen|bedroom / hallway / **kitchen**|all correct|
|idx 398|apple before kitchen?|bathroom|hallway / office / **bathroom**|all correct|
|idx 586|football before garden?|bathroom|kitchen / hallway / **bathroom**|all correct|

Both baselines produce wrong rooms at beg (facts 100K+ tokens from the question). RPE finds the correct answer regardless of position.

## Part 6 — Limitations and What's Missing

**RPE is not a silver bullet.** 293 cells (5.8%) where both baselines are right and RPE is wrong. 140 cells where YaRN-only gained something and RPE lost it. The 64K recency anomaly. These are real costs.

**What we don't have:**
- Attention maps confirming content-based vs position-based attention
- PoSE in this same experiment (would show whether the effect is specific to RPE or general to position randomization)
- Reasoning traces (currently max_new_tokens=10, single-word output — no insight into what the model attends to)

**What we do have:**
- YaRN fine-tuning alone is net harmful — concrete, surprising, reproducible
- Position randomization (RPE) makes YaRN training net positive — moderate but consistent (+365 net vs y2_base)
- The improvement includes both damage prevention (494 rescued cells) and new capability (304 unique solves)
- Error-mode collapse in YaRN-only training, absent in position-randomized training

## Summary

Training with fixed position encodings (YaRN f=2) causes positional overfitting: the model learns position-dependent shortcuts that break at eval time (YaRN f=4). Result: worse than not fine-tuning at all.

Position randomization during training (RPE) acts as both **regularization** (prevents 66% of YaRN damage, eliminates error-mode collapse) and **data augmentation** (304 cells solved that neither baseline can, 45 always-right samples vs 32). Like random cropping in vision, it doesn't fix one specific failure — it teaches more robust representations that generalize across positions.

The practical takeaway: if you extend context via position-encoding scaling, you likely need position regularization during training. Without it, you may be worse off than not fine-tuning at all.
