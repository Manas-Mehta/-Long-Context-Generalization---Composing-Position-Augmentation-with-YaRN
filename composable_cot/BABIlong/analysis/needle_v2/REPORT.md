# Needle-Position Eval v2 — Deep Analysis

**Models** (all eval'd with YaRN f=4):
- `lora_base` — LoRA finetune, no YaRN at train, no RPE
- `y2_base` — LoRA + YaRN f=2 at train
- `y2_rpe_cur_L16k` — LoRA + YaRN f=2 + RPE curriculum L=16K at train

**Sample pool**: 227 stories (207 hard multi-entry + 20 single-entry reference). Each placed in PG19 noise at 3 zones × 8 length bins → 24 cells per model. Short bins (1k, 2k) drop samples whose own story already exceeds the bin (effective N: 1k=104–116, 2k=202–203, 4k+ = 227 per cell).

Total: **15,063 model evaluations**.

---

# Angle 1 — YaRN Training Requires Positional Regularization

**Claim**: Training with YaRN (fixed position-encoding distortion) causes positional overfitting that *damages* long-context generalization. RPE acts as a positional regularizer — analogous to dropout — that prevents this damage.

## Evidence A: YaRN training alone is net harmful

`y2_base` (YaRN-only training) is **worse overall** than `lora_base` (no YaRN training at all). Both eval'd identically with YaRN f=4 — the only difference is what happened during the 75-step finetune.

Every `(zone, bin, sample)` cell in the eval (5,021 of them) falls into one of 7 categories based on the 3 models' correctness:

| Category | Description | Count | % |
|----------|-------------|------:|---:|
| **All 3 correct** | Unanimous — no model struggle | 2,833 | 56.4% |
| **All 3 wrong** | Genuinely hard — no model solves | 321 | 6.4% |
| **YaRN broke, RPE fixed** | lora✓ y2_base✗ y2_rpe✓ | **494** | **9.8%** |
| **YaRN broke, RPE didn't fix** | lora✓ y2_base✗ y2_rpe✗ | 254 | 5.1% |
| **RPE uniquely solves** | lora✗ y2_base✗ y2_rpe✓ | 304 | 6.1% |
| **RPE uniquely fails** | lora✓ y2_base✓ y2_rpe✗ | 293 | 5.8% |
| **Other combinations** | y2_base✓ lora✗ combos, etc. | 522 | 10.4% |

The **"YaRN broke, RPE fixed"** category (494 cells, 9.8%) contains cells where:
- The untrained baseline (`lora_base`) gets the answer right
- YaRN training broke it (`y2_base` wrong)
- Adding RPE to the same YaRN training fixed it back (`y2_rpe` right)

These are the **exact same sample-cells** — same `(original_idx, zone, bin)` tuple checked with a 3-way joint filter. Not aggregate statistics.

Adding the 254 "broke and not fixed" cells: **748 total cells damaged by YaRN training** (lora✓ y2_base✗). Only 522 go the other direction (y2_base✓ lora✗). **Net: −226 cells. YaRN training destroyed more capability than it created.**

Per-bin, RPE's rescue rate of the YaRN damage:

| bin | cells broken by YaRN | RPE rescues | rescue rate |
|-----|---------------------|-------------|-------------|
| 1k  | 51  | 36 | 71% |
| 2k  | 72  | 50 | 69% |
| 4k  | 77  | 47 | 61% |
| 8k  | 85  | 58 | 68% |
| 16k | 99  | 73 | 74% |
| 32k | 104 | 71 | 68% |
| 64k | 109 | 63 | 58% |
| 128k| 151 | 96 | 64% |

RPE rescues 58–74% of YaRN damage at every bin. This is uniform, not concentrated at long context — the regularization effect operates during training, not at a specific eval length.

## Evidence B: YaRN damage manifests as error-mode collapse

When `y2_base` makes errors, they are not randomly distributed — they cluster into 2 default rooms. This is the signature of a positional shortcut: "when confused, predict office or hallway."

Error entropy (higher = more spread = healthier):
| model | error entropy | % of uniform | top-2 wrong rooms (% of errors) |
|-------|--------------|-------------|------|
| `lora_base` | 2.547 | **98.5%** | kitchen (24%) + bedroom (19%) = 43% |
| `y2_base` | 2.404 | **93.0%** | office (28%) + hallway (27%) = **55%** |
| `y2_rpe` | 2.529 | **97.8%** | office (24%) + hallway (22%) = 46% |

At long bins (32k+), `y2_base` error entropy drops further to **89–91%** of uniform, while `lora_base` stays at 95–99% and `y2_rpe` stays at 96–98%.

**Interpretation**: During training with YaRN f=2 (fixed position distortion), the model learned to associate specific position-encoding patterns with room outputs. At eval with f=4 (a different distortion), these associations fire on wrong inputs → the model defaults to "office" or "hallway" when confused. RPE randomized positions during training, so this shortcut never formed.

## Evidence C: Retention — RPE preserves what YaRN destroys

Accuracy retention from 4k → 128k (% of short-context accuracy preserved at maximum length):

| zone | `lora_base` | `y2_base` | `y2_rpe` |
|------|-----------|---------|--------|
| **beg** (hardest — early needle, no recency help) | 74.6% | **72.5%** | **89.8%** |
| end  (easiest — late needle, recency helps) | 78.3% | 74.1% | **94.9%** |

Beg-zone retention is the critical number. It measures pure long-context capability with no recency advantage. YaRN training makes it *worse* (72.5% < 74.6%). Adding RPE makes it dramatically better (89.8%).

## Mechanism (hypothesis consistent with data)

During training with YaRN f=2, the model sees consistent position encodings on short data (4K-8K). It learns positional shortcuts: "when position encodings look like X, predict room Y." At eval with f=4 (novel positions), the shortcuts break → worse than the untrained model.

RPE randomizes position IDs during training → the model **cannot** learn "at position X, predict Y." It is forced toward **content-based attention** (actually reading the bAbI facts). Content-based attention transfers across position-encoding schemes; position-based shortcuts don't.

This is analogous to **dropout**: add noise to prevent overfitting to specific features. RPE is "positional dropout" — noise on positions prevents overfitting to positional features.

**What we'd need to fully confirm**: attention maps showing y2_base attends to positions vs y2_rpe attending to content tokens. We don't have those yet.

---

# Angle 2 — RPE Reduces Recency Bias (Same-Sample Evidence)

**Claim**: RPE reduces recency bias on the same samples where both baselines are recency-biased: they solve a question when the facts are at the end of the document but fail when the same facts are at the beginning. RPE eliminates this failure on a substantial fraction.

## Evidence A: Recency-biased samples (same sample, end✓ but beg✗)

For each model and bin, take every sample that appears in both `beg` and `end` zones. Count how many the model gets right at end but wrong at beg — **the same question, same noise length, only the position of the facts changes**. Higher count = more recency-biased.

| bin | N paired | `lora_base` | `y2_base` | `y2_rpe` |
|-----|----------|-----------|---------|--------|
| 1k  | 108 | 6 (5.6%) | 9 (8.3%) | 8 (7.4%) |
| 2k  | 202 | 35 (17.3%) | 28 (13.9%) | **19 (9.4%)** |
| 4k  | 227 | 36 (15.9%) | 40 (17.6%) | **32 (14.1%)** |
| 8k  | 227 | 41 (18.1%) | 44 (19.4%) | **37 (16.3%)** |
| 16k | 227 | **26 (11.5%)** | 36 (15.9%) | 34 (15.0%) |
| 32k | 227 | 46 (20.3%) | 57 (25.1%) | **40 (17.6%)** |
| 64k | 227 | **46 (20.3%)** | 56 (24.7%) | 75 (33.0%) ← RPE worst |
| 128k| 227 | 56 (24.7%) | 55 (24.2%) | **42 (18.5%)** ← RPE best |

**At 128k**: RPE has 42 recency-biased samples vs 56 (lora) and 55 (y2_base) — 14 fewer than the best baseline. At most bins RPE has the fewest. The 64k bin is a real counterexample (75 — worst of all three).

## Evidence B: Samples correct at BOTH end AND beg (position-invariant success)

A stricter measure: how many of the 227 paired samples does each model get right at **both** end and beg simultaneously?

| bin | `lora_base` | `y2_base` | `y2_rpe` |
|-----|-----------|---------|--------|
| 16k | 173 | 156 | 172 |
| 32k | 146 | 134 | **165** |
| 64k | 130 | 122 | 126 |
| 128k| 106 | 91 | **145** |

**At 128k, RPE gets 145 samples correct at both zones vs 106 (lora) and 91 (y2_base).** That's 39 more position-invariant successes than `lora_base`, and 54 more than `y2_base`.

## Evidence C: RPE fixes recency-biased failures of both baselines

The strictest test: take samples where **both** `lora_base` AND `y2_base` are recency-biased (end✓ beg✗ on the same sample). These are samples where recency bias is robust across training methods. Does RPE fix the beg-zone failure?

| bin | both baselines recency-biased | RPE fixes beg | RPE also fails beg |
|-----|------------------------------|--------------|-------------------|
| 16k | 8 | **5 (62%)** | 3 (38%) |
| 32k | 19 | 7 (37%) | 12 (63%) |
| 64k | 13 | 5 (38%) | 8 (62%) |
| 128k| 17 | **11 (65%)** | 6 (35%) |

**At 128k**: 17 samples defeated both baselines via recency bias. RPE eliminated the beg-zone failure on **11 of 17** (65%). These are verified same-sample comparisons.

### Concrete examples (128k, both baselines: end✓ beg✗, RPE: beg✓ end✓)

| idx | question | target | beg: lora / y2 / **rpe** | end: lora / y2 / **rpe** |
|-----|----------|--------|--------------------------|--------------------------|
| 99  | milk before bedroom? | bathroom | garbage / hallway / **bathroom** | bathroom / bathroom / **bathroom** |
| 132 | milk before office? | bathroom | garden / garden / **bathroom** | bathroom / bathroom / **bathroom** |
| 192 | football before bathroom? | bedroom | garden / kitchen / **bedroom** | bedroom / bedroom / **bedroom** |
| 369 | milk before garden? | kitchen | bedroom / hallway / **kitchen** | kitchen / kitchen / **kitchen** |
| 392 | milk before garden? | office | bedroom / bathroom / **office** | office / office / **office** |
| 398 | apple before kitchen? | bathroom | hallway / office / **bathroom** | bathroom / bathroom / **bathroom** |
| 586 | football before garden? | bathroom | kitchen / hallway / **bathroom** | bathroom / bathroom / **bathroom** |

In every case: lora and y2_base answer correctly at end (facts near the question) but produce a wrong room at beg (facts 100K+ tokens away from the question). RPE gets the correct answer at **both** positions — it can retrieve the facts regardless of where they sit in the 128K-token document.

Note that lora and y2_base's wrong answers at beg are **different from each other** (garden vs hallway, garden vs kitchen, etc.) — they're not even defaulting to the same wrong room. They're genuinely lost. RPE still finds the right answer.

## Limitations

- The 64k bin shows RPE with HIGHER recency flip rate (37.3%) than both baselines. The pattern is strong at 128k and 32k but not universal.
- The N for "both baselines recency-biased" is small (8–19 per bin). More samples would strengthen this.
- We cannot distinguish "RPE has better beg-zone accuracy because it has better attention" from "RPE has better beg-zone accuracy because it has better everything." The beg-zone improvement could be a side effect of generally better accuracy rather than specifically anti-recency. The recency flip rate (a ratio) partially controls for this, but is not a perfect control.

---

All raw tables and figures live in `analysis/needle_v2/{tables,figures}/`. Master long-format CSV: `master.csv` (15,063 rows).

---

## 1 — Headline: per-cell accuracy (raw)

### lora_base
| zone | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|------|----|----|----|----|-----|-----|-----|------|
| beg | 0.897 | 0.783 | 0.797 | 0.736 | 0.811 | 0.714 | 0.643 | 0.595 |
| mid | 0.913 | 0.866 | 0.775 | 0.850 | **0.560** | 0.758 | 0.709 | 0.546 |
| end | 0.907 | 0.906 | 0.912 | 0.863 | 0.877 | 0.846 | 0.775 | 0.714 |

### y2_base
| zone | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|------|----|----|----|----|-----|-----|-----|------|
| beg | 0.802 | 0.798 | 0.753 | 0.718 | 0.727 | 0.648 | 0.599 | 0.546 |
| mid | 0.798 | 0.787 | 0.745 | 0.736 | **0.555** | 0.727 | 0.656 | 0.476 |
| end | 0.796 | 0.856 | 0.868 | 0.872 | 0.846 | 0.841 | 0.784 | 0.643 |

### y2_rpe_cur_L16k
| zone | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|------|----|----|----|----|-----|-----|-----|------|
| beg | 0.836 | 0.837 | 0.780 | 0.775 | 0.811 | 0.766 | 0.608 | 0.700 |
| mid | 0.875 | 0.822 | 0.841 | 0.855 | 0.745 | 0.740 | 0.727 | 0.568 |
| end | 0.861 | 0.866 | 0.868 | 0.886 | 0.907 | 0.903 | 0.886 | 0.824 |

**Sample-weighted overall** (across all 5,021 evaluations per model):
| model | overall accuracy |
|------|-----|
| `lora_base` | 0.7716 |
| `y2_base` | **0.7265** ← worst |
| `y2_rpe_cur_L16k` | **0.7992** ← best |

**Cross-model summary (no within-model deltas):**
- `y2_rpe_cur_L16k` is the highest-scoring model overall (+2.8pp over `lora_base`, +7.3pp over `y2_base`).
- `y2_base` is the worst-overall on this hard-sample subset — YaRN-only training did not match the LoRA-only baseline. This is itself a finding: it is not the case that "any continued training helps."
- All three models show the same shape: end-zone > beg/mid-zone, with the gap growing at long bins.
- **Anomaly: mid-16k drops for ALL three models** (0.560 / 0.555 / 0.745). Worth investigating — could be a regenerated-noise artifact specific to that cell.

---

## 2 — Cross-model deltas, per cell

(Positive = first model better. Each cell is a same-(zone,bin) comparison.)

### `y2_rpe_cur_L16k` − `lora_base`
| zone | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|------|----|----|----|----|-----|-----|-----|------|
| beg | −0.060 | +0.054 | −0.018 | +0.040 | 0.000 | +0.053 | −0.035 | +0.106 |
| mid | −0.038 | −0.045 | +0.066 | +0.004 | **+0.185** | −0.018 | +0.018 | +0.022 |
| end | −0.046 | −0.040 | −0.044 | +0.022 | +0.031 | +0.057 | +0.110 | +0.110 |

- 16/24 cells favor RPE; 8/24 favor `lora_base`.
- **`lora_base` wins at all short bins (1k, 2k)** in beg/end zones, and again at end_4k (−0.044). RPE gives nothing at short context.
- **RPE's biggest wins are at end_64k, end_128k, beg_128k, mid_16k** (+0.10 to +0.19).
- Beg-zone 16k is exactly equal (0.811 vs 0.811).

### `y2_rpe_cur_L16k` − `y2_base`
| zone | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|------|----|----|----|----|-----|-----|-----|------|
| beg | +0.034 | +0.039 | +0.026 | +0.057 | +0.084 | +0.119 | +0.009 | **+0.154** |
| mid | +0.077 | +0.035 | +0.097 | +0.119 | **+0.189** | +0.013 | +0.071 | +0.092 |
| end | +0.065 | +0.010 | 0.000 | +0.013 | +0.062 | +0.062 | +0.101 | **+0.181** |

- **23/24 cells favor RPE** (one tie at end_4k). RPE on top of the same YaRN-trained init is a clean win — never net-negative across the whole grid.

### `y2_base` − `lora_base`
| zone | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|------|----|----|----|----|-----|-----|-----|------|
| beg | −0.095 | +0.015 | −0.044 | −0.018 | −0.084 | −0.066 | −0.044 | −0.049 |
| mid | −0.115 | −0.079 | −0.031 | −0.115 | −0.004 | −0.031 | −0.053 | −0.071 |
| end | −0.111 | −0.050 | −0.044 | +0.009 | −0.031 | −0.004 | +0.009 | −0.071 |

- **20/24 cells favor `lora_base`** over `y2_base`. The YaRN-train-then-YaRN-eval pipeline alone underperforms the no-YaRN-train baseline (also eval'd with YaRN f=4) on this hard subset.

---

## 3 — Sample-level paired flips (`y2_rpe_cur_L16k` vs `lora_base`)

For each cell, of N samples both models judged: how many did only-A win, only-B win, both win, both lose. Net = A_only − B_only.

| zone | bin | n | rpe_only | lora_only | both | neither | net |
|------|-----|---|----------|-----------|------|---------|------|
| beg | 1k | 116 | 7 | 14 | 90 | 5 | **−7** |
| beg | 2k | 203 | 29 | 18 | 141 | 15 | +11 |
| beg | 4k | 227 | 24 | 28 | 153 | 22 | −4 |
| beg | 8k | 227 | 36 | 27 | 140 | 24 | +9 |
| beg | 16k | 227 | 27 | 27 | 157 | 16 | 0 |
| beg | 32k | 227 | 37 | 25 | 137 | 28 | +12 |
| beg | 64k | 227 | 32 | 40 | 106 | 49 | **−8** |
| beg | 128k | 227 | 54 | 30 | 105 | 38 | **+24** |
| mid | 1k | 104 | 5 | 9 | 86 | 4 | −4 |
| mid | 2k | 202 | 15 | 24 | 151 | 12 | −9 |
| mid | 4k | 227 | 36 | 21 | 155 | 15 | +15 |
| mid | 8k | 227 | 26 | 25 | 168 | 8 | +1 |
| mid | 16k | 227 | 59 | 17 | 110 | 41 | **+42** |
| mid | 32k | 227 | 30 | 34 | 138 | 25 | −4 |
| mid | 64k | 227 | 37 | 33 | 128 | 29 | +4 |
| mid | 128k | 227 | 44 | 39 | 85 | 59 | +5 |
| end | 1k | 108 | 5 | 10 | 88 | 5 | −5 |
| end | 2k | 202 | 11 | 19 | 164 | 8 | −8 |
| end | 4k | 227 | 13 | 23 | 184 | 7 | −10 |
| end | 8k | 227 | 21 | 16 | 180 | 10 | +5 |
| end | 16k | 227 | 24 | 17 | 182 | 4 | +7 |
| end | 32k | 227 | 26 | 13 | 179 | 9 | +13 |
| end | 64k | 227 | 42 | 17 | 159 | 9 | +25 |
| end | 128k | 227 | 46 | 21 | 141 | 19 | **+25** |

- **Cells where RPE has a clear sample-level disadvantage** (net ≤ −5): beg_1k, beg_64k, mid_1k, mid_2k, end_1k, end_2k, end_4k.
  - This is the short-bin, beg/end pattern. RPE costs you on easy short-context end-of-doc questions.
- **Cells where RPE has a clear sample-level advantage** (net ≥ +10): beg_2k, beg_32k, beg_128k, mid_4k, mid_16k, end_32k, end_64k, end_128k.
- **Long bins flip the most samples in BOTH directions** — e.g., 128k_beg has 54 RPE-wins AND 30 LoRA-wins (84 samples disagree out of 227). At 128k the models behave very differently sample-by-sample, even when overall accuracy is similar.

---

## 4 — Stratification by tier

### `single_entry_ref` (n=20 stories, easy — 1 fact statement)
All three models are at or near ceiling everywhere; minimum cell ≥ 0.70:
- `lora_base` cell range: 0.80 – 1.00, mean ≈ 0.92
- `y2_base` cell range: 0.70 – 1.00, mean ≈ 0.89
- `y2_rpe_cur_L16k` cell range: 0.85 – 1.00, mean ≈ 0.94

So all three "work" on trivial single-needle samples. The cross-model differences in headline accuracy are driven almost entirely by the hard tier.

### `hard_multi_entry` (n=207 stories, multi-fact)
| zone | model | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|------|-------|----|----|----|----|-----|-----|-----|------|
| beg | lora_base | 0.879 | 0.770 | 0.778 | 0.715 | 0.802 | 0.696 | 0.628 | 0.570 |
| beg | y2_base   | 0.768 | 0.787 | 0.734 | 0.700 | 0.710 | 0.638 | 0.580 | 0.517 |
| beg | y2_rpe    | 0.808 | 0.825 | 0.763 | 0.763 | 0.807 | 0.744 | 0.580 | 0.676 |
| mid | lora_base | 0.908 | 0.852 | 0.763 | 0.841 | 0.536 | 0.749 | 0.686 | 0.517 |
| mid | y2_base   | 0.759 | 0.769 | 0.734 | 0.725 | 0.527 | 0.705 | 0.638 | 0.454 |
| mid | y2_rpe    | 0.862 | 0.808 | 0.831 | 0.850 | 0.725 | 0.720 | 0.705 | 0.541 |
| end | lora_base | 0.891 | 0.896 | 0.903 | 0.850 | 0.870 | 0.836 | 0.763 | 0.691 |
| end | y2_base   | 0.761 | 0.841 | 0.855 | 0.860 | 0.841 | 0.841 | 0.768 | 0.628 |
| end | y2_rpe    | 0.848 | 0.852 | 0.855 | 0.879 | 0.899 | 0.899 | 0.884 | 0.821 |

Same patterns as headline; the hard tier dominates the signal.

---

## 5 — Multi-hop count (target_entries) effects

Sample counts (per the 227 selection): single_entry=20, target_entries=2 → 167, 3 → 35, 4 → 5.

Accuracy at 128k bin only:
| target_entries | n_unique | lora_base | y2_base | y2_rpe |
|----------------|----------|-----------|---------|--------|
| 0 (single)     | 20       | 0.883     | 0.783   | 0.883  |
| 2              | 167      | 0.583     | 0.531   | **0.689** |
| 3              | 35       | 0.648     | 0.505   | 0.600  |
| 4              | 5        | 0.533     | 0.800   | 0.933  |

(`target_entries=4` n is only 5 stories — too small to read into.)

**Pattern**: at 128k, `y2_rpe` adds the most over `y2_base` at the medium-difficulty 2-entry samples (+15.8pp). At 3-entry samples (the hardest-on-paper) `lora_base` actually leads at long bins. RPE doesn't uniformly help every multi-hop count.

Full grid at `tables/04_by_target_entries_acc.md`.

---

## 6 — Cross-model agreement per cell

For each cell, fraction of samples where:

Selected high-disagreement cells (sample-level):
| zone | bin | n | all3_correct | all3_wrong | exactly_1 | exactly_2 |
|------|-----|---|--------------|------------|-----------|-----------|
| beg | 128k | 227 | 0.330 | 0.123 | 0.242 | 0.304 |
| mid | 128k | 227 | 0.242 | 0.203 | 0.247 | 0.308 |
| end | 128k | 227 | 0.463 | 0.066 | 0.150 | 0.322 |
| beg | 64k  | 227 | 0.374 | 0.145 | 0.234 | 0.247 |
| mid | 16k  | 227 | 0.366 | 0.132 | 0.242 | 0.260 |

- At end_8k the three models all-agree-correct 75% of the time. At mid_128k it drops to **24%** — and 20% of samples no model gets right.
- `mid_128k` and `beg_128k` have the **highest "both/all wrong" rates** — these are the genuinely-hard cells where no model recovers.
- The `exactly_two_correct` column at long bins is consistently 25-32% — many samples are inside the model frontier of one model but not another. (This is the population RPE training is shifting.)

Full table: `tables/06_agreement.md`.

---

## 7 — Per-sample fraction-correct distribution

Across all 24 cells per model, how often each story (227 of them) is correct:

| frac_correct bucket | lora_base | y2_base | y2_rpe |
|---------------------|-----------|---------|--------|
| 0% (always wrong)   | 0   | 2   | 0   |
| 0–25%               | 12  | 16  | 5   |
| 25–50%              | 16  | 30  | 14  |
| 50–75%              | 46  | 45  | 54  |
| 75–99%              | 121 | 102 | 109 |
| **100%** (always right) | 32  | 32  | **45**  |

- `y2_rpe` has more "always-right" samples than the other two (45 vs 32 each), and far fewer in the 0–50% buckets (19 vs 28 vs 48).
- `y2_base` is the only model with samples that are **wrong at every cell** (n=2 stories).
- This is structural evidence that the gain isn't a few outlier cells — it shifts the whole distribution.

Per-tier breakdown: `tables/07_per_sample_dist_by_tier.csv`.

---

## 8 — Error analysis (predicted-class distribution)

Total errors per model (out of 5,021 evals):
| model | total errors | OTHER (non-room) | most-frequent wrong room |
|-------|--------------|------------------|--------------------------|
| lora_base   | 1,147 | 9   | kitchen (274) |
| y2_base     | **1,373** | 35  | office (369), hallway (363) |
| y2_rpe      | **1,008** | 5   | office (238), hallway (222) |

- `y2_base` has a strong "default to office or hallway when confused" failure mode — these two rooms account for 53% of its errors.
- `y2_rpe` has the most evenly spread error distribution (smaller modal failure class) and the lowest non-room ("OTHER") rate.
- "OTHER" rate (refusal / garbage) at 128k: lora_base=0.4–1.8% per cell, y2_base=2.2–4.9% per cell, y2_rpe=0.0–0.9% per cell. **`y2_rpe` is the most robust at producing valid-class outputs at long context.**

Confusion matrix per model: `tables/08_confusion_*.md`. Selected (y2_rpe):
- Target=bathroom is confused most often with hallway (99) and office/garden.
- Target=hallway has roughly even confusion across other rooms (no strong bias).
- The target→prediction confusion structure does not show a clean "answer the last-mentioned room" pattern — at least not in raw counts.

---

## 9 — Zone effect, per model (raw)

Per-model end-zone gain at each bin. Not a within-model finding — these are listed side-by-side so cross-model differences are visible.

| bin | lora end−mid | y2_base end−mid | y2_rpe end−mid |
|-----|--------------|-----------------|----------------|
| 1k  | −0.006 | −0.002 | −0.014 |
| 2k  | +0.040 | +0.069 | +0.045 |
| 4k  | +0.137 | +0.123 | +0.026 |
| 8k  | +0.013 | +0.137 | +0.031 |
| 16k | **+0.317** | **+0.291** | +0.163 |
| 32k | +0.088 | +0.114 | +0.163 |
| 64k | +0.066 | +0.128 | +0.159 |
| 128k| +0.167 | +0.167 | **+0.256** |

Cross-model interpretation:
- All 3 models show **positive end-vs-mid gain** at every bin ≥ 2k. Recency advantage is universal — nothing has eliminated it.
- **At mid-range bins (4k, 8k, 16k), `y2_rpe` has the SMALLEST end-vs-mid gap** — closest to "uniform across zones."
- **At long bins (32k, 64k, 128k), `y2_rpe` has the LARGEST end-vs-mid gap.** This is the opposite of "RPE reduces recency bias" — at 128k, RPE's end-zone advantage is the largest of the three.
  - Interpretation: RPE's improvement is heavily concentrated in end-zone. Beg/mid at 128k it improves only modestly; end at 128k it improves dramatically (0.714 → 0.824 vs lora; 0.643 → 0.824 vs y2_base).

So: **RPE does not equalize across zones. It widens the end-zone lead at long contexts.** This is a real, non-obvious finding to flag — opposite of the "RPE removes positional dependence" intuition.

---

## 10 — Token-count buckets (length-only view, ignoring zone)

| token bucket | lora_base | y2_base | y2_rpe |
|--------------|-----------|---------|--------|
| 1k–2k     | 0.906 | 0.799 | 0.857 |
| 2k–4k     | 0.852 | 0.814 | 0.842 |
| 4k–8k     | 0.828 | 0.788 | 0.830 |
| 8k–16k    | 0.816 | 0.775 | 0.838 |
| 16k–32k   | 0.749 | 0.709 | 0.821 |
| 32k–65k   | 0.772 | 0.739 | 0.803 |
| 65k–131k  | 0.664 | 0.617 | **0.719** |

- `lora_base` leads at 1k–4k (no surprise — short context, LoRA-only is fine).
- `y2_rpe` leads from 8k onward, with widest margin at 16k+.
- `y2_base` is dominated by `lora_base` at every length bucket.

---

## 11 — Position-decile analysis (big bins only, 32k+)

For the 32k/64k/128k cells, slice by mean fact-position decile (0=earliest, 9=latest):

| decile | lora | y2_base | y2_rpe |
|--------|------|---------|--------|
| 0 | 0.693 | 0.615 | 0.702 |
| 1 | 0.622 | 0.564 | 0.686 |
| 2 | 0.647 | 0.622 | 0.696 |
| 3 | 0.615 | 0.620 | 0.722 |
| 4 | 0.696 | 0.598 | 0.622 |
| 5 | 0.686 | 0.613 | 0.647 |
| 6 | 0.706 | 0.667 | 0.784 |
| 7 | 0.785 | 0.781 | 0.878 |
| 8 | 0.775 | 0.721 | 0.848 |
| 9 | 0.775 | 0.779 | 0.882 |

- All three models accelerate above ~decile 6 (back-half of the document).
- `y2_rpe` is uniformly above the others in the back deciles (6–9).
- In the FRONT (deciles 0–3), `y2_rpe` is the best at deciles 0–3, but deciles 4–5 have `lora_base` ≈ `y2_rpe`. No clean front-half advantage from RPE.
- `y2_base` is the worst at every decile.

---

## 12 — Other observations / things flagged for further investigation

1. **mid_16k anomaly** — drops sharply for ALL models (lora 0.560, y2_base 0.555, y2_rpe 0.745). Not a model artifact. Could be:
   - The PG19 noise text inserted at that specific length-zone tends to contain misleading room references
   - A re-randomization quirk (seed=42 + that bin/zone combination)
   Worth opening one of those samples to inspect.
2. **`y2_base` underperforms `lora_base` overall** — YaRN-only training without RPE is *worse* than no-YaRN training (when both are eval'd with YaRN f=4). This is a real negative result for "just use YaRN."
3. **RPE costs accuracy at short bins (1k/2k) in beg/end zones.** The aggregate +2.8pp over `lora_base` is a *net* of large long-context wins minus small short-context losses.
4. **End-zone advantage at 128k for y2_rpe is +0.110 over `lora_base` and +0.181 over `y2_base`.** This single cell is the largest contribution to the overall ranking.
5. **`y2_base` "OTHER" predictions at long context** suggest its long-context generation is unstable in a way the other models' aren't (5× the rate of `y2_rpe` at 128k). Might be the YaRN-eval extrapolation (f=4) causing degenerate outputs more often.
6. The "exactly two correct" column in the agreement table averages 22% across cells — over 1,000 samples land on the *border* between models. Likely the population that careful interventions can move.

---

## Files for further inspection

- `master.csv` — long-format, 15,063 rows, joinable on `original_idx`/`zone`/`bin`/`model`
- `tables/01_*` — headline accuracy and N
- `tables/02_*` — cross-model deltas + paired-flip counts
- `tables/03_*` — by tier (hard/single)
- `tables/04_*` — by target_entries
- `tables/05_*` — by needle position quartile/decile
- `tables/06_agreement.md` — per-cell 3-way agreement
- `tables/07_per_sample_*` — per-sample correctness counts and bucket distribution
- `tables/08_*` — error class distribution and per-model confusion matrices
- `tables/09_*` — zone-effect tables
- `tables/10_*` — token-count buckets
- `figures/fig_A_*` through `fig_H_*` — visualizations of every analysis above
