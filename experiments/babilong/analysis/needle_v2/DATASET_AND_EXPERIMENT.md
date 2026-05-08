# Needle-Position Eval v2 — Dataset & Experiment Design

Self-contained description of what data we run on, how we built it, and why each choice was made.

---

## 1. The underlying task: BABILong QA3

**Source**: `RMT-team/babilong-1k-samples` (HuggingFace), split `qa3`, config `0k` — the 999 raw bAbI-task-3 stories with no noise injected (~0 background tokens).

**Task (QA3 = "three-supporting-facts / before-and-after location")**:
A story lists a sequence of small bAbI facts — one object is carried around, dropped, picked up by multiple agents across multiple rooms. The question asks *where the object was in a room before it moved to another room* — requiring the model to (a) find the query object in the story, (b) track its location timeline, and (c) return the previous-room when a specific move happens.

**Example** (idx 3):
```
Facts: ... Mary picked up the football. Mary went to the bathroom.
       Mary dropped the football. John picked up the football.
       John went to the garden. John dropped the football. ...
Question: Where was the football before the garden?
Answer: bathroom
```

**Output format**: single word (one of 6 rooms) — `bedroom | bathroom | office | kitchen | garden | hallway`.

**Why this task for studying long-context + position encoding**:
- Short "golden facts" (~5–20 tokens each) that must be located inside a large noise document
- Multi-hop reasoning (must chain 2+ facts about the query object) — not just single-needle retrieval
- Unambiguous scoring (exact-match of one of 6 canonical rooms)
- No knowledge cutoff / no ambiguity from world knowledge — the answer is fully determined by the story

---

## 2. Why a "hard" subset instead of random sampling

Running the full 999 stories across every (model × zone × bin) is:
- Expensive (×9 cells = ~9,000 GPU-minutes at 128k)
- Diluted — most BABILong QA3 samples are solved by all three models at every bin we care about. Averages over them compress the signal and make real differences invisible.

**Precursor eval (multi-entry eval, N=305)**:
Before this experiment, we ran a standard-BABILong eval on 305 multi-entry samples (samples where the query object appears in ≥2 rooms — genuinely multi-hop). That eval scored `lora_base`, `y2_base`, `y2_rpe_cur_L16k` on 9 length bins each (`0k, 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k`).

**Differentiating subset**: from that 305, we removed every sample that all three models got correct at every bin where the sample appears. Those "universally-easy" samples provide no signal — they are noise in the average.

- **Kept**: 207 samples where at least one model failed at at least one bin
- **Removed**: 98 universally-easy samples

See [hard_subset_analysis.md](../multi_entry_eval/hard_subset_analysis.md) and the JSON spec at [differentiating_subset_207.json](../multi_entry_eval/differentiating_subset_207.json).

**Why exclude short-bin failures too?** An earlier definition considered only long-bin failures. We rejected that: if a sample is failed at 1k (short) by one of the models, that's a real capability gap — not a "length" effect — and suppressing it biases the analysis toward "all models are fine at short context, RPE helps at long." The wider definition keeps failure evidence from every bin.

**Properties of the 207 hard multi-entry subset**:
| object | count |
|--------|------:|
| football | 75 |
| milk | 67 |
| apple | 65 |
| **target_entries** (how many times object moves) | |
| 2 | 167 |
| 3 | 35 |
| 4 | 5 |

All samples are 2-hop minimum (one object, moves between ≥2 rooms). Answers are distributed roughly uniformly across the 6 rooms.

---

## 3. Why +20 single-entry reference samples

The 207 multi-entry subset is, by construction, biased toward hard samples. A pure hard-only eval can't tell us whether a drop at long context is because:
- (A) the model genuinely loses the needle in noise, or
- (B) our subset is too hard for any model no matter the position encoding.

**Fix**: add 20 "single-entry" samples — stories where the query object appears in exactly one room, so the answer is essentially a single-needle retrieval. These are easy for a functioning long-context model at every bin. They serve as a **ceiling / sanity-check track**:
- If a model drops on single-entry at long context → it's a long-context / position-encoding problem.
- If a model is fine on single-entry but drops on multi-entry at long context → it's a multi-hop problem (possibly an attention-spread problem).

The 20 single-entry samples are **deterministically selected** (random-seed=42 from the single-entry pool) — identical to those used in v1 so results are comparable.

**Overlap check**: asserted `hard_indices ∩ single_entry_indices = ∅` (the 207 are all multi-entry; the 20 are all single-entry — disjoint by construction).

---

## 4. Final selection: 227 samples

Union of `207 hard multi-entry + 20 single-entry reference = 227`.

- **Spec**: [data/eval_needle_v2/selected_227_indices.json](../../data/eval_needle_v2/selected_227_indices.json)
- **Build script**: [scripts/build_needle_selection_v2.py](../../scripts/build_needle_selection_v2.py)

Each sample entry includes:
```json
{
  "idx": 3,                     // original index in RMT-team/babilong-1k-samples 0k/qa3
  "object": "football",
  "answer": "bathroom",
  "target_entries": 2,          // how many times object is mentioned in a room
  "tier": "hard_multi_entry",   // or "single_entry_ref"
  "failure_score": 23           // sum of failures across all (model, bin) from multi-entry eval
}
```

`failure_score` range: 0–27 (3 models × 9 bins). Higher = harder in the precursor eval.

---

## 5. The needle-position manipulation: zones

Standard BABILong injects golden facts at a random position inside a noise document. **v2 evaluates at three controlled zones** so we can separate "recency bias" from "raw length":

| Zone | Relative position | Semantics |
|------|-------------------|-----------|
| `beg` | [0.00, 0.33] of document | Facts are in the first third — model must look far back |
| `mid` | [0.33, 0.66] of document | Facts are in the middle |
| `end` | [0.66, 1.00] of document | Facts are in the last third (close to the question) |

**Why zones?** If a model scores 0.60 at 128k but 0.95 at 1k, two hypotheses fit:
1. Raw length hurts the model (RoPE extrapolation degrades attention quality)
2. The model is recency-biased (it pays attention to the last N tokens regardless of length, and at 128k a random needle lands outside that window with high probability)

Randomized-needle eval cannot distinguish (1) and (2). **Zones fix this**: if the model is recency-biased, accuracy at `end` ≫ accuracy at `beg/mid` regardless of bin. If it's a pure length effect, `end` ≈ `beg` ≈ `mid` at a given length. The interaction pattern between (zone, bin) diagnoses the mechanism.

**How the zone is enforced**: the upstream `NoiseInjectionDataset` (from [babilong](https://github.com/booydar/babilong)) accepts `task_start_pct` / `task_end_pct` args that constrain the uniform-random insertion window to the requested zone. We thread these through our generator.

---

## 6. Length bins: 1k → 128k

| Bin | Tokens | Rationale |
|-----|-------:|-----------|
| `1k`  | 1,024  | Baseline — story alone, minimal noise |
| `2k`  | 2,048  | Easy length |
| `4k`  | 4,096  | Still within training length |
| `8k`  | 8,192  | Edge of training length |
| `16k` | 16,384 | 2× training length — entry point for long-context behavior |
| `32k` | 32,768 | Heavy extrapolation territory |
| `64k` | 65,536 | 8× training length |
| `128k`| 130,700 | ~128K (16× training length) — leave ~372 tokens headroom for chat template + prompt to prevent left-truncation clipping the facts at beg zone |

**Why 128K cap?** Qwen2.5-7B-Instruct ships with YaRN-supported extension up to 131,072 tokens. 130,700 is the actual token budget after reserving room for the chat template and QA3 instruction.

**0k bin removed in v2**: v1 included `0k` (set to 1024 tokens), which duplicated the 1k bin since no noise is injected. Zone control is meaningless at 0k (no noise = no zone = facts are the only content). Removed to save compute.

**Effective N per cell** (some short bins can't fit the original stories without truncation):
- `1k`: 104 (mid) / 108 (end) / 116 (beg) — story + noise must fit in 1024 tokens; some stories exceed that even without noise
- `2k`: 202 (mid) / 202 (end) / 203 (beg)
- `4k`+: 227 across all cells (all 227 fit comfortably)

---

## 7. Noise source: PG19

- **Source**: `pg19` (HuggingFace), split `test`
- PG19 is the official BABILong noise source — long public-domain books, low topical overlap with bAbI rooms/objects so the noise doesn't accidentally contain the answer.
- Per-cell independent noise: the generator re-seeds the `SentenceSampler` with `random_seed=42 + hash((bin, zone)) % 10000`, so each of the 24 cells gets an independent noise draw.
- Samples are built with full PG19 sentences (not cut mid-sentence) by the upstream `SentenceSampler`.

**Why PG19 and not Wikipedia/etc.?**:
- Matches the original BABILong benchmark spec
- Long documents (enough to draw 128K of unique noise without cycling)
- Low risk of contaminating the 6-room vocabulary (PG19 = 19th century novels, hardly ever mentions "office" / "kitchen" / etc. in a bAbI-resembling pattern)

---

## 8. Full experiment grid

| Axis | Values | Count |
|------|--------|------:|
| Models | `lora_base`, `y2_base`, `y2_rpe_cur_L16k` | 3 |
| Zones  | `beg`, `mid`, `end`                        | 3 |
| Length bins | `1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k` | 8 |
| Samples/cell | 104–227 (see §6) | ~220 avg |

**Total evaluations**: 3 × 3 × 8 × ~220 = **15,063 model forward passes**

Wall-clock on H200 (HPC):
- `lora_base`: 5,084 s per zone × 3 zones ≈ 4.2 hr
- `y2_base`: ~3,500 s per zone × 3 ≈ 2.9 hr
- `y2_rpe_cur_L16k`: ~3,500 s per zone × 3 ≈ 2.9 hr
- **Data generation (CPU)**: ~1–2 hr (PG19 must be loaded and sampled)

---

## 9. Eval protocol

- **Checkpoint loading**: base = `Qwen/Qwen2.5-7B-Instruct`, LoRA adapter = `checkpoints/{model}_1k/checkpoint-2000` (all three models trained on the same 60-story 4k-8k bin for 75 steps)
- **All three models eval'd with YaRN f=4** (`enable_yarn=True, yarn_factor=4.0`):
    - `lora_base` trained with NO YaRN but eval'd with YaRN to stretch its 32K training context up to 128K
    - `y2_base` trained with YaRN f=2, eval'd with f=4 — 2× extrapolation on top of YaRN
    - `y2_rpe_cur_L16k` trained with YaRN f=2 + RPE curriculum L=16K, eval'd with f=4
- `max_seq_len = 131072`, `max_new_tokens = 10` (we only need one word)
- Greedy decoding, prediction normalized (lowercase + strip punctuation) for exact-match scoring against 6-room vocabulary

---

## 10. Why THIS specific v2 vs. the earlier v1

| Aspect | v1 | v2 | Reason for change |
|--------|-----|-----|-------------------|
| Samples | 100 (80 hard + 20 single) | **227** (207 hard + 20 single) | v1 pool too small; we had 207 pre-vetted hard samples ready |
| Models | 2 (`y2_base`, `y2_rpe_cur_L16k`) | **3** — added `lora_base` | v1 couldn't tell us whether differences were from YaRN training vs. RPE training |
| Length bins | 9 (`0k`-`128k`) | **8** (`1k`-`128k`) | 0k was a duplicate of 1k; wasted compute |
| Eval config | YaRN f=4 for all | Same | — |

**Analysis bias in v1**: v1's writeup reported only "RPE rescues" (cases where RPE beats the alternative). A balanced analysis also needs "RPE regressions" (cases where baselines beat RPE). v2 enables that by adding `lora_base` (to disentangle YaRN-only vs. no-continued-training) and a larger hard subset (to make regression counts reliable).

---

## 11. What's in the data directory

Under [`data/eval_needle_v2/`](../../data/eval_needle_v2/):

| File | Description |
|------|-------------|
| `selected_227_indices.json` | The 227-sample selection with per-sample metadata + failure-score |
| `manifest.json`             | Zone/bin grid spec + per-cell sample counts |
| `{zone}_{bin}.json`         | Pre-generated eval data, 24 files — ready-to-consume by `eval_babilong.py` |

Each eval sample in `{zone}_{bin}.json` has:
```json
{
  "messages": [ {"role":"user","content":"<PG19 noise + facts + Question>"},
                {"role":"assistant","content":"bathroom"} ],
  "answer": "bathroom",
  "question": "Where was the football before the garden?",
  "bin": "4k", "zone": "beg", "zone_pct": [0.0, 0.33],
  "token_count": 4257,
  "original_idx": 3,
  "tier": "hard_multi_entry",
  "object": "football",
  "target_entries": 2,
  "fact_positions_rel": [0.12, 0.18, 0.24, ...]
}
```

`fact_positions_rel` = per-fact insertion position as a fraction of the noise background. Used downstream for fine-grained position analysis (see §5 of REPORT.md).

---

## 12. What this experiment is designed to tell us

The zone × bin × model grid answers four questions cleanly:

1. **Raw length effect**: does accuracy drop as bin grows, holding zone fixed?
2. **Recency bias**: does `end` outperform `beg/mid` at each bin, and by how much?
3. **Does position encoding change the zone pattern?** i.e., does RPE training equalize zones, shift the recency curve, or just raise the `end` ceiling?
4. **Does training for long context cost anything at short context?** `lora_base` is untouched by YaRN/RPE training — it's the "do no harm" baseline. If `y2_*` models drop at `1k/2k` relative to `lora_base`, that's a training tradeoff to account for.

Findings for each are written up in [REPORT.md](REPORT.md).

---

## 13. Artifacts & reproduction

- **Selection build**: `python scripts/build_needle_selection_v2.py`
- **Data generation (HPC)**: `sbatch hpc/generate_needle_eval_v2.slurm`
- **Eval (HPC, 3 jobs in parallel with dependency on gen)**: `bash hpc/submit_needle_eval_v2.sh`
- **Analysis**: `python analysis/needle_v2/build_master.py && python analysis/needle_v2/analyze.py`
- **Results**:
    - `results/{model}_needle_v2_{beg,mid,end}/predictions_{bin}.json` — per-sample prediction JSONs (9 dirs × 8 files each)
    - `results/{model}_needle_v2_{beg,mid,end}/summary.json` — per-cell accuracy summary
