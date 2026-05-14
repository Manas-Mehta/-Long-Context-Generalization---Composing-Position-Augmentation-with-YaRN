# LongFaith → LongBench v2 — Revision Notes

Quick scan-back doc for the third-dataset experiment. Full context in
[HANDOFF.md §10](../HANDOFF.md).

## Goal (1-liner)
Train Qwen2.5-7B + LoRA with position-ID aug (RPE/PoSE) + YaRN f=2 at train;
eval with YaRN f=4. Composition beats either alone on long OOD context.
Third dataset extends the paper beyond BABILong + MRCR.

## Why composition works (recipe, not mechanism)
- Position-ID aug alone → collapses at 128K (below LoRA baseline)
- YaRN alone (train+eval) → destabilizes at most in-distribution bins
- **YaRN at train + position-ID aug at train + YaRN at eval** = only consistent win
- §5 zone study: lifts all positions uniformly → NOT recency-bias
- §6 retrieval heads: no clear differential
- → Paper framed as recipe, mechanism left open

## Headline numbers (BABILong QA3, 128K, paper)
| Cond (train/eval) | 128K | OOD-avg |
|---|---|---|
| none/none | 62 | 85 |
| YaRN/YaRN | 69 | 82 |
| RPE/none | 50 | 80 |
| RPE/YaRN | 81 | 89 |
| **YaRN+RPE/YaRN** | **84** | **90** |

## 7 conditions (same matrix for LongFaith)
1. none/none (LoRA only)
2. YaRN/YaRN
3. PoSE/none
4. RPE/none
5. YaRN+PoSE/YaRN
6. RPE/YaRN
7. YaRN+RPE/YaRN ← expected winner

## Hyperparams (frozen, mirror BABILong)
- Base: `Qwen/Qwen2.5-7B-Instruct`, LoRA rank 16, α 32, dropout 0.1
- lr 5e-5, 2 epochs, bs 1 × grad-accum 4, warmup 0.05, seed 42
- max-seq-len 9,216 (LongFaith max ≈ 5.7K Qwen tokens → fits cleanly)
- YaRN: f=2 train, f=4 eval
- RPE: curriculum L=16K · PoSE: target=32K

## Plan phases
1. **Inspect** 1 LongFaith sample + bucket v2 by Qwen tokens
2. **Scaffold** `experiments/longfaith/` mirroring `experiments/babilong/`
3. **Smoke** test: 1 step train → 4-example v2 eval
4. **Train** 7 conditions on h200
5. **Eval** on LongBench v2 QA subsets, MCQ letter parser
6. **Sanity**: rerun YaRN+RPE/YaRN on BABILong 128K to confirm no regression

## Risks / caveats
1. **Domain transfer unvalidated** — LongFaith=Wikipedia, v2=mixed (academic/legal/financial/etc). Mitigation: zero-shot Qwen v1 multi-doc QA baseline first.
2. **Cross-model transfer unvalidated** — LongFaith published on Llama-3.1-8B; we use Qwen. Same mitigation.
3. **Train=free-form CoT, eval=MCQ letter** — one-line prompt change but watch the parser hit-rate on v2.
4. **128K bin is sparse** — 273/300 v2 QA fit ≤128K, but per-domain that's ~30–40 examples.
5. **v2 examples = natural lengths, not stretched** — unlike BABILong, each bin contains *different* examples. Table 1 row reports length-bucketed avg, not length-stretched avg.

## What stays unchanged
- Training max-seq-len = 9,216
- Hyperparams above
- RPE/PoSE configs (`rpe_config_babilong_curriculum_L16k.yaml`, `pose_config_babilong_32k.yaml`)
- 7-condition matrix

## What changes
- Dataset loader: alpaca format (one-line read)
- Prompt: pass `instruction` directly (already CoC-formatted with [1]–[20])
- Eval parser: regex `The answer is (.+?)(\.|$)` for v1 sanity; override to A/B/C/D letter for v2 MCQ

---

# Dataset

## LongFaith-SFT (training)

**Source.** Yang et al., ACL Findings 2025, arXiv:2502.12583 ·
[github.com/IDEA-FinAI/LongFaith](https://github.com/IDEA-FinAI/LongFaith) · MIT

**Where the file lives.**
- Google Drive only — no HF release:
  https://drive.google.com/drive/folders/1f2306gR41glW9PzO6dJz8X5J53XsSNtC
- Local (ephemeral): `/tmp/longfaith_data/LongFaith_datasets/longfaith_syn/gpt-4o-mini/faith_sft_2k.json` (26 MB)
- TODO: move to HPC `/scratch/mm14444/positionaug-YaRN/experiments/longfaith/data/` or commit to repo

**Variant chosen.** `gpt-4o-mini` synthesizer, 2K-sample SFT file.
Their Table 4 ablation shows GPT-4o-mini-synthesized data is strongest.
**"gpt-4o-mini" = the LLM that wrote the reasoning chains, not our target model.**
Our target is Qwen2.5-7B-Instruct regardless.

**Size.** 2,048 examples. (2,038 with exactly 20 docs; ~10 outliers with
malformed bracket counts from raw Wikipedia — easy to filter.)

**Format (alpaca).**
```json
{
  "instruction": "<CoC header> DOCUMENTS: [1]… [20]… QUESTION: …",
  "input": "",
  "output": "Step 1: … Step 2: … The answer is X."
}
```
- `instruction` is fully pre-formatted: CoC prompt + 20 Wikipedia paragraphs marked `[1]…[20]` + question. **Feed directly as the prompt; no template work.**
- `output` is a synthesized reasoning chain ending in `The answer is X.` (typ. 200–2,000 chars)

**Length distribution (computed locally).**

| Stat | Instruction (chars) | Output (chars) | Total (chars) | Total (Qwen tokens ≈ chars/3.5) |
|---|---|---|---|---|
| min | 5,316 | 231 | 6,005 | ~1,715 |
| p50 | 11,556 | 1,013 | 12,596 | ~3,600 |
| p95 | ~16,000 | ~1,700 | ~17,800 | ~4,800 |
| max | 18,771 | 2,292 | 20,062 | ~5,732 |

**→ All 2,048 fit ≤ ~5.7K Qwen tokens. Existing max-seq-len 9,216 is enough. No infra changes.**

**Gotchas (do not re-trip).**
- LongFaith paper Table 6 "1K/2K/4K/8K" = **sample counts**, not context lengths. Released file is the 2K-sample variant.
- LongFaith paper Table 8 avg "11,542" = **chars**, not tokens. (Source of the "doesn't fit our window" red herring.)
- Qwen2.5-7B synthesizer folder has only raw components (`coc_*k_*.json`), no final SFT file. Don't go there — use gpt-4o-mini.

**Why we picked this (rejected alternatives).**
1. v1-train → v2-eval: only ~440 LongBench-E QA examples ≤8K Qwen tokens. Too small for 7-cond matrix.
2. v2-only train+eval: min v2 ctx ~14K; zero v2 examples fit our 8K training window.
3. Pad v1 with BABILong-style PG19 noise: `NoiseInjectionDataset` interleaves short discrete bAbI sentences — semantics don't transfer to v1's coherent docs.
4. ✓ **LongFaith**: fits natively + multi-hop QA matches v2 + proven on v1.

---

## LongBench v2 (evaluation)

**Source.** [`THUDM/LongBench-v2`](https://huggingface.co/datasets/THUDM/LongBench-v2) on HF
· local copy at `/tmp/longbench_v2.json` (503 examples)

**Format.** 4-choice MCQ. Each example has its own document(s); answer is a letter A/B/C/D.

**Splits (full v2, 503 examples).**
- By length category: 180 short / 215 medium / 108 long
- By difficulty: 311 hard / 192 easy
- By sub-domain:

| Sub-domain | N | Use? |
|---|---|---|
| Single-doc QA | 175 | ✓ primary |
| Multi-doc QA | 125 | ✓ primary |
| Long ICL | 81 | skip |
| Code | 50 | skip |
| Long-dialogue | 39 | skip |
| Long Structured Data | 33 | skip |

**Sub-domain breakdown (the QA ones we use).**
- Single-doc QA (175): Academic 44, Literary 30, Detective 22, Financial 22, Event ordering 20, Legal 19, Governmental 18
- Multi-doc QA (125): Academic 50, Multi-news 23, Governmental 23, Financial 15, Legal 14

**Length buckets (Qwen tokens, chars/3.5 approx, full v2).**
| Bucket | Count |
|---|---|
| ≤ 8K | 0 |
| ≤ 16K | 11 |
| ≤ 32K | 110 |
| ≤ 128K | 273 |

Raw range: short 14K–240K, medium 49K–638K, long 206K–4.6M.

**Our eval set.** QA subsets only → 300 examples → bucket into 16K / 32K / 64K / 128K → ≤128K filter → MCQ letter parser.

**OOD setup holds.** We train at ≤6K Qwen tokens; every v2 eval bin (16K+) is OOD on length. Same ratio shape as BABILong (train 8K, eval to 128K).

---

## LongBench v1 (sanity-check only)

**Source.** [`THUDM/LongBench`](https://huggingface.co/datasets/THUDM/LongBench) on HF
· local copy at `/tmp/longbench_v1/data/*_e.jsonl` (3,668 examples across 13 EN tasks)

**Length distribution (Qwen tokens).**
| Bucket | Count |
|---|---|
| ≤4K | 488 |
| 4–8K | 492 |
| 8–16K | 1,345 |
| 16–32K | 765 |
| >32K | 90 |

**Cannot serve as 128K OOD eval** — QA tasks max ~22K–28K tokens
(hotpotqa ~22K, 2wikimqa ~21K, qasper ~28K, multifieldqa ~18K).

**Use only for:** zero-shot Qwen baseline on v1 multi-doc QA *before* launching the 7-condition matrix, as the Llama→Qwen / Wikipedia→mixed-domain transfer sanity check.

---

# Implementation (locked 2026-05-14)

Scaffolded at [experiments/longfaith/](../experiments/longfaith/). Forks BABILong patterns; reuses `posaug` package verbatim.

## Files

```
experiments/longfaith/
  scripts/
    prepare_longfaith.py        # gdown LongFaith from Google Drive + filter ~10 outliers
    prepare_longbench_v2.py     # HF download v2 + filter QA + Qwen-tokenize + bucket
    train_longfaith_lora.py     # fork of train_babilong_lora.py
    eval_longbench_v2.py        # MCQ eval with CoT generation + letter parser
  configs/
    rpe_config_longfaith_curriculum_L16k.yaml   # identical to BABILong's
    pose_config_longfaith_32k.yaml              # identical to BABILong's
  hpc/
    smoke_test.slurm            # 1-step train + 2-sample 16k eval
  README.md
  REPRODUCIBILITY.md
```

## Frozen hyperparameters (every trained condition)

| Knob | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | paper convention |
| LoRA rank | 16 | BABILong default |
| LoRA α | 32 | BABILong default |
| LoRA dropout | 0.1 | BABILong default |
| LoRA target | `q,k,v,o,up,down,gate` projs | BABILong default |
| Learning rate | 5e-5 | BABILong default |
| Epochs | **2** | BABILong default (matches §4.1 frozen) |
| Batch size | 1 | BABILong default |
| Grad accumulation | 4 (effective batch 4) | BABILong default |
| LR scheduler | cosine | BABILong default |
| Warmup ratio | 0.05 | BABILong default |
| Max grad norm | 1.0 | BABILong default |
| Seed | 42 | BABILong default |
| max_seq_len (train) | 9,216 | LongFaith max ≈ 5.7K, headroom |
| Precision | bf16 | BABILong default |
| Gradient checkpointing | on | BABILong default |
| Save strategy | per-epoch (top 3 kept) | BABILong default |

## YaRN / RPE / PoSE settings

| Condition | YaRN train f | YaRN eval f | Position-ID aug | Aug config |
|---|---|---|---|---|
| `zero_shot_nyarn` | — | off | — | (no train) |
| `zero_shot_yarn4` | — | **4.0** | — | (no train) |
| `lora_base`       | off | off | — | — |
| `y2_base`         | **2.0** | **4.0** | — | — |
| `rpe_only`        | off | off | RPE | L=16K, curriculum {1:8K, 2:16K} |
| `pose_only`       | off | off | PoSE | target=32K |
| `y2_rpe_cur_L16k` | **2.0** | **4.0** | RPE | L=16K, curriculum {1:8K, 2:16K} |
| `y2_pose_32k`     | **2.0** | **4.0** | PoSE | target=32K |

Total: **7 trained + 2 zero-shot = 9 eval rows.**

Configs literally identical to BABILong's so cross-dataset claims are apples-to-apples.

## Eval pipeline

- Source: `THUDM/LongBench-v2` (503 examples) → filtered to Single-Doc QA (175) + Multi-Doc QA (125) = **300 examples**
- Bucketed by Qwen tokenizer on `context` field:
  - `16k`  : context_tokens ≤ 16384
  - `32k`  : 16384 < context_tokens ≤ 32768
  - `64k`  : 32768 < context_tokens ≤ 65536
  - `128k` : 65536 < context_tokens ≤ 131072
  - `>128k`: **dropped** (cannot fit even with YaRN f=4 at eval)
- Per-bin counts populated at prep time. From the full 503-example v2: ≤32K=110, ≤128K=273. After QA filter (300 of 503) per-bin counts will be re-tallied by `prepare_longbench_v2.py`.
- Generation: greedy decoding, `max_new_tokens=512`, CoT trace expected
- Parser:
  - **Primary**: `r"[Tt]he answer is\s*[:\-]?\s*\(?\s*([ABCD])\b"` on raw output
  - **Fallback**: last standalone `[ABCD]` in the final 300 chars of output
  - Each prediction logs `parser` field (`primary` / `fallback` / `miss`) for post-hoc debugging
- MCQ prompt drops the explicit `[N]` citation instruction (the prompt body just says "step-by-step reasoning"), but the eval context is **wrapped as a single `[1]` document** so the model has a citation handle from its training distribution (see "Eval prompt: Option A vs B" decision below)
- Uses `DOCUMENTS:` (plural) header — matches LongFaith training distribution

## Smoke test ([hpc/smoke_test.slurm](../experiments/longfaith/hpc/smoke_test.slurm))

l40s, ~45min budget. Three steps:
1. Import check (posaug, torch, transformers, peft)
2. 1-step train with RPE callback wired in → save checkpoint
3. 2-sample eval on the 16k v2 bin → produce per-bin predictions JSON

**Catches:** import / path errors, LoRA save+load roundtrip, prompt construction, parser regex, bin index loading.
**Does NOT catch:** YaRN f=4 inference logic, long-context handling, parser edge cases at scale.

## Known concerns (revisit if smoke or full run surfaces issues)

1. **Train-prompt / eval-prompt mismatch (partially mitigated).** Training: `DOCUMENTS:\n[1]…[20]…` with 20 short Wikipedia paragraphs. Eval: `DOCUMENTS:\n[1] {full v2 context}` — single `[1]` wrap (Option B) gives the model a citation handle but no multi-doc structure. Still a structural mismatch (1 large doc vs 20 small ones). Watch parser hit-rate; if it's high (>90%) the mitigation is sufficient.
2. **CoT eval cost.** Each example generates ~300–500 tokens. At 128K context, generation cost dominates prefill on h200. Budget ~12–15 GPU-h for the full 7-condition × 4-bin eval matrix.
3. **128K bin sparsity.** Each sub-domain has ~30–40 examples at 128K → wide per-row error bars. Worth reporting bin-averaged numbers + a 64K–128K composite.
4. **Cross-model transfer unvalidated.** LongFaith was published only on Llama-3.1-8B; Qwen2.5-7B was their synthesizer, never their target. Optional sanity: zero-shot Qwen on LongBench v1 multi-doc QA before launching the 7-condition matrix.
5. **`max_new_tokens=512` may truncate longest 5%** of CoT chains (LongFaith outputs p95 ≈ 500, max ≈ 650 tokens). If parser miss-rate is high, bump to 1024.

## Eval prompt: Option A vs B vs D (final decision: D, matching LongFaith authors)

**Switched 2026-05-14 after reading LongFaith's published eval code.**

The LongFaith authors' own [data_manager.py](https://github.com/IDEA-FinAI/LongFaith/blob/main/data_manager.py) shows that for single-doc benchmarks (qasper, multifieldqa_en) they artificially chunk the context into 20 equal char-slices labeled `[1]…[20]` and keep the citation instruction in the eval prompt. For multi-doc benchmarks (2wikimqa, hotpotqa, musique) they regex-rewrite existing "Passage N:\n" headers into `[N]` markers. Their `PREDICT_COC_PROMPT` is character-for-character the same as the training prompt.

**Our switch:** mirror their single-doc handling on v2 (Option D). Both Single-Doc and Multi-Doc v2 contexts get chunked into 20 char-equal slices labeled `[1]…[20]`. Citation instruction restored in `MCQ_PROMPT`. Final answer constraint overridden from "concise" to "exactly one letter A/B/C/D".

**Why we changed:**
- LongFaith authors published Llama-LongFaith results using this exact eval pattern → apples-to-apples claim
- Defensible to reviewers: "we used the authors' eval protocol"
- Preserves the trained 20-doc retrieval mechanism instead of collapsing to single `[1]` shallow-retrieval (the concern raised in our prior discussion)
- Implementation is trivial — see `_chunk_into_20()` in `eval_longbench_v2.py`, ~15 lines total

**LongBench v2's official protocol** uses no `[N]` markers (raw `<text>…</text>` wrap) and a two-stage CoT-then-extract eval. We diverge from this deliberately to match LongFaith training distribution; the two-stage extract isn't needed because the model has been trained to end with `The answer is X` directly.

**Options table (final):**

| | Description | Train-dist closeness | Status |
|---|---|---|---|
| A | Raw v2 context, no markers | low | rejected |
| B | Single `[1]` wrap | medium | superseded |
| C | Heuristic doc-boundary detection | high (multi) / low (single) | not pursued |
| **D** | **20 char-equal slices `[1]…[20]`** | **high** | **PICKED — matches LongFaith authors verbatim** |
| E | RAG-style retrieval | maximum | rejected (defeats long-context test) |

**Trade-off acknowledged:** char-based slicing splits mid-word. LongFaith authors accept this — training-distribution match wins.

## Eval prompt: Option A vs B (superseded — see above)

**The problem.** LongFaith trains the model on Chain-of-Citations (CoC) format: prompt includes 20 Wikipedia paragraphs labeled `[1]…[20]`, and the output is "Step 1: from [3], … The answer is X." with explicit citation markers. LongBench v2 contexts have NO `[N]` markers — single-doc QA is one coherent document; multi-doc QA is several documents concatenated without parseable boundaries.

**Options considered.**

| | Description | Train-dist closeness | Risk |
|---|---|---|---|
| A | Raw v2 context, no markers | low | distribution shift |
| **B** | **Wrap whole context as `[1] {context}`** | **medium** | **none — 5-char change** |
| C | Heuristic doc-boundary detection → label `[1]…[N]` | high (multi) / low (single) | fragile, may mis-split |
| D | Chunk by `\n\n` paragraphs → label `[1]…[N]` capped at 20 | high | arbitrary chunking; may exceed seen `[N]` range |
| E | RAG-style top-K retrieval | maximum | **disqualifies experiment** (defeats long-context test) |

**Decision: Option B** (implemented in `eval_longbench_v2.py:build_prompt`).

**Why:**
- Gives the model one citation handle it has seen in training, without imposing arbitrary chunking decisions on v2.
- D introduces research-decisions-with-no-right-answer (how many chunks? where to split? cap at 20?). Every choice is defensible, none is principled; reviewers can pick at any of them.
- D's chunks are structurally unlike training chunks: LongFaith's `[N]` were ~500-char Wikipedia paragraphs; v2-chunked-into-20 at 128K gives ~6.4K-token chunks. Same shape, very different scale — model's attention pattern over 500 chars is not the same machine as over 6.4K tokens.
- D risks `[N]` numbers the model never saw (uncapped → `[47]` markers at long context; capped → arbitrary merging).
- B is honest about what we know: v2 doesn't expose document boundaries reliably. Wrapping as one `[1]` is the truest statement of "here is the input, treat it as a single source."

**What we lose vs D:** the model trained on 20-doc retrieval pattern doesn't get the multi-doc structural prior at eval. Real concern at 128K, where finding the right sub-passage in one giant `[1]` blob is harder than finding the right `[N]` of 20 small ones.

**Phase-2 ablation (if results are flat).** Run Option D (paragraph-chunked `[N]`, capped at 20 by merging consecutive paragraphs) on the same checkpoints. Compare to B. If D meaningfully helps → that's a paper-worthy finding ("recipe transfers better when eval prompts retain training-time structural priors"). If D doesn't help → question closed.

**Experimental-validity trap to remember.** The more we massage v2 to look like LongFaith training, the less "OOD" the eval becomes — at the limit we'd be testing in-distribution-at-longer-lengths. That's still a valid claim ("recipe extends context length") but it's a different claim than "recipe handles diverse downstream tasks robustly." Option B sits at the conservative end of massaging, which keeps the OOD claim defensible.

## Operational checklist (when ready to run)

1. **HPC data setup** (login node, has internet):
   ```bash
   pip install gdown    # one-time
   python experiments/longfaith/scripts/prepare_longfaith.py \
       --output-dir experiments/longfaith/data
   python experiments/longfaith/scripts/prepare_longbench_v2.py \
       --output-dir experiments/longfaith/data
   ```
2. **Smoke**: `sbatch experiments/longfaith/hpc/smoke_test.slurm`
3. If smoke passes → write 7 condition train slurms (copy BABILong's per-condition slurms; swap script + config paths)
4. Train all 7 conditions (h200, ~1–2 GPU-h each)
5. Eval all 9 rows on v2 (l40s for ≤32K bins, h200 for 64K/128K bins)
6. Aggregate → fill paper Table 1 row for LongFaith
