# LongFaith → LongBench v2

Third-dataset experiment for the *Composing Position-ID Augmentation with
YaRN* paper. Trains Qwen2.5-7B-Instruct (LoRA) on LongFaith-SFT and
evaluates on LongBench v2 QA subsets at four length bins (16K/32K/64K/128K).

Tests whether the **YaRN-at-train + position-ID-aug + YaRN-at-eval** recipe
that won on BABILong QA3 and MRCR also transfers to a third multi-hop
benchmark with different content domains (academic / legal / financial /
multi-news vs. BABILong's PG19-padded bAbI facts).

## Directory layout

```
experiments/longfaith/
  scripts/
    prepare_longfaith.py        # gdown LongFaith from Google Drive + filter
    prepare_longbench_v2.py     # download v2 from HF + filter QA + bucket
    train_longfaith_lora.py     # fork of BABILong train; alpaca-format dataset
    eval_longbench_v2.py        # MCQ eval w/ CoT generation + letter parser
  configs/
    rpe_config_longfaith_curriculum_L16k.yaml
    pose_config_longfaith_32k.yaml
  hpc/
    smoke_test.slurm            # 1-step train + 2-sample eval pipeline check
  data/                          # produced by prepare scripts (gitignored)
  REPRODUCIBILITY.md             # frozen hyperparams + CLI invocations
```

## Conditions

Seven trained conditions + two zero-shot baselines (see `REPRODUCIBILITY.md`):

- **Baselines**: `zero_shot_nyarn`, `zero_shot_yarn4`, `lora_base`, `y2_base`
- **Ablations**: `rpe_only`, `pose_only`
- **Methods**: `y2_pose_32k`, `y2_rpe_cur_L16k` ← expected winner

Hyperparameters are frozen to match BABILong (rank 16, lr 5e-5, 2 epochs,
warmup 0.05, seed 42, max-seq-len 9216) so cross-dataset claims are
apples-to-apples.

## Why LongFaith and not raw LongBench v1 or v2

See top-level [notes/longfaith_experiment.md](../../notes/longfaith_experiment.md)
for the rejected alternatives. Short version: LongFaith's 2,048 multi-hop QA
examples all fit ≤ 6K Qwen tokens (max-seq-len 9216 is fine), and LongBench
v2's 16K+ bins are all OOD from training length — same ratio as BABILong.

## Differences from BABILong / MRCR

- **CoT training/eval.** LongFaith's `output` field is a Step-by-Step
  reasoning chain, unlike BABILong (single word) or MRCR (verbatim
  response). Eval also uses CoT — the model generates a chain ending
  `The answer is X` and the parser extracts the letter.
- **Letter answer.** LongBench v2 is MCQ. The eval prompt forces a final
  A/B/C/D; the training-time `output` ends with a free-form answer, so the
  prompt asks the model to emit a letter in the final position.
- **Natural length distribution.** v2 examples have their own natural
  lengths (no PG19 padding). Each bin contains *different* examples, not
  the same example padded — so per-bin numbers reflect a mix of difficulty
  and length, not pure length sensitivity.
