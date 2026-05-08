# MRCR — Cross-benchmark observation (§4.3)

Preliminary experiment that surfaced the composition pattern: pairing a
training-time position-ID augmentation (RPE / PoSE) with YaRN at evaluation
helped on multi-round coreference resolution as well as on BABILong. The
paper reports MRCR results in the §4.3 cross-benchmark paragraph; the
mechanism is otherwise identical to the BABILong setup.

## Layout

```
mrcr/
  scripts/
    prepare_data.py             Fetch + token-bin openai/mrcr by Qwen tokenizer
    train_mrcr_lora.py          LoRA training (LoRA, YaRN, RPE, PoSE variants)
    eval_mrcr.py                Eval with version-aware YaRN config
    analyze_top_models.py       §4.3 cross-condition recency-bias plot
    analyze_errors.py           Error-type breakdown
    visualize_predictions.py    Per-sample prediction visualisation
  configs/
    rpe_config_mrcr_curriculum_L16k.yaml   (used in paper)
    pose_config_mrcr_curriculum.yaml
    plus several L-sweep / v2 ablation configs
  hpc/                          SLURM templates
  data/, outputs/, checkpoints/ gitignored
```

## Reproduce

```bash
# 1. Prepare MRCR data (downloads openai/mrcr from HuggingFace)
python experiments/mrcr/scripts/prepare_data.py \
    --output-dir experiments/mrcr/data

# 2. Train LoRA on the 4K-8K bin (example: YaRN+RPE curriculum)
python experiments/mrcr/scripts/train_mrcr_lora.py \
    --enable-yarn --yarn-factor 4.0 \
    --rpe-config experiments/mrcr/configs/rpe_config_mrcr_curriculum_L16k.yaml \
    --train-file experiments/mrcr/data/bin0_4K-8K/train.json \
    --output-dir experiments/mrcr/checkpoints/y4_rpe_cur_L16k

# 3. Evaluate across all bins (LoRA merge is automatic)
python experiments/mrcr/scripts/eval_mrcr.py \
    --enable-yarn --yarn-factor 4.0 \
    --lora-ckpt   experiments/mrcr/checkpoints/y4_rpe_cur_L16k \
    --test-file   experiments/mrcr/data/bin1_8K-16K/test.json \
    --output-dir  experiments/mrcr/outputs/y4_rpe_cur_L16k_8K-16K

# 4. Analysis (recency-bias plot used in §4.3)
python experiments/mrcr/scripts/analyze_top_models.py
```

## Note on YaRN at training

The paper's BABILong setup uses YaRN factor 2 at training and factor 4 at
evaluation. The MRCR pipeline supports the same split via `--enable-yarn
--yarn-factor` on both train and eval scripts. `eval_mrcr.py` contains
version-aware YaRN config logic (transformers 4 vs 5) — see the
`_apply_yarn_manual` fallback inside the script.
