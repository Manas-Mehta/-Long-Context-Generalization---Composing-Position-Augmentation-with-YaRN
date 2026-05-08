# BABILong QA3 — Main paper experiment

This is the headline experiment of the paper. We finetune Qwen2.5-7B-Instruct
with LoRA on BABILong QA3 (Three Supporting Facts) under seven training/eval
configurations and measure accuracy across nine context-length bins (0K–128K).

Sections of the paper using this experiment:
- §4.1 Random-needle main table (7 conditions × 9 bins)
- §4.2 Multi-entry hard subset (3 conditions × 9 bins)
- §5 Position-sensitivity zone study (3 conditions × 8 bins × 3 zones)

## Layout

```
babilong/
  scripts/
    prepare_babilong.py             Fetch + tokenize BABILong QA3 from HuggingFace
    train_babilong_lora.py          LoRA training (6 conditions)
    eval_babilong.py                Greedy-decoding eval on 9 bins
    create_multi_entry_eval.py      §4.2 hard-subset selection
    build_needle_selection_v2.py    §5 sample selection
    generate_needle_position_eval.py §5 zone construction
    aggregate_results.py            Build the §4.1 table from summary.json
    analyze_training_progress.py    Per-step loss / mid-train accuracy plots
  analysis/
    analyze_results.py              §4.1 figures
    multi_entry_eval/               §4.2 hard-subset analysis
    needle_v2/                      §5 zone-study analysis (canonical)
    figures/                        Paper-referenced PNGs
  configs/
    rpe_config_babilong_curriculum_L16k.yaml
    pose_config_babilong_32k.yaml
  babilong_src/babilong/
    Vendored metric / prompt module from
    https://github.com/booydar/babilong (used by the needle-position eval
    generator only; eval_babilong.py reimplements the metric inline).
  hpc/                              SLURM templates
  data/, results/, checkpoints/     gitignored
```

## Training conditions

Six LoRA training conditions, all on the same 20K-sample mix of BABILong QA3
bins 0K/2K/4K/8K (1K from each of the four training bins). Same LoRA
hyperparameters (rank 16, α 32, dropout 0.1, lr 5e-5, batch 1, grad-accum 4,
2 epochs, seed 42, max_seq_len 9216).

| # | Name | Train-time position trick | Eval-time YaRN |
|---|---|---|---|
| 1 | `lora_base` | none | none |
| 2 | `y2_base` | YaRN factor 2 | factor 4 |
| 3 | `y2_rpe_cur_L16k` | YaRN f=2 + RPE curriculum L=10K→16K | factor 4 |
| 4 | `y2_pose_32k` | YaRN f=2 + PoSE target_length=32K | factor 4 |
| 5 | `rpe_only` | RPE curriculum L=10K→16K | none |
| 6 | `pose_only` | PoSE target_length=32K | none |

## Reproduce

```bash
# 1. Prepare data (downloads RMT-team/babilong-train-5k-samples to data/)
python experiments/babilong/scripts/prepare_babilong.py \
    --output-dir experiments/babilong/data

# 2. Train one condition (example: YaRN+RPE curriculum)
python experiments/babilong/scripts/train_babilong_lora.py \
    --enable-yarn --yarn-factor 2.0 \
    --rpe-config experiments/babilong/configs/rpe_config_babilong_curriculum_L16k.yaml \
    --train-file experiments/babilong/data/train/all_train.json \
    --output-dir experiments/babilong/checkpoints/y2_rpe_cur_L16k

# 3. Evaluate across all 9 bins
python experiments/babilong/scripts/eval_babilong.py \
    --checkpoint-dir experiments/babilong/checkpoints/y2_rpe_cur_L16k \
    --eval-dir       experiments/babilong/data/eval \
    --output-dir     experiments/babilong/results/y2_rpe_cur_L16k \
    --condition      y2_rpe_cur_L16k

# 4. Aggregate the §4.1 table once all six conditions are evaluated
python experiments/babilong/scripts/aggregate_results.py \
    --results-dir experiments/babilong/results/

# 5. Generate paper figures
python experiments/babilong/analysis/analyze_results.py
```

For §4.2 multi-entry and §5 needle-position evals, see the README in
`analysis/multi_entry_eval/` and `analysis/needle_v2/`.

## SLURM

Templates for the NYU Torch HPC environment are in `hpc/`. Adapt
`#SBATCH` headers (account, partition, conda env path, scratch dir) to
your cluster.
