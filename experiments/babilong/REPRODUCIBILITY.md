# BABILong — Reproducibility reference

Frozen hyperparameters from the paper-run SLURM scripts in `hpc/`. Recreate
these on any cluster by adapting only the SBATCH headers (account, partition,
conda env path, scratch dir). The Python invocations are exactly what produced
the §4.1 / §4.2 / §5 numbers.

## Common environment

```
PROJECT_DIR=/scratch/<NETID>/RPE          # repo root on HPC
conda env: <SCRATCH>/conda_envs/rpe        # python 3.11, torch 2.x, transformers 4.52.4
HF_HOME=<SCRATCH>/hf_cache, HF_HUB_OFFLINE=1
PYTHONPATH=$PROJECT_DIR
```

SBATCH (every train + most evals): `--account=torch_pr_219_courant
--partition=h200_courant --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00`.
Eval-only L40S variants use `--partition=l40s_courant --mem=48G --time=04:00:00`.

## §4.1 Main table — 6 training conditions

Shared train args for every condition below:
```
--base-model     Qwen/Qwen2.5-7B-Instruct
--train-file     experiments/babilong/data/train/all_train_1k.json
--eval-dir       experiments/babilong/data/eval
--max-seq-len    9216
--lora-rank      16     --lora-alpha 32  --lora-dropout 0.1
--lr             5e-5   --epochs 2       --batch-size 1   --grad-accum 4
--warmup-ratio   0.15   --max-grad-norm 1.0   --seed 42
--eval-every     1000   --eval-samples 200    --grad-log-every 100
```

| # | Name | Extra train args | Extra eval args |
|---|---|---|---|
| 1 | `lora_base`        | (none)                                                                              | (none)                          |
| 2 | `y2_base`          | `--enable-yarn --yarn-factor 2.0`                                                   | `--enable-yarn --yarn-factor 4.0` |
| 3 | `y2_rpe_cur_L16k`  | `--enable-yarn --yarn-factor 2.0  --rpe-config configs/rpe_config_babilong_curriculum_L16k.yaml` | `--enable-yarn --yarn-factor 4.0` |
| 4 | `y2_pose_32k`      | `--enable-yarn --yarn-factor 2.0  --pose-config configs/pose_config_babilong_32k.yaml`           | `--enable-yarn --yarn-factor 4.0` |
| 5 | `rpe_only`         | `--rpe-config configs/rpe_config_babilong_curriculum_L16k.yaml`                                  | (none)                          |
| 6 | `pose_only`        | `--pose-config configs/pose_config_babilong_32k.yaml`                                            | (none)                          |

Shared eval args (all conditions):
```
--max-samples 100   --max-seq-len 131072   --max-new-tokens 10
--checkpoint-dir experiments/babilong/checkpoints/<cond>_1k
--output-dir     experiments/babilong/results/<cond>_1k
--condition      <cond>_1k
```

## §4.2 Multi-entry hard subset

Reuses the §4.1 checkpoints. Conditions evaluated: `lora_base`, `y2_base`,
`y2_rpe_cur_L16k`. Each pinned to `checkpoint-2000`, full subset (305 samples).

```
--checkpoint-dir experiments/babilong/checkpoints/<cond>_1k/checkpoint-2000
--enable-yarn --yarn-factor 4.0          # applied at eval even for lora_base
--eval-dir       experiments/babilong/data/eval_multi_entry
--output-dir     experiments/babilong/results/<cond>_me
--condition      <cond>_me
--max-samples    0                       # full set
--max-seq-len    131072
--max-new-tokens 10
```

The `data/eval_multi_entry/` files are built by
`scripts/create_multi_entry_eval.py` (filters the 999-sample eval bins to the
305 samples where the target room is visited 2+ times).

## §5 Needle-position study (v2)

1. **Build sample selection (227 indices)**:
   ```
   python scripts/build_needle_selection_v2.py
   # output: data/eval_needle_v2/selected_227_indices.json
   ```
2. **Generate eval data**:
   ```
   python scripts/generate_needle_position_eval.py \
       --selected-indices data/eval_needle_v2/selected_227_indices.json \
       --output-dir       data/eval_needle_v2 \
       --bins             1k,2k,4k,8k,16k,32k,64k,128k \
       --zones            beg,mid,end \
       --noise-dataset    pg19 \
       --random-seed      42 \
       --babilong-src-dir babilong_src
   ```
3. **Run eval** (same conditions as §4.2). The `--eval-dir` swaps to
   `data/eval_needle_v2`; otherwise the args match §4.2.

## Configs

```
configs/rpe_config_babilong_curriculum_L16k.yaml   # RPE curriculum L=10K → 16K
configs/pose_config_babilong_32k.yaml               # PoSE target_length=32K
```

## Legacy SLURM templates

`hpc/*.slurm` still encode the pre-migration path layout
(`composable_cot/BABIlong/...`). They are kept for git-history continuity but
will not run as-is. Use `hpc/smoke_test.slurm` as the canonical template — it
matches the new `experiments/babilong/` layout.
