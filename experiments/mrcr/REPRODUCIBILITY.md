# MRCR — Reproducibility reference

Frozen hyperparameters from the paper-run SLURM scripts in `hpc/`. The
mechanism mirrors BABILong: LoRA at rank 16 on the 4K-8K bin, evaluated
across all five context bins. The §4.3 cross-benchmark observation comes
from the YaRN-at-eval rows.

## Common environment

```
PROJECT_DIR=/scratch/<NETID>/RPE          # repo root on HPC
conda env: <SCRATCH>/conda_envs/rpe        # python 3.11, torch 2.x, transformers 4.52.4
HF_HOME=<SCRATCH>/hf_cache, HF_HUB_OFFLINE=1
PYTHONPATH=$PROJECT_DIR
```

SBATCH (train): `--account=torch_pr_219_courant --partition=h200_courant
--gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00`.
SBATCH (eval): same headers but `--time=12:00:00` (5 bins, longest is 64K-128K).

## Step 1 — Prepare data (CPU only, ~10 min)

```
python experiments/mrcr/scripts/prepare_data.py \
    --tokenizer  Qwen/Qwen2.5-7B-Instruct \
    --output-dir experiments/mrcr/data \
    --train-ratio 0.7 \
    --n-needles  2 \
    --max-bin    2          # train data comes from bin 0 only (4K-8K); eval bins go up to bin 4
```

Produces `experiments/mrcr/data/{bin0_4K-8K,bin1_8K-16K,...,bin4_64K-128K}/{train,test}.json`.

## Step 2 — Train (paper conditions)

Shared train args:
```
--base-model     Qwen/Qwen2.5-7B-Instruct
--train-file     experiments/mrcr/data/bin0_4K-8K/train.json
--lora-rank      16     --lora-alpha 32  --lora-dropout 0.1
--lr             2e-4   --epochs 5       --batch-size 1   --grad-accum 4
--max-seq-len    8192   --warmup-ratio 0.1   --seed 42
```

| # | Name | Extra train args |
|---|---|---|
| 1 | `lora_baseline`         | (none) |
| 2 | `yarn_lora`             | `--enable-yarn --yarn-factor 4.0` |
| 3 | `rpe_lora_L16k`         | `--rpe-config configs/rpe_config_mrcr_L16k.yaml` |
| 4 | `rpe_curriculum_lora_L16k` | `--rpe-config configs/rpe_config_mrcr_curriculum_L16k.yaml` |
| 5 | `pose_lora`             | `--pose-config configs/pose_config_mrcr.yaml` |
| 6 | `pose_curriculum_lora`  | `--pose-config configs/pose_config_mrcr_curriculum.yaml` |

L-sweep variants (Phase 5): swap the RPE config file. Available:
`rpe_config_mrcr_{L16k,L64k,L128k,curriculum_L{4k,8k,16k,64k,128k}}.yaml`.

Output: `experiments/mrcr/checkpoints/<cond>/`.

## Step 3 — Evaluate (loop over 5 bins)

For each condition, loop over `bin0_4K-8K … bin4_64K-128K`:
```
python experiments/mrcr/scripts/eval_mrcr.py \
    --base-model     Qwen/Qwen2.5-7B-Instruct \
    --lora-ckpt      experiments/mrcr/checkpoints/<cond> \
    --test-file      experiments/mrcr/data/<bin>/test.json \
    --output-dir     experiments/mrcr/outputs/<cond>_<bin> \
    --max-new-tokens 2048
```

YaRN-at-eval rows additionally pass `--enable-yarn --yarn-factor 4.0`. The
paper's headline §4.3 result is `rpe_curriculum_lora_L16k` evaluated with
YaRN factor 4.

`eval_mrcr.py` auto-detects transformers 4 vs 5 for the YaRN config injection
— no manual fork needed. See `_apply_yarn_manual` in the script.

## Legacy ablations

Phase-6 L-sweep, seed-variance, and rank-128 compositional ablations now live
under `_archive/`. See `_archive/README.md` for the hyperparameter grid each
one swept. None of them are part of the §4.3 reproduction.

## Legacy SLURM templates

`hpc/*.slurm` still encode the pre-migration path layout
(`composable_cot/mrcr_context_extension/...`). They are kept for
git-history continuity but will not run as-is. Use `hpc/smoke_test.slurm`
as the canonical template — it matches the new `experiments/mrcr/` layout.
