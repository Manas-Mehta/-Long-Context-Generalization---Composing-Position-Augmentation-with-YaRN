# LongFaith — Reproducibility reference

Frozen hyperparameters for the seven-condition matrix on LongFaith-SFT
(gpt-4o-mini synthesizer) → LongBench v2 (QA subsets, 4 length bins).

## Common environment

```
PROJECT_DIR=/scratch/<NETID>/positionaug-YaRN
conda env:  <SCRATCH>/conda_envs/rpe   # python 3.11, torch 2.x, transformers 4.52.4
HF_HOME=<SCRATCH>/hf_cache, HF_HUB_OFFLINE=1
PYTHONPATH=$PROJECT_DIR
```

SBATCH (train): `--account=torch_pr_219_courant --partition=h200_courant
--gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00`.
SBATCH (eval, long bins): `--partition=h200_courant --mem=80G --time=06:00:00`.

## Data setup (one-time)

LongFaith is Google-Drive-only. Run on a node with internet:

```bash
pip install gdown                            # one-time
python ${EXPT_DIR}/scripts/prepare_longfaith.py \
    --output-dir ${EXPT_DIR}/data
# Produces:
#   data/faith_sft_2k.json
#   data/faith_sft_2k_filtered.json   (2,038 examples — drops ~10 outliers)
#   data/length_stats.json
```

LongBench v2 from HuggingFace (cache must contain `THUDM/LongBench-v2` or
provide a local copy via `--local-path`):

```bash
python ${EXPT_DIR}/scripts/prepare_longbench_v2.py \
    --output-dir ${EXPT_DIR}/data
# Produces:
#   data/longbench_v2_qa.json         (300 QA examples + n_tokens field)
#   data/eval_v2_bin_indices.json     ({16k|32k|64k|128k: [...indices]})
```

## §X.1 Main table — 6 training conditions + zero-shot

Shared train args for every trained condition:
```
--base-model     Qwen/Qwen2.5-7B-Instruct
--train-file     experiments/longfaith/data/faith_sft_2k_filtered.json
--max-seq-len    9216
--lora-rank      16     --lora-alpha 32   --lora-dropout 0.1
--lr             5e-5   --epochs 2        --batch-size 1   --grad-accum 4
--warmup-ratio   0.05   --max-grad-norm 1.0   --seed 42
--grad-log-every 100
```

| # | Name | Extra train args | Extra eval args |
|---|---|---|---|
| 1 | `lora_base`        | (none)                                                                              | (none)                            |
| 2 | `y2_base`          | `--enable-yarn --yarn-factor 2.0`                                                   | `--enable-yarn --yarn-factor 4.0` |
| 3 | `y2_rpe_cur_L16k`  | `--enable-yarn --yarn-factor 2.0  --rpe-config configs/rpe_config_longfaith_curriculum_L16k.yaml` | `--enable-yarn --yarn-factor 4.0` |
| 4 | `y2_pose_32k`      | `--enable-yarn --yarn-factor 2.0  --pose-config configs/pose_config_longfaith_32k.yaml`           | `--enable-yarn --yarn-factor 4.0` |
| 5 | `rpe_only`         | `--rpe-config configs/rpe_config_longfaith_curriculum_L16k.yaml`                                  | (none)                            |
| 6 | `pose_only`        | `--pose-config configs/pose_config_longfaith_32k.yaml`                                            | (none)                            |

### Zero-shot baselines (no training)

| Name | Eval args |
|---|---|
| `zero_shot_nyarn` | `--no-lora` |
| `zero_shot_yarn4` | `--no-lora --enable-yarn --yarn-factor 4.0` |

## Shared eval args (all conditions)

```
--data-dir       experiments/longfaith/data
--output-dir     experiments/longfaith/results/<cond>
--condition      <cond>
--bins           16k 32k 64k 128k
--max-samples    0                    # all examples in each bin
--max-seq-len    131072
--max-new-tokens 512
```

CoT eval: the model generates a full reasoning chain ending in
`The answer is X.` where X ∈ {A, B, C, D}. Parser: regex
`r"[Tt]he answer is\s*[:\-]?\s*\(?\s*([ABCD])\b"`. Fallback: last standalone
A/B/C/D in the final ~300 chars. `parser` field in per-bin predictions
records which path matched (`primary` / `fallback` / `miss`).

## Configs

```
configs/rpe_config_longfaith_curriculum_L16k.yaml   # RPE curriculum L=8K → 16K
configs/pose_config_longfaith_32k.yaml              # PoSE target_length=32K
```

Both files mirror BABILong's exactly (same L, same target, same curriculum)
so cross-dataset comparisons are clean.

## Smoke test

```bash
sbatch experiments/longfaith/hpc/smoke_test.slurm
```

Runs 1 training step + 2-sample eval on the 16k bin. Use to verify the
pipeline before launching full condition runs.
