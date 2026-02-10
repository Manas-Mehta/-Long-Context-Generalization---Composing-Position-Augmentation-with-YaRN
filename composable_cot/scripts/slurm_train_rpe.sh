#!/bin/bash
#SBATCH --job-name=rpe-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rpe-sft-%j.out
#SBATCH --error=logs/rpe-sft-%j.err

# ── NYU HPC Environment ───────────────────────────────────────────────
# Adjust module loads to match your cluster's environment
module purge
module load anaconda3/2023.09
module load cuda/12.1

conda activate rpe

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT="/path/to/RPE"
CCOT_ROOT="${PROJECT_ROOT}/composable_cot"
LLAMA_FACTORY="${CCOT_ROOT}/LLaMA-Factory"

# Add RPE to Python path so the callback can import rpe.patching
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ── Configuration ──────────────────────────────────────────────────────
# Training config (standard LLaMA-Factory YAML)
TRAIN_CONFIG="${CCOT_ROOT}/scripts/llamafactory/letter_concat_ascii_multiply_composable_cot.yaml"

# RPE config (our custom YAML — edit enabled: true/false for RPE vs baseline)
RPE_CONFIG="${CCOT_ROOT}/scripts/rpe_config.yaml"

# ── Launch ─────────────────────────────────────────────────────────────
cd "${LLAMA_FACTORY}"

# Single-GPU training with RPE callback injected via the patched tuner.py.
# The only change to LLaMA-Factory is a 4-line addition to tuner.py that
# reads RPE_CONFIG_PATH from the environment and adds RPETrainerCallback.
RPE_CONFIG_PATH="${RPE_CONFIG}" llamafactory-cli train "${TRAIN_CONFIG}"

# ── Notes ──────────────────────────────────────────────────────────────
# To run baseline (no RPE):
#   Option A: Set enabled: false in rpe_config.yaml
#   Option B: Unset RPE_CONFIG_PATH:
#       unset RPE_CONFIG_PATH && llamafactory-cli train "${TRAIN_CONFIG}"
#
# For multi-GPU training, replace the launch line with:
#   RPE_CONFIG_PATH="${RPE_CONFIG}" llamafactory-cli train \
#       "${TRAIN_CONFIG}" \
#       --deepspeed ds_config.json
#
# Or use torchrun directly:
#   RPE_CONFIG_PATH="${RPE_CONFIG}" torchrun \
#       --nproc_per_node=4 \
#       -m llamafactory.cli train "${TRAIN_CONFIG}"
