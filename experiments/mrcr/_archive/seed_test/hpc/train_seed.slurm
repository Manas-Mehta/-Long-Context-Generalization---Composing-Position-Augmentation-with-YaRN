#!/bin/bash
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/seed_test/slurm_logs/train_%x_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/seed_test/slurm_logs/train_%x_%j.err
#SBATCH --requeue

# ============================================================================
# Seed Test — Parameterized training script
# Usage: sbatch --job-name=<name> --export=MODEL=<name>,SEED=<seed>,YARN_FACTOR=<f>,POS_FLAG=<flag> train_seed.slurm
#
# MODEL:       short name (e.g. y2_pose_32k)
# SEED:        random seed (123 or 456)
# YARN_FACTOR: "" for no YaRN at train, "2.0" or "3.0" or "4.0"
# POS_FLAG:    "" for none, "--rpe-config <path>" or "--pose-config <path>"
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
MRCR_DIR="composable_cot/mrcr_context_extension"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1

cd "${PROJECT_DIR}"

# Build YaRN flag
YARN_FLAG=""
if [ -n "${YARN_FACTOR}" ]; then
    YARN_FLAG="--enable-yarn --yarn-factor ${YARN_FACTOR}"
fi

CKPT_DIR="${MRCR_DIR}/seed_test/checkpoints/${MODEL}_s${SEED}"

echo "=== Seed Test Training: ${MODEL} seed=${SEED} ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  YaRN:      ${YARN_FLAG:-none}"
echo "  Position:  ${POS_FLAG:-none}"
echo "  Seed:      ${SEED}"
echo "  Output:    ${CKPT_DIR}"
echo ""

python ${MRCR_DIR}/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    ${YARN_FLAG} \
    ${POS_FLAG} \
    --train-file "${MRCR_DIR}/data/bin0_4K-8K/train.json" \
    --output-dir "${CKPT_DIR}" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed ${SEED}

echo ""
echo "=== ${MODEL} seed=${SEED} training complete: $(date) ==="
