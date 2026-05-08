#!/bin/bash
#SBATCH --job-name=p6_Y2
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_yarn2_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_yarn2_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 6 — Group B (#3): Y2
# YaRN f=2.0 + LoRA only (no RPE/PoSE). Baseline for "train low, eval high".
# We already have Y4 from Phase 3. Adding Y2 and Y3 for interpolation.
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

echo "=== Phase 6 Group B: Y2 (YaRN f=2.0 + LoRA only) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${MRCR_DIR}/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 2.0 \
    --train-file "${MRCR_DIR}/data/bin0_4K-8K/train.json" \
    --output-dir "${MRCR_DIR}/phase6/checkpoints/yarn2_lora" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

echo ""
echo "=== Y2 training complete: $(date) ==="
