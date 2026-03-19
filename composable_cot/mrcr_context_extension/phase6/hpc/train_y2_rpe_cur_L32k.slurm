#!/bin/bash
#SBATCH --job-name=p6_Y2Rc32
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_y2_rpe_cur_L32k_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_y2_rpe_cur_L32k_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 6 — Group D (#11): Y2-Rc32
# YaRN f=2.0 + RPE curriculum L=32K. L = half of YaRN's 64K window.
# Tests: does RPE approaching YaRN's extended range help?
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

echo "=== Phase 6 Group D: Y2-Rc32 (YaRN f=2 + RPE cur L=32K) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  L vs YaRN window: 32K = half of 64K (approaching)"
echo ""

python ${MRCR_DIR}/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 2.0 \
    --rpe-config "${MRCR_DIR}/configs/rpe_config_mrcr_curriculum.yaml" \
    --train-file "${MRCR_DIR}/data/bin0_4K-8K/train.json" \
    --output-dir "${MRCR_DIR}/phase6/checkpoints/y2_rpe_cur_L32k" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

echo ""
echo "=== Y2-Rc32 training complete: $(date) ==="
