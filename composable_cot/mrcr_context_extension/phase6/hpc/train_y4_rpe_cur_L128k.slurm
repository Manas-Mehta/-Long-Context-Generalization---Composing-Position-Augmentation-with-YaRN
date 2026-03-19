#!/bin/bash
#SBATCH --job-name=p6_Y4Rc128
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_y4_rpe_cur_L128k_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_y4_rpe_cur_L128k_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 6 — Group D (#14): Y4-Rc128
# YaRN f=4.0 + RPE curriculum L=128K. L matches YaRN's 128K window exactly.
# Full coverage of YaRN-extended space. Maximum extension.
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

echo "=== Phase 6 Group D: Y4-Rc128 (YaRN f=4 + RPE cur L=128K) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  L vs YaRN window: 128K = matches 128K exactly (full coverage)"
echo ""

python ${MRCR_DIR}/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --rpe-config "${MRCR_DIR}/configs/rpe_config_mrcr_curriculum_L128k.yaml" \
    --train-file "${MRCR_DIR}/data/bin0_4K-8K/train.json" \
    --output-dir "${MRCR_DIR}/phase6/checkpoints/y4_rpe_cur_L128k" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

echo ""
echo "=== Y4-Rc128 training complete: $(date) ==="
