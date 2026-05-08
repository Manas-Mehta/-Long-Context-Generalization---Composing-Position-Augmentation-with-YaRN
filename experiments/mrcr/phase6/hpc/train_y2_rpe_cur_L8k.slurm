#!/bin/bash
#SBATCH --job-name=p6_Y2Rc8
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_y2_rpe_cur_L8k_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/phase6/slurm_logs/train_y2_rpe_cur_L8k_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 6 — Group C (#6): Y2-Rc8
# YaRN f=2.0 + RPE curriculum L=8K. Mid-range L + modest YaRN.
# L << YaRN window (8K << 64K): RPE for position robustness only.
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

echo "=== Phase 6 Group C: Y2-Rc8 (YaRN f=2 + RPE cur L=8K) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${MRCR_DIR}/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 2.0 \
    --rpe-config "${MRCR_DIR}/configs/rpe_config_mrcr_curriculum_L8k.yaml" \
    --train-file "${MRCR_DIR}/data/bin0_4K-8K/train.json" \
    --output-dir "${MRCR_DIR}/phase6/checkpoints/y2_rpe_cur_L8k" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

echo ""
echo "=== Y2-Rc8 training complete: $(date) ==="
