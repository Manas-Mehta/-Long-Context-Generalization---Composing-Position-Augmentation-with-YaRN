#!/bin/bash
#SBATCH --job-name=bab_y2_base
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/train_y2_base_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/train_y2_base_%j.err
#SBATCH --requeue

# ============================================================================
# BABILong — Condition 2: Y2-base
# YaRN f=2 during training, f=4 at eval. Standard context extension baseline.
# No RPE, no PoSE. Everything else is measured against this.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
BABILONG_DIR="composable_cot/BABIlong"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
mkdir -p "${PROJECT_DIR}/${BABILONG_DIR}/slurm_logs"

cd "${PROJECT_DIR}"

echo "=== BABILong Condition 2: Y2-base (YaRN f=2) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/train_babilong_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 2.0 \
    --train-file "${BABILONG_DIR}/data/train/all_train.json" \
    --eval-dir   "${BABILONG_DIR}/data/eval" \
    --output-dir "${BABILONG_DIR}/checkpoints/y2_base" \
    --max-seq-len 9216 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 3 --batch-size 1 --grad-accum 4 \
    --warmup-ratio 0.05 --max-grad-norm 1.0 --seed 42 \
    --eval-every 500 --eval-samples 200 \
    --grad-log-every 100 \
    --wandb-run-name "y2_base"

echo ""
echo "=== Y2-base training complete: $(date) ==="
