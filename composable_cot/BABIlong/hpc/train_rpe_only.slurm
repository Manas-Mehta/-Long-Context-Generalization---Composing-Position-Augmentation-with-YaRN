#!/bin/bash
#SBATCH --job-name=bab_rpe_only
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/train_rpe_only_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/train_rpe_only_%j.err
#SBATCH --requeue

# ============================================================================
# BABILong — Condition 5: RPE-only
# RPE curriculum L=16K only. No YaRN — default RoPE frequencies.
# Ablation: how much of the Y2-Rc16 gain comes from RPE alone vs YaRN?
# Expected: weaker than Y2-Rc16 at 64K-128K (no frequency extension).
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
export WANDB_MODE=offline
export WANDB_DIR="${SCRATCH}/wandb_logs"
export WANDB_PROJECT="babilong_rpe"

mkdir -p "${SCRATCH}/wandb_logs"
mkdir -p "${PROJECT_DIR}/${BABILONG_DIR}/slurm_logs"

cd "${PROJECT_DIR}"

echo "=== BABILong Condition 5: RPE-only (RPE curriculum L=16K, no YaRN) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/train_babilong_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config "${BABILONG_DIR}/configs/rpe_config_babilong_curriculum_L16k.yaml" \
    --train-file "${BABILONG_DIR}/data/train/all_train.json" \
    --eval-dir   "${BABILONG_DIR}/data/eval" \
    --output-dir "${BABILONG_DIR}/checkpoints/rpe_only" \
    --max-seq-len 9216 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 3 --batch-size 1 --grad-accum 4 \
    --warmup-ratio 0.05 --max-grad-norm 1.0 --seed 42 \
    --eval-every 500 --eval-samples 200 \
    --grad-log-every 100 \
    --wandb-run-name "rpe_only"

echo ""
echo "=== RPE-only training complete: $(date) ==="
