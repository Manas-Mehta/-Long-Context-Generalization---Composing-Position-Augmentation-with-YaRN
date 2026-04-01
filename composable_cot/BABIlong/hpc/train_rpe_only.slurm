#!/bin/bash
#SBATCH --job-name=bab_rpe_only
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/train_rpe_only_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/train_rpe_only_%j.err
#SBATCH --requeue

# ============================================================================
# BABILong — Condition 5: RPE-only
# RPE curriculum L=16K only. No YaRN.
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

echo "=== BABILong Condition 5: RPE-only (RPE curriculum L=16K, no YaRN) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/train_babilong_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config "${BABILONG_DIR}/configs/rpe_config_babilong_curriculum_L16k.yaml" \
    --train-file "${BABILONG_DIR}/data/train/all_train_1k.json" \
    --eval-dir   "${BABILONG_DIR}/data/eval" \
    --output-dir "${BABILONG_DIR}/checkpoints/rpe_only_1k" \
    --max-seq-len 9216 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 5e-5 --epochs 3 --batch-size 1 --grad-accum 4 \
    --warmup-ratio 0.15 --max-grad-norm 1.0 --seed 42 \
    --eval-every 1000 --eval-samples 200 \
    --grad-log-every 100 \
    --wandb-run-name "rpe_only"

echo ""
echo "=== RPE-only training complete: $(date) ==="
echo ""
echo "=== Starting eval: rpe_only (vanilla RoPE at eval) ==="

python ${BABILONG_DIR}/scripts/eval_babilong.py \
    --checkpoint-dir "${BABILONG_DIR}/checkpoints/rpe_only_1k" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --eval-dir       "${BABILONG_DIR}/data/eval" \
    --output-dir     "${BABILONG_DIR}/results/rpe_only_1k" \
    --condition      "rpe_only_1k" \
    --max-samples    100 \
    --max-seq-len    131072 \
    --max-new-tokens 10

echo ""
echo "=== rpe_only DONE (train + eval): $(date) ==="
