#!/bin/bash
#SBATCH --job-name=eval_lora_base
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/eval_lora_base_l40s_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/eval_lora_base_l40s_%j.err

# ============================================================================
# BABILong Eval — Condition 1: LoRA-base  [L40S version]
# Vanilla RoPE at eval (no YaRN). Will collapse at 32K+.
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

echo "=== BABILong Eval: lora_base [L40S] ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/eval_babilong.py \
    --checkpoint-dir "${PROJECT_DIR}/${BABILONG_DIR}/checkpoints/lora_base/checkpoint-5000" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --eval-dir       "${BABILONG_DIR}/data/eval" \
    --output-dir     "${BABILONG_DIR}/results/lora_base" \
    --condition      "lora_base" \
    --max-samples    100 \
    --max-seq-len    131072 \
    --max-new-tokens 10

echo ""
echo "=== lora_base eval complete: $(date) ==="
