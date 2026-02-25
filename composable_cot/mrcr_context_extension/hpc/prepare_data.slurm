#!/bin/bash
#SBATCH --job-name=mrcr_prep
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/mrcr_prep_%j.out
#SBATCH --error=slurm_logs/mrcr_prep_%j.err
#SBATCH --requeue

# ============================================================================
# MRCR Data Preparation: Download, tokenize, bin, split
# GPU requested to match standard HPC job allocation
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"

cd "${PROJECT_DIR}"

echo "=== MRCR Data Preparation ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Timestamp: $(date)"
echo ""

# Download and prepare data for bins 0 (4K-8K), 1 (8K-16K), 2 (16K-32K)
# Using 2-needle subset only, 70/30 train/test split
python composable_cot/mrcr_context_extension/scripts/prepare_data.py \
    --tokenizer "Qwen/Qwen2.5-7B-Instruct" \
    --output-dir "composable_cot/mrcr_context_extension/data" \
    --train-ratio 0.7 \
    --n-needles 2 \
    --max-bin 2

echo ""
echo "=== Data preparation complete: $(date) ==="
