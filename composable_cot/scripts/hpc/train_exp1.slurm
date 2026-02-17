#!/bin/bash
#SBATCH --job-name=rpe_rank16
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/exp1_rpe_rank16_%j.out
#SBATCH --error=slurm_logs/exp1_rpe_rank16_%j.err

# ============================================================================
# Experiment 1: RPE rank 16, L=1024
# ============================================================================

set -euo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"

cd "${PROJECT_DIR}/composable_cot/LLaMA-Factory"

echo "=== Exp 1: RPE rank 16 L=1024 ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

# Set RPE config
export RPE_CONFIG_PATH="${PROJECT_DIR}/composable_cot/scripts/rpe_config_L1024.yaml"
echo "  RPE_CONFIG_PATH: ${RPE_CONFIG_PATH}"

llamafactory-cli train "${PROJECT_DIR}/composable_cot/scripts/llamafactory/reverse_string_rpe_rank16.yaml"

echo ""
echo "=== Exp 1 training complete: $(date) ==="
