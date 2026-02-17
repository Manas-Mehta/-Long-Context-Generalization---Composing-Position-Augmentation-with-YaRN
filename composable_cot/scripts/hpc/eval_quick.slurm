#!/bin/bash
#SBATCH --job-name=rpe_eval
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/eval_quick_%j.out
#SBATCH --error=slurm_logs/eval_quick_%j.err

# ============================================================================
# Quick evaluation of all 4 experiments (stratified sample, ~20 min total)
# ============================================================================

set -euo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"

# Activate conda env
source activate "${SCRATCH}/conda_envs/rpe" 2>/dev/null || conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PROJECT_ROOT="${PROJECT_DIR}"
export HF_HOME="${SCRATCH}/hf_cache"

cd "${PROJECT_DIR}"

echo "=== Quick Eval: All 4 experiments ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

bash composable_cot/scripts/quick_eval_all.sh

echo ""
echo "=== Quick eval complete: $(date) ==="
