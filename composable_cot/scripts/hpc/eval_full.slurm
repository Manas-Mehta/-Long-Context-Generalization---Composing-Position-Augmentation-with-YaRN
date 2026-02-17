#!/bin/bash
#SBATCH --job-name=rpe_eval_full
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/eval_full_%j.out
#SBATCH --error=slurm_logs/eval_full_%j.err

# ============================================================================
# Full evaluation of all 4 experiments (1000 examples each, ~7 hours)
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

echo "=== Full Eval: All 4 experiments (1000 examples each) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

bash composable_cot/scripts/run_three_experiments.sh eval

echo ""
echo "=== Full eval complete: $(date) ==="
