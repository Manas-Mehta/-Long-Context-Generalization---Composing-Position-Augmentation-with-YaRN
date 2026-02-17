#!/bin/bash
#SBATCH --job-name=rpe_baseline
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/exp0_baseline_%j.out
#SBATCH --error=slurm_logs/exp0_baseline_%j.err

# ============================================================================
# Experiment 0: Baseline (LoRA rank 16, NO RPE)
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"

cd "${PROJECT_DIR}/composable_cot/LLaMA-Factory"

echo "=== Exp 0: Baseline (rank 16, no RPE) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

# Unset RPE — this is the baseline
unset RPE_CONFIG_PATH 2>/dev/null || true

llamafactory-cli train "${PROJECT_DIR}/composable_cot/scripts/llamafactory/reverse_string_baseline_rank16.yaml"

echo ""
echo "=== Exp 0 training complete: $(date) ==="
