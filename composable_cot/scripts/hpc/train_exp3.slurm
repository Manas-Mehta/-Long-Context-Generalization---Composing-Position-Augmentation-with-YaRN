#!/bin/bash
#SBATCH --job-name=rpe_curric
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/exp3_rpe_curriculum_%j.out
#SBATCH --error=slurm_logs/exp3_rpe_curriculum_%j.err

# ============================================================================
# Experiment 3: RPE curriculum (L: 256 → 512 → 768 → 1024 → 1024)
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
export HF_HUB_OFFLINE=1

cd "${PROJECT_DIR}/composable_cot/LLaMA-Factory"

echo "=== Exp 3: RPE curriculum (L: 256→1024) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

# Set RPE config — curriculum version
export RPE_CONFIG_PATH="${PROJECT_DIR}/composable_cot/scripts/rpe_config_curriculum.yaml"
echo "  RPE_CONFIG_PATH: ${RPE_CONFIG_PATH}"

llamafactory-cli train "${PROJECT_DIR}/composable_cot/scripts/llamafactory/reverse_string_rpe_curriculum.yaml"

echo ""
echo "=== Exp 3 training complete: $(date) ==="
