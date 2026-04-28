#!/bin/bash
#SBATCH --job-name=qrhead_phase0
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=composable_cot/retrieval_head_analysis/slurm_logs/phase0_smoketest_%j.out
#SBATCH --error=composable_cot/retrieval_head_analysis/slurm_logs/phase0_smoketest_%j.err

# ============================================================================
# RHA Phase 0 — verify qrhead env works on vanilla Qwen2.5-7B-Instruct.
# Submit from PROJECT_DIR: cd /scratch/mm14444/RPE && sbatch composable_cot/...
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
RHA_DIR="composable_cot/retrieval_head_analysis"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

# QRHead source lives at /scratch/mm14444/QRHead/src — prepend so qrretriever imports.
export PYTHONPATH="${SCRATCH}/QRHead/src:${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/${RHA_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== RHA Phase 0 — smoke test ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Python:    $(which python) $(python --version)"
echo "  HF_HOME:   ${HF_HOME}"
echo "  Timestamp: $(date)"
echo ""

python ${RHA_DIR}/scripts/phase0_smoketest.py

echo ""
echo "=== Phase 0 smoke test complete: $(date) ==="
