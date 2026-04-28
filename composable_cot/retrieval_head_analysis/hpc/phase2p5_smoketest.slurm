#!/bin/bash
#SBATCH --job-name=rha_p2p5
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=composable_cot/retrieval_head_analysis/slurm_logs/phase2p5_smoketest_%j.out
#SBATCH --error=composable_cot/retrieval_head_analysis/slurm_logs/phase2p5_smoketest_%j.err

# ============================================================================
# RHA Phase 2.5 — BABILongRetriever smoke test on one detection sample.
# Validates the prompt patch + span finder + forward pass before the full
# Phase 3 sweep over 36 (model × bin) cells.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
RHA_DIR="composable_cot/retrieval_head_analysis"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

# QRHead source + our scripts dir on PYTHONPATH so imports resolve.
export PYTHONPATH="${SCRATCH}/QRHead/src:${PROJECT_DIR}/${RHA_DIR}/scripts:${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/${RHA_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== RHA Phase 2.5 — BABILongRetriever smoke test ==="
echo "  Job:  ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Time: $(date)"
echo ""

python ${RHA_DIR}/scripts/phase2p5_smoketest.py

echo ""
echo "=== Phase 2.5 done: $(date) ==="
