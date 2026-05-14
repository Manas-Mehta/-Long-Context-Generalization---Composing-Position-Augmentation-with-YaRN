#!/bin/bash
#SBATCH --job-name=lf_zs_nc_all
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/lf_zs_nc_all_%j.out
#SBATCH --error=slurm_logs/lf_zs_nc_all_%j.err
#SBATCH --requeue

# ============================================================================
# DIAGNOSTIC for mismatch #2 (chunking-into-20) — zero_shot_yarn4 baseline.
# Tests: does feeding raw context (no [1]..[20] markers) move accuracy?
# ALL 4 BINS (206 examples). Companion: diag_y2rpe_nochunk_all.slurm.
# See notes/longfaith_diagnosis_2026-05-15.md and §15 of HANDOFF.md.
# Expected wall-time: ~3-5h on L40S, ~2-3h on H200.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/positionaug-YaRN"
EXPT_DIR="experiments/longfaith"
COND="diag_zs_yarn4_nochunk_all"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${PROJECT_DIR}/${EXPT_DIR}/results/${COND}"

cd "${PROJECT_DIR}"

echo "=== ${COND} ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Timestamp: $(date)"
echo ""

python ${EXPT_DIR}/scripts/eval_longbench_v2.py \
    --no-lora \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn    --yarn-factor 4.0 \
    --data-dir       "${EXPT_DIR}/data" \
    --output-dir     "${EXPT_DIR}/results/${COND}" \
    --condition      "${COND}" \
    --bins           16k 32k 64k 128k \
    --max-seq-len    131072 \
    --max-new-tokens 1024 \
    --no-chunk

echo "=== ${COND} DONE: $(date) ==="
