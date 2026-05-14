#!/bin/bash
#SBATCH --job-name=lf_rpe_yarn4
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/lf_rpe_yarn4_%j.out
#SBATCH --error=slurm_logs/lf_rpe_yarn4_%j.err

# ============================================================================
# Eval-only Row 6: rpe_only_yarn4 (RPE / YaRN)
#   Reuses rpe_only checkpoint (must run run_rpe_only.slurm first).
#   Eval: YaRN f=4
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/positionaug-YaRN"
EXPT_DIR="experiments/longfaith"
CKPT="rpe_only"
COND="rpe_only_yarn4"

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
echo "  Reusing checkpoint: ${EXPT_DIR}/checkpoints/${CKPT}"
echo ""

# Verify the rpe_only checkpoint exists
if [ ! -d "${EXPT_DIR}/checkpoints/${CKPT}" ]; then
    echo "ERROR: ${EXPT_DIR}/checkpoints/${CKPT} does not exist."
    echo "Run run_rpe_only.slurm first."
    exit 1
fi

# --- Eval (all 4 bins, YaRN f=4 on the rpe_only checkpoint) ---
python ${EXPT_DIR}/scripts/eval_longbench_v2.py \
    --checkpoint-dir "${EXPT_DIR}/checkpoints/${CKPT}" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn    --yarn-factor 4.0 \
    --data-dir       "${EXPT_DIR}/data" \
    --output-dir     "${EXPT_DIR}/results/${COND}" \
    --condition      "${COND}" \
    --bins           16k 32k 64k 128k \
    --max-seq-len    131072 \
    --max-new-tokens 1024

echo "=== ${COND} DONE: $(date) ==="
