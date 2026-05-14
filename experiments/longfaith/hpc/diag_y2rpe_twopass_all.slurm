#!/bin/bash
#SBATCH --job-name=lf_y2rpe_tp_all
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/lf_y2rpe_tp_all_%j.out
#SBATCH --error=slurm_logs/lf_y2rpe_tp_all_%j.err
#SBATCH --requeue

# ============================================================================
# DIAGNOSTIC for mismatch #1 (output format / CoT) — y2_rpe_cur_L16k headline,
# THUDM two-pass CoT protocol. Reuses the y2_rpe_cur_L16k checkpoint (no
# retraining). Tests whether the composition's deficit vs zero-shot is driven
# by single-pass CoT format collapse (preference drift in pass 1 -> wrong
# letter at extraction time).
# ALL 4 BINS (206 examples). Companion: diag_zs_yarn4_twopass_all.slurm.
# See notes/longbench_v2_protocol_match.md and §16 of HANDOFF.md.
# Wall-time: pass-1 generates 1024 tokens + short pass-2 — budget 6-10h on L40S.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/positionaug-YaRN"
EXPT_DIR="experiments/longfaith"
CKPT="y2_rpe_cur_L16k"
COND="diag_y2rpe_twopass_all"

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

if [ ! -d "${EXPT_DIR}/checkpoints/${CKPT}" ]; then
    echo "ERROR: ${EXPT_DIR}/checkpoints/${CKPT} does not exist."
    exit 1
fi

python ${EXPT_DIR}/scripts/eval_longbench_v2.py \
    --checkpoint-dir "${EXPT_DIR}/checkpoints/${CKPT}" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn    --yarn-factor 4.0 \
    --data-dir       "${EXPT_DIR}/data" \
    --output-dir     "${EXPT_DIR}/results/${COND}" \
    --condition      "${COND}" \
    --bins           16k 32k 64k 128k \
    --max-seq-len    131072 \
    --max-new-tokens 1024 \
    --two-pass

echo "=== ${COND} DONE: $(date) ==="
