#!/bin/bash
#SBATCH --job-name=p6_eval
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/phase6/slurm_logs/eval_%x_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/phase6/slurm_logs/eval_%x_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 6 — Parameterized eval script
# Usage: sbatch --job-name=<name> --export=CKPT=<path>,YARN_FLAG=<flag>,CONDITION=<name> eval_phase6.slurm
#
# CKPT:      checkpoint dir name under phase6/checkpoints/ (e.g. y2_rpe_cur_L16k)
# YARN_FLAG: "" for no yarn, "--enable-yarn --yarn-factor 2.0" etc.
# CONDITION: output subdirectory name (e.g. y2_rpe_cur_L16k_yarn2)
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
MRCR_DIR="composable_cot/mrcr_context_extension"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1

cd "${PROJECT_DIR}"

CKPT_PATH="${MRCR_DIR}/phase6/checkpoints/${CKPT}"
BINS=("bin0_4K-8K" "bin1_8K-16K" "bin2_16K-32K" "bin3_32K-64K" "bin4_64K-128K")
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
EVAL_SCRIPT="${MRCR_DIR}/scripts/eval_mrcr.py"

echo "=== Phase 6 Eval: ${CONDITION} ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  YaRN flag:  ${YARN_FLAG:-none}"
echo ""

for BIN in "${BINS[@]}"; do
    echo ""
    echo "--- ${CONDITION} on ${BIN} ---"
    echo "  Start: $(date)"

    python ${EVAL_SCRIPT} \
        --base-model "${BASE_MODEL}" \
        --lora-ckpt "${CKPT_PATH}" \
        ${YARN_FLAG} \
        --test-file "${MRCR_DIR}/data/${BIN}/test.json" \
        --output-dir "${MRCR_DIR}/phase6/outputs/${CONDITION}_${BIN}" \
        --max-new-tokens 2048

    echo "  Done:  $(date)"
done

echo ""
echo "=== ${CONDITION} eval complete: $(date) ==="
