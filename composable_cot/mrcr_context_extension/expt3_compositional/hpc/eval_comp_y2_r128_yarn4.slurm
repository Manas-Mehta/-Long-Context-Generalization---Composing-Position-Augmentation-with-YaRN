#!/bin/bash
#SBATCH --job-name=ev_y2r128_y4
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/expt3_compositional/slurm_logs/eval_comp_y2_r128_yarn4_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/expt3_compositional/slurm_logs/eval_comp_y2_r128_yarn4_%j.err
#SBATCH --requeue

# ============================================================================
# Experiment 3 — Eval: Comp-Y2-R128, SCALED-UP YaRN f=4 at eval
# TRIPLE EXTENSION: train YaRN 2x + train RPE to 128K + eval YaRN 4x
# This is the most exciting test — could push effective context beyond 128K
# without any additional training.
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

CONDITION="comp_y2_r128_yarn4"
CKPT="${MRCR_DIR}/expt3_compositional/checkpoints/comp_y2_r128"
BINS=("bin0_4K-8K" "bin1_8K-16K" "bin2_16K-32K" "bin3_32K-64K" "bin4_64K-128K")
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
EVAL_SCRIPT="${MRCR_DIR}/scripts/eval_mrcr.py"

echo "=== Experiment 3 Eval: ${CONDITION} (TRIPLE EXTENSION) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Checkpoint: ${CKPT}"
echo "  YaRN at eval: f=4.0 (SCALED UP from training f=2.0)"
echo ""

for BIN in "${BINS[@]}"; do
    echo ""
    echo "--- ${CONDITION} on ${BIN} ---"
    echo "  Start: $(date)"

    python ${EVAL_SCRIPT} \
        --base-model "${BASE_MODEL}" \
        --lora-ckpt "${CKPT}" \
        --enable-yarn --yarn-factor 4.0 \
        --test-file "${MRCR_DIR}/data/${BIN}/test.json" \
        --output-dir "${MRCR_DIR}/expt3_compositional/outputs/${CONDITION}_${BIN}" \
        --max-new-tokens 2048

    echo "  Done:  $(date)"
done

echo ""
echo "=== ${CONDITION} eval complete: $(date) ==="
