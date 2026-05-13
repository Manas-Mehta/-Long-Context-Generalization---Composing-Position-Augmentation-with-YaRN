#!/bin/bash
#SBATCH --job-name=ev_y3r128_y3
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/expt3_compositional/slurm_logs/eval_comp_y3_r128_yarn3_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/expt3_compositional/slurm_logs/eval_comp_y3_r128_yarn3_%j.err
#SBATCH --requeue

# ============================================================================
# Experiment 3 — Eval: Comp-Y3-R128, matching YaRN f=3 at eval
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

CONDITION="comp_y3_r128_yarn3"
CKPT="${MRCR_DIR}/expt3_compositional/checkpoints/comp_y3_r128"
BINS=("bin0_4K-8K" "bin1_8K-16K" "bin2_16K-32K" "bin3_32K-64K" "bin4_64K-128K")
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
EVAL_SCRIPT="${MRCR_DIR}/scripts/eval_mrcr.py"

echo "=== Experiment 3 Eval: ${CONDITION} ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Checkpoint: ${CKPT}"
echo "  YaRN at eval: f=3.0 (matching training)"
echo ""

for BIN in "${BINS[@]}"; do
    echo ""
    echo "--- ${CONDITION} on ${BIN} ---"
    echo "  Start: $(date)"

    python ${EVAL_SCRIPT} \
        --base-model "${BASE_MODEL}" \
        --lora-ckpt "${CKPT}" \
        --enable-yarn --yarn-factor 3.0 \
        --test-file "${MRCR_DIR}/data/${BIN}/test.json" \
        --output-dir "${MRCR_DIR}/expt3_compositional/outputs/${CONDITION}_${BIN}" \
        --max-new-tokens 2048

    echo "  Done:  $(date)"
done

echo ""
echo "=== ${CONDITION} eval complete: $(date) ==="
