#!/bin/bash
#SBATCH --job-name=mrcr_eval_lora
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/mrcr_eval_lora_%j.out
#SBATCH --error=slurm_logs/mrcr_eval_lora_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 3 — Evaluate ALL 4 LoRA conditions on bins 0-4
#
# Conditions:
#   1. lora_baseline     — normal RoPE
#   2. yarn_lora         — YaRN at load + LoRA (must pass --enable-yarn)
#   3. rpe_lora          — RPE-trained LoRA (eval with standard positions)
#   4. rpe_curriculum_lora — RPE curriculum LoRA (eval with standard positions)
#
# Each condition x 5 bins = 20 eval runs total.
# Estimated time: ~16h (bins 0: ~20min, 1: ~40min, 2: ~90min, 3: ~2h, 4: ~3h per cond)
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

echo "=== MRCR Phase 3: LoRA Evaluation (all conditions, bins 0-4) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

BINS=("bin0_4K-8K" "bin1_8K-16K" "bin2_16K-32K" "bin3_32K-64K" "bin4_64K-128K")
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
EVAL_SCRIPT="${MRCR_DIR}/scripts/eval_mrcr.py"

# ============================================================================
# Helper function
# ============================================================================
run_eval() {
    local CONDITION="$1"
    local CKPT="$2"
    local YARN_FLAG="$3"  # "--enable-yarn --yarn-factor 4.0" or ""

    echo ""
    echo "================================================================"
    echo "  Evaluating: ${CONDITION}"
    echo "  Checkpoint: ${CKPT}"
    echo "  YaRN: ${YARN_FLAG:-none}"
    echo "================================================================"

    for BIN in "${BINS[@]}"; do
        echo ""
        echo "--- ${CONDITION} on ${BIN} ---"
        echo "  Start: $(date)"

        python ${EVAL_SCRIPT} \
            --base-model "${BASE_MODEL}" \
            --lora-ckpt "${CKPT}" \
            ${YARN_FLAG} \
            --test-file "${MRCR_DIR}/data/${BIN}/test.json" \
            --output-dir "${MRCR_DIR}/outputs/${CONDITION}_${BIN}" \
            --max-new-tokens 2048

        echo "  Done:  $(date)"
    done
}

# ============================================================================
# Run all conditions
# ============================================================================

# Condition 1: LoRA Baseline
run_eval "lora_baseline" \
    "${MRCR_DIR}/checkpoints/lora_baseline" \
    ""

# Condition 2: YaRN+LoRA
run_eval "yarn_lora" \
    "${MRCR_DIR}/checkpoints/yarn_lora" \
    "--enable-yarn --yarn-factor 4.0"

# Condition 3: RPE+LoRA (fixed L)
run_eval "rpe_lora" \
    "${MRCR_DIR}/checkpoints/rpe_lora" \
    ""

# Condition 4: RPE+LoRA (curriculum)
run_eval "rpe_curriculum_lora" \
    "${MRCR_DIR}/checkpoints/rpe_curriculum_lora" \
    ""

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "================================================================"
echo "  ALL EVALUATIONS COMPLETE: $(date)"
echo "================================================================"
echo ""
echo "Results saved in ${MRCR_DIR}/outputs/"
echo ""

# Print summary table
echo "=== RESULTS SUMMARY ==="
echo ""
printf "%-25s %8s %8s %8s %8s %8s\n" "Condition" "4K-8K" "8K-16K" "16K-32K" "32K-64K" "64K-128K"
printf "%-25s %8s %8s %8s %8s %8s\n" "-------------------------" "--------" "--------" "--------" "--------" "--------"

for COND in "lora_baseline" "yarn_lora" "rpe_lora" "rpe_curriculum_lora"; do
    SCORES=""
    for BIN in "${BINS[@]}"; do
        RESULT_FILE="${MRCR_DIR}/outputs/${COND}_${BIN}/eval_results.json"
        if [ -f "${RESULT_FILE}" ]; then
            SCORE=$(python -c "import json; d=json.load(open('${RESULT_FILE}')); print(f'{d[\"overall_score\"]:.3f}')")
        else
            SCORE="N/A"
        fi
        SCORES="${SCORES} $(printf '%8s' ${SCORE})"
    done
    printf "%-25s%s\n" "${COND}" "${SCORES}"
done

echo ""
echo "=== Done ==="
