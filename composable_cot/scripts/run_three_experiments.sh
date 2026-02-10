#!/bin/bash
# ============================================================================
# RPE Experiment Suite: 3 RPE Variations + Baseline
# ============================================================================
#
# Runs on Lightning AI (A100/H100).
#
# Experiments:
#   0. Baseline   — LoRA rank 16, NO RPE (control)
#   1. RPE Rank16 — LoRA rank 16, RPE L=1024 (standard RPE)
#   2. RPE Asym   — LoRA rank 32 on Q/K + rank 8 on rest, RPE L=1024
#   3. RPE Curric  — LoRA rank 16, curriculum L: 256→512→768→1024→1024
#
# Each experiment: ~20-30 min training + ~15 min eval on A100.
# Total: ~2-3 hours for all 4 runs.
#
# Usage:
#   # Run all experiments end-to-end
#   bash composable_cot/scripts/run_three_experiments.sh
#
#   # Run only a specific experiment (0-3)
#   bash composable_cot/scripts/run_three_experiments.sh 1
#
#   # Run only evaluation for all experiments
#   bash composable_cot/scripts/run_three_experiments.sh eval
#
# ============================================================================

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CCOT_ROOT="${PROJECT_ROOT}/composable_cot"
LLAMA_FACTORY="${CCOT_ROOT}/LLaMA-Factory"
SCRIPTS="${CCOT_ROOT}/scripts"
CONFIGS="${SCRIPTS}/llamafactory"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "RPE Experiment Suite (3 Variations + Baseline)"
echo "============================================================"
echo "  PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "  CCOT_ROOT:      ${CCOT_ROOT}"
echo "  PYTHONPATH:     ${PYTHONPATH}"
echo "  Timestamp:      $(date)"
echo ""

# ── Step 0: Environment Setup ─────────────────────────────────────────
echo "=== Step 0: Environment Setup ==="

if ! command -v llamafactory-cli &> /dev/null; then
    echo "Installing LLaMA-Factory..."
    cd "${LLAMA_FACTORY}"
    pip install -e ".[torch,metrics]" --no-build-isolation
    cd "${PROJECT_ROOT}"
else
    echo "LLaMA-Factory already installed."
fi

pip install peft -q 2>/dev/null || true

mkdir -p "${CCOT_ROOT}/model_ckpt"
mkdir -p "${CCOT_ROOT}/outputs"

echo ""

# ── Step 1: Generate Data (if needed) ─────────────────────────────────
echo "=== Step 1: Check/Generate Reverse String Data ==="

if [ -f "${CCOT_ROOT}/data/atomic/reverse_string_composable_cot/train.json" ]; then
    TRAIN_COUNT=$(python -c "import json; print(len(json.load(open('${CCOT_ROOT}/data/atomic/reverse_string_composable_cot/train.json'))))")
    echo "Training data exists: ${TRAIN_COUNT} examples."
else
    echo "Generating reverse string data..."
    python "${SCRIPTS}/generate_reverse_string_data.py"
fi

if [ -f "${CCOT_ROOT}/data/reverse_string_eval/test_all.json" ]; then
    TEST_COUNT=$(python -c "import json; print(len(json.load(open('${CCOT_ROOT}/data/reverse_string_eval/test_all.json'))))")
    echo "Eval data exists: ${TEST_COUNT} examples."
else
    echo "ERROR: Eval data not found at ${CCOT_ROOT}/data/reverse_string_eval/test_all.json"
    echo "Please ensure eval data is generated or uploaded."
    exit 1
fi

echo ""

# ── Helper: Find LoRA adapter directory ──────────────────────────────
find_lora_dir() {
    local base_dir="$1"
    local found=$(find "${base_dir}" -name "adapter_config.json" -type f 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        dirname "$found"
    else
        echo "${base_dir}"
    fi
}

# ── Helper: Train one experiment ─────────────────────────────────────
train_experiment() {
    local name="$1"
    local yaml_config="$2"
    local rpe_config="${3:-}"  # empty string = no RPE
    local ckpt_dir="$4"

    echo "------------------------------------------------------------"
    echo "TRAINING: ${name}"
    echo "  Config:     ${yaml_config}"
    echo "  RPE Config: ${rpe_config:-NONE (baseline)}"
    echo "  Output:     ${ckpt_dir}"
    echo "------------------------------------------------------------"

    if [ -d "${ckpt_dir}" ] && [ "$(ls -A ${ckpt_dir} 2>/dev/null)" ]; then
        echo "  Checkpoint exists, skipping. Delete ${ckpt_dir} to re-train."
        return 0
    fi

    cd "${LLAMA_FACTORY}"

    if [ -n "${rpe_config}" ]; then
        RPE_CONFIG_PATH="${rpe_config}" llamafactory-cli train "${yaml_config}"
    else
        unset RPE_CONFIG_PATH 2>/dev/null || true
        llamafactory-cli train "${yaml_config}"
    fi

    cd "${PROJECT_ROOT}"
    echo "  Training complete: ${name}"
    echo ""
}

# ── Helper: Evaluate one experiment ──────────────────────────────────
eval_experiment() {
    local name="$1"
    local ckpt_dir="$2"
    local output_dir="$3"

    echo "------------------------------------------------------------"
    echo "EVALUATING: ${name}"
    echo "  Checkpoint: ${ckpt_dir}"
    echo "  Output:     ${output_dir}"
    echo "------------------------------------------------------------"

    if [ ! -d "${ckpt_dir}" ]; then
        echo "  ERROR: Checkpoint not found at ${ckpt_dir}. Skipping eval."
        return 1
    fi

    local lora_dir=$(find_lora_dir "${ckpt_dir}")
    echo "  LoRA dir: ${lora_dir}"

    # max-new-tokens=2048: OOD lengths 50-100 need 600-1200+ tokens for full CoT trace.
    # Default 512 truncates before "the answer is ..." causing false 0% accuracy.
    python "${SCRIPTS}/eval_length_generalization.py" \
        --base-model "Qwen/Qwen2.5-7B" \
        --lora-ckpt "${lora_dir}" \
        --test-file "${CCOT_ROOT}/data/reverse_string_eval/test_all.json" \
        --output-dir "${output_dir}" \
        --task-type reverse_string \
        --train-max-length 40 \
        --max-new-tokens 2048

    echo "  Eval complete: ${name}"
    echo ""
}

# ── Experiment Definitions ───────────────────────────────────────────
# Format: NAME | YAML_CONFIG | RPE_CONFIG | CKPT_DIR

EXP_NAMES=(
    "Baseline (rank 16, no RPE)"
    "Exp1: RPE rank 16 L=1024"
    "Exp2: RPE asymmetric (Q/K=32, rest=8) L=1024"
    "Exp3: RPE curriculum (L: 256→1024)"
)

EXP_YAMLS=(
    "${CONFIGS}/reverse_string_baseline_rank16.yaml"
    "${CONFIGS}/reverse_string_rpe_rank16.yaml"
    "${CONFIGS}/reverse_string_rpe_asymmetric.yaml"
    "${CONFIGS}/reverse_string_rpe_curriculum.yaml"
)

EXP_RPE_CONFIGS=(
    ""
    "${SCRIPTS}/rpe_config_L1024.yaml"
    "${SCRIPTS}/rpe_config_L1024.yaml"
    "${SCRIPTS}/rpe_config_curriculum.yaml"
)

EXP_CKPTS=(
    "${CCOT_ROOT}/model_ckpt/reverse_string_baseline_rank16"
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_rank16"
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_asymmetric"
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_curriculum"
)

EXP_EVAL_DIRS=(
    "${CCOT_ROOT}/outputs/reverse_string_baseline_rank16_eval"
    "${CCOT_ROOT}/outputs/reverse_string_rpe_rank16_eval"
    "${CCOT_ROOT}/outputs/reverse_string_rpe_asymmetric_eval"
    "${CCOT_ROOT}/outputs/reverse_string_rpe_curriculum_eval"
)

# ── Determine What to Run ────────────────────────────────────────────
RUN_MODE="${1:-all}"

if [ "${RUN_MODE}" = "eval" ]; then
    echo "=== Running Evaluation Only ==="
    for i in "${!EXP_NAMES[@]}"; do
        eval_experiment "${EXP_NAMES[$i]}" "${EXP_CKPTS[$i]}" "${EXP_EVAL_DIRS[$i]}" || true
    done
elif [[ "${RUN_MODE}" =~ ^[0-3]$ ]]; then
    i="${RUN_MODE}"
    echo "=== Running Single Experiment: ${EXP_NAMES[$i]} ==="
    train_experiment "${EXP_NAMES[$i]}" "${EXP_YAMLS[$i]}" "${EXP_RPE_CONFIGS[$i]}" "${EXP_CKPTS[$i]}"
    eval_experiment "${EXP_NAMES[$i]}" "${EXP_CKPTS[$i]}" "${EXP_EVAL_DIRS[$i]}"
else
    echo "=== Running All Experiments ==="
    for i in "${!EXP_NAMES[@]}"; do
        train_experiment "${EXP_NAMES[$i]}" "${EXP_YAMLS[$i]}" "${EXP_RPE_CONFIGS[$i]}" "${EXP_CKPTS[$i]}"
    done

    echo ""
    echo "=== All Training Complete. Starting Evaluation ==="
    echo ""

    for i in "${!EXP_NAMES[@]}"; do
        eval_experiment "${EXP_NAMES[$i]}" "${EXP_CKPTS[$i]}" "${EXP_EVAL_DIRS[$i]}" || true
    done
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "EXPERIMENT SUITE COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
for i in "${!EXP_NAMES[@]}"; do
    echo "  ${EXP_NAMES[$i]}:"
    RESULTS_FILE="${EXP_EVAL_DIRS[$i]}/eval_results.json"
    if [ -f "${RESULTS_FILE}" ]; then
        python -c "
import json
with open('${RESULTS_FILE}') as f:
    r = json.load(f)
print(f'    Overall: {r[\"overall_accuracy\"]:.4f}')
print(f'    In-dist: {r[\"in_dist_accuracy\"]:.4f}')
print(f'    OOD:     {r[\"ood_accuracy\"]:.4f}')
"
    else
        echo "    (no results yet)"
    fi
done

echo ""
echo "To re-run a specific experiment, delete its checkpoint and run:"
echo "  bash composable_cot/scripts/run_three_experiments.sh <0|1|2|3>"
echo ""
echo "To run evaluation only:"
echo "  bash composable_cot/scripts/run_three_experiments.sh eval"
echo "============================================================"
