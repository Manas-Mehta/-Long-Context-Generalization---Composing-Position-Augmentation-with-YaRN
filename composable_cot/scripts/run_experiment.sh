#!/bin/bash
# ============================================================================
# RPE + Composable CoT: Full Experiment Pipeline
# ============================================================================
#
# Runs on Lightning AI (A100 or H100).
# Trains Qwen2.5-7B + LoRA on reverse_string with and without RPE,
# then evaluates length generalization on in-distribution and OOD test sets.
#
# Usage:
#   bash composable_cot/scripts/run_experiment.sh
#
# Prerequisites:
#   - CUDA GPU available
#   - Python environment with torch, transformers, peft, llamafactory installed
#
# Expected total runtime: ~1-2 hours on A100 (2 training runs + evaluation)
# ============================================================================

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
# Adjust PROJECT_ROOT to match your Lightning AI workspace
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CCOT_ROOT="${PROJECT_ROOT}/composable_cot"
LLAMA_FACTORY="${CCOT_ROOT}/LLaMA-Factory"
SCRIPTS="${CCOT_ROOT}/scripts"

# Add RPE to Python path so callbacks can import rpe.patching
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "RPE + Composable CoT Experiment"
echo "============================================================"
echo "  PROJECT_ROOT:   ${PROJECT_ROOT}"
echo "  CCOT_ROOT:      ${CCOT_ROOT}"
echo "  LLAMA_FACTORY:  ${LLAMA_FACTORY}"
echo "  PYTHONPATH:     ${PYTHONPATH}"
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

# Ensure peft is installed for evaluation
pip install peft -q 2>/dev/null || true

# Create output directories
mkdir -p "${CCOT_ROOT}/model_ckpt"
mkdir -p "${CCOT_ROOT}/outputs"

echo ""

# ── Step 1: Generate Data ─────────────────────────────────────────────
echo "=== Step 1: Generate Reverse String Data ==="

if [ -f "${CCOT_ROOT}/data/atomic/reverse_string_composable_cot/train.json" ]; then
    echo "Data already exists, skipping generation."
else
    python "${SCRIPTS}/generate_reverse_string_data.py"
fi

echo ""

# ── Step 2: Train Baseline (No RPE) ──────────────────────────────────
echo "=== Step 2: Train Baseline (No RPE) ==="

BASELINE_CKPT="${CCOT_ROOT}/model_ckpt/reverse_string_baseline"
BASELINE_CONFIG="${SCRIPTS}/llamafactory/reverse_string_baseline.yaml"

if [ -d "${BASELINE_CKPT}" ] && [ "$(ls -A ${BASELINE_CKPT} 2>/dev/null)" ]; then
    echo "Baseline checkpoint exists at ${BASELINE_CKPT}, skipping training."
else
    echo "Training baseline model (standard positions)..."
    cd "${LLAMA_FACTORY}"

    # No RPE_CONFIG_PATH = no RPE patching
    unset RPE_CONFIG_PATH 2>/dev/null || true
    llamafactory-cli train "${BASELINE_CONFIG}"

    cd "${PROJECT_ROOT}"
    echo "Baseline training complete."
fi

echo ""

# ── Step 3: Train RPE ────────────────────────────────────────────────
echo "=== Step 3: Train RPE ==="

RPE_CKPT="${CCOT_ROOT}/model_ckpt/reverse_string_rpe"
RPE_CONFIG="${SCRIPTS}/rpe_config.yaml"
RPE_TRAIN_CONFIG="${SCRIPTS}/llamafactory/reverse_string_rpe.yaml"

if [ -d "${RPE_CKPT}" ] && [ "$(ls -A ${RPE_CKPT} 2>/dev/null)" ]; then
    echo "RPE checkpoint exists at ${RPE_CKPT}, skipping training."
else
    echo "Training RPE model (randomized positions)..."
    cd "${LLAMA_FACTORY}"

    # Set RPE_CONFIG_PATH to enable RPE callback
    RPE_CONFIG_PATH="${RPE_CONFIG}" \
        llamafactory-cli train "${RPE_TRAIN_CONFIG}"

    cd "${PROJECT_ROOT}"
    echo "RPE training complete."
fi

echo ""

# ── Step 4: Evaluate Both Conditions ──────────────────────────────────
echo "=== Step 4: Evaluate Length Generalization ==="

EVAL_SCRIPT="${SCRIPTS}/eval_length_generalization.py"
TEST_FILE="${CCOT_ROOT}/data/reverse_string_eval/test_all.json"

# Find the actual LoRA adapter directory (LLaMA-Factory saves in checkpoint-* subdirs)
find_lora_dir() {
    local base_dir="$1"
    # Look for adapter_model.safetensors or adapter_config.json
    local found=$(find "${base_dir}" -name "adapter_config.json" -type f 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        dirname "$found"
    else
        echo "${base_dir}"
    fi
}

echo "--- Evaluating Baseline ---"
BASELINE_LORA=$(find_lora_dir "${BASELINE_CKPT}")
echo "  LoRA dir: ${BASELINE_LORA}"
python "${EVAL_SCRIPT}" \
    --base-model "Qwen/Qwen2.5-7B" \
    --lora-ckpt "${BASELINE_LORA}" \
    --test-file "${TEST_FILE}" \
    --output-dir "${CCOT_ROOT}/outputs/reverse_string_baseline_eval" \
    --task-type reverse_string \
    --train-max-length 40

echo ""
echo "--- Evaluating RPE ---"
RPE_LORA=$(find_lora_dir "${RPE_CKPT}")
echo "  LoRA dir: ${RPE_LORA}"
python "${EVAL_SCRIPT}" \
    --base-model "Qwen/Qwen2.5-7B" \
    --lora-ckpt "${RPE_LORA}" \
    --test-file "${TEST_FILE}" \
    --output-dir "${CCOT_ROOT}/outputs/reverse_string_rpe_eval" \
    --task-type reverse_string \
    --train-max-length 40

echo ""

# ── Step 5: Generate Comparison Plots ─────────────────────────────────
echo "=== Step 5: Generate Comparison Plots ==="
python "${SCRIPTS}/plot_results.py" \
    --baseline-results "${CCOT_ROOT}/outputs/reverse_string_baseline_eval/eval_results.json" \
    --rpe-results "${CCOT_ROOT}/outputs/reverse_string_rpe_eval/eval_results.json" \
    --output-dir "${CCOT_ROOT}/outputs/comparison_plots" \
    --train-max-length 40

echo ""

# ── Summary ───────────────────────────────────────────────────────────
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
echo "  Baseline eval:  ${CCOT_ROOT}/outputs/reverse_string_baseline_eval/eval_results.json"
echo "  RPE eval:       ${CCOT_ROOT}/outputs/reverse_string_rpe_eval/eval_results.json"
echo "  Plots:          ${CCOT_ROOT}/outputs/comparison_plots/"
echo ""
echo "To view results:"
echo "  cat ${CCOT_ROOT}/outputs/reverse_string_baseline_eval/eval_results.json | python -m json.tool | head -20"
echo "  cat ${CCOT_ROOT}/outputs/reverse_string_rpe_eval/eval_results.json | python -m json.tool | head -20"
echo "============================================================"
