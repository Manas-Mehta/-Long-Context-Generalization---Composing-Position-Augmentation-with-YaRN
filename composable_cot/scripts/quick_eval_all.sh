#!/bin/bash
# ============================================================================
# Quick Evaluation: All 4 experiments, stratified sample (~20 min total)
# ============================================================================
#
# Samples 1 example per length at key checkpoints (every 5th length),
# giving 20 examples per model instead of 1000. Enough to see if the
# model learned anything, and whether OOD generalizes at all.
#
# Usage:
#   bash composable_cot/scripts/quick_eval_all.sh
#
# ============================================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CCOT_ROOT="${PROJECT_ROOT}/composable_cot"
SCRIPTS="${CCOT_ROOT}/scripts"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

QUICK_TEST="${CCOT_ROOT}/data/reverse_string_eval/test_quick.json"

# ── Step 1: Create stratified sample ─────────────────────────────────
echo "=== Creating stratified quick test set ==="

python3 -c "
import json

with open('${CCOT_ROOT}/data/reverse_string_eval/test_all.json') as f:
    data = json.load(f)

# Group by length
by_length = {}
for ex in data:
    L = ex.get('string_length', 0)
    by_length.setdefault(L, []).append(ex)

# Sample 1 example per every 5th length: 5, 10, 15, ..., 100
# Plus a few key lengths: 1, 40 (boundary), 41 (first OOD)
sample_lengths = [1] + list(range(5, 101, 5)) + [40, 41]
sample_lengths = sorted(set(sample_lengths))

sampled = []
for L in sample_lengths:
    if L in by_length and by_length[L]:
        sampled.append(by_length[L][0])  # first example at that length

print(f'Sampled {len(sampled)} examples across {len(sample_lengths)} lengths')
print(f'Lengths: {[s.get(\"string_length\", \"?\") for s in sampled]}')

with open('${QUICK_TEST}', 'w') as f:
    json.dump(sampled, f, indent=2)
print(f'Saved to ${QUICK_TEST}')
"

echo ""

# ── Step 2: Evaluate all 4 models ────────────────────────────────────

NAMES=(
    "Baseline (rank 16, no RPE)"
    "Exp1: RPE rank 16 L=1024"
    "Exp2: RPE asymmetric (Q/K=32, rest=8)"
    "Exp3: RPE curriculum (L: 256→1024)"
)

CKPTS=(
    "${CCOT_ROOT}/model_ckpt/reverse_string_baseline_rank16"
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_rank16"
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_asymmetric"
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_curriculum"
)

OUTPUT_DIRS=(
    "${CCOT_ROOT}/outputs/quick_eval_baseline"
    "${CCOT_ROOT}/outputs/quick_eval_rpe_rank16"
    "${CCOT_ROOT}/outputs/quick_eval_rpe_asymmetric"
    "${CCOT_ROOT}/outputs/quick_eval_rpe_curriculum"
)

# Helper to find LoRA adapter dir
find_lora_dir() {
    local base_dir="$1"
    local found=$(find "${base_dir}" -name "adapter_config.json" -type f 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        dirname "$found"
    else
        echo "${base_dir}"
    fi
}

for i in "${!NAMES[@]}"; do
    CKPT="${CKPTS[$i]}"
    if [ ! -d "${CKPT}" ]; then
        echo "SKIP: ${NAMES[$i]} — checkpoint not found at ${CKPT}"
        echo ""
        continue
    fi

    LORA_DIR=$(find_lora_dir "${CKPT}")

    echo "============================================================"
    echo "QUICK EVAL: ${NAMES[$i]}"
    echo "  LoRA: ${LORA_DIR}"
    echo "============================================================"

    python "${SCRIPTS}/eval_length_generalization.py" \
        --base-model "Qwen/Qwen2.5-7B" \
        --lora-ckpt "${LORA_DIR}" \
        --test-file "${QUICK_TEST}" \
        --output-dir "${OUTPUT_DIRS[$i]}" \
        --task-type reverse_string \
        --train-max-length 40 \
        --max-new-tokens 2048

    echo ""
done

# ── Summary table ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "QUICK EVAL SUMMARY"
echo "============================================================"
echo ""
printf "%-45s  %8s  %8s  %8s\n" "Experiment" "Overall" "In-Dist" "OOD"
printf "%-45s  %8s  %8s  %8s\n" "---------" "-------" "-------" "---"

for i in "${!NAMES[@]}"; do
    RESULTS="${OUTPUT_DIRS[$i]}/eval_results.json"
    if [ -f "${RESULTS}" ]; then
        python3 -c "
import json
with open('${RESULTS}') as f:
    r = json.load(f)
print(f'  {r[\"overall_accuracy\"]:.4f}    {r[\"in_dist_accuracy\"]:.4f}    {r[\"ood_accuracy\"]:.4f}')
" | while read line; do
            printf "%-45s %s\n" "${NAMES[$i]}" "$line"
        done
    else
        printf "%-45s  %8s\n" "${NAMES[$i]}" "(no results)"
    fi
done

echo ""
echo "============================================================"
echo "Done! Full eval can be run later with:"
echo "  bash composable_cot/scripts/run_three_experiments.sh eval"
echo "============================================================"
