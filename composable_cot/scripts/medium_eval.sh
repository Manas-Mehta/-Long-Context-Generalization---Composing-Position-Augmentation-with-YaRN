#!/bin/bash
# ============================================================================
# Medium Eval: 1 example per length (35-100) = 66 examples
# Skips lengths 1-34 (known to be perfect for all models).
# Runs baseline + curriculum (best RPE) first, then others if time allows.
# ~10 min per model. Total ~20 min for 2 models, ~40 min for all 4.
# ============================================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CCOT_ROOT="${PROJECT_ROOT}/composable_cot"
SCRIPTS="${CCOT_ROOT}/scripts"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MEDIUM_TEST="${CCOT_ROOT}/data/reverse_string_eval/test_medium.json"

# ── Create stratified test file: 1 example per length (35-100) ───────
MIN_LENGTH=35
echo "=== Creating medium test set (1 per length, ${MIN_LENGTH}-100) ==="

python3 -c "
import json

with open('${CCOT_ROOT}/data/reverse_string_eval/test_all.json') as f:
    data = json.load(f)

by_length = {}
for ex in data:
    L = ex.get('string_length', 0)
    by_length.setdefault(L, []).append(ex)

sampled = []
for L in range(${MIN_LENGTH}, 101):
    if L in by_length and by_length[L]:
        sampled.append(by_length[L][0])

print(f'Sampled {len(sampled)} examples (lengths ${MIN_LENGTH}-{max(s[\"string_length\"] for s in sampled)})')

with open('${MEDIUM_TEST}', 'w') as f:
    json.dump(sampled, f, indent=2)
print(f'Saved to ${MEDIUM_TEST}')
"

echo ""

# ── Helper ───────────────────────────────────────────────────────────
find_lora_dir() {
    local base_dir="$1"
    local found=$(find "${base_dir}" -name "adapter_config.json" -type f 2>/dev/null | head -1)
    if [ -n "$found" ]; then dirname "$found"; else echo "${base_dir}"; fi
}

run_eval() {
    local name="$1"
    local ckpt="$2"
    local out="$3"

    if [ ! -d "${ckpt}" ]; then
        echo "SKIP: ${name} — no checkpoint"
        return
    fi

    local lora_dir=$(find_lora_dir "${ckpt}")
    echo "============================================================"
    echo "MEDIUM EVAL: ${name}"
    echo "  Started: $(date)"
    echo "============================================================"

    python "${SCRIPTS}/eval_length_generalization.py" \
        --base-model "Qwen/Qwen2.5-7B" \
        --lora-ckpt "${lora_dir}" \
        --test-file "${MEDIUM_TEST}" \
        --output-dir "${out}" \
        --task-type reverse_string \
        --train-max-length 40 \
        --min-length ${MIN_LENGTH} \
        --max-new-tokens 2048

    echo "  Finished: $(date)"
    echo ""
}

# ── Priority 1: Baseline + Curriculum (most important comparison) ────
run_eval "Baseline (no RPE)" \
    "${CCOT_ROOT}/model_ckpt/reverse_string_baseline_rank16" \
    "${CCOT_ROOT}/outputs/medium_eval_baseline"

run_eval "Exp3: RPE curriculum" \
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_curriculum" \
    "${CCOT_ROOT}/outputs/medium_eval_rpe_curriculum"

# ── Priority 2: Other RPE variants (run if time allows) ─────────────
echo "=== Baseline + Curriculum done. Running remaining models... ==="
echo ""

run_eval "Exp1: RPE rank 16" \
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_rank16" \
    "${CCOT_ROOT}/outputs/medium_eval_rpe_rank16"

run_eval "Exp2: RPE asymmetric" \
    "${CCOT_ROOT}/model_ckpt/reverse_string_rpe_asymmetric" \
    "${CCOT_ROOT}/outputs/medium_eval_rpe_asymmetric"

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "MEDIUM EVAL SUMMARY"
echo "============================================================"
echo ""
printf "%-45s  %8s  %8s  %8s\n" "Experiment" "Overall" "In-Dist" "OOD"
printf "%-45s  %8s  %8s  %8s\n" "---------" "-------" "-------" "---"

for dir in \
    "${CCOT_ROOT}/outputs/medium_eval_baseline" \
    "${CCOT_ROOT}/outputs/medium_eval_rpe_rank16" \
    "${CCOT_ROOT}/outputs/medium_eval_rpe_asymmetric" \
    "${CCOT_ROOT}/outputs/medium_eval_rpe_curriculum"; do

    RESULTS="${dir}/eval_results.json"
    NAME=$(basename "$dir" | sed 's/medium_eval_//')
    if [ -f "${RESULTS}" ]; then
        python3 -c "
import json
with open('${RESULTS}') as f:
    r = json.load(f)
print(f'  {r[\"overall_accuracy\"]:.4f}    {r[\"in_dist_accuracy\"]:.4f}    {r[\"ood_accuracy\"]:.4f}')
" | while read line; do
            printf "%-45s %s\n" "${NAME}" "$line"
        done
    else
        printf "%-45s  %8s\n" "${NAME}" "(not run)"
    fi
done

echo ""
echo "============================================================"
