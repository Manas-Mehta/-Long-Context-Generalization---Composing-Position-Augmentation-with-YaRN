#!/bin/bash
#SBATCH --job-name=eval_base_needle_v2
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/eval_y2_base_needle_v2_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/eval_y2_base_needle_v2_%j.err

# ============================================================================
# BABILong Eval — YaRN-only on needle-position subset v2
# 227 samples × 8 bins × 3 zones = 5,448 evals
# Trained: YaRN f=2 | Eval: YaRN f=4
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
BABILONG_DIR="composable_cot/BABIlong"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1

mkdir -p "${PROJECT_DIR}/${BABILONG_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== BABILong Eval: y2_base — needle-position v2 (5448 samples) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

for ZONE in beg mid end; do
    echo "=============================================="
    echo "ZONE: ${ZONE}"
    echo "=============================================="

    TMP_DIR="${PROJECT_DIR}/${BABILONG_DIR}/data/eval_needle_v2/_tmp_${ZONE}"
    mkdir -p "${TMP_DIR}"
    rm -f "${TMP_DIR}"/*.json
    for BIN in 0k 1k 2k 4k 8k 16k 32k 64k 128k; do
        if [ -f "${PROJECT_DIR}/${BABILONG_DIR}/data/eval_needle_v2/${ZONE}_${BIN}.json" ]; then
            ln -sf "${PROJECT_DIR}/${BABILONG_DIR}/data/eval_needle_v2/${ZONE}_${BIN}.json" \
                   "${TMP_DIR}/${BIN}.json"
        fi
    done

    python ${BABILONG_DIR}/scripts/eval_babilong.py \
        --checkpoint-dir "${PROJECT_DIR}/${BABILONG_DIR}/checkpoints/y2_base_1k/checkpoint-2000" \
        --base-model     "Qwen/Qwen2.5-7B-Instruct" \
        --enable-yarn    --yarn-factor 4.0 \
        --eval-dir       "${TMP_DIR}" \
        --output-dir     "${BABILONG_DIR}/results/y2_base_needle_v2_${ZONE}" \
        --condition      "y2_base_needle_v2_${ZONE}" \
        --max-samples    0 \
        --max-seq-len    131072 \
        --max-new-tokens 10
done

echo ""
echo "=== y2_base needle v2 eval complete: $(date) ==="
