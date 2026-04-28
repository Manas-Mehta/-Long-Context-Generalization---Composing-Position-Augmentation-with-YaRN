#!/bin/bash
#SBATCH --job-name=rha_p3
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=composable_cot/retrieval_head_analysis/slurm_logs/phase3_%x_%j.out
#SBATCH --error=composable_cot/retrieval_head_analysis/slurm_logs/phase3_%x_%j.err

# ============================================================================
# RHA Phase 3 — run BABILongRetriever-based QR detection across all 9 bins
# for ONE condition. Submit four times, one per condition.
#
# Required env var:
#   COND : one of {vanilla_qwen, lora_base, y2_base, y2_rpe_cur_L16k}
#
# Submit:
#   cd /scratch/mm14444/RPE
#   sbatch --job-name=p3_vanilla --export=COND=vanilla_qwen \
#       composable_cot/retrieval_head_analysis/hpc/phase3_run_detection.slurm
#   sbatch --job-name=p3_lora    --export=COND=lora_base ...
#   sbatch --job-name=p3_y2      --export=COND=y2_base ...
#   sbatch --job-name=p3_rpe     --export=COND=y2_rpe_cur_L16k ...
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
RHA_DIR="composable_cot/retrieval_head_analysis"
MERGED_DIR="${SCRATCH}/merged_for_qrhead"

if [ -z "${COND:-}" ]; then
    echo "ERROR: COND env var is required" >&2
    echo "Submit with: sbatch --export=COND=<cond> ..." >&2
    exit 1
fi

# Map condition → model path. Vanilla uses HF id (resolved from cache);
# the three trained conditions use merged checkpoint dirs.
case "${COND}" in
    vanilla_qwen)
        MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
        ;;
    lora_base|y2_base|y2_rpe_cur_L16k)
        MODEL_PATH="${MERGED_DIR}/${COND}"
        if [ ! -d "${MODEL_PATH}" ]; then
            echo "ERROR: merged checkpoint not found at ${MODEL_PATH}" >&2
            echo "Run merge_lora.slurm first." >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unknown condition '${COND}'" >&2
        echo "Valid: vanilla_qwen, lora_base, y2_base, y2_rpe_cur_L16k" >&2
        exit 1
        ;;
esac

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${SCRATCH}/QRHead/src:${PROJECT_DIR}/${RHA_DIR}/scripts:${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/${RHA_DIR}/slurm_logs"
mkdir -p "${PROJECT_DIR}/${RHA_DIR}/results/${COND}"
cd "${PROJECT_DIR}"

echo "=== RHA Phase 3 — condition: ${COND} ==="
echo "  Job:        ${SLURM_JOB_ID}"
echo "  Job name:   ${SLURM_JOB_NAME}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Time:       $(date)"
echo "  Model path: ${MODEL_PATH}"
echo ""

python ${RHA_DIR}/scripts/phase3_run_detection.py \
    --condition  "${COND}" \
    --model-path "${MODEL_PATH}" \
    --data-dir   "${RHA_DIR}/data" \
    --output-dir "${RHA_DIR}/results"

echo ""
echo "=== Phase 3 done (${COND}): $(date) ==="
ls -lah "${PROJECT_DIR}/${RHA_DIR}/results/${COND}/"
