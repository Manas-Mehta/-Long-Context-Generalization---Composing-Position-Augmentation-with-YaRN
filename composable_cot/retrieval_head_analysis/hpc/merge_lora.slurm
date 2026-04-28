#!/bin/bash
#SBATCH --job-name=rha_merge
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=composable_cot/retrieval_head_analysis/slurm_logs/merge_lora_%j.out
#SBATCH --error=composable_cot/retrieval_head_analysis/slurm_logs/merge_lora_%j.err

# ============================================================================
# RHA Phase 2 — merge LoRA adapters for the 3 trained conditions into
# standalone Qwen2.5-7B checkpoints. For YaRN-using conditions, bakes
# rope_scaling into the saved config so subsequent loads apply YaRN.
#
# Submit from PROJECT_DIR: cd /scratch/mm14444/RPE && sbatch composable_cot/...
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
RHA_DIR="composable_cot/retrieval_head_analysis"
CKPT_DIR="${PROJECT_DIR}/composable_cot/BABIlong/checkpoints"
MERGED_DIR="${SCRATCH}/merged_for_qrhead"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/${RHA_DIR}/slurm_logs"
mkdir -p "${MERGED_DIR}"
cd "${PROJECT_DIR}"

echo "=== RHA Phase 2 — LoRA merge ==="
echo "  Job:       ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Time:      $(date)"
echo "  Output:    ${MERGED_DIR}"
echo ""

BASE="Qwen/Qwen2.5-7B-Instruct"

merge_one() {
    local cond="$1"
    local adapter_dir="$2"
    shift 2
    local extra_args=("$@")

    if [ ! -d "${adapter_dir}" ]; then
        echo "ERROR: adapter not found: ${adapter_dir}" >&2
        exit 1
    fi
    if [ ! -f "${adapter_dir}/adapter_model.safetensors" ]; then
        echo "ERROR: ${adapter_dir} does not contain adapter_model.safetensors" >&2
        exit 1
    fi

    echo ""
    echo "================================================================"
    echo "MERGING: ${cond}"
    echo "  adapter: ${adapter_dir}"
    echo "  output:  ${MERGED_DIR}/${cond}"
    echo "================================================================"

    python ${RHA_DIR}/scripts/merge_lora.py \
        --base "${BASE}" \
        --adapter "${adapter_dir}" \
        --output "${MERGED_DIR}/${cond}" \
        "${extra_args[@]}"
}

# 1. lora_base — none/none, no YaRN
merge_one "lora_base" \
    "${CKPT_DIR}/lora_base_1k/checkpoint-2000"

# 2. y2_base — YaRN/YaRN, factor 4 at eval
merge_one "y2_base" \
    "${CKPT_DIR}/y2_base_1k/checkpoint-2000" \
    --yarn-factor 4.0 --verify-yarn

# 3. y2_rpe_cur_L16k — YaRN+RPE/YaRN, factor 4 at eval
merge_one "y2_rpe_cur_L16k" \
    "${CKPT_DIR}/y2_rpe_cur_L16k_1k/checkpoint-2000" \
    --yarn-factor 4.0 --verify-yarn

echo ""
echo "================================================================"
echo "All merges complete: $(date)"
echo "================================================================"
ls -lah "${MERGED_DIR}/"
echo ""
du -sh "${MERGED_DIR}"/*/
