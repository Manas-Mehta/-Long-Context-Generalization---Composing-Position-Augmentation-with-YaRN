#!/bin/bash
#SBATCH --job-name=gen_needle_v2
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/gen_needle_v2_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/gen_needle_v2_%j.err

# ============================================================================
# BABILong — Generate needle-position eval data v2 (CPU only)
# 227 samples × 3 zones × 8 bins = 5,448 samples
# Uses h200 partition because account may be restricted to GPU partitions.
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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export HF_DATASETS_TRUST_REMOTE_CODE=1

mkdir -p "${PROJECT_DIR}/${BABILONG_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== Generate needle-position eval data v2 (227 samples) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/generate_needle_position_eval.py \
    --selected-indices ${BABILONG_DIR}/data/eval_needle_v2/selected_227_indices.json \
    --output-dir       ${BABILONG_DIR}/data/eval_needle_v2 \
    --bins             1k,2k,4k,8k,16k,32k,64k,128k \
    --zones            beg,mid,end \
    --noise-dataset    pg19 \
    --random-seed      42 \
    --babilong-src-dir ${BABILONG_DIR}/babilong_src

echo ""
echo "=== Generation complete: $(date) ==="
ls -la ${BABILONG_DIR}/data/eval_needle_v2/
