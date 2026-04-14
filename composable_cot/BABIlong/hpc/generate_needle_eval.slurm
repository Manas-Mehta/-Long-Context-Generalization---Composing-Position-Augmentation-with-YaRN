#!/bin/bash
#SBATCH --job-name=gen_needle
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# NOTE: CPU-only generation, but h200_courant used because account may be
# restricted to GPU partitions. Change to a CPU partition if available.
#SBATCH --output=composable_cot/BABIlong/slurm_logs/gen_needle_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/gen_needle_%j.err

# ============================================================================
# BABILong — Generate needle-position eval data (CPU only)
# 100 stories × 3 zones × 9 bins = 2700 samples
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
# PG19 is script-based; allow network for first download, else cached
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export HF_DATASETS_TRUST_REMOTE_CODE=1

mkdir -p "${PROJECT_DIR}/${BABILONG_DIR}/slurm_logs"
cd "${PROJECT_DIR}"

echo "=== Generate needle-position eval data ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/generate_needle_position_eval.py \
    --selected-indices ${BABILONG_DIR}/data/eval_needle/selected_100_indices.json \
    --output-dir       ${BABILONG_DIR}/data/eval_needle \
    --bins             1k,2k,4k,8k,16k,32k,64k,128k \
    --zones            beg,mid,end \
    --noise-dataset    pg19 \
    --random-seed      42 \
    --babilong-src-dir ${BABILONG_DIR}/babilong_src

echo ""
echo "=== Generation complete: $(date) ==="
ls -la ${BABILONG_DIR}/data/eval_needle/
