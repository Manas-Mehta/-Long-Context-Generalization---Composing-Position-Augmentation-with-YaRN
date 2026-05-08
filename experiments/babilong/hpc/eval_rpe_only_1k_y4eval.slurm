#!/bin/bash
#SBATCH --job-name=eval_rpe1k_y4
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/eval_rpe_only_1k_y4eval_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/eval_rpe_only_1k_y4eval_%j.err

# ============================================================================
# BABILong Eval — RPE-only (1k) with YaRN f=4 at eval
# Same checkpoint as rpe_only_1k but with YaRN applied at inference.
# This is the MRCR-style ablation: RPE training + YaRN eval only (no YaRN training).
# Compare against y2_rpe_cur_L16k_1k (YaRN at both train+eval) to isolate
# how much of the YaRN+RPE benefit comes from YaRN eval alone.
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

echo "=== BABILong Eval: rpe_only_1k with YaRN f=4 at eval ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/eval_babilong.py \
    --checkpoint-dir "${PROJECT_DIR}/${BABILONG_DIR}/checkpoints/rpe_only_1k/checkpoint-2000" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn    --yarn-factor 4.0 \
    --eval-dir       "${BABILONG_DIR}/data/eval" \
    --output-dir     "${BABILONG_DIR}/results/rpe_only_1k_y4eval" \
    --condition      "rpe_only_1k_y4eval" \
    --max-samples    100 \
    --max-seq-len    131072 \
    --max-new-tokens 10

echo ""
echo "=== rpe_only_1k_y4eval eval complete: $(date) ==="
