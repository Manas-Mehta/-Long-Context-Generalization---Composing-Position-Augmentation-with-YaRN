#!/bin/bash
#SBATCH --job-name=eval_rpe_me
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/eval_y2_rpe_cur_L16k_me_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/eval_y2_rpe_cur_L16k_me_%j.err

# ============================================================================
# BABILong Eval — YaRN+RPE on multi-entry subset (305 hard samples)
# Trained: YaRN f=2 + RPE curriculum L=16K | Eval: YaRN f=4
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

echo "=== BABILong Eval: y2_rpe_cur_L16k — multi-entry subset (305 samples) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python ${BABILONG_DIR}/scripts/eval_babilong.py \
    --checkpoint-dir "${PROJECT_DIR}/${BABILONG_DIR}/checkpoints/y2_rpe_cur_L16k_1k/checkpoint-2000" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn    --yarn-factor 4.0 \
    --eval-dir       "${BABILONG_DIR}/data/eval_multi_entry" \
    --output-dir     "${BABILONG_DIR}/results/y2_rpe_cur_L16k_me" \
    --condition      "y2_rpe_cur_L16k_me" \
    --max-samples    0 \
    --max-seq-len    131072 \
    --max-new-tokens 10

echo ""
echo "=== y2_rpe_cur_L16k multi-entry eval complete: $(date) ==="
