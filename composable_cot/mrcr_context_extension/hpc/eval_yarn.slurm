#!/bin/bash
#SBATCH --job-name=mrcr_yarn
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/mrcr_yarn_%j.out
#SBATCH --error=slurm_logs/mrcr_yarn_%j.err
#SBATCH --requeue

# ============================================================================
# MRCR Eval: Qwen2.5-7B-Instruct with YaRN (factor=4.0, no training)
# Evaluates on 4K-8K and 8K-16K bins
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1

cd "${PROJECT_DIR}"

echo "=== MRCR Eval: YaRN (factor=4.0) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

# --- Bin 0: 4K-8K ---
echo "=== Evaluating on 4K-8K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_4K-8K" \
    --max-new-tokens 2048

echo ""

# --- Bin 1: 8K-16K ---
echo "=== Evaluating on 8K-16K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin1_8K-16K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_8K-16K" \
    --max-new-tokens 2048

echo ""
echo "=== YaRN eval complete: $(date) ==="
