#!/bin/bash
#SBATCH --job-name=mrcr_vanilla
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/mrcr_vanilla_%j.out
#SBATCH --error=slurm_logs/mrcr_vanilla_%j.err
#SBATCH --requeue

# ============================================================================
# MRCR Eval: Vanilla Qwen2.5-7B-Instruct (no YaRN, no LoRA)
# Evaluates on all bins: 4K-8K, 8K-16K, 16K-32K, 32K-64K, 64K-128K
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

echo "=== MRCR Eval: Vanilla Qwen2.5-7B-Instruct ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
python -c "import transformers; print(f'  transformers: {transformers.__version__}')"
echo ""

# --- Bin 0: 4K-8K ---
echo "=== Evaluating on 4K-8K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/vanilla_4K-8K" \
    --max-new-tokens 2048

echo ""

# --- Bin 1: 8K-16K ---
echo "=== Evaluating on 8K-16K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --test-file "composable_cot/mrcr_context_extension/data/bin1_8K-16K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/vanilla_8K-16K" \
    --max-new-tokens 2048

echo ""

# --- Bin 2: 16K-32K ---
echo "=== Evaluating on 16K-32K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --test-file "composable_cot/mrcr_context_extension/data/bin2_16K-32K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/vanilla_16K-32K" \
    --max-new-tokens 2048

echo ""

# --- Bin 3: 32K-64K ---
echo "=== Evaluating on 32K-64K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --test-file "composable_cot/mrcr_context_extension/data/bin3_32K-64K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/vanilla_32K-64K" \
    --max-new-tokens 2048

echo ""

# --- Bin 4: 64K-128K ---
echo "=== Evaluating on 64K-128K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --test-file "composable_cot/mrcr_context_extension/data/bin4_64K-128K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/vanilla_64K-128K" \
    --max-new-tokens 2048

echo ""
echo "=== Vanilla eval complete: $(date) ==="
