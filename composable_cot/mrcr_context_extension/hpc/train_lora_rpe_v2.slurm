#!/bin/bash
#SBATCH --job-name=mrcr_rpev2
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/mrcr_lora_rpe_v2_%j.out
#SBATCH --error=slurm_logs/mrcr_lora_rpe_v2_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 3b — RPE+LoRA fixed v2 (L warmup + lower LR + tighter grad clip)
#
# Changes from v1:
#   - L warmup: epoch 1 at L=8192 (sequential), epochs 2-5 at L=32768
#   - LR: 1e-4 (halved from 2e-4) — gentler optimization under position noise
#   - max_grad_norm: 0.5 (halved from 1.0) — prevent gradient explosion
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

echo "=== MRCR LoRA Training: RPE fixed v2 (L warmup) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Python:    $(python --version)"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "  peft:      $(python -c 'import peft; print(peft.__version__)')"
echo ""

python composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config "composable_cot/mrcr_context_extension/configs/rpe_config_mrcr_fixed_v2.yaml" \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/rpe_lora_v2" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 1e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --max-grad-norm 0.5 --seed 42

echo ""
echo "=== RPE+LoRA fixed v2 training complete: $(date) ==="
