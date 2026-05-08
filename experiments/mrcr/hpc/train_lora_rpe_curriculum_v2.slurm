#!/bin/bash
#SBATCH --job-name=mrcr_rcv2
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/mrcr_lora_rpe_cur_v2_%j.out
#SBATCH --error=slurm_logs/mrcr_lora_rpe_cur_v2_%j.err
#SBATCH --requeue

# ============================================================================
# Phase 3b — RPE+LoRA curriculum v2 (aggressive L targeting bins 3-4)
#
# Changes from v1:
#   - L schedule: 16384→32768→49152→65536→65536 (v1: 10240→...→32768)
#   - Pushes to L=65536 (bin 3 max) to target longer-context generalization
#   - Same LR (2e-4) — v1 curriculum trained smoothly, no instability
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

echo "=== MRCR LoRA Training: RPE Curriculum v2 (aggressive L) ==="
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
    --rpe-config "composable_cot/mrcr_context_extension/configs/rpe_config_mrcr_curriculum_v2.yaml" \
    --train-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/train.json" \
    --output-dir "composable_cot/mrcr_context_extension/checkpoints/rpe_curriculum_lora_v2" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

echo ""
echo "=== RPE Curriculum v2+LoRA training complete: $(date) ==="
