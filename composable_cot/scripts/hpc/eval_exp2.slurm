#!/bin/bash
#SBATCH --job-name=eval_asym
#SBATCH --account=torch_pr_219_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/eval_exp2_%j.out
#SBATCH --error=slurm_logs/eval_exp2_%j.err
#SBATCH --requeue

# ============================================================================
# Eval Exp 2: RPE asymmetric (Q/K=32, rest=8), L=1024
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

echo "=== Eval Exp 2: RPE asymmetric (Q/K=32, rest=8) L=1024 ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

python composable_cot/scripts/eval_length_generalization.py \
    --base-model "Qwen/Qwen2.5-7B" \
    --lora-ckpt "composable_cot/model_ckpt/reverse_string_rpe_asymmetric" \
    --test-file "composable_cot/data/reverse_string_eval/test_all.json" \
    --output-dir "composable_cot/outputs/full_eval_rpe_asymmetric" \
    --task-type reverse_string \
    --train-max-length 40 \
    --min-length 35 \
    --max-new-tokens 2048

echo ""
echo "=== Eval Exp 2 complete: $(date) ==="
