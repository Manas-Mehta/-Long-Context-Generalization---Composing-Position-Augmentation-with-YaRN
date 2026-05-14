#!/bin/bash
#SBATCH --job-name=lf_rpe_only
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/lf_rpe_only_%j.out
#SBATCH --error=slurm_logs/lf_rpe_only_%j.err

# ============================================================================
# Condition 5: rpe_only (RPE / none)
#   Train: LoRA + RPE curriculum L=16K, no YaRN
#   Eval:  no YaRN  (rpe_only_yarn4 reuses this checkpoint via eval_rpe_only_yarn4.slurm)
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/positionaug-YaRN"
EXPT_DIR="experiments/longfaith"
COND="rpe_only"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${PROJECT_DIR}/${EXPT_DIR}/checkpoints/${COND}"
mkdir -p "${PROJECT_DIR}/${EXPT_DIR}/results/${COND}"

cd "${PROJECT_DIR}"

echo "=== ${COND} ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Timestamp: $(date)"
echo ""

# --- Train ---
echo "--- Training ${COND} ---"
python ${EXPT_DIR}/scripts/train_longfaith_lora.py \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config     "${EXPT_DIR}/configs/rpe_config_longfaith_curriculum_L16k.yaml" \
    --train-file     "${EXPT_DIR}/data/faith_sft_2k_filtered.json" \
    --output-dir     "${EXPT_DIR}/checkpoints/${COND}" \
    --max-seq-len    9216 \
    --lora-rank      16  --lora-alpha 32  --lora-dropout 0.1 \
    --lr             5e-5  --batch-size 1  --grad-accum 4 \
    --warmup-ratio   0.05  --max-grad-norm 1.0  --seed 42 \
    --epochs         2 \
    --wandb-run-name "${COND}"

# --- Eval (all 4 bins, no YaRN) ---
echo "--- Evaluating ${COND} on LongBench v2 (no YaRN) ---"
python ${EXPT_DIR}/scripts/eval_longbench_v2.py \
    --checkpoint-dir "${EXPT_DIR}/checkpoints/${COND}" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --data-dir       "${EXPT_DIR}/data" \
    --output-dir     "${EXPT_DIR}/results/${COND}" \
    --condition      "${COND}" \
    --bins           16k 32k 64k 128k \
    --max-seq-len    131072 \
    --max-new-tokens 1024

echo "=== ${COND} DONE: $(date) ==="
