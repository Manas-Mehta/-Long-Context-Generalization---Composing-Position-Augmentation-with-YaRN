#!/bin/bash
#SBATCH --job-name=mrcr_smoke
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/mrcr_smoke_%j.out
#SBATCH --error=slurm_logs/mrcr_smoke_%j.err

# ============================================================================
# MRCR smoke test — verify migrated repo runs end-to-end.
# Trains 1 step + evaluates 4 examples on bin 0 (4K-8K). Goal: catch import,
# path, and CLI-arg breakage post-migration. Not a real run.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
EXPT_DIR="experiments/mrcr"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${PROJECT_DIR}/${EXPT_DIR}/checkpoints/smoke"
mkdir -p "${PROJECT_DIR}/${EXPT_DIR}/outputs/smoke"

cd "${PROJECT_DIR}"

echo "=== MRCR smoke test ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Python:    $(python --version)"
echo "  posaug:    $(python -c 'import posaug; print(posaug.__file__)')"
echo ""

# --- Step 1: 1-step training with RPE-curriculum callback ---
echo "--- 1/2: train_mrcr_lora.py (1 step, RPE curriculum L=16K) ---"
python ${EXPT_DIR}/scripts/train_mrcr_lora.py \
    --base-model    "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config    "${EXPT_DIR}/configs/rpe_config_mrcr_curriculum_L16k.yaml" \
    --train-file    "${EXPT_DIR}/data/bin0_4K-8K/train.json" \
    --output-dir    "${EXPT_DIR}/checkpoints/smoke" \
    --lora-rank     16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr            2e-4 --batch-size 1 --grad-accum 4 \
    --max-seq-len   8192 --warmup-ratio 0.1 --seed 42 \
    --max-steps     1
echo ""

# --- Step 2: 4-example eval on bin 0 ---
echo "--- 2/2: eval_mrcr.py (4 examples, bin 0 4K-8K) ---"
python ${EXPT_DIR}/scripts/eval_mrcr.py \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --lora-ckpt      "${EXPT_DIR}/checkpoints/smoke" \
    --test-file      "${EXPT_DIR}/data/bin0_4K-8K/test.json" \
    --output-dir     "${EXPT_DIR}/outputs/smoke" \
    --max-examples   4 \
    --max-new-tokens 256
echo ""

echo "=== MRCR smoke test PASSED: $(date) ==="
ls -lah "${EXPT_DIR}/outputs/smoke/"
