#!/bin/bash
#SBATCH --job-name=bab_smoke
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/bab_smoke_%j.out
#SBATCH --error=slurm_logs/bab_smoke_%j.err

# ============================================================================
# BABILong smoke test — verify migrated repo runs end-to-end.
# Trains 1 step + evaluates 4 samples on the shortest bin. Goal: catch import,
# path, and CLI-arg breakage post-migration. Not a real run.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/positionaug-YaRN"
EXPT_DIR="experiments/babilong"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "${PROJECT_DIR}/slurm_logs"
mkdir -p "${PROJECT_DIR}/${EXPT_DIR}/checkpoints/smoke"
mkdir -p "${PROJECT_DIR}/${EXPT_DIR}/results/smoke"

cd "${PROJECT_DIR}"

echo "=== BABILong smoke test ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Python:    $(python --version)"
echo "  posaug:    $(python -c 'import posaug; print(posaug.__file__)')"
echo ""

# --- Step 1: import-only sanity check (no model download) ---
echo "--- 1/3: import sanity ---"
python -c "
import posaug, posaug.core, posaug.patching, posaug.pose, posaug.pose_patching
from posaug.callbacks_rpe import RPETrainerCallback, load_rpe_config
from posaug.callbacks_pose import PoSETrainerCallback
import torch, transformers, peft, datasets
print(f'  torch       {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  peft        {peft.__version__}')
print('  posaug imports OK')
"
echo ""

# --- Step 2: 1-step training with RPE callback wired in ---
echo "--- 2/3: train_babilong_lora.py (1 step, RPE curriculum) ---"
python ${EXPT_DIR}/scripts/train_babilong_lora.py \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config     "${EXPT_DIR}/configs/rpe_config_babilong_curriculum_L16k.yaml" \
    --train-file     "${EXPT_DIR}/data/train/all_train_1k.json" \
    --output-dir     "${EXPT_DIR}/checkpoints/smoke" \
    --max-seq-len    9216 \
    --lora-rank      16  --lora-alpha 32  --lora-dropout 0.1 \
    --lr             5e-5  --batch-size 1  --grad-accum 4 \
    --warmup-ratio   0.15  --max-grad-norm 1.0  --seed 42 \
    --max-steps      1 \
    --eval-every     999999 \
    --grad-log-every 999999 \
    --wandb-run-name "smoke"
echo ""

# --- Step 3: 4-sample eval on the shortest bin only ---
echo "--- 3/3: eval_babilong.py (4 samples, bin 0k) ---"
python ${EXPT_DIR}/scripts/eval_babilong.py \
    --checkpoint-dir "${EXPT_DIR}/checkpoints/smoke" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --eval-dir       "${EXPT_DIR}/data/eval" \
    --output-dir     "${EXPT_DIR}/results/smoke" \
    --condition      "smoke" \
    --bins           0k \
    --max-samples    4 \
    --max-seq-len    8192 \
    --max-new-tokens 10
echo ""

echo "=== BABILong smoke test PASSED: $(date) ==="
ls -lah "${EXPT_DIR}/results/smoke/"
