#!/bin/bash
#SBATCH --job-name=lf_smoke
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm_logs/lf_smoke_%j.out
#SBATCH --error=slurm_logs/lf_smoke_%j.err

# ============================================================================
# LongFaith smoke test — verify pipeline end-to-end.
# Trains 1 step + evaluates 2 samples on the 16k v2 bin. Goal: catch import,
# path, prompt, and parser breakage. Not a real run.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/positionaug-YaRN"
EXPT_DIR="experiments/longfaith"

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

echo "=== LongFaith smoke test ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Python:    $(python --version)"
echo "  posaug:    $(python -c 'import posaug; print(posaug.__file__)')"
echo ""

# --- Step 1: import sanity ---
echo "--- 1/3: import sanity ---"
python -c "
import posaug, posaug.core, posaug.patching, posaug.pose, posaug.pose_patching
from posaug.callbacks_rpe import RPETrainerCallback, load_rpe_config
from posaug.callbacks_pose import PoSETrainerCallback
import torch, transformers, peft, datasets
print(f'  torch        {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  peft         {peft.__version__}')
print('  posaug imports OK')
"
echo ""

# --- Step 2: 1-step training with RPE curriculum wired in ---
echo "--- 2/3: train_longfaith_lora.py (1 step, RPE curriculum) ---"
python ${EXPT_DIR}/scripts/train_longfaith_lora.py \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --rpe-config     "${EXPT_DIR}/configs/rpe_config_longfaith_curriculum_L16k.yaml" \
    --train-file     "${EXPT_DIR}/data/faith_sft_2k_filtered.json" \
    --output-dir     "${EXPT_DIR}/checkpoints/smoke" \
    --max-seq-len    9216 \
    --lora-rank      16  --lora-alpha 32  --lora-dropout 0.1 \
    --lr             5e-5  --batch-size 1  --grad-accum 4 \
    --warmup-ratio   0.05  --max-grad-norm 1.0  --seed 42 \
    --max-steps      1 \
    --grad-log-every 999999 \
    --wandb-run-name "smoke"
echo ""

# --- Step 3: 2-sample eval on the 16k v2 bin (smallest, fits l40s easily) ---
echo "--- 3/3: eval_longbench_v2.py (2 samples, bin 16k) ---"
python ${EXPT_DIR}/scripts/eval_longbench_v2.py \
    --checkpoint-dir "${EXPT_DIR}/checkpoints/smoke" \
    --base-model     "Qwen/Qwen2.5-7B-Instruct" \
    --data-dir       "${EXPT_DIR}/data" \
    --output-dir     "${EXPT_DIR}/results/smoke" \
    --condition      "smoke" \
    --bins           16k \
    --max-samples    2 \
    --max-seq-len    20000 \
    --max-new-tokens 256
echo ""

echo "=== LongFaith smoke test PASSED: $(date) ==="
ls -lah "${EXPT_DIR}/results/smoke/"
