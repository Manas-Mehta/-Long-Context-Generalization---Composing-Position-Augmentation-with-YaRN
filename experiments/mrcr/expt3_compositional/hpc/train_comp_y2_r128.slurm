#!/bin/bash
#SBATCH --job-name=comp_y2r128
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=composable_cot/mrcr_context_extension/expt3_compositional/slurm_logs/train_comp_y2_r128_%j.out
#SBATCH --error=composable_cot/mrcr_context_extension/expt3_compositional/slurm_logs/train_comp_y2_r128_%j.err
#SBATCH --requeue

# ============================================================================
# Experiment 3 — Compositional Context Extension
# Condition: Comp-Y2-R128
#   YaRN f=2.0 at model load (extends 32K → 64K)
#   + RPE curriculum to L=128K (pushes positions BEYOND YaRN's 64K window)
#
# Hypothesis: two small extension steps beat one large YaRN factor.
# Comparison: this vs Pure-Y4 (YaRN f=4 alone) for reaching 128K.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
MRCR_DIR="composable_cot/mrcr_context_extension"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1

cd "${PROJECT_DIR}"

echo "=== Experiment 3: Comp-Y2-R128 (YaRN f=2 + RPE curriculum L=128K) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo "  Python:    $(python --version)"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "  peft:      $(python -c 'import peft; print(peft.__version__)')"
echo ""
echo "  YaRN factor:    2.0 (extends 32K → 64K)"
echo "  RPE curriculum: 10240 → 32768 → 65536 → 131072 → 131072"
echo "  RPE pushes beyond YaRN window in epochs 4-5 (L=128K > YaRN's 64K)"
echo ""

python ${MRCR_DIR}/scripts/train_mrcr_lora.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 2.0 \
    --rpe-config "${MRCR_DIR}/configs/rpe_config_mrcr_curriculum_L128k.yaml" \
    --train-file "${MRCR_DIR}/data/bin0_4K-8K/train.json" \
    --output-dir "${MRCR_DIR}/expt3_compositional/checkpoints/comp_y2_r128" \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \
    --lr 2e-4 --epochs 5 --batch-size 1 --grad-accum 4 \
    --max-seq-len 8192 --warmup-ratio 0.1 --seed 42

echo ""
echo "=== Comp-Y2-R128 training complete: $(date) ==="
