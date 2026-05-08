#!/bin/bash
#SBATCH --job-name=bab_eval
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=l40s_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=composable_cot/BABIlong/slurm_logs/eval_%x_%j.out
#SBATCH --error=composable_cot/BABIlong/slurm_logs/eval_%x_%j.err

# ============================================================================
# BABILong — Evaluation
# Evaluates one trained condition across all 9 context-length bins (0K–128K).
#
# Usage:
#   sbatch --export=CONDITION=lora_base eval_babilong.slurm
#   sbatch --export=CONDITION=y2_base,YARN_EVAL=4.0 eval_babilong.slurm
#   sbatch --export=CONDITION=y2_rpe_cur_L16k,YARN_EVAL=4.0 eval_babilong.slurm
#   sbatch --export=CONDITION=y2_pose_32k,YARN_EVAL=4.0 eval_babilong.slurm
#   sbatch --export=CONDITION=rpe_only eval_babilong.slurm
#   sbatch --export=CONDITION=pose_only eval_babilong.slurm
#
# CONDITION:  name of checkpoint dir under composable_cot/BABIlong/checkpoints/
# YARN_EVAL:  YaRN factor at eval (default: none). Set 4.0 for YaRN conditions (2,3,4).
#
# NOTE: eval_babilong.py script is still being written (next session).
#       This SLURM stub documents the intended interface so it can be completed.
# ============================================================================

set -eo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"
BABILONG_DIR="composable_cot/BABIlong"

# Defaults
CONDITION="${CONDITION:-lora_base}"
YARN_EVAL="${YARN_EVAL:-}"

eval "$(conda shell.bash hook)"
conda activate "${SCRATCH}/conda_envs/rpe"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1

mkdir -p "${PROJECT_DIR}/${BABILONG_DIR}/slurm_logs"

cd "${PROJECT_DIR}"

echo "=== BABILong Evaluation: ${CONDITION} ==="
echo "  YARN_EVAL: ${YARN_EVAL:-none}"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

# Build optional YARN flag
YARN_FLAG=""
if [ -n "${YARN_EVAL}" ]; then
    YARN_FLAG="--enable-yarn --yarn-factor ${YARN_EVAL}"
fi

python ${BABILONG_DIR}/scripts/eval_babilong.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --checkpoint-dir "${BABILONG_DIR}/checkpoints/${CONDITION}" \
    --eval-dir "${BABILONG_DIR}/data/eval" \
    --output-file "${BABILONG_DIR}/results/${CONDITION}_results.json" \
    --bins 0k 1k 2k 4k 8k 16k 32k 64k 128k \
    --max-new-tokens 20 \
    ${YARN_FLAG}

echo ""
echo "=== Eval complete: $(date) ==="
echo "=== Results saved to ${BABILONG_DIR}/results/${CONDITION}_results.json ==="
