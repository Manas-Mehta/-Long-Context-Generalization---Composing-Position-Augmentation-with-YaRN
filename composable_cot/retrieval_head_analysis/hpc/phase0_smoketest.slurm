#!/bin/bash
#SBATCH --job-name=qrhead_phase0
#SBATCH --output=/scratch/mm14444/RPE/composable_cot/retrieval_head_analysis/slurm_logs/phase0_smoketest_%j.out
#SBATCH --error=/scratch/mm14444/RPE/composable_cot/retrieval_head_analysis/slurm_logs/phase0_smoketest_%j.err
#SBATCH --partition=h200_courant
#SBATCH --account=torch_pr_219_courant
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00

set -euo pipefail

mkdir -p /scratch/mm14444/RPE/composable_cot/retrieval_head_analysis/slurm_logs

echo "=== Phase 0 smoke test ==="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
nvidia-smi | head -10
echo

# Activate the dedicated qrhead env (NOT the rpe env).
source /scratch/mm14444/conda_envs/qrhead/etc/profile.d/conda.sh 2>/dev/null \
    || conda activate /scratch/mm14444/conda_envs/qrhead
which python
python --version

cd /scratch/mm14444/RPE
python composable_cot/retrieval_head_analysis/scripts/phase0_smoketest.py

echo
echo "=== DONE ==="
