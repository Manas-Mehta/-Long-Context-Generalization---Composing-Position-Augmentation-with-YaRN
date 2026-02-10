#!/bin/bash
# ============================================================================
# NYU Torch HPC: One-time environment setup
# ============================================================================
# Run this ONCE after SSH-ing into login.torch.hpc.nyu.edu
#
# Storage layout:
#   /home/mm14444/           — small (30K inode limit). Only symlinks/configs.
#   /scratch/mm14444/RPE/    — code, data, conda env, model checkpoints.
#
# Usage:
#   ssh mm14444@login.torch.hpc.nyu.edu
#   bash /scratch/mm14444/RPE/composable_cot/scripts/hpc/setup_hpc.sh
# ============================================================================

set -euo pipefail

NETID="mm14444"
SCRATCH="/scratch/${NETID}"
PROJECT_DIR="${SCRATCH}/RPE"

echo "============================================================"
echo "NYU Torch HPC Setup"
echo "============================================================"
echo "  SCRATCH:     ${SCRATCH}"
echo "  PROJECT_DIR: ${PROJECT_DIR}"
echo ""

# ── Step 1: Clone repo to /scratch ───────────────────────────────────
echo "=== Step 1: Clone/update repo ==="
if [ -d "${PROJECT_DIR}/.git" ]; then
    echo "  Repo exists. Pulling latest..."
    cd "${PROJECT_DIR}"
    git pull
else
    echo "  Cloning repo to ${PROJECT_DIR}..."
    mkdir -p "${SCRATCH}"
    cd "${SCRATCH}"
    git clone https://github.com/Manas-Mehta/Generalized-CCoT.git RPE
fi
cd "${PROJECT_DIR}"
echo "  Done."
echo ""

# ── Step 2: Create conda environment on /scratch ─────────────────────
# IMPORTANT: conda envs in /home will exhaust the 30K inode limit.
# We create the env on /scratch instead.
echo "=== Step 2: Conda environment ==="

CONDA_ENV_DIR="${SCRATCH}/conda_envs/rpe"

# Load conda module (adjust if your cluster uses a different module name)
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || {
    echo "WARNING: Could not load conda module. Trying system conda..."
    echo "  If this fails, run: module avail | grep -i conda"
    echo "  Then update the 'module load' line in this script."
}

if [ -d "${CONDA_ENV_DIR}" ]; then
    echo "  Conda env exists at ${CONDA_ENV_DIR}. Activating..."
else
    echo "  Creating conda env at ${CONDA_ENV_DIR}..."
    conda create -y --prefix "${CONDA_ENV_DIR}" python=3.10
fi

# Activate
source activate "${CONDA_ENV_DIR}" 2>/dev/null || conda activate "${CONDA_ENV_DIR}"

echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

# ── Step 3: Install dependencies ─────────────────────────────────────
echo "=== Step 3: Install packages ==="

# PyTorch with CUDA (adjust CUDA version if needed — check with: module avail cuda)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# PEFT, transformers, etc.
pip install peft transformers accelerate datasets sentencepiece protobuf -q

# LLaMA-Factory
cd "${PROJECT_DIR}/composable_cot/LLaMA-Factory"
pip install -e ".[torch,metrics]" --no-build-isolation -q
cd "${PROJECT_DIR}"

# Our RPE module (just needs to be on PYTHONPATH, no install needed)

echo "  Installed packages."
echo ""

# ── Step 4: Verify GPU access ────────────────────────────────────────
echo "=== Step 4: Verify setup ==="
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count:       {torch.cuda.device_count()}')
    print(f'  GPU name:        {torch.cuda.get_device_name(0)}')
else:
    print('  NOTE: No GPU on login node. This is normal — GPUs are on compute nodes.')
"

python -c "
import peft; print(f'  PEFT version: {peft.__version__}')
import transformers; print(f'  Transformers: {transformers.__version__}')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "To submit training jobs:"
echo "  cd ${PROJECT_DIR}"
echo "  sbatch composable_cot/scripts/hpc/train_exp1.slurm"
echo "  sbatch composable_cot/scripts/hpc/train_exp2.slurm"
echo "  sbatch composable_cot/scripts/hpc/train_exp3.slurm"
echo ""
echo "To check job status:"
echo "  squeue -u ${NETID}"
echo ""
echo "To check quota:"
echo "  myquota"
echo "============================================================"
