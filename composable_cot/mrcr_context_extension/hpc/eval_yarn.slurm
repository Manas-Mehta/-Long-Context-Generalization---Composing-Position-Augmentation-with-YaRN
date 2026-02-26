#!/bin/bash
#SBATCH --job-name=mrcr_yarn
#SBATCH --account=torch_pr_219_courant
#SBATCH --partition=h200_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/mrcr_yarn_%j.out
#SBATCH --error=slurm_logs/mrcr_yarn_%j.err
#SBATCH --requeue

# ============================================================================
# MRCR Eval: Qwen2.5-7B-Instruct with YaRN (factor=4.0, no training)
# Evaluates on all bins: 4K-8K, 8K-16K, 16K-32K, 32K-64K, 64K-128K
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

echo "=== MRCR Eval: YaRN (factor=4.0) ==="
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Timestamp: $(date)"
echo ""

# --- Pre-flight: verify YaRN math works before expensive model load ---
echo "=== Pre-flight: YaRN sanity check ==="
python -c "
import transformers
print(f'transformers version: {transformers.__version__}')

import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

config = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
rope_theta = getattr(config, 'rope_theta', None)
if rope_theta is None and hasattr(config, 'rope_parameters'):
    rope_theta = config.rope_parameters.get('rope_theta', 10000.0)
print(f'rope_theta: {rope_theta}')
print(f'max_position_embeddings: {config.max_position_embeddings}')

# Compute vanilla inv_freq
head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
inv_v = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

# Set YaRN config (version-agnostic)
if hasattr(config, 'rope_parameters') and isinstance(config.rope_parameters, dict):
    config.rope_parameters.update({'type': 'yarn', 'rope_type': 'yarn', 'factor': 4.0})
else:
    config.rope_scaling = {'type': 'yarn', 'factor': 4.0}
if hasattr(config, 'standardize_rope_params'):
    config.standardize_rope_params()

inv_y, attn = ROPE_INIT_FUNCTIONS['yarn'](config, device='cpu')
n_changed = ((inv_v - inv_y).abs() > 1e-8).sum().item()
print(f'YaRN dims changed: {n_changed}/{len(inv_v)}')
print(f'attention_factor: {attn}')
if n_changed > 0:
    print('PRE-FLIGHT PASSED: YaRN math produces different frequencies')
else:
    print('PRE-FLIGHT FAILED: YaRN frequencies identical to vanilla!')
    exit(1)
"
echo ""

# --- Bin 0: 4K-8K ---
echo "=== Evaluating on 4K-8K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin0_4K-8K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_4K-8K" \
    --max-new-tokens 2048

echo ""

# --- Bin 1: 8K-16K ---
echo "=== Evaluating on 8K-16K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin1_8K-16K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_8K-16K" \
    --max-new-tokens 2048

echo ""

# --- Bin 2: 16K-32K ---
echo "=== Evaluating on 16K-32K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin2_16K-32K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_16K-32K" \
    --max-new-tokens 2048

echo ""

# --- Bin 3: 32K-64K ---
echo "=== Evaluating on 32K-64K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin3_32K-64K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_32K-64K" \
    --max-new-tokens 2048

echo ""

# --- Bin 4: 64K-128K ---
echo "=== Evaluating on 64K-128K bin ==="
python composable_cot/mrcr_context_extension/scripts/eval_mrcr.py \
    --base-model "Qwen/Qwen2.5-7B-Instruct" \
    --enable-yarn --yarn-factor 4.0 \
    --test-file "composable_cot/mrcr_context_extension/data/bin4_64K-128K/test.json" \
    --output-dir "composable_cot/mrcr_context_extension/outputs/yarn_64K-128K" \
    --max-new-tokens 2048

echo ""
echo "=== YaRN eval complete: $(date) ==="
