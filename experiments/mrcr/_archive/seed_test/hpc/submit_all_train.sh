#!/bin/bash
# ============================================================================
# Submit seed test training jobs (top 4 models x 2 seeds = 8 jobs)
# Original seed=42 already exists. We add seed=123 and seed=456.
# ~21 min each on L40S/H200 = ~3 hours total (parallel on cluster)
# ============================================================================

set -eo pipefail

SCRIPT_DIR="composable_cot/mrcr_context_extension/seed_test/hpc"
MRCR_DIR="composable_cot/mrcr_context_extension"

# Create log directory
mkdir -p composable_cot/mrcr_context_extension/seed_test/slurm_logs

# ============================================================================
# Top 4 models from leaderboard (ranked by bin4 score with eval f=4):
#  1. Y2-P32     (0.777) — YaRN f=2 train + PoSE target=32K
#  2. Y2-Rc16    (0.753) — YaRN f=2 train + RPE cur L=16K
#  3. Y4-Rc16    (0.744) — YaRN f=4 train + RPE cur L=16K
#  4. Y2-Rc64    (0.648) — YaRN f=2 train + RPE cur L=64K
# ============================================================================

submit_train() {
    local MODEL=$1
    local SEED=$2
    local YARN_FACTOR=$3
    local POS_FLAG=$4

    echo "Submitting: ${MODEL} seed=${SEED}"
    sbatch --job-name="${MODEL}_s${SEED}" \
        --export="MODEL=${MODEL},SEED=${SEED},YARN_FACTOR=${YARN_FACTOR},POS_FLAG=${POS_FLAG}" \
        ${SCRIPT_DIR}/train_seed.slurm
}

for SEED in 123 456; do
    echo ""
    echo "=== Seed ${SEED} ==="

    # 1. Y2-P32: YaRN f=2 + PoSE target=32K
    submit_train "y2_pose_32k" ${SEED} "2.0" "--pose-config ${MRCR_DIR}/configs/pose_config_mrcr.yaml"

    # 2. Y2-Rc16: YaRN f=2 + RPE cur L=16K
    submit_train "y2_rpe_cur_L16k" ${SEED} "2.0" "--rpe-config ${MRCR_DIR}/configs/rpe_config_mrcr_curriculum_L16k.yaml"

    # 3. Y4-Rc16: YaRN f=4 + RPE cur L=16K
    submit_train "y4_rpe_cur_L16k" ${SEED} "4.0" "--rpe-config ${MRCR_DIR}/configs/rpe_config_mrcr_curriculum_L16k.yaml"

    # 4. Y2-Rc64: YaRN f=2 + RPE cur L=64K
    submit_train "y2_rpe_cur_L64k" ${SEED} "2.0" "--rpe-config ${MRCR_DIR}/configs/rpe_config_mrcr_curriculum_L64k.yaml"
done

echo ""
echo "=== Submitted 8 training jobs (4 models x 2 seeds) ==="
echo "Monitor: squeue -u mm14444"
