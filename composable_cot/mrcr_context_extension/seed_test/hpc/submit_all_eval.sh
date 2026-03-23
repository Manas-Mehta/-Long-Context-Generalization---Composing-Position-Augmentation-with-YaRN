#!/bin/bash
# ============================================================================
# Submit seed test eval jobs (top 4 models x 2 seeds = 8 jobs)
# All eval'd with YaRN f=4 at eval time (the winning eval config)
# Each eval: 5 bins, ~1-2 hours total per job on H200
# ============================================================================

set -eo pipefail

SCRIPT_DIR="composable_cot/mrcr_context_extension/seed_test/hpc"

# Create log directory
mkdir -p composable_cot/mrcr_context_extension/seed_test/slurm_logs

EVAL_YARN="--enable-yarn --yarn-factor 4.0"

submit_eval() {
    local MODEL=$1
    local SEED=$2
    local CKPT=$3

    local CONDITION="${MODEL}_s${SEED}_f4"

    echo "Submitting eval: ${CONDITION}"
    sbatch --job-name="ev_${MODEL}_s${SEED}" \
        --export="CKPT=${CKPT},YARN_FLAG=${EVAL_YARN},CONDITION=${CONDITION}" \
        ${SCRIPT_DIR}/eval_seed.slurm
}

for SEED in 123 456; do
    echo ""
    echo "=== Eval seed ${SEED} ==="

    # 1. Y2-P32
    submit_eval "y2_pose_32k" ${SEED} "seed_test/checkpoints/y2_pose_32k_s${SEED}"

    # 2. Y2-Rc16
    submit_eval "y2_rpe_cur_L16k" ${SEED} "seed_test/checkpoints/y2_rpe_cur_L16k_s${SEED}"

    # 3. Y4-Rc16
    submit_eval "y4_rpe_cur_L16k" ${SEED} "seed_test/checkpoints/y4_rpe_cur_L16k_s${SEED}"

    # 4. Y2-Rc64
    submit_eval "y2_rpe_cur_L64k" ${SEED} "seed_test/checkpoints/y2_rpe_cur_L64k_s${SEED}"
done

echo ""
echo "=== Submitted 8 eval jobs (4 models x 2 seeds) ==="
echo "Combined with seed=42 results from Phase 3/6, gives 3 seeds per model."
echo "Monitor: squeue -u mm14444"
