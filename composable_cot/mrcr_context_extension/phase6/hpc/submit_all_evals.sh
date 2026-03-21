#!/bin/bash
# Phase 6 — Submit all eval jobs for completed checkpoints
# Each job evals one model in one YaRN mode across all 5 bins
# Run from project root: bash composable_cot/mrcr_context_extension/phase6/hpc/submit_all_evals.sh

EVAL_SCRIPT="composable_cot/mrcr_context_extension/phase6/hpc/eval_phase6.slurm"

# --- Group A: RPE-only controls (no YaRN during training) ---
# Eval: no YaRN, +YaRN f=4

sbatch --job-name=ev_Rc4_noy  --export=ALL,CKPT=rpe_cur_L4k,YARN_FLAG="",CONDITION=rpe_cur_L4k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_Rc4_y4   --export=ALL,CKPT=rpe_cur_L4k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=rpe_cur_L4k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_Rc8_noy  --export=ALL,CKPT=rpe_cur_L8k,YARN_FLAG="",CONDITION=rpe_cur_L8k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_Rc8_y4   --export=ALL,CKPT=rpe_cur_L8k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=rpe_cur_L8k_yarn4  ${EVAL_SCRIPT}

# --- Group C: YaRN f=2 + RPE small L ---
# Eval: no YaRN, matching f=2, scaled f=4

sbatch --job-name=ev_y2Rc4_noy --export=ALL,CKPT=y2_rpe_cur_L4k,YARN_FLAG="",CONDITION=y2_rpe_cur_L4k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc4_y2  --export=ALL,CKPT=y2_rpe_cur_L4k,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=y2_rpe_cur_L4k_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc4_y4  --export=ALL,CKPT=y2_rpe_cur_L4k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y2_rpe_cur_L4k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y2Rc8_noy --export=ALL,CKPT=y2_rpe_cur_L8k,YARN_FLAG="",CONDITION=y2_rpe_cur_L8k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc8_y2  --export=ALL,CKPT=y2_rpe_cur_L8k,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=y2_rpe_cur_L8k_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc8_y4  --export=ALL,CKPT=y2_rpe_cur_L8k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y2_rpe_cur_L8k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y2Rc16_noy --export=ALL,CKPT=y2_rpe_cur_L16k,YARN_FLAG="",CONDITION=y2_rpe_cur_L16k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc16_y2  --export=ALL,CKPT=y2_rpe_cur_L16k,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=y2_rpe_cur_L16k_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc16_y4  --export=ALL,CKPT=y2_rpe_cur_L16k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y2_rpe_cur_L16k_yarn4  ${EVAL_SCRIPT}

# --- Group D: YaRN f=2 + RPE window-matching L ---
# Eval: no YaRN, matching f=2, scaled f=4

sbatch --job-name=ev_y2Rc32_noy --export=ALL,CKPT=y2_rpe_cur_L32k,YARN_FLAG="",CONDITION=y2_rpe_cur_L32k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc32_y2  --export=ALL,CKPT=y2_rpe_cur_L32k,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=y2_rpe_cur_L32k_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc32_y4  --export=ALL,CKPT=y2_rpe_cur_L32k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y2_rpe_cur_L32k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y2Rc64_noy --export=ALL,CKPT=y2_rpe_cur_L64k,YARN_FLAG="",CONDITION=y2_rpe_cur_L64k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc64_y2  --export=ALL,CKPT=y2_rpe_cur_L64k,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=y2_rpe_cur_L64k_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2Rc64_y4  --export=ALL,CKPT=y2_rpe_cur_L64k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y2_rpe_cur_L64k_yarn4  ${EVAL_SCRIPT}

# --- Group E: YaRN f=2 + PoSE ---
# Eval: no YaRN, matching f=2, scaled f=4

sbatch --job-name=ev_y2P16_noy --export=ALL,CKPT=y2_pose_16k,YARN_FLAG="",CONDITION=y2_pose_16k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2P16_y2  --export=ALL,CKPT=y2_pose_16k,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=y2_pose_16k_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2P16_y4  --export=ALL,CKPT=y2_pose_16k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y2_pose_16k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y2P32_noy --export=ALL,CKPT=y2_pose_32k,YARN_FLAG="",CONDITION=y2_pose_32k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2P32_y2  --export=ALL,CKPT=y2_pose_32k,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=y2_pose_32k_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2P32_y4  --export=ALL,CKPT=y2_pose_32k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y2_pose_32k_yarn4  ${EVAL_SCRIPT}

# --- Y4 + RPE (only y4_rpe_cur_L4k done so far) ---
# Eval: no YaRN, matching f=4

sbatch --job-name=ev_y4Rc4_noy --export=ALL,CKPT=y4_rpe_cur_L4k,YARN_FLAG="",CONDITION=y4_rpe_cur_L4k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y4Rc4_y4  --export=ALL,CKPT=y4_rpe_cur_L4k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y4_rpe_cur_L4k_yarn4  ${EVAL_SCRIPT}

echo ""
echo "Submitted 27 eval jobs total."
echo "  Group A (RPE-only):     4 jobs  (2 models x 2 modes)"
echo "  Group C (Y2+RPE small): 9 jobs  (3 models x 3 modes)"
echo "  Group D (Y2+RPE window): 6 jobs (2 models x 3 modes)"
echo "  Group E (Y2+PoSE):      6 jobs  (2 models x 3 modes)"
echo "  Y4+RPE:                  2 jobs  (1 model x 2 modes)"
