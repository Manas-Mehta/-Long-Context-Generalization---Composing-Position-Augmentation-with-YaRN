#!/bin/bash
# Phase 6 — Submit evals for the 8 newly trained models
# Run from project root: bash composable_cot/mrcr_context_extension/phase6/hpc/submit_remaining_evals.sh

EVAL_SCRIPT="composable_cot/mrcr_context_extension/phase6/hpc/eval_phase6.slurm"

# --- Group B: YaRN-only baselines ---
# Eval: no YaRN, matching factor, f=4

sbatch --job-name=ev_y2_noy   --export=ALL,CKPT=yarn2_lora,YARN_FLAG="",CONDITION=yarn2_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2_y2    --export=ALL,CKPT=yarn2_lora,YARN_FLAG="--enable-yarn --yarn-factor 2.0",CONDITION=yarn2_yarn2  ${EVAL_SCRIPT}
sbatch --job-name=ev_y2_y4    --export=ALL,CKPT=yarn2_lora,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=yarn2_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y3_noy   --export=ALL,CKPT=yarn3_lora,YARN_FLAG="",CONDITION=yarn3_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y3_y3    --export=ALL,CKPT=yarn3_lora,YARN_FLAG="--enable-yarn --yarn-factor 3.0",CONDITION=yarn3_yarn3  ${EVAL_SCRIPT}
sbatch --job-name=ev_y3_y4    --export=ALL,CKPT=yarn3_lora,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=yarn3_yarn4  ${EVAL_SCRIPT}

# --- Group C: Y4+RPE small L ---
# Eval: no YaRN, matching f=4

sbatch --job-name=ev_y4Rc8_noy  --export=ALL,CKPT=y4_rpe_cur_L8k,YARN_FLAG="",CONDITION=y4_rpe_cur_L8k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y4Rc8_y4   --export=ALL,CKPT=y4_rpe_cur_L8k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y4_rpe_cur_L8k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y4Rc16_noy --export=ALL,CKPT=y4_rpe_cur_L16k,YARN_FLAG="",CONDITION=y4_rpe_cur_L16k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y4Rc16_y4  --export=ALL,CKPT=y4_rpe_cur_L16k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y4_rpe_cur_L16k_yarn4  ${EVAL_SCRIPT}

# --- Group D: Y4+RPE window-matching L ---
# Eval: no YaRN, matching f=4

sbatch --job-name=ev_y4Rc64_noy  --export=ALL,CKPT=y4_rpe_cur_L64k,YARN_FLAG="",CONDITION=y4_rpe_cur_L64k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y4Rc64_y4   --export=ALL,CKPT=y4_rpe_cur_L64k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y4_rpe_cur_L64k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y4Rc128_noy --export=ALL,CKPT=y4_rpe_cur_L128k,YARN_FLAG="",CONDITION=y4_rpe_cur_L128k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y4Rc128_y4  --export=ALL,CKPT=y4_rpe_cur_L128k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y4_rpe_cur_L128k_yarn4  ${EVAL_SCRIPT}

# --- Group E: Y4+PoSE ---
# Eval: no YaRN, matching f=4

sbatch --job-name=ev_y4P16_noy --export=ALL,CKPT=y4_pose_16k,YARN_FLAG="",CONDITION=y4_pose_16k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y4P16_y4  --export=ALL,CKPT=y4_pose_16k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y4_pose_16k_yarn4  ${EVAL_SCRIPT}

sbatch --job-name=ev_y4P32_noy --export=ALL,CKPT=y4_pose_32k,YARN_FLAG="",CONDITION=y4_pose_32k_no_yarn  ${EVAL_SCRIPT}
sbatch --job-name=ev_y4P32_y4  --export=ALL,CKPT=y4_pose_32k,YARN_FLAG="--enable-yarn --yarn-factor 4.0",CONDITION=y4_pose_32k_yarn4  ${EVAL_SCRIPT}

echo ""
echo "Submitted 18 eval jobs for 8 newly trained models."
echo "  Group B (YaRN-only):    6 jobs (2 models x 3 modes)"
echo "  Group C (Y4+RPE small): 4 jobs (2 models x 2 modes)"
echo "  Group D (Y4+RPE window): 4 jobs (2 models x 2 modes)"
echo "  Group E (Y4+PoSE):      4 jobs (2 models x 2 modes)"
