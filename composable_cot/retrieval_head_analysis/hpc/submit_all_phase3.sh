#!/bin/bash
# Submit all 4 Phase 3 detection jobs at once (one per condition).
# Each job runs all 9 bins for its condition, ~3 hrs wall on h200.

set -e

SLURM=composable_cot/retrieval_head_analysis/hpc/phase3_run_detection.slurm

sbatch --job-name=p3_vanilla --export=COND=vanilla_qwen     "${SLURM}"
sbatch --job-name=p3_lora    --export=COND=lora_base        "${SLURM}"
sbatch --job-name=p3_y2      --export=COND=y2_base          "${SLURM}"
sbatch --job-name=p3_rpe     --export=COND=y2_rpe_cur_L16k  "${SLURM}"

squeue -u mm14444
