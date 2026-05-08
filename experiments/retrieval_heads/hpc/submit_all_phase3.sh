#!/bin/bash
# Submit all 3 Phase 3 detection jobs at once (one per condition).
# Each job runs all 9 bins for its condition, ~3-5 hrs wall on h200.
#
# All 3 conditions are evaluated with YaRN factor 4 at detection time
# (matching the multi-entry eval config that produced our correctness labels).
# vanilla Qwen2.5-7B-Instruct is NOT in scope — we have no BABILong-conditioned
# reference for it, and the per-sample correlation analysis needs all probed
# models to share the deployment config of the labels.

set -e

SLURM=composable_cot/retrieval_head_analysis/hpc/phase3_run_detection.slurm

sbatch --job-name=p3_lora    --export=COND=lora_base        "${SLURM}"
sbatch --job-name=p3_y2      --export=COND=y2_base          "${SLURM}"
sbatch --job-name=p3_rpe     --export=COND=y2_rpe_cur_L16k  "${SLURM}"

squeue -u mm14444
