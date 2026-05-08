#!/bin/bash
# Submit needle-position v2 experiment pipeline on HPC.
#
# v2 changes vs v1:
#   - 227 samples (207 hard multi-entry + 20 single-entry ref) vs 100
#   - 3 models (lora_base, y2_base, y2_rpe_cur_L16k) vs 2
#   - All models eval with YaRN f=4
#
# Step 1: Generate eval data (CPU, ~1-2 hrs with PG19 cached)
# Step 2: Once generation done, submit 3 GPU eval jobs in parallel (~4-8 hrs each)
#
# Usage:
#   bash composable_cot/BABIlong/hpc/submit_needle_eval_v2.sh

cd /scratch/mm14444/RPE

echo "=== Step 1: Generate needle-position eval data v2 (CPU, ~1-2 hrs) ==="
JOB_GEN=$(sbatch composable_cot/BABIlong/hpc/generate_needle_eval_v2.slurm | awk '{print $4}')
echo "  Generation job: ${JOB_GEN}"
echo ""

echo "=== Step 2: Submit GPU eval jobs (will wait for generation) ==="
JOB_LORA=$(sbatch --dependency=afterok:${JOB_GEN} \
    composable_cot/BABIlong/hpc/eval_lora_base_needle_v2.slurm | awk '{print $4}')
echo "  LoRA-base eval: job ${JOB_LORA} (afterok:${JOB_GEN})"

JOB_BASE=$(sbatch --dependency=afterok:${JOB_GEN} \
    composable_cot/BABIlong/hpc/eval_y2_base_needle_v2.slurm | awk '{print $4}')
echo "  YaRN-only eval: job ${JOB_BASE} (afterok:${JOB_GEN})"

JOB_RPE=$(sbatch --dependency=afterok:${JOB_GEN} \
    composable_cot/BABIlong/hpc/eval_y2_rpe_cur_L16k_needle_v2.slurm | awk '{print $4}')
echo "  YaRN+RPE eval:  job ${JOB_RPE} (afterok:${JOB_GEN})"

echo ""
echo "All submitted. Monitor with: squeue -u mm14444"
echo ""
echo "Expected outputs:"
echo "  Gen:  composable_cot/BABIlong/data/eval_needle_v2/{zone}_{bin}.json (24 files)"
echo "  Eval: composable_cot/BABIlong/results/lora_base_needle_v2_{beg,mid,end}/"
echo "        composable_cot/BABIlong/results/y2_base_needle_v2_{beg,mid,end}/"
echo "        composable_cot/BABIlong/results/y2_rpe_cur_L16k_needle_v2_{beg,mid,end}/"
