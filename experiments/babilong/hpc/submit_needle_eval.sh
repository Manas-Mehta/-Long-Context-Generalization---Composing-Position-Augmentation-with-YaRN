#!/bin/bash
# Submit needle-position experiment pipeline on HPC.
#
# Step 1: Generate eval data (CPU, ~30-60 min) — depends on PG19 cache
# Step 2: Once generation done, submit 2 GPU eval jobs in parallel (~2-4 hrs each)
#
# Usage:
#   bash composable_cot/BABIlong/hpc/submit_needle_eval.sh
#
# Or submit jobs sequentially with dependency:
#   JOB1=$(sbatch ... generate_needle_eval.slurm | awk '{print $4}')
#   sbatch --dependency=afterok:${JOB1} eval_y2_base_needle.slurm
#   sbatch --dependency=afterok:${JOB1} eval_y2_rpe_cur_L16k_needle.slurm

cd /scratch/mm14444/RPE

echo "=== Step 1: Generate needle-position eval data (CPU, ~30-60 min) ==="
JOB_GEN=$(sbatch composable_cot/BABIlong/hpc/generate_needle_eval.slurm | awk '{print $4}')
echo "  Generation job: ${JOB_GEN}"
echo ""

echo "=== Step 2: Submit GPU eval jobs (will wait for generation) ==="
JOB_BASE=$(sbatch --dependency=afterok:${JOB_GEN} \
    composable_cot/BABIlong/hpc/eval_y2_base_needle.slurm | awk '{print $4}')
echo "  YaRN-only eval: job ${JOB_BASE} (afterok:${JOB_GEN})"

JOB_RPE=$(sbatch --dependency=afterok:${JOB_GEN} \
    composable_cot/BABIlong/hpc/eval_y2_rpe_cur_L16k_needle.slurm | awk '{print $4}')
echo "  YaRN+RPE eval:  job ${JOB_RPE} (afterok:${JOB_GEN})"

echo ""
echo "All submitted. Monitor with: squeue -u mm14444"
echo ""
echo "Expected outputs:"
echo "  Gen:  composable_cot/BABIlong/data/eval_needle/{zone}_{bin}.json (27 files)"
echo "  Eval: composable_cot/BABIlong/results/y2_base_needle_{beg,mid,end}/"
echo "        composable_cot/BABIlong/results/y2_rpe_cur_L16k_needle_{beg,mid,end}/"
