#!/bin/bash
# Submit all 3 multi-entry eval jobs
# Estimated: ~1.7hrs each on H200, ~5hrs total (can run in parallel)
#
# Step 1: Generate filtered eval data (run once, fast, CPU-only)
# Step 2: Submit GPU eval jobs

cd /scratch/mm14444/RPE

echo "=== Step 1: Generate multi-entry eval subset ==="
python composable_cot/BABIlong/scripts/create_multi_entry_eval.py
echo ""

echo "=== Step 2: Submit eval jobs (305 hard samples, all 9 bins) ==="
echo ""

JOB1=$(sbatch composable_cot/BABIlong/hpc/eval_y2_rpe_cur_L16k_me.slurm | awk '{print $4}')
echo "  YaRN+RPE:  job ${JOB1}"

JOB2=$(sbatch composable_cot/BABIlong/hpc/eval_y2_base_me.slurm | awk '{print $4}')
echo "  YaRN-only: job ${JOB2}"

JOB3=$(sbatch composable_cot/BABIlong/hpc/eval_lora_base_me.slurm | awk '{print $4}')
echo "  LoRA+YaRN: job ${JOB3}"

echo ""
echo "All submitted. Monitor with: squeue -u mm14444"
echo "Results will be in: composable_cot/BABIlong/results/*_me/"
