# Multi-Entry Eval — File Index

All files needed to analyze the multi-entry (305-sample) eval run on 2026-04-13.

No analysis performed here — just a collection of inputs, outputs, and scripts.

---

## Folder contents

### `logs/` — SLURM eval outputs (completed runs)
- `eval_y2_base_me.out` — YaRN-only condition, trained f=2, eval f=4
- `eval_y2_rpe_cur_L16k_me.out` — YaRN+RPE curriculum L=16K condition, eval f=4

Each log contains per-bin accuracy, sample counts, timing. Final summary block at bottom.

*LoRA-base eval not run (decided unnecessary).*

### `prior_100sample_summaries/` — previous eval results for comparison
- `y2_base_1k_summary.json` — prior 100-sample eval (mixed single+multi entry)
- `y2_rpe_cur_L16k_1k_summary.json` — prior 100-sample eval

These are the "before" numbers from the original 100-sample sweep. Use these to compare against the new 305-multi-entry numbers.

### `dataset_info/` — eval dataset metadata + tags
- `metadata.json` — multi-entry dataset description (305 samples, per-object and per-entry-count distribution)
- `sample_difficulty_tags.json` — per-sample tags for ALL 999 samples in original eval: `{idx, obj, target_entries, multi}`. This is what filtered the 305 indices.
- `sample_excerpts.json` — first sample from each bin (0k through 128k), with long context fields truncated. Shows what the data actually looks like.

Full filtered dataset (not included here — too large): `composable_cot/BABIlong/data/eval_multi_entry/*.json` (~310 MB total)

### `scripts/` — reproducibility
- `create_multi_entry_eval.py` — filters full eval → multi-entry subset using tags
- `eval_y2_base_me.slurm` — SLURM script for YaRN-only eval
- `eval_y2_rpe_cur_L16k_me.slurm` — SLURM script for YaRN+RPE eval
- `eval_lora_base_me.slurm` — SLURM script for LoRA-base (not run)
- `submit_multi_entry_eval.sh` — orchestration: runs create script then sbatches all 3

---

## Raw prediction JSONs (on HPC only, not local)

Per-sample predictions are on HPC at:
- `/scratch/mm14444/RPE/composable_cot/BABIlong/results/y2_base_me/predictions_*.json`
- `/scratch/mm14444/RPE/composable_cot/BABIlong/results/y2_rpe_cur_L16k_me/predictions_*.json`

To pull locally for per-sample analysis:
```
scp -r mm14444@login.torch.hpc.nyu.edu:/scratch/mm14444/RPE/composable_cot/BABIlong/results/y2_base_me ./
scp -r mm14444@login.torch.hpc.nyu.edu:/scratch/mm14444/RPE/composable_cot/BABIlong/results/y2_rpe_cur_L16k_me ./
```

---

## Key numbers (for quick reference — from the .out logs)

**YaRN-only (y2_base_me):** Overall 86.3% (2333/2703)
- 0k: 87.2% | 1k: 91.7% | 2k: 92.4% | 4k: 91.5% | 8k: 89.5%
- 16k: 89.5% | 32k: 86.2% | 64k: 82.3% | 128k: 67.2%

**YaRN+RPE (y2_rpe_cur_L16k_me):** Overall 90.9% (2457/2703)
- 0k: 89.5% | 1k: 90.1% | 2k: 90.8% | 4k: 92.5% | 8k: 93.8%
- 16k: 93.8% | 32k: 93.4% | 64k: 90.2% | 128k: 83.9%

**Dataset:** 305 multi-entry samples (264 at 1k bin — fewer samples available), 2703 total eval examples across 9 bins.
