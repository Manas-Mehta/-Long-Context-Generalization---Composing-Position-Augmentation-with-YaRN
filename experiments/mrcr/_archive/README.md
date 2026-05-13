# MRCR _archive — legacy ablation subdirs

These three subdirs were Phase 5 / Phase 6 / Phase-3-compositional ablations on
MRCR. They contain **only SLURM scripts with pre-migration paths**
(`composable_cot/mrcr_context_extension/...`) and will not run as-is. Moved
here on 2026-05-13 to declutter `experiments/mrcr/` while preserving the
hyperparameter grid as documentation.

| Subdir | What it is | Records |
|---|---|---|
| `seed_test/`           | Seed-variance experiment (4 files) | one train + one eval slurm |
| `phase6/`              | L-sweep × YaRN-factor grid (23 files) | L ∈ {4K, 8K, 16K, 32K, 64K, 128K} × YaRN ∈ {f=2, f=3, f=4} × {RPE-curriculum, PoSE} |
| `expt3_compositional/` | rank-128 LoRA + YaRN-factor grid (9 files) | baseline (r=128, YaRN=4) + comp-y2 (r=128, no/y2/y4) + comp-y3 (r=128, no/y3/y4) |

**Where the actual numbers live:**
- Phase-5 L-sweep results: `~/.claude/projects/.../memory/MEMORY.md` and top-level `HANDOFF.md` §5.
- Per-condition prediction JSONs: HPC archive at `/scratch/mm14444/RPE/composable_cot/mrcr_context_extension/outputs/`.

**Do not run these slurms.** If you need to re-run any of the ablations,
write a fresh slurm using `experiments/mrcr/hpc/smoke_test.slurm` as the
template and the filename-encoded hyperparameters here as the grid.
