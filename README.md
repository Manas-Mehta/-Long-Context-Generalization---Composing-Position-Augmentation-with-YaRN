# Composing Position-ID Augmentation with YaRN for Long-Context Generalisation

Code accompanying the paper *"Composing Position-ID Augmentation with YaRN for Long-Context Generalisation"*.

We finetune Qwen2.5-7B-Instruct with LoRA on BABILong QA3 at up to 32K tokens and evaluate out-of-distribution up to 128K. Pairing a position-ID augmentation (RPE or PoSE) at training time with YaRN at evaluation consistently improves long-context accuracy over either alone. At 128K, YaRN alone gives 69%, the LoRA baseline gives 62%, RPE training with YaRN at eval reaches 81%, and combining YaRN at training and evaluation with RPE pushes this to 84%.

## Repository layout

```
posaug/                         Importable package
  core.py                       Randomized Positional Encodings (RPE)
  pose.py                       Positional Skip-wisE (PoSE)
  patching.py / pose_patching.py
                                Monkey-patches that swap position_ids during forward
  callbacks_rpe.py              HuggingFace TrainerCallback for RPE
  callbacks_pose.py             HuggingFace TrainerCallback for PoSE
  config.py                     Config-loading helpers
  tests/                        Unit tests (pytest posaug/tests)

experiments/
  babilong/                     Main BABILong QA3 experiment (§4, §5 of paper)
  mrcr/                         MRCR cross-benchmark observation (§4.3)
  retrieval_heads/              Mechanistic probe with QR heads (§6)

requirements.txt
pytest.ini
.gitignore
```

Each experiment folder is self-contained — its own `scripts/`, `analysis/`,
`configs/`, `data/` (gitignored), `results/` (gitignored except summaries),
`hpc/` (SLURM templates) and a per-experiment README.

## Install

```bash
git clone https://github.com/Manas-Mehta/-Long-Context-Generalization---Composing-Position-Augmentation-with-YaRN.git
cd -Long-Context-Generalization---Composing-Position-Augmentation-with-YaRN
pip install -r requirements.txt
```

The scripts add the repo root to `sys.path` themselves (so `import posaug`
works without a separate `pip install -e .`). Tested with Python 3.11,
PyTorch 2.x, transformers 4.52+.

## Reproducing the paper

| Section | Experiment dir | Entry-point |
|---|---|---|
| §4.1, §4.2 BABILong main + multi-entry | `experiments/babilong/` | `scripts/train_babilong_lora.py`, `scripts/eval_babilong.py` |
| §4.3 MRCR cross-benchmark | `experiments/mrcr/` | `scripts/train_mrcr_lora.py`, `scripts/eval_mrcr.py` |
| §5 Position sensitivity (zone study) | `experiments/babilong/` | `scripts/build_needle_selection_v2.py`, `scripts/generate_needle_position_eval.py` |
| §6 Retrieval-head probe | `experiments/retrieval_heads/` | `scripts/build_detection_set.py`, `scripts/phase3_run_detection.py` |

See each experiment's `README.md` for step-by-step commands.

## Data

Datasets are not committed. Each experiment's README documents how to
download or regenerate them:

- BABILong QA3: `RMT-team/babilong-train-5k-samples` on HuggingFace
- MRCR: `openai/mrcr` on HuggingFace
- Retrieval-head detection set: built from the BABILong multi-entry subset

Targets are gitignored (`experiments/*/data/`, `experiments/*/checkpoints/`,
`experiments/*/results/`).

## External dependencies

The `experiments/retrieval_heads/` probe uses the published QRHead detection
code from Wu et al. (2024). See `experiments/retrieval_heads/README.md`
for installation.

## Citation

```bibtex
@misc{positionaug-yarn,
  title  = {Composing Position-ID Augmentation with YaRN for Long-Context Generalisation},
  author = {Mehta, Manas and others},
  year   = {2026},
}
```
