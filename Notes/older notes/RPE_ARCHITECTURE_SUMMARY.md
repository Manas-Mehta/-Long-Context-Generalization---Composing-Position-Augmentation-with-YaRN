# PROJECT RPE: Architecture & Progress Summary

## For Presentation / Documentation

**Date**: January 2025  
**Institution**: TAUR Labs, NYU  
**Project**: Randomized Positional Encodings for Composable Chain-of-Thought

---

## 1. Executive Summary

We have successfully implemented Randomized Positional Encodings (RPE) from DeepMind's paper and integrated it with the Composable CoT training framework. The implementation is:

- **Encoding-agnostic**: Works with RoPE, ALiBi, or any position-based encoding
- **Non-invasive**: Only 4 lines added to LLaMA-Factory
- **Configurable**: YAML-based settings, easy enable/disable
- **HPC-ready**: SLURM scripts prepared for NYU cluster

---

## 2. Background: The Length Generalization Problem

### What Happens with Standard Transformers

```
Training:   Sequences of length 1-100      → Positions [0, 1, 2, ..., 99]
Inference:  Sequence of length 200         → Positions [0, 1, 2, ..., 199]
                                                              ↑
                                              Positions 100-199 are OUT OF DISTRIBUTION
                                              Model has never seen them → FAILS
```

### DeepMind's Solution: RPE (arXiv:2305.16843)

Instead of sequential positions, **randomize during training**:

```
Training Batch 1:  positions = [23, 456, 1200, 3400, 7800]    (sorted random from [0, 8192))
Training Batch 2:  positions = [5, 89, 201, 567, 8001]        (different random each batch)
Training Batch 3:  positions = [1000, 1001, 1002, 1003, 1004] (sometimes consecutive)

Result: Model learns to handle ANY position numbers
        → Length generalization achieved!
```

### Paper Results
- **+12% average accuracy** on length generalization tasks
- Tested on 15 algorithmic tasks (reverse string, parity, etc.)
- 6000 models evaluated

---

## 3. Our Implementation Architecture

### Design Decision: Encoding-Agnostic Approach

We operate at the **position_ids level**, not inside specific encodings:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STANDARD FLOW                                    │
│                                                                         │
│   Input Tokens                                                          │
│        ↓                                                                │
│   Position IDs: [0, 1, 2, 3, 4]                                        │
│        ↓                                                                │
│   RoPE / ALiBi / Absolute PE                                           │
│        ↓                                                                │
│   Attention Computation                                                 │
│        ↓                                                                │
│   Output                                                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          RPE FLOW                                        │
│                                                                         │
│   Input Tokens                                                          │
│        ↓                                                                │
│   Position IDs: [0, 1, 2, 3, 4]                                        │
│        ↓                                                                │
│   ┌─────────────────────────────┐                                      │
│   │     RPE TRANSFORMATION      │  ← OUR CODE INTERCEPTS HERE          │
│   │                             │                                       │
│   │  Sample random from [0,L)   │                                       │
│   │  Sort ascending             │                                       │
│   │  [0,1,2,3,4] → [127,892,...]│                                       │
│   └─────────────────────────────┘                                      │
│        ↓                                                                │
│   Randomized IDs: [127, 892, 2341, 5892, 7103]                         │
│        ↓                                                                │
│   RoPE / ALiBi / Absolute PE    ← UNCHANGED, just receives new IDs     │
│        ↓                                                                │
│   Attention Computation                                                 │
│        ↓                                                                │
│   Output                                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Design?

| Aspect | Our Approach | Alternative (Modify RoPE) |
|--------|--------------|---------------------------|
| Code changes | Minimal (monkey-patch) | Extensive (fork model code) |
| Model compatibility | Any HuggingFace model | Specific models only |
| Encoding support | RoPE, ALiBi, Absolute, Relative | Only one encoding type |
| Maintenance | Easy (external patch) | Hard (track upstream changes) |

---

## 4. Integration with Composable CoT

### What is Composable CoT?

Research project on compositional generalization of reasoning skills:
- Train on **atomic tasks** (letter concat, next letter, etc.)
- Combine via **multitask learning** or **model merging**
- Goal: Zero-shot performance on **compositional tasks**

**Paper**: "Learning Composable Chains-of-Thought" (arXiv:2505.22635)

### Why Add RPE to Composable CoT?

| Challenge | How RPE Helps |
|-----------|---------------|
| Longer reasoning chains | Model generalizes to longer sequences |
| Variable-length compositions | Not limited by training sequence length |
| Out-of-distribution lengths | Positions are never OOD after RPE training |

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPOSABLE CoT + RPE PIPELINE                      │
│                                                                         │
│   ┌─────────────────┐                                                  │
│   │  rpe_config.yaml │  ← Configuration (enable/disable, params)       │
│   └────────┬────────┘                                                  │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                     LLaMA-Factory                                │  │
│   │                                                                  │  │
│   │   tuner.py                                                      │  │
│   │      │                                                          │  │
│   │      ├── get_train_args()                                       │  │
│   │      │                                                          │  │
│   │      ├── ✚ RPE INTEGRATION (4 lines added)                      │  │
│   │      │      │                                                   │  │
│   │      │      └── if RPE_CONFIG_PATH:                             │  │
│   │      │              callbacks.append(RPETrainerCallback(...))    │  │
│   │      │                                                          │  │
│   │      └── run_sft() / run_dpo() / etc.                          │  │
│   │             │                                                   │  │
│   │             ▼                                                   │  │
│   │   ┌─────────────────────────────────────────┐                  │  │
│   │   │        RPETrainerCallback               │                  │  │
│   │   │                                         │                  │  │
│   │   │  on_train_begin():                      │                  │  │
│   │   │    → apply_rpe_patch(model)             │                  │  │
│   │   │    → model.forward is now wrapped       │                  │  │
│   │   │                                         │                  │  │
│   │   │  on_train_end():                        │                  │  │
│   │   │    → remove_rpe_patch(model)            │                  │  │
│   │   │    → clean state for checkpoint save    │                  │  │
│   │   └─────────────────────────────────────────┘                  │  │
│   │             │                                                   │  │
│   │             ▼                                                   │  │
│   │      Training Loop                                              │  │
│   │             │                                                   │  │
│   │             ▼                                                   │  │
│   │   ┌─────────────────────────────────────────┐                  │  │
│   │   │         Qwen 2.5 Model                  │                  │  │
│   │   │                                         │                  │  │
│   │   │  forward() [PATCHED]                    │                  │  │
│   │   │    │                                    │                  │  │
│   │   │    ├── Create position_ids              │                  │  │
│   │   │    ├── ✚ RPE: Randomize positions       │                  │  │
│   │   │    ├── Pass to RoPE                     │                  │  │
│   │   │    └── Continue normal forward          │                  │  │
│   │   └─────────────────────────────────────────┘                  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. File Structure

```
RPE/
├── rpe/                                    # Core RPE module (PHASE 1)
│   ├── __init__.py
│   ├── core.py                             # RandomizedPositionalEncoding class
│   ├── patching.py                         # RPEPatcher for HuggingFace models
│   ├── config.py                           # Configuration dataclasses
│   ├── tasks/
│   │   └── reverse_string.py               # Validation task
│   └── tests/
│       └── test_rpe.py                     # 19 pytest tests
│
├── composable_cot/                         # Composable CoT integration (PHASE 2)
│   ├── LLaMA-Factory/
│   │   └── src/llamafactory/train/
│   │       └── tuner.py                    # ✚ Modified: 4 lines added
│   ├── scripts/
│   │   ├── rpe_llamafactory_patch.py       # ✚ Bridge code & TrainerCallback
│   │   ├── rpe_config.yaml                 # ✚ RPE configuration
│   │   ├── slurm_train_rpe.sh              # ✚ HPC job script
│   │   ├── test_rpe_inference.py           # ✚ Inference validation
│   │   └── test_rpe_sanity_checks.py       # ✚ Numerical sanity checks
│   └── data/                               # Composable CoT datasets
│
├── scripts/
│   └── eval_reverse_string.py              # Phase 1 evaluation
│
└── requirements.txt
```

**Lines of code added/modified**:
- Core RPE module: ~770 lines (new)
- LLaMA-Factory: 4 lines (modified)
- Integration scripts: ~400 lines (new)
- **Total**: ~1,170 lines

---

## 6. Validation Results

### Phase 1: Standalone RPE (19/19 tests pass)

| Test Category | Tests | Status |
|---------------|-------|--------|
| Position ID Validity | 7 | ✅ All pass |
| Tensor Shapes | 2 | ✅ All pass |
| Numerical Stability | 2 | ✅ All pass |
| Determinism | 3 | ✅ All pass |
| Mode Switching | 5 | ✅ All pass |

### Phase 2: Composable CoT Integration

| Check | Baseline | RPE | Status |
|-------|----------|-----|--------|
| Positions sorted | N/A | ✅ Yes | Pass |
| Positions unique | N/A | ✅ Yes | Pass |
| Shapes match | ✅ | ✅ | Pass |
| No NaN/Inf | ✅ | ✅ | Pass |
| Logits differ | - | ~98% | Pass |
| Cosine similarity | - | 0.65-0.76 | Expected |
| Output tokens differ | - | 10/10 | Pass |

**Interpretation**: RPE is correctly affecting computation through RoPE without causing numerical issues.

---

## 7. Alignment with Original Paper

| Paper Aspect | Our Implementation | Status |
|--------------|-------------------|--------|
| Random sampling from [0, L) | ✅ `torch.randperm(L)[:seq_len]` | Matches |
| Sorted positions | ✅ `.sort().values` | Matches |
| Sampling without replacement | ✅ `randperm` guarantees this | Matches |
| L >> training sequence length | ✅ L=8192, typical seq=100-500 | Matches |
| Training mode randomizes | ✅ `training=True` flag | Matches |
| Inference mode standard | ✅ `training=False` passthrough | Matches |
| Works with relative PE | ✅ Tested with RoPE (Qwen) | Matches |

### Key Differences (Intentional)

| Paper | Our Implementation | Reason |
|-------|-------------------|--------|
| Encoder-only Transformer | Decoder-only (Qwen) | Modern LLMs are decoder-only |
| JAX/Haiku | PyTorch/HuggingFace | Ecosystem compatibility |
| Custom small models | Qwen 2.5 1.5B/7B | Practical for real tasks |
| Standalone training | LLaMA-Factory integration | Use existing infrastructure |

---

## 8. Deployment Architecture

### Development (Mac M3 Pro)

```
┌─────────────────────────────────────────┐
│           MacBook Pro M3                │
│                                         │
│  • MPS backend (Metal GPU)              │
│  • Qwen 2.5 1.5B (fits in 16GB)        │
│  • Fast iteration on code               │
│  • Run pytest, inference validation     │
└─────────────────────────────────────────┘
```

### Training (NYU HPC)

```
┌─────────────────────────────────────────┐
│           NYU HPC Cluster               │
│                                         │
│  • SLURM job scheduler                  │
│  • CUDA GPUs (A100/V100/RTX8000)       │
│  • Qwen 2.5 7B (full model)            │
│  • Full training runs                   │
│  • Multi-GPU support (DeepSpeed)        │
└─────────────────────────────────────────┘

Job Submission:
  RPE_CONFIG_PATH="scripts/rpe_config.yaml" sbatch scripts/slurm_train_rpe.sh

Baseline (no RPE):
  # Set enabled: false in rpe_config.yaml
  sbatch scripts/slurm_train_rpe.sh
```

---

## 9. Current Status & Next Steps

### Completed ✅

| Phase | Milestone | Status |
|-------|-----------|--------|
| 1 | Environment Setup | ✅ |
| 1 | Core RPE Module | ✅ |
| 1 | Model Patching | ✅ |
| 1 | Sanity Checks (19 tests) | ✅ |
| 1 | Reverse String Validation | ✅ |
| 2 | Composable CoT Inference Test | ✅ |
| 2 | Sanity Checks on CoT Data | ✅ |
| 2 | LLaMA-Factory Integration Code | ✅ |
| 2 | tuner.py Modification | ✅ |

### In Progress 🔄

| Phase | Milestone | Status |
|-------|-----------|--------|
| 2 | Training Pipeline Validation | 🔄 Next |

### Upcoming ⬜

| Phase | Milestone | Description |
|-------|-----------|-------------|
| 2 | Full Training Run | Train on Composable CoT tasks with RPE |
| 2 | Evaluation | Compare RPE vs baseline on length generalization |
| 3 | Benchmark Reproduction | (Optional) Reproduce DeepMind's exact benchmarks |

---

## 10. Expected Outcomes

After completing Phase 2 training:

| Metric | Baseline (No RPE) | With RPE | Improvement |
|--------|-------------------|----------|-------------|
| In-distribution accuracy | ~X% | ~X% | Similar |
| Length generalization | Poor | Good | Significant |
| Longer sequence handling | Fails | Works | Major |

**Based on paper**: Expect ~12% average improvement on length generalization tasks.

---

## 11. Commands Quick Reference

```bash
# Activate environment
cd "/Users/reach/CodingRepositories/02 TAUR Labs/RPE"
source .venv/bin/activate

# Run all tests
pytest rpe/tests/test_rpe.py -v

# Test RPE on Composable CoT data
python composable_cot/scripts/test_rpe_inference.py
python composable_cot/scripts/test_rpe_sanity_checks.py

# Local training validation (Mac)
RPE_CONFIG_PATH="composable_cot/scripts/rpe_config.yaml" \
  python -m llamafactory.cli train <config.yaml>

# HPC training (NYU cluster)
sbatch composable_cot/scripts/slurm_train_rpe.sh
```

---

## 12. References

1. **RPE Paper**: Ruoss et al., "Randomized Positional Encodings Boost Length Generalization of Transformers", ACL 2023. [arXiv:2305.16843](https://arxiv.org/abs/2305.16843)

2. **Composable CoT Paper**: Yin et al., "Learning Composable Chains-of-Thought", 2025. [arXiv:2505.22635](https://arxiv.org/abs/2505.22635)

3. **DeepMind RPE Code**: [github.com/google-deepmind/randomized_positional_encodings](https://github.com/google-deepmind/randomized_positional_encodings)

4. **Composable CoT Code**: [github.com/fc2869/composable_cot](https://github.com/fc2869/composable_cot)

---

## 13. Contact

**Project Lead**: [Your Name]  
**Institution**: TAUR Labs, NYU  
**Email**: [Your Email]
