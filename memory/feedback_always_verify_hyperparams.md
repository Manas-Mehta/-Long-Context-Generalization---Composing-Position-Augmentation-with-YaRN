---
name: Always verify hyperparameters against official paper before training
description: User explicitly asked to check LR against official BABILong paper multiple times before training. Was ignored. Cost real GPU money.
type: feedback
---

Always check hyperparameters against the official paper/codebase BEFORE writing training scripts. Do not use "standard defaults" without verification.

**Why:** User asked multiple times to verify LR before BABILong training. Was not done. Official BABILong paper used LR 3e-5–5e-5 for small models. We used 2e-4 (4-7x higher). This caused catastrophic forgetting in 4/6 conditions, wasting GPU budget.

**How to apply:** Before finalizing any training script, explicitly search for the official paper's hyperparameters and state them. If not found, say so clearly and ask the user to confirm the value before proceeding. Never assume "standard LoRA defaults" are appropriate without checking task-specific requirements.

**For BABILong retraining:** Use --lr 5e-5 (not 2e-4). Also add --save-steps 500 to capture best weights before potential collapse.
