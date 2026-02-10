# Ideas and Proposals for RPE + CCoT Experiments

*Potential improvements and new directions to discuss in meetings.*

---

## Idea 1: Scale-Matched L (Reduce max_simulation_length)

**Problem:** Our current L=8192 samples positions from a huge range. CCoT compositions only produce sequences of ~200-600 tokens. LoRA with rank 8 (~20M params) doesn't have the capacity to learn position-invariance across such a wide range.

**Proposal:** Reduce L to match the actual composition scale:
- L=512 (2-3x longest composition) -- **recommended starting point**
- L=1024 (5x longest composition) -- conservative
- Run an ablation: L=256, 512, 1024, 2048, 8192

**Why this makes sense:** DeepMind used large L because they trained from scratch with full parameter updates. With LoRA's limited capacity, a tighter position range means easier learning. CCoT's own random prefix shifts by only 50-100 tokens -- we should be in that ballpark, not 100x beyond it.

**Framing for meeting:** "We're asking LoRA to learn position-invariance across 8192 positions with only 20M params. If we scale L down to match the actual composition lengths (~500 tokens), the task becomes tractable."

---

## Idea 2: Asymmetric LoRA Rank for Position-Sensitive Layers

**Problem:** All 7 LoRA targets get the same rank (8), but position information flows through q_proj and k_proj specifically (RoPE is applied to Q and K, not V or MLP layers).

**Proposal:** Give higher rank to position-sensitive layers:
```yaml
# Hypothetical config (requires custom implementation)
q_proj: rank 32    # Directly affected by RoPE/RPE
k_proj: rank 32    # Directly affected by RoPE/RPE
v_proj: rank 8     # Not directly affected
o_proj: rank 8     # Not directly affected
gate_proj: rank 4  # Position-independent
up_proj: rank 4    # Position-independent
down_proj: rank 4  # Position-independent
```

**Why this makes sense:** RPE changes position_ids, which only affects Q and K through RoPE. The LoRA adapters on q_proj and k_proj need the most capacity to learn position-invariant transformations. MLP layers don't see positions directly.

**Caveat:** LLaMA-Factory doesn't natively support per-module rank. Would need a custom PEFT config or a code patch. But worth proposing as a direction.

**Framing for meeting:** "Since RPE operates through RoPE which only touches Q and K, the LoRA adapters on q_proj and k_proj need the most capacity for position-invariance. We could concentrate parameters where they matter most."

---

## Idea 3: Curriculum Learning for RPE

**Problem:** Jumping straight from sequential positions to fully random positions in [0, 8192) may be too hard for the LoRA adapters to learn in one step.

**Proposal:** Gradually increase the randomness of positions during training:
- Epoch 1: L=seq_len (basically sequential, with tiny perturbations)
- Epoch 2: L=2*seq_len (mild randomization)
- Epoch 3: L=5*seq_len (moderate)
- Epoch 4-5: L=target_L (full randomization)

**Why this makes sense:** Curriculum learning is well-established -- start easy, increase difficulty. Instead of asking the model to handle arbitrary positions immediately, let it first learn the task, then gradually handle wider position ranges.

**Implementation:** Modify the RPE callback to accept an epoch-dependent L schedule.

**Framing for meeting:** "Instead of shocking the model with random positions from [0, 8192) from the start, we warm up gradually. This lets the LoRA adapters first learn the task structure, then learn position-invariance."

---

## Idea 4: RPE with Position Anchoring

**Problem:** In vanilla RPE, ALL positions are randomized, including the instruction tokens. The model needs to understand the instruction to do the task -- randomizing instruction positions might confuse it.

**Proposal:** Keep instruction positions sequential, only randomize output/CoT positions. This is the "V2: Output-only RPE" variant from the meeting notes.

**Why this makes sense for CCoT specifically:** In CCoT composition, the instruction stays the same -- it's the CoT trace that appears at different positions. So the model should have stable instruction understanding and flexible CoT positioning.

**Implementation status:** Mentioned in meeting notes as medium priority. Requires modifying RPEPatcher to accept a boundary index separating instruction from output.

**Framing for meeting:** "The instruction format is fixed -- it doesn't need length generalization. Only the CoT trace shifts position during composition. We should match RPE to this reality."

---

## Idea 5: LoRA on Q/K Only + Full Fine-Tune MLP

**Problem:** LoRA rank 8 gives only 20M params. Full fine-tuning of 0.5B gave 490M. The gap is too large.

**Proposal:** Hybrid approach -- use LoRA for attention layers but fully fine-tune (unfreeze) the MLP layers:
- q_proj, k_proj, v_proj, o_proj: LoRA rank 8-16 (for position adaptation)
- gate_proj, up_proj, down_proj: Full fine-tuning (for task learning capacity)

**Parameter count:** MLP layers in Qwen2.5-7B = 28 layers * 3 * (3584 * 18944) ≈ 5.7B params. That's too many to fully unfreeze.

**Alternative:** LoRA on attention + LoRA with MUCH higher rank on MLP:
- Attention LoRA: rank 8
- MLP LoRA: rank 64

This would give: 28 * (22,400 * 8 + 67,584 * 64) ≈ 126M params. Still reasonable.

**Framing for meeting:** "We can allocate LoRA capacity where it's needed -- moderate rank for attention (position adaptation) and high rank for MLP (task learning capacity)."

---

## Idea 6: Full Fine-Tuning on Qwen2.5-0.5B (from Meeting Notes)

**This is already on the escalation path.** If LoRA on 7B doesn't work:

**Proposal:** Full fine-tuning of Qwen2.5-0.5B with RPE, then test on CCoT tasks.
- 490M trainable params vs 20M with LoRA rank 8 on 7B
- We already showed Phase 1 works with full fine-tuning
- The question: does 0.5B have enough pretrained knowledge to do CCoT tasks?

**Memory requirements:**
- Model: ~1GB (bf16)
- Optimizer: ~4GB (AdamW, 2 states per param)
- Gradients: ~1GB
- Total: ~6-8GB -- easily fits on a single GPU

**Framing for meeting:** "This is the clearest apples-to-apples comparison: same full fine-tuning as Phase 1, but on a model with pretrained knowledge. If this works, it confirms the issue is LoRA capacity, not RPE incompatibility with pretrained models."

---

## Idea 7: Compare RPE vs CCoT's Random Prefix Head-to-Head

**Problem:** CCoT's random prefix and RPE both simulate positional shifts. We don't know if they're redundant or complementary.

**Proposal:** 2x2 factorial experiment on letter_concat:

| | No Random Prefix | With Random Prefix (CCoT) |
|--|--|--|
| **No RPE** | Pure baseline | Standard CCoT |
| **RPE** | RPE replaces prefix | RPE + CCoT combined |

To do "No Random Prefix": use only Format A examples from letter_concat (the 2000 without gibberish).
To do "With Random Prefix": use the full 4000 (2000 A + 2000 B).

This cleanly separates the effects of CCoT's data-level position shifting from RPE's encoding-level position shifting.

**Framing for meeting:** "This tells us whether RPE adds value on top of CCoT, replaces it, or is redundant. It's the cleanest test of our core research question."

---

## Idea 8: RPE-Aware Evaluation (Use RPE at Inference Too)

**Problem:** We train with RPE (random positions) but evaluate with standard sequential positions. If the LoRA adapter has "overfit" to random positions, it fails at sequential inference.

**Proposal:** Try evaluating with several position strategies:
1. Standard sequential [0, 1, 2, ...] (current)
2. Positions sampled from [0, L) with the same strategy as training
3. Positions sampled multiple times, take majority vote (ensemble)

**Why this could help diagnostically:** If the model works with random positions at inference but fails with sequential, the LoRA adapter learned RPE-specific patterns rather than position-invariant patterns. This helps us understand whether the problem is capacity (not enough params) or generalization (wrong kind of learning).

**Framing for meeting:** "This diagnostic test tells us whether the LoRA adapter learned position-invariance (works with any positions) or position-specific patterns (only works with the same randomization it was trained with)."

---

## Priority Ranking

| Priority | Idea | Effort | Expected Impact |
|----------|------|--------|-----------------|
| 1 | Scale-Matched L | LOW (just change config) | HIGH |
| 2 | Output-only RPE (V2) | MEDIUM (code change) | HIGH |
| 3 | Curriculum Learning | MEDIUM (code change) | MEDIUM |
| 4 | Full FT on 0.5B | LOW (change config) | HIGH (diagnostic) |
| 5 | 2x2 Factorial (RPE vs Prefix) | LOW (data selection) | HIGH (research insight) |
| 6 | RPE-Aware Eval | LOW (eval script change) | MEDIUM (diagnostic) |
| 7 | Asymmetric LoRA Rank | HIGH (custom PEFT) | MEDIUM |
| 8 | Hybrid LoRA + Full FT | HIGH (custom training) | UNKNOWN |
