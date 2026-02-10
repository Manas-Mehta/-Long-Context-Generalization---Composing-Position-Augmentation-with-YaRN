#!/usr/bin/env python
"""Detailed sanity checks for RPE on Composable CoT data.

Runs numerical diagnostics across 3 examples:
  1. Position ID properties (sorted, unique, range, statistics)
  2. Tensor shapes (input/output preserved)
  3. Numerical stability (NaN, Inf, logit range)
  4. Logit comparison (MAD, max diff, cosine similarity)
  5. Output token comparison (decoded side-by-side)

Usage:
    python composable_cot/scripts/test_rpe_sanity_checks.py
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpe.core import RandomizedPositionalEncoding
from rpe.patching import RPEPatcher

CCOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = "data/composition/answer_only/letter_concat_ascii_multiply/test.json"

NUM_EXAMPLES = 3
MAX_SIM_LENGTH = 8192
RPE_SEED = 42
MAX_NEW_TOKENS = 10  # Short — we only need a few tokens for diagnostics


def section(title: str) -> None:
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def main():
    print("=" * 72)
    print("  RPE Sanity Checks on Composable CoT Data")
    print("=" * 72)

    # --- Setup ---
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.eval()

    with open(os.path.join(CCOT_ROOT, DATA_PATH)) as f:
        data = json.load(f)
    examples = data[:NUM_EXAMPLES]
    print(f"Loaded {NUM_EXAMPLES} examples from {DATA_PATH}")

    rpe = RandomizedPositionalEncoding(max_simulation_length=MAX_SIM_LENGTH, seed=RPE_SEED)

    # Storage for summary table
    summary_rows = []

    for ex_idx, ex in enumerate(examples):
        prompt = ex["instruction"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt")  # (1, seq_len)
        seq_len = input_ids.shape[1]

        print(f"\n{'━' * 72}")
        print(f"  EXAMPLE {ex_idx} — seq_len={seq_len} tokens")
        print(f"  prompt (tail): ...{prompt[-70:]}")
        print(f"{'━' * 72}")

        # ── 1. Position ID Checks ──────────────────────────────────────────
        section("1. Position ID Checks")

        standard_pos = torch.arange(seq_len)
        rpe.reset_seed(RPE_SEED)
        rpe_pos = rpe.get_randomized_positions(seq_len)

        is_sorted = torch.all(rpe_pos[1:] > rpe_pos[:-1]).item()
        is_unique = len(torch.unique(rpe_pos)) == seq_len
        in_range = (rpe_pos.min().item() >= 0) and (rpe_pos.max().item() < MAX_SIM_LENGTH)

        print(f"  Standard IDs (first 8):  {standard_pos[:8].tolist()}")
        print(f"  RPE IDs (first 8):       {rpe_pos[:8].tolist()}")
        print(f"  RPE IDs (last 8):        {rpe_pos[-8:].tolist()}")
        print(f"  Sorted (ascending):      {is_sorted}")
        print(f"  All unique:              {is_unique}")
        print(f"  In range [0, {MAX_SIM_LENGTH}):     {in_range}")
        print(f"  Min:   {rpe_pos.min().item()}")
        print(f"  Max:   {rpe_pos.max().item()}")
        print(f"  Mean:  {rpe_pos.float().mean().item():.1f}")
        print(f"  Stdev: {rpe_pos.float().std().item():.1f}")

        # ── 2. Tensor Shape Checks ─────────────────────────────────────────
        section("2. Tensor Shape Checks")

        print(f"  input_ids shape:  {tuple(input_ids.shape)}")

        with torch.no_grad():
            baseline_out = model(input_ids=input_ids)
        baseline_logits = baseline_out.logits
        print(f"  Baseline logits:  {tuple(baseline_logits.shape)}")

        patcher = RPEPatcher(model, {"max_simulation_length": MAX_SIM_LENGTH, "seed": RPE_SEED})
        patcher.patch()
        with torch.no_grad():
            rpe_out = model(input_ids=input_ids)
        rpe_logits = rpe_out.logits
        patcher.unpatch()

        print(f"  RPE logits:       {tuple(rpe_logits.shape)}")
        shapes_match = baseline_logits.shape == rpe_logits.shape
        print(f"  Shapes match:     {shapes_match}")

        # ── 3. Numerical Stability ─────────────────────────────────────────
        section("3. Numerical Stability")

        b_nan = torch.isnan(baseline_logits).sum().item()
        r_nan = torch.isnan(rpe_logits).sum().item()
        b_inf = torch.isinf(baseline_logits).sum().item()
        r_inf = torch.isinf(rpe_logits).sum().item()

        b_min = baseline_logits.min().item()
        b_max = baseline_logits.max().item()
        r_min = rpe_logits.min().item()
        r_max = rpe_logits.max().item()

        print(f"  Baseline NaN count:  {b_nan}")
        print(f"  RPE NaN count:       {r_nan}")
        print(f"  Baseline Inf count:  {b_inf}")
        print(f"  RPE Inf count:       {r_inf}")
        print(f"  Baseline logit range: [{b_min:.2f}, {b_max:.2f}]")
        print(f"  RPE logit range:      [{r_min:.2f}, {r_max:.2f}]")

        # ── 4. Logit Comparison ────────────────────────────────────────────
        section("4. Logit Comparison")

        # Cast to float32 for stable comparison math
        bl = baseline_logits.float()
        rl = rpe_logits.float()
        abs_diff = (bl - rl).abs()

        mad = abs_diff.mean().item()
        max_diff = abs_diff.max().item()
        pct_differ = (abs_diff > 0.01).float().mean().item() * 100
        total_elements = abs_diff.numel()

        bl_flat = bl.reshape(-1)
        rl_flat = rl.reshape(-1)
        cosine_sim = F.cosine_similarity(bl_flat.unsqueeze(0), rl_flat.unsqueeze(0)).item()

        print(f"  Mean absolute diff:     {mad:.4f}")
        print(f"  Max absolute diff:      {max_diff:.4f}")
        print(f"  Elements differing >0.01: {pct_differ:.1f}% of {total_elements:,}")
        print(f"  Cosine similarity:      {cosine_sim:.6f}")

        # ── 5. Output Token Comparison ─────────────────────────────────────
        section("5. Output Token Comparison")

        # Greedy next-token predictions from last position at each step
        # For simplicity, compare argmax token IDs from the logits at each position
        n_compare = min(MAX_NEW_TOKENS, seq_len)
        # Use last n_compare positions' logits to get predicted next tokens
        b_pred_ids = baseline_logits[0, -n_compare:, :].argmax(dim=-1)
        r_pred_ids = rpe_logits[0, -n_compare:, :].argmax(dim=-1)

        b_tokens = [tokenizer.decode([tid]) for tid in b_pred_ids]
        r_tokens = [tokenizer.decode([tid]) for tid in r_pred_ids]
        n_differ = (b_pred_ids != r_pred_ids).sum().item()

        print(f"  Comparing argmax tokens at last {n_compare} positions:")
        print(f"  Baseline IDs:  {b_pred_ids.tolist()}")
        print(f"  RPE IDs:       {r_pred_ids.tolist()}")
        print(f"  Tokens differ: {n_differ}/{n_compare}")
        print()
        print(f"  {'Pos':>4}  {'B_id':>7}  {'R_id':>7}  {'Match':>5}  {'B_token':<15}  {'R_token':<15}")
        for i in range(n_compare):
            match = "  =" if b_pred_ids[i] == r_pred_ids[i] else " !="
            print(f"  {i:>4}  {b_pred_ids[i].item():>7}  {r_pred_ids[i].item():>7}  {match:>5}  {repr(b_tokens[i]):<15}  {repr(r_tokens[i]):<15}")

        # Collect for summary
        summary_rows.append({
            "ex": ex_idx,
            "seq_len": seq_len,
            "sorted": is_sorted,
            "unique": is_unique,
            "in_range": in_range,
            "shapes_match": shapes_match,
            "b_nan": b_nan,
            "r_nan": r_nan,
            "b_inf": b_inf,
            "r_inf": r_inf,
            "mad": mad,
            "max_diff": max_diff,
            "cosine": cosine_sim,
            "tokens_differ": n_differ,
            "tokens_total": n_compare,
        })

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'━' * 72}")
    print("  SUMMARY TABLE")
    print(f"{'━' * 72}")

    hdr = (f"  {'Ex':>2}  {'SeqLen':>6}  {'Sort':>4}  {'Uniq':>4}  {'Range':>5}"
           f"  {'Shape':>5}  {'B_NaN':>5}  {'R_NaN':>5}  {'B_Inf':>5}  {'R_Inf':>5}"
           f"  {'MAD':>8}  {'MaxDif':>8}  {'Cosine':>8}  {'TokDif':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_pass = True
    for r in summary_rows:
        tok_str = f"{r['tokens_differ']}/{r['tokens_total']}"
        row_pass = (r["sorted"] and r["unique"] and r["in_range"]
                    and r["shapes_match"] and r["b_nan"] == 0 and r["r_nan"] == 0
                    and r["b_inf"] == 0 and r["r_inf"] == 0 and r["tokens_differ"] > 0)
        if not row_pass:
            all_pass = False
        print(f"  {r['ex']:>2}  {r['seq_len']:>6}  {'Y' if r['sorted'] else 'N':>4}"
              f"  {'Y' if r['unique'] else 'N':>4}  {'Y' if r['in_range'] else 'N':>5}"
              f"  {'Y' if r['shapes_match'] else 'N':>5}  {r['b_nan']:>5}  {r['r_nan']:>5}"
              f"  {r['b_inf']:>5}  {r['r_inf']:>5}"
              f"  {r['mad']:>8.4f}  {r['max_diff']:>8.2f}  {r['cosine']:>8.6f}  {tok_str:>8}")

    print(f"\n{'━' * 72}")
    print(f"  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'━' * 72}")


if __name__ == "__main__":
    main()
