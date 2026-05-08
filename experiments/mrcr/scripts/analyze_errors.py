#!/usr/bin/env python3
"""Qualitative error analysis on MRCR predictions.

Categorizes each prediction into error types and generates summary tables
across conditions and bins.

Usage:
    python experiments/mrcr/scripts/analyze_errors.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────

OUTPUTS_DIR = Path("experiments/mrcr/outputs")

# Conditions to analyze (directory prefix -> display name)
CONDITIONS = {
    "lora_baseline": "LoRA baseline",
    "rpe_curriculum_lora_L16k": "RPE cur L=16K",
    "rpe_lora_yarn_eval": "RPE fixed+YaRN",
    "pose_lora": "PoSE fixed",
    "yarn_lora": "YaRN+LoRA",
}

BINS = [
    ("bin0_4K-8K", "4K-8K"),
    ("bin1_8K-16K", "8K-16K"),
    ("bin2_16K-32K", "16K-32K"),
    ("bin3_32K-64K", "32K-64K"),
    ("bin4_64K-128K", "64K-128K"),
]

# ── Error categorization thresholds ───────────────────────────────────

PERFECT_THRESHOLD = 0.95    # score >= this → "perfect"
HIGH_PARTIAL_THRESHOLD = 0.5  # score >= this → "high_partial" (likely right needle, minor errors)
LOW_PARTIAL_THRESHOLD = 0.15  # score >= this → "low_partial" (likely wrong needle or major errors)
# score < LOW_PARTIAL_THRESHOLD and > 0 → "near_zero"
# score == 0.0 → check prefix


def categorize_error(pred):
    """Categorize a single prediction into an error type.

    Returns: (category, detail) tuple
        category: one of 'perfect', 'prefix_missing', 'high_partial',
                  'low_partial', 'near_zero', 'empty'
        detail: human-readable explanation
    """
    score = pred["score"]
    response = pred["response_preview"]
    answer = pred["answer_preview"]

    # Extract the random prefix from the answer (first 10 chars)
    # Answer always starts with the random string
    random_prefix = pred.get("random_string", "")
    if not random_prefix and len(answer) >= 10:
        # Infer prefix: answer starts with it, typically 10 alphanumeric chars
        # Find where the non-alnum content starts
        for i, c in enumerate(answer):
            if not c.isalnum():
                random_prefix = answer[:i]
                break
        if not random_prefix:
            random_prefix = answer[:10]

    generated_tokens = pred.get("generated_tokens", 0)

    # Category 1: Empty or near-empty response
    if generated_tokens <= 3 or len(response.strip()) == 0:
        return "empty", f"Generated only {generated_tokens} tokens"

    # Category 2: Perfect retrieval
    if score >= PERFECT_THRESHOLD:
        return "perfect", f"score={score:.3f}"

    # Category 3: Score is exactly 0.0 — prefix missing
    if score == 0.0:
        # Check if the response starts with something other than the prefix
        if response.strip().startswith(random_prefix):
            return "prefix_present_zero", f"Has prefix but score=0 (unexpected)"
        else:
            # What did it start with?
            preview = response[:80].replace('\n', '\\n')
            return "prefix_missing", f"Started with: '{preview}'"

    # Category 4: Has prefix, check quality
    has_prefix = response.startswith(random_prefix) if random_prefix else True

    if not has_prefix:
        # Has some content but wrong prefix — shouldn't happen with score > 0
        # (grade_mrcr returns 0 if prefix missing), but check anyway
        preview = response[:80].replace('\n', '\\n')
        return "prefix_wrong", f"score={score:.3f}, started with: '{preview}'"

    # Has correct prefix, but imperfect score — analyze quality
    if score >= HIGH_PARTIAL_THRESHOLD:
        return "high_partial", f"score={score:.3f} (right needle, minor errors)"
    elif score >= LOW_PARTIAL_THRESHOLD:
        return "low_partial", f"score={score:.3f} (possibly wrong needle or major truncation)"
    else:
        return "near_zero_with_prefix", f"score={score:.3f} (has prefix but very low match)"


def load_predictions(condition_prefix, bin_suffix):
    """Load predictions.json for a condition+bin combo."""
    dirname = f"{condition_prefix}_{bin_suffix}"
    path = OUTPUTS_DIR / dirname / "predictions.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_all():
    """Run analysis across all conditions and bins."""

    # ── Collect all results ───────────────────────────────────────────
    # results[condition][bin] = list of (category, detail, pred) tuples
    results = defaultdict(lambda: defaultdict(list))

    for cond_prefix, cond_name in CONDITIONS.items():
        for bin_suffix, bin_label in BINS:
            preds = load_predictions(cond_prefix, bin_suffix)
            if preds is None:
                print(f"  WARNING: Missing {cond_prefix}_{bin_suffix}")
                continue
            for pred in preds:
                cat, detail = categorize_error(pred)
                results[cond_name][bin_label].append((cat, detail, pred))

    # ── Table 1: Error type distribution per condition × bin ──────────
    categories = ["perfect", "high_partial", "low_partial", "near_zero_with_prefix",
                   "prefix_missing", "empty", "prefix_present_zero", "prefix_wrong"]
    cat_short = {
        "perfect": "Perfect",
        "high_partial": "High partial",
        "low_partial": "Low partial",
        "near_zero_with_prefix": "Near-zero+pfx",
        "prefix_missing": "No prefix",
        "empty": "Empty",
        "prefix_present_zero": "Pfx+zero",
        "prefix_wrong": "Wrong pfx",
    }

    print("\n" + "=" * 90)
    print("TABLE 1: ERROR TYPE COUNTS PER CONDITION × BIN")
    print("=" * 90)

    for cond_name in CONDITIONS.values():
        print(f"\n── {cond_name} ──")
        # Header
        header = f"{'Bin':<12}"
        for cat in categories:
            if cat in cat_short:
                header += f" {cat_short[cat]:>14}"
        header += f" {'Total':>7}"
        print(header)
        print("-" * len(header))

        for _, bin_label in BINS:
            entries = results[cond_name][bin_label]
            counts = defaultdict(int)
            for cat, _, _ in entries:
                counts[cat] += 1

            row = f"{bin_label:<12}"
            for cat in categories:
                c = counts.get(cat, 0)
                row += f" {c:>14}" if c > 0 else f" {'·':>14}"
            row += f" {len(entries):>7}"
            print(row)

    # ── Table 2: Condensed comparison (% perfect + % prefix_missing) ──
    print("\n" + "=" * 90)
    print("TABLE 2: KEY METRICS PER CONDITION × BIN (% of samples)")
    print("=" * 90)

    print(f"\n{'Condition':<20} {'Bin':<12} {'%Perfect':>9} {'%NoPfx':>8} {'%HiPart':>8} {'%LoPart':>8} {'AvgScore':>9}")
    print("-" * 76)

    for cond_name in CONDITIONS.values():
        for _, bin_label in BINS:
            entries = results[cond_name][bin_label]
            if not entries:
                continue
            n = len(entries)
            counts = defaultdict(int)
            for cat, _, _ in entries:
                counts[cat] += 1
            avg_score = sum(p["score"] for _, _, p in entries) / n

            pct_perfect = 100 * counts.get("perfect", 0) / n
            pct_noprefix = 100 * counts.get("prefix_missing", 0) / n
            pct_hi = 100 * counts.get("high_partial", 0) / n
            pct_lo = 100 * (counts.get("low_partial", 0) + counts.get("near_zero_with_prefix", 0)) / n

            print(f"{cond_name:<20} {bin_label:<12} {pct_perfect:>8.1f}% {pct_noprefix:>7.1f}% {pct_hi:>7.1f}% {pct_lo:>7.1f}% {avg_score:>9.3f}")
        print()

    # ── Table 3: Score distribution deep dive for failed samples ──────
    print("\n" + "=" * 90)
    print("TABLE 3: RESPONSE PREVIEWS FOR LOW-SCORING SAMPLES (score < 0.5)")
    print("=" * 90)

    for cond_name in CONDITIONS.values():
        low_scores = []
        for _, bin_label in BINS:
            for cat, detail, pred in results[cond_name][bin_label]:
                if pred["score"] < 0.5 and pred["score"] > 0:
                    low_scores.append((bin_label, cat, pred))

        if not low_scores:
            continue

        print(f"\n── {cond_name} ({len(low_scores)} low-scoring samples) ──")
        # Show up to 5 examples per condition
        for bin_label, cat, pred in low_scores[:8]:
            resp_preview = pred["response_preview"][:120].replace('\n', '\\n')
            ans_preview = pred["answer_preview"][:120].replace('\n', '\\n')
            print(f"  [{bin_label}] score={pred['score']:.3f} cat={cat} tokens={pred['generated_tokens']}")
            print(f"    RESP: {resp_preview}")
            print(f"    ANS:  {ans_preview}")
            print()

    # ── Table 4: Generated token length analysis ─────────────────────
    print("\n" + "=" * 90)
    print("TABLE 4: GENERATED TOKEN LENGTH BY CATEGORY")
    print("=" * 90)

    for cond_name in CONDITIONS.values():
        cat_tokens = defaultdict(list)
        for _, bin_label in BINS:
            for cat, _, pred in results[cond_name][bin_label]:
                cat_tokens[cat].append(pred["generated_tokens"])

        print(f"\n── {cond_name} ──")
        for cat in categories:
            tokens = cat_tokens.get(cat, [])
            if tokens:
                avg_t = sum(tokens) / len(tokens)
                min_t = min(tokens)
                max_t = max(tokens)
                print(f"  {cat_short.get(cat, cat):<18} n={len(tokens):>3}  avg_tokens={avg_t:>6.0f}  min={min_t:>5}  max={max_t:>5}")


if __name__ == "__main__":
    analyze_all()
