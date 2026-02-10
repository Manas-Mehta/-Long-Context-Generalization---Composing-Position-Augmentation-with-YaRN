#!/usr/bin/env python
"""Generate comparison plots for RPE vs Baseline on CCoT reverse_string.

Reads eval_results.json files from baseline and RPE runs, produces:
1. Length vs Accuracy curve with training boundary
2. Bar chart: In-dist vs OOD accuracy
3. Summary comparison table (printed to console)

Usage:
    python composable_cot/scripts/plot_results.py \
        --baseline-results outputs/reverse_string_baseline_eval/eval_results.json \
        --rpe-results outputs/reverse_string_rpe_eval/eval_results.json \
        --output-dir outputs/comparison_plots
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> dict:
    """Load eval_results.json."""
    with open(path) as f:
        return json.load(f)


def plot_length_vs_accuracy(
    baseline: dict,
    rpe: dict,
    train_max_length: int,
    output_path: str,
):
    """Plot per-length accuracy for baseline vs RPE."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Extract per-length data
    for label, data, color, marker in [
        ("Baseline (no RPE)", baseline, "#E07020", "o"),
        ("RPE (L=8192)", rpe, "#2070E0", "s"),
    ]:
        lengths = sorted(int(k) for k in data["per_length"].keys())
        accs = [data["per_length"][str(l)]["accuracy"] for l in lengths]
        ax.plot(lengths, accs, f"-{marker}", color=color, label=label,
                markersize=4, linewidth=1.5, alpha=0.8)

    # Training boundary
    ax.axvline(x=train_max_length, color="gray", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(train_max_length + 1, 0.95, "OOD boundary",
            fontsize=9, color="gray", ha="left", va="top")

    # Shade regions
    ax.axvspan(0, train_max_length, alpha=0.05, color="green", label="_nolegend_")
    ax.axvspan(train_max_length, max(int(k) for k in baseline["per_length"].keys()) + 5,
               alpha=0.05, color="red", label="_nolegend_")

    ax.set_xlabel("String Length", fontsize=12)
    ax.set_ylabel("Per-Token Accuracy", fontsize=12)
    ax.set_title("RPE vs Baseline: Length Generalization on Reverse String (CCoT + Qwen2.5-7B)",
                 fontsize=13, pad=15)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(int(k) for k in baseline["per_length"].keys()) + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_bar_comparison(
    baseline: dict,
    rpe: dict,
    train_max_length: int,
    output_path: str,
):
    """Bar chart comparing in-dist and OOD accuracy."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    categories = ["In-Distribution\n(lengths 1-40)", "Out-of-Distribution\n(lengths 41-100)"]
    baseline_vals = [baseline["in_dist_accuracy"], baseline["ood_accuracy"]]
    rpe_vals = [rpe["in_dist_accuracy"], rpe["ood_accuracy"]]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (no RPE)",
                   color="#E07020", alpha=0.8)
    bars2 = ax.bar(x + width/2, rpe_vals, width, label="RPE (L=8192)",
                   color="#2070E0", alpha=0.8)

    # Value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("RPE vs Baseline: In-Distribution vs OOD Accuracy", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def print_comparison_table(baseline: dict, rpe: dict):
    """Print a formatted comparison table to console."""
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"  {'Metric':<30} {'Baseline':>10} {'RPE':>10} {'Delta':>10}")
    print(f"  {'-'*60}")

    metrics = [
        ("Overall Accuracy", baseline["overall_accuracy"], rpe["overall_accuracy"]),
        ("In-Distribution Accuracy", baseline["in_dist_accuracy"], rpe["in_dist_accuracy"]),
        ("OOD Accuracy", baseline["ood_accuracy"], rpe["ood_accuracy"]),
        ("DeepMind Score", baseline["dm_score"], rpe["dm_score"]),
    ]

    for name, b_val, r_val in metrics:
        delta = r_val - b_val
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<30} {b_val:>10.4f} {r_val:>10.4f} {sign}{delta:>9.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot RPE vs Baseline comparison")
    parser.add_argument("--baseline-results", type=str, required=True,
                        help="Path to baseline eval_results.json")
    parser.add_argument("--rpe-results", type=str, required=True,
                        help="Path to RPE eval_results.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save plots")
    parser.add_argument("--train-max-length", type=int, default=40,
                        help="Max training length (for boundary line)")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)

    baseline = load_results(args.baseline_results)
    rpe = load_results(args.rpe_results)

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot 1: Length vs Accuracy curve
    print("\nPlot 1: Length vs Accuracy")
    plot_length_vs_accuracy(
        baseline, rpe,
        train_max_length=args.train_max_length,
        output_path=os.path.join(args.output_dir, "length_vs_accuracy.png"),
    )

    # Plot 2: Bar comparison
    print("Plot 2: Bar Comparison")
    plot_bar_comparison(
        baseline, rpe,
        train_max_length=args.train_max_length,
        output_path=os.path.join(args.output_dir, "bar_comparison.png"),
    )

    # Print comparison table
    print_comparison_table(baseline, rpe)

    # Save summary JSON
    summary = {
        "baseline": {
            "overall_accuracy": baseline["overall_accuracy"],
            "in_dist_accuracy": baseline["in_dist_accuracy"],
            "ood_accuracy": baseline["ood_accuracy"],
            "dm_score": baseline["dm_score"],
        },
        "rpe": {
            "overall_accuracy": rpe["overall_accuracy"],
            "in_dist_accuracy": rpe["in_dist_accuracy"],
            "ood_accuracy": rpe["ood_accuracy"],
            "dm_score": rpe["dm_score"],
        },
        "delta": {
            "overall_accuracy": rpe["overall_accuracy"] - baseline["overall_accuracy"],
            "in_dist_accuracy": rpe["in_dist_accuracy"] - baseline["in_dist_accuracy"],
            "ood_accuracy": rpe["ood_accuracy"] - baseline["ood_accuracy"],
            "dm_score": rpe["dm_score"] - baseline["dm_score"],
        },
    }

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")

    print(f"\nAll plots saved to {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
