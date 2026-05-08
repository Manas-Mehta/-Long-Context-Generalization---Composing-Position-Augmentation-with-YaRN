#!/usr/bin/env python3
"""Visualize MRCR prediction data across conditions and bins.

Generates 7 publication-quality plots analyzing failure patterns,
position effects, and context extension behavior.

Usage:
    python composable_cot/mrcr_context_extension/scripts/visualize_predictions.py

Outputs saved to: composable_cot/mrcr_context_extension/analysis/
"""

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Try seaborn for nicer defaults, fall back gracefully
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use("seaborn-v0_8-whitegrid")

# ── Configuration ─────────────────────────────────────────────────────

BASE_DIR = Path("composable_cot/mrcr_context_extension")
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis"

# Conditions we have predictions for (dir prefix -> display name + color)
CONDITIONS = {
    "lora_baseline": ("LoRA baseline", "#2196F3"),
    "rpe_curriculum_lora_L16k": ("RPE cur L=16K", "#E91E63"),
    "rpe_lora_yarn_eval": ("RPE fixed+YaRN", "#4CAF50"),
    "pose_lora": ("PoSE fixed", "#FF9800"),
    "yarn_lora": ("YaRN+LoRA", "#9C27B0"),
}

BINS = [
    ("bin0_4K-8K", "4K-8K"),
    ("bin1_8K-16K", "8K-16K"),
    ("bin2_16K-32K", "16K-32K"),
    ("bin3_32K-64K", "32K-64K"),
    ("bin4_64K-128K", "64K-128K"),
]

# Error categorization thresholds (same as analyze_errors.py)
PERFECT_THRESHOLD = 0.95
HIGH_PARTIAL_THRESHOLD = 0.5
LOW_PARTIAL_THRESHOLD = 0.15

# Plot settings (compact for presentation — fit on one screen)
FIGSIZE_WIDE = (9, 4)
FIGSIZE_TALL = (9, 6)
FIGSIZE_SQUARE = (7, 5)
DPI = 150


# ── Data Loading ──────────────────────────────────────────────────────

def load_predictions(cond_prefix, bin_suffix):
    """Load predictions.json for a condition+bin combo."""
    dirname = f"{cond_prefix}_{bin_suffix}"
    path = OUTPUTS_DIR / dirname / "predictions.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_test_data(bin_suffix):
    """Load test.json for a bin to get needle position metadata."""
    path = DATA_DIR / bin_suffix / "test.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_needle_positions(sample):
    """Extract target and distractor needle positions from a test sample.

    Returns dict with:
        target_pos: relative position (0-1) of the target needle
        distractor_pos: relative position (0-1) of the distractor
        target_msg_idx: message index of the target
        distractor_msg_idx: message index of the distractor
        n_messages: total number of messages
    """
    messages = json.loads(sample["prompt"])
    last_msg = messages[-1]["content"]
    random_str = sample["random_string_to_prepend"]
    needle_content = sample["answer"][len(random_str):]

    # Extract topic from the retrieval question
    topic_match = re.search(r"(?:st|nd|rd|th) \(1 indexed\) (.+?)\. Do not", last_msg)
    topic = topic_match.group(1).lower() if topic_match else ""

    # Extract requested needle index (1-indexed)
    idx_match = re.search(r"the (\d+)(?:st|nd|rd|th)", last_msg)
    requested_idx = int(idx_match.group(1)) if idx_match else 1

    # Compute character offsets for each message
    char_offsets = []
    offset = 0
    for m in messages:
        char_offsets.append(offset)
        offset += len(m["content"])
    total_chars = offset

    # Find user messages requesting this topic (excluding the retrieval question)
    needle_infos = []
    for i, m in enumerate(messages[:-1]):
        if (m["role"] == "user"
                and topic in m["content"].lower()
                and "Prepend" not in m["content"]
                and i + 1 < len(messages)
                and messages[i + 1]["role"] == "assistant"):
            rel_pos = char_offsets[i + 1] / total_chars if total_chars > 0 else 0
            needle_infos.append({
                "msg_idx": i + 1,
                "relative_pos": rel_pos,
            })

    result = {
        "n_messages": len(messages),
        "target_pos": None,
        "distractor_pos": None,
        "target_msg_idx": None,
        "distractor_msg_idx": None,
        "requested_idx": requested_idx,
    }

    if len(needle_infos) >= 2:
        target = needle_infos[requested_idx - 1]
        distractor_idx = 1 - (requested_idx - 1)  # the other one
        distractor = needle_infos[distractor_idx]
        result["target_pos"] = target["relative_pos"]
        result["distractor_pos"] = distractor["relative_pos"]
        result["target_msg_idx"] = target["msg_idx"]
        result["distractor_msg_idx"] = distractor["msg_idx"]
    elif len(needle_infos) == 1:
        # Only found one needle (topic matching may have missed one)
        result["target_pos"] = needle_infos[0]["relative_pos"]
        result["target_msg_idx"] = needle_infos[0]["msg_idx"]

    return result


def categorize_error(pred):
    """Categorize a prediction into an error type."""
    score = pred["score"]
    response = pred["response_preview"]
    answer = pred["answer_preview"]
    generated_tokens = pred.get("generated_tokens", 0)

    # Infer random prefix from answer
    random_prefix = ""
    for i, c in enumerate(answer):
        if not c.isalnum():
            random_prefix = answer[:i]
            break
    if not random_prefix:
        random_prefix = answer[:10]

    if generated_tokens <= 3 or len(response.strip()) == 0:
        return "Empty"
    if score >= PERFECT_THRESHOLD:
        return "Perfect"
    if score == 0.0:
        if response.strip().startswith(random_prefix):
            return "Pfx+zero"
        return "No prefix"
    has_prefix = response.startswith(random_prefix) if random_prefix else True
    if not has_prefix:
        return "Wrong pfx"
    if score >= HIGH_PARTIAL_THRESHOLD:
        return "High partial"
    if score >= LOW_PARTIAL_THRESHOLD:
        return "Low partial"
    return "Near-zero"


def load_all_data():
    """Load all predictions and test data, enrich with needle positions.

    Returns list of dicts, one per prediction, with all fields needed for plotting.
    """
    all_data = []

    # Pre-load test data and extract positions per bin
    test_positions = {}  # (bin_suffix, index) -> position_info
    for bin_suffix, bin_label in BINS:
        test_data = load_test_data(bin_suffix)
        if test_data is None:
            continue
        for i, sample in enumerate(test_data):
            pos_info = extract_needle_positions(sample)
            test_positions[(bin_suffix, i)] = pos_info

    for cond_prefix, (cond_name, color) in CONDITIONS.items():
        for bin_suffix, bin_label in BINS:
            preds = load_predictions(cond_prefix, bin_suffix)
            if preds is None:
                print(f"  WARNING: Missing {cond_prefix}_{bin_suffix}")
                continue
            for pred in preds:
                idx = pred["index"]
                pos_info = test_positions.get((bin_suffix, idx), {})

                row = {
                    "condition": cond_name,
                    "color": color,
                    "bin_label": bin_label,
                    "bin_suffix": bin_suffix,
                    "bin_index": BINS.index((bin_suffix, bin_label)),
                    "index": idx,
                    "score": pred["score"],
                    "token_count": pred["token_count"],
                    "generated_tokens": pred.get("generated_tokens", 0),
                    "gen_time_s": pred.get("gen_time_s", 0),
                    "error_type": categorize_error(pred),
                    "target_pos": pos_info.get("target_pos"),
                    "distractor_pos": pos_info.get("distractor_pos"),
                    "n_messages": pos_info.get("n_messages"),
                    "requested_idx": pos_info.get("requested_idx"),
                }
                all_data.append(row)

    print(f"Loaded {len(all_data)} predictions across "
          f"{len(CONDITIONS)} conditions × {len(BINS)} bins")
    return all_data


# ── Plotting Functions ────────────────────────────────────────────────

def _get_condition_order():
    """Return condition names in display order."""
    return [name for _, (name, _) in CONDITIONS.items()]


def _get_color_map():
    """Return condition name -> color mapping."""
    return {name: color for _, (name, color) in CONDITIONS.items()}


def _savefig(fig, name):
    """Save figure to analysis directory."""
    path = ANALYSIS_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def moving_average(x, y, window=5):
    """Compute moving average for scatter trendline."""
    sorted_indices = np.argsort(x)
    x_sorted = np.array(x)[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]

    if len(x_sorted) < window:
        return x_sorted, y_sorted

    # Pad edges
    y_smooth = np.convolve(y_sorted, np.ones(window) / window, mode="valid")
    x_smooth = x_sorted[(window - 1) // 2: (window - 1) // 2 + len(y_smooth)]
    return x_smooth, y_smooth


# ── Plot 1: Score Distribution (Violin/Box) ──────────────────────────

def plot_score_distributions(data):
    """Violin/box plots of score distribution per condition × bin."""
    print("Plot 1: Score distributions...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()
    bin_labels = [bl for _, bl in BINS]

    fig, axes = plt.subplots(1, len(BINS), figsize=(12, 3.5), sharey=True)
    fig.suptitle("Score Distribution by Condition × Context Length Bin",
                 fontsize=14, fontweight="bold", y=1.02)

    for bi, bin_label in enumerate(bin_labels):
        ax = axes[bi]
        bin_data = [r for r in data if r["bin_label"] == bin_label]

        positions = []
        box_data = []
        colors = []
        labels = []

        for ci, cond in enumerate(cond_order):
            scores = [r["score"] for r in bin_data if r["condition"] == cond]
            if scores:
                positions.append(ci)
                box_data.append(scores)
                colors.append(color_map[cond])
                labels.append(cond)

        if box_data:
            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showfliers=True,
                           flierprops=dict(marker="o", markersize=4, alpha=0.5))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_title(bin_label, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels([c.split()[0] for c in cond_order],
                          rotation=45, ha="right", fontsize=8)
        if bi == 0:
            ax.set_ylabel("Score")
        ax.axhline(y=PERFECT_THRESHOLD, color="gray", linestyle="--",
                   alpha=0.3, linewidth=0.8)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=color_map[c], alpha=0.6, label=c)
                      for c in cond_order]
    fig.legend(handles=legend_patches, loc="upper center",
              ncol=len(cond_order), bbox_to_anchor=(0.5, 0.98), fontsize=9)

    fig.tight_layout()
    _savefig(fig, "01_score_distributions")


# ── Plot 2: Score vs Token Count (Scatter) ───────────────────────────

def plot_score_vs_tokens(data):
    """Scatter plot of score vs exact token count, colored by condition."""
    print("Plot 2: Score vs token count...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for cond in cond_order:
        cond_data = [r for r in data if r["condition"] == cond]
        tokens = [r["token_count"] for r in cond_data]
        scores = [r["score"] for r in cond_data]
        color = color_map[cond]

        # Scatter
        ax.scatter(tokens, scores, c=color, alpha=0.35, s=30,
                  edgecolors="none", label=cond)

        # Trendline (moving average with larger window for smoothness)
        if len(tokens) >= 8:
            x_smooth, y_smooth = moving_average(tokens, scores, window=7)
            ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.85)

    ax.set_xlabel("Token Count (prompt)")
    ax.set_ylabel("Score")
    ax.set_title("Score vs. Context Length (Exact Token Count)",
                fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x / 1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)

    # Add bin boundaries as vertical lines
    bin_bounds = [4096, 8192, 16384, 32768, 65536, 131072]
    for b in bin_bounds:
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.15, linewidth=0.8)

    fig.tight_layout()
    _savefig(fig, "02_score_vs_tokens")


# ── Plot 3: Error Type Stacked Bars ──────────────────────────────────

def plot_error_types(data):
    """Stacked bar chart of error categories per condition × bin."""
    print("Plot 3: Error type stacked bars...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()
    bin_labels = [bl for _, bl in BINS]

    error_cats = ["Perfect", "High partial", "Low partial", "Near-zero",
                  "No prefix", "Empty", "Pfx+zero", "Wrong pfx"]
    error_colors = {
        "Perfect": "#4CAF50",
        "High partial": "#8BC34A",
        "Low partial": "#FFC107",
        "Near-zero": "#FF9800",
        "No prefix": "#F44336",
        "Empty": "#9E9E9E",
        "Pfx+zero": "#795548",
        "Wrong pfx": "#607D8B",
    }

    fig, axes = plt.subplots(len(cond_order), 1, figsize=(9, 1.8 * len(cond_order)),
                             sharex=True)
    fig.suptitle("Error Type Distribution by Condition × Bin",
                fontsize=14, fontweight="bold", y=1.01)

    for ci, cond in enumerate(cond_order):
        ax = axes[ci]
        cond_data = [r for r in data if r["condition"] == cond]

        x = np.arange(len(bin_labels))
        bottoms = np.zeros(len(bin_labels))

        for ecat in error_cats:
            heights = []
            for bi, bl in enumerate(bin_labels):
                bin_preds = [r for r in cond_data if r["bin_label"] == bl]
                n = len(bin_preds)
                count = sum(1 for r in bin_preds if r["error_type"] == ecat)
                pct = 100 * count / n if n > 0 else 0
                heights.append(pct)
            heights = np.array(heights)
            if np.any(heights > 0):
                ax.bar(x, heights, bottom=bottoms, color=error_colors[ecat],
                      label=ecat, width=0.65, edgecolor="white", linewidth=0.5)
                bottoms += heights

        ax.set_ylabel("% samples")
        ax.set_ylim(0, 105)
        ax.set_title(cond, fontsize=11, fontweight="bold",
                    color=color_map[cond])
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)

        if ci == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=4,
                     framealpha=0.9, edgecolor="gray")

    axes[-1].set_xlabel("Context Length Bin")
    fig.tight_layout()
    _savefig(fig, "03_error_types_stacked")


# ── Plot 4: Score vs Needle Position ─────────────────────────────────

def plot_score_vs_position(data):
    """Score vs relative needle position (0=start, 1=end) with trendlines."""
    print("Plot 4: Score vs needle position...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()

    # Facet by bin
    fig, axes = plt.subplots(1, len(BINS), figsize=(12, 3.5), sharey=True)
    fig.suptitle("Score vs. Needle Position in Conversation",
                fontsize=14, fontweight="bold", y=1.03)

    for bi, (bin_suffix, bin_label) in enumerate(BINS):
        ax = axes[bi]

        for cond in cond_order:
            cond_bin = [r for r in data
                       if r["condition"] == cond
                       and r["bin_label"] == bin_label
                       and r["target_pos"] is not None]
            if not cond_bin:
                continue

            positions = [r["target_pos"] for r in cond_bin]
            scores = [r["score"] for r in cond_bin]
            color = color_map[cond]

            ax.scatter(positions, scores, c=color, alpha=0.4, s=35,
                      edgecolors="none")

            # Trendline
            if len(positions) >= 5:
                x_smooth, y_smooth = moving_average(positions, scores, window=5)
                ax.plot(x_smooth, y_smooth, color=color, linewidth=2, alpha=0.8)

        ax.set_title(bin_label, fontsize=11)
        ax.set_xlabel("Needle Position (0=start, 1=end)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        if bi == 0:
            ax.set_ylabel("Score")

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=color_map[c], linewidth=2, label=c)
               for c in cond_order]
    fig.legend(handles=handles, loc="upper center", ncol=len(cond_order),
              bbox_to_anchor=(0.5, 0.99), fontsize=9)

    fig.tight_layout()
    _savefig(fig, "04_score_vs_needle_position")

    # Also make a pooled (all bins) version with quintile summary
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.set_title("Score vs. Needle Position (All Bins Pooled) — Quintile Averages",
                 fontsize=13, fontweight="bold")

    quintile_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    quintile_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    x_quintile = np.arange(len(quintile_labels))

    bar_width = 0.15
    for ci, cond in enumerate(cond_order):
        cond_data = [r for r in data
                    if r["condition"] == cond and r["target_pos"] is not None]
        if not cond_data:
            continue

        quintile_scores = []
        for qi in range(len(quintile_labels)):
            lo, hi = quintile_edges[qi], quintile_edges[qi + 1]
            q_scores = [r["score"] for r in cond_data
                       if lo <= r["target_pos"] < (hi + 0.001)]
            avg = np.mean(q_scores) if q_scores else 0
            quintile_scores.append(avg)

        offset = (ci - len(cond_order) / 2 + 0.5) * bar_width
        bars = ax2.bar(x_quintile + offset, quintile_scores,
                      width=bar_width, color=color_map[cond],
                      label=cond, alpha=0.8, edgecolor="white")

        # Add value labels on bars
        for bar, val in zip(bars, quintile_scores):
            if val > 0.05:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax2.set_xticks(x_quintile)
    ax2.set_xticklabels(quintile_labels)
    ax2.set_xlabel("Needle Position (quintile)")
    ax2.set_ylabel("Average Score")
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=9)

    fig2.tight_layout()
    _savefig(fig2, "04b_needle_position_quintiles")


# ── Plot 5: Distractor vs Target Position (Primacy/Recency) ──────────

def plot_distractor_analysis(data):
    """Analyze whether failures correlate with target/distractor relative position."""
    print("Plot 5: Distractor vs target position...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()

    # For each failed prediction (score < 0.95), check:
    # - Is target before or after distractor?
    # - Is there a recency or primacy bias?

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Score vs (target_pos - distractor_pos)
    # Positive = target is AFTER distractor (recency would help)
    # Negative = target is BEFORE distractor (primacy would help)
    ax = axes[0]
    ax.set_title("Score vs. Target-Distractor Position Gap",
                fontsize=12, fontweight="bold")
    ax.set_xlabel("Position Gap (target - distractor)\n"
                 "← Target earlier | Target later →")
    ax.set_ylabel("Score")

    for cond in cond_order:
        cond_data = [r for r in data
                    if r["condition"] == cond
                    and r["target_pos"] is not None
                    and r["distractor_pos"] is not None]
        if not cond_data:
            continue
        gaps = [r["target_pos"] - r["distractor_pos"] for r in cond_data]
        scores = [r["score"] for r in cond_data]
        ax.scatter(gaps, scores, c=color_map[cond], alpha=0.35, s=30,
                  edgecolors="none", label=cond)

    ax.axvline(x=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower left")

    # Panel B: Success rate when target is earlier vs later than distractor
    ax2 = axes[1]
    ax2.set_title("% Perfect by Target Position Relative to Distractor",
                 fontsize=12, fontweight="bold")

    bar_data = {"Target Earlier\n(must skip recency)": [], "Target Later\n(recency helps)": []}
    bar_labels = list(bar_data.keys())

    x_pos = np.arange(len(cond_order))
    bar_width = 0.35

    earlier_pcts = []
    later_pcts = []

    for cond in cond_order:
        cond_data = [r for r in data
                    if r["condition"] == cond
                    and r["target_pos"] is not None
                    and r["distractor_pos"] is not None]

        earlier = [r for r in cond_data if r["target_pos"] < r["distractor_pos"]]
        later = [r for r in cond_data if r["target_pos"] >= r["distractor_pos"]]

        pct_e = 100 * sum(1 for r in earlier if r["score"] >= PERFECT_THRESHOLD) / len(earlier) if earlier else 0
        pct_l = 100 * sum(1 for r in later if r["score"] >= PERFECT_THRESHOLD) / len(later) if later else 0

        earlier_pcts.append(pct_e)
        later_pcts.append(pct_l)

    bars1 = ax2.bar(x_pos - bar_width / 2, earlier_pcts, bar_width,
                   label="Target earlier (must skip recency)",
                   color="#E57373", alpha=0.8)
    bars2 = ax2.bar(x_pos + bar_width / 2, later_pcts, bar_width,
                   label="Target later (recency helps)",
                   color="#81C784", alpha=0.8)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.replace(" ", "\n") for c in cond_order], fontsize=9)
    ax2.set_ylabel("% Perfect (score ≥ 0.95)")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    _savefig(fig, "05_distractor_analysis")


# ── Plot 6: Per-Sample Cross-Condition Comparison ─────────────────────

def plot_cross_condition(data):
    """Compare per-sample scores across conditions — are hard samples universal?"""
    print("Plot 6: Cross-condition comparison...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()

    # Build score matrix: (bin, index) -> {condition: score}
    score_matrix = defaultdict(dict)
    for r in data:
        key = (r["bin_label"], r["index"])
        score_matrix[key][r["condition"]] = r["score"]

    # Panel A: Pairwise correlation heatmap
    # Build score vectors (only samples present in all conditions)
    all_keys = [k for k in score_matrix if len(score_matrix[k]) == len(cond_order)]
    if not all_keys:
        print("  Skipping: no samples with all conditions")
        return

    score_vectors = {}
    for cond in cond_order:
        score_vectors[cond] = [score_matrix[k][cond] for k in all_keys]

    corr_matrix = np.zeros((len(cond_order), len(cond_order)))
    for i, c1 in enumerate(cond_order):
        for j, c2 in enumerate(cond_order):
            corr_matrix[i, j] = np.corrcoef(score_vectors[c1], score_vectors[c2])[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cond_order)))
    ax.set_yticks(range(len(cond_order)))
    short_names = [c.split("(")[0].strip() for c in cond_order]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_title("Score Correlation Across Conditions\n(per-sample, all bins)",
                fontsize=12, fontweight="bold")

    # Annotate cells
    for i in range(len(cond_order)):
        for j in range(len(cond_order)):
            val = corr_matrix[i, j]
            text_color = "white" if val < 0.3 or val > 0.85 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   fontsize=10, color=text_color, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    # Panel B: Scatter — baseline vs best RPE condition
    ax2 = axes[1]
    baseline_scores = score_vectors[cond_order[0]]  # LoRA baseline
    rpe_scores = score_vectors[cond_order[1]]  # RPE cur L=16K

    # Color by bin
    bin_colors_map = {
        "4K-8K": "#1a9850",
        "8K-16K": "#91cf60",
        "16K-32K": "#fee08b",
        "32K-64K": "#fc8d59",
        "64K-128K": "#d73027",
    }
    sample_bins = [k[0] for k in all_keys]
    scatter_colors = [bin_colors_map.get(b, "gray") for b in sample_bins]

    ax2.scatter(baseline_scores, rpe_scores, c=scatter_colors, alpha=0.5, s=35,
               edgecolors="white", linewidth=0.3)
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax2.set_xlabel(f"{cond_order[0]} Score")
    ax2.set_ylabel(f"{cond_order[1]} Score")
    ax2.set_title(f"Per-Sample: {cond_order[0]} vs {cond_order[1]}",
                 fontsize=12, fontweight="bold")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    # Legend for bins
    from matplotlib.lines import Line2D
    bin_handles = [Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=bin_colors_map[bl], markersize=8, label=bl)
                  for _, bl in BINS]
    ax2.legend(handles=bin_handles, fontsize=8, title="Bin", title_fontsize=9)

    # Annotate quadrants
    ax2.text(0.25, 0.85, "RPE better", ha="center", fontsize=9,
            color="#E91E63", alpha=0.6, transform=ax2.transAxes)
    ax2.text(0.75, 0.15, "Baseline better", ha="center", fontsize=9,
            color="#2196F3", alpha=0.6, transform=ax2.transAxes)

    fig.tight_layout()
    _savefig(fig, "06_cross_condition_comparison")


# ── Plot 7: Normalized Degradation Curve ──────────────────────────────

def plot_normalized_degradation(data):
    """Score normalized by bin-0 performance — isolates extension ability."""
    print("Plot 7: Normalized degradation curve...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()
    bin_labels = [bl for _, bl in BINS]

    # Compute mean score per condition × bin
    mean_scores = {}
    for cond in cond_order:
        for bl in bin_labels:
            scores = [r["score"] for r in data
                     if r["condition"] == cond and r["bin_label"] == bl]
            if scores:
                mean_scores[(cond, bl)] = np.mean(scores)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Raw mean scores (line plot)
    ax = axes[0]
    ax.set_title("Mean Score by Context Length Bin", fontsize=12, fontweight="bold")
    x = np.arange(len(bin_labels))

    for cond in cond_order:
        y = [mean_scores.get((cond, bl), np.nan) for bl in bin_labels]
        ax.plot(x, y, "o-", color=color_map[cond], linewidth=2.5,
               markersize=8, label=cond, alpha=0.85)
        # Annotate values
        for xi, yi in zip(x, y):
            if not np.isnan(yi):
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=7.5,
                           color=color_map[cond])

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Context Length Bin")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9, loc="upper right")

    # Panel B: Normalized (score / bin0 score)
    ax2 = axes[1]
    ax2.set_title("Normalized Score (relative to Bin 0)\n— Isolates Context Extension Ability —",
                 fontsize=12, fontweight="bold")

    for cond in cond_order:
        bin0_score = mean_scores.get((cond, bin_labels[0]), None)
        if bin0_score is None or bin0_score == 0:
            continue
        y_norm = []
        for bl in bin_labels:
            raw = mean_scores.get((cond, bl), np.nan)
            y_norm.append(raw / bin0_score if not np.isnan(raw) else np.nan)

        ax2.plot(x, y_norm, "o-", color=color_map[cond], linewidth=2.5,
                markersize=8, label=cond, alpha=0.85)
        for xi, yi in zip(x, y_norm):
            if not np.isnan(yi):
                ax2.annotate(f"{yi:.0%}", (xi, yi), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=7.5,
                           color=color_map[cond])

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, fontsize=9)
    ax2.set_xlabel("Context Length Bin")
    ax2.set_ylabel("Score / Bin-0 Score")
    ax2.set_ylim(-0.05, 1.5)
    ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    _savefig(fig, "07_normalized_degradation")


# ── Plot 4c: Position Quintile Grid (per bin) ─────────────────────────

def plot_position_quintile_grid(data):
    """Quintile bar chart split by bin — 5×1 grid showing position effect per context length."""
    print("Plot 4c: Position quintile grid by bin...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()

    quintile_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    quintile_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

    fig, axes = plt.subplots(len(BINS), 1, figsize=(9, 2 * len(BINS)), sharex=True)
    fig.suptitle("Score by Needle Position Quintile — Split by Context Length",
                 fontsize=14, fontweight="bold", y=1.01)

    bar_width = 0.14
    x_quintile = np.arange(len(quintile_labels))

    for bi, (bin_suffix, bin_label) in enumerate(BINS):
        ax = axes[bi]

        for ci, cond in enumerate(cond_order):
            cond_bin_data = [r for r in data
                           if r["condition"] == cond
                           and r["bin_label"] == bin_label
                           and r["target_pos"] is not None]
            if not cond_bin_data:
                continue

            quintile_scores = []
            quintile_ns = []
            for qi in range(len(quintile_labels)):
                lo, hi = quintile_edges[qi], quintile_edges[qi + 1]
                q_scores = [r["score"] for r in cond_bin_data
                           if lo <= r["target_pos"] < (hi + 0.001)]
                avg = np.mean(q_scores) if q_scores else 0
                quintile_scores.append(avg)
                quintile_ns.append(len(q_scores))

            offset = (ci - len(cond_order) / 2 + 0.5) * bar_width
            bars = ax.bar(x_quintile + offset, quintile_scores,
                         width=bar_width, color=color_map[cond],
                         label=cond if bi == 0 else None, alpha=0.8,
                         edgecolor="white")

            # Value labels with n count
            for bar, val, n in zip(bars, quintile_scores, quintile_ns):
                if n > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + 0.01,
                           f"{val:.2f}\nn={n}", ha="center", va="bottom",
                           fontsize=5.5, color=color_map[cond])

        ax.set_ylabel("Avg Score")
        ax.set_ylim(0, 1.25)
        ax.set_title(f"{bin_label}", fontsize=11, fontweight="bold")
        ax.axhline(y=PERFECT_THRESHOLD, color="gray", linestyle="--",
                   alpha=0.3, linewidth=0.8)

    axes[-1].set_xticks(x_quintile)
    axes[-1].set_xticklabels(quintile_labels)
    axes[-1].set_xlabel("Needle Position (quintile)")
    axes[0].legend(fontsize=8, ncol=len(cond_order), loc="upper left")

    fig.tight_layout()
    _savefig(fig, "04c_needle_position_grid")


# ── Plot 8: Per-Sample Difficulty Analysis ─────────────────────────────

def plot_per_sample_difficulty(data):
    """Analyze per-sample difficulty: hardest, easiest, divisive samples."""
    print("Plot 8: Per-sample difficulty analysis...")

    cond_order = _get_condition_order()
    color_map = _get_color_map()

    # Build score matrix: (bin_label, index) -> {condition: score}
    score_matrix = defaultdict(dict)
    for r in data:
        key = (r["bin_label"], r["index"])
        score_matrix[key][r["condition"]] = r["score"]

    # Only samples with all conditions
    full_keys = sorted([k for k in score_matrix if len(score_matrix[k]) == len(cond_order)])

    # Compute difficulty metrics
    sample_stats = []
    for key in full_keys:
        scores = score_matrix[key]
        vals = [scores[c] for c in cond_order]
        sample_stats.append({
            "key": key,
            "bin": key[0],
            "idx": key[1],
            "mean": np.mean(vals),
            "std": np.std(vals),
            "min": np.min(vals),
            "max": np.max(vals),
            "n_perfect": sum(1 for v in vals if v >= PERFECT_THRESHOLD),
            "n_fail": sum(1 for v in vals if v < 0.5),
            "scores": scores,
        })

    # Sort by difficulty
    by_hardest = sorted(sample_stats, key=lambda x: x["mean"])
    by_divisive = sorted(sample_stats, key=lambda x: x["std"], reverse=True)

    # ── Panel A: Heatmap of hardest + easiest + most divisive ──
    # Select 10 hardest, 5 easiest, 10 most divisive (deduped)
    selected = []
    selected_keys = set()

    for s in by_hardest[:10]:
        if s["key"] not in selected_keys:
            selected.append(("HARD", s))
            selected_keys.add(s["key"])

    for s in by_divisive[:10]:
        if s["key"] not in selected_keys:
            selected.append(("DIVISIVE", s))
            selected_keys.add(s["key"])

    for s in reversed(by_hardest):
        if len([x for x in selected if x[0] == "EASY"]) >= 5:
            break
        if s["key"] not in selected_keys:
            selected.append(("EASY", s))
            selected_keys.add(s["key"])

    # Build heatmap data
    n_samples = len(selected)
    n_conds = len(cond_order)
    heatmap = np.zeros((n_samples, n_conds))
    y_labels = []

    for si, (category, s) in enumerate(selected):
        for ci, cond in enumerate(cond_order):
            heatmap[si, ci] = s["scores"][cond]
        y_labels.append(f"[{category}] {s['bin']} #{s['idx']} (μ={s['mean']:.2f})")

    fig, ax = plt.subplots(figsize=(9, max(6, n_samples * 0.28)))
    im = ax.imshow(heatmap, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n_conds))
    ax.set_xticklabels(cond_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title("Per-Sample Difficulty: Hardest, Most Divisive, Easiest",
                fontsize=13, fontweight="bold")

    # Annotate cells
    for si in range(n_samples):
        for ci in range(n_conds):
            val = heatmap[si, ci]
            text_color = "white" if val < 0.3 or val > 0.85 else "black"
            ax.text(ci, si, f"{val:.2f}", ha="center", va="center",
                   fontsize=8, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.6, label="Score")
    fig.tight_layout()
    _savefig(fig, "08_per_sample_difficulty")

    # ── Panel B: Samples solved by exactly 1 condition ──
    unique_solvers = []
    for s in sample_stats:
        perfect_conds = [c for c in cond_order if s["scores"][c] >= PERFECT_THRESHOLD]
        if len(perfect_conds) == 1:
            unique_solvers.append({
                **s,
                "solver": perfect_conds[0],
            })

    if unique_solvers:
        # Group by solver
        solver_groups = defaultdict(list)
        for u in unique_solvers:
            solver_groups[u["solver"]].append(u)

        n_unique = len(unique_solvers)
        fig2, ax2 = plt.subplots(figsize=(9, max(5, n_unique * 0.25)))

        heatmap2 = np.zeros((n_unique, n_conds))
        y_labels2 = []
        row = 0
        for cond in cond_order:
            for u in solver_groups.get(cond, []):
                for ci, c in enumerate(cond_order):
                    heatmap2[row, ci] = u["scores"][c]
                y_labels2.append(f"{u['bin']} #{u['idx']}")
                row += 1

        im2 = ax2.imshow(heatmap2[:row], cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks(range(n_conds))
        ax2.set_xticklabels(cond_order, rotation=45, ha="right", fontsize=9)
        ax2.set_yticks(range(row))
        ax2.set_yticklabels(y_labels2[:row], fontsize=8)
        ax2.set_title("Samples Solved by Exactly 1 Condition (score ≥ 0.95)",
                     fontsize=13, fontweight="bold")

        for si in range(row):
            for ci in range(n_conds):
                val = heatmap2[si, ci]
                text_color = "white" if val < 0.3 or val > 0.85 else "black"
                ax2.text(ci, si, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

        # Add solver group labels on right
        y_pos = 0
        for cond in cond_order:
            n = len(solver_groups.get(cond, []))
            if n > 0:
                mid = y_pos + n / 2 - 0.5
                ax2.text(n_conds + 0.3, mid, f"← {cond} ({n})",
                        ha="left", va="center", fontsize=8,
                        color=color_map[cond], fontweight="bold")
                y_pos += n

        fig2.colorbar(im2, ax=ax2, shrink=0.6, label="Score")
        fig2.tight_layout()
        _savefig(fig2, "08b_unique_solvers")

    # ── Panel C: Full sample heatmap — ALL samples sorted by difficulty ──
    # Sort all samples by mean score (hardest at top)
    sorted_stats = sorted(sample_stats, key=lambda x: x["mean"])

    n_all = len(sorted_stats)
    heatmap_full = np.zeros((n_all, n_conds))
    y_labels_full = []
    solve_counts_strip = []

    for si, s in enumerate(sorted_stats):
        for ci, cond in enumerate(cond_order):
            heatmap_full[si, ci] = s["scores"][cond]
        y_labels_full.append(f"{s['bin']} #{s['idx']}")
        solve_counts_strip.append(s["n_perfect"])

    fig3, axes3 = plt.subplots(1, 2, figsize=(9, max(10, n_all * 0.09)),
                                gridspec_kw={"width_ratios": [5, 1], "wspace": 0.05})

    # Main heatmap
    ax_main = axes3[0]
    im3 = ax_main.imshow(heatmap_full, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_main.set_xticks(range(n_conds))
    ax_main.set_xticklabels(cond_order, rotation=45, ha="right", fontsize=8)
    ax_main.set_yticks(range(n_all))
    ax_main.set_yticklabels(y_labels_full, fontsize=4)
    ax_main.set_title("All Samples Sorted by Difficulty (hardest → easiest)",
                      fontsize=11, fontweight="bold")

    # Right strip: # conditions that solved it
    ax_strip = axes3[1]
    strip_data = np.array(solve_counts_strip).reshape(-1, 1)
    ax_strip.imshow(strip_data, cmap="Blues", vmin=0, vmax=5, aspect="auto")
    ax_strip.set_xticks([0])
    ax_strip.set_xticklabels(["#Solved"], fontsize=8)
    ax_strip.set_yticks([])
    for si in range(n_all):
        ax_strip.text(0, si, str(solve_counts_strip[si]), ha="center", va="center",
                     fontsize=4, color="black" if solve_counts_strip[si] < 4 else "white")

    fig3.colorbar(im3, ax=axes3, shrink=0.3, label="Score", pad=0.02)
    fig3.tight_layout()
    _savefig(fig3, "09_full_sample_heatmap")

    # ── Panel D: Per-condition deviation — who uniquely fails/succeeds? ──
    # For each sample, compute: condition_score - mean_of_other_conditions
    deviation = np.zeros((n_all, n_conds))
    for si, s in enumerate(sorted_stats):
        vals = [s["scores"][c] for c in cond_order]
        for ci, cond in enumerate(cond_order):
            others = [v for j, v in enumerate(vals) if j != ci]
            deviation[si, ci] = s["scores"][cond] - np.mean(others)

    fig4, ax4 = plt.subplots(figsize=(9, max(10, n_all * 0.09)))
    im4 = ax4.imshow(deviation, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax4.set_xticks(range(n_conds))
    ax4.set_xticklabels(cond_order, rotation=45, ha="right", fontsize=8)
    ax4.set_yticks(range(n_all))
    ax4.set_yticklabels(y_labels_full, fontsize=4)
    ax4.set_title("Per-Sample Deviation from Mean of Others\n"
                  "Blue = this condition outperforms | Red = this condition underperforms",
                  fontsize=11, fontweight="bold")

    fig4.colorbar(im4, ax=ax4, shrink=0.3, label="Score − Mean(others)")
    fig4.tight_layout()
    _savefig(fig4, "09b_condition_deviation")

    # ── Write detailed text report ──
    report_path = ANALYSIS_DIR / "per_sample_difficulty.txt"
    with open(report_path, "w") as f:
        f.write("Per-Sample Difficulty Analysis\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total samples with all {len(cond_order)} conditions: {len(full_keys)}\n\n")

        f.write("TOP 10 HARDEST SAMPLES (lowest mean score):\n")
        f.write("-" * 70 + "\n")
        for i, s in enumerate(by_hardest[:10]):
            f.write(f"  #{i+1}: {s['bin']} sample {s['idx']}  "
                   f"mean={s['mean']:.3f}  std={s['std']:.3f}\n")
            for c in cond_order:
                f.write(f"      {c:<22} {s['scores'][c]:.3f}\n")
            f.write("\n")

        f.write("\nTOP 10 MOST DIVISIVE SAMPLES (highest std):\n")
        f.write("-" * 70 + "\n")
        for i, s in enumerate(by_divisive[:10]):
            f.write(f"  #{i+1}: {s['bin']} sample {s['idx']}  "
                   f"mean={s['mean']:.3f}  std={s['std']:.3f}\n")
            for c in cond_order:
                perfect = "✓" if s['scores'][c] >= PERFECT_THRESHOLD else " "
                f.write(f"      {perfect} {c:<22} {s['scores'][c]:.3f}\n")
            f.write("\n")

        f.write(f"\nSAMPLES SOLVED BY EXACTLY 1 CONDITION ({len(unique_solvers)} total):\n")
        f.write("-" * 70 + "\n")
        for cond in cond_order:
            items = solver_groups.get(cond, [])
            if items:
                f.write(f"\n  {cond} uniquely solves {len(items)} samples:\n")
                for u in items:
                    f.write(f"    {u['bin']} #{u['idx']}: ", )
                    for c in cond_order:
                        marker = "★" if c == cond else " "
                        f.write(f"{marker}{u['scores'][c]:.2f} ")
                    f.write("\n")

        # Per-condition summary
        f.write("\n\nSUMMARY: SAMPLES BY SOLVE COUNT\n")
        f.write("-" * 70 + "\n")
        solve_counts = defaultdict(int)
        for s in sample_stats:
            solve_counts[s["n_perfect"]] += 1
        for n in range(len(cond_order) + 1):
            c = solve_counts.get(n, 0)
            f.write(f"  Solved by {n}/{len(cond_order)} conditions: {c} samples "
                   f"({100*c/len(sample_stats):.1f}%)\n")

    print(f"  Saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {ANALYSIS_DIR}\n")

    print("Loading data...")
    data = load_all_data()
    print()

    plot_score_distributions(data)
    plot_score_vs_tokens(data)
    plot_error_types(data)
    plot_score_vs_position(data)
    plot_position_quintile_grid(data)
    plot_distractor_analysis(data)
    plot_cross_condition(data)
    plot_normalized_degradation(data)
    plot_per_sample_difficulty(data)

    # ── Summary stats to text file ────────────────────────────────────
    summary_path = ANALYSIS_DIR / "summary_stats.txt"
    with open(summary_path, "w") as f:
        f.write("MRCR Prediction Analysis — Summary Statistics\n")
        f.write("=" * 60 + "\n\n")

        cond_order = _get_condition_order()
        bin_labels = [bl for _, bl in BINS]

        # Per condition × bin: mean, median, std, %perfect
        f.write("Mean Score per Condition × Bin:\n")
        f.write(f"{'Condition':<22}" + "".join(f" {bl:>10}" for bl in bin_labels) + "\n")
        f.write("-" * 74 + "\n")
        for cond in cond_order:
            row = f"{cond:<22}"
            for bl in bin_labels:
                scores = [r["score"] for r in data
                         if r["condition"] == cond and r["bin_label"] == bl]
                row += f" {np.mean(scores):>10.3f}" if scores else f" {'N/A':>10}"
            f.write(row + "\n")

        f.write(f"\n% Perfect (score ≥ {PERFECT_THRESHOLD}) per Condition × Bin:\n")
        f.write(f"{'Condition':<22}" + "".join(f" {bl:>10}" for bl in bin_labels) + "\n")
        f.write("-" * 74 + "\n")
        for cond in cond_order:
            row = f"{cond:<22}"
            for bl in bin_labels:
                scores = [r["score"] for r in data
                         if r["condition"] == cond and r["bin_label"] == bl]
                pct = 100 * sum(1 for s in scores if s >= PERFECT_THRESHOLD) / len(scores) if scores else 0
                row += f" {pct:>9.1f}%" if scores else f" {'N/A':>10}"
            f.write(row + "\n")

        # Needle position stats
        f.write("\n\nNeedle Position Analysis:\n")
        f.write("-" * 60 + "\n")
        pos_data = [r for r in data if r["target_pos"] is not None
                   and r["distractor_pos"] is not None]
        if pos_data:
            earlier = [r for r in pos_data if r["target_pos"] < r["distractor_pos"]]
            later = [r for r in pos_data if r["target_pos"] >= r["distractor_pos"]]
            f.write(f"Samples with target BEFORE distractor: {len(earlier)}\n")
            f.write(f"Samples with target AFTER distractor:  {len(later)}\n")
            f.write(f"Avg score (target earlier): {np.mean([r['score'] for r in earlier]):.3f}\n")
            f.write(f"Avg score (target later):   {np.mean([r['score'] for r in later]):.3f}\n")

            f.write(f"\nPer condition:\n")
            for cond in cond_order:
                e = [r for r in earlier if r["condition"] == cond]
                l = [r for r in later if r["condition"] == cond]
                e_avg = np.mean([r["score"] for r in e]) if e else 0
                l_avg = np.mean([r["score"] for r in l]) if l else 0
                e_pct = 100 * sum(1 for r in e if r["score"] >= PERFECT_THRESHOLD) / len(e) if e else 0
                l_pct = 100 * sum(1 for r in l if r["score"] >= PERFECT_THRESHOLD) / len(l) if l else 0
                f.write(f"  {cond:<22} earlier: avg={e_avg:.3f} %perf={e_pct:.0f}%  "
                       f"later: avg={l_avg:.3f} %perf={l_pct:.0f}%  "
                       f"gap={l_avg - e_avg:+.3f}\n")

    print(f"\n  Saved: {summary_path}")
    print(f"\nDone! {len(list(ANALYSIS_DIR.glob('*.png')))} plots saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
