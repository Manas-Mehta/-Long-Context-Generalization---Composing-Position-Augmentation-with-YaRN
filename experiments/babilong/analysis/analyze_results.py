"""
BABILong Results Analysis — Phase 1 + 2
Generates all plots and error analysis from eval prediction files.

Usage:
    python experiments/babilong/analysis/analyze_results.py

Outputs to: experiments/babilong/analysis/figures/
"""

import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR.parent / "results"
FIGURES_DIR  = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── condition display config ──────────────────────────────────────────────────
CONDITIONS = {
    "lora_base":   {"label": "LoRA Baseline",       "color": "#888888", "ls": "--"},
    "rpe_only":    {"label": "RPE Only",             "color": "#e07b39", "ls": "-."},
    "y2_base":     {"label": "YaRN Only",            "color": "#4878cf", "ls": "-"},
    "y2_pose_32k": {"label": "YaRN + PoSE (32k)",   "color": "#d43f3a", "ls": "-"},
    "pose_only":   {"label": "PoSE Only",            "color": "#6acc65", "ls": ":"},
}

BINS_ORDERED = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
BIN_TOKENS   = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
ROOMS        = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]
CHANCE       = 1 / 6  # 6-class uniform

# ── data loading ─────────────────────────────────────────────────────────────
def load_condition(cond):
    """Returns dict: bin_name -> list of prediction dicts."""
    cond_dir = RESULTS_DIR / cond
    data = {}
    for b in BINS_ORDERED:
        path = cond_dir / f"predictions_{b}.json"
        if path.exists():
            with open(path) as f:
                data[b] = json.load(f)
    return data


def load_all():
    all_data = {}
    for cond in CONDITIONS:
        d = load_condition(cond)
        if d:
            all_data[cond] = d
    return all_data


def accuracy(preds):
    valid = [p for p in preds if p["prediction"] != "__OOM__"]
    if not valid:
        return None
    return sum(p["correct"] for p in valid) / len(valid)


# ── Plot 1: Accuracy vs Context Length ───────────────────────────────────────
def plot_accuracy_curves(all_data):
    fig, ax = plt.subplots(figsize=(10, 6))

    for cond, cfg in CONDITIONS.items():
        if cond not in all_data:
            continue
        xs, ys = [], []
        for b, tok in zip(BINS_ORDERED, BIN_TOKENS):
            if b in all_data[cond]:
                acc = accuracy(all_data[cond][b])
                if acc is not None:
                    xs.append(tok / 1000)
                    ys.append(acc * 100)
        ax.plot(xs, ys, label=cfg["label"], color=cfg["color"],
                ls=cfg["ls"], marker="o", linewidth=2.5, markersize=7)

    ax.axhline(CHANCE * 100, color="black", ls=":", lw=1.5, alpha=0.6, label="Chance (16.7%)")
    ax.set_xlabel("Context Length (K tokens)", fontsize=13)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_title("BABILong QA3: Accuracy vs Context Length\n(v1 epoch-1, 100 samples/bin)", fontsize=14)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    out = FIGURES_DIR / "01_accuracy_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 2: Prediction Bias (what models predict when wrong) ──────────────────
def plot_prediction_bias(all_data):
    """For y2_base and y2_pose_32k, show prediction distribution at long bins."""
    long_bins = ["32k", "64k", "128k"]
    compare_conds = ["y2_base", "y2_pose_32k"]
    n_bins = len(long_bins)

    fig, axes = plt.subplots(2, n_bins, figsize=(14, 8), sharey=False)

    for ci, cond in enumerate(compare_conds):
        if cond not in all_data:
            continue
        cfg = CONDITIONS[cond]
        for bi, b in enumerate(long_bins):
            ax = axes[ci][bi]
            if b not in all_data[cond]:
                ax.set_visible(False)
                continue

            preds = all_data[cond][b]
            wrong = [p for p in preds if not p["correct"] and p["prediction"] != "__OOM__"]
            total_wrong = len(wrong)

            room_counts = Counter(p["prediction"] for p in wrong)
            counts = [room_counts.get(r, 0) for r in ROOMS]
            pcts   = [c / total_wrong * 100 if total_wrong > 0 else 0 for c in counts]

            bars = ax.bar(ROOMS, pcts, color=cfg["color"], alpha=0.8, edgecolor="white")
            ax.axhline(100/6, color="black", ls="--", lw=1.2, alpha=0.7, label="Uniform (16.7%)")
            ax.set_title(f"{b} context\n(n_wrong={total_wrong})", fontsize=10)
            ax.set_ylim(0, 65)
            ax.set_ylabel("% of wrong predictions" if bi == 0 else "", fontsize=9)
            ax.tick_params(axis="x", rotation=35, labelsize=8)

            # Annotate bars
            for bar, pct in zip(bars, pcts):
                if pct > 2:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)

        axes[ci][0].set_ylabel(f"{cfg['label']}\n\n% of errors", fontsize=10)

    fig.suptitle("Prediction Bias: What rooms models predict when WRONG\n"
                 "(recency bias → mode collapse to 'office'/'hallway')", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "02_prediction_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(all_data, cond, bin_name):
    if cond not in all_data or bin_name not in all_data[cond]:
        return
    preds = [p for p in all_data[cond][bin_name] if p["prediction"] != "__OOM__"]

    matrix = np.zeros((6, 6), dtype=int)
    room_idx = {r: i for i, r in enumerate(ROOMS)}

    for p in preds:
        t = p["target"]
        pred = p["prediction"]
        if t in room_idx and pred in room_idx:
            matrix[room_idx[t]][room_idx[pred]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(ROOMS, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(ROOMS, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — {CONDITIONS[cond]['label']}\nBin: {bin_name}", fontsize=12)

    for i in range(6):
        for j in range(6):
            val = matrix[i][j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > matrix.max()*0.5 else "black", fontsize=11)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = FIGURES_DIR / f"03_confusion_{cond}_{bin_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 4: Per-bin Agreement (PoSE wins / YaRN wins / both / neither) ────────
def plot_agreement(all_data):
    """For each bin, how many samples fall into each agreement category."""
    if "y2_base" not in all_data or "y2_pose_32k" not in all_data:
        return

    both_correct, pose_only_correct, yarn_only_correct, both_wrong = [], [], [], []
    bin_labels = []

    for b in BINS_ORDERED:
        if b not in all_data["y2_base"] or b not in all_data["y2_pose_32k"]:
            continue
        yarn_preds = {i: p for i, p in enumerate(all_data["y2_base"][b])
                      if p["prediction"] != "__OOM__"}
        pose_preds = {i: p for i, p in enumerate(all_data["y2_pose_32k"][b])
                      if p["prediction"] != "__OOM__"}
        shared = set(yarn_preds) & set(pose_preds)

        bc = sum(1 for i in shared if yarn_preds[i]["correct"] and pose_preds[i]["correct"])
        pc = sum(1 for i in shared if not yarn_preds[i]["correct"] and pose_preds[i]["correct"])
        yc = sum(1 for i in shared if yarn_preds[i]["correct"] and not pose_preds[i]["correct"])
        bw = sum(1 for i in shared if not yarn_preds[i]["correct"] and not pose_preds[i]["correct"])

        both_correct.append(bc)
        pose_only_correct.append(pc)
        yarn_only_correct.append(yc)
        both_wrong.append(bw)
        bin_labels.append(b)

    x = np.arange(len(bin_labels))
    width = 0.6
    fig, ax = plt.subplots(figsize=(11, 6))

    bars1 = ax.bar(x, both_correct,     width, label="Both correct",           color="#2ecc71")
    bars2 = ax.bar(x, pose_only_correct, width, bottom=both_correct,            label="YaRN+PoSE only ✓", color="#d43f3a")
    b2 = np.array(both_correct) + np.array(pose_only_correct)
    bars3 = ax.bar(x, yarn_only_correct, width, bottom=b2,                      label="YaRN only ✓",      color="#4878cf")
    b3 = b2 + np.array(yarn_only_correct)
    bars4 = ax.bar(x, both_wrong,        width, bottom=b3,                      label="Both wrong",        color="#cccccc")

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.set_xlabel("Context Length Bin", fontsize=12)
    ax.set_ylabel("Number of Samples (out of 100)", fontsize=12)
    ax.set_title("Sample Agreement: YaRN+PoSE vs YaRN Alone per Bin\n"
                 "(Red = samples PoSE uniquely solves)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    out = FIGURES_DIR / "04_agreement_per_bin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Print text summary
    print("\n── Agreement Analysis ──")
    print(f"{'Bin':<8} {'Both✓':>8} {'PoSE✓only':>10} {'YaRN✓only':>10} {'Both✗':>8}")
    for i, b in enumerate(bin_labels):
        print(f"{b:<8} {both_correct[i]:>8} {pose_only_correct[i]:>10} "
              f"{yarn_only_correct[i]:>10} {both_wrong[i]:>8}")


# ── Plot 5: Per-label accuracy ─────────────────────────────────────────────────
def plot_per_label_accuracy(all_data):
    """Accuracy broken down by true answer label, for y2_base vs y2_pose_32k."""
    long_bins = ["32k", "64k", "128k"]
    compare_conds = ["y2_base", "y2_pose_32k", "lora_base"]
    markers = ["s", "o", "^"]

    fig, axes = plt.subplots(1, len(long_bins), figsize=(15, 5), sharey=True)

    for bi, b in enumerate(long_bins):
        ax = axes[bi]
        x = np.arange(len(ROOMS))
        width = 0.25

        for ci, (cond, mk) in enumerate(zip(compare_conds, markers)):
            if cond not in all_data or b not in all_data[cond]:
                continue
            preds = [p for p in all_data[cond][b] if p["prediction"] != "__OOM__"]
            accs = []
            for room in ROOMS:
                room_preds = [p for p in preds if p["target"] == room]
                if room_preds:
                    accs.append(sum(p["correct"] for p in room_preds) / len(room_preds) * 100)
                else:
                    accs.append(0)
            offset = (ci - 1) * width
            cfg = CONDITIONS[cond]
            ax.bar(x + offset, accs, width, label=cfg["label"],
                   color=cfg["color"], alpha=0.85, edgecolor="white")

        ax.axhline(CHANCE * 100, color="black", ls="--", lw=1.2, alpha=0.6)
        ax.set_title(f"Bin: {b}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(ROOMS, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 105)
        if bi == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Label Accuracy: Does YaRN fail on specific room types?", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "05_per_label_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Plot 6: n_tokens scatter (within-bin variation) ───────────────────────────
def plot_token_scatter(all_data):
    """Scatter: actual n_tokens vs correct, for y2_base and y2_pose_32k, at 128k bin."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ci, cond in enumerate(["y2_base", "y2_pose_32k"]):
        ax = axes[ci]
        if cond not in all_data or "128k" not in all_data[cond]:
            continue
        preds = [p for p in all_data[cond]["128k"] if p["prediction"] != "__OOM__"]
        tokens_correct = [p["n_tokens"] for p in preds if p["correct"]]
        tokens_wrong   = [p["n_tokens"] for p in preds if not p["correct"]]

        ax.scatter(tokens_correct, [1]*len(tokens_correct),
                   color=CONDITIONS[cond]["color"], alpha=0.5, label="Correct", s=50)
        ax.scatter(tokens_wrong,   [0]*len(tokens_wrong),
                   color="#888888", alpha=0.5, label="Wrong", s=50, marker="x")

        # Histogram by correctness
        ax2 = ax.twinx()
        if tokens_correct:
            ax2.hist(tokens_correct, bins=15, alpha=0.25, color=CONDITIONS[cond]["color"])
        if tokens_wrong:
            ax2.hist(tokens_wrong,   bins=15, alpha=0.15, color="gray")
        ax2.set_ylabel("Count", fontsize=9)

        acc = sum(p["correct"] for p in preds) / len(preds)
        ax.set_title(f"{CONDITIONS[cond]['label']}\n128k bin — accuracy {acc:.0%}", fontsize=11)
        ax.set_xlabel("n_tokens (actual context length)", fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Within-bin Token Length vs Correctness (128k bin)\n"
                 "Are longer samples harder?", fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "06_token_scatter_128k.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Text Analysis: PoSE wins — what was YaRN predicting? ─────────────────────
def analyze_pose_wins(all_data):
    """For each bin, examine samples where PoSE is correct and YaRN is wrong."""
    if "y2_base" not in all_data or "y2_pose_32k" not in all_data:
        return

    print("\n" + "="*70)
    print("ANALYSIS: Samples where YaRN+PoSE is correct but YaRN alone is WRONG")
    print("="*70)

    long_bins = ["32k", "64k", "128k"]
    for b in long_bins:
        if b not in all_data["y2_base"] or b not in all_data["y2_pose_32k"]:
            continue

        yarn  = all_data["y2_base"][b]
        pose  = all_data["y2_pose_32k"][b]

        wins = []
        for i in range(min(len(yarn), len(pose))):
            yp, pp = yarn[i], pose[i]
            if pp["prediction"] == "__OOM__" or yp["prediction"] == "__OOM__":
                continue
            if pp["correct"] and not yp["correct"]:
                wins.append((i, yp, pp))

        print(f"\n── Bin {b}: PoSE wins on {len(wins)}/100 samples ──")
        yarn_pred_dist = Counter(yp["prediction"] for _, yp, _ in wins)
        true_dist      = Counter(pp["target"]     for _, yp, pp in wins)
        print(f"  True answer distribution in PoSE-wins: {dict(true_dist.most_common())}")
        print(f"  YaRN predicted (wrong): {dict(yarn_pred_dist.most_common())}")
        print(f"  → YaRN bias: top prediction = '{yarn_pred_dist.most_common(1)[0][0]}' "
              f"({yarn_pred_dist.most_common(1)[0][1]}/{len(wins)} = "
              f"{yarn_pred_dist.most_common(1)[0][1]/len(wins):.0%})")

        # Sample examples
        print(f"\n  Example questions (PoSE correct / YaRN wrong):")
        for _, yp, pp in wins[:5]:
            print(f"    Q: {pp['question']}")
            print(f"       True: {pp['target']}  |  YaRN predicted: {yp['prediction']}  |  tokens: {pp['n_tokens']:,}")


# ── Plot 7: PoSE-win heatmap (which true answers does PoSE rescue?) ───────────
def plot_pose_rescue_heatmap(all_data):
    """Heatmap: for each (true_label, yarn_wrong_prediction) pair, how many does PoSE rescue?"""
    if "y2_base" not in all_data or "y2_pose_32k" not in all_data:
        return

    matrix = np.zeros((6, 6), dtype=int)
    room_idx = {r: i for i, r in enumerate(ROOMS)}

    for b in ["32k", "64k", "128k"]:
        if b not in all_data["y2_base"] or b not in all_data["y2_pose_32k"]:
            continue
        yarn = all_data["y2_base"][b]
        pose = all_data["y2_pose_32k"][b]
        for i in range(min(len(yarn), len(pose))):
            yp, pp = yarn[i], pose[i]
            if pp["correct"] and not yp["correct"]:
                t = pp["target"]
                ypred = yp["prediction"]
                if t in room_idx and ypred in room_idx:
                    matrix[room_idx[t]][room_idx[ypred]] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Reds")
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(ROOMS, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(ROOMS, fontsize=10)
    ax.set_xlabel("YaRN's wrong prediction", fontsize=11)
    ax.set_ylabel("True answer (PoSE correct)", fontsize=11)
    ax.set_title("PoSE Rescue Heatmap (32k+64k+128k combined)\n"
                 "True label vs what YaRN predicted instead", fontsize=12)

    for i in range(6):
        for j in range(6):
            val = matrix[i][j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > matrix.max()*0.5 else "black", fontsize=11)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out = FIGURES_DIR / "07_pose_rescue_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary_table(all_data):
    print("\n" + "="*70)
    print("ACCURACY SUMMARY TABLE")
    print("="*70)
    header = f"{'Condition':<20}" + "".join(f"{b:>7}" for b in BINS_ORDERED) + f"{'Overall':>9}"
    print(header)
    print("-" * len(header))

    for cond, cfg in CONDITIONS.items():
        if cond not in all_data:
            continue
        row = f"{cfg['label']:<20}"
        total_correct, total_n = 0, 0
        for b in BINS_ORDERED:
            if b in all_data[cond]:
                valid = [p for p in all_data[cond][b] if p["prediction"] != "__OOM__"]
                if valid:
                    acc = sum(p["correct"] for p in valid) / len(valid)
                    row += f"{acc*100:>6.0f}%"
                    total_correct += sum(p["correct"] for p in valid)
                    total_n += len(valid)
                else:
                    row += f"{'—':>7}"
            else:
                row += f"{'—':>7}"
        if total_n > 0:
            row += f"{total_correct/total_n*100:>8.1f}%"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading results...")
    all_data = load_all()
    print(f"Loaded conditions: {list(all_data.keys())}")

    print_summary_table(all_data)

    print("\nGenerating plots...")
    plot_accuracy_curves(all_data)
    plot_prediction_bias(all_data)
    plot_confusion_matrix(all_data, "y2_base",     "128k")
    plot_confusion_matrix(all_data, "y2_pose_32k", "128k")
    plot_confusion_matrix(all_data, "y2_base",     "32k")
    plot_confusion_matrix(all_data, "y2_pose_32k", "32k")
    plot_agreement(all_data)
    plot_per_label_accuracy(all_data)
    plot_token_scatter(all_data)
    plot_pose_rescue_heatmap(all_data)

    analyze_pose_wins(all_data)

    print(f"\nAll outputs saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
