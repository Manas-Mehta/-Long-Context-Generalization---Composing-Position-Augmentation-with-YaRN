"""
Phase 4f — Δ-score on the PUBLISHED zero-shot Qwen2.5-7B-Instruct top-16.

This is the analysis the original suggestion called for. Instead of using
lora_base as the reference, we use the QR heads PUBLISHED by the QRHead paper
authors (BEIR-NQ-derived, on vanilla Qwen2.5-7B-Instruct).

For each (condition, bin):
  - Look up each of the 16 published heads' calibrated QR score in our detection
  - Compute Δ vs reference = trained_score - mean_published_score
    (we don't have the published scores per-bin since the paper computed them
     on BEIR-NQ; we use the rank-position itself as the signal)

What we report:
  1. Direct score of each published top-16 head, per (condition, bin)
  2. Mean and median rank of the published top-16 in each (condition, bin) ranking
  3. How many of the published top-16 still appear in each condition's
     top-16 / top-32 / top-64

Output: analysis/published_head_tracking.csv
        analysis/figures/published_head_scores_heatmap.png
"""
import json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
CONDITIONS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]

# Published Qwen2.5-7B-Instruct top-16 QR heads, from
# https://huggingface.co/datasets/PrincetonPLI/QRHead/tree/main/head_detection
# (BEIR-NQ-detected on vanilla Qwen).
PUBLISHED_TOP16 = [
    "16-19", "16-2", "16-20", "16-14", "16-0", "15-24", "16-18", "17-18",
    "16-1", "19-18", "19-19", "18-16", "16-17", "19-25", "19-17", "20-21",
]

ROOT = Path("experiments/retrieval_heads")
OUT = ROOT / "analysis"


def load_head_scores(cond, b):
    p = ROOT / "results" / cond / f"{b}_head_scores.json"
    return [(h, s) for h, s in json.loads(p.read_text())]


def main():
    print("Phase 4f — Tracking published zero-shot Qwen top-16 across our conditions\n")
    print(f"Published top-16 (BEIR-NQ-detected on vanilla Qwen):")
    print(f"  {PUBLISHED_TOP16}\n")

    rows = []
    # heatmap data: (condition × bin) of mean(score of published top-16)
    heatmap_mean = np.full((len(CONDITIONS), len(BINS)), np.nan)
    # in-top-K-survival: how many of the 16 are still in top-{16, 32, 64}
    survival = {}

    for ci, cond in enumerate(CONDITIONS):
        for bi, b in enumerate(BINS):
            try:
                ranking = load_head_scores(cond, b)  # [(head, score), ...] sorted desc
            except FileNotFoundError:
                continue
            score_lookup = dict(ranking)
            rank_lookup = {h: i + 1 for i, (h, _) in enumerate(ranking)}  # 1-indexed

            scores_pub = [score_lookup.get(h, float("nan")) for h in PUBLISHED_TOP16]
            ranks_pub = [rank_lookup.get(h, len(ranking)) for h in PUBLISHED_TOP16]

            mean_score = float(np.nanmean(scores_pub))
            mean_rank = float(np.mean(ranks_pub))
            median_rank = float(np.median(ranks_pub))
            in_top16 = sum(1 for r in ranks_pub if r <= 16)
            in_top32 = sum(1 for r in ranks_pub if r <= 32)
            in_top64 = sum(1 for r in ranks_pub if r <= 64)

            heatmap_mean[ci, bi] = mean_score
            survival[(cond, b)] = (in_top16, in_top32, in_top64)

            for h, s, r in zip(PUBLISHED_TOP16, scores_pub, ranks_pub):
                rows.append({
                    "condition": cond, "bin": b, "head": h,
                    "score": s, "rank": r,
                })

            print(f"  [{cond:<20s} {b:>5s}]  mean_score={mean_score:.4f}  "
                  f"mean_rank={mean_rank:5.1f}  median_rank={median_rank:5.1f}  "
                  f"in_top16={in_top16:2d}  in_top32={in_top32:2d}  in_top64={in_top64:2d}")

    # Write per-head detail
    csv = OUT / "published_head_tracking.csv"
    with csv.open("w") as f:
        f.write("condition,bin,head,score,rank\n")
        for r in rows:
            f.write(f"{r['condition']},{r['bin']},{r['head']},{r['score']:.6f},{r['rank']}\n")
    print(f"\nWrote {csv}")

    # Write summary table
    sum_csv = OUT / "published_head_summary.csv"
    with sum_csv.open("w") as f:
        f.write("condition,bin,mean_score,mean_rank,median_rank,in_top16,in_top32,in_top64\n")
        for ci, cond in enumerate(CONDITIONS):
            for bi, b in enumerate(BINS):
                if (cond, b) not in survival:
                    continue
                cell_rows = [r for r in rows if r["condition"] == cond and r["bin"] == b]
                if not cell_rows:
                    continue
                scs = [r["score"] for r in cell_rows]
                rks = [r["rank"] for r in cell_rows]
                in16, in32, in64 = survival[(cond, b)]
                f.write(f"{cond},{b},{np.mean(scs):.6f},{np.mean(rks):.2f},"
                        f"{np.median(rks):.1f},{in16},{in32},{in64}\n")
    print(f"Wrote {sum_csv}")

    # Print survival table
    print(f"\n  Survival of published top-16 in our conditions (count in top-16 of trained model):")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>5s}" for b in BINS))
    for cond in CONDITIONS:
        cells = [survival.get((cond, b), (np.nan, np.nan, np.nan))[0] for b in BINS]
        print(f"  {cond:<20s} " + " ".join(f"{c:>5d}" if not np.isnan(c) else "   nan"
                                            for c in cells))

    print(f"\n  Mean score on published top-16 per (condition, bin):")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>7s}" for b in BINS))
    for ci, cond in enumerate(CONDITIONS):
        cells = [heatmap_mean[ci, bi] for bi in range(len(BINS))]
        print(f"  {cond:<20s} " + " ".join(f"{c:>7.4f}" if not np.isnan(c) else "    nan"
                                            for c in cells))

    # Heatmap figure
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap_mean, aspect="auto", cmap="RdYlGn",
                    vmin=np.nanmin(heatmap_mean), vmax=np.nanmax(heatmap_mean))
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS)
    ax.set_yticks(range(len(CONDITIONS))); ax.set_yticklabels(CONDITIONS)
    ax.set_title("Mean QR score on published zero-shot Qwen top-16 per (condition, bin)")
    for i in range(len(CONDITIONS)):
        for j in range(len(BINS)):
            v = heatmap_mean[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = OUT / "figures" / "published_head_scores_heatmap.png"
    p.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
