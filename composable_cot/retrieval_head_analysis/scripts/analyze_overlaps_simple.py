"""
Simple head-overlap report.

Per professor's scope: "compare heads and check for overlaps, no masking."

Outputs all pairwise overlaps + overlap with published zero-shot Qwen top-16,
per bin, at K ∈ {8, 16, 32}.

Output:
  analysis/overlaps_simple.csv
  analysis/figures/overlaps_pairwise.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
CONDITIONS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
SHORT = {"lora_base": "lora", "y2_base": "y2", "y2_rpe_cur_L16k": "y2_rpe"}
K_VALUES = [8, 16, 32]

PUBLISHED_TOP16 = [
    "16-19", "16-2", "16-20", "16-14", "16-0", "15-24", "16-18", "17-18",
    "16-1", "19-18", "19-19", "18-16", "16-17", "19-25", "19-17", "20-21",
]

ROOT = Path("composable_cot/retrieval_head_analysis")
OUT = ROOT / "analysis"


def load_topk(cond, b, k):
    p = ROOT / "results" / cond / f"{b}_head_scores.json"
    pairs = json.loads(p.read_text())
    return [h for h, _ in pairs[:k]]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1), len(sa & sb)


def main():
    print("Head overlap report — pairwise + vs published\n")
    rows = []

    # 1. All pairwise (condition_A, condition_B) overlaps per bin per K
    for k in K_VALUES:
        for b in BINS:
            cond_topk = {c: load_topk(c, b, k) for c in CONDITIONS}
            # all 3 pairs
            for ca, cb in combinations(CONDITIONS, 2):
                jac, n = jaccard(cond_topk[ca], cond_topk[cb])
                rows.append({"k": k, "bin": b,
                              "comparison": f"{SHORT[ca]} vs {SHORT[cb]}",
                              "jaccard": jac, "n_overlap": n})
            # each condition vs published
            for c in CONDITIONS:
                jac, n = jaccard(cond_topk[c], PUBLISHED_TOP16[:k])
                rows.append({"k": k, "bin": b,
                              "comparison": f"{SHORT[c]} vs published",
                              "jaccard": jac, "n_overlap": n})

    csv = OUT / "overlaps_simple.csv"
    with csv.open("w") as f:
        f.write("k,bin,comparison,jaccard,n_overlap\n")
        for r in rows:
            f.write(f"{r['k']},{r['bin']},{r['comparison']},"
                    f"{r['jaccard']:.4f},{r['n_overlap']}\n")
    print(f"Wrote {csv}")

    # Headline: K=16 overlap matrix per bin
    print("\n=== K = 16 — pairwise count of overlapping heads (out of 16) ===")
    print(f"{'comparison':<28s} " + " ".join(f"{b:>5s}" for b in BINS))
    comparisons = (
        [f"{SHORT[a]} vs {SHORT[b]}" for a, b in combinations(CONDITIONS, 2)]
        + [f"{SHORT[c]} vs published" for c in CONDITIONS]
    )
    for comp in comparisons:
        cells = []
        for b in BINS:
            r = next((r for r in rows
                     if r["k"] == 16 and r["bin"] == b and r["comparison"] == comp),
                     None)
            cells.append(f"{r['n_overlap']:>5d}" if r else "    -")
        print(f"  {comp:<28s} " + " ".join(cells))

    print("\n=== K = 16 — Jaccard ===")
    print(f"{'comparison':<28s} " + " ".join(f"{b:>5s}" for b in BINS))
    for comp in comparisons:
        cells = []
        for b in BINS:
            r = next((r for r in rows
                     if r["k"] == 16 and r["bin"] == b and r["comparison"] == comp),
                     None)
            cells.append(f"{r['jaccard']:>5.2f}" if r else "    -")
        print(f"  {comp:<28s} " + " ".join(cells))

    # Plot — one panel per comparison, line over bins, multiple K
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, comp in zip(axes.flat, comparisons):
        for k in K_VALUES:
            xs, ys = [], []
            for b in BINS:
                r = next((r for r in rows
                         if r["k"] == k and r["bin"] == b and r["comparison"] == comp),
                         None)
                if r:
                    xs.append(b)
                    ys.append(r["jaccard"])
            ax.plot(xs, ys, marker="o", label=f"K={k}")
        ax.set_title(comp)
        ax.set_xlabel("bin")
        ax.set_ylabel("Jaccard overlap")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Pairwise top-K head overlaps across bins (per comparison, K∈{8,16,32})")
    fig.tight_layout()
    p = OUT / "figures" / "overlaps_pairwise.png"
    p.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
