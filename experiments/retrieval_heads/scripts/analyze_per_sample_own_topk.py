"""
Phase 4h — Per-sample QR-vs-correctness correlation, using EACH CONDITION'S OWN top-K.

This is the reference-free version of the per-sample test. Each model is probed
on its own forward passes, against its own correctness labels, using its own
top-K QR heads. No external reference (no lora_base, no published top-16).

The question this answers:
  "Does each model's actual retrieval pattern predict whether it gets each
   sample right, sample by sample?"

If yes for a condition, the heads detected as that model's top-K are
mechanistically tied to correctness. If no, the heads are decorative —
correctness is driven by something else (MLP, value mixing, etc.).

For each (condition, bin):
  1. Load this condition's own top-K head ranking from {bin}_head_scores.json.
  2. Load per-doc QR-score tensors from {bin}_doc_scores_per_head.pt.
  3. Per sample: S_sample = Σ over (head ∈ own top-K) of Σ over (gold doc) of score.
  4. Pair with per-sample BABILong correctness from predictions_*.json.
  5. Compute point-biserial r and Spearman r between S_sample and correctness.

We sweep K ∈ {8, 16, 32} so the headline isn't an artifact of K choice.

Outputs:
  analysis/per_sample_own_topk.csv
  analysis/figures/per_sample_own_topk.png
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pointbiserialr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
CONDITIONS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
K_VALUES = [8, 16, 32]

ROOT = Path("composable_cot/retrieval_head_analysis")
DATA = ROOT / "data"
PRED = Path("composable_cot/BABIlong/results")
OUT = ROOT / "analysis"


def parse_head(hid):
    l, h = hid.split("-")
    return int(l), int(h)


def load_head_scores(cond, b):
    p = ROOT / "results" / cond / f"{b}_head_scores.json"
    return [(h, s) for h, s in json.loads(p.read_text())]


def own_topk(cond, b, k):
    return [h for h, _ in load_head_scores(cond, b)[:k]]


def load_pt(cond, b):
    return torch.load(ROOT / "results" / cond / f"{b}_doc_scores_per_head.pt",
                      map_location="cpu", weights_only=False)


def load_detection_set(b):
    return json.loads((DATA / f"detection_set_{b}.json").read_text())


def load_correctness(cond, b):
    p = PRED / f"{cond}_me" / f"predictions_{b}.json"
    if not p.exists():
        return None
    return [int(bool(pred.get("correct"))) for pred in json.loads(p.read_text())]


def per_sample_signature(doc_scores_pt, sample, head_ids):
    """Sum over (head ∈ head_ids) × (gold doc) of QR score."""
    head_idx = [parse_head(h) for h in head_ids]
    qid = sample["idx"]
    gt = set(sample.get("gt_docs", []))
    if qid not in doc_scores_pt or not gt:
        return None
    per_doc = doc_scores_pt[qid]
    total = 0.0
    for did in gt:
        if did not in per_doc:
            continue
        t = per_doc[did]
        for l, h in head_idx:
            total += float(t[l, h].item())
    return total


def main():
    print("Phase 4h — Per-sample QR-vs-correctness using EACH CONDITION'S OWN top-K\n")
    rows = []

    for k in K_VALUES:
        print(f"\n=== K = {k} ===")
        for b in BINS:
            try:
                detset = load_detection_set(b)
            except FileNotFoundError:
                continue
            for cond in CONDITIONS:
                try:
                    pt = load_pt(cond, b)
                except FileNotFoundError:
                    continue
                heads = own_topk(cond, b, k)
                corr_all = load_correctness(cond, b)
                if corr_all is None:
                    continue

                sigs, labels = [], []
                for sample in detset:
                    sig = per_sample_signature(pt, sample, heads)
                    if sig is None:
                        continue
                    story_idx = sample["story_idx"]
                    if story_idx >= len(corr_all):
                        continue
                    sigs.append(sig)
                    labels.append(corr_all[story_idx])

                n = len(sigs)
                if n < 5 or len(set(labels)) < 2:
                    r_pb = p_pb = r_sp = float("nan")
                else:
                    r_pb, p_pb = pointbiserialr(labels, sigs)
                    r_sp, _ = spearmanr(sigs, labels)

                rows.append({
                    "k": k, "bin": b, "condition": cond,
                    "n": n, "n_correct": int(sum(labels)),
                    "mean_sig": float(np.mean(sigs)) if sigs else float("nan"),
                    "mean_sig_correct": float(np.mean([s for s, l in zip(sigs, labels) if l == 1]))
                                        if any(labels) else float("nan"),
                    "mean_sig_wrong": float(np.mean([s for s, l in zip(sigs, labels) if l == 0]))
                                      if not all(labels) else float("nan"),
                    "r_pb": r_pb, "p_pb": p_pb, "r_sp": r_sp,
                })
                star = "★" if (not np.isnan(p_pb) and p_pb < 0.05) else " "
                print(f"  [{b:>5s} {cond:<20s}]  n={n:2d}  correct={int(sum(labels)):2d}/{n:2d}  "
                      f"r_pb={r_pb:+.3f} (p={p_pb:.3f}){star}  r_sp={r_sp:+.3f}")

    csv = OUT / "per_sample_own_topk.csv"
    with csv.open("w") as f:
        f.write("k,bin,condition,n,n_correct,mean_sig,mean_sig_correct,mean_sig_wrong,"
                "r_pointbiserial,p_pointbiserial,r_spearman\n")
        for r in rows:
            f.write(f"{r['k']},{r['bin']},{r['condition']},{r['n']},{r['n_correct']},"
                    f"{r['mean_sig']:.4f},{r['mean_sig_correct']:.4f},{r['mean_sig_wrong']:.4f},"
                    f"{r['r_pb']:.4f},{r['p_pb']:.4e},{r['r_sp']:.4f}\n")
    print(f"\nWrote {csv}")

    # Plot 3 subplots, one per K. r_pb on y, bin on x, line per condition.
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 5),
                              sharey=True)
    for ax, k in zip(axes, K_VALUES):
        for cond in CONDITIONS:
            xs, ys = [], []
            for b in BINS:
                r = next((r for r in rows
                         if r["k"] == k and r["bin"] == b and r["condition"] == cond),
                         None)
                if r and not np.isnan(r["r_pb"]):
                    xs.append(b)
                    ys.append(r["r_pb"])
            ax.plot(xs, ys, marker="o", label=cond)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        # Significance threshold for n=60: |r| ≈ 0.25 ⇒ p ≈ 0.05
        ax.axhline(0.25, color="green", ls=":", lw=0.6, alpha=0.5)
        ax.axhline(-0.25, color="red", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("bin")
        ax.set_ylabel("point-biserial r" if k == K_VALUES[0] else "")
        ax.set_title(f"K = {k} (own heads)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-sample QR-vs-correctness — each condition's OWN top-K")
    fig.tight_layout()
    p = OUT / "figures" / "per_sample_own_topk.png"
    p.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # Headline summary table — at K=16
    print(f"\n  HEADLINE: point-biserial r at K=16, per (condition, bin)")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>7s}" for b in BINS))
    for cond in CONDITIONS:
        cells = []
        for b in BINS:
            r = next((r for r in rows
                     if r["k"] == 16 and r["bin"] == b and r["condition"] == cond),
                     None)
            cells.append(f"{r['r_pb']:>+7.3f}" if r and not np.isnan(r['r_pb']) else "    nan")
        print(f"  {cond:<20s} " + " ".join(cells))


if __name__ == "__main__":
    main()
