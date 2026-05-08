"""
Phase 4g — Recall@K (paper's own metric).

For each (condition, bin), use the per-doc QR scores (already saved in
{bin}_doc_scores_per_head.pt) to build a QRRetriever-style score per
(qid, doc_id), where score = mean over top-K heads of the per-(layer, head)
tensor entry.

Then compute recall@K against gt_docs from detection_set_{bin}.json — exactly
the same as `eval_retrieval.py` from the QRHead repo.

Two head sets per condition:
  (1) condition's OWN top-16 (paper's standard usage)
  (2) PUBLISHED zero-shot Qwen top-16 (the reference we should have led with)

Output:
  analysis/retrieval_recall.csv           per (condition, bin, head_set, K)
  analysis/figures/recall_per_bin.png
"""
import json
from pathlib import Path
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
CONDITIONS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]

PUBLISHED_TOP16 = [
    "16-19", "16-2", "16-20", "16-14", "16-0", "15-24", "16-18", "17-18",
    "16-1", "19-18", "19-19", "18-16", "16-17", "19-25", "19-17", "20-21",
]

ROOT = Path("composable_cot/retrieval_head_analysis")
DATA = ROOT / "data"
OUT = ROOT / "analysis"


def parse_head(hid):
    l, h = hid.split("-")
    return int(l), int(h)


def load_head_scores(cond, b):
    p = ROOT / "results" / cond / f"{b}_head_scores.json"
    return [(h, s) for h, s in json.loads(p.read_text())]


def load_pt(cond, b):
    p = ROOT / "results" / cond / f"{b}_doc_scores_per_head.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def load_detection_set(b):
    return json.loads((DATA / f"detection_set_{b}.json").read_text())


def topk_heads_own(cond, b, k):
    """This condition's own top-K heads at this bin."""
    ranking = load_head_scores(cond, b)
    return [h for h, _ in ranking[:k]]


def compute_per_doc_scores(doc_scores_pt, head_ids):
    """For each qid, compute {doc_id: mean_over_head_ids(tensor[l, h])}."""
    head_idx = [parse_head(h) for h in head_ids]
    out = {}
    for qid, doc_to_tensor in doc_scores_pt.items():
        per_doc = {}
        for did, t in doc_to_tensor.items():
            vals = [float(t[l, h].item()) for l, h in head_idx]
            per_doc[did] = float(np.mean(vals))
        out[qid] = per_doc
    return out


def compute_recall(samples, scores, k):
    """Same logic as exp_scripts/retrieval/eval_retrieval.py:compute_recall."""
    recall = recall_any = recall_all = 0
    n = 0
    for sample in samples:
        qid = sample["idx"]
        gt = [str(g) for g in sample.get("gt_docs", [])]
        if qid not in scores or not gt:
            continue
        ds = scores[qid]
        sorted_docs = sorted(ds.items(), key=lambda kv: kv[1], reverse=True)
        topk = {d for d, _ in sorted_docs[:k]}
        any_match = any(g in topk for g in gt)
        all_match = all(g in topk for g in gt)
        count = sum(1 for g in gt if g in topk)
        recall += count / len(gt)
        recall_any += int(any_match)
        recall_all += int(all_match)
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0
    return recall / n, recall_any / n, recall_all / n


def main():
    print("Phase 4g — Recall@K using QRHead's own metric (eval_retrieval.py logic)\n")
    rows = []

    for b in BINS:
        try:
            samples = load_detection_set(b)
        except FileNotFoundError:
            print(f"  [{b}] missing detection_set; skip")
            continue
        for cond in CONDITIONS:
            try:
                pt = load_pt(cond, b)
            except FileNotFoundError:
                continue

            # head set 1: condition's own top-16
            own_heads = topk_heads_own(cond, b, 16)
            scores_own = compute_per_doc_scores(pt, own_heads)

            # head set 2: published zero-shot Qwen top-16
            scores_pub = compute_per_doc_scores(pt, PUBLISHED_TOP16)

            for K in [3, 5, 10]:
                r_own, ra_own, rall_own = compute_recall(samples, scores_own, K)
                r_pub, ra_pub, rall_pub = compute_recall(samples, scores_pub, K)
                rows.append({
                    "condition": cond, "bin": b, "K": K,
                    "head_set": "own_top16",
                    "recall": r_own, "recall_any": ra_own, "recall_all": rall_own,
                })
                rows.append({
                    "condition": cond, "bin": b, "K": K,
                    "head_set": "published_top16",
                    "recall": r_pub, "recall_any": ra_pub, "recall_all": rall_pub,
                })
            # print K=5 for terminal readout
            r_own5, ra_own5, _ = compute_recall(samples, scores_own, 5)
            r_pub5, ra_pub5, _ = compute_recall(samples, scores_pub, 5)
            print(f"  [{cond:<20s} {b:>5s}]  R@5(own)={r_own5:.3f}/any={ra_own5:.3f}   "
                  f"R@5(pub)={r_pub5:.3f}/any={ra_pub5:.3f}")

    csv = OUT / "retrieval_recall.csv"
    with csv.open("w") as f:
        f.write("condition,bin,K,head_set,recall,recall_any,recall_all\n")
        for r in rows:
            f.write(f"{r['condition']},{r['bin']},{r['K']},{r['head_set']},"
                    f"{r['recall']:.4f},{r['recall_any']:.4f},{r['recall_all']:.4f}\n")
    print(f"\nWrote {csv}")

    # Plot R@5 per bin, two head sets, per condition
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, hs, title in zip(axes, ["own_top16", "published_top16"],
                              ["Condition's OWN top-16", "PUBLISHED zero-shot Qwen top-16"]):
        for cond in CONDITIONS:
            xs, ys = [], []
            for b in BINS:
                r = next((r for r in rows
                         if r["condition"] == cond and r["bin"] == b
                         and r["K"] == 5 and r["head_set"] == hs), None)
                if r:
                    xs.append(b)
                    ys.append(r["recall"])
            ax.plot(xs, ys, marker="o", label=cond)
        ax.set_xlabel("bin")
        ax.set_ylabel("Recall@5")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Retrieval Recall@5 — paper's own metric, two head-set choices")
    fig.tight_layout()
    p = OUT / "figures" / "recall_per_bin.png"
    p.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # Print headline tables
    print(f"\n  Recall@5 using condition's OWN top-16 (per bin):")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>6s}" for b in BINS))
    for cond in CONDITIONS:
        cells = []
        for b in BINS:
            r = next((r for r in rows if r["condition"] == cond and r["bin"] == b
                      and r["K"] == 5 and r["head_set"] == "own_top16"), None)
            cells.append(f"{r['recall']:>6.3f}" if r else "    nan")
        print(f"  {cond:<20s} " + " ".join(cells))

    print(f"\n  Recall@5 using PUBLISHED zero-shot Qwen top-16 (per bin):")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>6s}" for b in BINS))
    for cond in CONDITIONS:
        cells = []
        for b in BINS:
            r = next((r for r in rows if r["condition"] == cond and r["bin"] == b
                      and r["K"] == 5 and r["head_set"] == "published_top16"), None)
            cells.append(f"{r['recall']:>6.3f}" if r else "    nan")
        print(f"  {cond:<20s} " + " ".join(cells))


if __name__ == "__main__":
    main()
