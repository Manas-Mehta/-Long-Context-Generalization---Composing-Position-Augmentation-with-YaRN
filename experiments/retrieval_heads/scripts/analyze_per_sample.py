"""
Phase 4d — per-sample QR-vs-correctness correlation.

For each (condition × bin) cell:
  1. Load per-doc QR-score tensors from {bin}_doc_scores_per_head.pt
     (raw output of QRHead's get_doc_scores_per_head — dict of
      {qid: {doc_id: tensor(n_layers, n_heads)}})
  2. For each sample, compute the sample's QR signature: sum of per-doc scores
     over the GOLD docs, restricted to the lora_base top-K heads.
     One scalar per sample.
  3. Pair with the sample's BABILong correctness label (from predictions_*.json
     in the multi-entry eval).
  4. Compute correlation between sample-level QR signature and correctness:
       point-biserial   (per Spearman convention for continuous–binary pairs)

Mechanistic interpretation: if the QR head's attention to gold sentences
predicts whether the model gets the answer right, the heads are mechanistically
linked to retrieval-driven correctness. If correlation is near zero, attention
is decorative — correctness is driven by something else.

Outputs:
  analysis/per_sample_correlation.csv   (condition, bin, k, r, p, n)
  analysis/figures/correlation_per_bin.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy.stats import pointbiserialr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
CONDITIONS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
REFERENCE = "lora_base"


def load_head_scores(results_dir: Path, condition: str, bin_label: str) -> dict:
    p = results_dir / condition / f"{bin_label}_head_scores.json"
    pairs = json.loads(p.read_text())
    return {h: s for h, s in pairs}


def reference_topk_heads(results_dir: Path, bin_label: str, k: int) -> list:
    """Return list of K head IDs (e.g. '19-22') from lora_base's top-K at this bin."""
    scores = load_head_scores(results_dir, REFERENCE, bin_label)
    return [h for h, _ in sorted(scores.items(), key=lambda kv: kv[1],
                                  reverse=True)[:k]]


def head_id_to_indices(head_id: str) -> tuple:
    """'19-22' -> (19, 22)"""
    layer, head = head_id.split("-")
    return int(layer), int(head)


def load_doc_scores_pt(results_dir: Path, condition: str, bin_label: str):
    """Return the QRHead-saved doc_scores_per_head dict.

    Format: {qid: {doc_id: tensor(n_layers, n_heads)}}
    """
    p = results_dir / condition / f"{bin_label}_doc_scores_per_head.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def load_detection_set(data_dir: Path, bin_label: str) -> list:
    p = data_dir / f"detection_set_{bin_label}.json"
    return json.loads(p.read_text())


def load_correctness(predictions_dir: Path, condition: str, bin_label: str) -> list:
    """Return per-sample correctness ordered positionally — index in this list
    corresponds to the story_idx in the multi-entry eval JSON."""
    me_dir = f"{condition}_me"
    p = predictions_dir / me_dir / f"predictions_{bin_label}.json"
    if not p.exists():
        return None
    preds = json.loads(p.read_text())
    return [int(bool(p.get("correct"))) for p in preds]


def compute_sample_qr_signature(doc_scores: dict, detection_sample: dict,
                                  topk_heads: list) -> float:
    """For one sample, return scalar = sum over (head in topk_heads) of
    sum over (gold docs) of doc_score[head]."""
    qid = detection_sample["idx"]
    gt_doc_ids = set(detection_sample["gt_docs"])
    per_doc = doc_scores[qid]
    if len(gt_doc_ids) == 0:
        return 0.0
    total = 0.0
    for doc_id in gt_doc_ids:
        if doc_id not in per_doc:
            continue
        tensor = per_doc[doc_id]
        for hid in topk_heads:
            l, h = head_id_to_indices(hid)
            total += float(tensor[l, h].item())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="composable_cot/retrieval_head_analysis/results")
    ap.add_argument("--data-dir",
                    default="composable_cot/retrieval_head_analysis/data")
    ap.add_argument("--predictions-dir",
                    default="composable_cot/BABIlong/results")
    ap.add_argument("--output-dir",
                    default="composable_cot/retrieval_head_analysis/analysis")
    ap.add_argument("--top-k", type=int, default=16,
                    help="K = number of top heads (from lora_base reference) "
                         "to aggregate into the per-sample signature")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    predictions_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_SCIPY:
        print("ERROR: scipy required for correlation. pip install scipy.")
        return

    print(f"Per-sample QR-vs-correctness correlation (K={args.top_k})")
    print(f"  reference (top-K heads from): {REFERENCE}")
    print()

    rows = []
    for b in BINS:
        try:
            topk_heads = reference_topk_heads(results_dir, b, args.top_k)
        except FileNotFoundError:
            print(f"  [{b}] missing reference head_scores")
            continue
        try:
            detset = load_detection_set(data_dir, b)
        except FileNotFoundError:
            print(f"  [{b}] missing detection set")
            continue

        for cond in CONDITIONS:
            try:
                doc_scores = load_doc_scores_pt(results_dir, cond, b)
            except FileNotFoundError:
                print(f"  [{b}/{cond}] missing doc_scores_per_head.pt — skip")
                continue

            corr_labels_all = load_correctness(predictions_dir, cond, b)
            if corr_labels_all is None:
                print(f"  [{b}/{cond}] missing predictions — skip")
                continue

            # Build per-sample QR signature + correctness, paired by story_idx.
            qr_signatures = []
            correctness = []
            for sample in detset:
                story_idx = sample["story_idx"]
                if story_idx >= len(corr_labels_all):
                    continue
                # QR signature = sum of top-K head scores across gold docs.
                sig = compute_sample_qr_signature(doc_scores, sample, topk_heads)
                qr_signatures.append(sig)
                correctness.append(corr_labels_all[story_idx])

            n = len(qr_signatures)
            if n < 5 or len(set(correctness)) < 2:
                # Not enough variance for correlation.
                r_pb, p_pb = float("nan"), float("nan")
                r_sp, p_sp = float("nan"), float("nan")
            else:
                r_pb, p_pb = pointbiserialr(correctness, qr_signatures)
                r_sp, p_sp = spearmanr(qr_signatures, correctness)

            rows.append({
                "bin": b,
                "condition": cond,
                "k": args.top_k,
                "n_samples": n,
                "n_correct": int(np.sum(correctness)),
                "mean_qr_sig": float(np.mean(qr_signatures)),
                "r_pointbiserial": r_pb,
                "p_pointbiserial": p_pb,
                "r_spearman": r_sp,
                "p_spearman": p_sp,
            })

            print(f"  [{b:5s} {cond:<20s}]  n={n:2d}  correct={int(np.sum(correctness)):2d}/{n:2d}  "
                  f"r_pb={r_pb:+.3f} (p={p_pb:.3f})  r_sp={r_sp:+.3f}")

    csv_path = out_dir / "per_sample_correlation.csv"
    with csv_path.open("w") as f:
        f.write("bin,condition,k,n_samples,n_correct,mean_qr_sig,"
                "r_pointbiserial,p_pointbiserial,r_spearman,p_spearman\n")
        for r in rows:
            f.write(f"{r['bin']},{r['condition']},{r['k']},{r['n_samples']},"
                    f"{r['n_correct']},{r['mean_qr_sig']:.6f},"
                    f"{r['r_pointbiserial']:.4f},{r['p_pointbiserial']:.6e},"
                    f"{r['r_spearman']:.4f},{r['p_spearman']:.6e}\n")
    print(f"\nWrote {csv_path}")

    # Plot per-condition r curves over bins.
    if HAS_MPL and rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        for cond in CONDITIONS:
            xs, ys = [], []
            for b in BINS:
                r = next((r for r in rows if r["condition"] == cond and r["bin"] == b),
                         None)
                if r is None or np.isnan(r["r_pointbiserial"]):
                    continue
                xs.append(b)
                ys.append(r["r_pointbiserial"])
            ax.plot(xs, ys, marker="o", label=cond)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("bin")
        ax.set_ylabel(f"point-biserial r (top-{args.top_k} QR signature vs correctness)")
        ax.set_title("Per-sample QR-vs-correctness correlation per condition × bin")
        ax.legend()
        fig.tight_layout()
        p = out_dir / "figures" / "correlation_per_bin.png"
        p.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(p, dpi=120)
        plt.close(fig)
        print(f"Wrote {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
