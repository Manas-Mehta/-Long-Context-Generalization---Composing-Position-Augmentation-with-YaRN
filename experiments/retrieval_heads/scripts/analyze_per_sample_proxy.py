"""
Per-sample QR-vs-correctness correlation — PROXY version (no detection-set JSON
needed, runs locally with only the .pt tensors + meta.json + predictions_*.json).

The proper version (analyze_per_sample.py) needs the detection-set JSONs
(detection_set_{bin}.json) which mark which doc_ids are gold for each story.
Those live on HPC; we have local .pt tensors but not the JSONs.

This proxy uses the max QR score across ALL docs for the top-K heads
(reference = lora_base). Intuition: "did the QR heads activate strongly on
SOMETHING in this sample's haystack?" Without gold annotation we can't say
"on the right thing", but max-activation-correlation is still informative.

Outputs:
  analysis/per_sample_correlation_proxy.csv
  analysis/figures/correlation_per_bin_proxy.png
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
REFERENCE = "lora_base"
TOP_K = 16

ROOT = Path("experiments/retrieval_heads")
PRED_DIR = Path("experiments/babilong/results")
OUT_DIR = ROOT / "analysis"


def load_head_scores(cond, b):
    p = ROOT / "results" / cond / f"{b}_head_scores.json"
    return {h: s for h, s in json.loads(p.read_text())}


def load_meta(cond, b):
    return json.loads((ROOT / "results" / cond / f"{b}_meta.json").read_text())


def load_pt(cond, b):
    return torch.load(ROOT / "results" / cond / f"{b}_doc_scores_per_head.pt",
                      map_location="cpu", weights_only=False)


def topk_heads(b, k):
    s = load_head_scores(REFERENCE, b)
    return [h for h, _ in sorted(s.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def parse_head(hid):
    l, h = hid.split("-")
    return int(l), int(h)


def load_correctness(cond, b):
    p = PRED_DIR / f"{cond}_me" / f"predictions_{b}.json"
    if not p.exists():
        return None
    preds = json.loads(p.read_text())
    return [int(bool(pred.get("correct"))) for pred in preds]


def main():
    rows = []
    print(f"Per-sample QR-vs-correctness PROXY (max across all docs, K={TOP_K})\n")
    for b in BINS:
        try:
            tk = topk_heads(b, TOP_K)
        except FileNotFoundError:
            continue
        head_idx = [parse_head(h) for h in tk]
        for cond in CONDITIONS:
            try:
                doc_scores = load_pt(cond, b)
                meta = load_meta(cond, b)
            except FileNotFoundError:
                continue
            corr_all = load_correctness(cond, b)
            if corr_all is None:
                print(f"  [{b} {cond}] no predictions")
                continue
            sigs, labels = [], []
            for story_idx, qid in enumerate(meta["story_id_order"]):
                key = f"story{qid}_{b}"
                if key not in doc_scores:
                    continue
                # Build (n_docs, K) matrix of top-K head scores per doc
                docs = doc_scores[key]
                if len(docs) == 0:
                    continue
                doc_mat = []
                for did, tensor in docs.items():
                    vals = [float(tensor[l, h].item()) for l, h in head_idx]
                    doc_mat.append(vals)
                doc_mat = np.array(doc_mat)  # (n_docs, K)
                # PROXY signature: sum over top-K heads of MAX across docs
                sig = float(np.sum(np.max(doc_mat, axis=0)))
                sigs.append(sig)
                if qid < len(corr_all):
                    labels.append(corr_all[qid])
                else:
                    labels.append(0)
            n = len(sigs)
            if n < 5 or len(set(labels)) < 2:
                r_pb, p_pb, r_sp = float("nan"), float("nan"), float("nan")
            else:
                r_pb, p_pb = pointbiserialr(labels, sigs)
                r_sp, _ = spearmanr(sigs, labels)
            rows.append({"bin": b, "condition": cond, "n": n,
                         "n_correct": int(sum(labels)),
                         "mean_sig": float(np.mean(sigs)) if sigs else float("nan"),
                         "r_pb": r_pb, "p_pb": p_pb, "r_sp": r_sp})
            print(f"  [{b:>5s} {cond:<20s}]  n={n:2d}  correct={int(sum(labels)):2d}/{n:2d}  "
                  f"r_pb={r_pb:+.3f} (p={p_pb:.3f})  r_sp={r_sp:+.3f}")

    csv = OUT_DIR / "per_sample_correlation_proxy.csv"
    with csv.open("w") as f:
        f.write("bin,condition,n,n_correct,mean_sig,r_pointbiserial,p_pointbiserial,r_spearman\n")
        for r in rows:
            f.write(f"{r['bin']},{r['condition']},{r['n']},{r['n_correct']},"
                    f"{r['mean_sig']:.4f},{r['r_pb']:.4f},{r['p_pb']:.4e},{r['r_sp']:.4f}\n")
    print(f"\nWrote {csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for cond in CONDITIONS:
        xs, ys = [], []
        for b in BINS:
            r = next((r for r in rows if r["condition"] == cond and r["bin"] == b), None)
            if r and not np.isnan(r["r_pb"]):
                xs.append(b)
                ys.append(r["r_pb"])
        ax.plot(xs, ys, marker="o", label=cond)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("bin")
    ax.set_ylabel(f"point-biserial r (top-{TOP_K} max-doc QR proxy vs correctness)")
    ax.set_title("Per-sample QR-vs-correctness correlation (PROXY) per condition × bin")
    ax.legend()
    fig.tight_layout()
    fig_path = OUT_DIR / "figures" / "correlation_per_bin_proxy.png"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
