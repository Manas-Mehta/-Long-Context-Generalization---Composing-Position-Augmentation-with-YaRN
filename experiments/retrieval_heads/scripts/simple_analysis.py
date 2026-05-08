"""
Simple analysis answering prof's 3 questions.

K = 16 only (paper standard for sub-10B models). No invented thresholds.

Q1. Differences in # of QR heads / distribution / overlap?
Q2. Are scores larger/smaller/unchanged?
Q3. Does the ranking correspond to BABILong downstream?

4 models:
  zero_shot       Qwen2.5-7B-Instruct (paper, BEIR-NQ-detected, nq_TRAIN.json)
  lora_base       LoRA only
  y2_base         LoRA + YaRN
  y2_rpe_cur_L16k LoRA + RPE + YaRN

Domain note: zero_shot scores are BEIR-NQ-detected; trained models are
BABILong-detected. Length transfer holds (paper §6.2); cross-domain transfer is
a working assumption (Wu et al. "intrinsic to pretraining").

Outputs:
  analysis/simple_results.csv          per-(model, bin) summary
  analysis/figures/F1_top16_mean.png   bar chart, mean top-16 score per (model, bin)
  analysis/figures/F2_overlap.png      pairwise top-16 overlap matrix at 128k
  analysis/figures/F3_ranking.png      mean top-16 score vs BABILong accuracy
"""
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("experiments/retrieval_heads")
PRED = Path("experiments/babilong/results")
OUT = ROOT / "analysis"
FIG = OUT / "figures"

BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
TRAINED = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
SHORT = {"zero_shot": "zero_shot", "lora_base": "lora",
         "y2_base": "yarn", "y2_rpe_cur_L16k": "rpe+yarn"}
COLOR = {"zero_shot": "#777777", "lora_base": "C0",
         "y2_base": "C1", "y2_rpe_cur_L16k": "C2"}
K = 16


def load_ranking(model: str, bin_label: str | None) -> list[tuple[str, float]]:
    if model == "zero_shot":
        p = ROOT / "results" / "nq_TRAIN.json"
    else:
        p = ROOT / "results" / model / f"{bin_label}_head_scores.json"
    return [(h, s) for h, s in json.loads(p.read_text())]


def load_accuracy(model: str, bin_label: str, story_id_order: list[int]) -> float | None:
    """Restrict accuracy to the 60 detection-set stories (1-indexed story_idx)."""
    p = PRED / f"{model}_me" / f"predictions_{bin_label}.json"
    if not p.exists():
        return None
    preds = json.loads(p.read_text())
    # story_idx is 1-based; preds is 0-indexed positional in same multi-entry order
    hits = []
    for sidx in story_id_order:
        i = sidx - 1
        if 0 <= i < len(preds):
            hits.append(int(bool(preds[i].get("correct"))))
    return sum(hits) / len(hits) if hits else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # -------- zero_shot — full ranking + top-16 stats --------
    zs_ranking = load_ranking("zero_shot", None)
    zs_top16 = [h for h, _ in zs_ranking[:K]]
    zs_top16_set = set(zs_top16)
    zs_scores = dict(zs_ranking)
    zs_mean = float(np.mean([zs_scores[h] for h in zs_top16]))
    zs_max = max(zs_scores[h] for h in zs_top16)
    # Detection threshold: score at zero-shot's rank-16 head.
    # Origin: this is the score level at which the QRHead paper defines
    # the "QR head set" of zero-shot Qwen2.5-7B (K=16, sub-10B model).
    # Each model independently reports # heads with score >= this threshold.
    THRESHOLD = zs_ranking[K - 1][1]
    print(f"Detection threshold (= zero_shot's rank-{K} score): {THRESHOLD:.4f}\n")

    # -------- per-(model, bin) summary --------
    rows = []
    # zero_shot row (single, no bin)
    n_above_zs = sum(1 for _, s in zs_ranking if s >= THRESHOLD)
    rows.append({
        "model": "zero_shot", "bin": "(BEIR-NQ)",
        "top16": zs_top16,
        "mean_top16": zs_mean,
        "max_top16": zs_max,
        "n_above_threshold": n_above_zs,
        "babilong_acc": None,
    })
    for cond in TRAINED:
        for b in BINS:
            ranking = load_ranking(cond, b)
            top16 = [h for h, _ in ranking[:K]]
            scores = dict(ranking)
            top16_scores = [scores[h] for h in top16]
            mean_top16 = float(np.mean(top16_scores))
            max_top16 = float(np.max(top16_scores))
            n_above = sum(1 for _, s in ranking if s >= THRESHOLD)
            # use the meta to restrict accuracy to the 60 detection-set stories
            meta = json.loads((ROOT / "results" / cond / f"{b}_meta.json").read_text())
            acc = load_accuracy(cond, b, meta["story_id_order"])
            rows.append({
                "model": cond, "bin": b,
                "top16": top16,
                "mean_top16": mean_top16,
                "max_top16": max_top16,
                "n_above_threshold": n_above,
                "babilong_acc": acc,
            })

    # -------- write CSV --------
    csv = OUT / "simple_results.csv"
    with csv.open("w") as f:
        f.write("model,bin,mean_top16,max_top16,n_above_threshold,babilong_acc,top16_heads\n")
        for r in rows:
            acc = f"{r['babilong_acc']:.4f}" if r["babilong_acc"] is not None else ""
            f.write(f"{r['model']},{r['bin']},{r['mean_top16']:.4f},"
                    f"{r['max_top16']:.4f},{r['n_above_threshold']},{acc},"
                    f"{';'.join(r['top16'])}\n")
    print(f"Wrote {csv}")

    # -------- pairwise top-16 overlaps per bin --------
    overlap_csv = OUT / "simple_overlaps.csv"
    overlap_rows = []
    all_models = ["zero_shot"] + TRAINED
    for b in BINS:
        # zero_shot's top-16 doesn't depend on bin; trained's does
        per_model_top16 = {"zero_shot": zs_top16_set}
        for cond in TRAINED:
            per_model_top16[cond] = set([h for h, _ in load_ranking(cond, b)[:K]])
        for ma, mb in combinations(all_models, 2):
            sa, sb = per_model_top16[ma], per_model_top16[mb]
            overlap_rows.append({
                "bin": b, "a": ma, "b": mb,
                "n_overlap": len(sa & sb),
                "jaccard": len(sa & sb) / len(sa | sb),
            })
    with overlap_csv.open("w") as f:
        f.write("bin,a,b,n_overlap,jaccard\n")
        for r in overlap_rows:
            f.write(f"{r['bin']},{r['a']},{r['b']},{r['n_overlap']},{r['jaccard']:.4f}\n")
    print(f"Wrote {overlap_csv}")

    # ============================================================
    # FIGURE 1 — bar chart, mean top-16 score per (model, bin)
    # ============================================================
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(BINS))
    width = 0.27
    offsets = {"lora_base": -width, "y2_base": 0, "y2_rpe_cur_L16k": width}
    for cond in TRAINED:
        ys = [next(r["mean_top16"] for r in rows if r["model"] == cond and r["bin"] == b)
              for b in BINS]
        ax.bar(x + offsets[cond], ys, width, label=SHORT[cond], color=COLOR[cond])
    # zero-shot reference line
    ax.axhline(zs_mean, color=COLOR["zero_shot"], linestyle="--", linewidth=1.5,
                label=f"zero_shot (BEIR-NQ): {zs_mean:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_xlabel("BABILong context bin")
    ax.set_ylabel("Mean calibrated QR score on top-16 heads")
    ax.set_title("Q1b/Q2 — Mean top-16 QR score across the 4 models")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "F1_top16_mean.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE 2 — pairwise overlap matrix at 128k (and 32k for context)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, b in zip(axes, ["32k", "128k"]):
        per_model_top16 = {"zero_shot": zs_top16_set}
        for cond in TRAINED:
            per_model_top16[cond] = set([h for h, _ in load_ranking(cond, b)[:K]])
        mat = np.zeros((len(all_models), len(all_models)), dtype=int)
        for i, ma in enumerate(all_models):
            for j, mb in enumerate(all_models):
                mat[i, j] = len(per_model_top16[ma] & per_model_top16[mb])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=K)
        ax.set_xticks(range(len(all_models)))
        ax.set_yticks(range(len(all_models)))
        ax.set_xticklabels([SHORT[m] for m in all_models], rotation=20)
        ax.set_yticklabels([SHORT[m] for m in all_models])
        ax.set_title(f"Top-16 overlap @ {b}")
        for i in range(len(all_models)):
            for j in range(len(all_models)):
                color = "white" if mat[i, j] > K * 0.55 else "black"
                ax.text(j, i, mat[i, j], ha="center", va="center", color=color, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Q1c — # of overlapping heads in top-16 (K=16)")
    fig.tight_layout()
    p = FIG / "F2_overlap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE 3 — ranking correspondence (Q3)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    for cond in TRAINED:
        xs, ys = [], []
        for b in BINS:
            r = next(r for r in rows if r["model"] == cond and r["bin"] == b)
            if r["babilong_acc"] is not None:
                xs.append(r["babilong_acc"])
                ys.append(r["mean_top16"])
        ax.scatter(xs, ys, label=SHORT[cond], color=COLOR[cond], s=60)
        for r in rows:
            if r["model"] == cond and r["babilong_acc"] is not None:
                ax.annotate(r["bin"], (r["babilong_acc"], r["mean_top16"]),
                            fontsize=7, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("BABILong-QA3 accuracy on the 60 detection stories")
    ax.set_ylabel("Mean calibrated QR score on top-16")
    ax.set_title("Q3 — Does mean top-16 QR score track BABILong accuracy?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "F3_ranking.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # -------- terminal summary tables --------
    print(f"\n=== Q1a/Q1b/Q2 — # heads above threshold ({THRESHOLD:.3f}), "
          f"top-16 mean/max ===")
    print(f"  {'model':<12s} {'bin':>10s}  {'mean':>7s}  {'max':>7s}  "
          f"{'n>=thr':>7s}  {'BABILong_acc':>12s}")
    for r in rows:
        acc = f"{r['babilong_acc']:.3f}" if r["babilong_acc"] is not None else "    -"
        print(f"  {SHORT.get(r['model'], r['model']):<12s} {r['bin']:>10s}  "
              f"{r['mean_top16']:>7.4f}  {r['max_top16']:>7.4f}  "
              f"{r['n_above_threshold']:>7d}  {acc:>12s}")

    print("\n=== Q1c — pairwise overlap counts at 128K ===")
    n128 = {(r['a'], r['b']): r['n_overlap'] for r in overlap_rows if r['bin'] == "128k"}
    print(f"  {'pair':<35s}  n_overlap")
    for r in overlap_rows:
        if r['bin'] == "128k":
            print(f"  {SHORT[r['a']]} vs {SHORT[r['b']]:<25s} {r['n_overlap']:>3d} / 16")

    print("\n=== Q3 — ranking @ 128k (3 trained models) ===")
    by_acc = sorted([r for r in rows if r['model'] in TRAINED and r['bin'] == "128k"],
                     key=lambda r: -r["babilong_acc"])
    by_score = sorted([r for r in rows if r['model'] in TRAINED and r['bin'] == "128k"],
                       key=lambda r: -r["mean_top16"])
    print(f"  by BABILong acc:   " + " > ".join(f"{SHORT[r['model']]} ({r['babilong_acc']:.3f})" for r in by_acc))
    print(f"  by mean top-16:    " + " > ".join(f"{SHORT[r['model']]} ({r['mean_top16']:.3f})" for r in by_score))


if __name__ == "__main__":
    main()
