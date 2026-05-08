"""
Present QR-head distributions in the same style as the QRHead paper:
layer × head heatmaps so the same head sits at the same grid position
across models. Each panel stands alone — no model is treated as the
reference the others are scored against.

4 models:
  zero_shot       Qwen2.5-7B-Instruct (paper, BEIR-NQ-detected)
  lora_base       LoRA only
  y2_base         LoRA + YaRN (training factor 2, eval factor 4)
  y2_rpe_cur_L16k LoRA + RPE curriculum L=16K + YaRN

Outputs:
  analysis/distributions/per_model_summary.csv
  analysis/distributions/per_model_topk_heads.csv
  analysis/distributions/figures/H1_layer_head_heatmap.png   QR score, common scale
  analysis/distributions/figures/H2_topk_mask.png            top-16 mask per model
  analysis/distributions/figures/H3_fixed_order_bars.png     all 784 heads at fixed x
  analysis/distributions/figures/H4_per_bin_heatmap.png      one trained model × bins
  analysis/distributions/figures/H5_histogram.png            score histograms
  analysis/distributions/figures/H6_per_bin_metrics.png      counts/scores/acc per bin
"""
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("composable_cot/retrieval_head_analysis")
PRED = Path("composable_cot/BABIlong/results")
OUT = ROOT / "analysis" / "distributions"
FIG = OUT / "figures"

BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
TRAINED = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
ALL_MODELS = ["zero_shot"] + TRAINED
SHORT = {"zero_shot": "zero-shot", "lora_base": "lora",
         "y2_base": "yarn", "y2_rpe_cur_L16k": "rpe+yarn"}
COLOR = {"zero_shot": "#444444", "lora_base": "#1f77b4",
         "y2_base": "#ff7f0e", "y2_rpe_cur_L16k": "#2ca02c"}
N_LAYERS = 28
N_HEADS = 28
K = 16


def load_ranking(model: str, bin_label: str | None) -> list[tuple[str, float]]:
    if model == "zero_shot":
        p = ROOT / "results" / "nq_TRAIN.json"
    else:
        p = ROOT / "results" / model / f"{bin_label}_head_scores.json"
    return [(h, float(s)) for h, s in json.loads(p.read_text())]


def to_grid(ranking: list[tuple[str, float]]) -> np.ndarray:
    """Return [n_layers, n_heads] array of QR scores indexed by (layer, head)."""
    g = np.full((N_LAYERS, N_HEADS), np.nan, dtype=float)
    for hid, s in ranking:
        l, h = map(int, hid.split("-"))
        g[l, h] = s
    return g


def load_overall_accuracy(model: str, bin_label: str) -> float | None:
    p = PRED / f"{model}_me" / f"predictions_{bin_label}.json"
    if not p.exists():
        return None
    preds = json.loads(p.read_text())
    if not preds:
        return None
    return sum(1 for r in preds if r.get("correct")) / len(preds)


def primary_ranking(model: str) -> list[tuple[str, float]]:
    return load_ranking(model, "128k") if model != "zero_shot" else load_ranking(model, None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    primary = {m: primary_ranking(m) for m in ALL_MODELS}
    grids   = {m: to_grid(primary[m]) for m in ALL_MODELS}

    # ============================================================
    # CSVs
    # ============================================================
    thresholds = [0.05, 0.10, 0.15, 0.20]
    summary_csv = OUT / "per_model_summary.csv"
    with summary_csv.open("w") as f:
        cols = (["model", "ranking_source", "n_heads", "mean", "median", "std",
                 "max", "min", "mean_top16", "max_top16"]
                + [f"n_above_{t}" for t in thresholds])
        f.write(",".join(cols) + "\n")
        for m in ALL_MODELS:
            scores = np.array([s for _, s in primary[m]])
            top16 = sorted(scores, reverse=True)[:K]
            src = "BEIR-NQ" if m == "zero_shot" else "BABILong-128k"
            row = [m, src, len(scores),
                   f"{scores.mean():.4f}", f"{np.median(scores):.4f}",
                   f"{scores.std():.4f}", f"{scores.max():.4f}",
                   f"{scores.min():.4f}",
                   f"{np.mean(top16):.4f}", f"{np.max(top16):.4f}"]
            for t in thresholds:
                row.append(str(int((scores >= t).sum())))
            f.write(",".join(map(str, row)) + "\n")
    print(f"Wrote {summary_csv}")

    topk_csv = OUT / "per_model_topk_heads.csv"
    with topk_csv.open("w") as f:
        f.write("model,rank,layer,head_idx,score\n")
        for m in ALL_MODELS:
            r = sorted(primary[m], key=lambda x: -x[1])[:K]
            for i, (hid, s) in enumerate(r, 1):
                layer, hidx = hid.split("-")
                f.write(f"{m},{i},{layer},{hidx},{s:.4f}\n")
    print(f"Wrote {topk_csv}")

    # ============================================================
    # FIGURE H1 — Layer × Head heatmap, one panel per model.
    # SAME (layer, head) position across panels — directly visually comparable.
    # Common color scale so colors mean the same thing across models.
    # ============================================================
    vmax = max(grids[m].max() for m in ALL_MODELS)
    vmin = min(grids[m].min() for m in ALL_MODELS)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), sharey=True)
    for ax, m in zip(axes, ALL_MODELS):
        im = ax.imshow(grids[m], cmap="viridis", vmin=vmin, vmax=vmax,
                       aspect="auto", origin="lower")
        # mark this model's top-16 with red squares
        r = sorted(primary[m], key=lambda x: -x[1])[:K]
        for hid, _ in r:
            l, h = map(int, hid.split("-"))
            ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1,
                                        fill=False, edgecolor="red", linewidth=1.4))
        src = "BEIR-NQ" if m == "zero_shot" else "BABILong @ 128k"
        ax.set_title(f"{SHORT[m]}\n({src})", fontsize=12)
        ax.set_xlabel("head index (0–27)")
        ax.set_xticks(range(0, N_HEADS, 4))
    axes[0].set_ylabel("layer (0–27)")
    axes[0].set_yticks(range(0, N_LAYERS, 4))
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                 label="calibrated QR score")
    fig.suptitle("H1 — QR score per (layer, head). Red boxes = each model's top-16.",
                 fontsize=13)
    p = FIG / "H1_layer_head_heatmap.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE H2 — top-16 binary mask per model in layer×head space.
    # Tells you "which heads is each model using" at a glance.
    # ============================================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), sharey=True)
    for ax, m in zip(axes, ALL_MODELS):
        mask = np.zeros_like(grids[m])
        r = sorted(primary[m], key=lambda x: -x[1])[:K]
        for hid, _ in r:
            l, h = map(int, hid.split("-"))
            mask[l, h] = 1
        ax.imshow(mask, cmap="Reds", vmin=0, vmax=1, aspect="auto", origin="lower")
        # annotate with rank
        for rank, (hid, s) in enumerate(r, 1):
            l, h = map(int, hid.split("-"))
            ax.text(h, l, str(rank), ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")
        src = "BEIR-NQ" if m == "zero_shot" else "BABILong @ 128k"
        ax.set_title(f"{SHORT[m]} top-16\n({src})", fontsize=12)
        ax.set_xlabel("head index")
        ax.set_xticks(range(0, N_HEADS, 4))
    axes[0].set_ylabel("layer")
    axes[0].set_yticks(range(0, N_LAYERS, 4))
    fig.suptitle("H2 — Top-16 head positions per model (numbers = rank within top-16)",
                 fontsize=13)
    p = FIG / "H2_topk_mask.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE H3 — All 784 heads at fixed (layer, head) position, all
    # 4 models overlaid. x = head index 0..783 in (layer*28+head) order.
    # Lets you see at the same x where each model has a peak.
    # ============================================================
    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)
    x = np.arange(N_LAYERS * N_HEADS)
    for ax, m in zip(axes, ALL_MODELS):
        flat = grids[m].flatten()
        ax.bar(x, flat, color=COLOR[m], width=1.0, edgecolor="none")
        # mark top-16 positions with vertical red ticks
        r = sorted(primary[m], key=lambda x: -x[1])[:K]
        for hid, s in r:
            l, h = map(int, hid.split("-"))
            ax.plot([l * N_HEADS + h], [s + 0.02], marker="v",
                    color="red", markersize=5)
        # layer separators
        for li in range(1, N_LAYERS):
            ax.axvline(li * N_HEADS - 0.5, color="grey",
                       linewidth=0.3, alpha=0.5)
        src = "BEIR-NQ" if m == "zero_shot" else "BABILong @ 128k"
        ax.set_title(f"{SHORT[m]} ({src})", fontsize=11, loc="left")
        ax.set_ylabel("QR score")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("head index in (layer × 28 + head_idx) order — "
                        "vertical lines = layer boundaries")
    axes[-1].set_xticks(np.arange(0, N_LAYERS * N_HEADS, N_HEADS * 2))
    axes[-1].set_xticklabels([f"L{li}" for li in range(0, N_LAYERS, 2)])
    fig.suptitle("H3 — All 784 heads at fixed (layer, head) position. "
                 "Red ▼ = top-16 for that model.",
                 fontsize=13)
    fig.tight_layout()
    p = FIG / "H3_fixed_order_bars.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE H4 — per-bin heatmap, all 3 trained models × 4 bins.
    # Shows how the QR-head map shifts with context length within each model.
    # ============================================================
    bins_show = ["8k", "32k", "64k", "128k"]
    fig, axes = plt.subplots(len(TRAINED), len(bins_show),
                              figsize=(20, 12), sharex=True, sharey=True)
    # common color scale across this figure's panels
    grids_all = {(c, b): to_grid(load_ranking(c, b))
                 for c in TRAINED for b in bins_show}
    vmax_pb = max(g.max() for g in grids_all.values())
    vmin_pb = min(g.min() for g in grids_all.values())
    for i, cond in enumerate(TRAINED):
        for j, b in enumerate(bins_show):
            ax = axes[i, j]
            g = grids_all[(cond, b)]
            im = ax.imshow(g, cmap="viridis", vmin=vmin_pb, vmax=vmax_pb,
                            aspect="auto", origin="lower")
            r = sorted(load_ranking(cond, b), key=lambda x: -x[1])[:K]
            for hid, _ in r:
                l, h = map(int, hid.split("-"))
                ax.add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1,
                                            fill=False, edgecolor="red",
                                            linewidth=1.0))
            if i == 0:
                ax.set_title(f"@ {b}", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{SHORT[cond]}\nlayer", fontsize=11)
            if i == len(TRAINED) - 1:
                ax.set_xlabel("head index")
                ax.set_xticks(range(0, N_HEADS, 4))
            ax.set_yticks(range(0, N_LAYERS, 4))
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                 label="calibrated QR score")
    fig.suptitle("H4 — Per-bin QR-score map per trained model. "
                 "Red boxes = top-16 at that bin.",
                 fontsize=13)
    p = FIG / "H4_per_bin_heatmap.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE H5 — score histograms, one per model
    # ============================================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.0), sharex=True, sharey=True)
    bins_h = np.linspace(-0.1, 0.55, 50)
    for ax, m in zip(axes, ALL_MODELS):
        scores = np.array([s for _, s in primary[m]])
        ax.hist(scores, bins=bins_h, color=COLOR[m],
                edgecolor="black", linewidth=0.3)
        for t in [0.10, 0.20]:
            ax.axvline(t, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
            n = int((scores >= t).sum())
            ax.text(t + 0.005, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] else 100,
                    f"n≥{t:.2f}: {n}", fontsize=8, color="red")
        src = "BEIR-NQ" if m == "zero_shot" else "BABILong @ 128k"
        ax.set_title(f"{SHORT[m]} ({src})", fontsize=11)
        ax.set_xlabel("calibrated QR score")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("# heads")
    fig.suptitle("H5 — Per-model score histogram (all 784 heads)", fontsize=13)
    fig.tight_layout()
    p = FIG / "H5_histogram.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE H6 — per-bin metrics for trained models (counts / mean / acc)
    # ============================================================
    n_above_per = {cond: [] for cond in TRAINED}
    mean_top16_per = {cond: [] for cond in TRAINED}
    acc_per = {cond: [] for cond in TRAINED}
    for cond in TRAINED:
        for b in BINS:
            r = load_ranking(cond, b)
            scores = np.array([s for _, s in r])
            n_above_per[cond].append(int((scores >= 0.10).sum()))
            mean_top16_per[cond].append(
                float(np.mean(sorted(scores, reverse=True)[:K])))
            acc_per[cond].append(load_overall_accuracy(cond, b))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("# heads ≥ 0.10",        n_above_per,    "count"),
        ("mean top-16 QR score",  mean_top16_per, "score"),
        ("BABILong-QA3 accuracy", acc_per,        "accuracy"),
    ]
    x = np.arange(len(BINS))
    width = 0.27
    offsets = {"lora_base": -width, "y2_base": 0, "y2_rpe_cur_L16k": width}
    for ax, (title, data, ylabel) in zip(axes, metrics):
        for cond in TRAINED:
            ys = [v if v is not None else 0 for v in data[cond]]
            ax.bar(x + offsets[cond], ys, width,
                   color=COLOR[cond], label=SHORT[cond])
        ax.set_xticks(x)
        ax.set_xticklabels(BINS, fontsize=9)
        ax.set_xlabel("BABILong context bin")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("H6 — Per-bin: # heads, mean top-16 score, downstream accuracy",
                 fontsize=13)
    fig.tight_layout()
    p = FIG / "H6_per_bin_metrics.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE H7 — Top-16 overlap map. For each (layer, head) cell,
    # count how many of the 4 models have it in their top-16 (0..4).
    # ============================================================
    top16_sets = {}
    top16_score_lookup = {}  # (model, hid) -> score
    for m in ALL_MODELS:
        r = sorted(primary[m], key=lambda x: -x[1])[:K]
        top16_sets[m] = set(hid for hid, _ in r)
        for hid, s in primary[m]:
            top16_score_lookup[(m, hid)] = s

    overlap_count = np.zeros((N_LAYERS, N_HEADS), dtype=int)
    for m in ALL_MODELS:
        for hid in top16_sets[m]:
            l, h = map(int, hid.split("-"))
            overlap_count[l, h] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("YlOrRd", 5)  # 0..4 discrete
    im = ax.imshow(overlap_count, cmap=cmap, vmin=0, vmax=4,
                    aspect="auto", origin="lower")
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            c = overlap_count[l, h]
            if c > 0:
                ax.text(h, l, str(c), ha="center", va="center",
                        color="black" if c < 3 else "white",
                        fontsize=8, fontweight="bold")
    ax.set_xlabel("head index")
    ax.set_ylabel("layer")
    ax.set_xticks(range(0, N_HEADS, 2))
    ax.set_yticks(range(0, N_LAYERS, 2))
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.set_label("# models with this head in top-16 (out of 4)")
    n_in_any = int((overlap_count > 0).sum())
    n_in_all = int((overlap_count == 4).sum())
    n_in_3   = int((overlap_count == 3).sum())
    ax.set_title(f"H7 — Top-16 overlap across 4 models\n"
                  f"{n_in_any} unique heads in any top-16 | "
                  f"{n_in_all} in all 4 | "
                  f"{n_in_3} in exactly 3",
                  fontsize=12)
    fig.tight_layout()
    p = FIG / "H7_topk_overlap_count.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # ============================================================
    # FIGURE H8 — For each head in any model's top-16 (the "union top-16"),
    # show the mean of its QR score across the 4 models on the same
    # layer × head grid. Cells outside the union are greyed out.
    # ============================================================
    union_heads = set()
    for m in ALL_MODELS:
        union_heads |= top16_sets[m]

    mean_grid = np.full((N_LAYERS, N_HEADS), np.nan, dtype=float)
    rows = []  # for CSV / table
    for hid in union_heads:
        l, h = map(int, hid.split("-"))
        scores = [top16_score_lookup[(m, hid)] for m in ALL_MODELS]
        mean_grid[l, h] = float(np.mean(scores))
        rows.append({
            "head": hid, "layer": l, "head_idx": h,
            "n_models_top16": int(overlap_count[l, h]),
            "mean_score": float(np.mean(scores)),
            "zero_shot": top16_score_lookup[("zero_shot", hid)],
            "lora": top16_score_lookup[("lora_base", hid)],
            "yarn": top16_score_lookup[("y2_base", hid)],
            "rpe_yarn": top16_score_lookup[("y2_rpe_cur_L16k", hid)],
        })

    fig, ax = plt.subplots(figsize=(9, 8))
    masked = np.ma.masked_invalid(mean_grid)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgrey")
    im = ax.imshow(masked, cmap=cmap, aspect="auto", origin="lower")
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            v = mean_grid[l, h]
            if not np.isnan(v):
                color = "white" if v < 0.18 else "black"
                ax.text(h, l, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=6)
    ax.set_xlabel("head index")
    ax.set_ylabel("layer")
    ax.set_xticks(range(0, N_HEADS, 2))
    ax.set_yticks(range(0, N_LAYERS, 2))
    fig.colorbar(im, ax=ax, label="mean QR score across 4 models")
    ax.set_title(f"H8 — Mean QR score across the 4 models, "
                  f"on union of top-16 ({len(union_heads)} heads)",
                  fontsize=12)
    fig.tight_layout()
    p = FIG / "H8_union_top16_mean.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"Wrote {p}")

    # CSV: per-union-head per-model scores + mean + n_models_top16
    union_csv = OUT / "union_top16_per_model.csv"
    rows_sorted = sorted(rows, key=lambda r: (-r["n_models_top16"],
                                                -r["mean_score"]))
    with union_csv.open("w") as f:
        f.write("head,layer,head_idx,n_models_top16,mean_score,"
                "zero_shot,lora,yarn,rpe_yarn\n")
        for r in rows_sorted:
            f.write(f"{r['head']},{r['layer']},{r['head_idx']},"
                    f"{r['n_models_top16']},{r['mean_score']:.4f},"
                    f"{r['zero_shot']:.4f},{r['lora']:.4f},"
                    f"{r['yarn']:.4f},{r['rpe_yarn']:.4f}\n")
    print(f"Wrote {union_csv}")

    # ============================================================
    # Terminal summary
    # ============================================================
    print("\n=== Per-model summary at primary ranking ===")
    print(f"  {'model':<10s}  {'src':<14s}  {'mean':>7s}  {'max':>7s}  "
          f"{'top16_mean':>10s}  {'top16_max':>9s}  "
          f"{'>=0.10':>6s}  {'>=0.20':>6s}")
    for m in ALL_MODELS:
        scores = np.array([s for _, s in primary[m]])
        top16 = sorted(scores, reverse=True)[:K]
        src = "BEIR-NQ" if m == "zero_shot" else "BABILong-128k"
        print(f"  {SHORT[m]:<10s}  {src:<14s}  {scores.mean():>7.4f}  "
              f"{scores.max():>7.4f}  "
              f"{np.mean(top16):>10.4f}  {np.max(top16):>9.4f}  "
              f"{int((scores >= 0.10).sum()):>6d}  "
              f"{int((scores >= 0.20).sum()):>6d}")

    print("\n=== 128k overlap (top-16) among trained ===")
    per_model_128 = {cond: set([h for h, _ in
                                sorted(load_ranking(cond, '128k'),
                                       key=lambda x: -x[1])[:K]])
                     for cond in TRAINED}
    for ma, mb in combinations(TRAINED, 2):
        n = len(per_model_128[ma] & per_model_128[mb])
        print(f"  {SHORT[ma]:<10s} ∩ {SHORT[mb]:<10s} = {n:>2d} / 16")


if __name__ == "__main__":
    main()
