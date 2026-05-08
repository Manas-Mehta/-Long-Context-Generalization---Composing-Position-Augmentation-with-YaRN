"""
Phase 4 (analyses 4a, 4b, 4c, 4e) — head structure analysis.

Inputs (JSON only, no .pt needed):
  results/{condition}/{bin}_head_scores.json   (sorted [(layer-head, score), ...])
  results/{condition}/{bin}_meta.json          (n_samples, story_id_order, ...)

Outputs:
  analysis/head_overlap.csv          per-bin Jaccard between each trained
                                      condition and the lora_base reference,
                                      at K ∈ {8, 16, 32, 64}
  analysis/head_deltas_top16.csv     per-head Δscore vs lora_base for the top-16
                                      heads of each condition × bin
  analysis/distribution_ks.csv       KS-test p-values comparing each condition's
                                      full-head score distribution against lora_base
  analysis/figures/head_overlap_heatmap.png
  analysis/figures/score_distribution_per_bin.png
  analysis/summary.md                 auto-generated summary of findings

Reference baseline: lora_base. (Vanilla Qwen was dropped from scope; lora_base
acts as the "minimal fine-tuning" reference against which YaRN and YaRN+RPE
training are compared.)

Phase 4e (across-condition correlation with BABILong accuracy) is computed at
the bottom — needs per-condition per-bin BABILong accuracy (we use the
multi-entry-eval correctness counts on the same 60 stories for consistency).
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Optional plotting — only if matplotlib is available.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — plots will be skipped")


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
CONDITIONS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
REFERENCE = "lora_base"  # baseline for deltas / Jaccard / KS

# Published Qwen QR top-16 (BEIR-NQ derived) — secondary sanity reference.
PUBLISHED_QR_TOP16 = [
    "16-19", "16-2", "16-20", "16-14", "16-0", "15-24", "16-18", "17-18",
    "16-1", "19-18", "19-19", "18-16", "16-17", "19-25", "19-17", "20-21",
]


# --------------------------------------------------------------------------
# Loading

def load_head_scores(results_dir: Path, condition: str, bin_label: str) -> dict:
    """Return dict: head_id -> score."""
    p = results_dir / condition / f"{bin_label}_head_scores.json"
    if not p.exists():
        raise FileNotFoundError(p)
    pairs = json.loads(p.read_text())
    return {h: s for h, s in pairs}


def load_meta(results_dir: Path, condition: str, bin_label: str) -> dict:
    p = results_dir / condition / f"{bin_label}_meta.json"
    return json.loads(p.read_text())


# --------------------------------------------------------------------------
# 4a — top-K Jaccard

def topk_set(scores: dict, k: int) -> set:
    sorted_heads = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {h for h, _ in sorted_heads[:k]}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def analysis_4a(results_dir: Path, out_dir: Path) -> dict:
    """Top-K Jaccard overlap between each condition and the lora_base reference,
    plus overlap with the published Qwen QR top-16."""
    print("\n[4a] Top-K Jaccard overlaps")

    rows = []
    for k in [8, 16, 32, 64]:
        for cond in CONDITIONS:
            for b in BINS:
                try:
                    scores_cond = load_head_scores(results_dir, cond, b)
                    scores_ref = load_head_scores(results_dir, REFERENCE, b)
                except FileNotFoundError as e:
                    print(f"  skip k={k} {cond}/{b}: {e.filename} missing")
                    continue
                top_cond = topk_set(scores_cond, k)
                top_ref = topk_set(scores_ref, k)
                jac_vs_ref = jaccard(top_cond, top_ref)
                jac_vs_pub = jaccard(top_cond, set(PUBLISHED_QR_TOP16[:k]))
                rows.append({
                    "k": k,
                    "condition": cond,
                    "bin": b,
                    "jaccard_vs_lora_base": jac_vs_ref,
                    "jaccard_vs_published_top16": jac_vs_pub,
                    "n_overlap_vs_ref": len(top_cond & top_ref),
                })

    csv_path = out_dir / "head_overlap.csv"
    with csv_path.open("w") as f:
        f.write("k,condition,bin,jaccard_vs_lora_base,jaccard_vs_published_top16,n_overlap_vs_ref\n")
        for r in rows:
            f.write(f"{r['k']},{r['condition']},{r['bin']},"
                    f"{r['jaccard_vs_lora_base']:.4f},"
                    f"{r['jaccard_vs_published_top16']:.4f},"
                    f"{r['n_overlap_vs_ref']}\n")
    print(f"  wrote {csv_path}")

    # Print K=16 summary table
    print(f"\n  K=16 Jaccard vs lora_base reference (per bin):")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>5s}" for b in BINS))
    for cond in CONDITIONS:
        row = [r for r in rows if r["k"] == 16 and r["condition"] == cond]
        row_by_bin = {r["bin"]: r["jaccard_vs_lora_base"] for r in row}
        print(f"  {cond:<20s} " + " ".join(f"{row_by_bin.get(b, float('nan')):>5.2f}"
                                            for b in BINS))

    return {"rows": rows}


# --------------------------------------------------------------------------
# 4b — per-head Δscore on the top-16 heads of lora_base reference

def analysis_4b(results_dir: Path, out_dir: Path) -> dict:
    """For each (condition, bin), compute Δscore = trained_score - lora_base_score
    on the top-16 heads identified by lora_base at that bin.

    Why: this asks "do the top-16 heads of the reference model retain their
    scores under YaRN / YaRN+RPE training, or do they erode?"
    """
    print("\n[4b] Per-head Δscore on lora_base top-16 (per bin)")

    rows = []
    for b in BINS:
        try:
            ref_scores = load_head_scores(results_dir, REFERENCE, b)
        except FileNotFoundError:
            continue
        ref_top16 = sorted(ref_scores.items(), key=lambda kv: kv[1], reverse=True)[:16]
        ref_top16_heads = [h for h, _ in ref_top16]

        for cond in CONDITIONS:
            try:
                cond_scores = load_head_scores(results_dir, cond, b)
            except FileNotFoundError:
                continue
            for head_id in ref_top16_heads:
                ref_s = ref_scores[head_id]
                cond_s = cond_scores.get(head_id, float("nan"))
                rows.append({
                    "bin": b,
                    "head": head_id,
                    "condition": cond,
                    "ref_score": ref_s,
                    "score": cond_s,
                    "delta": cond_s - ref_s,
                })

    csv_path = out_dir / "head_deltas_top16.csv"
    with csv_path.open("w") as f:
        f.write("bin,head,condition,ref_score,score,delta\n")
        for r in rows:
            f.write(f"{r['bin']},{r['head']},{r['condition']},"
                    f"{r['ref_score']:.6f},{r['score']:.6f},{r['delta']:.6f}\n")
    print(f"  wrote {csv_path}")

    # Per-condition mean Δ on the lora_base top-16 (positive = condition strengthens
    # those heads, negative = erodes them, zero = preserves).
    print(f"\n  Mean Δscore on lora_base top-16 per (condition, bin):")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>7s}" for b in BINS))
    for cond in CONDITIONS:
        means_by_bin = {}
        for b in BINS:
            deltas = [r["delta"] for r in rows
                      if r["condition"] == cond and r["bin"] == b
                      and not np.isnan(r["delta"])]
            means_by_bin[b] = np.mean(deltas) if deltas else float("nan")
        print(f"  {cond:<20s} " + " ".join(f"{means_by_bin.get(b, float('nan')):>+7.4f}"
                                            for b in BINS))

    return {"rows": rows}


# --------------------------------------------------------------------------
# 4c — score distribution KS tests

def analysis_4c(results_dir: Path, out_dir: Path) -> dict:
    """KS test comparing the full 784-head score distribution between each
    condition and lora_base, per bin. Tells us whether the *distribution*
    shifted (not just specific heads)."""
    print("\n[4c] KS tests on score distribution vs lora_base")

    try:
        from scipy.stats import ks_2samp
    except ImportError:
        print("  WARNING: scipy not available — skipping KS tests")
        return {"rows": []}

    rows = []
    for b in BINS:
        try:
            ref_scores = list(load_head_scores(results_dir, REFERENCE, b).values())
        except FileNotFoundError:
            continue
        for cond in CONDITIONS:
            if cond == REFERENCE:
                continue
            try:
                cond_scores = list(load_head_scores(results_dir, cond, b).values())
            except FileNotFoundError:
                continue
            stat, p = ks_2samp(ref_scores, cond_scores)
            rows.append({
                "bin": b,
                "condition": cond,
                "ks_stat": stat,
                "p_value": p,
                "median_ref": float(np.median(ref_scores)),
                "median_cond": float(np.median(cond_scores)),
                "max_ref": float(np.max(ref_scores)),
                "max_cond": float(np.max(cond_scores)),
            })

    csv_path = out_dir / "distribution_ks.csv"
    with csv_path.open("w") as f:
        f.write("bin,condition,ks_stat,p_value,median_ref,median_cond,max_ref,max_cond\n")
        for r in rows:
            f.write(f"{r['bin']},{r['condition']},{r['ks_stat']:.4f},{r['p_value']:.6e},"
                    f"{r['median_ref']:.4f},{r['median_cond']:.4f},"
                    f"{r['max_ref']:.4f},{r['max_cond']:.4f}\n")
    print(f"  wrote {csv_path}")

    print(f"\n  KS p-value vs lora_base (per bin) — small p means distribution moved:")
    print(f"  {'cond':<20s} " + " ".join(f"{b:>9s}" for b in BINS))
    for cond in CONDITIONS:
        if cond == REFERENCE:
            continue
        ps = {r["bin"]: r["p_value"] for r in rows if r["condition"] == cond}
        print(f"  {cond:<20s} " + " ".join(f"{ps.get(b, float('nan')):>9.2e}" for b in BINS))

    return {"rows": rows}


# --------------------------------------------------------------------------
# Plot helpers

def plot_jaccard_heatmap(rows_4a, out_dir):
    if not HAS_MPL:
        return
    k = 16
    rows = [r for r in rows_4a["rows"] if r["k"] == k]
    if not rows:
        return
    matrix = np.full((len(CONDITIONS), len(BINS)), np.nan)
    for r in rows:
        i = CONDITIONS.index(r["condition"])
        j = BINS.index(r["bin"])
        matrix[i, j] = r["jaccard_vs_lora_base"]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS)
    ax.set_yticks(range(len(CONDITIONS))); ax.set_yticklabels(CONDITIONS)
    ax.set_title(f"Top-{k} Jaccard overlap with lora_base reference")
    for i in range(len(CONDITIONS)):
        for j in range(len(BINS)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = out_dir / "figures" / "head_overlap_heatmap.png"
    p.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_score_distributions(results_dir, out_dir):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=False, sharey=True)
    for ax, b in zip(axes.flat, BINS):
        for cond in CONDITIONS:
            try:
                scores = list(load_head_scores(results_dir, cond, b).values())
            except FileNotFoundError:
                continue
            ax.hist(scores, bins=40, alpha=0.4, label=cond)
        ax.set_title(f"bin {b}")
        ax.set_xlabel("head score")
        ax.legend(fontsize=7)
    fig.suptitle("Head score distributions per condition × bin")
    fig.tight_layout()
    p = out_dir / "figures" / "score_distribution_per_bin.png"
    p.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  wrote {p}")


# --------------------------------------------------------------------------
# 4e — across-condition correlation with downstream BABILong accuracy

def load_correctness(predictions_dir: Path, condition: str, bin_label: str,
                     story_idx_order: list) -> list:
    """Return per-sample correctness (0/1) for the given condition × bin in the
    same order as story_idx_order. The multi-entry eval predictions are
    indexed by position in the multi-entry data, which matches our story_idx
    values directly."""
    me_dir_name = f"{condition}_me"
    p = predictions_dir / me_dir_name / f"predictions_{bin_label}.json"
    if not p.exists():
        return None
    preds = json.loads(p.read_text())
    return [int(bool(preds[i].get("correct"))) for i in story_idx_order
            if i < len(preds)]


def analysis_4e(results_dir: Path, predictions_dir: Path, out_dir: Path) -> dict:
    """Across-condition correlation: does mean QR score on lora_base top-16
    predict mean BABILong accuracy at the same bin?

    With 3 conditions × 9 bins = 27 points, this is a directional / scatter
    statement, not a strict correlation. We report per-bin accuracy and per-bin
    mean QR score for plotting.
    """
    print("\n[4e] Across-condition: mean top-16 QR score vs BABILong accuracy")

    rows = []
    for b in BINS:
        try:
            ref_scores = load_head_scores(results_dir, REFERENCE, b)
        except FileNotFoundError:
            continue
        ref_top16_heads = sorted(ref_scores.items(), key=lambda kv: kv[1],
                                  reverse=True)[:16]
        ref_top16_heads = [h for h, _ in ref_top16_heads]

        for cond in CONDITIONS:
            try:
                cond_scores = load_head_scores(results_dir, cond, b)
                meta = load_meta(results_dir, cond, b)
            except FileNotFoundError:
                continue
            mean_top16 = float(np.mean(
                [cond_scores.get(h, 0.0) for h in ref_top16_heads]))

            corr = load_correctness(predictions_dir, cond, b, meta["story_id_order"])
            if corr is None:
                acc = float("nan")
            else:
                acc = float(np.mean(corr))
            rows.append({
                "bin": b,
                "condition": cond,
                "mean_top16_score": mean_top16,
                "babilong_accuracy": acc,
            })

    csv_path = out_dir / "score_vs_accuracy.csv"
    with csv_path.open("w") as f:
        f.write("bin,condition,mean_top16_score,babilong_accuracy\n")
        for r in rows:
            f.write(f"{r['bin']},{r['condition']},"
                    f"{r['mean_top16_score']:.6f},{r['babilong_accuracy']:.4f}\n")
    print(f"  wrote {csv_path}")

    return {"rows": rows}


# --------------------------------------------------------------------------
# Summary writer

def write_summary(out_dir: Path, results: dict):
    p = out_dir / "summary.md"
    lines = []
    lines.append(f"# Phase 4 Analysis Summary\n")
    lines.append(f"Generated by `analyze_head_structure.py`. "
                 f"Reference baseline: `{REFERENCE}`.\n")
    lines.append(f"## 4a — Top-16 Jaccard overlap with reference (per bin)\n")
    rows4a = [r for r in results["4a"]["rows"] if r["k"] == 16]
    lines.append(f"| condition | " + " | ".join(BINS) + " |")
    lines.append(f"|---" * (len(BINS) + 1) + "|")
    for cond in CONDITIONS:
        cells = []
        for b in BINS:
            r = next((r for r in rows4a if r["condition"] == cond and r["bin"] == b),
                     None)
            cells.append(f"{r['jaccard_vs_lora_base']:.2f}" if r else "—")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append(f"## 4b — Mean Δ-score on lora_base top-16 (per bin)\n")
    rows4b = results["4b"]["rows"]
    lines.append(f"| condition | " + " | ".join(BINS) + " |")
    lines.append(f"|---" * (len(BINS) + 1) + "|")
    for cond in CONDITIONS:
        cells = []
        for b in BINS:
            ds = [r["delta"] for r in rows4b if r["condition"] == cond and r["bin"] == b]
            cells.append(f"{np.mean(ds):+.4f}" if ds else "—")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines.append("")

    if results.get("4c"):
        lines.append(f"## 4c — KS p-values vs lora_base (small p ⇒ distribution shifted)\n")
        lines.append(f"| condition | " + " | ".join(BINS) + " |")
        lines.append(f"|---" * (len(BINS) + 1) + "|")
        for cond in CONDITIONS:
            if cond == REFERENCE:
                continue
            cells = []
            for b in BINS:
                r = next((r for r in results["4c"]["rows"]
                          if r["condition"] == cond and r["bin"] == b), None)
                cells.append(f"{r['p_value']:.2e}" if r else "—")
            lines.append(f"| {cond} | " + " | ".join(cells) + " |")
        lines.append("")

    if results.get("4e"):
        lines.append(f"## 4e — Mean top-16 QR score vs BABILong accuracy\n")
        lines.append(f"(For each (condition × bin), the mean QR score on lora_base's "
                     f"top-16 heads, paired with the model's accuracy on the same 60 "
                     f"stories at the multi-entry eval. Visual scatter to be made; "
                     f"raw values in `score_vs_accuracy.csv`.)\n")
        lines.append(f"| condition | bin | mean_top16_score | babilong_accuracy |")
        lines.append(f"|---|---|---|---|")
        for r in results["4e"]["rows"]:
            lines.append(f"| {r['condition']} | {r['bin']} | "
                         f"{r['mean_top16_score']:.4f} | "
                         f"{r['babilong_accuracy']:.3f} |")
        lines.append("")

    p.write_text("\n".join(lines))
    print(f"\n  wrote {p}")


# --------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="experiments/retrieval_heads/results")
    ap.add_argument("--predictions-dir",
                    default="experiments/babilong/results",
                    help="dir containing {condition}_me/predictions_*.json")
    ap.add_argument("--output-dir",
                    default="experiments/retrieval_heads/analysis")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    predictions_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    results["4a"] = analysis_4a(results_dir, out_dir)
    results["4b"] = analysis_4b(results_dir, out_dir)
    results["4c"] = analysis_4c(results_dir, out_dir)
    results["4e"] = analysis_4e(results_dir, predictions_dir, out_dir)

    plot_jaccard_heatmap(results["4a"], out_dir)
    plot_score_distributions(results_dir, out_dir)

    write_summary(out_dir, results)
    print(f"\nAll analyses written to {out_dir}/")


if __name__ == "__main__":
    main()
