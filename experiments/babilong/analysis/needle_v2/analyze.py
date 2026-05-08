"""
Comprehensive unbiased analysis of v2 needle-position eval.

Produces tables (CSV + markdown) and figures across many slicings:
  1. Headline: accuracy per (model, zone, bin)
  2. Cross-model deltas at each (zone, bin) cell
  3. By tier: hard_multi_entry vs single_entry_ref, separately
  4. By target_entries (1, 2, 3, ...): how does multi-hop count interact
  5. Position-of-needles within zone (mean_fact_pos quartiles)
  6. Pairwise sample-level agreement between models (per cell)
  7. Per-sample correctness counts: how many cells does each model get right
  8. Confusion: which rooms get predicted in errors
  9. Token-count-vs-accuracy (continuous, ignoring bin labels)
 10. Refusal/garbage rates (predictions outside the 6 valid rooms)

Output: tables/*.{csv,md} and figures/*.png
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path("/Users/reach/CodingRepositories/02 TAUR Labs/RPE/experiments/babilong")
ANA = ROOT / "analysis" / "needle_v2"
TABLES = ANA / "tables"
FIGS = ANA / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

MODELS = ["lora_base", "y2_base", "y2_rpe_cur_L16k"]
ZONES = ["beg", "mid", "end"]
BINS = ["1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
BIN_TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 130700}
ROOMS = {"bedroom", "bathroom", "office", "kitchen", "garden", "hallway"}

mpl.rcParams["figure.dpi"] = 110
mpl.rcParams["savefig.bbox"] = "tight"

df = pd.read_csv(ANA / "master.csv")
df["bin"] = pd.Categorical(df["bin"], categories=BINS, ordered=True)
df["zone"] = pd.Categorical(df["zone"], categories=ZONES, ordered=True)
df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)


def write_table(name, table, float_fmt=None):
    table.to_csv(TABLES / f"{name}.csv")
    md = table.to_markdown(floatfmt=float_fmt or ".4f")
    (TABLES / f"{name}.md").write_text(md)


def acc_grid(d):
    """Pivot to model x bin x zone accuracy grid (with N)."""
    g = d.groupby(["model", "zone", "bin"], observed=True).agg(
        n=("correct", "size"),
        acc=("correct", "mean"),
    ).reset_index()
    return g


# -------------------------------------------------------------------
# (1) HEADLINE: per-cell accuracy + N
# -------------------------------------------------------------------
g_all = acc_grid(df)
g_all.to_csv(TABLES / "01_headline_long.csv", index=False)

# Pivot: rows = (model, zone), cols = bin, values = acc
acc_wide = g_all.pivot_table(index=["model", "zone"], columns="bin", values="acc", observed=True)
n_wide = g_all.pivot_table(index=["model", "zone"], columns="bin", values="n", observed=True)
write_table("01_headline_acc", acc_wide, float_fmt=".4f")
write_table("01_headline_n", n_wide, float_fmt=".0f")

# Each model's overall (un-weighted-mean across cells) and sample-weighted overall
overall = g_all.groupby("model", observed=True).apply(
    lambda x: pd.Series({
        "cells": len(x),
        "samples": x["n"].sum(),
        "mean_cell_acc": x["acc"].mean(),
        "sample_weighted_acc": (x["n"] * x["acc"]).sum() / x["n"].sum(),
    })
)
write_table("01_overall", overall)

# -------------------------------------------------------------------
# (2) CROSS-MODEL DELTAS at each cell (only across-model deltas)
# -------------------------------------------------------------------
pairs = [
    ("y2_base", "lora_base"),
    ("y2_rpe_cur_L16k", "lora_base"),
    ("y2_rpe_cur_L16k", "y2_base"),
]
delta_rows = []
for a, b in pairs:
    aw = acc_wide.xs(a, level="model")
    bw = acc_wide.xs(b, level="model")
    delta = aw - bw
    delta.index = pd.MultiIndex.from_product([[f"{a} - {b}"], delta.index], names=["pair", "zone"])
    delta_rows.append(delta)
all_delta = pd.concat(delta_rows)
write_table("02_cross_model_delta", all_delta, float_fmt="+.4f")

# Sample-level paired McNemar-style: for each (zone, bin), among samples where
# both models judged, how many flips A>B and B>A?
def paired_flips(d, model_a, model_b):
    """For each (zone,bin), return n_AB (A correct, B wrong), n_BA, n_both, n_neither."""
    out = []
    for (zone, b), grp in d.groupby(["zone", "bin"], observed=True):
        wa = grp[grp.model == model_a].set_index("row_in_cell")["correct"]
        wb = grp[grp.model == model_b].set_index("row_in_cell")["correct"]
        common = wa.index.intersection(wb.index)
        wa, wb = wa.loc[common], wb.loc[common]
        n = len(common)
        both = int((wa & wb).sum())
        a_only = int((wa & ~wb).sum())
        b_only = int((~wa & wb).sum())
        neither = int((~wa & ~wb).sum())
        out.append({
            "zone": zone, "bin": b, "n": n,
            f"{model_a}_only": a_only,
            f"{model_b}_only": b_only,
            "both_correct": both,
            "both_wrong": neither,
            "net_a_minus_b": a_only - b_only,
        })
    return pd.DataFrame(out)


for a, b in pairs:
    flips = paired_flips(df, a, b)
    flips.to_csv(TABLES / f"02_paired_flips__{a}_vs_{b}.csv", index=False)


# -------------------------------------------------------------------
# (3) BY TIER (hard vs single-entry reference) — separate analyses
# -------------------------------------------------------------------
for tier in df["tier"].unique():
    sub = df[df.tier == tier]
    g = acc_grid(sub)
    wide = g.pivot_table(index=["model", "zone"], columns="bin", values="acc", observed=True)
    nwide = g.pivot_table(index=["model", "zone"], columns="bin", values="n", observed=True)
    write_table(f"03_tier_{tier}_acc", wide, float_fmt=".4f")
    write_table(f"03_tier_{tier}_n", nwide, float_fmt=".0f")

# -------------------------------------------------------------------
# (4) BY target_entries (multi-hop count: 1, 2, 3 ...)
# -------------------------------------------------------------------
te_counts = df.drop_duplicates("original_idx")["target_entries"].value_counts().sort_index()
te_counts.to_csv(TABLES / "04_target_entries_counts.csv")

g_te = df.groupby(["model", "target_entries", "bin"], observed=True).agg(
    n=("correct", "size"), acc=("correct", "mean")
).reset_index()
te_wide = g_te.pivot_table(index=["model", "target_entries"], columns="bin", values="acc", observed=True)
te_n = g_te.pivot_table(index=["model", "target_entries"], columns="bin", values="n", observed=True)
write_table("04_by_target_entries_acc", te_wide, float_fmt=".4f")
write_table("04_by_target_entries_n", te_n, float_fmt=".0f")

# -------------------------------------------------------------------
# (5) Position of needle WITHIN zone — quartile of mean_fact_pos
# -------------------------------------------------------------------
df_pos = df.dropna(subset=["mean_fact_pos"]).copy()
df_pos["mean_pos_q"] = pd.qcut(df_pos["mean_fact_pos"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
g_pos = df_pos.groupby(["model", "zone", "mean_pos_q"], observed=True).agg(
    n=("correct", "size"), acc=("correct", "mean")
).reset_index()
pos_wide = g_pos.pivot_table(index=["model", "mean_pos_q"], columns="zone", values="acc", observed=True)
pos_n = g_pos.pivot_table(index=["model", "mean_pos_q"], columns="zone", values="n", observed=True)
write_table("05_pos_quartile_acc", pos_wide, float_fmt=".4f")
write_table("05_pos_quartile_n", pos_n, float_fmt=".0f")

# Also: for the largest bins (32k+), where mean_fact_pos has most spread, do a finer slice
big = df_pos[df_pos["bin"].isin(["32k", "64k", "128k"])]
big = big.copy()
big["mean_pos_decile"] = pd.qcut(big["mean_fact_pos"], q=10, labels=False, duplicates="drop")
g_big = big.groupby(["model", "mean_pos_decile"], observed=True).agg(
    n=("correct", "size"), acc=("correct", "mean")
).reset_index()
big_wide = g_big.pivot_table(index="mean_pos_decile", columns="model", values="acc", observed=True)
big_n = g_big.pivot_table(index="mean_pos_decile", columns="model", values="n", observed=True)
write_table("05_pos_decile_big_bins_acc", big_wide, float_fmt=".4f")
write_table("05_pos_decile_big_bins_n", big_n, float_fmt=".0f")

# -------------------------------------------------------------------
# (6) PAIRWISE AGREEMENT MATRICES per cell
# -------------------------------------------------------------------
agree_rows = []
for (zone, b), grp in df.groupby(["zone", "bin"], observed=True):
    wide = grp.pivot_table(index="row_in_cell", columns="model", values="correct", observed=True)
    if wide.empty:
        continue
    n = len(wide)
    row = {"zone": zone, "bin": b, "n": n}
    for a, c in [("lora_base", "y2_base"),
                 ("lora_base", "y2_rpe_cur_L16k"),
                 ("y2_base", "y2_rpe_cur_L16k")]:
        agree = (wide[a] == wide[c]).sum()
        row[f"agree_{a}_{c}"] = agree / n
    # Three-way joint
    all_correct = (wide.sum(axis=1) == 3).sum()
    none_correct = (wide.sum(axis=1) == 0).sum()
    row["all3_correct"] = all_correct / n
    row["all3_wrong"] = none_correct / n
    row["exactly_one_correct"] = (wide.sum(axis=1) == 1).sum() / n
    row["exactly_two_correct"] = (wide.sum(axis=1) == 2).sum() / n
    agree_rows.append(row)
agree = pd.DataFrame(agree_rows)
agree.to_csv(TABLES / "06_agreement.csv", index=False)
write_table("06_agreement", agree.set_index(["zone", "bin"]), float_fmt=".4f")

# -------------------------------------------------------------------
# (7) PER-SAMPLE CORRECTNESS COUNTS per model (out of all cells where the sample is present)
# -------------------------------------------------------------------
per_sample = df.groupby(["model", "original_idx"], observed=True).agg(
    n_cells=("correct", "size"),
    n_correct=("correct", "sum"),
).reset_index()
per_sample["frac_correct"] = per_sample["n_correct"] / per_sample["n_cells"]
per_sample.to_csv(TABLES / "07_per_sample_summary.csv", index=False)

# Distribution of per-sample fraction-correct, per model
hist_rows = []
bins_edges = [0, 0.0001, 0.25, 0.5, 0.75, 0.9999, 1.01]
labels = ["0%", "0-25%", "25-50%", "50-75%", "75-99%", "100%"]
for m in MODELS:
    sub = per_sample[per_sample.model == m]
    cuts = pd.cut(sub["frac_correct"], bins=bins_edges, labels=labels, include_lowest=True, right=False)
    counts = cuts.value_counts().reindex(labels).fillna(0).astype(int)
    hist_rows.append(pd.Series(counts, name=m))
hist = pd.concat(hist_rows, axis=1)
write_table("07_per_sample_dist", hist, float_fmt=".0f")

# Stratify by tier
hist_tier_rows = []
for tier in df["tier"].unique():
    ps = df[df.tier == tier].groupby(["model", "original_idx"], observed=True).agg(
        n_cells=("correct", "size"), n_correct=("correct", "sum")).reset_index()
    ps["frac_correct"] = ps["n_correct"] / ps["n_cells"]
    for m in MODELS:
        sub = ps[ps.model == m]
        cuts = pd.cut(sub["frac_correct"], bins=bins_edges, labels=labels, include_lowest=True, right=False)
        counts = cuts.value_counts().reindex(labels).fillna(0).astype(int)
        for lab, c in counts.items():
            hist_tier_rows.append({"tier": tier, "model": m, "frac_correct_bucket": lab, "n_samples": c})
pd.DataFrame(hist_tier_rows).to_csv(TABLES / "07_per_sample_dist_by_tier.csv", index=False)

# -------------------------------------------------------------------
# (8) ERROR ANALYSIS — predicted-room distribution among errors
# -------------------------------------------------------------------
err = df[~df["correct"]].copy()
err["pred_class"] = err["pred_norm"].apply(lambda s: s if s in ROOMS else "OTHER")
err_class = err.groupby(["model", "pred_class"], observed=True).size().unstack(fill_value=0)
err_class["TOTAL"] = err_class.sum(axis=1)
write_table("08_error_pred_class", err_class, float_fmt=".0f")

# Refusal/garbage rate (per cell)
refusal = err[err.pred_class == "OTHER"].groupby(["model", "zone", "bin"], observed=True).size()
total = df.groupby(["model", "zone", "bin"], observed=True).size()
refusal_rate = (refusal / total).unstack("bin").fillna(0)
write_table("08_refusal_rate_OTHER", refusal_rate, float_fmt=".4f")

# Distribution of OTHER predictions for each model (top 20 strings)
other_rows = []
for m in MODELS:
    sub = err[(err.model == m) & (err.pred_class == "OTHER")]
    top = sub["pred_norm"].value_counts().head(20)
    for s, c in top.items():
        other_rows.append({"model": m, "pred": s, "count": c})
pd.DataFrame(other_rows).to_csv(TABLES / "08_other_pred_top20.csv", index=False)

# Confusion matrix per model: target -> top wrong prediction (only valid rooms)
conf_rows = []
for m in MODELS:
    sub = err[(err.model == m) & (err.pred_class != "OTHER")]
    tab = pd.crosstab(sub["target_norm"], sub["pred_norm"])
    tab.to_csv(TABLES / f"08_confusion_{m}.csv")
    write_table(f"08_confusion_{m}", tab, float_fmt=".0f")

# -------------------------------------------------------------------
# (9) Recency / zone effect — per model, raw zone accuracy at each bin
# -------------------------------------------------------------------
zone_grid = g_all.pivot_table(index=["model", "bin"], columns="zone", values="acc", observed=True)
write_table("09_zone_per_bin_acc", zone_grid, float_fmt=".4f")

# Cross-model recency comparison: end-vs-mid GAP, model by model side by side
end_mid_gap = (g_all.pivot_table(index=["model", "bin"], columns="zone", values="acc", observed=True))
end_mid_gap["end_minus_mid"] = end_mid_gap["end"] - end_mid_gap["mid"]
end_mid_gap["end_minus_beg"] = end_mid_gap["end"] - end_mid_gap["beg"]
end_mid_gap["mid_minus_beg"] = end_mid_gap["mid"] - end_mid_gap["beg"]
gap_only = end_mid_gap[["end_minus_mid", "end_minus_beg", "mid_minus_beg"]]
gap_pivot = gap_only.unstack("model")
write_table("09_zone_gaps", gap_pivot, float_fmt="+.4f")

# -------------------------------------------------------------------
# (10) Token-count buckets (ignore bin labels; bucket by actual token_count)
# -------------------------------------------------------------------
df_tc = df.dropna(subset=["token_count"]).copy()
edges = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
df_tc["tc_bucket"] = pd.cut(df_tc["token_count"], bins=edges, right=True)
g_tc = df_tc.groupby(["model", "tc_bucket"], observed=True).agg(
    n=("correct", "size"), acc=("correct", "mean")).reset_index()
tc_wide = g_tc.pivot_table(index="tc_bucket", columns="model", values="acc", observed=True)
tc_n = g_tc.pivot_table(index="tc_bucket", columns="model", values="n", observed=True)
write_table("10_token_count_acc", tc_wide, float_fmt=".4f")
write_table("10_token_count_n", tc_n, float_fmt=".0f")

# -------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------
COLORS = {"lora_base": "#888", "y2_base": "#1f77b4", "y2_rpe_cur_L16k": "#d62728"}
ZONE_LS = {"beg": "-", "mid": "--", "end": ":"}


# Fig A: per-zone bin curves (3 zones x 1 panel each, 3 lines per panel)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, zone in zip(axes, ZONES):
    for m in MODELS:
        sub = g_all[(g_all.zone == zone) & (g_all.model == m)].sort_values("bin")
        ax.plot(range(len(BINS)), sub["acc"], "o-", color=COLORS[m], label=m, linewidth=2)
        for x, (acc, n) in enumerate(zip(sub["acc"], sub["n"])):
            ax.annotate(f"n={n}", (x, acc), fontsize=6, ha="center", va="bottom", alpha=0.6)
    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels(BINS, rotation=0)
    ax.set_title(f"zone = {zone}")
    ax.set_xlabel("bin")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("accuracy")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Accuracy vs context length, per zone (3 models)", y=1.02)
fig.savefig(FIGS / "fig_A_zones_x_bins.png")
plt.close(fig)

# Fig B: per-model bin curves with all 3 zones overlaid
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, m in zip(axes, MODELS):
    for zone in ZONES:
        sub = g_all[(g_all.zone == zone) & (g_all.model == m)].sort_values("bin")
        ax.plot(range(len(BINS)), sub["acc"], "o-", linestyle=ZONE_LS[zone], label=zone, linewidth=2)
    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels(BINS, rotation=0)
    ax.set_title(m)
    ax.set_xlabel("bin")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
axes[0].set_ylabel("accuracy")
axes[0].legend(fontsize=9, loc="lower left")
fig.suptitle("Accuracy vs context length, per model (3 zones overlaid)", y=1.02)
fig.savefig(FIGS / "fig_B_models_x_zones.png")
plt.close(fig)

# Fig C: heatmap, model x zone x bin (one heatmap per model)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, m in zip(axes, MODELS):
    sub = g_all[g_all.model == m].pivot_table(index="zone", columns="bin", values="acc", observed=True)
    im = ax.imshow(sub.values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            ax.text(j, i, f"{sub.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels(BINS)
    ax.set_yticks(range(len(ZONES)))
    ax.set_yticklabels(ZONES)
    ax.set_title(m, fontsize=10)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
fig.suptitle("Per-cell accuracy heatmap (zone x bin)", y=1.05)
fig.savefig(FIGS / "fig_C_heatmaps.png")
plt.close(fig)

# Fig D: cross-model deltas heatmap
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, (a, b) in zip(axes, pairs):
    aw = g_all[g_all.model == a].pivot_table(index="zone", columns="bin", values="acc", observed=True)
    bw = g_all[g_all.model == b].pivot_table(index="zone", columns="bin", values="acc", observed=True)
    delta = aw - bw
    vmax = max(abs(delta.values.min()), abs(delta.values.max()))
    im = ax.imshow(delta.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{delta.values[i, j]:+.2f}", ha="center", va="center", fontsize=8, color="black")
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS)
    ax.set_yticks(range(len(ZONES))); ax.set_yticklabels(ZONES)
    ax.set_title(f"{a} - {b}", fontsize=9)
fig.suptitle("Cross-model accuracy delta (positive = first model better)", y=1.05)
fig.savefig(FIGS / "fig_D_cross_model_deltas.png")
plt.close(fig)

# Fig E: per-sample fraction-correct distribution (histogram)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for ax, m in zip(axes, MODELS):
    sub = per_sample[per_sample.model == m]
    ax.hist(sub["frac_correct"], bins=20, color=COLORS[m], edgecolor="white")
    ax.set_title(m); ax.set_xlabel("frac of cells correct (per sample)"); ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("# samples")
fig.suptitle("Per-sample fraction of cells correct (across all 24 cells)", y=1.02)
fig.savefig(FIGS / "fig_E_per_sample_dist.png")
plt.close(fig)

# Fig F: by tier (hard vs single-entry)
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
for ti, tier in enumerate(["hard_multi_entry", "single_entry_ref"]):
    for zi, zone in enumerate(ZONES):
        ax = axes[ti, zi]
        for m in MODELS:
            sub = df[(df.tier == tier) & (df.zone == zone) & (df.model == m)]
            agg = sub.groupby("bin", observed=True)["correct"].agg(["mean", "size"]).reset_index()
            agg["bin_idx"] = agg["bin"].map({b: i for i, b in enumerate(BINS)})
            ax.plot(agg["bin_idx"], agg["mean"], "o-", color=COLORS[m], label=m, linewidth=2)
        ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS, rotation=0)
        ax.set_title(f"tier={tier}, zone={zone}", fontsize=9)
        ax.set_ylim(0, 1.0); ax.grid(alpha=0.3)
        if zi == 0: ax.set_ylabel("accuracy")
        if ti == 1: ax.set_xlabel("bin")
        if ti == 0 and zi == 0: ax.legend(fontsize=8, loc="lower left")
fig.suptitle("By tier: hard_multi_entry (top) vs single_entry_ref (bottom)", y=1.01)
fig.savefig(FIGS / "fig_F_by_tier.png")
plt.close(fig)

# Fig G: accuracy by target_entries
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
te_levels = sorted(df["target_entries"].dropna().unique().tolist())
for ax, m in zip(axes, MODELS):
    for te in te_levels:
        sub = df[(df.model == m) & (df.target_entries == te)]
        agg = sub.groupby("bin", observed=True)["correct"].mean().reindex(BINS)
        ax.plot(range(len(BINS)), agg.values, "o-", label=f"target_entries={te}", linewidth=2)
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(BINS)
    ax.set_title(m); ax.set_xlabel("bin"); ax.grid(alpha=0.3); ax.set_ylim(0, 1.0)
axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Accuracy by number of target entries (object moves)", y=1.02)
fig.savefig(FIGS / "fig_G_by_target_entries.png")
plt.close(fig)

# Fig H: needle position (decile) — big bins only
big_pivot = big.groupby(["model", "mean_pos_decile"], observed=True)["correct"].mean().unstack("model")
fig, ax = plt.subplots(figsize=(10, 4.5))
for m in MODELS:
    if m in big_pivot.columns:
        ax.plot(big_pivot.index, big_pivot[m], "o-", color=COLORS[m], label=m, linewidth=2)
ax.set_xlabel("mean fact-position decile (within document)")
ax.set_ylabel("accuracy")
ax.set_title("Big bins (32k/64k/128k): accuracy vs needle position decile")
ax.set_xticks(range(10)); ax.grid(alpha=0.3); ax.set_ylim(0, 1.0); ax.legend(fontsize=9)
fig.savefig(FIGS / "fig_H_needle_pos_decile_big_bins.png")
plt.close(fig)

print(f"Wrote tables to {TABLES}")
print(f"Wrote figures to {FIGS}")
print(f"Tables: {sorted(p.name for p in TABLES.glob('*.csv'))}")
print(f"Figures: {sorted(p.name for p in FIGS.glob('*.png'))}")
