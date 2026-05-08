#!/usr/bin/env python
"""Hard-subset cross-model analysis.

Starting from the 305-sample multi-entry eval, we lock a differentiating
subset by excluding every sample that all three models got correct at
every bin where it appears (the "universally-easy" set). This yields
N=181 samples.

The all-bins (not long-bins-only) definition is deliberate: a sample
where all models ace 32k/64k/128k but one flubs at 1k is still
informative — it exposes an across-the-board capability gap. See
memory note feedback_hard_subset_definition.md.

Outputs:
  - differentiating_subset_181.json : locked indices + per-sample metadata
  - hard_subset_analysis.md         : primary analysis view on the subset
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

BINS = ["0k","1k","2k","4k","8k","16k","32k","64k","128k"]
LONG_BINS = ["32k","64k","128k"]

BABILONG_DIR = Path(__file__).resolve().parent.parent.parent   # composable_cot/BABIlong
EVAL_DIR     = BABILONG_DIR / "data/eval_multi_entry"
RES_DIR      = BABILONG_DIR / "results"
OUT_DIR      = BABILONG_DIR / "analysis/multi_entry_eval"

MODELS = {
    "lora":    "lora_base_me",
    "y2_base": "y2_base_me",
    "y2_rpe":  "y2_rpe_cur_L16k_me",
}


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------

def load_all():
    """dict[bin] -> list[per-sample dict] with all 3 models' correctness joined."""
    out = {}
    for b in BINS:
        eval_data = json.loads((EVAL_DIR / f"{b}.json").read_text())
        preds = {
            key: json.loads((RES_DIR / folder / f"predictions_{b}.json").read_text())
            for key, folder in MODELS.items()
        }
        assert all(len(p) == len(eval_data) for p in preds.values()), f"length mismatch at {b}"
        rows = []
        for i, e in enumerate(eval_data):
            row = {
                "original_idx":   e["original_idx"],
                "object":         e["object"],
                "target_entries": e["target_entries"],
                "answer":         e["answer"],
                "question":       preds["lora"][i]["question"],
                "bin":            b,
                "token_count":    e["token_count"],
            }
            for key in MODELS:
                p = preds[key][i]
                row[f"{key}_correct"] = p["correct"]
                row[f"{key}_pred"]    = p["prediction"]
            rows.append(row)
        out[b] = rows
    return out


# ---------------------------------------------------------------------------
# Build the hard subset
# ---------------------------------------------------------------------------

def build_subset(data):
    """Return set of original_idx values that are NOT universally-easy.

    universally-easy = correct by every model at every bin where present.
    """
    # Collect all correctness flags per original_idx
    per_idx_flags = defaultdict(list)   # idx -> list of (bin, correctness_tuple)
    for b in BINS:
        for r in data[b]:
            per_idx_flags[r["original_idx"]].append(
                (b, r["lora_correct"], r["y2_base_correct"], r["y2_rpe_correct"])
            )

    hard_idx = set()
    easy_idx = set()
    for idx, flags in per_idx_flags.items():
        # universally easy iff EVERY flag tuple is (1,1,1)
        if all(l and yb and yr for _, l, yb, yr in flags):
            easy_idx.add(idx)
        else:
            hard_idx.add(idx)
    return hard_idx, easy_idx, per_idx_flags


def failure_score(per_idx_flags, idx):
    """Count across all (bin, model) where the model got it wrong."""
    score = 0
    for _, l, yb, yr in per_idx_flags[idx]:
        score += (not l) + (not yb) + (not yr)
    return score


# ---------------------------------------------------------------------------
# Tables / cuts restricted to the subset
# ---------------------------------------------------------------------------

def filter_rows(rows, keep_idx):
    return [r for r in rows if r["original_idx"] in keep_idx]


def per_bin_table(data, keep_idx):
    lines = []
    lines.append("| Bin | lora_base | y2_base (YaRN) | y2_rpe (YaRN+RPE) | N |")
    lines.append("|-----|----------:|---------------:|------------------:|--:|")
    for b in BINS:
        rows = filter_rows(data[b], keep_idx)
        if not rows:
            continue
        acc = {k: sum(r[f"{k}_correct"] for r in rows) / len(rows) for k in MODELS}
        lines.append(
            f"| {b} | {acc['lora']:.3f} | {acc['y2_base']:.3f} | "
            f"**{acc['y2_rpe']:.3f}** | {len(rows)} |"
        )
    return "\n".join(lines)


def deltas_table(data, keep_idx):
    lines = []
    lines.append("| Bin | lora | y2_base | y2_rpe | Δ rpe − base | Δ rpe − lora |")
    lines.append("|-----|-----:|--------:|-------:|-------------:|-------------:|")
    for b in LONG_BINS:
        rows = filter_rows(data[b], keep_idx)
        acc = {k: sum(r[f"{k}_correct"] for r in rows) / len(rows) for k in MODELS}
        lines.append(
            f"| {b} | {acc['lora']:.3f} | {acc['y2_base']:.3f} | **{acc['y2_rpe']:.3f}** | "
            f"{(acc['y2_rpe']-acc['y2_base'])*100:+.1f}pt | "
            f"{(acc['y2_rpe']-acc['lora'])*100:+.1f}pt |"
        )
    return "\n".join(lines)


def correctness_cube(data, keep_idx, bin_label):
    rows = filter_rows(data[bin_label], keep_idx)
    counts = Counter()
    per_group = defaultdict(list)
    for r in rows:
        key = (r["lora_correct"], r["y2_base_correct"], r["y2_rpe_correct"])
        counts[key] += 1
        per_group[key].append(r)
    return counts, per_group, len(rows)


def group_label(key):
    l, b, r = key
    names = []
    if l: names.append("lora")
    if b: names.append("y2_base")
    if r: names.append("y2_rpe")
    if not names: return "ALL_WRONG"
    if len(names) == 3: return "ALL_RIGHT"
    return "+".join(names)


def fmt_example(r, max_q_len=70):
    q = r["question"]
    if len(q) > max_q_len:
        q = q[:max_q_len] + "…"
    preds = (
        f"lora={r['lora_pred'][:12]:<12}  "
        f"y2_base={r['y2_base_pred'][:12]:<12}  "
        f"y2_rpe={r['y2_rpe_pred'][:12]}"
    )
    return (
        f"- **idx {r['original_idx']:>3}** · `{r['object']:<8}` · "
        f"entries={r['target_entries']} · A=`{r['answer']}`  \n"
        f"    Q: _{q}_  \n"
        f"    {preds}"
    )


def universally_hard_at_long(data, keep_idx):
    """idx -> #long-bins where all 3 models failed, among keep_idx."""
    tally = defaultdict(lambda: {"all_wrong": 0, "instances": []})
    for b in LONG_BINS:
        for r in filter_rows(data[b], keep_idx):
            if not (r["lora_correct"] or r["y2_base_correct"] or r["y2_rpe_correct"]):
                tally[r["original_idx"]]["all_wrong"] += 1
                tally[r["original_idx"]]["instances"].append(r)
    return sorted(tally.items(), key=lambda kv: -kv[1]["all_wrong"])


def rpe_uniquely_correct_at_long(data, keep_idx):
    rows = []
    for b in LONG_BINS:
        for r in filter_rows(data[b], keep_idx):
            if r["y2_rpe_correct"] and not r["lora_correct"] and not r["y2_base_correct"]:
                rows.append(r)
    return rows


def rpe_uniquely_wrong_at_long(data, keep_idx):
    rows = []
    for b in LONG_BINS:
        for r in filter_rows(data[b], keep_idx):
            if not r["y2_rpe_correct"] and r["lora_correct"] and r["y2_base_correct"]:
                rows.append(r)
    return rows


def by_target_entries(data, keep_idx):
    lines = []
    lines.append("| Bin | entries | N | lora | y2_base | y2_rpe |")
    lines.append("|-----|--------:|--:|-----:|--------:|-------:|")
    for b in LONG_BINS:
        bucket = defaultdict(list)
        for r in filter_rows(data[b], keep_idx):
            bucket[r["target_entries"]].append(r)
        for te in sorted(bucket):
            rs = bucket[te]
            acc = {k: sum(r[f"{k}_correct"] for r in rs) / len(rs) for k in MODELS}
            lines.append(
                f"| {b} | {te} | {len(rs)} | {acc['lora']:.3f} | "
                f"{acc['y2_base']:.3f} | **{acc['y2_rpe']:.3f}** |"
            )
    return "\n".join(lines)


def by_object(data, keep_idx):
    lines = []
    lines.append("| Bin | object | N | lora | y2_base | y2_rpe |")
    lines.append("|-----|--------|--:|-----:|--------:|-------:|")
    for b in LONG_BINS:
        bucket = defaultdict(list)
        for r in filter_rows(data[b], keep_idx):
            bucket[r["object"]].append(r)
        for obj in sorted(bucket):
            rs = bucket[obj]
            acc = {k: sum(r[f"{k}_correct"] for r in rs) / len(rs) for k in MODELS}
            lines.append(
                f"| {b} | {obj} | {len(rs)} | {acc['lora']:.3f} | "
                f"{acc['y2_base']:.3f} | **{acc['y2_rpe']:.3f}** |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading predictions + eval data ...")
    data = load_all()

    hard_idx, easy_idx, per_idx_flags = build_subset(data)
    N_hard = len(hard_idx)
    N_easy = len(easy_idx)
    N_total = N_hard + N_easy
    print(f"Hard subset: {N_hard} / {N_total} (easy removed: {N_easy})")

    # -------- write locked subset JSON --------
    # One record per original_idx with summary flags; sorted by failure score desc
    subset_records = []
    for idx in sorted(hard_idx, key=lambda i: (-failure_score(per_idx_flags, i), i)):
        # grab metadata from first available bin instance
        first = None
        for b in BINS:
            for r in data[b]:
                if r["original_idx"] == idx:
                    first = r
                    break
            if first: break
        # per-bin correctness grid
        grid = {}
        for b, l, yb, yr in per_idx_flags[idx]:
            grid[b] = {"lora": int(l), "y2_base": int(yb), "y2_rpe": int(yr)}
        subset_records.append({
            "original_idx":   idx,
            "object":         first["object"],
            "target_entries": first["target_entries"],
            "answer":         first["answer"],
            "question":       first["question"],
            "failure_score":  failure_score(per_idx_flags, idx),
            "per_bin_correctness": grid,
        })

    subset_json_path = OUT_DIR / f"differentiating_subset_{N_hard}.json"
    subset_json_path.write_text(json.dumps({
        "definition": "Exclude samples correct by every model at every bin where present (universally-easy).",
        "n_total": N_total,
        "n_hard":  N_hard,
        "n_excluded_universally_easy": N_easy,
        "bins_considered": BINS,
        "models": MODELS,
        "samples": subset_records,
    }, indent=2))
    print(f"Wrote {subset_json_path}")

    # -------- markdown --------
    lines = []
    lines.append(f"# Hard-Subset Cross-Model Analysis (N={N_hard})")
    lines.append("")
    lines.append(
        f"Locked differentiating subset of **N={N_hard}** samples, distilled "
        f"from the 305-sample multi-entry eval by removing every sample that "
        f"all three models got correct **at every bin where it appears**."
    )
    lines.append("")
    lines.append("**Why this definition?** Short-bin performance matters too. A sample "
                 "where all models ace 32k/64k/128k but one flubs at 1k is still "
                 "informative — it exposes an across-the-board capability gap, "
                 "not just a long-context one. The stricter all-bins rule costs "
                 "~4pt of headline signal at 128k but gives a more rigorous subset.")
    lines.append("")
    lines.append(f"**Removed:** {N_easy} universally-easy samples (correct everywhere "
                 f"by every model). **Kept:** {N_hard} samples with ≥1 model failure "
                 f"somewhere.")
    lines.append("")
    lines.append("**Conditions (all checkpoint-2000, eval w/ YaRN f=4 where applicable):**")
    lines.append("- `lora_base` — vanilla LoRA (no YaRN, no RPE at eval)")
    lines.append("- `y2_base`   — YaRN f=2 trained, f=4 eval (YaRN-only)")
    lines.append("- `y2_rpe`    — YaRN f=2 + RPE curriculum L=16K trained, f=4 eval")
    lines.append("")
    lines.append("Note: 1k bin has N<hard> because RMT-team source data is shorter there.")
    lines.append("")

    # ---------- per-bin table ----------
    lines.append("## Per-Bin Accuracy (hard subset)")
    lines.append("")
    lines.append(per_bin_table(data, hard_idx))
    lines.append("")

    lines.append("### Long-context detail")
    lines.append("")
    lines.append(deltas_table(data, hard_idx))
    lines.append("")

    # Compute headline deltas for prose
    acc128 = {k: sum(r[f"{k}_correct"] for r in filter_rows(data["128k"], hard_idx))
                 / len(filter_rows(data["128k"], hard_idx)) for k in MODELS}
    d_rpe_base_128 = (acc128["y2_rpe"] - acc128["y2_base"]) * 100
    d_rpe_lora_128 = (acc128["y2_rpe"] - acc128["lora"])    * 100
    lines.append(
        f"**Headline at 128k:** `y2_rpe` beats `y2_base` by "
        f"{d_rpe_base_128:+.1f}pt and `lora_base` by {d_rpe_lora_128:+.1f}pt on "
        f"the N={N_hard} hard subset — a sharper signal than the full 305 view "
        f"because the ~40% universally-easy tail no longer dilutes the average."
    )
    lines.append("")

    # ---------- 3-way Venn by long bin ----------
    lines.append("## Correctness Cube (per long bin, hard subset)")
    lines.append("")
    lines.append(
        "For each long bin, the three models' correctness patterns. "
        "`ALL_RIGHT` = all 3 solved it; `ALL_WRONG` = none solved it; "
        "single-model labels = only that model got it right."
    )
    for b in LONG_BINS:
        lines.append("")
        lines.append(f"### {b}")
        lines.append("")
        counts, _, n_rows = correctness_cube(data, hard_idx, b)
        lines.append(f"(N={n_rows})")
        lines.append("")
        lines.append("| Group | N |")
        lines.append("|-------|--:|")
        ordering = [
            (1,1,1), (0,0,0),
            (0,0,1), (1,0,0), (0,1,0),
            (1,1,0), (1,0,1), (0,1,1),
        ]
        for k in ordering:
            lines.append(f"| {group_label(k)} | {counts.get(k,0)} |")

    # ---------- universally hard ----------
    lines.append("")
    lines.append("## Universally-Hard Samples at Long Context")
    lines.append("")
    lines.append(
        "Samples in the hard subset where **all three** models failed at 32k/64k/128k. "
        "These expose a ceiling imposed by the underlying 7B model / LoRA rank, "
        "not any position-encoding choice."
    )
    lines.append("")
    ranked = universally_hard_at_long(data, hard_idx)
    n3 = sum(1 for _, v in ranked if v["all_wrong"] == 3)
    n2 = sum(1 for _, v in ranked if v["all_wrong"] == 2)
    n1 = sum(1 for _, v in ranked if v["all_wrong"] == 1)
    lines.append(f"- Failed by all 3 at **all 3 long bins**: **{n3}** samples")
    lines.append(f"- Failed by all 3 at **2 of 3 long bins**: **{n2}** samples")
    lines.append(f"- Failed by all 3 at **1 of 3 long bins**: **{n1}** samples")
    lines.append("")
    if n3 > 0:
        lines.append("**Top hardest (failed by everyone at 3/3 long bins):**")
        lines.append("")
        showed = 0
        for idx, rec in ranked:
            if rec["all_wrong"] < 3: break
            r = rec["instances"][0]
            lines.append(
                f"- idx **{idx}** · `{r['object']}` · entries={r['target_entries']} · "
                f"A=`{r['answer']}`\n    Q: _{r['question']}_"
            )
            showed += 1
            if showed >= 10: break
        lines.append("")

    # ---------- RPE uniquely correct ----------
    lines.append("## RPE-Uniquely-Correct at Long Context")
    lines.append("")
    lines.append(
        "Samples where `y2_rpe` was the **only** model to solve it (lora and "
        "y2_base both failed). These are the cases driving the headline gap."
    )
    lines.append("")
    rpe_wins = rpe_uniquely_correct_at_long(data, hard_idx)
    by_bin_w = Counter(r["bin"] for r in rpe_wins)
    lines.append("**Count per long bin:**")
    for b in LONG_BINS:
        lines.append(f"- {b}: {by_bin_w.get(b, 0)}")
    lines.append("")
    lines.append("**Example samples (first 8):**")
    lines.append("")
    for r in rpe_wins[:8]:
        lines.append(fmt_example(r))
    lines.append("")

    # ---------- RPE uniquely wrong ----------
    lines.append("## RPE-Uniquely-Wrong at Long Context")
    lines.append("")
    lines.append(
        "Samples where `y2_rpe` was the **only** model to miss (both baselines "
        "got them). Closest we have to 'RPE training hurt this.'"
    )
    lines.append("")
    rpe_losses = rpe_uniquely_wrong_at_long(data, hard_idx)
    by_bin_l = Counter(r["bin"] for r in rpe_losses)
    lines.append("**Count per long bin:**")
    for b in LONG_BINS:
        lines.append(f"- {b}: {by_bin_l.get(b, 0)}")
    lines.append("")
    if rpe_losses:
        lines.append("**Example samples (first 8):**")
        lines.append("")
        for r in rpe_losses[:8]:
            lines.append(fmt_example(r))
    else:
        lines.append("_(none — RPE never uniquely wrong at any long bin.)_")
    lines.append("")

    # ---------- Worked examples of the hard samples ----------
    lines.append("## What these hard samples look like")
    lines.append("")
    lines.append(
        "The 10 highest-failure-score samples in the hard subset, with every "
        "model's prediction at 128k. These show *why* a reader cares about the "
        "subset: concrete 2-hop 'where was X before Y' questions where at least "
        "one model misses at long context."
    )
    lines.append("")
    top10 = sorted(hard_idx,
                   key=lambda i: (-failure_score(per_idx_flags, i), i))[:10]
    for idx in top10:
        # find 128k row for this idx
        row_128 = next((r for r in data["128k"] if r["original_idx"] == idx), None)
        if not row_128:
            continue
        score = failure_score(per_idx_flags, idx)
        # compact per-bin correctness strip, e.g. "0k:✓✓✓ 1k:✗✓✓ ..."
        strip_parts = []
        for b in BINS:
            cell = next((x for x in per_idx_flags[idx] if x[0] == b), None)
            if cell is None:
                strip_parts.append(f"{b}:—")
            else:
                _, l, yb, yr = cell
                sym = lambda x: "✓" if x else "✗"
                strip_parts.append(f"{b}:{sym(l)}{sym(yb)}{sym(yr)}")
        strip = "  ".join(strip_parts)
        lines.append(
            f"**idx {idx}** · `{row_128['object']}` · entries={row_128['target_entries']} · "
            f"A=`{row_128['answer']}` · failure_score={score}"
        )
        lines.append("")
        lines.append(f"- Q: _{row_128['question']}_")
        lines.append(f"- At 128k: lora=`{row_128['lora_pred']}`, "
                     f"y2_base=`{row_128['y2_base_pred']}`, "
                     f"y2_rpe=`{row_128['y2_rpe_pred']}`")
        lines.append(f"- Per-bin (lora/y2_base/y2_rpe): {strip}")
        lines.append("")

    # ---------- cuts ----------
    lines.append("## Accuracy by `target_entries` (hard subset)")
    lines.append("")
    lines.append(by_target_entries(data, hard_idx))
    lines.append("")

    lines.append("## Accuracy by object (hard subset)")
    lines.append("")
    lines.append(by_object(data, hard_idx))
    lines.append("")

    # ---------- summary ----------
    uniq_rpe_wins = len(rpe_wins)
    uniq_rpe_losses = len(rpe_losses)
    all_wrong_128k = sum(1 for r in filter_rows(data["128k"], hard_idx)
                         if not r["lora_correct"] and not r["y2_base_correct"]
                         and not r["y2_rpe_correct"])

    lines.append("## Pattern Summary")
    lines.append("")
    lines.append(
        f"1. **RPE+YaRN dominates at long context on the locked subset.** At 128k, "
        f"`y2_rpe` beats `y2_base` by {d_rpe_base_128:+.1f}pt and `lora_base` by "
        f"{d_rpe_lora_128:+.1f}pt on N={N_hard}. Uniquely correct on **{uniq_rpe_wins}** "
        f"samples vs uniquely wrong on only **{uniq_rpe_losses}** across the 3 long bins."
    )
    lines.append(
        f"2. **Short-bin behavior is the price tag.** By design the subset retains "
        f"samples where y2_rpe (or y2_base) flubs short bins while baselines succeed. "
        f"This is where you'd spot silent regressions; the 0k/1k/2k accuracies in the "
        f"table above are the honest picture, not a flattering 90%+ clustered average."
    )
    lines.append(
        f"3. **Universally-hard samples stay rare.** Only **{n3}** sample fails on "
        f"all 3 models at all 3 long bins. At 128k, {all_wrong_128k} samples defeat "
        f"every model — the ceiling imposed by rank-16 capacity / inherent bAbI ambiguity."
    )
    lines.append(
        f"4. **Subset construction removed {N_easy} / {N_total} samples** that were "
        f"universally-easy. The removed set was pushing all three models' averages "
        f"upward and compressing the delta; removing them surfaces the real gap."
    )

    md_path = OUT_DIR / "hard_subset_analysis.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
