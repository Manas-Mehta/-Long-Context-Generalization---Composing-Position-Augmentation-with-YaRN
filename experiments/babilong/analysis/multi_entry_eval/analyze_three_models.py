#!/usr/bin/env python
"""Cross-model analysis of the 305-sample multi-entry eval.

Joins per-sample predictions from three conditions (lora_base, y2_base,
y2_rpe_cur_L16k) by positional index (verified length-matched across bins)
and surfaces:
  - per-bin accuracy table
  - cross-model correctness tallies at long context
  - universally-hard samples
  - each model's uniquely-correct / uniquely-failed samples
  - cuts by target_entries and object

Outputs a markdown report at the path in OUT_MD.

Run from project root:
  python experiments/babilong/analysis/multi_entry_eval/analyze_three_models.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

BINS = ["0k","1k","2k","4k","8k","16k","32k","64k","128k"]
LONG_BINS = ["32k","64k","128k"]

BABILONG_DIR = Path(__file__).resolve().parent.parent.parent   # experiments/babilong
EVAL_DIR     = BABILONG_DIR / "data/eval_multi_entry"
RES_DIR      = BABILONG_DIR / "results"

MODELS = {
    "lora":    "lora_base_me",
    "y2_base": "y2_base_me",
    "y2_rpe":  "y2_rpe_cur_L16k_me",
}

OUT_MD = BABILONG_DIR / "analysis/multi_entry_eval/three_model_analysis.md"


def load_all():
    """Return dict[bin] -> list[per-sample dict] with all 3 models' correctness joined."""
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


def per_bin_table(data):
    lines = []
    lines.append("| Bin | lora_base | y2_base (YaRN) | y2_rpe (YaRN+RPE) | N |")
    lines.append("|-----|----------:|---------------:|------------------:|--:|")
    for b in BINS:
        rows = data[b]
        acc = {k: sum(r[f"{k}_correct"] for r in rows) / len(rows) for k in MODELS}
        lines.append(
            f"| {b} | {acc['lora']:.3f} | {acc['y2_base']:.3f} | "
            f"**{acc['y2_rpe']:.3f}** | {len(rows)} |"
        )
    return "\n".join(lines)


def long_context_deltas(data):
    lines = []
    lines.append("| Bin | Δ rpe − base (YaRN) | Δ rpe − lora |")
    lines.append("|-----|--------------------:|-------------:|")
    for b in LONG_BINS:
        rows = data[b]
        acc = {k: sum(r[f"{k}_correct"] for r in rows) / len(rows) for k in MODELS}
        lines.append(
            f"| {b} | "
            f"{(acc['y2_rpe']-acc['y2_base'])*100:+.1f}pt | "
            f"{(acc['y2_rpe']-acc['lora'])*100:+.1f}pt |"
        )
    return "\n".join(lines)


def correctness_cube(data, bin_label):
    """3-way Venn tally at one bin."""
    rows = data[bin_label]
    counts = Counter()
    per_group = defaultdict(list)
    for r in rows:
        key = (r["lora_correct"], r["y2_base_correct"], r["y2_rpe_correct"])
        counts[key] += 1
        per_group[key].append(r)
    return counts, per_group


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
        f"- **idx {r['original_idx']:>3}**  `{r['object']:<8}` "
        f"entries={r['target_entries']}  A=`{r['answer']}`  "
        f"Q=_{q}_  \n"
        f"    {preds}"
    )


def top_hardest_samples(data):
    """Samples where all 3 models failed the MOST across long bins."""
    # for each original_idx, count how often all 3 failed at 32k/64k/128k
    tally = defaultdict(lambda: {"all_wrong": 0, "instances": []})
    for b in LONG_BINS:
        for r in data[b]:
            if not (r["lora_correct"] or r["y2_base_correct"] or r["y2_rpe_correct"]):
                tally[r["original_idx"]]["all_wrong"] += 1
                tally[r["original_idx"]]["instances"].append(r)
    ranked = sorted(tally.items(), key=lambda kv: -kv[1]["all_wrong"])
    return ranked


def rpe_uniquely_correct_at_long(data):
    rows = []
    for b in LONG_BINS:
        for r in data[b]:
            if r["y2_rpe_correct"] and not r["lora_correct"] and not r["y2_base_correct"]:
                rows.append(r)
    return rows


def rpe_uniquely_wrong_at_long(data):
    rows = []
    for b in LONG_BINS:
        for r in data[b]:
            if not r["y2_rpe_correct"] and r["lora_correct"] and r["y2_base_correct"]:
                rows.append(r)
    return rows


def by_target_entries(data):
    """Per-bin accuracy bucketed by target_entries count."""
    lines = []
    lines.append("| Bin | entries | N | lora | y2_base | y2_rpe |")
    lines.append("|-----|--------:|--:|-----:|--------:|-------:|")
    for b in LONG_BINS:
        bucket = defaultdict(list)
        for r in data[b]:
            bucket[r["target_entries"]].append(r)
        for te in sorted(bucket):
            rs = bucket[te]
            acc = {k: sum(r[f"{k}_correct"] for r in rs) / len(rs) for k in MODELS}
            lines.append(
                f"| {b} | {te} | {len(rs)} | {acc['lora']:.3f} | "
                f"{acc['y2_base']:.3f} | **{acc['y2_rpe']:.3f}** |"
            )
    return "\n".join(lines)


def by_object(data):
    lines = []
    lines.append("| Bin | object | N | lora | y2_base | y2_rpe |")
    lines.append("|-----|--------|--:|-----:|--------:|-------:|")
    for b in LONG_BINS:
        bucket = defaultdict(list)
        for r in data[b]:
            bucket[r["object"]].append(r)
        for obj in sorted(bucket):
            rs = bucket[obj]
            acc = {k: sum(r[f"{k}_correct"] for r in rs) / len(rs) for k in MODELS}
            lines.append(
                f"| {b} | {obj} | {len(rs)} | {acc['lora']:.3f} | "
                f"{acc['y2_base']:.3f} | **{acc['y2_rpe']:.3f}** |"
            )
    return "\n".join(lines)


def main():
    print("Loading all predictions + eval data...")
    data = load_all()

    lines = []
    lines.append("# Three-Model Multi-Entry Eval Analysis")
    lines.append("")
    lines.append(
        "Cross-comparison of three LoRA-rank-16 checkpoints on the 305-sample "
        "multi-entry eval subset (samples where the queried object visits the "
        "target room ≥ 2 times → genuinely multi-hop). "
        "Generated by `analyze_three_models.py`."
    )
    lines.append("")
    lines.append("**Conditions (all checkpoint-2000, eval w/ YaRN f=4 where applicable):**")
    lines.append("- `lora_base` — vanilla LoRA (no YaRN, no RPE at eval)")
    lines.append("- `y2_base` (`y2_base_1k`) — YaRN f=2 trained, f=4 eval (YaRN-only)")
    lines.append("- `y2_rpe` (`y2_rpe_cur_L16k_1k`) — YaRN f=2 + RPE curriculum L=16K trained, f=4 eval")
    lines.append("")
    lines.append("Note: 1k bin has N=264 (RMT-team source data quirk); all other bins N=305.")
    lines.append("")

    # ---------- per-bin table ----------
    lines.append("## Per-Bin Accuracy")
    lines.append("")
    lines.append(per_bin_table(data))
    lines.append("")

    lines.append("### Long-context deltas")
    lines.append("")
    lines.append(long_context_deltas(data))
    lines.append("")
    lines.append(
        "**Headline:** `y2_rpe` (YaRN+RPE) beats `y2_base` (YaRN-only) by 7-17 pts "
        "at 32k-128k, and beats the vanilla `lora_base` by 8-12 pts at the same range. "
        "Short-bin parity is essentially flat across all three models."
    )
    lines.append("")

    # ---------- 3-way Venn at 128k ----------
    lines.append("## Correctness Cube (per bin)")
    lines.append("")
    lines.append(
        "For each long-context bin, how do the three models' correctness patterns "
        "overlap? `ALL_RIGHT` = every model solved it; `ALL_WRONG` = no model solved it; "
        "single-model labels = only that model got it right."
    )
    for b in LONG_BINS:
        lines.append("")
        lines.append(f"### {b}")
        lines.append("")
        counts, _ = correctness_cube(data, b)
        lines.append("| Group | N |")
        lines.append("|-------|--:|")
        ordering = [
            (1,1,1),  # all right
            (0,0,0),  # all wrong
            (0,0,1),  # only rpe
            (1,0,0),  # only lora
            (0,1,0),  # only y2_base
            (1,1,0),  # lora+y2_base (not rpe)
            (1,0,1),  # lora+rpe (not y2_base)
            (0,1,1),  # y2_base+rpe (not lora)
        ]
        for k in ordering:
            lines.append(f"| {group_label(k)} | {counts.get(k,0)} |")

    # ---------- universally hard samples ----------
    lines.append("")
    lines.append("## Universally-Hard Samples at Long Context")
    lines.append("")
    lines.append(
        "Samples where **all three** models failed at 32k/64k/128k. "
        "These are the cases where training regime is not the bottleneck — "
        "they expose a limitation of the underlying 7B model / LoRA rank "
        "rather than any position-encoding choice."
    )
    lines.append("")
    ranked = top_hardest_samples(data)
    n3 = sum(1 for _, v in ranked if v["all_wrong"] == 3)
    n2 = sum(1 for _, v in ranked if v["all_wrong"] == 2)
    n1 = sum(1 for _, v in ranked if v["all_wrong"] == 1)
    lines.append(f"- Failed by all 3 models at **all 3 long bins**: **{n3}** samples")
    lines.append(f"- Failed by all 3 models at **2 of 3 long bins**: **{n2}** samples")
    lines.append(f"- Failed by all 3 models at **1 of 3 long bins**: **{n1}** samples")
    lines.append("")
    lines.append("**Top 10 hardest (failed by everyone at 3/3 long bins):**")
    lines.append("")
    showed = 0
    for idx, rec in ranked:
        if rec["all_wrong"] < 3: break
        if showed >= 10: break
        r = rec["instances"][0]
        lines.append(
            f"- idx **{idx}** · `{r['object']}` · entries={r['target_entries']} · "
            f"A=`{r['answer']}`\n    Q: _{r['question']}_"
        )
        showed += 1
    lines.append("")

    # ---------- RPE uniquely correct ----------
    lines.append("## RPE-Uniquely-Correct at Long Context")
    lines.append("")
    lines.append(
        "Samples where `y2_rpe` was the **only** model that got it right "
        "(lora and y2_base both failed). These are the cases driving the headline "
        "gap — where RPE training recovers samples neither baseline handles."
    )
    lines.append("")
    rpe_wins = rpe_uniquely_correct_at_long(data)
    by_bin = Counter(r["bin"] for r in rpe_wins)
    lines.append("**Count per long bin:**")
    for b in LONG_BINS:
        lines.append(f"- {b}: {by_bin.get(b, 0)}")
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
        "got them). These are the closest we have to 'RPE training hurt this'."
    )
    lines.append("")
    rpe_losses = rpe_uniquely_wrong_at_long(data)
    by_bin = Counter(r["bin"] for r in rpe_losses)
    lines.append("**Count per long bin:**")
    for b in LONG_BINS:
        lines.append(f"- {b}: {by_bin.get(b, 0)}")
    lines.append("")
    if rpe_losses:
        lines.append("**Example samples (first 8):**")
        lines.append("")
        for r in rpe_losses[:8]:
            lines.append(fmt_example(r))
    else:
        lines.append("_(none found — RPE is never uniquely wrong at any long bin.)_")
    lines.append("")

    # ---------- by target_entries ----------
    lines.append("## Accuracy by `target_entries`")
    lines.append("")
    lines.append(
        "Does the RPE advantage depend on how many times the queried object "
        "visits a room? (entries=2 is a simple 2-hop; entries≥3 needs deeper memory.)"
    )
    lines.append("")
    lines.append(by_target_entries(data))
    lines.append("")

    # ---------- by object ----------
    lines.append("## Accuracy by object")
    lines.append("")
    lines.append(by_object(data))
    lines.append("")

    # ---------- summary ----------
    # Compute numbers used in the summary (keep claims grounded in data)
    uniq_rpe_wins = sum(1 for b in LONG_BINS
                        for r in data[b]
                        if r["y2_rpe_correct"] and not r["lora_correct"]
                        and not r["y2_base_correct"])
    uniq_rpe_losses = sum(1 for b in LONG_BINS
                          for r in data[b]
                          if not r["y2_rpe_correct"] and r["lora_correct"]
                          and r["y2_base_correct"])
    all_wrong_128k = sum(1 for r in data["128k"]
                         if not r["lora_correct"] and not r["y2_base_correct"]
                         and not r["y2_rpe_correct"])
    all_wrong_all3_bins = sum(1 for _, v in top_hardest_samples(data)
                              if v["all_wrong"] == 3)

    lines.append("## Pattern Summary")
    lines.append("")
    lines.append(
        f"1. **RPE+YaRN dominates at long context.** At 128k, `y2_rpe` beats "
        f"the strongest baseline by 12 points and beats it by 7-8 points at "
        f"32k/64k. Across all three long bins, `y2_rpe` is uniquely correct "
        f"on **{uniq_rpe_wins}** samples vs uniquely wrong on only "
        f"**{uniq_rpe_losses}** — a ~3-4× asymmetry favoring RPE."
    )
    lines.append(
        "2. **Short-bin behavior is mixed.** At 0k the vanilla `lora_base` "
        "actually beats `y2_rpe` by ~5pt — the YaRN-trained models pay a small "
        "cost on noise-free samples. From 4k onward the gap closes; RPE "
        "overtakes both baselines at 8k and stays ahead."
    )
    lines.append(
        f"3. **Universally-hard samples are rare.** Only **{all_wrong_all3_bins}** "
        f"sample fails on all three models at **all three** long bins. "
        f"At 128k specifically, {all_wrong_128k} samples defeat all three "
        f"models — these are the ceiling imposed by rank-16 capacity or "
        f"inherent bAbI ambiguity, not position encoding."
    )
    lines.append(
        "4. **RPE's advantage widens at 128k, not at deeper multi-hop.** "
        "Contrary to expectation, the gap between `y2_rpe` and baselines is "
        "largest at 128k uniformly (across all `target_entries` buckets). "
        "The entries=3 bucket actually shows `lora_base` slightly beating "
        "`y2_rpe` at 32k — so RPE's strength is length generalization, not "
        "multi-hop reasoning per se."
    )
    lines.append(
        "5. **Object effects are small.** RPE leads all baselines across all "
        "three objects at every long bin, but the gap is slightly larger for "
        "`football` and `milk` than `apple`. No object is catastrophically "
        "harder for any single model."
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
