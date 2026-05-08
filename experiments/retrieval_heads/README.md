# Retrieval-head probe (§6)

Mechanistic probe that compares query-aware retrieval heads (QR heads,
Wu et al. 2024) across our four trained conditions
(`zero_shot`, `lora_base`, `y2_base`, `y2_rpe_cur_L16k`) and asks whether
the §4 accuracy ranking is reflected in the retrieval-head structure.

## External dependency — QRHead

This experiment depends on the **published QRHead code** by Wu et al.
(2024), which is **not vendored** in this repo. You need to clone it
separately:

```bash
# Outside this repo
git clone https://github.com/<owner>/QRHead.git /path/to/QRHead
pip install -e /path/to/QRHead
export QRHEAD_REPO=/path/to/QRHead
```

`scripts/phase3_run_detection.py` adds `${QRHEAD_REPO}/exp_scripts/detection/`
to `sys.path` and imports `get_doc_scores_per_head` and `score_heads` verbatim
from `detect_qrhead_lme.py`. We do not modify QRHead's aggregation math.

`scripts/babilong_retriever.py` subclasses QRHead's `FullHeadRetriever`
(from `qrretriever.attn_retriever`) so detection runs on the BABILong QA3
prompt format instead of QRHead's default Wikipedia/QA wrapper.

## Layout

```
retrieval_heads/
  scripts/
    build_detection_set.py        Build the 60-story detection set from BABILong
    babilong_retriever.py         QRHead subclass for BABILong prompt
    phase3_run_detection.py       Run QR detection across 9 bins per condition
    merge_lora.py                 Merge LoRA into Qwen2.5-7B for trained conds
    analyze_*.py                  Eight analysis scripts
                                   (overlaps, per-sample, head structure,
                                    published reference, retrieval recall)
    dump_head_identities.py       Snapshot top-16 heads per (condition, bin)
    present_distributions.py      Score-distribution figures
    simple_analysis.py            One-shot table of all numbers
    analyze_per_sample.py         Tracked variant
    analyze_head_structure.py     Tracked variant
  analysis/                       Output CSVs + figures (committed)
  hpc/                            SLURM templates
  data/                           Detection set (gitignored)
  results/                        Per-bin detection outputs (gitignored)
```

## Reproduce

```bash
# 0. Set QRHead repo location
export QRHEAD_REPO=/path/to/QRHead

# 1. Build the 60-story detection set (one-time)
python experiments/retrieval_heads/scripts/build_detection_set.py \
    --selected experiments/babilong/analysis/multi_entry_eval/qr_selection/selected_60_stories.json \
    --eval-dir experiments/babilong/data/eval_multi_entry \
    --output-dir experiments/retrieval_heads/data

# 2. Merge LoRAs for the trained conditions
python experiments/retrieval_heads/scripts/merge_lora.py \
    --condition y2_rpe_cur_L16k

# 3. Run detection across all 9 bins per condition
python experiments/retrieval_heads/scripts/phase3_run_detection.py \
    --condition  y2_rpe_cur_L16k \
    --output-dir experiments/retrieval_heads/results

# 4. Analyse — e.g. simple overlap table at 128K
python experiments/retrieval_heads/scripts/simple_analysis.py
```

The §6 figures referenced in the paper (`F1_top16_mean.png`, `F2_overlap.png`,
`F3_ranking.png`) are produced by `present_distributions.py` and
`simple_analysis.py` and live in `analysis/figures/`.
