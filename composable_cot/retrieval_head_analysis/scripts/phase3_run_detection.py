"""
Phase 3 — run BABILongRetriever-based QR detection across all 9 bins for one
condition, on the 60-story BABILong-derived detection set.

Per-bin output (under {output_dir}/{condition}/):
  {bin}_head_scores.json   sorted [(layer-head, score), ...] (mean over samples
                            of sum-over-gold-docs per head)
  {bin}_per_sample.npz     scores: (n_samples, n_layers, n_heads) — per-sample
                            sum-over-gold per head; story_id_order: (n_samples,)
  {bin}_meta.json          {condition, bin, n_samples, n_layers, n_heads,
                            story_id_order, elapsed_seconds, ...}

Behavior:
  - Saves output incrementally per bin. On re-run, skips bins whose
    head_scores.json already exists. So a slurm timeout costs at most one bin's
    worth of work.
  - Defensive logging at startup: rope_scaling, n_layers, n_heads, dtype.
  - Per-sample logging: degenerate-span count, gold count, elapsed.

Math note (vectorized aggregation):
  QRHead's published `detect_qrhead_lme.py` aggregates per-head scores via
  three nested Python loops. At our 128K bin (~3700 sentence-docs/sample) that
  would be ~200M scalar extractions per bin → ~18 hours total. We do the same
  math by stacking per-doc tensors per sample and summing over gold indices in
  tensor ops. Mathematically identical up to fp16 sum-order noise.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# HF cache config — must be set before importing transformers.
os.environ.setdefault("HF_HOME", "/scratch/mm14444/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from babilong_retriever import BABILongRetriever  # noqa: E402


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


def run_one_bin(retriever, detset_path, output_dir, condition, bin_label,
                n_layers, n_heads):
    """Run detection for one bin. Save aggregated head scores + per-sample tensors."""
    head_scores_path = output_dir / f"{bin_label}_head_scores.json"
    per_sample_path = output_dir / f"{bin_label}_per_sample.npz"
    meta_path = output_dir / f"{bin_label}_meta.json"

    if head_scores_path.exists():
        print(f"  [{bin_label}] SKIP — output already exists", flush=True)
        return

    detset = json.loads(detset_path.read_text())
    n_samples = len(detset)
    print(f"  [{bin_label}] {n_samples} samples", flush=True)

    bin_start = time.time()

    per_sample_gold_sum = torch.zeros(n_samples, n_layers, n_heads,
                                      dtype=torch.float32)
    story_id_order = []

    total_degenerate = 0
    for sample_idx, sample in enumerate(detset):
        story_id_order.append(sample["story_idx"])
        query = sample["question"]
        docs = sample["paragraphs"]
        gt_doc_ids = set(sample["gt_docs"])

        sample_start = time.time()

        # Run BABILongRetriever's per-doc QR scoring (calibrated; covers the
        # FA2 + DynamicCacheWithQuery + manual Q·K^T softmax + null-query
        # subtraction pipeline).
        per_doc_score_tensors = retriever.score_docs_per_head_for_detection(
            query, docs)

        # Vectorized aggregation: stack per-doc tensors into
        # (n_docs, n_layers, n_heads), then sum over gold indices.
        stacked = torch.stack(
            [per_doc_score_tensors[d["idx"]] for d in docs])  # (n_docs, L, H)
        gold_mask = torch.tensor(
            [d["idx"] in gt_doc_ids for d in docs], dtype=torch.bool)
        per_sample_gold_sum[sample_idx] = stacked[gold_mask].sum(
            dim=0).cpu().float()

        # Defensive: per-doc tensors with all-zero values across heads
        # indicate a degenerate-span sentence (couldn't locate in prompt).
        # Count them — the BABILongRetriever already prints a warning.
        sample_elapsed = time.time() - sample_start
        if sample_idx % 10 == 0 or sample_idx == n_samples - 1:
            print(f"    [{bin_label}] {sample_idx + 1:3d}/{n_samples} "
                  f"docs={len(docs):4d} gold={int(gold_mask.sum().item()):3d} "
                  f"({sample_elapsed:5.1f}s)", flush=True)

    bin_elapsed = time.time() - bin_start

    # Aggregate to per-head ranking: mean across samples.
    per_head_score = per_sample_gold_sum.mean(dim=0)  # (L, H)
    head_scores_list = []
    for layer in range(n_layers):
        for head in range(n_heads):
            head_scores_list.append(
                (f"{layer}-{head}", per_head_score[layer, head].item()))
    head_scores_list.sort(key=lambda x: x[1], reverse=True)

    head_scores_path.write_text(json.dumps(head_scores_list, indent=2))
    np.savez_compressed(per_sample_path,
                        scores=per_sample_gold_sum.numpy(),
                        story_id_order=np.array(story_id_order))
    meta_path.write_text(json.dumps({
        "condition": condition,
        "bin": bin_label,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "story_id_order": story_id_order,
        "elapsed_seconds": bin_elapsed,
        "top10_heads": [h for h, _ in head_scores_list[:10]],
    }, indent=2))

    print(f"  [{bin_label}] DONE in {bin_elapsed/60:.1f} min. "
          f"top-5: {[h for h, _ in head_scores_list[:5]]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    help="condition label (e.g. vanilla_qwen, lora_base, "
                         "y2_base, y2_rpe_cur_L16k)")
    ap.add_argument("--model-path", required=True,
                    help="HF model id OR absolute path to merged checkpoint dir")
    ap.add_argument("--data-dir",
                    default="composable_cot/retrieval_head_analysis/data")
    ap.add_argument("--output-dir",
                    default="composable_cot/retrieval_head_analysis/results")
    ap.add_argument("--bins", default=",".join(BINS),
                    help="comma-separated bins to run (default: all 9)")
    args = ap.parse_args()

    bins_to_run = args.bins.split(",")
    out_dir = Path(args.output_dir) / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print(f"Phase 3 detection — condition: {args.condition}")
    print(f"  model:      {args.model_path}")
    print(f"  output:     {out_dir}")
    print(f"  bins:       {bins_to_run}")
    print("=" * 70, flush=True)

    # Build the config dict directly so model_name_or_path stays as we passed
    # it (the YAML lookup would otherwise overwrite it with vanilla Qwen).
    config_dict = {
        "attn_head_set": "full_heads",
        "model_base_class": "Qwen2.5-7B-Instruct",
        "model_name_or_path": args.model_path,
    }

    print(f"Loading model via BABILongRetriever (config dict)...", flush=True)
    retriever = BABILongRetriever(config_or_config_path=config_dict)

    n_layers = retriever.llm.config.num_hidden_layers
    n_heads = retriever.llm.config.num_attention_heads
    rope_scaling = getattr(retriever.llm.config, "rope_scaling", None)
    print(f"  n_layers:     {n_layers}", flush=True)
    print(f"  n_heads:      {n_heads}", flush=True)
    print(f"  rope_scaling: {rope_scaling}", flush=True)
    print(f"  dtype:        {retriever.llm.dtype}", flush=True)

    for bin_label in bins_to_run:
        if bin_label not in BINS:
            print(f"  WARN: unknown bin '{bin_label}', skipping", flush=True)
            continue
        detset_path = data_dir / f"detection_set_{bin_label}.json"
        if not detset_path.exists():
            print(f"  WARN: {detset_path} missing, skipping", flush=True)
            continue
        print(f"\n--- BIN: {bin_label} ---", flush=True)
        run_one_bin(retriever, detset_path, out_dir, args.condition,
                    bin_label, n_layers, n_heads)

    print(f"\nAll bins complete for condition '{args.condition}'.", flush=True)


if __name__ == "__main__":
    main()
