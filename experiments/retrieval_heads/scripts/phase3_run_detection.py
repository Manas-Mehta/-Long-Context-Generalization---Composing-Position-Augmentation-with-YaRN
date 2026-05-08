"""
Phase 3 — run QR detection across all 9 bins for one condition.

This wrapper does ONLY orchestration:
  1. Build BABILongRetriever (our subclass — necessary for BABILong prompt
     format per prof guidance; everything else inherited from QRHead).
  2. For each bin, call QRHead's published functions verbatim:
       - get_doc_scores_per_head(retriever, data_instances)
       - score_heads(doc_scores_per_head, data_instances)
     These come from /scratch/mm14444/QRHead/exp_scripts/detection/detect_qrhead_lme.py.
  3. Save outputs per bin (incremental, with skip-if-exists for resume).

NO custom math, NO custom aggregation. Aggregation is QRHead's published
score_heads function unchanged. If 128K turns out too slow, we'll discuss
optimization with the prof — not by default.

Outputs (under {output_dir}/{condition}/):
  {bin}_head_scores.json     — QRHead's score_heads output, sorted desc
  {bin}_doc_scores_per_head.pt — raw per-(qid, doc_id) tensor dict from QRHead
                                  (saved so we can later compute any per-sample
                                   stat we need for Phase 4d without re-running)
  {bin}_meta.json            — {condition, bin, n_samples, n_layers, n_heads,
                                story_id_order, elapsed_seconds, ...}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# HF cache config — must precede transformers import.
os.environ.setdefault("HF_HOME", "/scratch/mm14444/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

# QRHead's published detection functions. We import them directly and pass our
# BABILongRetriever instance — duck typing on score_docs_per_head_for_detection.
QRHEAD_DETECT_DIR = "/scratch/mm14444/QRHead/exp_scripts/detection"
if QRHEAD_DETECT_DIR not in sys.path:
    sys.path.insert(0, QRHEAD_DETECT_DIR)
from detect_qrhead_lme import get_doc_scores_per_head, score_heads  # noqa: E402

from babilong_retriever import BABILongRetriever  # noqa: E402


BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


def run_one_bin(retriever, detset_path, output_dir, condition, bin_label):
    """Call QRHead's get_doc_scores_per_head + score_heads. Save outputs."""
    head_scores_path = output_dir / f"{bin_label}_head_scores.json"
    doc_scores_path = output_dir / f"{bin_label}_doc_scores_per_head.pt"
    meta_path = output_dir / f"{bin_label}_meta.json"

    if head_scores_path.exists():
        print(f"  [{bin_label}] SKIP — output exists at {head_scores_path}",
              flush=True)
        return

    data_instances = json.loads(detset_path.read_text())
    n_samples = len(data_instances)
    print(f"  [{bin_label}] {n_samples} samples", flush=True)

    bin_start = time.time()

    # Step 1: per-doc QR scoring via QRHead's published function (uses our
    # retriever's score_docs_per_head_for_detection).
    print(f"  [{bin_label}] computing per-doc scores (forward passes)...",
          flush=True)
    doc_scores_per_head = get_doc_scores_per_head(retriever, data_instances)
    fwd_elapsed = time.time() - bin_start

    # Step 2: per-head aggregation via QRHead's published function.
    print(f"  [{bin_label}] aggregating per-head scores "
          f"(QRHead's score_heads, ~Python loop)...", flush=True)
    agg_start = time.time()
    head_scores_list = score_heads(doc_scores_per_head, data_instances)
    agg_elapsed = time.time() - agg_start
    bin_elapsed = time.time() - bin_start

    # Save outputs.
    head_scores_path.write_text(json.dumps(head_scores_list, indent=2))
    # Save raw per-doc tensors so Phase 4 can compute any per-sample stat
    # without re-running detection. Stored as a torch dict.
    torch.save(doc_scores_per_head, doc_scores_path)

    # Pull n_layers / n_heads from the first tensor for metadata.
    first_qid = next(iter(doc_scores_per_head))
    first_doc = next(iter(doc_scores_per_head[first_qid]))
    sample_tensor = doc_scores_per_head[first_qid][first_doc]
    n_layers, n_heads = sample_tensor.shape

    story_id_order = [d["story_idx"] for d in data_instances]

    meta_path.write_text(json.dumps({
        "condition": condition,
        "bin": bin_label,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "story_id_order": story_id_order,
        "fwd_elapsed_seconds": fwd_elapsed,
        "agg_elapsed_seconds": agg_elapsed,
        "total_elapsed_seconds": bin_elapsed,
        "top10_heads": [h for h, _ in head_scores_list[:10]],
    }, indent=2))

    print(f"  [{bin_label}] DONE in {bin_elapsed/60:.1f} min "
          f"(fwd {fwd_elapsed/60:.1f}, agg {agg_elapsed/60:.1f}). "
          f"top-5: {[h for h, _ in head_scores_list[:5]]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    help="condition label (vanilla_qwen | lora_base | "
                         "y2_base | y2_rpe_cur_L16k)")
    ap.add_argument("--model-path", required=True,
                    help="HF model id OR absolute path to a merged checkpoint dir")
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
    print(f"  model:  {args.model_path}")
    print(f"  output: {out_dir}")
    print(f"  bins:   {bins_to_run}")
    print(f"  using QRHead's published get_doc_scores_per_head + score_heads")
    print("=" * 70, flush=True)

    # Pass config as a dict so model_name_or_path stays as our merged path
    # (the YAML lookup would otherwise overwrite it with vanilla Qwen).
    config_dict = {
        "attn_head_set": "full_heads",
        "model_base_class": "Qwen2.5-7B-Instruct",
        "model_name_or_path": args.model_path,
    }

    print(f"Loading model via BABILongRetriever...", flush=True)
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
        run_one_bin(retriever, detset_path, out_dir, args.condition, bin_label)

    print(f"\nAll bins complete for condition '{args.condition}'.", flush=True)


if __name__ == "__main__":
    main()
