#!/usr/bin/env python
"""Generate needle-position eval data for BABILong QA3.

For each (bin × zone) cell, takes the 100 pre-selected hard sample stories from
RMT-team/babilong-1k-samples (0k config, raw bAbI facts), and injects them into
PG19 noise at a controlled zone (beginning/middle/end) using the upstream
`babilong.babilong_utils.NoiseInjectionDataset`.

Output matches the format consumed by `eval_babilong.py` — drop-in compatible.

Decisions (locked 2026-04-13; see Notes/BABILong/Needle_Position_Experiment.md):
  - 3 zones: 0.00-0.33 (beginning), 0.33-0.66 (middle), 0.66-1.00 (end)
  - 9 bins: 0k-128k (full), but short bins (<=8k) skip end zone since facts
    already compete for space
  - 100 samples per cell: 80 hardest multi-entry + 20 single-entry reference
  - Source stories: RMT-team/babilong-1k-samples (0k) -> deterministic reuse
  - Noise: PG19 test split (per official BABILong spec)

Dependencies:
  - Uses vendored babilong/ package (copy source files to experiments/babilong/babilong_src/)
  - OR pip install git+https://github.com/booydar/babilong

Usage:
  python generate_needle_position_eval.py \\
      --selected-indices experiments/babilong/data/eval_needle/selected_100_indices.json \\
      --output-dir experiments/babilong/data/eval_needle \\
      --bins 0k,1k,2k,4k,8k,16k,32k,64k,128k \\
      --noise-dataset pg19

Fast local test with small bins + wikitext:
  python generate_needle_position_eval.py \\
      --selected-indices ... \\
      --bins 1k,2k \\
      --noise-dataset wikitext \\
      --max-samples 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZONES = {
    "beg": (0.00, 0.33),
    "mid": (0.33, 0.66),
    "end": (0.66, 1.00),
}

BIN_TO_TOKENS = {
    # NOTE: "0k" removed — zone control is meaningless without noise, and the
    # previous 0k value (1024) duplicated the 1k bin.
    "1k":   1024,
    "2k":   2048,
    "4k":   4096,
    "8k":   8192,
    "16k":  16384,
    "32k":  32768,
    "64k":  65536,
    # 128k: reduced below 131072 to leave headroom for eval-time prompt
    # overhead (QA3_INSTRUCTION ~110 + QA3_POST_PROMPT ~35 + chat template
    # ~30 + <context> wrapper ~15 ≈ 190 tokens). Without this the eval's
    # left-truncation would eat facts at the very start of the beg-zone.
    "128k": 130700,
}

# Noise-dataset configs: (hf_id, config, split, text_field)
NOISE_CONFIGS = {
    "pg19":      ("pg19",     None,               "test", "text"),
    "wikitext":  ("wikitext", "wikitext-2-raw-v1", "test", "text"),
}

TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Custom TaskDataset wrapper — parses RMT-team 0k `input` into (facts, Q, A)
# ---------------------------------------------------------------------------

class CustomTaskDataset:
    """TaskDataset-compatible wrapper over a list of {facts, question, answer}.

    The upstream `TaskDataset` expects a bAbI-formatted text file with phrase
    numbers. Instead, we pre-parse the RMT-team HF 0k samples (where `input`
    is a space-joined list of bAbI facts) and wrap them in the same interface.
    """
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "facts":      s["facts"],
            "question":   s["question"],
            "answer":     s["answer"],
            "references": [],
        }

    def __len__(self):
        return len(self.samples)


def parse_babi_facts(input_text: str) -> list[str]:
    """Split the RMT-team 0k `input` field into individual bAbI fact sentences.

    Input: 'Mary got the milk. John moved to the bedroom. ...'
    Output: ['Mary got the milk.', 'John moved to the bedroom.', ...]
    """
    # Split on '. ' but keep the period on each fact.
    text = input_text.strip()
    if text.endswith("."):
        text = text[:-1]
    facts = [f.strip() + "." for f in text.split(". ") if f.strip()]
    return facts


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(
    selected_indices_path: str,
    output_dir: str,
    bins: list[str],
    zones: list[str],
    noise_dataset_name: str,
    max_samples: int,
    random_seed: int,
    babilong_src_dir: str | None,
):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Optionally add vendored babilong to path
    if babilong_src_dir:
        sys.path.insert(0, os.path.abspath(babilong_src_dir))
    from babilong.babilong_utils import (
        SentenceSampler,
        NoiseInjectionDataset,
    )

    # Ensure NLTK punkt is available
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading NLTK punkt_tab...")
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # ----------------------------------------------------------------
    # Load selection
    # ----------------------------------------------------------------
    selection = json.loads(Path(selected_indices_path).read_text())
    sel_indices = set(selection["selected_indices"])
    per_sample_meta = {e["idx"]: e for e in selection["per_sample"]}
    print(f"Selected indices: {len(sel_indices)}")

    # ----------------------------------------------------------------
    # Load 999 RMT-team 0k samples (raw bAbI)
    # ----------------------------------------------------------------
    print(f"Loading RMT-team/babilong-1k-samples 0k/qa3 ...")
    ds_0k = load_dataset("RMT-team/babilong-1k-samples", "0k", split="qa3")
    print(f"  loaded {len(ds_0k)} raw stories")

    # Build list of {facts, question, answer, original_idx, meta} for selected indices
    raw_samples = []
    for idx in sorted(sel_indices):
        if idx >= len(ds_0k):
            print(f"  WARN: idx {idx} out of range, skipping")
            continue
        s = ds_0k[idx]
        facts = parse_babi_facts(s["input"])
        raw_samples.append({
            "original_idx": idx,
            "facts":        facts,
            "question":     s["question"].strip(),
            "answer":       s["target"].strip().lower(),
            "meta":         per_sample_meta[idx],
        })
    print(f"  parsed {len(raw_samples)} selected stories")
    if max_samples:
        raw_samples = raw_samples[:max_samples]
        print(f"  truncated to {len(raw_samples)} samples for testing")

    task_ds = CustomTaskDataset(raw_samples)

    # ----------------------------------------------------------------
    # Load noise dataset + build sampler
    # ----------------------------------------------------------------
    hf_id, config, split, text_field = NOISE_CONFIGS[noise_dataset_name]
    print(f"\nLoading noise dataset: {hf_id} {config or ''} split={split} ...")
    if config:
        noise_ds = load_dataset(hf_id, config, split=split)
    else:
        noise_ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    print(f"  loaded {len(noise_ds)} noise documents")

    # Tokenizer
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ----------------------------------------------------------------
    # Generate per (bin × zone)
    # ----------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    manifest_entries = []

    for bin_label in bins:
        sample_size = BIN_TO_TOKENS[bin_label]

        for zone_name in zones:
            start_pct, end_pct = ZONES[zone_name]

            # Per-sample IndexError from NoiseInjectionDataset (below) already
            # handles cases where facts can't fit in the requested zone for a
            # particular story — no pre-skip needed.

            out_path = Path(output_dir) / f"{zone_name}_{bin_label}.json"
            print(f"\n=== Generating {zone_name}_{bin_label}  (size={sample_size}, zone=[{start_pct},{end_pct}])")

            # Fresh noise sampler per cell — independent noise per cell
            noise_sampler = SentenceSampler(
                noise_ds, tokenizer=tokenizer, shuffle=True,
                random_seed=random_seed + hash((bin_label, zone_name)) % 10000,
            )

            # At 1k/2k some samples will fail to fit facts into the target
            # zone — those are caught by the IndexError below.
            ds = NoiseInjectionDataset(
                task_dataset=task_ds,
                noise_sampler=noise_sampler,
                tokenizer=tokenizer,
                task_start_pct=start_pct,
                task_end_pct=end_pct,
                sample_size=sample_size,
                random_seed=random_seed,
            )

            out_records = []
            for i in range(len(task_ds)):
                try:
                    s = ds[i]
                except IndexError as e:
                    # Happens if facts don't fit in zone for this bin
                    print(f"    [{i}] skipped: {e}")
                    continue

                context_text = tokenizer.decode(s["input_tokens"], skip_special_tokens=True)
                user_content = (
                    f"{context_text}\n"
                    f"Question: {s['question'].strip()}\n"
                    f"Answer with only one word."
                )
                assistant_content = s["answer"].strip().lower()

                messages = [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                token_count = len(tokenizer.encode(full_text, add_special_tokens=False))

                # fact_positions from NoiseInjectionDataset are token-offset-ish
                # (indices into background_text array). Convert to relative 0-1.
                n_bg = len(s["background_text"])
                fp_rel = [float(p) / max(1, n_bg) for p in s["fact_positions"]] if n_bg else []

                meta = raw_samples[i]["meta"]
                record = {
                    "messages":       messages,
                    "answer":         assistant_content,
                    "question":       s["question"].strip(),
                    "bin":            bin_label,
                    "zone":           zone_name,
                    "zone_pct":       [start_pct, end_pct],
                    "token_count":    token_count,
                    "original_idx":   raw_samples[i]["original_idx"],
                    "tier":           meta.get("tier"),
                    "object":         meta.get("object"),
                    "target_entries": meta.get("target_entries"),
                    "fact_positions_rel": fp_rel,
                }
                out_records.append(record)
                if (i + 1) % 20 == 0:
                    print(f"    {i+1}/{len(task_ds)} generated", flush=True)

            with open(out_path, "w") as f:
                json.dump(out_records, f)
            print(f"  WROTE {len(out_records)} samples -> {out_path}")
            manifest_entries.append({
                "file":        out_path.name,
                "bin":         bin_label,
                "zone":        zone_name,
                "n_samples":   len(out_records),
                "sample_size": sample_size,
            })

    # ----------------------------------------------------------------
    # Manifest
    # ----------------------------------------------------------------
    manifest = {
        "description": (
            "Needle-position eval: 100 selected stories (80 hardest multi-entry "
            "from prior multi-entry eval + 20 single-entry reference) placed in "
            "PG19 noise at 3 controlled zones across 9 context-length bins. "
            "Used to separate recency-bias effects from raw length effects."
        ),
        "source_stories":    "RMT-team/babilong-1k-samples (0k/qa3) — 100 selected indices",
        "noise_source":      noise_dataset_name,
        "zones":             ZONES,
        "bins_tokens":       {b: BIN_TO_TOKENS[b] for b in bins},
        "tokenizer":         TOKENIZER_NAME,
        "random_seed":       random_seed,
        "selection_file":    selected_indices_path,
        "cells":             manifest_entries,
    }
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n=== MANIFEST -> {manifest_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selected-indices", default="experiments/babilong/data/eval_needle/selected_100_indices.json")
    p.add_argument("--output-dir",       default="experiments/babilong/data/eval_needle")
    p.add_argument("--bins",             default="1k,2k,4k,8k,16k,32k,64k,128k")
    p.add_argument("--zones",            default="beg,mid,end")
    p.add_argument("--noise-dataset",    default="pg19", choices=list(NOISE_CONFIGS.keys()))
    p.add_argument("--max-samples",      type=int, default=0, help="Cap samples per cell for testing (0=all)")
    p.add_argument("--random-seed",      type=int, default=42)
    p.add_argument("--babilong-src-dir", default=None,
                   help="Path to vendored babilong source if not pip-installed (e.g. experiments/babilong/babilong_src)")
    args = p.parse_args()

    generate(
        selected_indices_path=args.selected_indices,
        output_dir=args.output_dir,
        bins=[b.strip() for b in args.bins.split(",") if b.strip()],
        zones=[z.strip() for z in args.zones.split(",") if z.strip()],
        noise_dataset_name=args.noise_dataset,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        babilong_src_dir=args.babilong_src_dir,
    )


if __name__ == "__main__":
    main()
