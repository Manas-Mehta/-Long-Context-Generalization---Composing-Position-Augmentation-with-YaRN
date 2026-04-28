"""
Phase 2.5 — BABILongRetriever smoke test.

Runs our BABILongRetriever (the prompt-patched FullHeadRetriever) on ONE
detection sample at one bin, on vanilla Qwen2.5-7B-Instruct, and verifies:
  1. The custom get_prompt produces a valid BABILong-format prompt.
  2. compose_scoring_prompt locates the query span and per-sentence spans.
  3. The forward pass + calibration runs end-to-end without error.
  4. Output has the expected (n_layers, n_heads) shape per doc.
  5. Gold-doc attention is meaningfully larger than non-gold-doc attention
     (vanilla Qwen has functional retrieval circuitry, so this should be true).

Catches BABILongRetriever bugs before they cost us compute across 36 cells.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/scratch/mm14444/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

from babilong_retriever import BABILongRetriever  # noqa: E402


MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATA_DIR = Path("composable_cot/retrieval_head_analysis/data")
TEST_BIN = "8k"  # short enough to run fast, long enough to be representative


def main():
    detset_path = DATA_DIR / f"detection_set_{TEST_BIN}.json"
    print(f"Loading detection set: {detset_path}")
    detset = json.loads(detset_path.read_text())
    print(f"  {len(detset)} entries; using sample[0]")

    sample = detset[0]
    query = sample["question"]
    docs = sample["paragraphs"]
    gt_doc_ids = set(sample["gt_docs"])

    print(f"  Story: {sample['story_idx']}, bin: {sample['bin']}")
    print(f"  Question: {query}")
    print(f"  Docs: {len(docs)}, gold: {len(gt_doc_ids)}")
    print()

    print(f"Loading {MODEL} via BABILongRetriever...")
    retriever = BABILongRetriever(model_name_or_path=MODEL)
    print(f"  Loaded.")
    print()

    # Step 1: prompt sanity check
    print("=" * 60)
    print("STEP 1 — get_prompt sanity")
    print("=" * 60)
    prompt = retriever.get_prompt(query, docs)
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Prompt prefix (first 400 chars):")
    print(repr(prompt[:400]))
    print(f"\nPrompt suffix (last 400 chars):")
    print(repr(prompt[-400:]))
    assert "<context>" in prompt, "missing <context>"
    assert "</context>" in prompt, "missing </context>"
    assert f"Question: {query}" in prompt, "missing Question line"
    assert "<|im_start|>user" in prompt and "<|im_end|>" in prompt
    print("\n✓ prompt structure looks right")
    print()

    # Step 2: span finder sanity check
    print("=" * 60)
    print("STEP 2 — compose_scoring_prompt span finder")
    print("=" * 60)
    prompt2, token_ids, query_span, doc_spans = retriever.compose_scoring_prompt(query, docs)
    assert prompt == prompt2
    print(f"Total tokens: {len(token_ids)}")
    print(f"Query span: {query_span} ({query_span[1] - query_span[0] + 1} tokens)")
    print(f"Doc spans found: {len(doc_spans)}")
    nonzero = sum(1 for (s, e) in doc_spans if e >= s)
    degenerate = len(doc_spans) - nonzero
    print(f"  Non-degenerate: {nonzero}/{len(doc_spans)}")
    if degenerate:
        print(f"  ⚠️  {degenerate} sentences could not be located (whitespace drift)")
    # Show first 3 doc spans + the corresponding sentence prefix
    print(f"\n  First 3 doc spans:")
    for i in range(min(3, len(docs))):
        s, e = doc_spans[i]
        print(f"    doc[{i}] tokens [{s}, {e}] (len {e-s+1})  text: {repr(docs[i]['paragraph_text'][:80])}")
    print()

    # Step 3: forward pass + calibration
    print("=" * 60)
    print("STEP 3 — forward pass + calibration (this takes ~30s on H200)")
    print("=" * 60)
    per_doc_score_tensors = retriever.score_docs_per_head_for_detection(query, docs)
    print(f"  Returned {len(per_doc_score_tensors)} doc score tensors")
    # Pick the first doc, look at its tensor
    first_doc_id = next(iter(per_doc_score_tensors))
    first_tensor = per_doc_score_tensors[first_doc_id]
    print(f"  first doc: {first_doc_id}, tensor shape: {tuple(first_tensor.shape)}")
    assert first_tensor.shape == (28, 32), f"unexpected shape {first_tensor.shape}"
    print(f"  ✓ shape (n_layers=28, n_heads=32) per doc")
    print()

    # Step 4: gold-vs-noise comparison
    print("=" * 60)
    print("STEP 4 — gold-doc vs distractor attention comparison")
    print("=" * 60)
    # Stack into (n_docs, 28, 32). Order matches the docs input.
    stacked = torch.stack([per_doc_score_tensors[d["idx"]] for d in docs])  # (n_docs, 28, 32)
    print(f"  Stacked shape: {tuple(stacked.shape)}")
    is_gold = torch.tensor([d["idx"] in gt_doc_ids for d in docs])
    gold_scores = stacked[is_gold].sum(dim=0)  # (28, 32) — sum over gold docs
    nogold_scores = stacked[~is_gold].sum(dim=0)  # (28, 32) — sum over distractors
    n_gold = is_gold.sum().item()
    n_nogold = (~is_gold).sum().item()
    print(f"  {n_gold} gold docs, {n_nogold} distractors")

    # Per-head: gold attention per gold-doc vs distractor attention per distractor
    gold_per_head_per_gold = gold_scores / max(n_gold, 1)
    nogold_per_head_per_distractor = nogold_scores / max(n_nogold, 1)
    ratio = gold_per_head_per_gold / (nogold_per_head_per_distractor.abs() + 1e-9)

    # Top 10 heads by gold attention
    flat = gold_per_head_per_gold.flatten()
    top_idx = flat.argsort(descending=True)[:10]
    print(f"\n  Top 10 heads by mean gold-doc attention:")
    print(f"    {'rank':>4}  {'head':>8}  {'mean_gold':>10}  {'mean_dist':>10}  {'ratio':>8}")
    for rank, idx in enumerate(top_idx.tolist()):
        layer = idx // 32
        head = idx % 32
        g = gold_per_head_per_gold[layer, head].item()
        d = nogold_per_head_per_distractor[layer, head].item()
        print(f"    {rank+1:>4}  {layer:>2}-{head:<3}  {g:>10.4f}  {d:>10.4f}  {g/(d+1e-9):>8.2f}")

    # Sanity: top-K heads should attend more to gold than distractors
    top1_layer, top1_head = top_idx[0].item() // 32, top_idx[0].item() % 32
    top1_gold = gold_per_head_per_gold[top1_layer, top1_head].item()
    top1_dist = nogold_per_head_per_distractor[top1_layer, top1_head].item()
    if top1_gold > top1_dist:
        print(f"\n  ✓ Top head ({top1_layer}-{top1_head}): gold > distractor "
              f"({top1_gold:.4f} > {top1_dist:.4f})")
    else:
        print(f"\n  ⚠️  Top head ({top1_layer}-{top1_head}): gold ≤ distractor "
              f"({top1_gold:.4f} vs {top1_dist:.4f})")
        print("     This is suspicious — the highest-attending head should attend to gold more than noise.")

    # Cross-check vs published Qwen QR top-16 (layers 15-20 cluster)
    PUBLISHED_TOP16 = [
        (16, 19), (16, 2), (16, 20), (16, 14), (16, 0), (15, 24), (16, 18), (17, 18),
        (16, 1), (19, 18), (19, 19), (18, 16), (16, 17), (19, 25), (19, 17), (20, 21),
    ]
    print(f"\n  Published Qwen top-16 (BEIR-NQ derived) gold-attention scores:")
    for layer, head in PUBLISHED_TOP16:
        g = gold_per_head_per_gold[layer, head].item()
        d = nogold_per_head_per_distractor[layer, head].item()
        print(f"    {layer:>2}-{head:<3}  gold={g:>8.4f}  dist={d:>8.4f}  ratio={g/(d+1e-9):>6.2f}")
    pub_gold_mean = sum(gold_per_head_per_gold[l, h].item() for l, h in PUBLISHED_TOP16) / 16
    pub_dist_mean = sum(nogold_per_head_per_distractor[l, h].item() for l, h in PUBLISHED_TOP16) / 16
    print(f"  Published top-16 mean: gold={pub_gold_mean:.4f}, dist={pub_dist_mean:.4f}, "
          f"ratio={pub_gold_mean/(pub_dist_mean+1e-9):.2f}")

    print()
    print("=" * 60)
    print("Phase 2.5 smoke test PASSED" if top1_gold > top1_dist else "Phase 2.5 smoke test FLAGGED")
    print("=" * 60)


if __name__ == "__main__":
    main()
