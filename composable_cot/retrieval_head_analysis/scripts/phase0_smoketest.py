"""
Phase 0 smoke test — verify QRHead's pipeline runs on vanilla Qwen2.5-7B-Instruct
inside our `qrhead` conda env on HPC.

Success criterion: scores returned for both passages, with test1 (Levikha,
200,000 people) clearly higher than test0 (Kushva, 1,000 people) — the model
should send more attention from "largest population" toward the passage that
mentions a large number.

The exact numbers WILL differ from the README's published `{'test0': 0.63,
'test1': 1.17}` — that example is for Llama-3.1-8B; we run Qwen2.5-7B which
has different attention. Direction is what matters: test1 > test0.
"""

import os

# Force HuggingFace offline mode and point at the HPC cache.
os.environ.setdefault("HF_HOME", "/scratch/mm14444/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from qrretriever.attn_retriever import QRRetriever  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"

print(f"Loading {MODEL} via QRRetriever (this takes ~60s on H200)...")
retriever = QRRetriever(model_name_or_path=MODEL)
print("Model loaded.\n")

query = "Which town in Nizhnyaya has the largest population?"
docs = [
    {
        "idx": "test0",
        "title": "Kushva",
        "paragraph_text": "Kushva is the largest town in Nizhnyaya. It has a population of 1,000.",
    },
    {
        "idx": "test1",
        "title": "Levikha",
        "paragraph_text": "Levikha is a bustling town in Nizhnyaya. It has a population of 200,000.",
    },
]

print(f"Query: {query}")
scores = retriever.score_docs(query, docs)
print("Scores:", scores)

assert "test0" in scores and "test1" in scores, "missing scores in output"
ok = scores["test1"] > scores["test0"]
print()
print(f"test1 > test0? {ok}")
print("PASS" if ok else "FAIL — wrong direction; pipeline may be broken")
