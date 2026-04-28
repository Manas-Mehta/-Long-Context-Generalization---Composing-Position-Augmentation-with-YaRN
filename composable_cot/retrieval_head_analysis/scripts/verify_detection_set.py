"""
Verify the detection set's gold-marking is correct.

Three checks:
  1. EYEBALL — dump gold + non-gold sentences from a few stories at one bin,
     so you can visually confirm gold = bAbI facts and non-gold = PG19 noise.
  2. CROSS-BIN — the same story has the same bAbI facts at every bin (just
     embedded in different noise). Verify gold-fact sets are stable across
     bins by comparing to the 0K reference for the same story.
  3. COUNT SANITY — gold count per story should be roughly bin-independent
     (some variance expected when PG19 fragments inflate sentence counts).
"""

import argparse
import json
import re
import sys
from pathlib import Path


def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()


def load_bin(data_dir: Path, bin_label: str) -> list[dict]:
    return json.loads((data_dir / f"detection_set_{bin_label}.json").read_text())


def gold_sentences(sample: dict) -> list[str]:
    gold_ids = set(sample["gt_docs"])
    return [p["paragraph_text"] for p in sample["paragraphs"] if p["idx"] in gold_ids]


def non_gold_sentences(sample: dict) -> list[str]:
    gold_ids = set(sample["gt_docs"])
    return [p["paragraph_text"] for p in sample["paragraphs"] if p["idx"] not in gold_ids]


# --------------------------------------------------------------------------

def eyeball_check(data_dir: Path, bin_label: str, n_stories: int = 2):
    """Print gold + a few non-gold sentences for a few stories."""
    print(f"\n{'='*70}")
    print(f"EYEBALL CHECK — {bin_label}")
    print(f"{'='*70}")
    data = load_bin(data_dir, bin_label)
    for s in data[:n_stories]:
        print(f"\nStory {s['story_idx']}, bin {s['bin']}")
        print(f"  Question: {s['question']}")
        print(f"  Total sentences: {s['n_paragraphs']}, marked gold: {s['n_gold']}")
        gold = gold_sentences(s)
        print(f"  First 5 gold (should be bAbI fact-shaped):")
        for g in gold[:5]:
            print(f"    GOLD : {repr(g[:100])}")
        non_gold = non_gold_sentences(s)
        print(f"  First 5 non-gold (should be PG19 prose):")
        for ng in non_gold[:5]:
            print(f"    NOISE: {repr(ng[:100])}")


# --------------------------------------------------------------------------

def cross_bin_check(data_dir: Path, bins: list[str]):
    """For each story, compare gold-fact text across bins.

    Returns a per-story diagnosis: which bins missed which facts (relative to 0K).
    """
    print(f"\n{'='*70}")
    print(f"CROSS-BIN CONSISTENCY")
    print(f"{'='*70}")
    print("Comparing gold facts per story across bins.")
    print("Reference: 0K bin (pure bAbI, all sentences are facts).")
    print()

    # Load all bins.
    per_bin = {b: load_bin(data_dir, b) for b in bins}
    # Index 0K by story_idx for quick lookup.
    zerok_by_story = {s["story_idx"]: s for s in per_bin["0k"]}

    # For each story, compute the set of unique gold-fact strings (normalized) at each bin
    # and compare against 0K.
    print(f"{'story':>6}  {'0k_facts':>9}  {'1k':>4} {'2k':>4} {'4k':>4} {'8k':>4} "
          f"{'16k':>4} {'32k':>4} {'64k':>4} {'128k':>4}   missing_at_long_bins")
    print("-" * 100)

    issues = []
    for story in per_bin["0k"]:
        sidx = story["story_idx"]
        # Treat ALL 0K sentences as the canonical fact list (deduplicated).
        ref_facts_norm = {normalize(s) for s in gold_sentences(story)}
        n_ref = len(ref_facts_norm)

        per_bin_counts = {}
        long_misses = []
        for b in bins:
            if b == "0k":
                continue
            # Find the same story_idx in this bin.
            match = next((x for x in per_bin[b] if x["story_idx"] == sidx), None)
            if match is None:
                per_bin_counts[b] = "—"
                continue
            bin_facts_norm = {normalize(s) for s in gold_sentences(match)}
            # We use substring containment for matching, so normalize each bin's
            # gold sentence and check against the ref fact set.
            recovered = set()
            for bnorm in bin_facts_norm:
                for ref in ref_facts_norm:
                    if ref in bnorm:
                        recovered.add(ref)
            per_bin_counts[b] = f"{len(recovered)}/{n_ref}"
            if b in ("32k", "64k", "128k"):
                missing = ref_facts_norm - recovered
                if missing:
                    long_misses.extend(list(missing)[:1])  # show one example

        flag = ""
        if any(per_bin_counts.get(b) != "—" and "/" in per_bin_counts[b] and
               int(per_bin_counts[b].split("/")[0]) < n_ref * 0.9
               for b in ("32k", "64k", "128k")):
            flag = " ⚠️"
        print(f"{sidx:>6}  {n_ref:>9}  "
              f"{per_bin_counts.get('1k','—'):>4} {per_bin_counts.get('2k','—'):>4} "
              f"{per_bin_counts.get('4k','—'):>4} {per_bin_counts.get('8k','—'):>4} "
              f"{per_bin_counts.get('16k','—'):>4} {per_bin_counts.get('32k','—'):>4} "
              f"{per_bin_counts.get('64k','—'):>4} {per_bin_counts.get('128k','—'):>4}{flag}"
              f"   {long_misses[0][:60] if long_misses else ''}")
        if long_misses:
            issues.append((sidx, long_misses))

    print()
    if issues:
        print(f"⚠️  {len(issues)} stories have facts missing from at least one long bin.")
        print(f"   First few missing-fact examples printed above.")
    else:
        print("✅ Every story recovers its full 0K fact list at every long bin.")


# --------------------------------------------------------------------------

def count_sanity(data_dir: Path, bins: list[str]):
    print(f"\n{'='*70}")
    print(f"COUNT SANITY — gold count per bin (mean, min, max)")
    print(f"{'='*70}")
    print(f"{'bin':>5}   {'n_stories':>10}  {'mean_gold':>10}  {'min':>4}  {'max':>4}  "
          f"{'mean_paras':>11}")
    for b in bins:
        data = load_bin(data_dir, b)
        gcs = [s["n_gold"] for s in data]
        pcs = [s["n_paragraphs"] for s in data]
        if not gcs:
            print(f"{b:>5}   (empty)")
            continue
        print(f"{b:>5}   {len(data):>10}  {sum(gcs)/len(gcs):>10.1f}  "
              f"{min(gcs):>4}  {max(gcs):>4}  {sum(pcs)/len(pcs):>11.1f}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default="composable_cot/retrieval_head_analysis/data")
    ap.add_argument("--eyeball-bins", default="1k,32k,128k",
                    help="bins to dump samples from (comma-separated)")
    ap.add_argument("--n-stories", type=int, default=2,
                    help="how many stories per bin to dump in eyeball check")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found", file=sys.stderr)
        sys.exit(1)

    bins = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]
    eyeball_bins = args.eyeball_bins.split(",")

    count_sanity(data_dir, bins)
    cross_bin_check(data_dir, bins)
    for b in eyeball_bins:
        eyeball_check(data_dir, b, n_stories=args.n_stories)


if __name__ == "__main__":
    main()
