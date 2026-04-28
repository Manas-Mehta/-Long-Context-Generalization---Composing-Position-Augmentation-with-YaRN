"""
Phase 1 — build the BABILong-derived detection set for QR-head detection.

Inputs:
  - composable_cot/BABIlong/analysis/multi_entry_eval/qr_selection/selected_60_stories.json
      The 60 story indices we locked.
  - composable_cot/BABIlong/data/eval_multi_entry/{bin}.json
      The 305-sample multi-entry eval data, one file per bin (HPC-only path).

Outputs:
  - composable_cot/retrieval_head_analysis/data/detection_set_{bin}.json
      One file per bin, each containing 60 entries in QRHead's LME-style format:
      {idx, question, paragraphs: [{idx, paragraph_text, title}, ...], gt_docs: [...]}.

Method (Option E — no regex, exact string match against 0K reference):
  1. For each story, the 0K bin's haystack contains ONLY bAbI fact sentences
     (no PG19 noise). Split it into sentences — that's the canonical fact list
     for that story.
  2. For longer bins (1K..128K) of the same story, split the noisy haystack into
     sentences. A sentence is "gold" iff it exact-matches one of the 0K facts.
  3. Each individual sentence is a separate "document" (per prof guidance).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# --------------------------------------------------------------------------
# Sentence splitting

# Splits on ., !, ? followed by whitespace OR end-of-string. Robust enough for
# bAbI's deterministic punctuation; handles PG19's mixed quoting reasonably.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Returns trimmed non-empty pieces."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# --------------------------------------------------------------------------
# Haystack / question extraction (mirrors eval_babilong.py:build_prompt)

def extract_haystack_and_question(sample: dict) -> tuple[str, str]:
    """Return (haystack, question) parsed from a BABILong eval sample."""
    if "messages" in sample:
        user_content = sample["messages"][0]["content"]
        question = sample.get("question", "").strip()
    else:
        user_content = sample["input"].strip()
        question = sample.get("question", "").strip()

    parts = user_content.rsplit("\nQuestion:", 1)
    haystack = parts[0].strip() if len(parts) > 1 else user_content.strip()

    if not question and len(parts) > 1:
        q_line = parts[1].replace("\nAnswer with only one word.", "").strip()
        question = q_line.split("\n")[0].strip()

    return haystack, question


# --------------------------------------------------------------------------
# Whitespace-tolerant exact-match

def normalize(s: str) -> str:
    """Collapse all whitespace runs to single spaces. Helps if the same fact
    appears with slightly different surrounding whitespace at different bins."""
    return re.sub(r'\s+', ' ', s).strip()


def build_one_story(
    story_idx: int,
    bin_data: dict,
    fact_sentences_norm: set[str],
    bin_label: str,
) -> dict:
    """Build one QR detection entry for a (story_idx, bin) cell."""
    sample = bin_data[story_idx]
    haystack, question = extract_haystack_and_question(sample)
    sentences = split_sentences(haystack)

    paragraphs = []
    gt_docs = []
    for i, sent in enumerate(sentences):
        sid = f"s{i}"
        paragraphs.append({
            "idx": sid,
            "paragraph_text": sent,
            "title": None,
        })
        if normalize(sent) in fact_sentences_norm:
            gt_docs.append(sid)

    return {
        "idx": f"story{story_idx}_{bin_label}",
        "story_idx": story_idx,
        "bin": bin_label,
        "question": question,
        "paragraphs": paragraphs,
        "gt_docs": gt_docs,
        "n_paragraphs": len(paragraphs),
        "n_gold": len(gt_docs),
    }


# --------------------------------------------------------------------------
# Main

BINS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default=".",
                    help="repo root (use current dir by default)")
    ap.add_argument("--selection-file",
                    default="composable_cot/BABIlong/analysis/multi_entry_eval/qr_selection/selected_60_stories.json")
    ap.add_argument("--multi-entry-dir",
                    default="composable_cot/BABIlong/data/eval_multi_entry",
                    help="directory containing the 305-sample multi-entry per-bin JSONs")
    ap.add_argument("--output-dir",
                    default="composable_cot/retrieval_head_analysis/data")
    args = ap.parse_args()

    proj = Path(args.project_dir).resolve()
    sel_path = proj / args.selection_file
    me_dir = proj / args.multi_entry_dir
    out_dir = proj / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selection file: {sel_path}")
    print(f"Multi-entry dir: {me_dir}")
    print(f"Output dir: {out_dir}")
    print()

    sel = json.loads(sel_path.read_text())
    selected_indices = sel["all_indices"]
    print(f"Selected stories: {len(selected_indices)} indices")

    # Sanity: confirm multi-entry data exists for all bins.
    missing = [b for b in BINS if not (me_dir / f"{b}.json").exists()]
    if missing:
        print(f"ERROR: missing multi-entry files for bins: {missing}", file=sys.stderr)
        print(f"Expected at: {me_dir}/{{bin}}.json", file=sys.stderr)
        sys.exit(1)

    # Load the 0K bin once — that's our fact-string reference per story.
    print("Loading 0K reference (bAbI facts only, no noise)...")
    bin_data_0k = json.loads((me_dir / "0k.json").read_text())
    print(f"  0K samples: {len(bin_data_0k)}")

    fact_sentences_per_story: dict[int, set[str]] = {}
    for idx in selected_indices:
        if idx >= len(bin_data_0k):
            print(f"  WARN: story {idx} not in 0K bin (only {len(bin_data_0k)} samples)")
            continue
        haystack_0k, _ = extract_haystack_and_question(bin_data_0k[idx])
        facts = split_sentences(haystack_0k)
        fact_sentences_per_story[idx] = {normalize(s) for s in facts}

    fact_counts = [len(v) for v in fact_sentences_per_story.values()]
    print(f"  fact sentences per story: min={min(fact_counts)}, "
          f"max={max(fact_counts)}, mean={sum(fact_counts)/len(fact_counts):.1f}")
    print()

    # Build per-bin output.
    per_bin_summary = {}
    for b in BINS:
        bin_data = json.loads((me_dir / f"{b}.json").read_text())
        bin_n = len(bin_data)

        entries = []
        miss_count = 0
        gold_counts = []
        para_counts = []
        for idx in selected_indices:
            if idx >= bin_n:
                miss_count += 1
                continue
            facts_norm = fact_sentences_per_story.get(idx)
            if facts_norm is None:
                miss_count += 1
                continue
            entry = build_one_story(idx, bin_data, facts_norm, b)
            entries.append(entry)
            gold_counts.append(entry["n_gold"])
            para_counts.append(entry["n_paragraphs"])

        out_path = out_dir / f"detection_set_{b}.json"
        out_path.write_text(json.dumps(entries))

        if entries:
            mean_gold = sum(gold_counts) / len(gold_counts)
            mean_para = sum(para_counts) / len(para_counts)
            avg_gold_recall = sum(
                e["n_gold"] / len(fact_sentences_per_story[e["story_idx"]])
                for e in entries
            ) / len(entries)
            print(f"  [{b:5s}] {len(entries):2d} entries  miss={miss_count}  "
                  f"paras/sample mean={mean_para:5.1f}  "
                  f"gold/sample mean={mean_gold:4.1f}  "
                  f"gold-recall mean={avg_gold_recall:.3f}")
            per_bin_summary[b] = {
                "n_entries": len(entries),
                "missing": miss_count,
                "mean_paragraphs": mean_para,
                "mean_gold": mean_gold,
                "mean_gold_recall": avg_gold_recall,
            }
        else:
            print(f"  [{b:5s}] EMPTY ({miss_count} stories missing)")
            per_bin_summary[b] = {"n_entries": 0, "missing": miss_count}

    # Write a single summary file alongside the per-bin outputs.
    summary_path = out_dir / "detection_set_summary.json"
    summary_path.write_text(json.dumps(per_bin_summary, indent=2))
    print()
    print(f"Wrote {len(BINS)} per-bin files + summary to {out_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
