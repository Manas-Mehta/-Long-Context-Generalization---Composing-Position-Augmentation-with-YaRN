#!/usr/bin/env python
"""Download and prepare LongFaith-SFT dataset for training.

LongFaith is NOT on HuggingFace — only on Google Drive. Run this on a node
with internet (HPC login node, not compute node):

    pip install gdown
    python prepare_longfaith.py --output-dir experiments/longfaith/data

Source paper: "LongFaith: Enhancing Long-Context Reasoning in LLMs with
Faithful Synthetic Data," Yang et al., ACL Findings 2025 (arXiv:2502.12583).
GitHub: https://github.com/IDEA-FinAI/LongFaith (MIT).

Variant used: gpt-4o-mini synthesizer, 2K-sample SFT file
  (Table 4 ablation in the paper shows this synthesizer is strongest).
  "gpt-4o-mini" is the LLM that wrote the reasoning chains. Our target
  finetune model is Qwen2.5-7B-Instruct regardless.

Output:
    <output-dir>/faith_sft_2k.json   — raw alpaca file (2,048 examples)
    <output-dir>/faith_sft_2k_filtered.json — 2,038 examples (drop the ~10
                                              with malformed bracket counts)
    <output-dir>/length_stats.json   — quick distribution sanity check
"""

import argparse
import json
import os
import re
import sys

# Google Drive file ID for faith_sft_2k.json under longfaith_syn/gpt-4o-mini/
# Folder: https://drive.google.com/drive/folders/1f2306gR41glW9PzO6dJz8X5J53XsSNtC
# (File ID resolved by browsing the folder; gdown can pull by file or folder.)
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1f2306gR41glW9PzO6dJz8X5J53XsSNtC"
TARGET_FILENAME = "faith_sft_2k.json"
TARGET_SUBPATH = "longfaith_syn/gpt-4o-mini/" + TARGET_FILENAME


def _check_gdown():
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        print("ERROR: gdown not installed. Run: pip install gdown", file=sys.stderr)
        return False


def _looks_like_valid_json(path: str) -> bool:
    """Cheap pre-check: try to parse JSON, return True iff successful."""
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def download_via_folder(output_dir: str, force: bool = False) -> str:
    """Download the entire LongFaith folder; return path to target SFT file.

    gdown supports folder downloads natively. The folder is small (~few hundred
    MB across all variants); we only keep the gpt-4o-mini SFT file.
    """
    import gdown

    cache_dir = os.path.join(output_dir, "_longfaith_cache")
    os.makedirs(cache_dir, exist_ok=True)

    target_path = os.path.join(output_dir, TARGET_FILENAME)
    if os.path.exists(target_path) and not force:
        if _looks_like_valid_json(target_path):
            print(f"  [skip] Already present and valid: {target_path}")
            return target_path
        print(f"  [warn] {target_path} exists but is not valid JSON "
              f"(likely a partial download). Re-downloading.")
        os.remove(target_path)

    print(f"  Downloading LongFaith folder to {cache_dir} ...")
    print(f"  (this may take a few minutes — the full folder is ~few hundred MB)")
    gdown.download_folder(
        url=GDRIVE_FOLDER_URL,
        output=cache_dir,
        quiet=False,
        use_cookies=False,
    )

    # Find the target file inside the downloaded tree
    found = None
    for root, _, files in os.walk(cache_dir):
        for fn in files:
            if fn == TARGET_FILENAME and "gpt-4o-mini" in root:
                found = os.path.join(root, fn)
                break
        if found:
            break
    if found is None:
        raise FileNotFoundError(
            f"Could not locate {TARGET_SUBPATH} in downloaded folder.\n"
            f"Listing of {cache_dir}:\n"
            + "\n".join(os.listdir(cache_dir))
        )

    # Copy/move to output dir
    import shutil
    shutil.copy(found, target_path)
    print(f"  Saved -> {target_path}")
    return target_path


def filter_malformed(input_path: str, output_path: str) -> tuple[int, int]:
    """Drop the ~10 examples with malformed bracket counts (≠ 20 docs).

    LongFaith's instruction field embeds documents marked [1] [2] ... [20].
    The released file has ~10 outliers where the bracket count is off due to
    raw Wikipedia content inside the documents. Easy to filter — count
    occurrences of `\n[N]` at start of line where N is 1..20.
    """
    with open(input_path) as f:
        data = json.load(f)

    kept = []
    dropped = 0
    for ex in data:
        instr = ex.get("instruction", "")
        # Count document markers [1] [2] ... [20] — match at the start of
        # bracketed segments only.
        markers = set(re.findall(r"\[(\d+)\]\s", instr))
        marker_ints = {int(m) for m in markers if m.isdigit()}
        if not all(i in marker_ints for i in range(1, 21)):
            dropped += 1
            continue
        kept.append(ex)

    with open(output_path, "w") as f:
        json.dump(kept, f, indent=2)
    return len(kept), dropped


def compute_stats(filtered_path: str, output_path: str):
    """Quick length distribution sanity check (chars/3.5 token approx)."""
    with open(filtered_path) as f:
        data = json.load(f)

    instr_chars = [len(ex["instruction"]) for ex in data]
    out_chars = [len(ex["output"]) for ex in data]
    total_chars = [i + o for i, o in zip(instr_chars, out_chars)]

    def percentile(xs, p):
        xs = sorted(xs)
        k = int(len(xs) * p / 100)
        return xs[min(k, len(xs) - 1)]

    stats = {
        "n_examples": len(data),
        "instruction_chars": {
            "min": min(instr_chars), "p50": percentile(instr_chars, 50),
            "p95": percentile(instr_chars, 95), "max": max(instr_chars),
        },
        "output_chars": {
            "min": min(out_chars), "p50": percentile(out_chars, 50),
            "p95": percentile(out_chars, 95), "max": max(out_chars),
        },
        "total_chars": {
            "min": min(total_chars), "p50": percentile(total_chars, 50),
            "p95": percentile(total_chars, 95), "max": max(total_chars),
        },
        "approx_qwen_tokens_total": {
            "min": round(min(total_chars) / 3.5),
            "p50": round(percentile(total_chars, 50) / 3.5),
            "p95": round(percentile(total_chars, 95) / 3.5),
            "max": round(max(total_chars) / 3.5),
        },
        "note": "Tokens approximated as chars/3.5 for English. Re-run with the actual Qwen tokenizer for exact counts.",
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Length distribution (chars):")
    print(f"    instruction: {stats['instruction_chars']}")
    print(f"    output:      {stats['output_chars']}")
    print(f"    total:       {stats['total_chars']}")
    print(f"  Approx Qwen tokens (total): {stats['approx_qwen_tokens_total']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="Where to save faith_sft_2k.json and derived files.")
    ap.add_argument("--source", default="gdrive", choices=["gdrive", "local"],
                    help="Where to pull the SFT file from.")
    ap.add_argument("--local-path", default=None,
                    help="If --source=local, path to an existing faith_sft_2k.json.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if file exists.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == "gdrive":
        if not _check_gdown():
            sys.exit(1)
        raw_path = download_via_folder(args.output_dir, force=args.force)
    else:
        if not args.local_path or not os.path.exists(args.local_path):
            print(f"ERROR: --local-path required and must exist when --source=local",
                  file=sys.stderr)
            sys.exit(1)
        import shutil
        raw_path = os.path.join(args.output_dir, TARGET_FILENAME)
        shutil.copy(args.local_path, raw_path)
        print(f"  Copied {args.local_path} -> {raw_path}")

    filtered_path = os.path.join(args.output_dir, "faith_sft_2k_filtered.json")
    print(f"\n  Filtering malformed examples (bracket count != 20)...")
    kept, dropped = filter_malformed(raw_path, filtered_path)
    print(f"  Kept {kept}, dropped {dropped} -> {filtered_path}")

    stats_path = os.path.join(args.output_dir, "length_stats.json")
    compute_stats(filtered_path, stats_path)
    print(f"\n  Stats -> {stats_path}")
    print(f"\n  DONE. Use {filtered_path} as --train-file.")


if __name__ == "__main__":
    main()
