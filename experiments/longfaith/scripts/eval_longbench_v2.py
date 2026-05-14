#!/usr/bin/env python
"""Evaluate a LongFaith-trained LoRA checkpoint on LongBench v2 QA subsets.

Forked from experiments/babilong/scripts/eval_babilong.py. Differences:
  - Loads pre-filtered LongBench v2 QA examples (300 total: Single-Doc 175 +
    Multi-Doc 125) and pre-computed Qwen-token bin indices (16K/32K/64K/128K).
  - MCQ prompt: context + question + four choices + letter-answer instruction.
  - CoT decoding: model generates a chain ending in "The answer is X."
    where X is A/B/C/D. Parser extracts the letter via regex.
  - Supports zero-shot mode via --no-lora (no checkpoint dir).

Eval YaRN setting per condition (all trained conditions with YaRN train f=2
get f=4 at eval; LoRA-only and zero-shot get either no-YaRN or YaRN f=4):
  lora_base / zero_shot : --enable-yarn optional (provides two zero-shot rows)
  y2_*                  : --enable-yarn --yarn-factor 4.0

Output layout:
  <output-dir>/
    predictions_16k.json     # list of {id, question, target, prediction, raw_output, correct, n_tokens}
    predictions_32k.json
    predictions_64k.json
    predictions_128k.json
    summary.json             # {bin: {n, n_correct, accuracy, n_parser_miss}, overall}

Usage:
  # Trained checkpoint
  python eval_longbench_v2.py \\
      --checkpoint-dir experiments/longfaith/checkpoints/y2_rpe_cur_L16k \\
      --enable-yarn --yarn-factor 4.0 \\
      --data-dir experiments/longfaith/data \\
      --output-dir experiments/longfaith/results/y2_rpe_cur_L16k \\
      --condition y2_rpe_cur_L16k

  # Zero-shot, no YaRN
  python eval_longbench_v2.py \\
      --no-lora \\
      --data-dir experiments/longfaith/data \\
      --output-dir experiments/longfaith/results/zero_shot_nyarn \\
      --condition zero_shot_nyarn

  # Zero-shot, YaRN f=4 at eval
  python eval_longbench_v2.py \\
      --no-lora --enable-yarn --yarn-factor 4.0 \\
      --data-dir experiments/longfaith/data \\
      --output-dir experiments/longfaith/results/zero_shot_yarn4 \\
      --condition zero_shot_yarn4
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


EVAL_BINS = ["16k", "32k", "64k", "128k"]

# Prompt template for LongBench v2 MCQ.
#
# Mirrors the LongFaith authors' PREDICT_COC_PROMPT (data_manager.py in
# https://github.com/IDEA-FinAI/LongFaith) verbatim except for the final-
# answer constraint: original is "concise without other words" for free-form
# QA; ours requires exactly one letter A/B/C/D for v2 MCQ.
#
# Context formatting mirrors LongFaith's single-doc eval handling for
# qasper / multifieldqa_en: the input is sliced into 20 equal character-based
# chunks labeled [1]..[20]. See build_prompt() below.
MCQ_PROMPT = (
    "You are provided with documents and a complex logical reasoning question.\n"
    "You must refer to the documents to perform step-by-step logical reasoning "
    "and reach the correct answer.\n"
    "Each reasoning step must be on a separate line, ending with a newline character.\n"
    "Cite the document properly during reasoning, e.g., `[1]`, `[2]`, etc.\n"
    "The final answer must begin with `The answer is` followed by exactly one "
    "letter — A, B, C, or D — and nothing else.\n\n"
    "DOCUMENTS:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "CHOICES:\n"
    "A) {choice_A}\n"
    "B) {choice_B}\n"
    "C) {choice_C}\n"
    "D) {choice_D}\n"
)


# THUDM/LongBench prompts/0shot_cot.txt — verbatim copy (curly apostrophe
# preserved). Used as pass-1 of the official two-pass CoT protocol when
# --two-pass is set. See notes/longbench_v2_protocol_match.md.
OFFICIAL_COT_PROMPT = (
    "Please read the following text and answer the questions below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "Let’s think step by step:"
)

# THUDM/LongBench prompts/0shot_cot_ans.txt — verbatim. Pass-2 of the
# two-pass protocol: context is replaced with a stub, the pass-1 CoT is
# injected, and the model is asked to emit "The correct answer is (X)".
OFFICIAL_COT_ANS_PROMPT = (
    "Please read the following text and answer the questions below.\n\n"
    "The text is too long and omitted here.\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "Let’s think step by step: {cot}\n\n"
    "Based on the above, what is the single, most likely answer choice? "
    "Format your response as follows: \"The correct answer is (insert answer here)\"."
)


def _chunk_into_20(context: str) -> str:
    """Slice context into 20 equal character-based chunks labeled [1]..[20].

    Mirrors LongFaith's data_manager.build_pred_coc_prompt() single-doc path
    (qasper / multifieldqa_en branch) verbatim. Character-based slicing can
    split mid-word; the LongFaith authors accept this trade-off in exchange
    for matching the training distribution shape (20 labeled documents).
    """
    n = len(context)
    return "\n".join(
        f"[{i + 1}] {context[i * n // 20 : (i + 1) * n // 20]}"
        for i in range(20)
    )


def build_prompt(sample: dict, chunk: bool = True, official: bool = False) -> str:
    """Build the MCQ pass-1 prompt for one v2 example.

    chunk=True  (default): slice the context into 20 char-equal chunks
                           labeled [1]..[20] — matches LongFaith training shape.
    chunk=False: pass the raw context verbatim — diagnostic for mismatch #2
                 in notes/longfaith_diagnosis_2026-05-15.md.
    official=True: use the THUDM/LongBench 0shot_cot.txt template verbatim
                   (pass-1 of the official two-pass protocol). Chunking still
                   applies to the $DOC$ slot if chunk=True.
    """
    context = _chunk_into_20(sample["context"]) if chunk else sample["context"]
    template = OFFICIAL_COT_PROMPT if official else MCQ_PROMPT
    return template.format(
        context=context,
        question=sample["question"],
        choice_A=sample["choice_A"],
        choice_B=sample["choice_B"],
        choice_C=sample["choice_C"],
        choice_D=sample["choice_D"],
    )


def build_official_answer_extraction_prompt(sample: dict, cot: str) -> str:
    """Pass-2 prompt of the THUDM two-pass protocol.

    Drops the long context (replaced with a stub), injects the pass-1 CoT,
    and instructs the model to emit "The correct answer is (X)". The model
    therefore only needs short-context attention for pass 2.
    """
    return OFFICIAL_COT_ANS_PROMPT.format(
        question=sample["question"],
        choice_A=sample["choice_A"],
        choice_B=sample["choice_B"],
        choice_C=sample["choice_C"],
        choice_D=sample["choice_D"],
        cot=cot,
    )


# Three-stage parser lives in _parser.py so rescore.py can import it without
# pulling in torch. See _parser.py and notes/longfaith_diagnosis_2026-05-15.md.
from _parser import parse_answer  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading — identical to BABILong eval (YaRN logic unchanged)
# ---------------------------------------------------------------------------

def load_model(
    base_model:  str,
    checkpoint:  str | None,
    enable_yarn: bool,
    yarn_factor: float,
    no_cuda:     bool,
):
    print(f"\n{'='*70}")
    print("MODEL LOADING")
    print(f"{'='*70}")
    print(f"  Base model:  {base_model}")
    print(f"  Checkpoint:  {checkpoint or '(none — zero-shot)'}")
    print(f"  YaRN:        {enable_yarn} (factor={yarn_factor})")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s")

    config = AutoConfig.from_pretrained(base_model)
    model_kwargs = {"torch_dtype": torch.bfloat16}

    if enable_yarn:
        print(f"\n  Enabling YaRN: factor={yarn_factor}")
        rope_theta = None
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            rope_theta = config.rope_parameters.get("rope_theta")
        if rope_theta is None:
            rope_theta = getattr(config, "rope_theta", 1000000.0)

        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            config.rope_parameters.update({
                "type": "yarn", "rope_type": "yarn", "factor": yarn_factor,
            })
            if config.rope_parameters.get("rope_theta") is None:
                config.rope_parameters["rope_theta"] = rope_theta
        else:
            config.rope_scaling = {"type": "yarn", "factor": yarn_factor}

        model_kwargs["config"] = config

    device_map = None if no_cuda else "auto"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        **model_kwargs,
    )
    print(f"  Base model loaded in {time.time()-t0:.1f}s")

    if checkpoint is not None:
        t0 = time.time()
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        print(f"  LoRA merged in {time.time()-t0:.1f}s")

    model.eval()
    print(f"  Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Eval one bin
# ---------------------------------------------------------------------------

def _letter_logit_bias_decode(model, tokenizer, input_ids, device):
    """Decode a single token with logits masked to {A, B, C, D}.

    Used by the --letter-forced diagnostic to skip CoT entirely. Returns the
    chosen letter as a raw string (one of "A"/"B"/"C"/"D") plus the top-4
    logit gap for inspection.
    """
    letter_ids = []
    for L in "ABCD":
        ids = tokenizer.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(" " + L, add_special_tokens=False)
        letter_ids.append(ids[0])
    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1]  # last-position logits
    sub = logits[letter_ids]
    pick = int(sub.argmax().item())
    return "ABCD"[pick], sub.tolist()


def eval_bin(
    model,
    tokenizer,
    qa_data: list[dict],
    indices: list[int],
    bin_label: str,
    max_samples: int,
    max_seq_len: int,
    max_new_tokens: int,
    no_cuda: bool,
    chunk: bool = True,
    letter_forced: bool = False,
    two_pass: bool = False,
) -> tuple[list[dict], dict]:
    if max_samples > 0:
        indices = indices[:max_samples]

    device  = "cpu" if no_cuda else next(model.parameters()).device
    results = []
    n_skip  = 0
    n_parser_miss = 0

    t_bin = time.time()
    for k, qa_idx in enumerate(indices):
        if (k + 1) % 10 == 0 or k == 0:
            elapsed = time.time() - t_bin
            rate = (k + 1) / elapsed if elapsed > 0 else 0
            eta = (len(indices) - k - 1) / rate if rate > 0 else 0
            print(
                f"    [{bin_label}] {k + 1}/{len(indices)} | "
                f"{rate*60:.1f} samples/min | ETA {eta/60:.1f}min",
                flush=True,
            )

        sample = qa_data[qa_idx]
        target = sample["answer"].strip().upper()
        user_content = build_prompt(sample, chunk=chunk, official=two_pass)

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # In letter-forced mode, append "The answer is " so the very next token
        # is the model's letter pick. Skips CoT entirely.
        if letter_forced:
            prompt_text = prompt_text + "The answer is "

        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
        n_tokens  = input_ids.shape[1]

        if n_tokens > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]
            n_tokens  = input_ids.shape[1]

        if not no_cuda:
            input_ids = input_ids.to(device)

        letter_logits = None
        cot_response   = None
        n_tokens_pass2 = None
        try:
            if letter_forced:
                letter, letter_logits = _letter_logit_bias_decode(
                    model, tokenizer, input_ids, device
                )
                raw_output = letter
            elif two_pass:
                # Pass 1: CoT generation with the full long-context prompt.
                # 1024-token cap mirrors THUDM/LongBench pred.py CoT setting.
                with torch.no_grad():
                    out_p1 = model.generate(
                        input_ids,
                        max_new_tokens=max(max_new_tokens, 1024),
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                cot_response = tokenizer.decode(
                    out_p1[0][input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                # Pass 2: drop the long context, inject the CoT, force the
                # final letter. Pass-2 prompts are short (~few hundred tokens)
                # so no max_seq_len clipping is needed.
                user_content_p2 = build_official_answer_extraction_prompt(
                    sample, cot_response
                )
                prompt_text_p2 = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content_p2}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids_p2  = tokenizer.encode(prompt_text_p2, return_tensors="pt")
                n_tokens_pass2 = input_ids_p2.shape[1]
                if not no_cuda:
                    input_ids_p2 = input_ids_p2.to(device)
                with torch.no_grad():
                    out_p2 = model.generate(
                        input_ids_p2,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw_output = tokenizer.decode(
                    out_p2[0][input_ids_p2.shape[1]:], skip_special_tokens=True
                ).strip()
            else:
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw_output = tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=True
                ).strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raw_output = "__OOM__"
            n_skip += 1
        except Exception as e:
            raw_output = f"__ERROR__: {e}"
            n_skip += 1

        choices = {L: sample[f"choice_{L}"] for L in "ABCD"}
        if letter_forced and raw_output in "ABCD":
            parsed, parser_stage = raw_output, "letter_forced"
        else:
            parsed, parser_stage = parse_answer(raw_output, choices)
            if parser_stage == "miss":
                n_parser_miss += 1
        correct = parsed == target

        rec = {
            "id":           sample.get("_id"),
            "sub_domain":   sample.get("sub_domain"),
            "question":     sample["question"],
            "target":       target,
            "parsed":       parsed,
            "correct":      correct,
            "parser":       parser_stage,
            "raw_output":   raw_output,
            "n_tokens":     n_tokens,
        }
        if letter_logits is not None:
            rec["letter_logits"] = {L: round(l, 4) for L, l in zip("ABCD", letter_logits)}
        if cot_response is not None:
            rec["cot_response"]   = cot_response
            rec["n_tokens_pass2"] = n_tokens_pass2
        results.append(rec)

    n_correct  = sum(r["correct"] for r in results)
    n_recovery = sum(1 for r in results if r["parser"] == "recovery")
    n_fallback = sum(1 for r in results if r["parser"] == "fallback")
    accuracy   = n_correct / len(results) if results else 0.0

    stats = {
        "n":             len(results),
        "n_correct":     n_correct,
        "accuracy":      round(accuracy, 4),
        "n_parser_miss": n_parser_miss,
        "n_recovery":    n_recovery,
        "n_fallback":    n_fallback,
        "n_skipped":     n_skip,
        "elapsed_s":     round(time.time() - t_bin, 1),
    }
    print(
        f"    [{bin_label}] DONE: {n_correct}/{len(results)} correct "
        f"({accuracy:.1%}) | parser miss={n_parser_miss} fallback={n_fallback} "
        f"recovery={n_recovery} | {stats['elapsed_s']:.0f}s",
        flush=True,
    )
    return results, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Eval LongFaith LoRA on LongBench v2")

    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--no-lora",        action="store_true",
                   help="Skip LoRA loading (zero-shot baseline eval).")
    p.add_argument("--base-model",     default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--enable-yarn",    action="store_true")
    p.add_argument("--yarn-factor",    type=float, default=4.0)
    p.add_argument("--no-cuda",        action="store_true")

    p.add_argument("--data-dir", required=True,
                   help="Dir containing longbench_v2_qa.json and eval_v2_bin_indices.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--condition",  default="unknown")

    p.add_argument("--bins", nargs="+", default=EVAL_BINS,
                   help="Which bins to eval (default: 16k 32k 64k 128k)")
    p.add_argument("--max-samples",    type=int, default=0,
                   help="Max samples per bin (0 = all)")
    p.add_argument("--max-seq-len",    type=int, default=131072)
    p.add_argument("--max-new-tokens", type=int, default=512)

    # Diagnostic prompt / decoding variants — see
    # notes/longfaith_diagnosis_2026-05-15.md Tier 2.
    p.add_argument("--no-chunk", action="store_true",
                   help="Skip _chunk_into_20: feed the raw context verbatim. "
                        "Tests whether the [1]..[20] char-equal chunking fights "
                        "v2's continuous documents (mismatch #2).")
    p.add_argument("--letter-forced", action="store_true",
                   help="Append 'The answer is ' to prompt and decode a single "
                        "letter (A/B/C/D) via logit masking. Skips CoT entirely. "
                        "Tests whether verbose-CoT overthinking is the bottleneck "
                        "(mismatch #1).")
    p.add_argument("--two-pass", action="store_true",
                   help="Use the THUDM/LongBench v2 two-pass CoT protocol. "
                        "Pass 1 reasons with the official 0shot_cot.txt prompt "
                        "(1024 tokens). Pass 2 drops the context, injects the "
                        "pass-1 CoT, and forces 'The correct answer is (X)' "
                        "(128 tokens). Only pass-2 output is parsed. Borrowed "
                        "(not full match) from THUDM's protocol — we keep YaRN "
                        "and our long-context filtering. See "
                        "notes/longbench_v2_protocol_match.md.")

    args = p.parse_args()
    if args.two_pass and args.letter_forced:
        print("ERROR: --two-pass and --letter-forced are mutually exclusive.",
              file=sys.stderr)
        sys.exit(2)
    return args


def main():
    args = parse_args()

    qa_path     = os.path.join(args.data_dir, "longbench_v2_qa.json")
    bucket_path = os.path.join(args.data_dir, "eval_v2_bin_indices.json")
    for p in (qa_path, bucket_path):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}. Run prepare_longbench_v2.py first.",
                  file=sys.stderr)
            sys.exit(1)
    with open(qa_path) as f:
        qa_data = json.load(f)
    with open(bucket_path) as f:
        buckets = json.load(f)

    print("=" * 70)
    print("LongBench v2 — LongFaith evaluation")
    print("=" * 70)
    print(f"  Condition:    {args.condition}")
    print(f"  Checkpoint:   {args.checkpoint_dir or '(zero-shot)'}")
    print(f"  YaRN:         {args.enable_yarn} (factor={args.yarn_factor})")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Bins:         {args.bins}")
    print(f"  Max samples:  {args.max_samples or 'all'}")
    print(f"  Max seq len:  {args.max_seq_len}")
    print(f"  Max new toks: {args.max_new_tokens}")
    print(f"  Chunk into 20:{not args.no_chunk}")
    print(f"  Letter-forced:{args.letter_forced}")
    print(f"  Two-pass CoT: {args.two_pass}")
    print(f"  Timestamp:    {datetime.now().isoformat()}")
    print(f"  QA examples:  {len(qa_data)}")
    for b in args.bins:
        print(f"    bin {b:>5}: {len(buckets.get(b, []))} examples")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = None if args.no_lora else args.checkpoint_dir

    model, tokenizer = load_model(
        base_model  = args.base_model,
        checkpoint  = checkpoint,
        enable_yarn = args.enable_yarn,
        yarn_factor = args.yarn_factor,
        no_cuda     = args.no_cuda,
    )

    all_stats = {}
    t_total   = time.time()

    for bin_label in args.bins:
        indices = buckets.get(bin_label, [])
        if not indices:
            print(f"\n  [{bin_label}] SKIP — 0 examples in bucket", flush=True)
            continue

        print(f"\n{'='*70}")
        print(f"BIN: {bin_label}  ({len(indices)} examples)")
        print(f"{'='*70}")

        predictions, stats = eval_bin(
            model         = model,
            tokenizer     = tokenizer,
            qa_data       = qa_data,
            indices       = indices,
            bin_label     = bin_label,
            max_samples   = args.max_samples,
            max_seq_len   = args.max_seq_len,
            max_new_tokens= args.max_new_tokens,
            no_cuda       = args.no_cuda,
            chunk         = not args.no_chunk,
            letter_forced = args.letter_forced,
            two_pass      = args.two_pass,
        )

        pred_path = os.path.join(args.output_dir, f"predictions_{bin_label}.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)

        all_stats[bin_label] = stats
        _save_summary(args, all_stats, t_total)

    _save_summary(args, all_stats, t_total)
    _print_summary(args.condition, all_stats)


def _save_summary(args, all_stats: dict, t_total: float):
    total_n       = sum(s["n"]         for s in all_stats.values())
    total_correct = sum(s["n_correct"] for s in all_stats.values())
    summary = {
        "condition":     args.condition,
        "checkpoint":    args.checkpoint_dir,
        "enable_yarn":   args.enable_yarn,
        "yarn_factor":   args.yarn_factor,
        "no_chunk":      args.no_chunk,
        "letter_forced": args.letter_forced,
        "two_pass":      args.two_pass,
        "timestamp":     datetime.now().isoformat(),
        "elapsed_s":     round(time.time() - t_total, 1),
        "bins":          all_stats,
        "overall": {
            "n":         total_n,
            "n_correct": total_correct,
            "accuracy":  round(total_correct / total_n, 4) if total_n > 0 else 0.0,
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _print_summary(condition: str, all_stats: dict):
    print(f"\n{'='*70}")
    print(f"SUMMARY — {condition}")
    print(f"{'='*70}")
    print(f"  {'Bin':<6} {'N':>4} {'Correct':>8} {'Accuracy':>10} {'ParserMiss':>11}")
    print(f"  {'-'*6} {'-'*4} {'-'*8} {'-'*10} {'-'*11}")
    for bin_label, s in all_stats.items():
        print(f"  {bin_label:<6} {s['n']:>4} {s['n_correct']:>8} "
              f"{s['accuracy']:>10.1%} {s['n_parser_miss']:>11}")
    total_n       = sum(s["n"]         for s in all_stats.values())
    total_correct = sum(s["n_correct"] for s in all_stats.values())
    if total_n > 0:
        print(f"  {'OVERALL':<6} {total_n:>4} {total_correct:>8} "
              f"{total_correct/total_n:>10.1%}")
    print(f"{'='*70}")
    print("EVAL COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
