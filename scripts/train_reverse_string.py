#!/usr/bin/env python
"""Train a tiny Qwen2 from scratch on binary string reversal with RPE.

Reproduces the DeepMind RPE paper (arXiv:2305.16843) "reverse_string"
experiment, adapted from their JAX/Haiku encoder-only transformer to a
PyTorch decoder-only Qwen2 architecture with RoPE position patching.

Key design: we train a TINY model FROM SCRATCH (not fine-tune a pretrained
one) to isolate the RPE variable and match DeepMind's methodology.

DeepMind paper settings (from example.py / constants.py):
    Task:           reverse_string, vocab_size=2 (binary)
    Curriculum:     UniformCurriculum(values=range(1, 41))  → lengths [1, 40]
    Model:          5-layer encoder transformer, embedding_dim=64, 8 heads
    Pos encoding:   NOISY_ROTARY, noise_max_length=2048
    Batch size:     128
    Learning rate:  1e-3
    Optimizer:      Adam, grad clip 1.0, no weight decay
    Training steps: 10,000
    Eval:           Accuracy on lengths 1-100, 512 samples per length

Our adaptation:
    Model:          Tiny Qwen2 decoder (~2M params), 5 layers, hidden=64, 8 heads
    Tokenizer:      Character-level (vocab ~15 tokens, not BPE)
    RPE:            Patch position_ids with sorted random samples from [0, 2048)
    Training:       HF Trainer, causal LM loss on output tokens only
    Everything else matches paper exactly.

Usage:
    # Full training (matches DeepMind, ~2-5 min on Mac M3)
    python scripts/train_reverse_string.py

    # Pipeline test (quick infra validation)
    python scripts/train_reverse_string.py --pipeline-test

    # Baseline comparison (no RPE)
    python scripts/train_reverse_string.py --no-rpe --output-dir outputs/reverse_string_baseline
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Trainer,
    TrainingArguments,
)

from rpe.patching import RPEPatcher
from rpe.tasks.reverse_string_dataset import ReverseStringCollator, ReverseStringDataset


# ---------------------------------------------------------------------------
# Character-level tokenizer (replaces BPE tokenizer for scratch training)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Minimal character-level tokenizer for the reverse string task.

    Maps each character to a unique integer ID. No subword merges, no BPE.
    Vocabulary: <pad>, <eos>, space, newline, colon, digits 0-9, letters in
    "reverse" — about 20 tokens total.

    This replaces the 151,936-token Qwen BPE tokenizer, eliminating the
    massive embedding matrix and matching DeepMind's small-vocab setup.
    """

    def __init__(self) -> None:
        # Build vocabulary from all characters the task can produce.
        # "reverse: 01101\n10110" uses: r, e, v, s, ' ', ':', '0', '1', '\n'
        task_chars = sorted(set("reverse: 01\n"))
        # Reserve 0=pad, 1=eos, then map task characters
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}

        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._char_to_id["<pad>"] = 0
        self._char_to_id["<eos>"] = 1
        self._id_to_char[0] = "<pad>"
        self._id_to_char[1] = "<eos>"

        for i, c in enumerate(task_chars, start=2):
            self._char_to_id[c] = i
            self._id_to_char[i] = c

        self.vocab_size = len(self._char_to_id)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = None,
    ):
        ids = [self._char_to_id.get(c, self.pad_token_id) for c in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        for i in ids:
            if skip_special_tokens and i in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(self._id_to_char.get(i, "?"))
        return "".join(chars)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_length_generalization(
    model,
    tokenizer,
    patcher,
    min_length: int = 1,
    max_length: int = 100,
    num_samples_per_length: int = 64,
    seed: int = 1,
) -> list[dict]:
    """Autoregressive per-token accuracy across lengths.

    Generates output tokens one at a time using greedy decoding, then compares
    each generated character against the expected reversed string.

    Note: DeepMind uses encoder-only models (single forward pass, no error
    compounding). Our decoder-only model generates autoregressively, which is
    a stricter test — errors at early positions can cascade. This is an
    inherent difference between encoder-only and decoder-only evaluation.
    """
    import random as stdlib_random

    rng = stdlib_random.Random(seed)
    results = []

    model.eval()
    device = next(model.parameters()).device

    for length in range(min_length, max_length + 1):
        correct = 0
        total = 0

        for _ in range(num_samples_per_length):
            # Generate binary string (matching DeepMind's vocab_size=2)
            input_str = "".join(rng.choice("01") for _ in range(length))
            expected = input_str[::-1]
            prompt = f"reverse: {input_str}\n"

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=length + 5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0, prompt_len:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Per-token accuracy (matching DeepMind's accuracy_fn)
            gen_chars = list(generated[:length])
            exp_chars = list(expected)
            if len(gen_chars) == len(exp_chars):
                token_correct = sum(g == e for g, e in zip(gen_chars, exp_chars))
                correct += token_correct
                total += length
            else:
                # Length mismatch — count as all wrong
                total += length

        accuracy = correct / total if total > 0 else 0.0
        results.append({"length": length, "accuracy": accuracy})
        if length % 10 == 0 or length <= 5:
            print(f"  Length {length:3d}: accuracy={accuracy:.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train tiny Qwen2 from scratch on reverse string with RPE"
    )

    # Mode
    parser.add_argument("--pipeline-test", action="store_true",
                        help="Quick infrastructure validation (~30s)")
    parser.add_argument("--no-rpe", action="store_true",
                        help="Train baseline without RPE (standard positions)")

    # === All defaults match DeepMind paper exactly ===
    # Task (paper Section 4, Table 1)
    parser.add_argument("--min-train-length", type=int, default=1,
                        help="Min training string length (paper: 1)")
    parser.add_argument("--max-train-length", type=int, default=40,
                        help="Max training string length (paper: 40)")

    # RPE (paper Section 3)
    parser.add_argument("--rpe-max-sim-length", type=int, default=2048,
                        help="RPE L parameter (paper: noise_max_length=2048)")

    # Model architecture (paper Section 4, matches constants.py)
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="Embedding dimension (paper: 64)")
    parser.add_argument("--num-layers", type=int, default=5,
                        help="Number of transformer layers (paper: 5)")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads (paper: 8)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability (paper: 0.1)")

    # Training (paper Section 4)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Training batch size (paper: 128)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate (paper: 1e-3)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm (paper: 1.0)")
    parser.add_argument("--training-steps", type=int, default=10000,
                        help="Number of training steps (paper: 10,000)")

    # Eval (paper Section 4)
    parser.add_argument("--max-eval-length", type=int, default=100,
                        help="Max evaluation length (paper: 100)")
    parser.add_argument("--eval-samples-per-length", type=int, default=512,
                        help="Samples per length for eval (paper: 512)")

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/reverse_string_rpe",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (paper: 0)")

    args = parser.parse_args()

    # === Pipeline test overrides ===
    if args.pipeline_test:
        args.training_steps = 50
        args.batch_size = 16
        args.max_eval_length = 20
        args.eval_samples_per_length = 8
        args.output_dir = "outputs/pipeline_test"
        print("=" * 70)
        print("PIPELINE TEST MODE — quick infra validation")
        print("=" * 70)

    # --- Print config ---
    rpe_label = "OFF (baseline)" if args.no_rpe else f"ON (L={args.rpe_max_sim_length})"
    print(f"\n{'=' * 70}")
    print("Reverse String — From-Scratch Training (DeepMind Reproduction)")
    print(f"{'=' * 70}")
    print(f"  Task:            binary string reversal (vocab_size=2)")
    print(f"  Curriculum:      Uniform[{args.min_train_length}, {args.max_train_length}]")
    print(f"  RPE:             {rpe_label}")
    print(f"  Model:           Qwen2 decoder, {args.num_layers}L, {args.hidden_size}d, {args.num_heads}H")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  LR:              {args.learning_rate}")
    print(f"  Grad clip:       {args.max_grad_norm}")
    print(f"  Training steps:  {args.training_steps}")
    print(f"  Eval range:      [1, {args.max_eval_length}]")
    print(f"  Seed:            {args.seed}")

    # === Character tokenizer ===
    tokenizer = CharTokenizer()
    print(f"\n  Tokenizer:       {tokenizer} (character-level)")

    # === Build tiny Qwen2 model from scratch ===
    print(f"\nInitializing random Qwen2 model...")
    config = Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads,  # No GQA for tiny model
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.rpe_max_sim_length,
        attention_dropout=args.dropout,
        tie_word_embeddings=True,
        use_sliding_window=False,
    )
    model = Qwen2ForCausalLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {type(model).__name__}: {config.hidden_size}d, {config.num_hidden_layers}L, "
          f"{config.num_attention_heads}H, vocab={config.vocab_size}")
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # === Detect device ===
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model_dtype = torch.bfloat16
        model = model.to(dtype=model_dtype)
    # MPS and CPU stay in fp32 (default)

    # === Create datasets ===
    # DeepMind generates fresh batches each step (effectively infinite data).
    # We pre-generate a large dataset and loop via epochs.
    # Need: training_steps * batch_size examples for 1 epoch of unique data.
    # Generate a generous pool; HF Trainer will loop as needed via max_steps.
    num_train = min(args.training_steps * args.batch_size, 500_000)
    num_val = max(num_train // 20, 64)
    print(f"\nGenerating datasets...")
    train_dataset = ReverseStringDataset(
        tokenizer=tokenizer,
        num_examples=num_train,
        min_length=args.min_train_length,
        max_length=args.max_train_length,
        seed=args.seed,
    )
    val_dataset = ReverseStringDataset(
        tokenizer=tokenizer,
        num_examples=num_val,
        min_length=args.min_train_length,
        max_length=args.max_train_length,
        seed=args.seed + 1000,
    )
    collator = ReverseStringCollator(tokenizer, padding_side="right")

    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Val:   {len(val_dataset):,} examples")

    # Show a few examples
    for i in range(min(3, len(train_dataset))):
        ex = train_dataset.examples[i]
        print(f"  Sample {i}: \"{ex['input_str']}\" -> \"{ex['output_str']}\" (len={ex['length']})")

    # === Patch model with RPE (or skip for baseline) ===
    patcher = None
    if not args.no_rpe:
        print(f"\nApplying RPE patch (L={args.rpe_max_sim_length})...")
        rpe_config = {"max_simulation_length": args.rpe_max_sim_length}
        patcher = RPEPatcher(model, rpe_config)
        patcher.patch()
    else:
        print("\nRPE disabled — training with standard sequential positions (baseline).")

    # === Training arguments ===
    # We use max_steps to match DeepMind's step-based training.
    # num_train_epochs is set high so HF Trainer doesn't stop early.
    steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
    epochs_needed = max(1, (args.training_steps // steps_per_epoch) + 1)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.training_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        # Match DeepMind: Adam, no weight decay, no warmup
        weight_decay=0.0,
        optim="adamw_torch",
        warmup_steps=0,
        lr_scheduler_type="constant",
        # Logging
        logging_steps=max(1, args.training_steps // 50),
        logging_dir=os.path.join(args.output_dir, "logs"),
        eval_strategy="no",
        # Save final checkpoint (for re-evaluation without retraining)
        save_strategy="steps",
        save_steps=args.training_steps,  # save only at the very end
        save_total_limit=1,
        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
        # Performance
        fp16=False,
        bf16=use_cuda,
        gradient_checkpointing=False,  # Tiny model, not needed
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        # Disable unused features
        report_to="none",
        remove_unused_columns=False,
    )

    # === Train ===
    print(f"\nStarting training ({args.training_steps} steps, batch={args.batch_size})...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    train_result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Train loss:    {train_result.training_loss:.4f}")
    print(f"  Samples/sec:   {train_result.metrics.get('train_samples_per_second', 0):.1f}")

    # Save training loss history
    loss_log = [
        {"step": entry["step"], "loss": entry["loss"]}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_loss_log.json"), "w") as f:
        json.dump(loss_log, f, indent=2)
    print(f"  Loss curve saved ({len(loss_log)} points) to {args.output_dir}/train_loss_log.json")

    # === Unpatch before evaluation ===
    # IMPORTANT: Must unpatch before model.generate(). The rpe_forward wrapper
    # uses **kwargs which hides the `position_ids` parameter name from
    # inspect.signature(). HF's prepare_inputs_for_generation checks
    # "position_ids" in signature to decide whether to create position_ids
    # for KV-cache generation steps. With the wrapper active, this check fails
    # and every generated token gets position_id=0, corrupting output.
    if patcher is not None and patcher.is_patched:
        patcher.unpatch()

    # === Evaluate length generalization ===
    print(f"\n{'=' * 70}")
    print("Length Generalization Evaluation")
    print(f"{'=' * 70}")
    print(f"Testing lengths [{args.min_train_length}, {args.max_eval_length}] "
          f"with {args.eval_samples_per_length} samples per length")
    print(f"Lengths > {args.max_train_length} are OOD (out-of-distribution)\n")

    eval_results = evaluate_length_generalization(
        model=model,
        tokenizer=tokenizer,
        patcher=None,
        min_length=args.min_train_length,
        max_length=args.max_eval_length,
        num_samples_per_length=args.eval_samples_per_length,
        seed=1,
    )

    # === Report ===
    in_dist = [r for r in eval_results if r["length"] <= args.max_train_length]
    ood = [r for r in eval_results if r["length"] > args.max_train_length]

    in_dist_acc = sum(r["accuracy"] for r in in_dist) / len(in_dist) if in_dist else 0
    ood_acc = sum(r["accuracy"] for r in ood) / len(ood) if ood else 0

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  In-distribution  (len 1-{args.max_train_length}):  {in_dist_acc:.3f}")
    print(f"  Out-of-dist      (len {args.max_train_length + 1}-{args.max_eval_length}): {ood_acc:.3f}")
    print(f"  Overall          (len 1-{args.max_eval_length}):  "
          f"{sum(r['accuracy'] for r in eval_results) / len(eval_results):.3f}")

    # DeepMind's score metric: mean accuracy on lengths > training length
    # (matches example.py line 138: score = np.mean(accuracies[sequence_length + 1:]))
    dm_score = ood_acc
    print(f"\n  DeepMind 'score' (mean OOD accuracy): {dm_score:.3f}")
    print(f"  (Paper reports ~0.8+ for RPE on reverse_string)")

    # Collect experiment metadata
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "git_hash": git_hash,
            "config": {
                "model": "tiny-qwen2-from-scratch",
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "num_params": num_params,
                "rpe_enabled": not args.no_rpe,
                "rpe_max_sim_length": args.rpe_max_sim_length,
                "max_train_length": args.max_train_length,
                "max_eval_length": args.max_eval_length,
                "eval_samples_per_length": args.eval_samples_per_length,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "training_steps": args.training_steps,
                "seed": args.seed,
            },
            "train_loss": train_result.training_loss,
            "in_dist_accuracy": in_dist_acc,
            "ood_accuracy": ood_acc,
            "dm_score": dm_score,
            "per_length": eval_results,
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Unpatch if needed
    if patcher is not None:
        patcher.unpatch()

    print(f"\n{'=' * 70}")
    if args.pipeline_test:
        print("Pipeline test complete! Infrastructure is working.")
        print("Low accuracy expected — this was a quick validation run.")
    else:
        print("Training complete. See results above.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
