#!/usr/bin/env python
"""Train LoRA adapters on MRCR (Multi-Round Coreference Resolution).

Supports four conditions:
  1. LoRA baseline:       Normal RoPE, no position tricks
  2. YaRN+LoRA:           YaRN applied at model load (factor=4.0)
  3. RPE+LoRA (fixed):    RPE with fixed L during training
  4. RPE+LoRA (curriculum): RPE with increasing L per epoch

Uses HuggingFace Trainer + PEFT directly (not LLaMA-Factory) because:
  - MRCR has multi-turn conversations (10+ messages, 4K-8K tokens)
  - Need YaRN config injection at model load
  - Need fine-grained control over logging/checkpointing

Usage:
    # LoRA baseline
    python train_mrcr_lora.py \
        --train-file data/bin0_4K-8K/train.json \
        --output-dir checkpoints/lora_baseline

    # YaRN+LoRA
    python train_mrcr_lora.py \
        --enable-yarn --yarn-factor 4.0 \
        --train-file data/bin0_4K-8K/train.json \
        --output-dir checkpoints/yarn_lora

    # RPE+LoRA (fixed L)
    python train_mrcr_lora.py \
        --rpe-config configs/rpe_config_mrcr.yaml \
        --train-file data/bin0_4K-8K/train.json \
        --output-dir checkpoints/rpe_lora

    # RPE+LoRA (curriculum)
    python train_mrcr_lora.py \
        --rpe-config configs/rpe_config_mrcr_curriculum.yaml \
        --train-file data/bin0_4K-8K/train.json \
        --output-dir checkpoints/rpe_curriculum_lora
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Ensure project root is importable (for rpe/ and composable_cot/)
# Script is at: RPE/composable_cot/mrcr_context_extension/scripts/train_mrcr_lora.py
# Project root:  RPE/  (3 levels up from scripts/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MRCRDataset(Dataset):
    """MRCR training dataset with proper chat template and loss masking.

    Each sample is a multi-turn conversation where the model must recall
    and prepend a random string to a specific piece of content.
    We train only on the final assistant response (the answer).
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 8192):
        with open(data_path) as f:
            self.raw_data = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Pre-tokenize all samples and report stats
        self.samples = []
        n_truncated = 0
        total_answer_tokens = 0

        print(f"\n  Tokenizing {len(self.raw_data)} training samples...", flush=True)
        for i, item in enumerate(self.raw_data):
            sample = self._prepare_sample(item)
            if sample is None:
                continue
            if sample["truncated"]:
                n_truncated += 1
            total_answer_tokens += sample["n_answer_tokens"]
            self.samples.append(sample)

        avg_answer = total_answer_tokens / len(self.samples) if self.samples else 0
        print(f"  Tokenized: {len(self.samples)} samples", flush=True)
        print(f"  Truncated: {n_truncated} (>{self.max_seq_len} tokens)", flush=True)
        print(f"  Avg answer tokens: {avg_answer:.0f}", flush=True)

        # Print first sample for verification
        if self.samples:
            s = self.samples[0]
            n_masked = sum(1 for l in s["labels"] if l == -100)
            n_trained = len(s["labels"]) - n_masked
            print(f"\n  === Sample 0 verification ===", flush=True)
            print(f"  Total tokens: {len(s['input_ids'])}", flush=True)
            print(f"  Prompt tokens (masked): {n_masked}", flush=True)
            print(f"  Answer tokens (trained): {n_trained}", flush=True)
            # Decode first few answer tokens to verify alignment
            answer_start = n_masked
            answer_preview = self.tokenizer.decode(
                s["input_ids"][answer_start:answer_start + 30],
                skip_special_tokens=False,
            )
            print(f"  Answer starts with: {answer_preview[:100]!r}", flush=True)
            print(f"  Expected answer starts with: {self.raw_data[0]['answer'][:80]!r}", flush=True)

    def _prepare_sample(self, item: dict) -> dict | None:
        """Convert a raw MRCR sample to tokenized training format."""
        messages = json.loads(item["prompt"])
        answer = item["answer"]

        # Get prompt tokens (everything before the answer)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # Get full sequence (prompt + answer + end token)
        messages_with_answer = messages + [{"role": "assistant", "content": answer}]
        full_text = self.tokenizer.apply_chat_template(
            messages_with_answer, tokenize=False, add_generation_prompt=False,
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Number of answer tokens
        n_answer_tokens = len(full_ids) - len(prompt_ids)
        if n_answer_tokens <= 0:
            return None

        # Truncate if needed (keep answer, truncate prompt from left)
        truncated = False
        if len(full_ids) > self.max_seq_len:
            truncated = True
            excess = len(full_ids) - self.max_seq_len
            # Remove tokens from the start of the prompt
            full_ids = full_ids[excess:]
            prompt_ids = prompt_ids[excess:]

        # Create labels: -100 for prompt tokens, actual ids for answer
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        assert len(labels) == len(full_ids)

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "n_answer_tokens": n_answer_tokens,
            "truncated": truncated,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": s["input_ids"],
            "attention_mask": s["attention_mask"],
            "labels": s["labels"],
        }


# ---------------------------------------------------------------------------
# Model loading (reuses YaRN logic from eval_mrcr.py)
# ---------------------------------------------------------------------------

def load_model_for_training(
    base_model_name: str,
    enable_yarn: bool = False,
    yarn_factor: float = 4.0,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_target: str = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    use_gradient_checkpointing: bool = True,
    torch_dtype=torch.bfloat16,
):
    """Load base model with optional YaRN, attach LoRA, enable grad checkpointing.

    Returns (model, tokenizer, config_desc).
    """
    # --- Tokenizer ---
    print(f"\n{'='*70}", flush=True)
    print("MODEL SETUP", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s", flush=True)

    # --- Config + YaRN ---
    config = AutoConfig.from_pretrained(base_model_name)
    config_desc = "lora_baseline"
    model_kwargs = {"torch_dtype": torch_dtype}

    if enable_yarn:
        print(f"\n  Enabling YaRN with factor={yarn_factor}", flush=True)
        # Read rope_theta before potential loss (transformers 5.0+ bug)
        rope_theta = None
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            rope_theta = config.rope_parameters.get("rope_theta")
        if rope_theta is None:
            rope_theta = getattr(config, "rope_theta", 1000000.0)

        # Apply YaRN config (version-agnostic)
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            config.rope_parameters.update({
                "type": "yarn",
                "rope_type": "yarn",
                "factor": yarn_factor,
            })
            if config.rope_parameters.get("rope_theta") is None:
                config.rope_parameters["rope_theta"] = rope_theta
        else:
            config.rope_scaling = {"type": "yarn", "factor": yarn_factor}

        print(f"  rope_theta = {rope_theta}", flush=True)
        print(f"  rope_parameters = {getattr(config, 'rope_parameters', getattr(config, 'rope_scaling', 'N/A'))}", flush=True)
        model_kwargs["config"] = config
        config_desc = f"yarn_lora"

    # --- Load base model ---
    print(f"\n  Loading base model: {base_model_name}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        **model_kwargs,
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    # --- Verify YaRN ---
    if enable_yarn:
        _verify_yarn(model, config)

    # --- Gradient checkpointing ---
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        print(f"  Gradient checkpointing: enabled", flush=True)

    # --- LoRA ---
    target_modules = [m.strip() for m in lora_target.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    print(f"\n  LoRA config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}", flush=True)
    print(f"  Target modules: {target_modules}", flush=True)
    model.print_trainable_parameters()

    return model, tokenizer, config_desc


def _verify_yarn(model, config):
    """Check that YaRN actually modified inv_freq."""
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        rotary = model.model.layers[0].self_attn.rotary_emb

    rope_type = getattr(rotary, "rope_type", "unknown")
    print(f"  Actual rope_type: {rope_type}", flush=True)

    if hasattr(rotary, "inv_freq") and rotary.inv_freq is not None:
        dim = rotary.inv_freq.shape[0] * 2
        base = None
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            base = config.rope_parameters.get("rope_theta")
        if base is None:
            base = getattr(config, "rope_theta", 1000000.0)

        vanilla_inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        diff = (rotary.inv_freq.float().cpu() - vanilla_inv).abs()
        n_changed = (diff > 1e-8).sum().item()
        print(f"  YaRN verification: {n_changed}/{len(diff)} inv_freq dims differ from vanilla", flush=True)

        if n_changed == 0:
            print("  WARNING: YaRN NOT APPLIED! inv_freq identical to vanilla.", flush=True)
            print("  Attempting manual YaRN patch...", flush=True)
            # Import fallback from eval script
            from composable_cot.mrcr_context_extension.scripts.eval_mrcr import _apply_yarn_manual
            _apply_yarn_manual(model, 4.0, config)
    else:
        print("  inv_freq: not found as buffer (computed dynamically)", flush=True)


# ---------------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------------

class TrainingProgressCallback(TrainerCallback):
    """Rich per-step logging with timing, memory, and ETA."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.start_time = None
        self.step_logs = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n{'='*70}", flush=True)
        print("TRAINING STARTED", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Total steps: {state.max_steps}", flush=True)
        print(f"  Epochs: {args.num_train_epochs}", flush=True)
        print(f"  Batch size: {args.per_device_train_batch_size}", flush=True)
        print(f"  Grad accumulation: {args.gradient_accumulation_steps}", flush=True)
        print(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}", flush=True)
        print(f"  Learning rate: {args.learning_rate}", flush=True)
        print(f"  Warmup ratio: {args.warmup_ratio}", flush=True)
        print(f"  Scheduler: {args.lr_scheduler_type}", flush=True)
        print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'='*70}\n", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.start_time is None:
            return

        step = state.global_step
        max_steps = state.max_steps
        epoch = state.epoch or 0
        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)
        elapsed = time.time() - self.start_time

        # GPU memory
        gpu_mem = "N/A"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB"

        # ETA
        eta_str = "N/A"
        if step > 0 and max_steps > 0:
            time_per_step = elapsed / step
            remaining = (max_steps - step) * time_per_step
            eta_h = int(remaining // 3600)
            eta_m = int((remaining % 3600) // 60)
            eta_s = int(remaining % 60)
            eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"

        # Log entry
        entry = {
            "step": step,
            "epoch": round(epoch, 2),
            "loss": round(loss, 4) if loss is not None else None,
            "learning_rate": lr,
            "elapsed_s": round(elapsed, 1),
            "gpu_mem_gb": gpu_mem,
        }
        self.step_logs.append(entry)

        # Print progress line
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        lr_str = f"{lr:.2e}" if lr is not None else "N/A"
        elapsed_m = int(elapsed // 60)
        elapsed_s = int(elapsed % 60)
        print(
            f"  Step {step:>3}/{max_steps} | "
            f"Epoch {epoch:>5.2f} | "
            f"Loss {loss_str:>8} | "
            f"LR {lr_str:>10} | "
            f"GPU {gpu_mem:>8} | "
            f"Elapsed {elapsed_m:02d}:{elapsed_s:02d} | "
            f"ETA {eta_str}",
            flush=True,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        epoch = int(state.epoch) if state.epoch else 0
        elapsed_m = int(elapsed // 60)
        elapsed_s = int(elapsed % 60)
        print(f"\n  --- Epoch {epoch} complete | Elapsed: {elapsed_m:02d}:{elapsed_s:02d} ---\n", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        total_m = int(total_time // 60)
        total_s = int(total_time % 60)

        print(f"\n{'='*70}", flush=True)
        print("TRAINING COMPLETE", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Total time: {total_m:02d}:{total_s:02d}", flush=True)
        print(f"  Total steps: {state.global_step}", flush=True)
        print(f"  Final loss: {self.step_logs[-1]['loss'] if self.step_logs else 'N/A'}", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        # Save training metrics
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({
                "total_time_s": round(total_time, 1),
                "total_steps": state.global_step,
                "step_logs": self.step_logs,
            }, f, indent=2)
        print(f"  Metrics saved to {metrics_path}", flush=True)

        # Plot loss curve
        self._plot_loss()

    def _plot_loss(self):
        """Save loss curve as PNG."""
        losses = [(e["step"], e["loss"]) for e in self.step_logs if e["loss"] is not None]
        if not losses:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps, loss_vals = zip(*losses)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, loss_vals, "b-", linewidth=1.5, label="Training Loss")
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("MRCR LoRA Training Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = os.path.join(self.output_dir, "training_loss.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Loss plot saved to {plot_path}", flush=True)
        except ImportError:
            print("  matplotlib not available, skipping loss plot", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on MRCR")

    # Model
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--enable-yarn", action="store_true", help="Apply YaRN at model load")
    parser.add_argument("--yarn-factor", type=float, default=4.0)

    # RPE
    parser.add_argument("--rpe-config", type=str, default=None,
                        help="Path to RPE YAML config (enables RPE during training)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")

    # Data
    parser.add_argument("--train-file", type=str, required=True, help="Path to MRCR train JSON")
    parser.add_argument("--max-seq-len", type=int, default=8192)

    # Training
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70, flush=True)
    print("MRCR LoRA Training", flush=True)
    print("=" * 70, flush=True)
    print(f"  Base model:       {args.base_model}", flush=True)
    print(f"  YaRN enabled:     {args.enable_yarn} (factor={args.yarn_factor})", flush=True)
    print(f"  RPE config:       {args.rpe_config or '(none)'}", flush=True)
    print(f"  LoRA:             rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}", flush=True)
    print(f"  Train file:       {args.train_file}", flush=True)
    print(f"  Output dir:       {args.output_dir}", flush=True)
    print(f"  Max seq len:      {args.max_seq_len}", flush=True)
    print(f"  LR:               {args.lr}", flush=True)
    print(f"  Epochs:           {args.epochs}", flush=True)
    print(f"  Batch size:       {args.batch_size} (x{args.grad_accum} accum = {args.batch_size * args.grad_accum} effective)", flush=True)
    print(f"  Seed:             {args.seed}", flush=True)
    print(f"  bf16:             {args.bf16}", flush=True)
    print(f"  Grad checkpoint:  {args.gradient_checkpointing}", flush=True)
    print(f"  Timestamp:        {datetime.now().isoformat()}", flush=True)

    # Determine condition name
    if args.rpe_config and "curriculum" in args.rpe_config:
        condition = "rpe_curriculum_lora"
    elif args.rpe_config:
        condition = "rpe_lora"
    elif args.enable_yarn:
        condition = "yarn_lora"
    else:
        condition = "lora_baseline"
    print(f"  Condition:        {condition}", flush=True)

    # --- Load model ---
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model, tokenizer, config_desc = load_model_for_training(
        args.base_model,
        enable_yarn=args.enable_yarn,
        yarn_factor=args.yarn_factor,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        use_gradient_checkpointing=args.gradient_checkpointing,
        torch_dtype=torch_dtype,
    )

    # --- Create dataset ---
    print(f"\n{'='*70}", flush=True)
    print("DATASET", flush=True)
    print(f"{'='*70}", flush=True)
    train_dataset = MRCRDataset(args.train_file, tokenizer, max_seq_len=args.max_seq_len)
    print(f"  Training samples: {len(train_dataset)}", flush=True)

    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps per epoch:  {steps_per_epoch}", flush=True)
    print(f"  Total steps:      {total_steps}", flush=True)

    # --- Callbacks ---
    callbacks = [TrainingProgressCallback(args.output_dir)]

    # RPE callback (reuses existing RPETrainerCallback)
    if args.rpe_config:
        from composable_cot.scripts.rpe_llamafactory_patch import RPETrainerCallback
        rpe_callback = RPETrainerCallback(args.rpe_config)
        callbacks.append(rpe_callback)
        print(f"\n  RPE callback loaded from config: {args.rpe_config}", flush=True)

    # --- Data collator ---
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=args.epochs,  # Keep all epoch checkpoints
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ddp_timeout=180000000,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # --- Train ---
    trainer.train()

    # --- Save final model ---
    print(f"\n  Saving final LoRA weights to {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Save run config ---
    run_config = {
        "condition": condition,
        "config_desc": config_desc,
        "base_model": args.base_model,
        "enable_yarn": args.enable_yarn,
        "yarn_factor": args.yarn_factor,
        "rpe_config": args.rpe_config,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target": args.lora_target,
        "train_file": args.train_file,
        "max_seq_len": args.max_seq_len,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "num_train_samples": len(train_dataset),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = os.path.join(args.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"  Run config saved to {config_path}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("ALL DONE", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
