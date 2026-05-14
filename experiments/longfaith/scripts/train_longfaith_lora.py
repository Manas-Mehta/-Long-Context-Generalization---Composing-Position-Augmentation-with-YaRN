#!/usr/bin/env python
"""Train LoRA adapters on LongFaith-SFT (gpt-4o-mini synthesizer variant).

Forked from experiments/babilong/scripts/train_babilong_lora.py. Only the
dataset class and prompt construction differ — all YaRN / RPE / PoSE
plumbing is identical.

Supports the seven conditions matching the experiment design:
  1. lora_base     : no YaRN, no position-ID aug
  2. y2_base       : YaRN f=2 train (use f=4 at eval)
  3. y2_rpe_cur    : YaRN f=2 + RPE curriculum L=16K
  4. y2_pose_32k   : YaRN f=2 + PoSE target_length=32K
  5. rpe_only      : RPE curriculum L=16K, no YaRN
  6. pose_only     : PoSE target_length=32K, no YaRN
  (Zero-shot rows are handled in eval_longbench_v2.py with no checkpoint.)

Key differences from BABILong training:
  - LongFaith is alpaca-format {instruction, input, output}. `instruction` is
    already a fully formatted CoC prompt with documents [1]..[20] + question;
    `output` is a CoT chain ending in "The answer is X."
  - No mid-training accuracy eval — LongFaith has no fixed label set, and
    LongBench v2 eval is too slow to run mid-training.
  - max_seq_len=9216 covers the longest LongFaith example (~5.7K Qwen tokens)
    with headroom.

Usage:
    # LoRA baseline
    python train_longfaith_lora.py \\
        --train-file experiments/longfaith/data/faith_sft_2k_filtered.json \\
        --output-dir experiments/longfaith/checkpoints/lora_base

    # YaRN + RPE composition
    python train_longfaith_lora.py \\
        --enable-yarn --yarn-factor 2.0 \\
        --rpe-config experiments/longfaith/configs/rpe_config_longfaith_curriculum_L16k.yaml \\
        --train-file experiments/longfaith/data/faith_sft_2k_filtered.json \\
        --output-dir experiments/longfaith/checkpoints/y2_rpe_cur_L16k

    # Smoke test (CPU, 1 step):
    python train_longfaith_lora.py \\
        --train-file experiments/longfaith/data/faith_sft_2k_filtered.json \\
        --output-dir /tmp/longfaith_smoke \\
        --max-steps 1 --no-cuda --no-bf16
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

# Project root on sys.path (script is at experiments/longfaith/scripts/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Prompt construction — LongFaith alpaca format
# ---------------------------------------------------------------------------
#
# Each LongFaith example has the shape:
#   {
#     "instruction": "<CoC header> DOCUMENTS: [1]...[20]... QUESTION: ...",
#     "input": "",
#     "output": "Step 1: ... Step 2: ... The answer is X."
#   }
#
# The `instruction` field is a fully formatted CoC (Chain-of-Citations) prompt
# — we feed it directly as the user message without further templating.
# `input` is always empty in the released SFT file.
# `output` is the assistant-side CoT trace.
#
# Loss is computed only on the `output` portion (prompt tokens masked to -100).
# ---------------------------------------------------------------------------


def build_messages(ex: dict) -> list[dict]:
    user_content = ex["instruction"].strip()
    if ex.get("input"):
        user_content = user_content + "\n\n" + ex["input"].strip()
    assistant_content = ex["output"].strip()
    return [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LongFaithDataset(Dataset):
    """LongFaith-SFT training dataset.

    Each sample is a single-turn QA: one user message (CoC prompt + 20 docs
    + question) and one assistant message (CoT reasoning + final answer).
    Loss is computed on the assistant CoT only.
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 9216):
        with open(data_path) as f:
            self.raw_data = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.samples = []
        n_truncated = 0
        token_counts = []

        print(f"\n  Tokenizing {len(self.raw_data)} samples from {data_path}...", flush=True)

        for i, item in enumerate(self.raw_data):
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(self.raw_data)}...", flush=True)
            sample = self._prepare_sample(item)
            if sample is None:
                continue
            if sample["truncated"]:
                n_truncated += 1
            token_counts.append(len(sample["input_ids"]))
            self.samples.append(sample)

        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        print(f"  Done. {len(self.samples)} samples | "
              f"tokens: {min(token_counts)}–{max(token_counts)} (avg {avg_tokens:.0f}) | "
              f"truncated: {n_truncated}", flush=True)

        if self.samples:
            s = self.samples[0]
            n_masked  = sum(1 for l in s["labels"] if l == -100)
            n_trained = len(s["labels"]) - n_masked
            answer_preview = self.tokenizer.decode(
                s["input_ids"][n_masked:n_masked + 30], skip_special_tokens=False
            )
            print(f"\n  Sample 0 verification:", flush=True)
            print(f"    Total tokens:           {len(s['input_ids'])}", flush=True)
            print(f"    Prompt tokens (masked): {n_masked}", flush=True)
            print(f"    Answer tokens (trained): {n_trained}", flush=True)
            print(f"    Answer starts with:     {answer_preview!r}", flush=True)

    def _prepare_sample(self, item: dict) -> dict | None:
        messages = build_messages(item)

        prompt_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        n_answer_tokens = len(full_ids) - len(prompt_ids)
        if n_answer_tokens <= 0:
            return None

        truncated = False
        if len(full_ids) > self.max_seq_len:
            truncated = True
            excess = len(full_ids) - self.max_seq_len
            full_ids   = full_ids[excess:]
            prompt_ids = prompt_ids[excess:]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        assert len(labels) == len(full_ids)

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
            "truncated":      truncated,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids":      s["input_ids"],
            "attention_mask": s["attention_mask"],
            "labels":         s["labels"],
        }


# ---------------------------------------------------------------------------
# Model loading — identical to BABILong (YaRN logic unchanged)
# ---------------------------------------------------------------------------

def load_model_for_training(
    base_model_name: str,
    enable_yarn:     bool  = False,
    yarn_factor:     float = 2.0,
    lora_rank:       int   = 16,
    lora_alpha:      int   = 32,
    lora_dropout:    float = 0.1,
    lora_target:     str   = "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
    use_gradient_checkpointing: bool = True,
    torch_dtype = torch.bfloat16,
    no_cuda: bool = False,
):
    print(f"\n{'='*70}", flush=True)
    print("MODEL SETUP", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.pad_token_id  = tokenizer.eos_token_id
    print(f"  Tokenizer loaded in {time.time()-t0:.1f}s", flush=True)

    config = AutoConfig.from_pretrained(base_model_name)
    model_kwargs = {"torch_dtype": torch_dtype}

    if enable_yarn:
        print(f"\n  Enabling YaRN: factor={yarn_factor}", flush=True)
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

        print(f"  rope_theta={rope_theta}", flush=True)
        model_kwargs["config"] = config

    device_map = None if no_cuda else "auto"
    print(f"\n  Loading: {base_model_name}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map=device_map, **model_kwargs
    )
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    if enable_yarn:
        _verify_yarn(model, config)

    if use_gradient_checkpointing and not no_cuda:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        print("  Gradient checkpointing: enabled", flush=True)

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
    print(f"\n  LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}", flush=True)
    model.print_trainable_parameters()

    return model, tokenizer


def _verify_yarn(model, config):
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        rotary = model.model.layers[0].self_attn.rotary_emb
    rope_type = getattr(rotary, "rope_type", "unknown")
    print(f"  Actual rope_type: {rope_type}", flush=True)
    if hasattr(rotary, "inv_freq") and rotary.inv_freq is not None:
        base = getattr(config, "rope_theta", 1000000.0)
        dim  = rotary.inv_freq.shape[0] * 2
        vanilla = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        n_changed = (rotary.inv_freq.float().cpu() - vanilla).abs().gt(1e-8).sum().item()
        print(f"  YaRN verification: {n_changed}/{len(vanilla)} inv_freq dims differ", flush=True)
        if n_changed == 0:
            print("  WARNING: YaRN NOT APPLIED — inv_freq unchanged.", flush=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TrainingProgressCallback(TrainerCallback):
    """Per-step logging with timing, memory, ETA. Also logs per-layer LoRA
    gradient magnitudes every `grad_log_every` steps for mechanistic analysis.
    """

    def __init__(self, output_dir: str, grad_log_every: int = 100):
        self.output_dir      = output_dir
        self.grad_log_every  = grad_log_every
        self.start_time      = None
        self.step_logs       = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n{'='*70}", flush=True)
        print("TRAINING STARTED", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Total steps:    {state.max_steps}", flush=True)
        print(f"  Epochs:         {args.num_train_epochs}", flush=True)
        print(f"  Effective batch:{args.per_device_train_batch_size * args.gradient_accumulation_steps}", flush=True)
        print(f"  LR:             {args.learning_rate}", flush=True)
        print(f"  Warmup ratio:   {args.warmup_ratio}", flush=True)
        print(f"  Start:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"{'='*70}\n", flush=True)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None or self.start_time is None:
            return

        step     = state.global_step
        max_steps = state.max_steps
        epoch    = state.epoch or 0
        loss     = logs.get("loss")
        lr       = logs.get("learning_rate")
        elapsed  = time.time() - self.start_time

        gpu_mem = "N/A"
        if torch.cuda.is_available():
            gpu_mem = f"{torch.cuda.max_memory_allocated() / 1e9:.1f}GB"

        eta_str = "N/A"
        if step > 0 and max_steps > 0:
            sps = elapsed / step
            rem = (max_steps - step) * sps
            eta_str = f"{int(rem//3600):02d}:{int((rem%3600)//60):02d}:{int(rem%60):02d}"

        entry = {
            "step": step, "epoch": round(epoch, 2),
            "loss": round(loss, 4) if loss else None,
            "learning_rate": lr, "elapsed_s": round(elapsed, 1),
        }
        self.step_logs.append(entry)

        loss_str = f"{loss:.4f}" if loss else "N/A"
        lr_str   = f"{lr:.2e}"   if lr   else "N/A"
        print(
            f"  Step {step:>5}/{max_steps} | Ep {epoch:>5.2f} | "
            f"Loss {loss_str} | LR {lr_str} | GPU {gpu_mem} | ETA {eta_str}",
            flush=True,
        )

        if model is not None and step % self.grad_log_every == 0 and step > 0:
            self._log_layer_gradients(model, step)

    def _log_layer_gradients(self, model, step: int):
        try:
            import wandb
            if not wandb.run:
                return
            grad_log = {}
            for name, param in model.named_parameters():
                if param.grad is not None and "lora_" in name:
                    parts = name.split(".")
                    layer_idx = None
                    for j, p in enumerate(parts):
                        if p == "layers" and j + 1 < len(parts):
                            try:
                                layer_idx = int(parts[j + 1])
                            except ValueError:
                                pass
                    if layer_idx is None:
                        continue
                    module = None
                    for mod in ["q_proj", "k_proj", "v_proj", "o_proj",
                                "up_proj", "down_proj", "gate_proj"]:
                        if mod in name:
                            module = mod
                            break
                    if module is None:
                        continue
                    key = f"grad/layer_{layer_idx:02d}_{module}"
                    val = param.grad.abs().mean().item()
                    grad_log[key] = max(grad_log.get(key, 0.0), val)
            if grad_log:
                wandb.log(grad_log, step=step)
        except Exception:
            pass

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        print(
            f"\n  --- Epoch {int(state.epoch or 0)} complete | "
            f"Elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d} ---\n",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self.start_time
        print(f"\n{'='*70}", flush=True)
        print("TRAINING COMPLETE", flush=True)
        print(f"  Total time: {int(total//60):02d}:{int(total%60):02d}", flush=True)
        print(f"  Steps: {state.global_step}", flush=True)
        if self.step_logs:
            print(f"  Final loss: {self.step_logs[-1]['loss']}", flush=True)

        os.makedirs(self.output_dir, exist_ok=True)
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"total_time_s": round(total, 1), "step_logs": self.step_logs}, f, indent=2)
        print(f"  Metrics -> {metrics_path}", flush=True)
        self._plot_loss()

    def _plot_loss(self):
        losses = [(e["step"], e["loss"]) for e in self.step_logs if e["loss"] is not None]
        if not losses:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            steps, vals = zip(*losses)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, vals, "b-", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("LongFaith LoRA Training Loss")
            ax.grid(True, alpha=0.3)
            path = os.path.join(self.output_dir, "training_loss.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Loss plot -> {path}", flush=True)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA on LongFaith-SFT")

    p.add_argument("--base-model",  default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--enable-yarn", action="store_true")
    p.add_argument("--yarn-factor", type=float, default=2.0)
    p.add_argument("--no-cuda",     action="store_true")

    p.add_argument("--rpe-config",  default=None)
    p.add_argument("--pose-config", default=None)

    p.add_argument("--lora-rank",    type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target",  default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")

    p.add_argument("--train-file",   required=True)
    p.add_argument("--max-seq-len",  type=int, default=9216)

    p.add_argument("--output-dir",    required=True)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--epochs",        type=int,   default=2)
    p.add_argument("--batch-size",    type=int,   default=1)
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--warmup-ratio",  type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max-steps",     type=int,   default=-1, help="Override epochs (for smoke testing)")
    p.add_argument("--bf16",          action="store_true", default=True)
    p.add_argument("--no-bf16",       dest="bf16", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)

    p.add_argument("--grad-log-every", type=int, default=100)
    p.add_argument("--wandb-run-name", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.pose_config and args.enable_yarn:
        condition = "y2_pose"
    elif args.rpe_config and args.enable_yarn:
        condition = "y2_rpe_cur"
    elif args.pose_config:
        condition = "pose_only"
    elif args.rpe_config:
        condition = "rpe_only"
    elif args.enable_yarn:
        condition = "y2_base"
    else:
        condition = "lora_base"

    run_name = args.wandb_run_name or condition

    print("=" * 70, flush=True)
    print("LongFaith — LoRA Training", flush=True)
    print("=" * 70, flush=True)
    print(f"  Condition:    {condition}", flush=True)
    print(f"  Base model:   {args.base_model}", flush=True)
    print(f"  YaRN:         {args.enable_yarn} (factor={args.yarn_factor})", flush=True)
    print(f"  RPE config:   {args.rpe_config or '(none)'}", flush=True)
    print(f"  PoSE config:  {args.pose_config or '(none)'}", flush=True)
    print(f"  Train file:   {args.train_file}", flush=True)
    print(f"  Output dir:   {args.output_dir}", flush=True)
    print(f"  max_seq_len:  {args.max_seq_len}", flush=True)
    print(f"  LR:           {args.lr}", flush=True)
    print(f"  Epochs:       {args.epochs}", flush=True)
    print(f"  Batch×accum:  {args.batch_size}×{args.grad_accum}={args.batch_size*args.grad_accum}", flush=True)
    print(f"  Warmup:       {args.warmup_ratio}", flush=True)
    print(f"  Seed:         {args.seed}", flush=True)
    print(f"  W&B run:      {run_name}", flush=True)
    print(f"  Timestamp:    {datetime.now().isoformat()}", flush=True)

    torch_dtype = torch.bfloat16 if (args.bf16 and not args.no_cuda) else torch.float32
    model, tokenizer = load_model_for_training(
        args.base_model,
        enable_yarn=args.enable_yarn,
        yarn_factor=args.yarn_factor,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        use_gradient_checkpointing=args.gradient_checkpointing,
        torch_dtype=torch_dtype,
        no_cuda=args.no_cuda,
    )

    print(f"\n{'='*70}", flush=True)
    print("DATASET", flush=True)
    print(f"{'='*70}", flush=True)
    train_dataset = LongFaithDataset(args.train_file, tokenizer, args.max_seq_len)
    print(f"  Training samples: {len(train_dataset)}", flush=True)

    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.grad_accum))
    total_steps     = steps_per_epoch * args.epochs if args.max_steps < 0 else args.max_steps
    print(f"  Steps/epoch:      {steps_per_epoch}", flush=True)
    print(f"  Total steps:      {total_steps}", flush=True)

    callbacks = [TrainingProgressCallback(args.output_dir, grad_log_every=args.grad_log_every)]

    if args.rpe_config:
        from posaug.callbacks_rpe import RPETrainerCallback
        callbacks.append(RPETrainerCallback(args.rpe_config))
        print(f"\n  RPE callback: {args.rpe_config}", flush=True)

    if args.pose_config:
        from posaug.callbacks_pose import PoSETrainerCallback
        callbacks.append(PoSETrainerCallback(args.pose_config))
        print(f"\n  PoSE callback: {args.pose_config}", flush=True)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=(args.bf16 and not args.no_cuda),
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=4 if not args.no_cuda else 0,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing and not args.no_cuda,
        gradient_checkpointing_kwargs={"use_reentrant": False} if (args.gradient_checkpointing and not args.no_cuda) else None,
        max_grad_norm=args.max_grad_norm,
        no_cuda=args.no_cuda,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train()

    print(f"\n  Saving LoRA weights -> {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    run_config = {
        "condition": condition, "base_model": args.base_model,
        "enable_yarn": args.enable_yarn, "yarn_factor": args.yarn_factor,
        "rpe_config": args.rpe_config, "pose_config": args.pose_config,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "train_file": args.train_file, "max_seq_len": args.max_seq_len,
        "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "warmup_ratio": args.warmup_ratio, "seed": args.seed,
        "num_train_samples": len(train_dataset),
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print("ALL DONE", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    main()
