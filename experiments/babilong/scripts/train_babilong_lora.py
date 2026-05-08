#!/usr/bin/env python
"""Train LoRA adapters on BABILong QA3 (Three Supporting Facts).

Supports six conditions matching our experiment design:
  1. LoRA-base:    Normal RoPE, no position tricks
  2. Y2-base:      YaRN f=2 during training, use f=4 at eval
  3. Y2-Rc16:      YaRN f=2 + RPE curriculum L=16K
  4. Y2-P32:       YaRN f=2 + PoSE target_length=32K
  5. RPE-only:     RPE curriculum L=16K, no YaRN
  6. PoSE-only:    PoSE target_length=32K, no YaRN

Core model loading (YaRN, RPE, PoSE) is reused directly from the MRCR
training pipeline. Only the dataset class and logging are new.

Key differences from MRCR training:
  - BABILong is single-turn QA, not multi-turn conversation
  - Answer is 1 token (a room name) vs ~30 tokens in MRCR
  - max_seq_len=9216 (8K bin peaks at 8,569 tokens)
  - 3 epochs (20K samples = 5K steps/epoch)
  - warmup_ratio=0.05 (not 0.1 — 10% would waste 1,500 steps on warmup)
  - W&B offline logging for HPC (no internet on compute nodes)
  - Per-layer gradient magnitude logged for mechanistic analysis
  - Mid-training accuracy eval every N steps on small subsets

Usage:
    # LoRA baseline
    python train_babilong_lora.py \
        --train-file composable_cot/BABIlong/data/train/all_train.json \
        --output-dir composable_cot/BABIlong/checkpoints/lora_base

    # YaRN + LoRA
    python train_babilong_lora.py \
        --enable-yarn --yarn-factor 2.0 \
        --train-file composable_cot/BABIlong/data/train/all_train.json \
        --output-dir composable_cot/BABIlong/checkpoints/y2_base

    # YaRN + RPE curriculum
    python train_babilong_lora.py \
        --enable-yarn --yarn-factor 2.0 \
        --rpe-config composable_cot/BABIlong/configs/rpe_config_babilong_curriculum_L16k.yaml \
        --train-file composable_cot/BABIlong/data/train/all_train.json \
        --output-dir composable_cot/BABIlong/checkpoints/y2_rpe_cur_L16k

    # Smoke test (local, no GPU, 3 steps):
    python train_babilong_lora.py \
        --train-file composable_cot/BABIlong/data/train/0k.json \
        --output-dir /tmp/babilong_smoke \
        --max-steps 3 --no-cuda --no-bf16
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

# Project root on sys.path (script is at RPE/composable_cot/BABIlong/scripts/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# QA3 Prompt Template
#
# Follows the official BABILong prompt structure from:
#   https://github.com/booydar/babilong/blob/main/babilong/prompts.py
#
# Official structure:
#   {instruction} + {examples} + {post_prompt} + <context>{context}</context> + Question: {q}
#
# We use:
#   - Official QA3 instruction text (verbatim from DEFAULT_PROMPTS['qa3'])
#   - Official <context> tags around the document
#   - NO few-shot examples: official examples show sentence-format answers
#     ("Before the kitchen the apple was in the bathroom.") but our training
#     labels are single words ("bathroom"). Including them would create a
#     train/label inconsistency. Fine-tuning does not need few-shot examples.
#   - Post-prompt changed to single-word format to match training labels.
# ---------------------------------------------------------------------------

# Source: DEFAULT_PROMPTS['qa3']['instruction'] in babilong/prompts.py
QA3_INSTRUCTION = (
    "I give you context with the facts about locations and actions of different persons "
    "hidden in some random text and a question. "
    "You need to answer the question based only on the information from the facts.\n"
    "If a person got an item in the first location and travelled to the second location "
    "the item is also in the second location. "
    "If a person dropped an item in the first location and moved to the second location "
    "the item remains in the first location."
)

# Changed from official sentence format to single-word format to match training labels.
QA3_POST_PROMPT = (
    "Your answer must be exactly one word — one of: "
    "bathroom, bedroom, garden, hallway, kitchen, office. "
    "Do not write anything else."
)

QA3_LABELS = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]


def build_messages(sample: dict) -> list[dict]:
    """Build chat messages for a BABILong QA3 sample.

    Follows the official BABILong prompt structure (babilong/prompts.py):
        {instruction}

        <context>
        {context}
        </context>

        Question: {question}
        {post_prompt}

    The context is extracted from the pre-formatted user message in our
    JSON files. The raw question is read from the 'question' field saved
    by prepare_babilong.py.
    """
    if "messages" in sample:
        user_content = sample["messages"][0]["content"]
        answer = sample["answer"]
        question_text = sample.get("question", "").strip()
    else:
        # Raw HuggingFace format fallback
        user_content = f"{sample['input'].strip()}\nQuestion: {sample['question'].strip()}"
        answer = sample["target"].strip().lower()
        question_text = sample["question"].strip()

    # Extract just the context (everything before "\nQuestion:").
    # Prepared format: "{input}\nQuestion: {question}\nAnswer with only one word."
    parts = user_content.rsplit("\nQuestion:", 1)
    context = parts[0].strip() if len(parts) > 1 else user_content.strip()

    # If 'question' field wasn't saved (old data format), fall back to parsing.
    if not question_text:
        if len(parts) > 1:
            question_text = parts[1].replace("\nAnswer with only one word.", "").strip()

    full_user_content = (
        f"{QA3_INSTRUCTION}\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question_text}\n"
        f"{QA3_POST_PROMPT}"
    )

    return [
        {"role": "user",      "content": full_user_content},
        {"role": "assistant", "content": answer},
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BABILongDataset(Dataset):
    """BABILong QA3 training dataset.

    Each sample is a single-turn QA: one user message (context + question)
    and one assistant message (a single room name). Loss is computed only
    on the assistant's answer token.

    The context can be very short (0K bin: ~350 tokens) or up to 8K+
    tokens. We tokenize on-the-fly during __init__ and report stats.
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
            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{len(self.raw_data)}...", flush=True)
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

        # Print first sample for verification
        if self.samples:
            s = self.samples[0]
            n_masked  = sum(1 for l in s["labels"] if l == -100)
            n_trained = len(s["labels"]) - n_masked
            answer_preview = self.tokenizer.decode(
                s["input_ids"][n_masked:n_masked + 10], skip_special_tokens=False
            )
            print(f"\n  Sample 0 verification:", flush=True)
            print(f"    Total tokens:           {len(s['input_ids'])}", flush=True)
            print(f"    Prompt tokens (masked): {n_masked}", flush=True)
            print(f"    Answer tokens (trained): {n_trained}", flush=True)
            print(f"    Answer starts with:     {answer_preview!r}", flush=True)
            print(f"    Expected answer:        {self.raw_data[0].get('answer', '?')!r}", flush=True)

    def _prepare_sample(self, item: dict) -> dict | None:
        messages = build_messages(item)

        # Prompt only (no answer) — for masking
        prompt_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # Full sequence (prompt + answer)
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        n_answer_tokens = len(full_ids) - len(prompt_ids)
        if n_answer_tokens <= 0:
            return None

        # Left-truncate if over max_seq_len (remove from start of prompt)
        truncated = False
        if len(full_ids) > self.max_seq_len:
            truncated = True
            excess = len(full_ids) - self.max_seq_len
            full_ids   = full_ids[excess:]
            prompt_ids = prompt_ids[excess:]

        # Labels: -100 for prompt, actual ids for answer only
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
# Model loading — reused from MRCR (identical YaRN logic)
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
    """Per-step logging with timing, memory, ETA, and W&B metrics.

    Also logs per-layer LoRA gradient magnitudes every `grad_log_every` steps.
    These gradients indicate which layers RPE/PoSE are forcing to change,
    which is useful for mechanistic analysis.
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

        # Per-layer gradient magnitudes for mechanistic analysis
        if model is not None and step % self.grad_log_every == 0 and step > 0:
            self._log_layer_gradients(model, step)

    def _log_layer_gradients(self, model, step: int):
        """Log mean absolute gradient per LoRA module per layer.

        Logged as grad/layer_{i}_{module} in W&B.
        This tells us which layers RPE/PoSE are forcing to change most
        compared to LoRA-base — the mechanistic analysis signal.
        """
        try:
            import wandb
            if not wandb.run:
                return
            grad_log = {}
            for name, param in model.named_parameters():
                if param.grad is not None and "lora_" in name:
                    # name format: base_model.model.layers.{i}.self_attn.q_proj.lora_A.weight
                    parts = name.split(".")
                    # Find layer index
                    layer_idx = None
                    for j, p in enumerate(parts):
                        if p == "layers" and j + 1 < len(parts):
                            try:
                                layer_idx = int(parts[j + 1])
                            except ValueError:
                                pass
                    if layer_idx is None:
                        continue
                    # Find module name (q_proj, k_proj, etc.)
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
                    if key not in grad_log:
                        grad_log[key] = val
                    else:
                        grad_log[key] = max(grad_log[key], val)  # lora_A and lora_B — take max
            if grad_log:
                wandb.log(grad_log, step=step)
        except Exception:
            pass  # Never crash training due to logging

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
            ax.set_title("BABILong LoRA Training Loss")
            ax.grid(True, alpha=0.3)
            path = os.path.join(self.output_dir, "training_loss.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Loss plot -> {path}", flush=True)
        except ImportError:
            pass


class MidTrainingEvalCallback(TrainerCallback):
    """Run accuracy eval on small fixed subsets every `eval_every` steps.

    Evaluates on 3 context bins: 0K (fast, in-distribution),
    8K (top of training range), 32K (out-of-distribution).
    Uses a fixed random subset of `n_samples` per bin for speed.

    Catches problems early:
    - Fast memorization: 0K acc -> 99% in epoch 1, 32K stays at 25%
    - Underfitting: all bins stay near 25% after epoch 2
    - Healthy training: 0K > 8K > 32K, all improving
    """

    def __init__(
        self,
        tokenizer,
        eval_dir:      str,
        output_dir:    str,
        eval_every:    int   = 500,
        n_samples:     int   = 200,
        max_new_tokens: int  = 20,
        yarn_factor:   float = 4.0,   # eval YaRN factor (usually 4.0)
        enable_yarn_eval: bool = False,
        seed:          int   = 42,
    ):
        self.tokenizer       = tokenizer
        self.eval_dir        = eval_dir
        self.output_dir      = output_dir
        self.eval_every      = eval_every
        self.n_samples       = n_samples
        self.max_new_tokens  = max_new_tokens
        self.enable_yarn_eval = enable_yarn_eval
        self.yarn_factor     = yarn_factor
        self.seed            = seed
        self.eval_results    = []

        # Load fixed eval subsets once at init
        self._subsets = {}
        for bin_label in ["0k", "8k", "32k"]:
            path = os.path.join(eval_dir, f"{bin_label}.json")
            if not os.path.exists(path):
                print(f"  MidTrainingEval: {path} not found, skipping {bin_label}", flush=True)
                continue
            import random
            rng = random.Random(seed)
            with open(path) as f:
                all_samples = json.load(f)
            subset = rng.sample(all_samples, min(n_samples, len(all_samples)))
            self._subsets[bin_label] = subset
            print(f"  MidTrainingEval: loaded {len(subset)} samples for {bin_label} eval", flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            return
        if state.global_step % self.eval_every != 0:
            return
        if model is None or not self._subsets:
            return

        print(f"\n  [Mid-training eval @ step {state.global_step}]", flush=True)
        model.eval()
        results = {}

        with torch.no_grad():
            for bin_label, subset in self._subsets.items():
                acc = self._eval_bin(model, subset, bin_label)
                results[bin_label] = acc
                print(f"    {bin_label} acc: {acc:.3f}", flush=True)

        model.train()

        # Log to W&B
        try:
            import wandb
            if wandb.run:
                wandb.log(
                    {f"eval/acc_{b}": v for b, v in results.items()},
                    step=state.global_step,
                )
        except Exception:
            pass

        self.eval_results.append({"step": state.global_step, **results})
        self._save_eval_results()

    def _eval_bin(self, model, subset: list, bin_label: str) -> float:
        correct = 0
        device  = next(model.parameters()).device

        for sample in subset:
            messages  = build_messages(sample)
            prompt    = self.tokenizer.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Skip if too long for current GPU memory
            if input_ids.shape[1] > 70000:
                continue

            try:
                out = model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=True
                ).strip().lower()
                target = sample.get("answer", "").strip().lower()
                # Pass only the question sentence to _grade(), NOT the full context
                # or the post_prompt (which lists all 6 labels — passing it would
                # exclude all valid answers, making accuracy permanently 0%).
                # Prefer the raw 'question' field saved by prepare_babilong.py;
                # fall back to parsing from the message content.
                question_only = (
                    sample.get("question", "").strip()
                    or self._extract_question(messages[0]["content"])
                )
                if self._grade(response, target, question_only):
                    correct += 1
            except Exception:
                continue

        return correct / len(subset) if subset else 0.0

    @staticmethod
    def _extract_question(user_content: str) -> str:
        """Extract just the question line from the full user message.

        IMPORTANT: must extract ONLY the question sentence, not the post_prompt.
        Our post_prompt lists all 6 labels — passing it to _grade() would cause
        all labels to be excluded, making accuracy permanently 0%.

        We take only the text from "Question:" up to the first newline.
        e.g.  "Question: Where was the football before the garden?"
        """
        idx = user_content.rfind("Question:")
        if idx == -1:
            return ""
        question_line = user_content[idx:]
        newline_idx = question_line.find("\n")
        if newline_idx != -1:
            question_line = question_line[:newline_idx]
        return question_line  # "Question: Where was the football before the garden?"

    def _grade(self, response: str, target: str, question: str) -> bool:
        """Official BABILong grading: closed-vocabulary label detection.

        preprocess_output and compare_answers logic taken verbatim from:
          https://github.com/booydar/babilong/blob/main/babilong/metrics.py
        """
        # preprocess_output — exact match to official metrics.py
        response = response.lower()
        response = response.split('.')[0]
        response = response.split('<context>')[0]
        response = response.split('<example>')[0]
        response = response.split('Question')[0]   # matches official (capital Q, post-lowercase)

        question = question.lower()
        labels   = set(QA3_LABELS)

        labels_in_question = {l for l in labels if l in question}
        labels_in_response = {l for l in labels if l in response}
        labels_in_response -= labels_in_question

        return target in labels_in_response and len(labels_in_response) == 1

    def _save_eval_results(self):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "mid_training_eval.json")
        with open(path, "w") as f:
            json.dump(self.eval_results, f, indent=2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA on BABILong QA3")

    # Model
    p.add_argument("--base-model",  default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--enable-yarn", action="store_true")
    p.add_argument("--yarn-factor", type=float, default=2.0)
    p.add_argument("--no-cuda",     action="store_true", help="CPU-only (for smoke testing)")

    # Position methods
    p.add_argument("--rpe-config",  default=None)
    p.add_argument("--pose-config", default=None)

    # LoRA
    p.add_argument("--lora-rank",    type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-target",  default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")

    # Data
    p.add_argument("--train-file",   required=True)
    p.add_argument("--eval-dir",     default=None, help="Dir with 0k.json, 8k.json, 32k.json for mid-training eval")
    p.add_argument("--max-seq-len",  type=int, default=9216)

    # Training
    p.add_argument("--output-dir",    required=True)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=1)
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--warmup-ratio",  type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max-steps",     type=int,   default=-1, help="Override epochs (for smoke testing)")
    p.add_argument("--bf16",          action="store_true", default=True)
    p.add_argument("--no-bf16",       dest="bf16", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)

    # Monitoring
    p.add_argument("--eval-every",    type=int, default=500, help="Mid-training eval interval (steps)")
    p.add_argument("--eval-samples",  type=int, default=200, help="Samples per bin in mid-training eval")
    p.add_argument("--grad-log-every",type=int, default=100, help="Per-layer gradient log interval")
    p.add_argument("--wandb-run-name",default=None)  # kept for run_name logging only

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Determine condition name for logging
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
    print("BABILong QA3 — LoRA Training", flush=True)
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

    # W&B disabled — metrics logged to JSON by TrainingProgressCallback instead.

    # Load model
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

    # Dataset
    print(f"\n{'='*70}", flush=True)
    print("DATASET", flush=True)
    print(f"{'='*70}", flush=True)
    train_dataset = BABILongDataset(args.train_file, tokenizer, args.max_seq_len)
    print(f"  Training samples: {len(train_dataset)}", flush=True)

    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.grad_accum))
    total_steps     = steps_per_epoch * args.epochs if args.max_steps < 0 else args.max_steps
    print(f"  Steps/epoch:      {steps_per_epoch}", flush=True)
    print(f"  Total steps:      {total_steps}", flush=True)

    # Callbacks
    callbacks = [TrainingProgressCallback(args.output_dir, grad_log_every=args.grad_log_every)]

    if args.eval_dir and os.path.isdir(args.eval_dir):
        callbacks.append(MidTrainingEvalCallback(
            tokenizer=tokenizer,
            eval_dir=args.eval_dir,
            output_dir=args.output_dir,
            eval_every=args.eval_every,
            n_samples=args.eval_samples,
        ))
        print(f"\n  Mid-training eval: every {args.eval_every} steps on 0K/8K/32K subsets", flush=True)

    if args.rpe_config:
        from composable_cot.scripts.rpe_llamafactory_patch import RPETrainerCallback
        callbacks.append(RPETrainerCallback(args.rpe_config))
        print(f"\n  RPE callback: {args.rpe_config}", flush=True)

    if args.pose_config:
        from composable_cot.scripts.pose_patch import PoSETrainerCallback
        callbacks.append(PoSETrainerCallback(args.pose_config))
        print(f"\n  PoSE callback: {args.pose_config}", flush=True)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    # Training arguments
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
        save_total_limit=3,           # Keep all 3 epoch checkpoints
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

    # Save
    print(f"\n  Saving LoRA weights -> {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save run config
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

    try:
        import wandb
        if wandb.run:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
