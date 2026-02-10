Complete Audit & Walkthrough: RPE + Composable CoT
1. THE CORE IDEA — What Is RPE and Why Does It Matter?
The Problem: Standard transformers use sequential position IDs [0, 1, 2, ..., N-1]. If you train on sequences up to length 40, the model has never seen position 41, 42, etc. At test time, those positions are out-of-distribution (OOD) — the model fails catastrophically.

DeepMind's Solution (Ruoss et al., ACL 2023): During training, instead of [0, 1, 2, 3, 4], randomly sample N integers from a much larger range [0, L) and sort them: e.g. [127, 892, 2341, 5892, 7103]. At inference, use standard sequential positions [0, 1, ..., N-1]. Since the model trained with diverse position patterns spanning [0, L), the sequential inference positions fall well within the seen range regardless of sequence length.

The algorithm (from DeepMind's positional_encodings.py:196-226):


# DeepMind's JAX version (noisy_fixed_positional_encodings):
indexes = jrandom.choice(rng, jnp.arange(noise_max_length), shape=(sequence_length,), replace=False)
indexes = jnp.sort(indexes)
encodings = fixed_positional_encodings[indexes]
Your implementation (rpe/core.py:72-80):


# Your PyTorch version:
perm = torch.randperm(self.max_simulation_length)
positions = perm[:seq_length].sort().values
Verdict: Algorithmically identical. Both sample seq_length unique integers from [0, L) without replacement, then sort ascending. torch.randperm(L)[:N] is equivalent to jrandom.choice(arange(L), shape=(N,), replace=False). Both guarantee uniqueness, ascending order, and uniform coverage of the range.

2. PHASE 1 — Standalone RPE Reproduction (Tiny Qwen2 from scratch)
This is scripts/train_reverse_string.py. The goal was to reproduce DeepMind's key result before integrating with CCoT.

Hyperparameter Comparison
Parameter	DeepMind (example.py)	Your Phase 1 (train_reverse_string.py)	Match?
Task	reverse_string, vocab_size=2	Binary string reversal	Yes
Train lengths	UniformCurriculum(values=range(1, 41))	randint(1, 40)	Yes
Eval lengths	max_range_test_length=100	max_eval_length=100	Yes
RPE L	noise_max_length: 2048	max_simulation_length=2048	Yes
Hidden dim	embedding_dim: 64	hidden_size=64	Yes
Layers	num_layers: 5	num_layers=5	Yes
Heads	8 (implicit)	num_heads=8	Yes
Dropout	dropout_prob: 0.1	attention_dropout=0.1	Yes
Batch size	batch_size=128	batch_size=128	Yes
LR	learning_rate=1e-3	learning_rate=1e-3	Yes
Optimizer	optax.adam (no weight decay)	AdamW(weight_decay=0)	Equivalent
Grad clip	max_grad_norm=1.0	max_grad_norm=1.0	Yes
Steps	training_steps=10_000	training_steps=10000	Yes
Eval samples	range_test_total_batch_size=512	eval_samples_per_length=512	Yes
Seed	seed=0	seed=0	Yes
Verdict: All hyperparameters match. The one architectural difference — encoder-only (DeepMind) vs decoder-only (yours) — is deliberate and documented. DeepMind uses bidirectional attention; you use causal masking. This makes your test stricter because errors compound autoregressively.

Why Train from Scratch?
Phase 1 trains a tiny ~330K parameter Qwen2 from random weights, not a pretrained 7B model. This isolates the RPE variable — you're not confounding with pretrained knowledge. DeepMind also trains from scratch on each task. This is the correct scientific methodology.

Phase 1 Results
Baseline: perfect in-distribution (1.0), complete failure OOD (0.0). RPE: 0.56 at length 50 (OOD). The core finding reproduces — RPE enables generalization that is completely absent in the baseline. The lower absolute OOD accuracy vs DeepMind's ~0.8 is attributed to decoder-only autoregressive error compounding, which is well-documented in EXPERIMENT_REPORT.md.

3. PHASE 2 — RPE + Composable CoT Integration
This is the migration from "tiny model from scratch" to "Qwen2.5-7B with LoRA fine-tuning via LLaMA-Factory."

3.1 The RPE Patching Pipeline
The patching architecture has three layers:

Layer 1: rpe/core.py — Pure algorithm. RandomizedPositionalEncoding.get_randomized_positions(seq_length) returns sorted random positions. No model awareness.

Layer 2: rpe/patching.py — Model integration. RPEPatcher wraps model.forward() with a closure that:

Intercepts the forward call
Checks model.training (line 82)
If training: generates random positions per batch element (lines 84-88)
If eval: passes through standard sequential positions (line 90)
Uses @functools.wraps(original_forward) to preserve the function signature (line 60) — this was a critical bug fix from Phase 1 where the inspect.signature issue broke HF's generation utilities
Layer 3: composable_cot/scripts/rpe_llamafactory_patch.py — LLaMA-Factory integration. RPETrainerCallback is a standard HuggingFace TrainerCallback that:

on_train_begin: calls apply_rpe_patch(model, config) (line 137)
on_train_end: calls remove_rpe_patch(patcher) (line 142)
Triggered by RPE_CONFIG_PATH environment variable
Flow:


LLaMA-Factory starts training
  → RPETrainerCallback.on_train_begin()
    → apply_rpe_patch(model, config)
      → RPEPatcher(model, {"max_simulation_length": 8192}).patch()
        → model.forward = rpe_forward (wrapped)
          → Every forward call during training:
            → model.training == True?
              → Yes: generate random positions from [0, 8192)
              → No: use standard [0, 1, ..., N-1]
  → Training loop runs normally (LLaMA-Factory SFT)
  → RPETrainerCallback.on_train_end()
    → patcher.unpatch() → restore original forward
Verdict: The patching logic is sound. The model.training gate is the correct mechanism — during LLaMA-Factory's SFT training loop, the model is in .train() mode, so RPE is active. During any evaluation within training, HF Trainer calls model.eval(), so standard positions are used. The unpatch at on_train_end ensures the saved checkpoint has a clean forward method.

3.2 The Data Generation
composable_cot/scripts/generate_reverse_string_data.py creates data in CCoT format.

Training data format (with CoT trace):


{
  "instruction": "Reverse the following binary string: 0 1 0 0 0 0 0 1 answer: ",
  "output": "<prefix> The 1st character from the end is 1. The 2nd character from the end is 0. ... So the answer is 10000010.</prefix><|endoftext|>"
}
OOD test data format (answer-only for evaluation):


{
  "instruction": "Reverse the following binary string: 1 0 1 1 ... answer: ",
  "output": "11110010000101100111101101011001010111101<|endoftext|>",
  "string_length": 41
}
Key design decisions:

Space-separated digits (0 1 0 0 not 0100) — ensures Qwen's BPE tokenizer tokenizes each digit as a separate token. Without spaces, BPE would merge digits into multi-character tokens, breaking the positional alignment.
Binary alphabet (vocab_size=2) — matches DeepMind exactly
Lengths 1-40 for training, 41-100 for OOD — matches DeepMind exactly
5000 train / 500 val — reasonable for LoRA fine-tuning
10 test samples per length (stratified) — ensures every length from 1-100 is tested equally
The CCoT trace provides step-by-step reasoning ("The 1st character from the end is X"). This is the key contribution from the CCoT paper — the model learns to generate reasoning traces that decompose the problem. The question is whether RPE + CCoT reasoning traces together enable better length generalization than either alone.

Verdict: Data generation is correct. The length ranges, binary alphabet, and uniform curriculum all match DeepMind. The CCoT format follows the composable_cot paper's conventions.

3.3 Training Configuration
composable_cot/scripts/llamafactory/reverse_string_composable_cot.yaml:

Parameter	Value	Notes
Base model	Qwen/Qwen2.5-7B	7B parameter pretrained model
Fine-tuning	LoRA (rank=8, alpha=16, dropout=0.2)	~0.1% of params trainable
LoRA targets	q,k,v,o,up,down,gate_proj	All attention + MLP projections
Stage	SFT	Supervised fine-tuning
Template	empty	Raw instruction/output format, no chat template
Cutoff length	1024	Max token sequence length per example
Batch size	4 per device	Smaller than Phase 1's 128 (7B model is much larger)
LR	1e-3	Same as DeepMind/Phase 1
Scheduler	linear	Different from Phase 1's constant schedule
Epochs	5	Different from Phase 1's step-based training
Precision	bf16	Standard for A100
Eval strategy	epoch	Evaluate at end of each epoch
Best model	by eval_loss	Saves best checkpoint
Seed	42	Different from Phase 1's 0
RPE config (composable_cot/scripts/rpe_config.yaml):

max_simulation_length: 8192 — increased from Phase 1's 2048 because Qwen2.5 supports up to 32K positions. The paper states L should be >> max training length. With CCoT traces, training sequences can be hundreds of tokens long, so 8192 provides good headroom.
training_mode: true, inference_mode: false — RPE only during training, matching the paper.
3.4 Why Train Baseline AND RPE?
This is the standard controlled experiment design:

Baseline (no RPE): Train Qwen2.5-7B + LoRA on the same data with standard sequential positions. This establishes what the model can do without RPE.

RPE: Train the identical model with identical data but with randomized positions during training.

The only variable between the two conditions is RPE. Same model, same data, same hyperparameters, same hardware. The environment variable RPE_CONFIG_PATH is the switch — unset for baseline, set for RPE. This is visible in run_experiment.sh:86-111:


# Baseline: no RPE_CONFIG_PATH
unset RPE_CONFIG_PATH
llamafactory-cli train ...

# RPE: set RPE_CONFIG_PATH
RPE_CONFIG_PATH="${RPE_CONFIG}" llamafactory-cli train ...
If you only ran RPE without a baseline, you couldn't attribute the results to RPE — the model might just be good at the task regardless.

3.5 Evaluation Pipeline
composable_cot/scripts/eval_length_generalization.py:

Load Qwen2.5-7B base model
Load LoRA adapter checkpoint
Merge LoRA weights into base model (merge_and_unload())
For each test example: prompt the model, generate autoregressively (greedy, do_sample=False), extract the answer
Compare predicted answer to ground truth
Group by string length, compute per-length accuracy
Compute in-dist (lengths 1-40) vs OOD (41-100) mean accuracy
Report dm_score = mean OOD accuracy (matching DeepMind's np.mean(accuracies[seq_len+1:]) from example.py:138)
Answer extraction uses regex to find "the answer is <binary_string>" in the CCoT output, falling back to the last binary string in the generation. This handles both cases: the model generating a full CoT trace, or outputting just the answer.

Verdict: Evaluation is sound. Autoregressive generation with greedy decoding is the correct approach (no teacher forcing, no information leakage). The per-length stratification ensures fair comparison.

4. ISSUES AND CONCERNS I FOUND
4.1 Learning Rate Schedule Mismatch
Phase 1 and DeepMind both use constant LR. Phase 2's YAML uses lr_scheduler_type: linear (linearly decaying). This means the learning rate starts at 1e-3 and linearly decreases to 0 over training. This is a deviation from the paper and Phase 1. It could affect convergence behavior and RPE dynamics.

Severity: Medium. For a controlled experiment, using constant would be more faithful. However, since both baseline and RPE use the same schedule, the comparison between them is still valid — the absolute numbers just may differ from Phase 1.

4.2 Epoch-based vs Step-based Training
DeepMind and Phase 1 use step-based training (10,000 steps). Phase 2 uses num_train_epochs: 5 with 5000 training examples and batch_size 4 = 1250 steps/epoch = 6250 total steps. This is different from 10,000. Again, since both conditions share this, the comparison is valid, but the total training compute is different.

4.3 LoRA + Frozen RoPE Embeddings
This is the most interesting concern. Qwen2.5-7B's RoPE embeddings are computed dynamically (not learned weights) — they're sinusoidal functions of position IDs. So RPE works by feeding different position IDs into the same RoPE computation. The model's attention patterns are adjusted by LoRA adapters to work with the randomized positions it sees during training.

However, LoRA only modifies ~0.1% of parameters. The base model has 7B parameters of pretrained weights that "expect" sequential positions. The LoRA adapters must learn to override these expectations. This is a meaningful question: can a small LoRA delta overcome 7B parameters of positional inductive bias?

Verdict: This is actually fine because RoPE is a functional transform, not learned weights. The position IDs go into cos(position * theta) and sin(position * theta) which are computed on the fly. RPE changes the input to this function, not the function itself. The LoRA adapters then adjust the attention/MLP to handle the resulting attention patterns. The question is empirical — hence the experiment.

4.4 The RPE Callback Integration Point
The RPETrainerCallback in rpe_llamafactory_patch.py is defined but I don't see where it's registered with LLaMA-Factory's training loop. The run_experiment.sh uses RPE_CONFIG_PATH as an environment variable, but LLaMA-Factory won't automatically read this. Somewhere, LLaMA-Factory's tuner.py or equivalent must check for RPE_CONFIG_PATH and add the callback.

Let me verify this.

Grep "RPE_CONFIG_PATH|RPETrainerCallback|rpe_llamafactory" (in /Users/reach/CodingRepositories/02 TAUR Labs/RPE/composable_cot/LLaMA-Factory/src)
10 lines of output
The callback is registered in composable_cot/LLaMA-Factory/src/llamafactory/train/tuner.py:58-61. When RPE_CONFIG_PATH is set, it imports RPETrainerCallback and appends it to the callbacks list. This is correct.

5. FULL PIPELINE FLOW (End-to-End)
Here is exactly what happens when you run bash composable_cot/scripts/run_experiment.sh:


Step 0: Environment Setup
  └── Install LLaMA-Factory, peft, create output dirs

Step 1: Generate Data
  └── python generate_reverse_string_data.py
      ├── 5000 training examples (lengths 1-40, CCoT format with reasoning traces)
      ├── 500 validation examples (same range, CCoT format)
      ├── 400 in-dist test examples (10 per length, 1-40, answer-only)
      ├── 600 OOD test examples (10 per length, 41-100, answer-only)
      └── 1000 combined test examples

Step 2: Train Baseline
  └── llamafactory-cli train reverse_string_composable_cot.yaml
      ├── NO RPE_CONFIG_PATH set → no RPE callback
      ├── Loads Qwen2.5-7B → applies LoRA adapters
      ├── SFT on 5000 examples, 5 epochs, batch_size=4, LR=1e-3
      ├── Standard sequential positions [0, 1, ..., N-1]
      └── Saves best checkpoint by eval_loss

Step 3: Train RPE
  └── RPE_CONFIG_PATH=rpe_config.yaml llamafactory-cli train ...
      ├── RPE_CONFIG_PATH is set → RPETrainerCallback added
      ├── on_train_begin: RPEPatcher wraps model.forward
      ├── Every training forward pass: positions = sorted random from [0, 8192)
      ├── Eval within training: model.eval() → standard positions
      ├── on_train_end: unpatch model
      └── Saves best checkpoint

Step 4: Evaluate Both
  └── For each {baseline, rpe}:
      ├── Load Qwen2.5-7B + LoRA checkpoint
      ├── Merge LoRA into base model
      ├── For each test example (1000 total):
      │   ├── Feed instruction as prompt
      │   ├── Generate autoregressively (greedy)
      │   ├── Extract predicted answer
      │   └── Compare to ground truth
      ├── Compute per-length accuracy
      └── Compute in-dist/OOD/dm_score

Step 5: Generate Plots
  └── Length vs accuracy curves, bar charts, summary JSON
6. SUMMARY OF FINDINGS
What's correct:

The RPE algorithm is a faithful translation of DeepMind's JAX code to PyTorch
The monkey-patch approach correctly gates on model.training
The @functools.wraps fix preserves the function signature for HF internals
The data generation matches DeepMind's parameters (binary strings, lengths 1-40 train, 41-100 OOD)
The controlled experiment design (baseline vs RPE, only variable is position randomization) is sound
The evaluation (autoregressive, greedy, per-length accuracy, dm_score) matches the paper's methodology
The LLaMA-Factory integration via TrainerCallback + environment variable is clean
Minor concerns:

LR schedule is linear (decaying) instead of constant — both conditions use it, so comparison is still valid, but differs from DeepMind's setup
Epoch-based training gives ~6250 steps vs DeepMind's 10,000 — again, same for both conditions
The LoRA fine-tuning interaction with RPE is the key open question — pretrained positional expectations may resist modification through a small adapter
No bugs found in the current code. The training=True bug mentioned in the summary was already fixed. The inspect.signature bug from Phase 1 doesn't apply to Phase 2 because the callback unpatches before training ends, and evaluation loads a fresh model from checkpoint with merge_and_unload() (no wrapper present).