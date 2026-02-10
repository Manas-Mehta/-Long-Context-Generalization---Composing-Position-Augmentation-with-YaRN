"""HuggingFace-compatible dataset for the reverse string task.

Mirrors the DeepMind setup:
  - Binary alphabet (vocab_size=2): strings of '0' and '1'
  - Uniform length curriculum: sample lengths from [1, max_train_length]
  - Sequence-to-sequence format for causal LM fine-tuning

DeepMind reference:
  randomized_positional_encodings/tasks/dcf/reverse_string.py
  randomized_positional_encodings/experiments/example.py
"""

import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ReverseStringDataset(Dataset):
    """Generates binary string reversal examples for causal LM training.

    Each example is formatted as:
        input:  "reverse: 01101\n10110"
        labels: [-100, ..., -100, tok(1), tok(0), tok(1), tok(1), tok(0)]

    The -100 mask ensures loss is only computed on the output (reversed) tokens,
    matching the DeepMind setup where loss is computed on output positions only.

    Args:
        tokenizer: HuggingFace tokenizer.
        num_examples: Total number of examples to generate.
        min_length: Minimum string length (inclusive).
        max_length: Maximum string length (inclusive). DeepMind default: 40.
        seed: Random seed for reproducibility.
    """

    PROMPT_PREFIX = "reverse: "
    SEPARATOR = "\n"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examples: int,
        min_length: int = 1,
        max_length: int = 40,
        seed: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.min_length = min_length
        self.max_length = max_length

        self.rng = random.Random(seed)
        self.examples = [self._generate_example() for _ in range(num_examples)]

    def _generate_example(self) -> dict:
        """Generate a single binary string reversal example."""
        # Uniform curriculum: sample length uniformly from [min_length, max_length]
        # Matches DeepMind: curriculum_lib.UniformCurriculum(values=list(range(1, 41)))
        length = self.rng.randint(self.min_length, self.max_length)

        # Binary alphabet (vocab_size=2), matching DeepMind's reverse_string.ReverseString(vocab_size=2)
        input_str = "".join(self.rng.choice("01") for _ in range(length))
        output_str = input_str[::-1]

        # Format: "reverse: 01101\n10110"
        prompt = f"{self.PROMPT_PREFIX}{input_str}{self.SEPARATOR}"
        full_text = f"{prompt}{output_str}"

        # Tokenize prompt and full text separately to find the boundary
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # Add EOS token
        if self.tokenizer.eos_token_id is not None:
            full_ids.append(self.tokenizer.eos_token_id)

        # Labels: mask prompt tokens with -100, keep output tokens
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "length": length,
            "input_str": input_str,
            "output_str": output_str,
        }

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


class ReverseStringCollator:
    """Pads batches of variable-length reverse string examples.

    Left-pads input_ids with pad_token_id and labels with -100.
    Right-padding would also work but left-padding is standard for causal LMs.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, padding_side: str = "right"):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.padding_side = padding_side

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].size(0) for f in features)

        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for f in features:
            seq_len = f["input_ids"].size(0)
            pad_len = max_len - seq_len

            if self.padding_side == "right":
                input_ids = torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                labels = torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
                attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            else:
                input_ids = torch.cat([torch.full((pad_len,), self.pad_token_id, dtype=torch.long), f["input_ids"]])
                labels = torch.cat([torch.full((pad_len,), -100, dtype=torch.long), f["labels"]])
                attention_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(seq_len, dtype=torch.long)])

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_mask_batch.append(attention_mask)

        return {
            "input_ids": torch.stack(input_ids_batch),
            "labels": torch.stack(labels_batch),
            "attention_mask": torch.stack(attention_mask_batch),
        }
