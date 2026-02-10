"""Reverse string task for evaluating length generalization.

Reproduces the evaluation from DeepMind's RPE paper (arXiv:2305.16843).
Task: given "reverse: abcde", the model should output "edcba".

This is a pipeline validation task. Without fine-tuning on reversal,
a pretrained LLM will perform poorly — the goal is to verify that:
  1. The evaluation harness works end-to-end
  2. RPE-patched and baseline models produce different outputs
  3. The infrastructure is ready for training experiments
"""

import random
import string
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class ReversalExample:
    """A single reverse-string example."""
    input_str: str      # The letters to reverse (e.g. "abcde")
    expected: str       # The reversed output (e.g. "edcba")
    prompt: str         # Full prompt sent to model (e.g. "reverse: abcde\n")
    length: int         # Length of input_str


class ReverseStringTask:
    """Generates and evaluates reverse-string examples.

    Args:
        seed: Random seed for reproducible example generation.
    """

    PROMPT_TEMPLATE = "reverse: {}\n"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def generate_example(self, length: int) -> ReversalExample:
        """Create a single reverse-string example.

        Args:
            length: Number of characters in the string to reverse.

        Returns:
            A ReversalExample with input, expected output, and full prompt.
        """
        letters = [self.rng.choice(string.ascii_lowercase) for _ in range(length)]
        input_str = "".join(letters)
        expected = input_str[::-1]
        prompt = self.PROMPT_TEMPLATE.format(input_str)
        return ReversalExample(
            input_str=input_str,
            expected=expected,
            prompt=prompt,
            length=length,
        )

    def generate_dataset(
        self,
        min_len: int,
        max_len: int,
        num_per_length: int = 10,
    ) -> list[ReversalExample]:
        """Generate examples across a range of lengths.

        Args:
            min_len: Minimum string length (inclusive).
            max_len: Maximum string length (inclusive).
            num_per_length: Number of examples per length.

        Returns:
            List of ReversalExample, grouped by length.
        """
        examples = []
        for length in range(min_len, max_len + 1):
            for _ in range(num_per_length):
                examples.append(self.generate_example(length))
        return examples

    @torch.no_grad()
    def evaluate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        examples: list[ReversalExample],
        max_new_tokens: int | None = None,
    ) -> dict:
        """Evaluate model on reverse-string examples.

        Uses greedy decoding. Compares the model's generated text (stripped)
        to the expected reversed string.

        Args:
            model: HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
            examples: List of ReversalExample to evaluate.
            max_new_tokens: Max tokens to generate. Defaults to 2x the input
                length (generous budget for BPE overhead).

        Returns:
            Dict with per-length accuracy and example details:
            {
                "per_length": {5: {"correct": 3, "total": 10, "accuracy": 0.3}, ...},
                "overall_accuracy": float,
                "details": [{"prompt": ..., "expected": ..., "generated": ..., "correct": bool}, ...],
            }
        """
        per_length: dict[int, dict] = {}
        details = []
        total_correct = 0
        total = 0

        for ex in examples:
            budget = max_new_tokens or (ex.length * 2 + 5)

            input_ids = tokenizer.encode(ex.prompt, return_tensors="pt")
            input_ids = input_ids.to(next(model.parameters()).device)
            prompt_len = input_ids.shape[1]

            output_ids = model.generate(
                input_ids,
                max_new_tokens=budget,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.eos_token_id,
            )

            # Decode only the generated portion (after prompt)
            generated_ids = output_ids[0, prompt_len:]
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Compare raw strings (strip whitespace / newlines)
            correct = generated == ex.expected

            if ex.length not in per_length:
                per_length[ex.length] = {"correct": 0, "total": 0}
            per_length[ex.length]["total"] += 1
            if correct:
                per_length[ex.length]["correct"] += 1
                total_correct += 1
            total += 1

            details.append({
                "length": ex.length,
                "input": ex.input_str,
                "expected": ex.expected,
                "generated": generated[:ex.length + 20],  # Truncate for display
                "correct": correct,
            })

        # Compute per-length accuracy
        for length in per_length:
            d = per_length[length]
            d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

        return {
            "per_length": dict(sorted(per_length.items())),
            "overall_accuracy": total_correct / total if total > 0 else 0.0,
            "details": details,
        }
