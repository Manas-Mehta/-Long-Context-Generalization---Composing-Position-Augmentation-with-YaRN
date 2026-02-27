"""PoSE (Positional Skip-wisE) module.

Implements PoSE from "Efficient Context Window Extension of LLMs via
Positional Skip-wise Training" (Zhu et al., ICLR 2024, arXiv:2309.10400).

Key idea: split the training sequence into 2 contiguous chunks, then insert
a random gap between them. Each chunk has sequential position IDs internally,
preserving the local structure the model learned during pretraining. The gap
simulates long-range relative positions without needing long sequences.

Contrast with RPE (fully random positions):
  RPE:  [12, 45, 89, 156, 203]     -- gaps between every token
  PoSE: [0, 1, 2, ..., 511, 6560, 6561, ..., 8095]  -- two contiguous blocks

PoSE preserves within-chunk continuity, which is why it retains language
modeling ability better than RPE (see paper Table 1).
"""

import random
from typing import Optional

import torch


class PositionalSkipWise:
    """Generates PoSE position IDs for transformer models.

    Splits a sequence into 2 chunks with a random skip between them.
    Chunk 1 starts at position 0, chunk 2 starts at a random offset
    in [0, target_length - seq_len].

    Args:
        target_length: Upper bound L_t for the extended context window.
            Positions in chunk 2 can go up to target_length - 1.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        target_length: int = 32768,
        seed: Optional[int] = None,
    ) -> None:
        self.target_length = target_length
        self.seed = seed
        self._rng = random.Random(seed)

    def get_pose_positions(
        self,
        seq_length: int,
        device: Optional[torch.device] = None,
    ) -> torch.LongTensor:
        """Generate PoSE position IDs for a single sequence.

        Algorithm (2-chunk, following the paper):
          1. Pick random split point rt1 in [1, seq_length // 2]
          2. Chunk 1: positions [0, 1, ..., rt1 - 1]
          3. Pick random skip rt in [0, target_length - seq_length]
          4. Chunk 2: positions [rt, rt+1, ..., rt + (seq_length - rt1) - 1]

        The result has exactly seq_length positions, all unique, monotonically
        increasing within each chunk. There is one gap (the "skip") between
        the two chunks.

        Args:
            seq_length: Number of positions to generate.
            device: Target device for the output tensor.

        Returns:
            Tensor of shape (seq_length,) with PoSE position IDs.
        """
        effective_target = max(self.target_length, seq_length)

        # Random split: chunk 1 gets [0, rt1) tokens
        rt1 = self._rng.randint(1, max(1, (seq_length + 1) // 2))

        # Chunk 1 always starts at position 0
        # Chunk 2 starts at a random offset
        max_skip = effective_target - seq_length
        rt = self._rng.randint(0, max(0, max_skip))

        pos_ids = torch.arange(seq_length, dtype=torch.long)
        # Chunk 1: positions [0, 1, ..., rt1-1] — already correct
        # Chunk 2: shift by rt (adds the skip gap)
        pos_ids[rt1:] += rt

        if device is not None:
            pos_ids = pos_ids.to(device)

        return pos_ids

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the random generator."""
        if seed is None:
            seed = self.seed
        self._rng = random.Random(seed)
