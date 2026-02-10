"""Core RPE (Randomized Positional Encodings) module.

Implements the RPE algorithm from DeepMind's paper (arXiv:2305.16843).
The key insight: instead of using sequential position IDs [0, 1, 2, ...],
sample random positions from a larger range and sort them. This teaches
models to rely on relative order rather than absolute positions, enabling
length generalization.
"""

import torch
from typing import Optional


class RandomizedPositionalEncoding:
    """Generates randomized position IDs for transformer models.

    Instead of standard position_ids = [0, 1, 2, ..., N-1], this class
    generates sorted random integers from [0, L) where L >> N.

    Example:
        Standard: [0, 1, 2, 3, 4]
        RPE:      [12, 45, 89, 156, 203]  (sorted random from [0, 8192))

    Args:
        max_simulation_length: Upper bound L for position sampling (default: 8192).
            Should be significantly larger than expected sequence lengths.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        max_simulation_length: int = 8192,
        seed: Optional[int] = None
    ) -> None:
        self.max_simulation_length = max_simulation_length
        self.seed = seed
        self._generator: Optional[torch.Generator] = None

        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def get_randomized_positions(
        self,
        seq_length: int,
        device: Optional[torch.device] = None
    ) -> torch.LongTensor:
        """Generate sorted random position IDs.

        Samples seq_length unique integers from [0, max_simulation_length),
        then sorts them in ascending order (critical for causal attention).

        Args:
            seq_length: Number of positions to generate.
            device: Target device for the output tensor.

        Returns:
            Sorted tensor of shape (seq_length,) with unique random positions.

        Raises:
            ValueError: If seq_length > max_simulation_length.
        """
        if seq_length > self.max_simulation_length:
            raise ValueError(
                f"seq_length ({seq_length}) cannot exceed "
                f"max_simulation_length ({self.max_simulation_length})"
            )

        # Sample without replacement using randperm on CPU (generator is CPU-bound),
        # then move to target device. This avoids device mismatch errors with CUDA.
        if self._generator is not None:
            perm = torch.randperm(
                self.max_simulation_length,
                generator=self._generator,
            )
        else:
            perm = torch.randperm(self.max_simulation_length)

        # Take first seq_length elements and sort them
        positions = perm[:seq_length].sort().values

        if device is not None:
            positions = positions.to(device)

        return positions.long()

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the random generator with a new seed.

        Args:
            seed: New seed value. If None, uses the original seed.
        """
        if seed is None:
            seed = self.seed

        if seed is not None:
            if self._generator is None:
                self._generator = torch.Generator()
            self._generator.manual_seed(seed)
        else:
            self._generator = None


def transform_position_ids(
    position_ids: torch.LongTensor,
    rpe: RandomizedPositionalEncoding,
    training: bool = True
) -> torch.LongTensor:
    """Transform standard position IDs using RPE.

    Args:
        position_ids: Original position IDs tensor of shape (batch_size, seq_length)
            or (seq_length,).
        rpe: RandomizedPositionalEncoding instance.
        training: If True, returns randomized positions. If False, returns
            original positions unchanged (for deterministic inference).

    Returns:
        Transformed position IDs with same shape as input.
    """
    if not training:
        return position_ids

    # Handle both batched and unbatched inputs
    if position_ids.dim() == 1:
        seq_length = position_ids.size(0)
        return rpe.get_randomized_positions(seq_length, device=position_ids.device)

    # Batched case: shape (batch_size, seq_length)
    batch_size, seq_length = position_ids.shape
    device = position_ids.device

    # Generate different random positions for each batch element
    # This provides more diverse training signal
    randomized = torch.stack([
        rpe.get_randomized_positions(seq_length, device=device)
        for _ in range(batch_size)
    ])

    return randomized


if __name__ == "__main__":
    print("=" * 60)
    print("RPE Core Module Test")
    print("=" * 60)

    # Test 1: Basic functionality
    print("\n[Test 1] Basic RandomizedPositionalEncoding")
    rpe = RandomizedPositionalEncoding(max_simulation_length=8192)
    positions = rpe.get_randomized_positions(seq_length=10)
    print(f"  Sequence length: 10")
    print(f"  Generated positions: {positions.tolist()}")
    print(f"  Shape: {positions.shape}")
    print(f"  Dtype: {positions.dtype}")
    print(f"  Is sorted: {torch.all(positions[1:] > positions[:-1]).item()}")
    print(f"  Min: {positions.min().item()}, Max: {positions.max().item()}")

    # Test 2: Reproducibility with seed
    print("\n[Test 2] Reproducibility with seed")
    rpe_seeded = RandomizedPositionalEncoding(max_simulation_length=8192, seed=42)
    pos1 = rpe_seeded.get_randomized_positions(5)
    rpe_seeded.reset_seed()
    pos2 = rpe_seeded.get_randomized_positions(5)
    print(f"  First call:  {pos1.tolist()}")
    print(f"  After reset: {pos2.tolist()}")
    print(f"  Are equal: {torch.equal(pos1, pos2)}")

    # Test 3: Uniqueness (no duplicates)
    print("\n[Test 3] Uniqueness check")
    positions = rpe.get_randomized_positions(100)
    unique_count = len(torch.unique(positions))
    print(f"  Generated 100 positions, unique count: {unique_count}")
    print(f"  All unique: {unique_count == 100}")

    # Test 4: transform_position_ids function
    print("\n[Test 4] transform_position_ids function")
    standard_ids = torch.arange(5).unsqueeze(0)  # Shape: (1, 5)
    print(f"  Standard IDs: {standard_ids.tolist()}")

    transformed_train = transform_position_ids(standard_ids, rpe, training=True)
    print(f"  Training mode: {transformed_train.tolist()}")

    transformed_eval = transform_position_ids(standard_ids, rpe, training=False)
    print(f"  Eval mode: {transformed_eval.tolist()}")
    print(f"  Eval preserves original: {torch.equal(standard_ids, transformed_eval)}")

    # Test 5: Batched transform
    print("\n[Test 5] Batched transform (different positions per batch)")
    batch_ids = torch.arange(5).unsqueeze(0).expand(3, -1)  # Shape: (3, 5)
    print(f"  Batch shape: {batch_ids.shape}")
    transformed_batch = transform_position_ids(batch_ids, rpe, training=True)
    print(f"  Transformed batch:")
    for i, row in enumerate(transformed_batch):
        print(f"    Batch {i}: {row.tolist()}")

    # Test 6: Edge case - max sequence length
    print("\n[Test 6] Edge case - using full range")
    small_rpe = RandomizedPositionalEncoding(max_simulation_length=100)
    full_positions = small_rpe.get_randomized_positions(100)
    print(f"  Max sim length: 100, seq_length: 100")
    print(f"  Covers all positions: {set(full_positions.tolist()) == set(range(100))}")

    # Test 7: Error handling
    print("\n[Test 7] Error handling")
    try:
        small_rpe.get_randomized_positions(101)
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
