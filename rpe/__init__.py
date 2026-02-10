"""RPE (Randomized Positional Encodings) package.

Implements RPE from DeepMind's paper (arXiv:2305.16843) for decoder-only models.
"""

from .core import RandomizedPositionalEncoding, transform_position_ids
from .patching import RPEPatcher

__all__ = ["RandomizedPositionalEncoding", "transform_position_ids", "RPEPatcher"]
