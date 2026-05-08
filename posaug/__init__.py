"""RPE (Randomized Positional Encodings) package.

Implements RPE from DeepMind's paper (arXiv:2305.16843) for decoder-only models.
Also includes PoSE (Positional Skip-wisE) from Zhu et al. (ICLR 2024, arXiv:2309.10400).
"""

from .core import RandomizedPositionalEncoding, transform_position_ids
from .patching import RPEPatcher
from .pose import PositionalSkipWise
from .pose_patching import PoSEPatcher

__all__ = [
    "RandomizedPositionalEncoding", "transform_position_ids", "RPEPatcher",
    "PositionalSkipWise", "PoSEPatcher",
]
