"""PoSE integration for training.

Provides PoSETrainerCallback that applies PoSE position manipulation
during training. Mirrors the RPETrainerCallback interface exactly.

Usage:
    from pose_patch import PoSETrainerCallback
    callbacks.append(PoSETrainerCallback("path/to/pose_config.yaml"))
"""

import os
import sys
from typing import Optional

import yaml

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import TrainerCallback
from typing_extensions import override

from posaug.pose_patching import PoSEPatcher


def load_pose_config(config_path: str) -> dict:
    """Load PoSE configuration from a YAML file.

    Args:
        config_path: Path to pose_config.yaml.

    Returns:
        Dict with keys: enabled, target_length, seed,
        and optionally curriculum.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pose_cfg = cfg.get("pose", cfg)
    result = {
        "enabled": pose_cfg.get("enabled", False),
        "target_length": pose_cfg.get("target_length", 32768),
        "seed": pose_cfg.get("seed", None),
    }

    # Curriculum: epoch-to-target_length schedule
    curriculum = pose_cfg.get("curriculum", None)
    if curriculum is not None:
        result["curriculum"] = {int(k): int(v) for k, v in curriculum.items()}

    return result


class PoSETrainerCallback(TrainerCallback):
    """HuggingFace TrainerCallback that applies PoSE during training.

    Mirrors RPETrainerCallback interface:
      - on_train_begin: patches model.forward with PoSE positions
      - on_epoch_begin: updates target_length for curriculum learning
      - on_train_end: unpatches (clean state for saving)

    Usage:
        callbacks.append(PoSETrainerCallback("/path/to/pose_config.yaml"))
    """

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config_path = config_path
        self.pose_config = load_pose_config(config_path)
        self._patcher: Optional[PoSEPatcher] = None
        self._curriculum = self.pose_config.get("curriculum", None)
        self._current_epoch = 0

    @override
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Apply PoSE patch when training starts."""
        if model is not None and self.pose_config.get("enabled", False):
            target_length = self.pose_config["target_length"]

            if self._curriculum:
                target_length = self._curriculum.get(1, target_length)
                print(f"[PoSE Curriculum] Starting with target_length={target_length} for epoch 1")

            patcher_config = {"target_length": target_length}
            seed = self.pose_config.get("seed")
            if seed is not None:
                patcher_config["seed"] = seed

            self._patcher = PoSEPatcher(model, patcher_config)
            self._patcher.patch()

    @override
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Update target_length for curriculum learning."""
        if self._curriculum and self._patcher is not None:
            epoch = int(state.epoch) + 1 if state.epoch is not None else 1
            if epoch != self._current_epoch:
                self._current_epoch = epoch
                new_L = self._curriculum.get(epoch, self._curriculum.get(
                    max(k for k in self._curriculum if k <= epoch),
                    self.pose_config["target_length"]
                ))
                old_L = self._patcher.pose.target_length
                if new_L != old_L:
                    self._patcher.pose.target_length = new_L
                    print(f"[PoSE Curriculum] Epoch {epoch}: target_length changed from {old_L} to {new_L}")

    @override
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Remove PoSE patch when training ends."""
        if self._patcher is not None:
            self._patcher.unpatch()
            self._patcher = None
