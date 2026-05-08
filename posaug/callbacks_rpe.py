"""RPE integration for LLaMA-Factory training.

Provides functions to apply/remove RPE patching on a model loaded by
LLaMA-Factory's `load_model()`.  Designed to be called from a
TrainerCallback or directly in a workflow file.

Usage (standalone):
    from rpe_llamafactory_patch import apply_rpe_patch, remove_rpe_patch, load_rpe_config

    rpe_config = load_rpe_config("composable_cot/scripts/rpe_config.yaml")
    patcher = apply_rpe_patch(model, rpe_config)
    # ... training ...
    remove_rpe_patch(patcher)

Usage (as TrainerCallback):
    from rpe_llamafactory_patch import RPETrainerCallback
    callbacks.append(RPETrainerCallback("composable_cot/scripts/rpe_config.yaml"))
"""

import os
import sys
from typing import Any, Optional

import yaml

# Ensure the RPE package is importable regardless of working directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import PreTrainedModel, TrainerCallback
from typing_extensions import override

from rpe.patching import RPEPatcher


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_rpe_config(config_path: str) -> dict:
    """Load RPE configuration from a YAML file.

    Args:
        config_path: Path to rpe_config.yaml.

    Returns:
        Dict with keys: enabled, max_simulation_length, seed,
        training_mode, inference_mode, and optionally curriculum.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rpe_cfg = cfg.get("rpe", cfg)  # support nested or flat
    result = {
        "enabled": rpe_cfg.get("enabled", False),
        "max_simulation_length": rpe_cfg.get("max_simulation_length", 8192),
        "seed": rpe_cfg.get("seed", None),
        "training_mode": rpe_cfg.get("training_mode", True),
        "inference_mode": rpe_cfg.get("inference_mode", False),
    }

    # Curriculum learning: epoch-to-L schedule
    # Format in YAML: curriculum: {1: 256, 2: 512, 3: 768, 4: 1024, 5: 1024}
    curriculum = rpe_cfg.get("curriculum", None)
    if curriculum is not None:
        # Convert keys to int (YAML may parse them as int already, but be safe)
        result["curriculum"] = {int(k): int(v) for k, v in curriculum.items()}

    return result


# ---------------------------------------------------------------------------
# Patch / unpatch functions
# ---------------------------------------------------------------------------

def apply_rpe_patch(
    model: PreTrainedModel,
    rpe_config: dict,
) -> Optional[RPEPatcher]:
    """Apply RPE patching to a model if enabled in config.

    RPEPatcher automatically gates on model.training: randomized positions
    when model.training=True, standard positions when model.training=False.

    Args:
        model: HuggingFace PreTrainedModel.
        rpe_config: Dict from load_rpe_config().

    Returns:
        RPEPatcher instance (for later unpatching), or None if RPE is disabled.
    """
    if not rpe_config.get("enabled", False):
        print("[RPE] Disabled in config — skipping patch.")
        return None

    patcher_config = {
        "max_simulation_length": rpe_config["max_simulation_length"],
    }
    seed = rpe_config.get("seed")
    if seed is not None:
        patcher_config["seed"] = seed

    patcher = RPEPatcher(model, patcher_config)
    patcher.patch()
    return patcher


def remove_rpe_patch(patcher: Optional[RPEPatcher]) -> None:
    """Remove RPE patching from model.

    Args:
        patcher: RPEPatcher instance returned by apply_rpe_patch(), or None.
    """
    if patcher is not None:
        patcher.unpatch()


# ---------------------------------------------------------------------------
# TrainerCallback — single integration point for all training stages
# ---------------------------------------------------------------------------

class RPETrainerCallback(TrainerCallback):
    """HuggingFace TrainerCallback that applies RPE at the start of training.

    This is the recommended integration approach because it:
      - Works with ALL training stages (pt, sft, rm, dpo, kto, ppo)
      - Requires only ONE insertion point in tuner.py
      - Follows standard HuggingFace patterns
      - Works with distributed training

    Usage in tuner.py:
        callbacks.append(RPETrainerCallback("/path/to/rpe_config.yaml"))
    """

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config_path = config_path
        self.rpe_config = load_rpe_config(config_path)
        self._patcher: Optional[RPEPatcher] = None
        self._curriculum = self.rpe_config.get("curriculum", None)
        self._current_epoch = 0

    @override
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Apply RPE patch when training starts."""
        if model is not None:
            # If curriculum is set, start with epoch 1's L value
            if self._curriculum:
                initial_L = self._curriculum.get(1, self.rpe_config["max_simulation_length"])
                self.rpe_config["max_simulation_length"] = initial_L
                print(f"[RPE Curriculum] Starting with L={initial_L} for epoch 1")
            self._patcher = apply_rpe_patch(model, self.rpe_config)

    @override
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Update L value for curriculum learning at each epoch boundary."""
        if self._curriculum and self._patcher is not None:
            epoch = int(state.epoch) + 1 if state.epoch is not None else 1
            if epoch != self._current_epoch:
                self._current_epoch = epoch
                new_L = self._curriculum.get(epoch, self._curriculum.get(
                    max(k for k in self._curriculum if k <= epoch),
                    self.rpe_config["max_simulation_length"]
                ))
                old_L = self._patcher.rpe.max_simulation_length
                if new_L != old_L:
                    self._patcher.rpe.max_simulation_length = new_L
                    print(f"[RPE Curriculum] Epoch {epoch}: L changed from {old_L} to {new_L}")

    @override
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Remove RPE patch when training ends (clean state for saving)."""
        remove_rpe_patch(self._patcher)
        self._patcher = None

    @override
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Optionally apply RPE during evaluation."""
        if model is not None and self.rpe_config.get("inference_mode", False):
            pass
