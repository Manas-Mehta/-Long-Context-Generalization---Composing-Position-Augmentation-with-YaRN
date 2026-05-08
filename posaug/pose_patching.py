"""Model patching module for PoSE.

Monkey-patches HuggingFace model forward methods to inject PoSE
(Positional Skip-wisE) position IDs. Same architecture as RPEPatcher
but uses 2-chunk skip positions instead of fully random positions.

Like RPE, PoSE is training-only: when model.training=True, positions
are modified; when model.training=False, standard sequential positions
are used.
"""

import functools
from typing import Any, Optional

import torch
from transformers import PreTrainedModel

from .pose import PositionalSkipWise


class PoSEPatcher:
    """Patches a HuggingFace model to use PoSE position IDs during training.

    Usage:
        patcher = PoSEPatcher(model, {"target_length": 131072})
        patcher.patch()
        outputs = model(input_ids)  # uses PoSE positions if model.training
        patcher.unpatch()

    Args:
        model: A HuggingFace PreTrainedModel.
        pose_config: Dict with keys: target_length (int), seed (int|None).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        pose_config: dict,
    ) -> None:
        self.model = model
        self.pose = PositionalSkipWise(**pose_config)
        self._original_forward: Optional[Any] = None
        self._patched = False

    def patch(self) -> None:
        """Replace model.forward with PoSE-wrapped version."""
        if self._patched:
            print("[PoSEPatcher] Already patched, skipping.")
            return

        self._original_forward = self.model.forward

        patcher = self
        original_forward = self._original_forward

        @functools.wraps(original_forward)
        def pose_forward(
            input_ids: torch.LongTensor | None = None,
            **kwargs: Any,
        ):
            if input_ids is not None:
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
            elif "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                batch_size, seq_length, _ = kwargs["inputs_embeds"].shape
                device = kwargs["inputs_embeds"].device
            else:
                return original_forward(input_ids=input_ids, **kwargs)

            position_ids = kwargs.get("position_ids", None)

            if position_ids is None:
                position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

            if patcher.model.training:
                pose_positions = torch.stack([
                    patcher.pose.get_pose_positions(seq_length, device=device)
                    for _ in range(batch_size)
                ])
                kwargs["position_ids"] = pose_positions
            else:
                kwargs["position_ids"] = position_ids

            return original_forward(input_ids=input_ids, **kwargs)

        self.model.forward = pose_forward
        self._patched = True
        print(f"[PoSEPatcher] Patched {type(self.model).__name__} "
              f"(target_length={self.pose.target_length}). "
              f"PoSE positions when model.training=True, sequential when False.")

    def unpatch(self) -> None:
        """Restore the original model.forward."""
        if not self._patched:
            print("[PoSEPatcher] Not patched, nothing to restore.")
            return

        self.model.forward = self._original_forward
        self._original_forward = None
        self._patched = False
        print(f"[PoSEPatcher] Unpatched {type(self.model).__name__} — original forward restored.")

    @property
    def is_patched(self) -> bool:
        return self._patched
