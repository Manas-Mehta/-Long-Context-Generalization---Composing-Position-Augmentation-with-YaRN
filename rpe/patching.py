"""Model patching module for RPE.

Monkey-patches HuggingFace model forward methods to inject randomized
position IDs before they reach the rotary embedding layer.

Patching strategy:
    We wrap the top-level CausalLM.forward(). When called, the wrapper
    intercepts `position_ids`, replaces them with RPE-generated ones,
    and delegates to the original forward. This works because Qwen2
    (and most HF models) pass position_ids straight through to the
    base model, which feeds them into rotary embeddings.
"""

import functools
from typing import Any, Optional

import torch
from transformers import PreTrainedModel

from .core import RandomizedPositionalEncoding


class RPEPatcher:
    """Patches a HuggingFace model to use randomized position IDs.

    Usage:
        patcher = RPEPatcher(model, {"max_simulation_length": 8192, "seed": 42})
        patcher.patch()      # model.forward now uses RPE positions
        outputs = model(input_ids)
        patcher.unpatch()    # restore original forward

    Args:
        model: A HuggingFace PreTrainedModel (e.g. Qwen2ForCausalLM).
        rpe_config: Dict passed to RandomizedPositionalEncoding.__init__.
            Keys: max_simulation_length (int), seed (int|None).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        rpe_config: dict,
    ) -> None:
        self.model = model
        self.rpe = RandomizedPositionalEncoding(**rpe_config)
        self._original_forward: Optional[Any] = None
        self._patched = False

    def patch(self) -> None:
        """Replace model.forward with RPE-wrapped version."""
        if self._patched:
            print("[RPEPatcher] Already patched, skipping.")
            return

        self._original_forward = self.model.forward

        # Capture self (the patcher) for the closure
        patcher = self
        original_forward = self._original_forward

        @functools.wraps(original_forward)
        def rpe_forward(
            input_ids: torch.LongTensor | None = None,
            **kwargs: Any,
        ):
            # Determine sequence length from input_ids or inputs_embeds
            if input_ids is not None:
                batch_size, seq_length = input_ids.shape
                device = input_ids.device
            elif "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                batch_size, seq_length, _ = kwargs["inputs_embeds"].shape
                device = kwargs["inputs_embeds"].device
            else:
                # Can't determine shape — fall through to original
                return original_forward(input_ids=input_ids, **kwargs)

            position_ids = kwargs.get("position_ids", None)

            # Build standard position_ids if none provided
            if position_ids is None:
                position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

            if patcher.model.training:
                # Generate randomized positions for each batch element
                randomized = torch.stack([
                    patcher.rpe.get_randomized_positions(seq_length, device=device)
                    for _ in range(batch_size)
                ])
                kwargs["position_ids"] = randomized
            else:
                kwargs["position_ids"] = position_ids

            return original_forward(input_ids=input_ids, **kwargs)

        self.model.forward = rpe_forward
        self._patched = True
        print(f"[RPEPatcher] Patched {type(self.model).__name__} "
              f"(L={self.rpe.max_simulation_length}). "
              f"Randomizes when model.training=True, passes through when False.")

    def unpatch(self) -> None:
        """Restore the original model.forward."""
        if not self._patched:
            print("[RPEPatcher] Not patched, nothing to restore.")
            return

        self.model.forward = self._original_forward
        self._original_forward = None
        self._patched = False
        print(f"[RPEPatcher] Unpatched {type(self.model).__name__} — original forward restored.")

    @property
    def is_patched(self) -> bool:
        return self._patched


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("RPEPatcher Integration Test with Qwen 2.5 1.5B")
    print("=" * 60)

    # --- Load model and tokenizer ---
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
    model.eval()
    print(f"Loaded {type(model).__name__} ({model.config.hidden_size}d, {model.config.num_hidden_layers}L)")

    # --- Prepare dummy input ---
    text = "The quick brown fox jumps over"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"\nInput: \"{text}\"")
    print(f"Token IDs: {input_ids[0].tolist()} (len={seq_len})")

    # --- Test 1: Baseline (unpatched) ---
    print("\n" + "-" * 40)
    print("[Test 1] Baseline forward (no patch)")
    print("-" * 40)
    with torch.no_grad():
        baseline_out = model(input_ids=input_ids)
    baseline_logits = baseline_out.logits
    print(f"Output logits shape: {baseline_logits.shape}")
    print(f"Logits sample (last token, first 5 vocab): {baseline_logits[0, -1, :5].tolist()}")

    # --- Test 2: Patched, model.train() (should randomize) ---
    print("\n" + "-" * 40)
    print("[Test 2] Patched forward (model.train() — should randomize)")
    print("-" * 40)
    rpe_config = {"max_simulation_length": 8192, "seed": 42}
    patcher = RPEPatcher(model, rpe_config)
    patcher.patch()

    model.train()
    with torch.no_grad():
        patched_out = model(input_ids=input_ids)
    patched_logits = patched_out.logits
    print(f"Output logits shape: {patched_logits.shape}")
    print(f"Logits sample (last token, first 5 vocab): {patched_logits[0, -1, :5].tolist()}")

    # Logits should differ from baseline because positions changed
    logits_differ = not torch.allclose(baseline_logits, patched_logits, atol=1e-3)
    print(f"Logits differ from baseline: {logits_differ}")

    # --- Test 3: Patched, model.eval() (should pass through) ---
    print("\n" + "-" * 40)
    print("[Test 3] Patched forward (model.eval() — should pass through)")
    print("-" * 40)
    model.eval()

    with torch.no_grad():
        eval_out = model(input_ids=input_ids)
    eval_logits = eval_out.logits
    print(f"Output logits shape: {eval_logits.shape}")

    # Eval mode should produce same logits as baseline (standard positions)
    logits_match = torch.allclose(baseline_logits, eval_logits, atol=1e-3)
    print(f"Logits match baseline (eval passthrough): {logits_match}")

    patcher.unpatch()

    # --- Test 4: Unpatch restores original ---
    print("\n" + "-" * 40)
    print("[Test 4] Unpatch restores original forward")
    print("-" * 40)
    with torch.no_grad():
        restored_out = model(input_ids=input_ids)
    restored_logits = restored_out.logits
    logits_restored = torch.allclose(baseline_logits, restored_logits, atol=1e-3)
    print(f"Logits match baseline after unpatch: {logits_restored}")

    # --- Test 5: Double patch/unpatch safety ---
    print("\n" + "-" * 40)
    print("[Test 5] Double patch/unpatch safety")
    print("-" * 40)
    patcher2 = RPEPatcher(model, rpe_config)
    patcher2.patch()
    patcher2.patch()  # Should print "Already patched"
    patcher2.unpatch()
    patcher2.unpatch()  # Should print "Not patched"

    # --- Summary ---
    print("\n" + "=" * 60)
    results = {
        "Shapes preserved": baseline_logits.shape == patched_logits.shape,
        "RPE changes logits": logits_differ,
        "Eval mode passthrough": logits_match,
        "Unpatch restores original": logits_restored,
    }
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    print("=" * 60)
    print(f"{'All tests passed!' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 60)
