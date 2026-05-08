"""Comprehensive sanity checks for the RPE module.

Run all tests:       pytest rpe/tests/test_rpe.py -v
Skip slow tests:     pytest rpe/tests/test_rpe.py -v -m "not slow"
Only slow tests:     pytest rpe/tests/test_rpe.py -v -m slow
"""

import pytest
import torch

from rpe.core import RandomizedPositionalEncoding, transform_position_ids
from rpe.patching import RPEPatcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


@pytest.fixture(scope="module")
def model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.float16)
    m.eval()
    return m


@pytest.fixture
def rpe():
    return RandomizedPositionalEncoding(max_simulation_length=8192)


@pytest.fixture
def rpe_seeded():
    return RandomizedPositionalEncoding(max_simulation_length=8192, seed=42)


@pytest.fixture(scope="module")
def dummy_input(tokenizer):
    return tokenizer("The quick brown fox", return_tensors="pt")["input_ids"]


# ---------------------------------------------------------------------------
# 1. Position ID tests
# ---------------------------------------------------------------------------

class TestPositionIDs:
    def test_positions_are_sorted(self, rpe):
        positions = rpe.get_randomized_positions(50)
        assert torch.all(positions[1:] > positions[:-1]).item(), \
            "Positions must be strictly ascending (causal requirement)"

    def test_positions_are_unique(self, rpe):
        positions = rpe.get_randomized_positions(200)
        assert len(torch.unique(positions)) == 200, \
            "All positions must be unique (sampled without replacement)"

    def test_positions_in_range(self, rpe):
        positions = rpe.get_randomized_positions(100)
        assert positions.min().item() >= 0, "Positions must be >= 0"
        assert positions.max().item() < 8192, "Positions must be < max_simulation_length"

    def test_positions_correct_length(self, rpe):
        for length in [1, 5, 64, 512]:
            positions = rpe.get_randomized_positions(length)
            assert positions.shape == (length,), \
                f"Expected shape ({length},), got {positions.shape}"

    def test_positions_dtype(self, rpe):
        positions = rpe.get_randomized_positions(10)
        assert positions.dtype == torch.int64, \
            f"Expected torch.int64, got {positions.dtype}"

    def test_seq_length_exceeds_max_raises(self):
        small_rpe = RandomizedPositionalEncoding(max_simulation_length=100)
        with pytest.raises(ValueError, match="cannot exceed"):
            small_rpe.get_randomized_positions(101)

    def test_seq_length_equals_max(self):
        small_rpe = RandomizedPositionalEncoding(max_simulation_length=100)
        positions = small_rpe.get_randomized_positions(100)
        assert set(positions.tolist()) == set(range(100))


# ---------------------------------------------------------------------------
# 2. Tensor shape tests
# ---------------------------------------------------------------------------

class TestTensorShapes:
    @pytest.mark.slow
    def test_output_shape_unchanged(self, model, dummy_input):
        """Model output shape must be identical with and without RPE."""
        model.eval()
        with torch.no_grad():
            baseline = model(input_ids=dummy_input)

        patcher = RPEPatcher(model, {"max_simulation_length": 8192})
        patcher.patch()
        try:
            model.train()
            with torch.no_grad():
                patched = model(input_ids=dummy_input)
            assert baseline.logits.shape == patched.logits.shape, \
                f"Shape mismatch: {baseline.logits.shape} vs {patched.logits.shape}"
        finally:
            patcher.unpatch()
            model.eval()

    @pytest.mark.slow
    def test_batch_dimension_preserved(self, model, tokenizer):
        """Batch size must be preserved through RPE patching."""
        texts = ["Hello world", "Foo bar baz", "Testing RPE"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)

        patcher = RPEPatcher(model, {"max_simulation_length": 8192})
        patcher.patch()
        try:
            model.train()
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"])
            assert out.logits.shape[0] == 3, \
                f"Expected batch_size=3, got {out.logits.shape[0]}"
        finally:
            patcher.unpatch()
            model.eval()


# ---------------------------------------------------------------------------
# 3. Numerical stability tests
# ---------------------------------------------------------------------------

class TestNumericalStability:
    @pytest.mark.slow
    def test_no_nan_in_output(self, model, dummy_input):
        patcher = RPEPatcher(model, {"max_simulation_length": 8192})
        patcher.patch()
        try:
            model.train()
            with torch.no_grad():
                out = model(input_ids=dummy_input)
            assert not torch.any(torch.isnan(out.logits)).item(), \
                "NaN detected in model output with RPE"
        finally:
            patcher.unpatch()
            model.eval()

    @pytest.mark.slow
    def test_no_inf_in_output(self, model, dummy_input):
        patcher = RPEPatcher(model, {"max_simulation_length": 8192})
        patcher.patch()
        try:
            model.train()
            with torch.no_grad():
                out = model(input_ids=dummy_input)
            assert not torch.any(torch.isinf(out.logits)).item(), \
                "Inf detected in model output with RPE"
        finally:
            patcher.unpatch()
            model.eval()


# ---------------------------------------------------------------------------
# 4. Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_positions(self):
        rpe1 = RandomizedPositionalEncoding(max_simulation_length=8192, seed=123)
        rpe2 = RandomizedPositionalEncoding(max_simulation_length=8192, seed=123)
        pos1 = rpe1.get_randomized_positions(20)
        pos2 = rpe2.get_randomized_positions(20)
        assert torch.equal(pos1, pos2), "Same seed must produce identical positions"

    def test_seed_reset_reproduces(self, rpe_seeded):
        pos1 = rpe_seeded.get_randomized_positions(15)
        rpe_seeded.reset_seed()
        pos2 = rpe_seeded.get_randomized_positions(15)
        assert torch.equal(pos1, pos2), "reset_seed must reproduce original sequence"

    def test_different_calls_different_positions(self, rpe):
        """Without a seed, consecutive calls should (almost certainly) differ."""
        pos1 = rpe.get_randomized_positions(50)
        pos2 = rpe.get_randomized_positions(50)
        assert not torch.equal(pos1, pos2), \
            "Unseeded consecutive calls should produce different positions"


# ---------------------------------------------------------------------------
# 5. Mode tests
# ---------------------------------------------------------------------------

class TestModes:
    def test_training_mode_randomizes(self, rpe):
        standard = torch.arange(20).unsqueeze(0)  # (1, 20)
        transformed = transform_position_ids(standard, rpe, training=True)
        assert not torch.equal(standard, transformed), \
            "Training mode must change position IDs"
        # Verify the transformed positions are still sorted
        assert torch.all(transformed[0, 1:] > transformed[0, :-1]).item(), \
            "Transformed positions must remain sorted"

    def test_eval_mode_passthrough(self, rpe):
        standard = torch.arange(20).unsqueeze(0)  # (1, 20)
        transformed = transform_position_ids(standard, rpe, training=False)
        assert torch.equal(standard, transformed), \
            "Eval mode must return positions unchanged"

    @pytest.mark.slow
    def test_patched_training_changes_logits(self, model, dummy_input):
        """Patched model in training mode must produce different logits."""
        model.eval()
        with torch.no_grad():
            baseline = model(input_ids=dummy_input).logits

        patcher = RPEPatcher(model, {"max_simulation_length": 8192, "seed": 99})
        patcher.patch()
        try:
            model.train()
            with torch.no_grad():
                patched = model(input_ids=dummy_input).logits
            assert not torch.allclose(baseline, patched, atol=1e-3), \
                "RPE training mode should change model logits"
        finally:
            patcher.unpatch()
            model.eval()

    @pytest.mark.slow
    def test_patched_eval_preserves_logits(self, model, dummy_input):
        """Patched model in eval mode must produce identical logits to baseline."""
        model.eval()
        with torch.no_grad():
            baseline = model(input_ids=dummy_input).logits

        patcher = RPEPatcher(model, {"max_simulation_length": 8192})
        patcher.patch()
        try:
            model.eval()
            with torch.no_grad():
                eval_out = model(input_ids=dummy_input).logits
            assert torch.allclose(baseline, eval_out, atol=1e-3), \
                "RPE eval mode should produce identical logits to unpatched model"
        finally:
            patcher.unpatch()

    @pytest.mark.slow
    def test_patched_eval_after_train_preserves_logits(self, model, dummy_input):
        """Switching from train to eval mid-patch must stop randomization."""
        model.eval()
        with torch.no_grad():
            baseline = model(input_ids=dummy_input).logits

        patcher = RPEPatcher(model, {"max_simulation_length": 8192})
        patcher.patch()
        try:
            # First do a train-mode forward
            model.train()
            with torch.no_grad():
                _ = model(input_ids=dummy_input).logits
            # Switch to eval — should now pass through standard positions
            model.eval()
            with torch.no_grad():
                eval_out = model(input_ids=dummy_input).logits
            assert torch.allclose(baseline, eval_out, atol=1e-3), \
                "Switching to eval mode must stop randomization"
        finally:
            patcher.unpatch()
            model.eval()

    @pytest.mark.slow
    def test_unpatch_restores_original(self, model, dummy_input):
        """After unpatch, model must behave identically to before patching."""
        model.eval()
        with torch.no_grad():
            baseline = model(input_ids=dummy_input).logits

        patcher = RPEPatcher(model, {"max_simulation_length": 8192})
        patcher.patch()
        # Run a forward to exercise the patch
        model.train()
        with torch.no_grad():
            _ = model(input_ids=dummy_input)
        patcher.unpatch()

        model.eval()
        with torch.no_grad():
            restored = model(input_ids=dummy_input).logits
        assert torch.allclose(baseline, restored, atol=1e-3), \
            "Unpatching must fully restore original model behavior"
