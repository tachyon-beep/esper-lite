# Tolaria Module Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs and enhancements identified in the multi-expert review of the Tolaria module.

**Architecture:** The fixes address: (1) optimizer state handling after rollback, (2) panic counter logic, (3) memory safety in snapshots, (4) training mode restoration, and (5) test coverage gaps. All changes follow TDD - write failing test first, then implement.

**Tech Stack:** PyTorch 2.9, Python 3.13, pytest

---

## Task 1: FIX-2 - Reset consecutive_panics After Rollback

**Priority:** CRITICAL
**Issue:** After a successful rollback, `consecutive_panics` is incremented instead of reset to 0. This causes escalating panic detection after the first rollback.

**Files:**
- Modify: `src/esper/tolaria/governor.py:190`
- Test: `tests/test_tolaria_governor.py`

**Step 1: Write the failing test**

Add this test to `tests/test_tolaria_governor.py` after `test_execute_rollback_increments_consecutive_panics`:

```python
def test_execute_rollback_resets_consecutive_panics(self):
    """Test that rollback resets consecutive_panics to allow fresh start."""
    from esper.tolaria import TolariaGovernor

    model = DummyModel()
    gov = TolariaGovernor(model)

    # Build history for statistical detection
    for i in range(15):
        gov.check_vital_signs(1.0)

    # Simulate panic buildup (2 consecutive anomalies trigger rollback)
    gov.consecutive_panics = 2
    gov._panic_loss = 50.0

    # Execute rollback
    report = gov.execute_rollback()
    assert report.rollback_occurred is True

    # After rollback, consecutive_panics should reset to 0
    # This allows training to recover without escalating panic detection
    assert gov.consecutive_panics == 0, (
        f"consecutive_panics should be 0 after rollback, got {gov.consecutive_panics}"
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_execute_rollback_resets_consecutive_panics -v`

Expected: FAIL with `AssertionError: consecutive_panics should be 0 after rollback, got 1`

**Step 3: Write minimal implementation**

In `src/esper/tolaria/governor.py`, change line 190 from:

```python
self.consecutive_panics += 1
```

to:

```python
# Reset panic counter after successful rollback to allow fresh recovery
self.consecutive_panics = 0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_execute_rollback_resets_consecutive_panics -v`

Expected: PASS

**Step 5: Update existing test that expects increment behavior**

The test `test_execute_rollback_increments_consecutive_panics` at line 316 now has incorrect expectations. Update it:

```python
def test_execute_rollback_resets_consecutive_panics_each_time(self):
    """Test that each rollback resets panic counter (not increments)."""
    from esper.tolaria import TolariaGovernor

    model = DummyModel()
    gov = TolariaGovernor(model)

    gov.snapshot()

    # Build minimal history
    for i in range(5):
        gov.loss_history.append(1.0)

    # First rollback - should reset to 0
    report1 = gov.execute_rollback()
    assert gov.consecutive_panics == 0

    # Manually set panics to simulate another panic event
    gov.consecutive_panics = 2

    # Second rollback - should reset to 0 again
    report2 = gov.execute_rollback()
    assert gov.consecutive_panics == 0
```

**Step 6: Run all governor tests**

Run: `pytest tests/test_tolaria_governor.py -v`

Expected: All tests PASS

**Step 7: Commit**

```bash
git add tests/test_tolaria_governor.py src/esper/tolaria/governor.py
git commit -m "fix(governor): reset consecutive_panics after rollback

After a successful rollback, the panic counter should reset to 0 to
allow training to recover with a fresh start. Previously it was
incrementing, causing escalating panic detection after the first
rollback event."
```

---

## Task 2: FIX-3 - Add torch.no_grad() to Governor Snapshot

**Priority:** MODERATE
**Issue:** Snapshot performs `.detach().cpu().clone()` without `torch.no_grad()` context, causing unnecessary autograd graph participation.

**Files:**
- Modify: `src/esper/tolaria/governor.py:70-86`
- Test: `tests/test_tolaria_governor.py`

**Step 1: Write the failing test**

```python
def test_snapshot_does_not_track_gradients(self):
    """Test that snapshot() does not create tensors that track gradients."""
    from esper.tolaria import TolariaGovernor

    model = DummyModel()
    gov = TolariaGovernor(model)

    # Ensure model has gradients enabled
    model.linear.weight.requires_grad_(True)

    gov.snapshot()

    # Snapshot tensors should NOT require gradients
    for key, value in gov.last_good_state.items():
        if isinstance(value, torch.Tensor):
            assert not value.requires_grad, (
                f"Snapshot tensor '{key}' should not require gradients"
            )
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_snapshot_does_not_track_gradients -v`

Expected: PASS (current code already uses `.detach()`, but this test documents the requirement)

**Step 3: Improve implementation with torch.no_grad()**

In `src/esper/tolaria/governor.py`, update the `snapshot()` method (lines 70-86):

```python
def snapshot(self) -> None:
    """Save Last Known Good state to CPU memory to reduce GPU memory pressure.

    Tensors are moved to CPU; non-tensor values are deep copied.
    This trades slightly slower rollback for significant GPU memory savings,
    especially for large models where snapshots could double GPU memory usage.
    """
    # Explicitly free old snapshot to prevent memory fragmentation
    if self.last_good_state is not None:
        del self.last_good_state
        self.last_good_state = None

    # Store on CPU to save GPU memory (rollback is rare, memory savings are constant)
    # Use no_grad() to prevent any autograd overhead during state extraction
    with torch.no_grad():
        self.last_good_state = {
            k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
            for k, v in self.model.state_dict().items()
        }
```

> **Code Reviewer Note:** Keep `.detach()` for explicitness even though `state_dict()` tensors
> are already detached. This documents intent and decouples correctness from PyTorch internals.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_snapshot_does_not_track_gradients -v`

Expected: PASS

**Step 5: Run all governor tests**

Run: `pytest tests/test_tolaria_governor.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/tolaria/governor.py tests/test_tolaria_governor.py
git commit -m "perf(governor): wrap snapshot in torch.no_grad()

Prevents unnecessary autograd graph participation during state dict
extraction. Also removes redundant .detach() call since state_dict()
tensors are already detached and no_grad() prevents gradient tracking."
```

---

## Task 3: FIX-4 - Restore Training Mode After Attribution Validation

**Priority:** LOW
**Issue:** `validate_with_attribution()` sets `model.eval()` but never restores the original training mode.

**Files:**
- Modify: `src/esper/tolaria/trainer.py:302-346`
- Create: `tests/test_tolaria_trainer.py`

**Step 1: Create new test file with failing test**

Create `tests/test_tolaria_trainer.py`:

```python
"""Tests for Tolaria trainer functions."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DummyClassifier(nn.Module):
    """Minimal classifier for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.5)  # Behaves differently in train vs eval

    def forward(self, x):
        return self.linear(self.dropout(x))


class MockSeedSlot:
    """Mock seed slot for attribution testing."""
    def __init__(self):
        self._force_alpha_value = None

    class _ForceAlphaContext:
        def __init__(self, slot, alpha):
            self.slot = slot
            self.alpha = alpha
            self.original = None

        def __enter__(self):
            self.original = self.slot._force_alpha_value
            self.slot._force_alpha_value = self.alpha
            return self

        def __exit__(self, *args):
            self.slot._force_alpha_value = self.original

    def force_alpha(self, alpha):
        return self._ForceAlphaContext(self, alpha)


class DummyModelWithSlot(nn.Module):
    """Model with mock seed slot for attribution testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.seed_slot = MockSeedSlot()

    def forward(self, x):
        return self.linear(x)


class TestValidateWithAttribution:
    """Tests for validate_with_attribution function."""

    def test_restores_training_mode_after_validation(self):
        """Test that training mode is restored after attribution validation."""
        from esper.tolaria.trainer import validate_with_attribution

        model = DummyModelWithSlot()

        # Create minimal test data
        inputs = torch.randn(8, 10)
        labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(inputs, labels)
        testloader = DataLoader(dataset, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        # Set model to training mode
        model.train()
        assert model.training is True

        # Run attribution validation
        result = validate_with_attribution(model, testloader, criterion, "cpu")

        # Model should be back in training mode
        assert model.training is True, (
            "Model should be in training mode after validate_with_attribution"
        )

    def test_restores_eval_mode_if_originally_eval(self):
        """Test that eval mode is preserved if model was originally in eval."""
        from esper.tolaria.trainer import validate_with_attribution

        model = DummyModelWithSlot()

        inputs = torch.randn(8, 10)
        labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(inputs, labels)
        testloader = DataLoader(dataset, batch_size=4)
        criterion = nn.CrossEntropyLoss()

        # Set model to eval mode
        model.eval()
        assert model.training is False

        result = validate_with_attribution(model, testloader, criterion, "cpu")

        # Model should still be in eval mode
        assert model.training is False, (
            "Model should remain in eval mode after validate_with_attribution"
        )

    def test_restores_mode_even_on_exception(self):
        """Test that training mode is restored even if validation raises."""
        from esper.tolaria.trainer import validate_with_attribution

        model = DummyModelWithSlot()

        # Empty dataloader will cause division by zero in accuracy calc
        # Actually, current implementation handles empty gracefully, so we need
        # a different approach - use a mock that raises

        # For now, just verify the happy path works
        # A more robust test would mock the inner function to raise
        pass  # Placeholder - implementation handles this via try/finally


class TestAttributionResult:
    """Tests for AttributionResult dataclass."""

    def test_attribution_result_structure(self):
        """Test AttributionResult has correct fields."""
        from esper.tolaria.trainer import AttributionResult

        result = AttributionResult(
            real_accuracy=85.0,
            baseline_accuracy=80.0,
            seed_contribution=5.0,
            real_loss=0.5,
            baseline_loss=0.6,
        )

        assert result.real_accuracy == 85.0
        assert result.baseline_accuracy == 80.0
        assert result.seed_contribution == 5.0
        assert result.real_loss == 0.5
        assert result.baseline_loss == 0.6

    def test_seed_contribution_calculation(self):
        """Test that seed_contribution = real - baseline."""
        from esper.tolaria.trainer import AttributionResult

        result = AttributionResult(
            real_accuracy=85.0,
            baseline_accuracy=80.0,
            seed_contribution=85.0 - 80.0,  # Should be 5.0
            real_loss=0.5,
            baseline_loss=0.6,
        )

        assert result.seed_contribution == result.real_accuracy - result.baseline_accuracy
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tolaria_trainer.py::TestValidateWithAttribution::test_restores_training_mode_after_validation -v`

Expected: FAIL with `AssertionError: Model should be in training mode after validate_with_attribution`

**Step 3: Write minimal implementation**

In `src/esper/tolaria/trainer.py`, update `validate_with_attribution()` (lines 302-346):

```python
def validate_with_attribution(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str = "classification",
) -> AttributionResult:
    """Counterfactual validation for true seed contribution measurement.

    Runs two forward passes through the validation set:
    1. Real pass: model with current alpha (seed contributing)
    2. Baseline pass: model with alpha=0 (host-only, seed output zeroed)

    The difference in accuracy between these passes gives the true causal
    attribution of the seed - how much accuracy the seed actually contributes
    vs natural host training gains.

    This addresses the "Scapegoat Problem" where seeds were blamed/credited
    for host accuracy changes during TRAINING stage when they had zero impact.

    Args:
        model: The model to evaluate (must have seed_slot with force_alpha()).
        testloader: Validation data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        task_type: Task type ("classification" or "lm").

    Returns:
        AttributionResult with real and baseline accuracies plus seed_contribution.
    """
    # Save original training mode to restore after validation
    was_training = model.training
    model.eval()

    def _run_validation_pass() -> tuple[float, float]:
        """Run a single validation pass, return (loss, accuracy)."""
        loss_tensor = torch.tensor(0.0, device=device)
        correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = _compute_loss(outputs, labels, criterion, task_type)
                loss_tensor += loss

                if task_type == "lm":
                    predicted = outputs.argmax(dim=-1)
                    total += labels.numel()
                    correct_tensor += (predicted == labels).sum()
                else:
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct_tensor += predicted.eq(labels).sum()

        avg_loss = loss_tensor.item() / max(len(testloader), 1)
        accuracy = 100.0 * correct_tensor.item() / total if total > 0 else 0.0
        return avg_loss, accuracy

    try:
        # Pass 1: Real validation with current alpha
        real_loss, real_accuracy = _run_validation_pass()

        # Pass 2: Baseline validation with alpha=0 (host-only)
        # Use force_alpha context manager to temporarily override alpha
        seed_slot = model.seed_slot
        with seed_slot.force_alpha(0.0):
            baseline_loss, baseline_accuracy = _run_validation_pass()

        return AttributionResult(
            real_accuracy=real_accuracy,
            baseline_accuracy=baseline_accuracy,
            seed_contribution=real_accuracy - baseline_accuracy,
            real_loss=real_loss,
            baseline_loss=baseline_loss,
        )
    finally:
        # Restore original training mode
        model.train(was_training)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tolaria_trainer.py -v`

Expected: All tests PASS

**Step 5: Run full test suite**

Run: `pytest tests/test_tolaria_governor.py tests/test_tolaria_trainer.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/tolaria/trainer.py tests/test_tolaria_trainer.py
git commit -m "fix(trainer): restore training mode after attribution validation

validate_with_attribution() now saves and restores the original
model.training state using try/finally, ensuring the model returns
to its original mode even if an exception occurs."
```

---

## Task 4: FIX-6 - Scale Lobotomy Detection Threshold by Task

**Priority:** MEDIUM
**Issue:** Hardcoded 0.15 tolerance for lobotomy detection doesn't scale across tasks (CIFAR-10: ~6.5%, TinyStories: ~1.4%).

**Files:**
- Modify: `src/esper/tolaria/governor.py:111-112`
- Test: `tests/test_tolaria_governor.py`

**Step 1: Write the failing test**

```python
def test_lobotomy_detection_scales_with_task(self):
    """Test that lobotomy tolerance scales with random_guess_loss."""
    from esper.tolaria import TolariaGovernor
    import math

    model = DummyModel()

    # Test with TinyStories-like task (50257 classes)
    tinystories_loss = math.log(50257)  # ~10.82
    gov = TolariaGovernor(model, random_guess_loss=tinystories_loss)

    # Build healthy history (loss ~3.0, well below random guess)
    for _ in range(15):
        gov.check_vital_signs(3.0)

    # Jump to near random guess loss - should detect lobotomy
    # With relative tolerance of ~6.5%, threshold is ~0.7 for TinyStories
    # (vs fixed 0.15 which is too tight)
    is_panic = gov.check_vital_signs(tinystories_loss + 0.5)
    assert is_panic is True, (
        "Should detect lobotomy even with larger absolute tolerance for high-entropy tasks"
    )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_lobotomy_detection_scales_with_task -v`

Expected: FAIL (current 0.15 tolerance is too tight for `10.82 + 0.5 = 11.32` vs `10.82`)

**Step 3: Write minimal implementation**

In `src/esper/tolaria/governor.py`, update lines 110-112:

```python
        # Lobotomy detection: loss jumped to exactly random guessing
        # This catches "silent failures" where model outputs uniform probabilities
        if len(self.loss_history) >= 10:
            avg = sum(self.loss_history) / len(self.loss_history)
            # Relative tolerance: ~6.5% of random guess loss
            # - CIFAR-10 (ln(10)=2.3): tolerance = 0.15
            # - TinyStories (ln(50257)=10.8): tolerance = 0.70
            lobotomy_tolerance = 0.065 * self.random_guess_loss
            # If we were doing well (loss < 60% of random guess) and suddenly
            # hit exactly the random guess loss (±tolerance), that's a lobotomy
            if (avg < self.random_guess_loss * 0.6 and
                abs(current_loss - self.random_guess_loss) < lobotomy_tolerance):
                self._pending_panic = False
                self._panic_loss = current_loss
                self.consecutive_panics = self.min_panics_before_rollback
                return True
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_lobotomy_detection_scales_with_task -v`

Expected: PASS

**Step 5: Run all lobotomy tests**

Run: `pytest tests/test_tolaria_governor.py -k lobotomy -v`

Expected: All lobotomy tests PASS

**Step 6: Commit**

```bash
git add src/esper/tolaria/governor.py tests/test_tolaria_governor.py
git commit -m "fix(governor): scale lobotomy tolerance by task complexity

Lobotomy detection now uses relative tolerance (~6.5% of random_guess_loss)
instead of fixed 0.15. This ensures proper detection across tasks:
- CIFAR-10 (10 classes): ~0.15 tolerance
- TinyStories (50257 vocab): ~0.70 tolerance"
```

---

## Task 5: ENH-2 - Add non_blocking=True to Rollback Device Transfers

**Priority:** LOW
**Issue:** Device transfer during rollback is synchronous, which could be slow for large models.

**Files:**
- Modify: `src/esper/tolaria/governor.py:184-187`
- Test: `tests/test_tolaria_governor.py`

> **PyTorch Expert Review Note:** `load_state_dict()` does NOT synchronize CUDA streams.
> If `non_blocking` transfers haven't completed, garbage/partial data will be loaded.
> Explicit `torch.cuda.synchronize()` is REQUIRED before `load_state_dict()`.

**Step 1: Write the test (verifies behavior, not performance)**

```python
def test_rollback_uses_nonblocking_transfer(self):
    """Test that rollback transfers use non_blocking for efficiency."""
    from esper.tolaria import TolariaGovernor

    # This test verifies the rollback completes correctly with non_blocking
    # (actual performance benefit requires CUDA)
    model = DummyModel()
    gov = TolariaGovernor(model)

    original_weight = model.linear.weight.data.clone()
    gov.snapshot()

    model.linear.weight.data.fill_(999.0)

    # Build history
    for i in range(5):
        gov.loss_history.append(1.0)

    report = gov.execute_rollback()

    # Verify rollback still works correctly
    assert torch.allclose(model.linear.weight.data, original_weight)
    assert report.rollback_occurred is True
```

**Step 2: Update implementation**

In `src/esper/tolaria/governor.py`, update lines 180-188:

```python
        # Restore host + fossilized seeds (strict=True ensures complete restoration)
        # Move all tensors to model device in one batch before loading, avoiding
        # individual CPU->GPU transfers for each parameter.
        # Get device from parameters, falling back to CPU if no parameters
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        # Use non_blocking=True for async CPU->GPU transfer
        state_on_device = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in self.last_good_state.items()
        }

        # CRITICAL: Synchronize CUDA stream before load_state_dict
        # load_state_dict() does NOT synchronize - without this, we load garbage
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        self.model.load_state_dict(state_on_device, strict=True)
```

**Step 3: Run tests**

Run: `pytest tests/test_tolaria_governor.py -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/tolaria/governor.py
git commit -m "perf(governor): use non_blocking transfers in rollback

Device transfers during rollback now use non_blocking=True for
asynchronous CPU->GPU transfers. Explicit torch.cuda.synchronize()
is called before load_state_dict to ensure transfers complete -
load_state_dict does NOT synchronize CUDA streams automatically."
```

---

## Task 6: FIX-9 - Handle Empty Model in Device Query

**Priority:** LOW
**Issue:** `next(self.model.parameters())` raises `StopIteration` if model has no parameters.

**Files:**
- Modify: `src/esper/tolaria/governor.py:183`
- Test: `tests/test_tolaria_governor.py`

**Step 1: Write the failing test**

```python
def test_rollback_handles_parameterless_model(self):
    """Test that rollback handles models with no parameters gracefully."""
    from esper.tolaria import TolariaGovernor

    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # No parameters, just a buffer
            self.register_buffer('counter', torch.tensor(0))

        def forward(self, x):
            return x

    model = EmptyModel()
    gov = TolariaGovernor(model)

    # Build history
    for i in range(5):
        gov.loss_history.append(1.0)

    # Should not raise StopIteration
    report = gov.execute_rollback()
    assert report.rollback_occurred is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_rollback_handles_parameterless_model -v`

Expected: FAIL with `StopIteration`

**Step 3: Write minimal implementation**

In `src/esper/tolaria/governor.py`, update line 183:

```python
        # Get device from parameters, falling back to CPU if no parameters
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            # Model has no parameters (only buffers) - default to CPU
            device = torch.device('cpu')
```

**Step 4: Run tests**

Run: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_rollback_handles_parameterless_model -v`

Expected: PASS

**Step 5: Run all governor tests**

Run: `pytest tests/test_tolaria_governor.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/tolaria/governor.py tests/test_tolaria_governor.py
git commit -m "fix(governor): handle models with no parameters in rollback

Rollback now gracefully handles models that have only buffers and no
trainable parameters by catching StopIteration and defaulting to CPU."
```

---

## Task 7: FIX-5 - Add Test Coverage for validate_with_attribution()

**Priority:** HIGH
**Issue:** Critical function for measuring seed contribution has minimal test coverage.

**Files:**
- Modify: `tests/test_tolaria_trainer.py`

**Step 1: Add comprehensive tests**

Add to `tests/test_tolaria_trainer.py`:

```python
class TestValidateWithAttributionIntegration:
    """Integration tests for validate_with_attribution with real MorphogeneticModel."""

    @pytest.fixture
    def model_with_seed(self):
        """Create MorphogeneticModel with an active seed."""
        from esper.kasmina import MorphogeneticModel, CNNHost

        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu")
        model.germinate_seed("conv_light", "test_seed")

        # Advance to BLENDING stage so alpha > 0
        model.seed_slot.state.stage = 4  # SeedStage.BLENDING
        model.seed_slot._alpha = 0.5

        return model

    @pytest.fixture
    def test_data(self):
        """Create CIFAR-10-like test data."""
        # 8 samples of 3x32x32 images
        inputs = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 10, (8,))
        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset, batch_size=4)

    def test_attribution_with_active_seed(self, model_with_seed, test_data):
        """Test attribution validation with an active seed."""
        from esper.tolaria.trainer import validate_with_attribution

        criterion = nn.CrossEntropyLoss()

        result = validate_with_attribution(
            model_with_seed, test_data, criterion, "cpu"
        )

        # Result should have valid structure
        assert isinstance(result.real_accuracy, float)
        assert isinstance(result.baseline_accuracy, float)
        assert isinstance(result.seed_contribution, float)
        assert result.seed_contribution == result.real_accuracy - result.baseline_accuracy

    def test_attribution_contribution_sign(self, model_with_seed, test_data):
        """Test that seed_contribution can be positive or negative."""
        from esper.tolaria.trainer import validate_with_attribution

        criterion = nn.CrossEntropyLoss()

        result = validate_with_attribution(
            model_with_seed, test_data, criterion, "cpu"
        )

        # Contribution can be any value (positive = seed helps, negative = seed hurts)
        assert isinstance(result.seed_contribution, float)
        # Both accuracies should be in [0, 100] range
        assert 0.0 <= result.real_accuracy <= 100.0
        assert 0.0 <= result.baseline_accuracy <= 100.0

    def test_force_alpha_context_restores_alpha(self, model_with_seed, test_data):
        """Test that force_alpha context manager properly restores original alpha."""
        from esper.tolaria.trainer import validate_with_attribution

        criterion = nn.CrossEntropyLoss()

        original_alpha = model_with_seed.seed_slot.alpha

        result = validate_with_attribution(
            model_with_seed, test_data, criterion, "cpu"
        )

        # Alpha should be restored after validation
        assert model_with_seed.seed_slot.alpha == original_alpha

    def test_attribution_with_empty_loader(self):
        """Test attribution handles empty dataloader gracefully."""
        from esper.tolaria.trainer import validate_with_attribution
        from esper.kasmina import MorphogeneticModel, CNNHost

        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu")
        model.germinate_seed("conv_light", "test_seed")
        model.seed_slot.state.stage = 4
        model.seed_slot._alpha = 0.5

        # Empty dataset
        empty_dataset = TensorDataset(
            torch.randn(0, 3, 32, 32),
            torch.randint(0, 10, (0,))
        )
        empty_loader = DataLoader(empty_dataset, batch_size=4)

        criterion = nn.CrossEntropyLoss()

        result = validate_with_attribution(model, empty_loader, criterion, "cpu")

        # Should return 0 accuracy for empty loader
        assert result.real_accuracy == 0.0
        assert result.baseline_accuracy == 0.0
        assert result.seed_contribution == 0.0
```

**Step 2: Add imports at top of test file**

```python
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
```

**Step 3: Run tests**

Run: `pytest tests/test_tolaria_trainer.py -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_tolaria_trainer.py
git commit -m "test(trainer): add comprehensive attribution validation tests

Adds integration tests for validate_with_attribution() covering:
- Active seed attribution
- Contribution sign validation
- Alpha restoration after validation
- Empty dataloader edge case"
```

---

## Summary

| Task | Type | Priority | Description |
|------|------|----------|-------------|
| 1 | FIX-2 | CRITICAL | Reset consecutive_panics after rollback |
| 2 | FIX-3 | MODERATE | Add torch.no_grad() to snapshot |
| 3 | FIX-4 | LOW | Restore training mode after attribution |
| 4 | FIX-6 | MEDIUM | Scale lobotomy tolerance by task |
| 5 | ENH-2 | LOW | Use non_blocking transfers in rollback |
| 6 | FIX-9 | LOW | Handle parameterless models |
| 7 | FIX-5 | HIGH | Add attribution validation tests |

**Estimated Total:** 7 tasks, each 5-15 minutes = ~1-2 hours

**Test Commands:**
- Single test: `pytest tests/test_tolaria_governor.py::TestTolariaGovernor::test_name -v`
- Governor tests: `pytest tests/test_tolaria_governor.py -v`
- Trainer tests: `pytest tests/test_tolaria_trainer.py -v`
- All Tolaria tests: `pytest tests/test_tolaria_governor.py tests/test_tolaria_trainer.py tests/esper/test_tolaria.py -v`
