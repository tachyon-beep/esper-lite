# Remaining HIGH Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the 5 remaining HIGH priority fixes from the Phase 2 Kasmina audit.

**Architecture:** Three categories of fixes:
1. **Policy fixes** (K-DRL-H4, K-DRL-H2): Change gate/reward logic to prevent gaming
2. **Observation space expansion** (K-DRL-H1, K-DRL-H3): Add new signals for better policy learning
3. **PyTorch fix** (K-PT-H2): Clear shape probe cache on device transfer

**Tech Stack:** Python, PyTorch, dataclasses, NamedTuple

---

## Task 1: K-DRL-H4 - Make G5 Gate Require Counterfactual

**Files:**
- Modify: `src/esper/kasmina/slot.py:454-485`
- Test: `tests/test_seed_slot.py`

**Context:** G5 is the fossilization readiness gate. Currently it falls back to `total_improvement` when counterfactual is unavailable. This is problematic because G5 should only be reachable from PROBATIONARY stage where counterfactual validation is mandatory.

**Step 1: Write the failing test**

Add to `tests/test_seed_slot.py`:

```python
class TestG5RequiresCounterfactual:
    """G5 gate must require counterfactual - no fallback to total_improvement."""

    def test_g5_fails_without_counterfactual(self):
        """G5 should fail if counterfactual_contribution is None."""
        from esper.kasmina.slot import SeedState, SeedMetrics, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
        )
        # Set high total_improvement but NO counterfactual
        state.metrics.initial_val_accuracy = 50.0
        state.metrics.current_val_accuracy = 60.0  # 10% total improvement
        state.metrics.counterfactual_contribution = None  # No counterfactual!

        result = gates._check_g5(state)

        assert not result.passed, "G5 should fail without counterfactual"
        assert "counterfactual_not_available" in result.checks_failed

    def test_g5_passes_with_positive_counterfactual(self):
        """G5 should pass with positive counterfactual contribution."""
        from esper.kasmina.slot import SeedState, SeedMetrics, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
            is_healthy=True,
        )
        state.metrics.counterfactual_contribution = 2.5  # Positive contribution

        result = gates._check_g5(state)

        assert result.passed, f"G5 should pass: {result.checks_failed}"
        assert "positive_contribution" in str(result.checks_passed)

    def test_g5_fails_with_negative_counterfactual(self):
        """G5 should fail with negative counterfactual contribution."""
        from esper.kasmina.slot import SeedState, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
            is_healthy=True,
        )
        state.metrics.counterfactual_contribution = -1.0  # Negative!

        result = gates._check_g5(state)

        assert not result.passed
        assert "negative_contribution" in result.checks_failed

    def test_g5_fails_with_zero_counterfactual(self):
        """G5 should fail with zero counterfactual contribution (no value added)."""
        from esper.kasmina.slot import SeedState, QualityGates
        from esper.leyline import SeedStage

        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="test_bp",
            stage=SeedStage.PROBATIONARY,
            is_healthy=True,
        )
        state.metrics.counterfactual_contribution = 0.0  # Zero = no value added

        result = gates._check_g5(state)

        assert not result.passed, "Zero contribution should not pass G5"
        assert "negative_contribution" in result.checks_failed  # 0 is not > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_slot.py::TestG5RequiresCounterfactual -v`
Expected: FAIL - current code falls back to total_improvement

**Step 3: Implement the fix**

In `src/esper/kasmina/slot.py`, replace the `_check_g5` method (lines 454-485):

```python
    def _check_g5(self, state: SeedState) -> GateResult:
        """G5: Fossilization readiness - requires counterfactual validation.

        G5 is only reachable from PROBATIONARY stage where counterfactual
        validation is mandatory. No fallback to total_improvement.
        """
        checks_passed = []
        checks_failed = []

        # REQUIRE counterfactual - no fallback
        contribution = state.metrics.counterfactual_contribution
        if contribution is None:
            return GateResult(
                gate=GateLevel.G5,
                passed=False,
                score=0.0,
                checks_passed=[],
                checks_failed=["counterfactual_not_available"],
            )

        # Check contribution is positive
        if contribution > 0:
            checks_passed.append(f"positive_contribution_{contribution:.2f}%")
        else:
            checks_failed.append("negative_contribution")

        # Check health
        if state.is_healthy:
            checks_passed.append("healthy")
        else:
            checks_failed.append("unhealthy")

        passed = len(checks_failed) == 0
        return GateResult(
            gate=GateLevel.G5,
            passed=passed,
            score=min(1.0, contribution / 10.0) if contribution > 0 else 0.0,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_slot.py::TestG5RequiresCounterfactual -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x`
Expected: All tests pass

---

## Task 2: K-DRL-H2 - Add Fossilization Legitimacy Discount

**Files:**
- Modify: `src/esper/leyline/__init__.py` (add constant)
- Modify: `src/esper/simic/rewards.py:815-836`
- Test: `tests/test_simic_rewards.py`

**Context:** Seeds can game fossilization rewards by rapidly creating dependencies during BLENDING/SHADOWING that inflate both contribution and progress metrics. The fix adds a legitimacy discount based on PROBATIONARY duration.

**Step 1: Write the failing test**

Add to `tests/test_simic_rewards.py`:

```python
class TestFossilizeLegitimacyDiscount:
    """Fossilization bonus should be discounted for rapid fossilization."""

    def test_short_probation_gets_discounted(self):
        """Seeds with short PROBATIONARY get reduced fossilize bonus."""
        from esper.simic.rewards import (
            _contribution_fossilize_shaping,
            ContributionRewardConfig,
            SeedInfo,
            STAGE_PROBATIONARY,
        )
        from esper.leyline import MIN_PROBATION_EPOCHS

        config = ContributionRewardConfig()

        # Seed with very short probation (1 epoch)
        short_probation = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=1,  # Just entered probation
            seed_age_epochs=15,
        )

        # Seed with full probation
        full_probation = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=MIN_PROBATION_EPOCHS,  # Full probation
            seed_age_epochs=20,
        )

        short_bonus = _contribution_fossilize_shaping(short_probation, 3.0, config)
        full_bonus = _contribution_fossilize_shaping(full_probation, 3.0, config)

        # Short probation should get less bonus
        assert short_bonus < full_bonus, (
            f"Short probation ({short_bonus}) should be less than full ({full_bonus})"
        )

        # Discount should be proportional
        expected_discount = 1 / MIN_PROBATION_EPOCHS
        assert short_bonus == pytest.approx(full_bonus * expected_discount, rel=0.01)

    def test_zero_probation_gets_zero_bonus(self):
        """Seeds with 0 epochs in PROBATIONARY get no bonus."""
        from esper.simic.rewards import (
            _contribution_fossilize_shaping,
            ContributionRewardConfig,
            SeedInfo,
            STAGE_PROBATIONARY,
        )

        config = ContributionRewardConfig()
        seed = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,
            total_improvement=10.0,
            epochs_in_stage=0,  # Just entered, no validation yet
            seed_age_epochs=15,
        )

        bonus = _contribution_fossilize_shaping(seed, 3.0, config)

        # Should get penalty, not bonus (legitimacy_discount = 0)
        assert bonus <= 0, f"Zero probation should not get positive bonus: {bonus}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_rewards.py::TestFossilizeLegitimacyDiscount -v`
Expected: FAIL - constant doesn't exist and no discount applied

**Step 3: Add MIN_PROBATION_EPOCHS constant**

In `src/esper/leyline/__init__.py`, after line 26, add:

```python
MIN_PROBATION_EPOCHS = 5     # Minimum epochs in PROBATIONARY to earn full fossilize bonus
```

Also add to `__all__` list (around line 93):

```python
    "MIN_PROBATION_EPOCHS",
```

**Step 4: Implement the legitimacy discount**

In `src/esper/simic/rewards.py`, first add the import at the top (around line 20):

```python
from esper.leyline import MIN_CULL_AGE, MIN_PROBATION_EPOCHS
```

Then replace `_contribution_fossilize_shaping` function (lines 815-836):

```python
def _contribution_fossilize_shaping(
    seed_info: SeedInfo | None,
    seed_contribution: float | None,
    config: ContributionRewardConfig,
) -> float:
    """Shaping for FOSSILIZE action - with legitimacy discount.

    Rapid fossilization (short PROBATIONARY period) is discounted to prevent
    dependency gaming where seeds create artificial dependencies during
    BLENDING/SHADOWING that inflate metrics.
    """
    if seed_info is None:
        return config.invalid_fossilize_penalty

    # FOSSILIZE only valid from PROBATIONARY
    if seed_info.stage != STAGE_PROBATIONARY:
        return config.invalid_fossilize_penalty

    # Legitimacy discount: must have spent time in PROBATIONARY to earn full bonus
    # This prevents rapid fossilization gaming
    legitimacy_discount = min(1.0, seed_info.epochs_in_stage / MIN_PROBATION_EPOCHS)

    # Use seed_contribution to determine if fossilization is earned
    if seed_contribution is not None and seed_contribution > 0:
        # Bonus scales with actual contribution AND legitimacy
        base_bonus = (
            config.fossilize_base_bonus
            + config.fossilize_contribution_scale * seed_contribution
        )
        return base_bonus * legitimacy_discount

    # Non-contributing or no counterfactual - penalty (no discount on penalties)
    return config.fossilize_noncontributing_penalty
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_simic_rewards.py::TestFossilizeLegitimacyDiscount -v`
Expected: PASS

**Step 6: Run full test suite**

Run: `uv run pytest tests/ -x`
Expected: All tests pass

---

## Task 3: K-PT-H2 - Clear Shape Probe Cache on Device Transfer

**Files:**
- Modify: `src/esper/kasmina/slot.py` (add `to()` override)
- Test: `tests/test_seed_slot.py`

**Context:** `_shape_probe_cache` stores tensors keyed by device string. When `.to(device)` is called, cached tensors on the old device become stale. The cache checks device on access, but explicit clearing is cleaner.

**Step 1: Write the failing test**

Add to `tests/test_seed_slot.py`:

```python
class TestShapeProbeCacheDeviceTransfer:
    """Shape probe cache should be cleared on device transfer."""

    def test_cache_cleared_on_to_call(self):
        """Calling .to(device) should clear the shape probe cache."""
        import torch
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        # Warm the cache
        _ = slot._get_shape_probe("cnn")
        _ = slot._get_shape_probe("transformer")
        assert len(slot._shape_probe_cache) == 2, "Cache should have 2 entries"

        # Transfer to same device (simulating .to() call)
        slot.to("cpu")

        # Cache should be cleared
        assert len(slot._shape_probe_cache) == 0, "Cache should be empty after .to()"

    def test_to_returns_self(self):
        """The .to() method should return self for chaining."""
        import torch
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        result = slot.to("cpu")

        assert result is slot, ".to() should return self"

    def test_to_updates_device(self):
        """The .to() method should update self.device."""
        import torch
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot.to("cpu")  # No-op but should still work

        assert slot.device == torch.device("cpu")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_slot.py::TestShapeProbeCacheDeviceTransfer -v`
Expected: FAIL - SeedSlot has no `.to()` method

**Step 3: Implement the `.to()` method**

In `src/esper/kasmina/slot.py`, add this method to the `SeedSlot` class (after `_get_shape_probe`, around line 567):

**IMPORTANT:** SeedSlot inherits from `nn.Module`, so we MUST call `super().to()` and preserve the full signature.

```python
    def to(self, *args, **kwargs) -> "SeedSlot":
        """Transfer slot to a new device/dtype, clearing cached tensors.

        Extends nn.Module.to() to clear the shape probe cache. The cache
        stores device-specific tensors that become invalid after device
        transfer.

        Args:
            *args, **kwargs: Passed to nn.Module.to() - supports device,
                dtype, memory_format, and non_blocking arguments.

        Returns:
            Self for method chaining (standard nn.Module behavior).
        """
        # Defensive: clear cache before transfer to avoid holding references
        # to tensors on the old device. The cache self-invalidates on access,
        # but explicit clearing prevents potential memory leaks during transfer.
        self._shape_probe_cache.clear()

        # Call parent to handle all registered parameters, buffers, submodules
        super().to(*args, **kwargs)

        # Update self.device tracking if a device was specified
        device = kwargs.get('device')
        if device is None and args:
            arg = args[0]
            if isinstance(arg, (str, torch.device)):
                device = arg
            elif isinstance(arg, torch.Tensor):
                device = arg.device

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device

        return self
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_slot.py::TestShapeProbeCacheDeviceTransfer -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `uv run pytest tests/ -x`
Expected: All tests pass

---

## Task 4: K-DRL-H1 - Add Counterfactual to Observation Space

**Files:**
- Modify: `src/esper/leyline/signals.py` (TensorSchema, FastTrainingSignals)
- Modify: `src/esper/simic/features.py` (obs_to_base_features)
- Modify: `src/esper/simic/ppo.py` (build_observation)
- Test: `tests/test_simic_ppo.py` (or new test file)

**Context:** The policy receives `SEED_IMPROVEMENT` (improvement_since_stage_start) which conflates host learning with seed impact. Adding counterfactual directly to observations lets the policy see TRUE causal attribution.

**Step 1: Write the failing test**

Create or add to `tests/test_simic_features.py`:

```python
"""Tests for feature extraction."""

import pytest


class TestCounterfactualInObservation:
    """Counterfactual contribution should be in observation space."""

    def test_tensor_schema_has_counterfactual(self):
        """TensorSchema should include SEED_COUNTERFACTUAL."""
        from esper.leyline.signals import TensorSchema, TENSOR_SCHEMA_SIZE

        assert hasattr(TensorSchema, 'SEED_COUNTERFACTUAL')
        # Should be index 27 (after AVAILABLE_SLOTS at 26)
        assert TensorSchema.SEED_COUNTERFACTUAL == 27
        # Schema size should be 28
        assert TENSOR_SCHEMA_SIZE == 28

    def test_fast_signals_has_counterfactual(self):
        """FastTrainingSignals should include seed_counterfactual."""
        from esper.leyline.signals import FastTrainingSignals

        # Check the field exists
        empty = FastTrainingSignals.empty()
        assert hasattr(empty, 'seed_counterfactual')
        assert empty.seed_counterfactual == 0.0

    def test_to_vector_includes_counterfactual(self):
        """to_vector() should include counterfactual at correct index."""
        from esper.leyline.signals import FastTrainingSignals, TensorSchema

        signals = FastTrainingSignals.empty()._replace(seed_counterfactual=1.5)
        vec = signals.to_vector()

        assert len(vec) == 28
        assert vec[TensorSchema.SEED_COUNTERFACTUAL] == 1.5

    def test_obs_to_base_features_includes_counterfactual(self):
        """obs_to_base_features should handle seed_counterfactual."""
        from esper.simic.features import obs_to_base_features

        obs = {
            'epoch': 10,
            'global_step': 1000,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'loss_delta': -0.01,
            'train_accuracy': 80.0,
            'val_accuracy': 75.0,
            'accuracy_delta': 1.0,
            'plateau_epochs': 2,
            'best_val_accuracy': 76.0,
            'best_val_loss': 0.55,
            'loss_history_5': [0.7, 0.65, 0.6, 0.58, 0.55],
            'accuracy_history_5': [70, 72, 73, 74, 75],
            'has_active_seed': 1,
            'seed_stage': 6,  # PROBATIONARY
            'seed_epochs_in_stage': 3,
            'seed_alpha': 0.8,
            'seed_improvement': 2.0,
            'available_slots': 1,
            'seed_counterfactual': 1.5,  # NEW
        }

        features = obs_to_base_features(obs)

        assert len(features) == 28, f"Expected 28 features, got {len(features)}"
        # Last feature should be counterfactual (normalized)
        assert features[27] == pytest.approx(1.5 / 10.0, rel=0.01)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_features.py::TestCounterfactualInObservation -v`
Expected: FAIL - attribute doesn't exist

**Step 3: Update TensorSchema**

In `src/esper/leyline/signals.py`, update `TensorSchema` (around line 21):

```python
class TensorSchema(IntEnum):
    """Feature indices for the observation vector.

    Maps feature names to tensor indices for vectorized PPO training.
    Use this to slice state vectors by name without string lookups.

    Total: 28 base features (V2 with counterfactual)
    """
    # Core state (2)
    EPOCH = 0
    GLOBAL_STEP = 1

    # Loss metrics (3)
    TRAIN_LOSS = 2
    VAL_LOSS = 3
    LOSS_DELTA = 4

    # Accuracy metrics (4)
    TRAIN_ACCURACY = 5
    VAL_ACCURACY = 6
    ACCURACY_DELTA = 7
    PLATEAU_EPOCHS = 8

    # Best tracking (2)
    BEST_VAL_ACCURACY = 9
    BEST_VAL_LOSS = 10

    # History (loss - 5 slots: 11-15)
    LOSS_HIST_0 = 11
    LOSS_HIST_1 = 12
    LOSS_HIST_2 = 13
    LOSS_HIST_3 = 14
    LOSS_HIST_4 = 15

    # History (accuracy - 5 slots: 16-20)
    ACC_HIST_0 = 16
    ACC_HIST_1 = 17
    ACC_HIST_2 = 18
    ACC_HIST_3 = 19
    ACC_HIST_4 = 20

    # Seed state (7) - added SEED_COUNTERFACTUAL
    HAS_ACTIVE_SEED = 21
    SEED_STAGE = 22
    SEED_EPOCHS_IN_STAGE = 23
    SEED_ALPHA = 24
    SEED_IMPROVEMENT = 25
    AVAILABLE_SLOTS = 26
    SEED_COUNTERFACTUAL = 27  # True causal attribution
```

Update the constant (around line 72):

```python
# Total feature count for V2 (with counterfactual)
TENSOR_SCHEMA_SIZE = 28
```

**Step 4: Update FastTrainingSignals**

In `src/esper/leyline/signals.py`, update `FastTrainingSignals` (around line 75):

Add new field after `available_slots`:

```python
    available_slots: int
    seed_counterfactual: float  # True causal attribution (0.0 if unavailable)
```

Update `to_vector()` method to include the new field (add at end of return list):

```python
            float(self.available_slots),
            self.seed_counterfactual / 10.0,  # Normalize to ~[-1, 1]
        ]
```

Update `empty()` method:

```python
    @staticmethod
    def empty() -> "FastTrainingSignals":
        """Create empty/default signals."""
        return FastTrainingSignals(
            epoch=0, global_step=0,
            train_loss=0.0, val_loss=0.0, loss_delta=0.0,
            train_accuracy=0.0, val_accuracy=0.0, accuracy_delta=0.0,
            plateau_epochs=0, best_val_accuracy=0.0, best_val_loss=float('inf'),
            loss_history_5=(0.0, 0.0, 0.0, 0.0, 0.0),
            accuracy_history_5=(0.0, 0.0, 0.0, 0.0, 0.0),
            has_active_seed=0, seed_stage=0, seed_epochs_in_stage=0,
            seed_alpha=0.0, seed_improvement=0.0, available_slots=1,
            seed_counterfactual=0.0,
        )
```

**Step 5: Update TrainingSignals.to_fast()**

In `src/esper/leyline/signals.py`, update `to_fast()` method signature and implementation:

Add parameter:

```python
    def to_fast(
        self,
        seed_stage: int | None = None,
        seed_epochs_in_stage: int | None = None,
        seed_alpha: float | None = None,
        seed_improvement: float | None = None,
        seed_counterfactual: float | None = None,
    ) -> FastTrainingSignals:
```

Add field to dataclass (around line 200):

```python
    seed_counterfactual: float = 0.0
```

Update return statement to include:

```python
            seed_counterfactual=self.seed_counterfactual if seed_counterfactual is None else seed_counterfactual,
```

**Step 6: Update obs_to_base_features**

In `src/esper/simic/features.py`, update `obs_to_base_features` (around line 67):

Update docstring:

```python
    """Extract V2-style base features (28 dims) with pre-normalization.
    ...
    - Seed state: has_active_seed, seed_stage, seed_epochs_in_stage,
                  seed_alpha, seed_improvement, seed_counterfactual (6)
    - Slots: available_slots (1)

    Total: 28 features
    ...
```

Add to the return list (after `available_slots`):

```python
        float(obs['available_slots']),                        # Usually 0-2, small scale ok
        obs.get('seed_counterfactual', 0.0) / 10.0,          # ~[-1, 1] typical range
    ]
```

**Step 7: Update build_observation in ppo.py**

In `src/esper/simic/ppo.py`, update `build_observation` (around line 77):

After setting `seed_improvement`, add:

```python
        obs['seed_counterfactual'] = seed_state.metrics.counterfactual_contribution or 0.0
```

And in the else branch (around line 88), add:

```python
        obs['seed_counterfactual'] = 0.0
```

**Step 8: Run tests to verify**

Run: `uv run pytest tests/test_simic_features.py::TestCounterfactualInObservation -v`
Expected: PASS

Run: `uv run pytest tests/ -x`
Expected: All tests pass (may need to fix other tests expecting 27 features)

---

## Task 5: K-DRL-H3 - Add Host State Observability

**Files:**
- Modify: `src/esper/leyline/signals.py` (TensorSchema, FastTrainingSignals, TrainingMetrics)
- Modify: `src/esper/simic/features.py` (obs_to_base_features)
- Test: `tests/test_simic_features.py`

**Context:** The host network learns continuously, but this non-stationarity is invisible to the RL policy. Adding host grad norm and learning phase helps the agent adapt to host training dynamics.

**Step 1: Write the failing test**

Add to `tests/test_simic_features.py`:

```python
class TestHostStateObservability:
    """Host network state should be observable by policy."""

    def test_tensor_schema_has_host_signals(self):
        """TensorSchema should include host state signals."""
        from esper.leyline.signals import TensorSchema, TENSOR_SCHEMA_SIZE

        assert hasattr(TensorSchema, 'HOST_GRAD_NORM')
        assert hasattr(TensorSchema, 'HOST_LEARNING_PHASE')
        # Should be indices 28, 29 (after SEED_COUNTERFACTUAL at 27)
        assert TensorSchema.HOST_GRAD_NORM == 28
        assert TensorSchema.HOST_LEARNING_PHASE == 29
        # Schema size should be 30
        assert TENSOR_SCHEMA_SIZE == 30

    def test_fast_signals_has_host_state(self):
        """FastTrainingSignals should include host state."""
        from esper.leyline.signals import FastTrainingSignals

        empty = FastTrainingSignals.empty()
        assert hasattr(empty, 'host_grad_norm')
        assert hasattr(empty, 'host_learning_phase')

    def test_obs_to_base_features_includes_host_state(self):
        """obs_to_base_features should handle host state."""
        from esper.simic.features import obs_to_base_features

        obs = {
            'epoch': 10,
            'global_step': 1000,
            'train_loss': 0.5,
            'val_loss': 0.6,
            'loss_delta': -0.01,
            'train_accuracy': 80.0,
            'val_accuracy': 75.0,
            'accuracy_delta': 1.0,
            'plateau_epochs': 2,
            'best_val_accuracy': 76.0,
            'best_val_loss': 0.55,
            'loss_history_5': [0.7, 0.65, 0.6, 0.58, 0.55],
            'accuracy_history_5': [70, 72, 73, 74, 75],
            'has_active_seed': 1,
            'seed_stage': 6,
            'seed_epochs_in_stage': 3,
            'seed_alpha': 0.8,
            'seed_improvement': 2.0,
            'available_slots': 1,
            'seed_counterfactual': 1.5,
            'host_grad_norm': 0.5,  # NEW
            'host_learning_phase': 0.4,  # NEW (epoch 10 / max 25)
        }

        features = obs_to_base_features(obs)

        assert len(features) == 30, f"Expected 30 features, got {len(features)}"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simic_features.py::TestHostStateObservability -v`
Expected: FAIL

**Step 3: Update TensorSchema**

In `src/esper/leyline/signals.py`, add to `TensorSchema`:

```python
    SEED_COUNTERFACTUAL = 27  # True causal attribution
    HOST_GRAD_NORM = 28       # Host gradient norm (training dynamics)
    HOST_LEARNING_PHASE = 29  # epoch / max_epochs (training progress)
```

Update constant:

```python
TENSOR_SCHEMA_SIZE = 30
```

**Step 4: Update FastTrainingSignals**

Add fields:

```python
    seed_counterfactual: float
    host_grad_norm: float      # Host gradient norm
    host_learning_phase: float # epoch / max_epochs
```

Update `to_vector()`:

```python
            self.seed_counterfactual / 10.0,
            min(self.host_grad_norm, 10.0) / 10.0,  # Clamp and normalize
            self.host_learning_phase,  # Already [0, 1]
        ]
```

Update `empty()`:

```python
            seed_counterfactual=0.0,
            host_grad_norm=0.0,
            host_learning_phase=0.0,
```

**Step 5: Update TrainingSignals**

**IMPORTANT (Code Review Feedback):** `grad_norm_host` already exists in `TrainingMetrics` (line 179). Do NOT duplicate it in `TrainingSignals`. Use `self.metrics.grad_norm_host` instead.

Add ONLY `seed_counterfactual` field (around line 200):

```python
    seed_counterfactual: float = 0.0  # Seed-specific, not in TrainingMetrics
```

Update `to_fast()` signature - add `max_epochs` parameter for learning_phase calculation:

```python
    def to_fast(
        self,
        seed_stage: int | None = None,
        seed_epochs_in_stage: int | None = None,
        seed_alpha: float | None = None,
        seed_improvement: float | None = None,
        seed_counterfactual: float | None = None,
        max_epochs: int = 200,  # For host_learning_phase calculation
    ) -> FastTrainingSignals:
```

Update return statement - use existing `metrics.grad_norm_host`:

```python
            seed_counterfactual=self.seed_counterfactual if seed_counterfactual is None else seed_counterfactual,
            host_grad_norm=self.metrics.grad_norm_host,  # Already in TrainingMetrics!
            host_learning_phase=self.metrics.epoch / max(1, max_epochs),  # Computed from epoch
```

**Step 6: Update obs_to_base_features**

In `src/esper/simic/features.py`:

Update docstring:

```python
    """Extract V2-style base features (30 dims) with pre-normalization.
    ...
    - Host state: host_grad_norm, host_learning_phase (2)

    Total: 30 features
```

Add to return list:

```python
        obs.get('seed_counterfactual', 0.0) / 10.0,
        min(obs.get('host_grad_norm', 0.0), 10.0) / 10.0,  # Clamp and normalize
        obs.get('host_learning_phase', 0.0),  # Already [0, 1]
    ]
```

**Step 7: Update signals_to_features in ppo.py**

**IMPORTANT (Code Review Feedback):** `max_epochs` is NOT available in the current function signature. Use the default pattern from `obs_to_base_features` (max_epochs=200) as a sensible default.

First, update the function signature to accept `max_epochs`:

```python
def signals_to_features(
    signals: TrainingSignals,
    model=None,
    tracker=None,
    use_telemetry: bool = True,
    max_epochs: int = 200,  # Add this parameter
) -> list[float]:
```

Then after the counterfactual line, add:

```python
        obs['host_grad_norm'] = signals.metrics.grad_norm_host
        obs['host_learning_phase'] = signals.metrics.epoch / max(1, max_epochs)
```

And in the else branch (no live model), add:

```python
        obs['host_grad_norm'] = signals.metrics.grad_norm_host
        obs['host_learning_phase'] = signals.metrics.epoch / max(1, max_epochs)
```

**Step 8: Run tests**

Run: `uv run pytest tests/test_simic_features.py -v`
Expected: PASS

Run: `uv run pytest tests/ -x`
Expected: All tests pass (may need to update tests expecting 27/28 features)

---

## Task 6: Commit and Verify

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 2: Type check**

Run: `python3 -m py_compile src/esper/kasmina/slot.py src/esper/simic/rewards.py src/esper/leyline/signals.py src/esper/simic/features.py src/esper/simic/ppo.py`
Expected: No errors

**Step 3: Commit**

```bash
git add src/esper/kasmina/slot.py src/esper/simic/rewards.py \
        src/esper/leyline/__init__.py src/esper/leyline/signals.py \
        src/esper/simic/features.py src/esper/simic/ppo.py \
        tests/test_seed_slot.py tests/test_simic_rewards.py tests/test_simic_features.py

git commit -m "fix(audit): implement remaining HIGH priority fixes from Phase 2

K-DRL-H4: Make G5 gate require counterfactual (no fallback)
K-DRL-H2: Add fossilization legitimacy discount based on PROBATIONARY duration
K-PT-H2: Clear shape probe cache on .to(device) calls
K-DRL-H1: Add counterfactual to observation space (SEED_COUNTERFACTUAL)
K-DRL-H3: Add host state observability (HOST_GRAD_NORM, HOST_LEARNING_PHASE)

Observation space expanded from 27 to 30 features (V2)."
```

---

## Summary

| Task | Fix ID | Description | Lines Changed |
|------|--------|-------------|---------------|
| 1 | K-DRL-H4 | G5 gate requires counterfactual | ~35 |
| 2 | K-DRL-H2 | Fossilization legitimacy discount | ~25 |
| 3 | K-PT-H2 | Clear shape probe cache on .to() | ~15 |
| 4 | K-DRL-H1 | Add counterfactual to obs space | ~50 |
| 5 | K-DRL-H3 | Add host state observability | ~40 |

**Total:** ~165 lines of changes + tests

**Dependencies:**
- Task 4 and 5 both modify signals.py - do them **STRICTLY** in order (4 → 5)
- Tasks 1, 2, 3 are independent - can be done in any order

---

## Reviewer Feedback Incorporated

This plan was reviewed by 3 specialist agents. The following feedback was incorporated:

### DRL Expert (APPROVED)
- All fixes align with RL best practices
- Suggestion: Consider superlinear legitimacy discount (deferred to future refinement)
- Note: Observation space expansion is a breaking change for existing policies

### Code Reviewer (CHANGES MADE)
- **Fixed:** Added `max_epochs` parameter to `signals_to_features()`
- **Fixed:** Use existing `metrics.grad_norm_host` instead of duplicating in TrainingSignals
- **Added:** Edge case test for counterfactual=0.0

### PyTorch Expert (CHANGES MADE)
- **Fixed:** `.to()` method now calls `super().to()` with proper `*args, **kwargs` signature
- **Fixed:** Device extraction handles all nn.Module.to() argument patterns
- Note: Cache clearing is defensive (cache self-invalidates, but explicit is safer)
