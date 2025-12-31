# Comprehensive Telemetry System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add comprehensive RL and PyTorch telemetry to diagnose training issues at both "Ops Normal" (routine monitoring) and "Debug" (catch everything) levels.

**Architecture:** Extend existing telemetry infrastructure with: (1) new TelemetryEventTypes, (2) dataclasses for structured metrics, (3) TelemetryLevel enum for verbosity control, (4) hooks in PPO update loops and training functions. Debug telemetry triggers automatically on anomaly detection.

**Tech Stack:** Python dataclasses, PyTorch gradient/memory APIs, existing Nissa hub, TelemetryEvent system

---

## Phase 1: Foundation - Telemetry Configuration and Event Types

### Task 1: Add TelemetryLevel Enum and Config

**Files:**
- Create: `src/esper/simic/telemetry_config.py`
- Test: `tests/simic/test_telemetry_config.py`

**Step 1: Write the failing test**

Create `tests/simic/test_telemetry_config.py`:

```python
"""Tests for telemetry configuration."""

import pytest

from esper.simic.telemetry_config import TelemetryLevel, TelemetryConfig


class TestTelemetryLevel:
    """Tests for TelemetryLevel enum."""

    def test_levels_are_ordered(self):
        """Telemetry levels have correct ordering."""
        assert TelemetryLevel.OFF < TelemetryLevel.MINIMAL
        assert TelemetryLevel.MINIMAL < TelemetryLevel.NORMAL
        assert TelemetryLevel.NORMAL < TelemetryLevel.DEBUG

    def test_level_comparison(self):
        """Can compare levels for conditional logging."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        assert config.level >= TelemetryLevel.NORMAL
        assert config.level < TelemetryLevel.DEBUG


class TestTelemetryConfig:
    """Tests for TelemetryConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = TelemetryConfig()
        assert config.level == TelemetryLevel.NORMAL
        assert config.auto_escalate_on_anomaly is True

    def test_should_collect_ops_normal(self):
        """should_collect returns True for ops normal at NORMAL level."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        assert config.should_collect("ops_normal") is True
        assert config.should_collect("debug") is False

    def test_should_collect_debug(self):
        """should_collect returns True for debug at DEBUG level."""
        config = TelemetryConfig(level=TelemetryLevel.DEBUG)
        assert config.should_collect("ops_normal") is True
        assert config.should_collect("debug") is True

    def test_escalate_temporarily(self):
        """escalate_temporarily increases level for N epochs."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        config.escalate_temporarily(epochs=5)
        assert config.effective_level == TelemetryLevel.DEBUG
        assert config.escalation_epochs_remaining == 5

    def test_tick_escalation(self):
        """tick_escalation decrements and returns to normal."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        config.escalate_temporarily(epochs=2)
        config.tick_escalation()
        assert config.escalation_epochs_remaining == 1
        config.tick_escalation()
        assert config.escalation_epochs_remaining == 0
        assert config.effective_level == TelemetryLevel.NORMAL
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_telemetry_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'esper.simic.telemetry_config'`

**Step 3: Implement TelemetryLevel and TelemetryConfig**

Create `src/esper/simic/telemetry_config.py`:

```python
"""Telemetry Configuration for Simic Training.

Controls what telemetry is collected at different verbosity levels,
with automatic escalation to DEBUG mode on anomaly detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class TelemetryLevel(IntEnum):
    """Telemetry verbosity levels."""

    OFF = 0      # No telemetry collection
    MINIMAL = 1  # Episode summaries only
    NORMAL = 2   # Per-batch PPO metrics (Ops Normal)
    DEBUG = 3    # Full diagnostics (Oh Shit mode)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection.

    Supports automatic escalation to DEBUG mode when anomalies
    are detected, with configurable escalation duration.
    """

    level: TelemetryLevel = TelemetryLevel.NORMAL

    # Ops Normal collection intervals
    gradient_check_interval: int = 1   # Every N epochs
    memory_check_interval: int = 5     # Every N epochs

    # Debug settings
    per_layer_gradients: bool = False
    activation_monitoring: bool = False
    weight_tracking: bool = False
    weight_track_interval: int = 10

    # Auto-escalation
    auto_escalate_on_anomaly: bool = True
    anomaly_escalation_epochs: int = 5

    # Internal state
    _escalation_epochs_remaining: int = field(default=0, repr=False)

    @property
    def effective_level(self) -> TelemetryLevel:
        """Return effective level (accounting for temporary escalation)."""
        if self._escalation_epochs_remaining > 0:
            return TelemetryLevel.DEBUG
        return self.level

    @property
    def escalation_epochs_remaining(self) -> int:
        """Return remaining escalation epochs."""
        return self._escalation_epochs_remaining

    def should_collect(self, category: str) -> bool:
        """Check if telemetry category should be collected.

        Args:
            category: One of 'ops_normal' or 'debug'

        Returns:
            True if category should be collected at current level
        """
        level = self.effective_level
        if category == "ops_normal":
            return level >= TelemetryLevel.NORMAL
        elif category == "debug":
            return level >= TelemetryLevel.DEBUG
        return False

    def escalate_temporarily(self, epochs: int | None = None) -> None:
        """Temporarily escalate to DEBUG level.

        Args:
            epochs: Number of epochs to stay escalated (default: anomaly_escalation_epochs)
        """
        if epochs is None:
            epochs = self.anomaly_escalation_epochs
        self._escalation_epochs_remaining = epochs

    def tick_escalation(self) -> None:
        """Decrement escalation counter (call once per epoch)."""
        if self._escalation_epochs_remaining > 0:
            self._escalation_epochs_remaining -= 1


__all__ = ["TelemetryLevel", "TelemetryConfig"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_telemetry_config.py -v`

Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/telemetry_config.py tests/simic/test_telemetry_config.py
git commit -m "feat(simic): add TelemetryLevel and TelemetryConfig"
```

---

### Task 2: Extend TelemetryEventType with New Events

**Files:**
- Modify: `src/esper/leyline/telemetry.py:27-51`
- Test: `tests/leyline/test_telemetry_events.py` (create)

**Step 1: Write the failing test**

Create `tests/leyline/test_telemetry_events.py`:

```python
"""Tests for telemetry event types."""

from esper.leyline import TelemetryEventType


class TestTelemetryEventTypes:
    """Tests for new telemetry event types."""

    def test_ppo_update_event_exists(self):
        """PPO_UPDATE_COMPLETED event type exists."""
        assert TelemetryEventType.PPO_UPDATE_COMPLETED

    def test_debug_event_types_exist(self):
        """Debug-level event types exist."""
        assert TelemetryEventType.RATIO_EXPLOSION_DETECTED
        assert TelemetryEventType.VALUE_COLLAPSE_DETECTED
        assert TelemetryEventType.GRADIENT_PATHOLOGY_DETECTED
        assert TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED

    def test_ops_normal_event_types_exist(self):
        """Ops normal event types exist."""
        assert TelemetryEventType.MEMORY_WARNING
        assert TelemetryEventType.REWARD_HACKING_SUSPECTED
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/leyline/test_telemetry_events.py -v`

Expected: FAIL with `AttributeError: PPO_UPDATE_COMPLETED`

**Step 3: Add new event types**

In `src/esper/leyline/telemetry.py`, update TelemetryEventType (after line 50):

```python
class TelemetryEventType(Enum):
    """Types of telemetry events."""

    # Lifecycle events
    SEED_GERMINATED = auto()
    SEED_STAGE_CHANGED = auto()
    SEED_FOSSILIZED = auto()
    SEED_CULLED = auto()

    # Training events
    EPOCH_COMPLETED = auto()
    PLATEAU_DETECTED = auto()
    IMPROVEMENT_DETECTED = auto()
    TAMIYO_INITIATED = auto()

    # Health events
    ISOLATION_VIOLATION = auto()
    GRADIENT_ANOMALY = auto()
    PERFORMANCE_DEGRADATION = auto()

    # Command events
    COMMAND_ISSUED = auto()
    COMMAND_EXECUTED = auto()
    COMMAND_FAILED = auto()

    # === NEW: PPO Training Events (Ops Normal) ===
    PPO_UPDATE_COMPLETED = auto()
    MEMORY_WARNING = auto()
    REWARD_HACKING_SUSPECTED = auto()

    # === NEW: Debug Events (triggered by anomalies) ===
    RATIO_EXPLOSION_DETECTED = auto()
    VALUE_COLLAPSE_DETECTED = auto()
    GRADIENT_PATHOLOGY_DETECTED = auto()
    NUMERICAL_INSTABILITY_DETECTED = auto()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/leyline/test_telemetry_events.py -v`

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/esper/leyline/telemetry.py tests/leyline/test_telemetry_events.py
git commit -m "feat(leyline): add PPO and debug telemetry event types"
```

---

## Phase 2: Quick Wins - PPO Core Metrics

### Task 3: Add PPOHealthTelemetry Dataclass

**Files:**
- Create: `src/esper/simic/ppo_telemetry.py`
- Test: `tests/simic/test_ppo_telemetry.py`

**Step 1: Write the failing test**

Create `tests/simic/test_ppo_telemetry.py`:

```python
"""Tests for PPO telemetry dataclasses."""

import pytest

from esper.simic.ppo_telemetry import PPOHealthTelemetry, ValueFunctionTelemetry


class TestPPOHealthTelemetry:
    """Tests for PPOHealthTelemetry."""

    def test_create_from_metrics(self):
        """Can create PPOHealthTelemetry from raw metrics."""
        telemetry = PPOHealthTelemetry(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=0.8,
            approx_kl=0.01,
            clip_fraction=0.15,
            ratio_mean=1.0,
            ratio_std=0.1,
            ratio_max=1.5,
            ratio_min=0.7,
        )
        assert telemetry.policy_loss == 0.5
        assert telemetry.ratio_max == 1.5

    def test_is_ratio_healthy(self):
        """is_ratio_healthy detects problematic ratios."""
        healthy = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=1.5, ratio_min=0.7,
        )
        assert healthy.is_ratio_healthy() is True

        unhealthy = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.5, ratio_max=6.0, ratio_min=0.1,
        )
        assert unhealthy.is_ratio_healthy() is False

    def test_to_dict(self):
        """Can convert to dict for telemetry event."""
        telemetry = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=1.5, ratio_min=0.7,
        )
        d = telemetry.to_dict()
        assert d["policy_loss"] == 0.5
        assert "ratio_max" in d


class TestValueFunctionTelemetry:
    """Tests for ValueFunctionTelemetry."""

    def test_explained_variance_calculation(self):
        """explained_variance is computed correctly."""
        import torch

        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([1.1, 1.9, 3.1, 3.9])

        telemetry = ValueFunctionTelemetry.from_tensors(returns, values)
        # Should be close to 1.0 (good prediction)
        assert telemetry.explained_variance > 0.9

    def test_is_healthy(self):
        """is_healthy detects value function collapse."""
        import torch

        # Good predictions
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([1.1, 1.9, 3.1, 3.9])
        healthy = ValueFunctionTelemetry.from_tensors(returns, values)
        assert healthy.is_healthy() is True

        # Bad predictions (constant value)
        bad_values = torch.tensor([2.5, 2.5, 2.5, 2.5])
        unhealthy = ValueFunctionTelemetry.from_tensors(returns, bad_values)
        assert unhealthy.explained_variance < 0.1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_ppo_telemetry.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement PPOHealthTelemetry and ValueFunctionTelemetry**

Create `src/esper/simic/ppo_telemetry.py`:

```python
"""PPO Telemetry Dataclasses.

Structured telemetry for PPO training diagnostics, covering:
- Policy health (ratio statistics, KL, entropy)
- Value function health (explained variance)
- Action distribution analysis
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class PPOHealthTelemetry:
    """Per-update PPO health metrics.

    These metrics are collected every PPO update (Ops Normal level).
    """

    # Loss components
    policy_loss: float
    value_loss: float
    entropy: float

    # KL and clipping
    approx_kl: float
    clip_fraction: float

    # Ratio statistics (CRITICAL for detecting explosions)
    ratio_mean: float
    ratio_std: float
    ratio_max: float
    ratio_min: float

    # Optional: early stopping info
    early_stopped: bool = False
    early_stop_epoch: int | None = None

    def is_ratio_healthy(
        self,
        max_ratio_threshold: float = 5.0,
        min_ratio_threshold: float = 0.1,
    ) -> bool:
        """Check if ratio statistics indicate healthy training.

        Args:
            max_ratio_threshold: Ratio above this is concerning
            min_ratio_threshold: Ratio below this is concerning

        Returns:
            True if ratios are within healthy bounds
        """
        return (
            self.ratio_max < max_ratio_threshold
            and self.ratio_min > min_ratio_threshold
        )

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


@dataclass(slots=True)
class ValueFunctionTelemetry:
    """Value function health metrics.

    Tracks how well the critic is predicting returns.
    """

    explained_variance: float
    value_mean: float
    value_std: float
    return_mean: float
    return_std: float
    advantage_mean: float
    advantage_std: float

    @classmethod
    def from_tensors(
        cls,
        returns: "torch.Tensor",
        values: "torch.Tensor",
        advantages: "torch.Tensor | None" = None,
    ) -> "ValueFunctionTelemetry":
        """Create from PyTorch tensors.

        Args:
            returns: Computed returns [N]
            values: Predicted values [N]
            advantages: Computed advantages [N] (optional)

        Returns:
            ValueFunctionTelemetry instance
        """
        import torch

        # Explained variance: 1 - Var(returns - values) / Var(returns)
        var_returns = returns.var()
        if var_returns > 1e-8:
            explained_var = 1.0 - (returns - values).var() / var_returns
            explained_var = explained_var.item()
        else:
            explained_var = 0.0

        # Advantage stats
        if advantages is not None:
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
        else:
            adv_mean = 0.0
            adv_std = 1.0

        return cls(
            explained_variance=explained_var,
            value_mean=values.mean().item(),
            value_std=values.std().item(),
            return_mean=returns.mean().item(),
            return_std=returns.std().item(),
            advantage_mean=adv_mean,
            advantage_std=adv_std,
        )

    def is_healthy(self, min_explained_variance: float = 0.1) -> bool:
        """Check if value function is healthy.

        Args:
            min_explained_variance: Below this indicates broken critic

        Returns:
            True if value function appears healthy
        """
        return self.explained_variance >= min_explained_variance

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


__all__ = ["PPOHealthTelemetry", "ValueFunctionTelemetry"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_ppo_telemetry.py -v`

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/ppo_telemetry.py tests/simic/test_ppo_telemetry.py
git commit -m "feat(simic): add PPOHealthTelemetry and ValueFunctionTelemetry"
```

---

### Task 4: Add Ratio Statistics to PPO Update

**Files:**
- Modify: `src/esper/simic/ppo.py:304-346`
- Test: `tests/simic/test_ppo_ratio_stats.py`

**Step 1: Write the failing test**

Create `tests/simic/test_ppo_ratio_stats.py`:

```python
"""Tests for PPO ratio statistics collection."""

import torch
import pytest

from esper.simic.ppo import PPOAgent
from esper.simic.buffers import RolloutBuffer


class TestPPORatioStats:
    """Tests for ratio statistics in PPO update."""

    @pytest.fixture
    def agent(self):
        """Create a simple PPO agent."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_buffer(self, agent):
        """Create buffer with some transitions."""
        for _ in range(20):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            action, log_prob, value, _ = agent.get_action(state, action_mask)
            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0.1,
                done=False,
                action_mask=action_mask,
            )
        return agent.buffer

    def test_update_returns_ratio_stats(self, agent, filled_buffer):
        """PPO update returns ratio statistics."""
        metrics = agent.update(last_value=0.0)

        assert "ratio_mean" in metrics
        assert "ratio_std" in metrics
        assert "ratio_max" in metrics
        assert "ratio_min" in metrics

    def test_ratio_stats_are_reasonable(self, agent, filled_buffer):
        """Ratio stats have reasonable values for fresh policy."""
        metrics = agent.update(last_value=0.0)

        # For a fresh policy with no updates, ratios should be near 1.0
        assert 0.5 < metrics["ratio_mean"] < 2.0
        assert metrics["ratio_std"] < 1.0
        assert metrics["ratio_max"] < 5.0
        assert metrics["ratio_min"] > 0.1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_ppo_ratio_stats.py -v`

Expected: FAIL with `KeyError: 'ratio_mean'`

**Step 3: Add ratio statistics collection to PPO update**

In `src/esper/simic/ppo.py`, modify the batch loop in `update()` method. After line 305 (`ratio = torch.exp(log_probs - old_log_probs)`), add ratio tracking:

```python
                ratio = torch.exp(log_probs - old_log_probs)

                # Track ratio statistics for telemetry
                metrics['ratio_mean'].append(ratio.mean().item())
                metrics['ratio_std'].append(ratio.std().item())
                metrics['ratio_max'].append(ratio.max().item())
                metrics['ratio_min'].append(ratio.min().item())

                surr1 = ratio * batch_advantages
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_ppo_ratio_stats.py -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_ratio_stats.py
git commit -m "feat(simic): add ratio statistics to PPO update metrics"
```

---

### Task 5: Add Explained Variance to PPO Update

**Files:**
- Modify: `src/esper/simic/ppo.py:273-277`
- Test: `tests/simic/test_ppo_explained_variance.py`

**Step 1: Write the failing test**

Create `tests/simic/test_ppo_explained_variance.py`:

```python
"""Tests for explained variance in PPO update."""

import torch
import pytest

from esper.simic.ppo import PPOAgent


class TestPPOExplainedVariance:
    """Tests for explained variance calculation."""

    @pytest.fixture
    def agent(self):
        """Create a simple PPO agent."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_buffer(self, agent):
        """Create buffer with some transitions."""
        for _ in range(20):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            action, log_prob, value, _ = agent.get_action(state, action_mask)
            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0.1,
                done=False,
                action_mask=action_mask,
            )
        return agent.buffer

    def test_update_returns_explained_variance(self, agent, filled_buffer):
        """PPO update returns explained_variance metric."""
        metrics = agent.update(last_value=0.0)
        assert "explained_variance" in metrics

    def test_explained_variance_is_reasonable(self, agent, filled_buffer):
        """Explained variance is in valid range."""
        metrics = agent.update(last_value=0.0)
        # Can be negative (worse than mean) or up to 1.0 (perfect)
        assert -2.0 < metrics["explained_variance"] < 1.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_ppo_explained_variance.py -v`

Expected: FAIL with `KeyError: 'explained_variance'`

**Step 3: Add explained variance computation**

In `src/esper/simic/ppo.py`, after line 276 (after computing returns and advantages), add:

```python
        # Compute returns and advantages directly on device
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda, device=self.device
        )

        # Compute explained variance for value function diagnostics
        # Uses values from buffer before any updates
        values_tensor = torch.tensor(
            [t.value for t in self.buffer.transitions],
            device=self.device,
        )
        var_returns = returns.var()
        if var_returns > 1e-8:
            explained_variance = 1.0 - (returns - values_tensor).var() / var_returns
            explained_variance = explained_variance.item()
        else:
            explained_variance = 0.0

        metrics = defaultdict(list)
        metrics['explained_variance'] = [explained_variance]  # Single value, not per-batch
        early_stopped = False
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_ppo_explained_variance.py -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_explained_variance.py
git commit -m "feat(simic): add explained variance to PPO update metrics"
```

---

### Task 6: Add Reward Components Telemetry

**Files:**
- Create: `src/esper/simic/reward_telemetry.py`
- Modify: `src/esper/simic/rewards.py` (add return_components parameter)
- Test: `tests/simic/test_reward_telemetry.py`

**Step 1: Write the failing test**

Create `tests/simic/test_reward_telemetry.py`:

```python
"""Tests for reward component telemetry."""

import pytest

from esper.simic.reward_telemetry import RewardComponentsTelemetry
from esper.simic.rewards import (
    compute_shaped_reward,
    compute_contribution_reward,
    SeedInfo,
    RewardConfig,
    ContributionRewardConfig,
)


class TestRewardComponentsTelemetry:
    """Tests for RewardComponentsTelemetry."""

    def test_from_shaped_reward(self):
        """Can capture components from compute_shaped_reward."""
        # Create a mock action enum
        from enum import IntEnum

        class MockAction(IntEnum):
            WAIT = 0
            GERMINATE = 1
            FOSSILIZE = 2
            CULL = 3

        seed_info = SeedInfo(
            stage=3,  # TRAINING
            improvement_since_stage_start=1.5,
            total_improvement=2.0,
            epochs_in_stage=5,
            seed_params=10000,
            previous_stage=2,
            seed_age_epochs=8,
        )

        # Use the extended version that returns components
        reward, components = compute_shaped_reward(
            action=MockAction.WAIT,
            acc_delta=0.5,
            val_acc=65.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=10000,
            host_params=100000,
            return_components=True,
        )

        assert isinstance(components, RewardComponentsTelemetry)
        assert components.base_acc_delta != 0.0
        assert components.total_reward == reward

    def test_components_sum_to_total(self):
        """Component rewards sum to total reward."""
        from enum import IntEnum

        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=3,
            improvement_since_stage_start=1.0,
            total_improvement=1.5,
            epochs_in_stage=3,
            seed_params=5000,
            previous_stage=2,
            seed_age_epochs=5,
        )

        reward, components = compute_shaped_reward(
            action=MockAction.WAIT,
            acc_delta=0.3,
            val_acc=60.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=25,
            total_params=5000,
            host_params=100000,
            return_components=True,
        )

        computed_sum = (
            components.base_acc_delta
            + components.compute_rent
            + components.stage_bonus
            + components.pbrs_bonus
            + components.action_shaping
            + components.terminal_bonus
        )
        assert abs(computed_sum - components.total_reward) < 1e-6
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_reward_telemetry.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create RewardComponentsTelemetry dataclass**

Create `src/esper/simic/reward_telemetry.py`:

```python
"""Reward Telemetry Dataclasses.

Captures per-component breakdown of reward computation
for diagnosing reward hacking and tuning reward weights.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(slots=True)
class RewardComponentsTelemetry:
    """Breakdown of reward components for debugging.

    Each field represents one component of the total reward.
    All components should sum to total_reward.
    """

    # Base signal
    base_acc_delta: float = 0.0

    # Penalties
    compute_rent: float = 0.0

    # Bonuses
    stage_bonus: float = 0.0
    pbrs_bonus: float = 0.0
    action_shaping: float = 0.0
    terminal_bonus: float = 0.0

    # Contribution-primary specific (if applicable)
    seed_contribution: float | None = None
    bounded_attribution: float | None = None

    # Total
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


__all__ = ["RewardComponentsTelemetry"]
```

**Step 4: Modify compute_shaped_reward to support return_components**

In `src/esper/simic/rewards.py`, modify `compute_shaped_reward()` signature and implementation. Add `return_components: bool = False` parameter and track components throughout:

```python
def compute_shaped_reward(
    action: IntEnum,
    acc_delta: float,
    val_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: RewardConfig | None = None,
    action_enum: type | None = None,
    return_components: bool = False,
) -> float | tuple[float, "RewardComponentsTelemetry"]:
    """Compute shaped reward for seed lifecycle control.

    ... existing docstring ...

    Args:
        ... existing args ...
        return_components: If True, return (reward, RewardComponentsTelemetry) tuple

    Returns:
        Shaped reward value, or (reward, components) if return_components=True
    """
    from esper.simic.reward_telemetry import RewardComponentsTelemetry

    if config is None:
        config = _DEFAULT_CONFIG

    # Track components if requested
    components = RewardComponentsTelemetry() if return_components else None

    reward = 0.0

    # Base: accuracy improvement
    base_acc = acc_delta * config.acc_delta_weight
    reward += base_acc
    if components:
        components.base_acc_delta = base_acc

    # Compute rent
    rent_penalty = 0.0
    if host_params > 0 and total_params > 0:
        growth_ratio = total_params / host_params
        scaled_cost = math.log(1.0 + max(0.0, growth_ratio))
        rent_penalty = config.compute_rent_weight * scaled_cost
        rent_penalty = min(rent_penalty, config.max_rent_penalty)
        reward -= rent_penalty
    if components:
        components.compute_rent = -rent_penalty

    # Stage bonuses
    stage_bonus = 0.0
    if seed_info is not None:
        stage = seed_info.stage
        improvement = seed_info.improvement_since_stage_start

        if stage == STAGE_TRAINING:
            stage_bonus += config.training_bonus
            if improvement > 0:
                stage_bonus += improvement * config.stage_improvement_weight
        elif stage == STAGE_BLENDING:
            stage_bonus += config.blending_bonus
            if acc_delta > 0:
                stage_bonus += config.blending_improvement_bonus
        elif stage == STAGE_FOSSILIZED:
            stage_bonus += config.fossilized_bonus

    reward += stage_bonus
    if components:
        components.stage_bonus = stage_bonus

    # PBRS bonus
    pbrs_bonus = 0.0
    if seed_info is not None:
        # ... existing PBRS logic ...
        if seed_info.epochs_in_stage == 0:
            prev_stage = seed_info.previous_stage
            prev_epochs = 0
        else:
            prev_stage = seed_info.stage
            prev_epochs = seed_info.epochs_in_stage - 1

        current_obs = {
            "has_active_seed": 1,
            "seed_stage": seed_info.stage,
            "seed_epochs_in_stage": seed_info.epochs_in_stage,
        }
        prev_obs = {
            "has_active_seed": 1,
            "seed_stage": prev_stage,
            "seed_epochs_in_stage": prev_epochs,
        }
        phi_t = compute_seed_potential(current_obs)
        phi_t_prev = compute_seed_potential(prev_obs)
        pb_bonus = compute_pbrs_bonus(phi_t_prev, phi_t, gamma=0.99)
        pbrs_bonus = config.seed_potential_weight * pb_bonus
        reward += pbrs_bonus
    if components:
        components.pbrs_bonus = pbrs_bonus

    # Action shaping
    action_shaping = 0.0
    action_name = action.name
    if is_germinate_action(action):
        action_shaping = _germinate_shaping(seed_info, epoch, max_epochs, config)
    elif action_name == "FOSSILIZE":
        action_shaping = _advance_shaping(seed_info, config)
    elif action_name == "CULL":
        action_shaping = _cull_shaping(seed_info, config)
    elif action_name == "WAIT":
        action_shaping = _wait_shaping(seed_info, acc_delta, config)
    reward += action_shaping
    if components:
        components.action_shaping = action_shaping

    # Terminal bonus
    terminal_bonus = 0.0
    if epoch == max_epochs:
        terminal_bonus = val_acc * config.terminal_acc_weight
        reward += terminal_bonus
    if components:
        components.terminal_bonus = terminal_bonus
        components.total_reward = reward

    if return_components:
        return reward, components
    return reward
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_reward_telemetry.py -v`

Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add src/esper/simic/reward_telemetry.py src/esper/simic/rewards.py tests/simic/test_reward_telemetry.py
git commit -m "feat(simic): add reward components telemetry"
```

---

## Phase 3: Ops Normal Infrastructure

### Task 7: Add Memory Telemetry Module

**Files:**
- Create: `src/esper/simic/memory_telemetry.py`
- Test: `tests/simic/test_memory_telemetry.py`

**Step 1: Write the failing test**

Create `tests/simic/test_memory_telemetry.py`:

```python
"""Tests for memory telemetry."""

import torch
import pytest

from esper.simic.memory_telemetry import MemoryMetrics, collect_memory_metrics


class TestMemoryMetrics:
    """Tests for MemoryMetrics."""

    def test_collect_cpu_returns_zeros(self):
        """CPU device returns zero metrics."""
        metrics = collect_memory_metrics(torch.device("cpu"))
        assert metrics.allocated_mb == 0.0
        assert metrics.reserved_mb == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_collect_cuda_returns_positive(self):
        """CUDA device returns positive metrics when memory used."""
        device = torch.device("cuda:0")
        # Allocate some memory
        x = torch.randn(1000, 1000, device=device)

        metrics = collect_memory_metrics(device)
        assert metrics.allocated_mb > 0
        assert metrics.reserved_mb >= metrics.allocated_mb
        assert metrics.headroom_mb >= 0

        del x

    def test_oom_risk_score_range(self):
        """OOM risk score is in valid range."""
        metrics = MemoryMetrics(
            allocated_mb=8000,
            reserved_mb=10000,
            max_allocated_mb=9000,
            fragmentation_ratio=1.25,
            utilization=0.8,
            headroom_mb=2000,
            oom_risk_score=0.3,
        )
        assert 0.0 <= metrics.oom_risk_score <= 1.0

    def test_is_healthy(self):
        """is_healthy detects low headroom."""
        healthy = MemoryMetrics(
            allocated_mb=5000,
            reserved_mb=6000,
            max_allocated_mb=5500,
            fragmentation_ratio=1.2,
            utilization=0.5,
            headroom_mb=4000,
            oom_risk_score=0.1,
        )
        assert healthy.is_healthy() is True

        unhealthy = MemoryMetrics(
            allocated_mb=9000,
            reserved_mb=9800,
            max_allocated_mb=9500,
            fragmentation_ratio=1.1,
            utilization=0.95,
            headroom_mb=50,
            oom_risk_score=0.8,
        )
        assert unhealthy.is_healthy() is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_memory_telemetry.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement MemoryMetrics**

Create `src/esper/simic/memory_telemetry.py`:

```python
"""GPU Memory Telemetry.

Tracks CUDA memory usage for OOM prevention and fragmentation detection.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass(slots=True)
class MemoryMetrics:
    """GPU memory statistics for training monitoring.

    Ops Normal level - collected periodically (every N epochs).
    """

    # CUDA memory (MB)
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float

    # Derived indicators
    fragmentation_ratio: float  # reserved / allocated
    utilization: float  # allocated / total
    headroom_mb: float  # total - reserved

    # Risk indicator
    oom_risk_score: float  # 0-1, higher = more risk

    # Optional: seed-specific
    seed_param_mb: float = 0.0

    def is_healthy(
        self,
        min_headroom_mb: float = 200.0,
        max_fragmentation: float = 2.5,
    ) -> bool:
        """Check if memory situation is healthy.

        Args:
            min_headroom_mb: Minimum free memory buffer
            max_fragmentation: Maximum fragmentation ratio

        Returns:
            True if memory is in healthy state
        """
        return (
            self.headroom_mb >= min_headroom_mb
            and self.fragmentation_ratio <= max_fragmentation
        )

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


def collect_memory_metrics(
    device: torch.device,
    seed_module: nn.Module | None = None,
) -> MemoryMetrics:
    """Collect GPU memory statistics.

    Call after backward() for accurate peak tracking.
    Returns zero metrics for CPU device.

    Args:
        device: PyTorch device to check
        seed_module: Optional seed module for param memory calculation

    Returns:
        MemoryMetrics instance
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        return MemoryMetrics(
            allocated_mb=0.0,
            reserved_mb=0.0,
            max_allocated_mb=0.0,
            fragmentation_ratio=1.0,
            utilization=0.0,
            headroom_mb=float("inf"),
            oom_risk_score=0.0,
            seed_param_mb=0.0,
        )

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2

    # Fragmentation ratio
    fragmentation = reserved / max(allocated, 1.0)

    # Utilization
    utilization = allocated / total

    # Headroom
    headroom = total - reserved

    # Seed param memory
    seed_param_mb = 0.0
    if seed_module is not None:
        seed_param_mb = sum(
            p.numel() * p.element_size() for p in seed_module.parameters()
        ) / 1024**2

    # OOM risk score (heuristic)
    oom_risk = 0.0
    if reserved > 0.9 * total:
        oom_risk += 0.5
    if fragmentation > 2.0:
        oom_risk += 0.3
    if headroom < 200:
        oom_risk += 0.2
    oom_risk = min(1.0, oom_risk)

    return MemoryMetrics(
        allocated_mb=allocated,
        reserved_mb=reserved,
        max_allocated_mb=max_alloc,
        fragmentation_ratio=fragmentation,
        utilization=utilization,
        headroom_mb=headroom,
        oom_risk_score=oom_risk,
        seed_param_mb=seed_param_mb,
    )


__all__ = ["MemoryMetrics", "collect_memory_metrics"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_memory_telemetry.py -v`

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/memory_telemetry.py tests/simic/test_memory_telemetry.py
git commit -m "feat(simic): add GPU memory telemetry"
```

---

### Task 8: Enhance Gradient Collector with Layer Metrics

**Files:**
- Modify: `src/esper/simic/gradient_collector.py`
- Test: `tests/simic/test_gradient_collector_enhanced.py`

**Step 1: Write the failing test**

Create `tests/simic/test_gradient_collector_enhanced.py`:

```python
"""Tests for enhanced gradient collector."""

import torch
import torch.nn as nn
import pytest

from esper.simic.gradient_collector import (
    collect_seed_gradients,
    GradientHealthMetrics,
)


class TestGradientHealthMetrics:
    """Tests for GradientHealthMetrics dataclass."""

    def test_dataclass_fields(self):
        """GradientHealthMetrics has all required fields."""
        metrics = GradientHealthMetrics(
            gradient_norm=1.0,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            min_layer_norm=0.1,
            max_layer_norm=2.0,
            norm_ratio=20.0,
            zero_grad_fraction=0.01,
            nan_count=0,
            inf_count=0,
        )
        assert metrics.norm_ratio == 20.0

    def test_is_healthy(self):
        """is_healthy detects gradient pathologies."""
        healthy = GradientHealthMetrics(
            gradient_norm=1.0,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            min_layer_norm=0.5,
            max_layer_norm=2.0,
            norm_ratio=4.0,
            zero_grad_fraction=0.01,
            nan_count=0,
            inf_count=0,
        )
        assert healthy.is_healthy() is True

        # NaN detected
        nan_grads = GradientHealthMetrics(
            gradient_norm=1.0,
            gradient_health=0.5,
            has_vanishing=False,
            has_exploding=False,
            min_layer_norm=0.5,
            max_layer_norm=2.0,
            norm_ratio=4.0,
            zero_grad_fraction=0.0,
            nan_count=5,
            inf_count=0,
        )
        assert nan_grads.is_healthy() is False


class TestEnhancedCollector:
    """Tests for enhanced gradient collection."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        return model

    def test_collect_returns_enhanced_metrics(self, simple_model):
        """collect_seed_gradients returns GradientHealthMetrics."""
        # Forward/backward to generate gradients
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()

        result = collect_seed_gradients(
            simple_model.parameters(),
            return_enhanced=True,
        )

        assert isinstance(result, GradientHealthMetrics)
        assert result.gradient_norm > 0
        assert result.min_layer_norm > 0
        assert result.max_layer_norm >= result.min_layer_norm
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_gradient_collector_enhanced.py -v`

Expected: FAIL with `ImportError: cannot import name 'GradientHealthMetrics'`

**Step 3: Enhance gradient_collector.py**

Add GradientHealthMetrics dataclass and enhanced collection to `src/esper/simic/gradient_collector.py`:

```python
"""Lightweight Gradient Collector for Seed Telemetry.

Collects gradient statistics for seed parameters without the full
overhead of DiagnosticTracker. Designed for per-epoch collection
during comparison and training loops.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator

import torch
import torch.nn as nn


@dataclass(slots=True)
class GradientHealthMetrics:
    """Enhanced gradient health metrics for Ops Normal monitoring.

    Extends basic gradient stats with layer-wise information
    and numerical stability indicators.
    """

    # Basic stats (existing)
    gradient_norm: float
    gradient_health: float
    has_vanishing: bool
    has_exploding: bool

    # Layer-wise summary
    min_layer_norm: float
    max_layer_norm: float
    norm_ratio: float  # max/min - high ratio indicates imbalance

    # Quality indicators
    zero_grad_fraction: float
    nan_count: int
    inf_count: int

    def is_healthy(
        self,
        max_norm_ratio: float = 1000.0,
        max_zero_fraction: float = 0.5,
    ) -> bool:
        """Check if gradients indicate healthy training.

        Returns:
            True if gradients are healthy
        """
        return (
            self.nan_count == 0
            and self.inf_count == 0
            and self.norm_ratio < max_norm_ratio
            and self.zero_grad_fraction < max_zero_fraction
            and not self.has_exploding
        )

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


# ... existing SeedGradientCollector class ...


def collect_seed_gradients(
    seed_parameters: Iterator[nn.Parameter],
    vanishing_threshold: float = 1e-7,
    exploding_threshold: float = 100.0,
    return_enhanced: bool = False,
) -> dict | GradientHealthMetrics:
    """Convenience function to collect gradient stats.

    Args:
        seed_parameters: Iterator of seed parameters
        vanishing_threshold: Threshold for vanishing detection
        exploding_threshold: Threshold for exploding detection
        return_enhanced: If True, return GradientHealthMetrics dataclass

    Returns:
        Dict with gradient statistics, or GradientHealthMetrics if return_enhanced
    """
    # Convert to list to allow multiple passes
    params = list(seed_parameters)
    grads = [p.grad for p in params if p.grad is not None]

    if not grads:
        if return_enhanced:
            return GradientHealthMetrics(
                gradient_norm=0.0,
                gradient_health=1.0,
                has_vanishing=False,
                has_exploding=False,
                min_layer_norm=0.0,
                max_layer_norm=0.0,
                norm_ratio=1.0,
                zero_grad_fraction=0.0,
                nan_count=0,
                inf_count=0,
            )
        return {
            'gradient_norm': 0.0,
            'gradient_health': 1.0,
            'has_vanishing': False,
            'has_exploding': False,
        }

    # Compute per-layer norms
    per_layer_norms = [g.norm(2).item() for g in grads]
    n_grads = len(grads)

    # Aggregate stats
    total_norm = sum(n**2 for n in per_layer_norms) ** 0.5
    avg_norm = total_norm / n_grads

    min_norm = min(per_layer_norms)
    max_norm = max(per_layer_norms)
    norm_ratio = max_norm / max(min_norm, 1e-10)

    # Count vanishing/exploding
    n_vanishing = sum(1 for n in per_layer_norms if n < vanishing_threshold)
    n_exploding = sum(1 for n in per_layer_norms if n > exploding_threshold)

    # Quality checks
    all_grads = torch.cat([g.view(-1) for g in grads])
    zero_fraction = (all_grads == 0).float().mean().item()
    nan_count = torch.isnan(all_grads).sum().item()
    inf_count = torch.isinf(all_grads).sum().item()

    # Health score
    vanishing_ratio = n_vanishing / n_grads
    exploding_ratio = n_exploding / n_grads
    health = 1.0
    health -= vanishing_ratio * 0.5
    health -= exploding_ratio * 0.8
    health = max(0.0, min(1.0, health))

    if return_enhanced:
        return GradientHealthMetrics(
            gradient_norm=avg_norm,
            gradient_health=health,
            has_vanishing=n_vanishing > 0,
            has_exploding=n_exploding > 0,
            min_layer_norm=min_norm,
            max_layer_norm=max_norm,
            norm_ratio=norm_ratio,
            zero_grad_fraction=zero_fraction,
            nan_count=int(nan_count),
            inf_count=int(inf_count),
        )

    return {
        'gradient_norm': avg_norm,
        'gradient_health': health,
        'has_vanishing': n_vanishing > 0,
        'has_exploding': n_exploding > 0,
    }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_gradient_collector_enhanced.py -v`

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/gradient_collector.py tests/simic/test_gradient_collector_enhanced.py
git commit -m "feat(simic): enhance gradient collector with layer metrics"
```

---

## Phase 4: Debug Mode Telemetry

### Task 9: Add Per-Layer Gradient Statistics

**Files:**
- Create: `src/esper/simic/debug_telemetry.py`
- Test: `tests/simic/test_debug_telemetry.py`

**Step 1: Write the failing test**

Create `tests/simic/test_debug_telemetry.py`:

```python
"""Tests for debug-level telemetry."""

import torch
import torch.nn as nn
import pytest

from esper.simic.debug_telemetry import (
    LayerGradientStats,
    collect_per_layer_gradients,
    NumericalStabilityReport,
    check_numerical_stability,
)


class TestPerLayerGradients:
    """Tests for per-layer gradient collection."""

    @pytest.fixture
    def model_with_grads(self):
        """Create model with gradients."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        return model

    def test_collect_returns_list_of_stats(self, model_with_grads):
        """collect_per_layer_gradients returns list of LayerGradientStats."""
        stats = collect_per_layer_gradients(model_with_grads)
        assert len(stats) > 0
        assert all(isinstance(s, LayerGradientStats) for s in stats)

    def test_stats_have_layer_names(self, model_with_grads):
        """Each stats entry has layer name."""
        stats = collect_per_layer_gradients(model_with_grads)
        names = [s.layer_name for s in stats]
        assert "0.weight" in names or any("weight" in n for n in names)

    def test_stats_have_valid_values(self, model_with_grads):
        """Stats have reasonable gradient values."""
        stats = collect_per_layer_gradients(model_with_grads)
        for s in stats:
            assert s.grad_norm >= 0
            assert 0 <= s.zero_fraction <= 1
            assert s.nan_count == 0  # No NaNs in normal model


class TestNumericalStability:
    """Tests for numerical stability checking."""

    @pytest.fixture
    def healthy_model(self):
        """Create healthy model."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        return model, loss

    def test_healthy_model_passes(self, healthy_model):
        """Healthy model has clean stability report."""
        model, loss = healthy_model
        report = check_numerical_stability(model, loss)

        assert isinstance(report, NumericalStabilityReport)
        assert len(report.nan_in_weights) == 0
        assert len(report.nan_in_gradients) == 0
        assert report.loss_is_finite is True

    def test_detects_nan_in_gradients(self):
        """Detects NaN in gradients."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()

        # Inject NaN
        model.weight.grad[0, 0] = float('nan')

        report = check_numerical_stability(model)
        assert len(report.nan_in_gradients) > 0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_debug_telemetry.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement debug telemetry module**

Create `src/esper/simic/debug_telemetry.py`:

```python
"""Debug-Level Telemetry for Simic Training.

These functions are expensive (5-50ms) and should only be called
when anomalies are detected or debug mode is enabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


@dataclass(slots=True)
class LayerGradientStats:
    """Per-layer gradient statistics for debugging."""

    layer_name: str
    param_count: int

    # Distribution statistics
    grad_norm: float
    grad_mean: float
    grad_std: float
    grad_min: float
    grad_max: float

    # Health indicators
    zero_fraction: float
    small_fraction: float  # < 1e-6
    large_fraction: float  # > 10.0
    nan_count: int
    inf_count: int

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)


def collect_per_layer_gradients(
    model: nn.Module,
    small_threshold: float = 1e-6,
    large_threshold: float = 10.0,
) -> list[LayerGradientStats]:
    """Collect detailed per-layer gradient statistics.

    WARNING: This is expensive (~10-50ms for large models).
    Only use in debug mode.

    Args:
        model: PyTorch model with gradients computed
        small_threshold: Threshold for "small" gradient count
        large_threshold: Threshold for "large" gradient count

    Returns:
        List of LayerGradientStats, one per parameter
    """
    stats = []

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.detach()
        flat = grad.view(-1)

        layer_stats = LayerGradientStats(
            layer_name=name,
            param_count=param.numel(),
            grad_norm=grad.norm().item(),
            grad_mean=flat.mean().item(),
            grad_std=flat.std().item(),
            grad_min=flat.min().item(),
            grad_max=flat.max().item(),
            zero_fraction=(flat == 0).float().mean().item(),
            small_fraction=(flat.abs() < small_threshold).float().mean().item(),
            large_fraction=(flat.abs() > large_threshold).float().mean().item(),
            nan_count=int(torch.isnan(flat).sum().item()),
            inf_count=int(torch.isinf(flat).sum().item()),
        )
        stats.append(layer_stats)

    return stats


@dataclass(slots=True)
class NumericalStabilityReport:
    """Detailed numerical stability analysis for debugging."""

    # NaN/Inf locations
    nan_in_weights: list[str] = field(default_factory=list)
    nan_in_gradients: list[str] = field(default_factory=list)
    inf_in_weights: list[str] = field(default_factory=list)
    inf_in_gradients: list[str] = field(default_factory=list)

    # Value ranges
    max_weight: float = 0.0
    max_gradient: float = 0.0

    # Loss stability
    loss_value: float = 0.0
    loss_is_finite: bool = True

    def has_issues(self) -> bool:
        """Check if any numerical issues detected."""
        return (
            len(self.nan_in_weights) > 0
            or len(self.nan_in_gradients) > 0
            or len(self.inf_in_weights) > 0
            or len(self.inf_in_gradients) > 0
            or not self.loss_is_finite
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)


def check_numerical_stability(
    model: nn.Module,
    loss: torch.Tensor | None = None,
) -> NumericalStabilityReport:
    """Check for numerical stability issues.

    Call after backward() but before optimizer.step() to catch issues
    before they propagate.

    Args:
        model: Model to check
        loss: Optional loss tensor to check

    Returns:
        NumericalStabilityReport
    """
    nan_weights = []
    nan_grads = []
    inf_weights = []
    inf_grads = []
    max_weight = 0.0
    max_grad = 0.0

    for name, param in model.named_parameters():
        # Check weights
        if torch.isnan(param.data).any():
            nan_weights.append(name)
        if torch.isinf(param.data).any():
            inf_weights.append(name)
        max_weight = max(max_weight, param.data.abs().max().item())

        # Check gradients
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if torch.isinf(param.grad).any():
                inf_grads.append(name)
            max_grad = max(max_grad, param.grad.abs().max().item())

    # Check loss
    loss_val = 0.0
    loss_finite = True
    if loss is not None:
        loss_val = loss.item()
        loss_finite = not (math.isnan(loss_val) or math.isinf(loss_val))

    return NumericalStabilityReport(
        nan_in_weights=nan_weights,
        nan_in_gradients=nan_grads,
        inf_in_weights=inf_weights,
        inf_in_gradients=inf_grads,
        max_weight=max_weight,
        max_gradient=max_grad,
        loss_value=loss_val,
        loss_is_finite=loss_finite,
    )


__all__ = [
    "LayerGradientStats",
    "collect_per_layer_gradients",
    "NumericalStabilityReport",
    "check_numerical_stability",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_debug_telemetry.py -v`

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/debug_telemetry.py tests/simic/test_debug_telemetry.py
git commit -m "feat(simic): add per-layer gradient and numerical stability debug telemetry"
```

---

### Task 10: Add Ratio Explosion Diagnostic

**Files:**
- Modify: `src/esper/simic/debug_telemetry.py`
- Test: `tests/simic/test_ratio_explosion.py`

**Step 1: Write the failing test**

Create `tests/simic/test_ratio_explosion.py`:

```python
"""Tests for ratio explosion diagnostics."""

import torch
import pytest

from esper.simic.debug_telemetry import RatioExplosionDiagnostic


class TestRatioExplosionDiagnostic:
    """Tests for RatioExplosionDiagnostic."""

    def test_create_from_tensors(self):
        """Can create diagnostic from tensors."""
        ratio = torch.tensor([0.5, 1.0, 1.5, 6.0, 0.05])
        old_log_probs = torch.tensor([-1.0, -0.5, -0.8, -0.3, -2.0])
        new_log_probs = torch.tensor([-1.2, -0.5, -0.4, 1.5, -5.0])
        states = torch.randn(5, 10)
        actions = torch.tensor([0, 1, 2, 1, 0])
        action_masks = torch.ones(5, 4)

        diag = RatioExplosionDiagnostic.from_batch(
            ratio=ratio,
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            states=states,
            actions=actions,
            action_masks=action_masks,
            max_threshold=5.0,
            min_threshold=0.1,
        )

        assert len(diag.worst_ratio_indices) == 2  # 6.0 > 5.0, 0.05 < 0.1
        assert diag.logit_diff_max > 0

    def test_to_dict_serializable(self):
        """Diagnostic can be serialized to dict."""
        diag = RatioExplosionDiagnostic(
            worst_ratio_indices=[3, 4],
            worst_ratio_values=[6.0, 0.05],
            worst_ratio_actions=[1, 0],
            logit_diff_mean=0.5,
            logit_diff_max=2.0,
        )
        d = diag.to_dict()
        assert "worst_ratio_indices" in d
        assert d["logit_diff_max"] == 2.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_ratio_explosion.py -v`

Expected: FAIL with `ImportError`

**Step 3: Add RatioExplosionDiagnostic to debug_telemetry.py**

Add to `src/esper/simic/debug_telemetry.py`:

```python
@dataclass(slots=True)
class RatioExplosionDiagnostic:
    """Diagnostic data when PPO ratios explode.

    Captures the specific transitions that caused ratio explosion
    for post-hoc debugging.
    """

    # Indices of problematic transitions
    worst_ratio_indices: list[int] = field(default_factory=list)
    worst_ratio_values: list[float] = field(default_factory=list)
    worst_ratio_actions: list[int] = field(default_factory=list)

    # Log prob divergence
    logit_diff_mean: float = 0.0
    logit_diff_max: float = 0.0

    @classmethod
    def from_batch(
        cls,
        ratio: "torch.Tensor",
        old_log_probs: "torch.Tensor",
        new_log_probs: "torch.Tensor",
        states: "torch.Tensor",
        actions: "torch.Tensor",
        action_masks: "torch.Tensor",
        max_threshold: float = 5.0,
        min_threshold: float = 0.1,
    ) -> "RatioExplosionDiagnostic":
        """Create diagnostic from batch tensors.

        Args:
            ratio: PPO ratio tensor [N]
            old_log_probs: Old log probabilities [N]
            new_log_probs: New log probabilities [N]
            states: State observations [N, state_dim]
            actions: Actions taken [N]
            action_masks: Valid action masks [N, action_dim]
            max_threshold: Ratio above this is problematic
            min_threshold: Ratio below this is problematic

        Returns:
            RatioExplosionDiagnostic
        """
        # Find problematic indices
        bad_mask = (ratio > max_threshold) | (ratio < min_threshold)
        bad_indices = bad_mask.nonzero(as_tuple=True)[0].tolist()

        # Extract worst values
        worst_values = ratio[bad_indices].tolist() if bad_indices else []
        worst_actions = actions[bad_indices].tolist() if bad_indices else []

        # Compute log prob divergence
        logit_diff = (new_log_probs - old_log_probs).abs()
        logit_diff_mean = logit_diff.mean().item()
        logit_diff_max = logit_diff.max().item()

        return cls(
            worst_ratio_indices=bad_indices,
            worst_ratio_values=worst_values,
            worst_ratio_actions=worst_actions,
            logit_diff_mean=logit_diff_mean,
            logit_diff_max=logit_diff_max,
        )

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)
```

Update `__all__` to include `RatioExplosionDiagnostic`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_ratio_explosion.py -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/debug_telemetry.py tests/simic/test_ratio_explosion.py
git commit -m "feat(simic): add RatioExplosionDiagnostic for debugging policy divergence"
```

---

## Phase 5: Integration

### Task 11: Wire Telemetry into PPO Update

**Files:**
- Modify: `src/esper/simic/ppo.py`
- Test: `tests/simic/test_ppo_telemetry_integration.py`

**Step 1: Write the failing test**

Create `tests/simic/test_ppo_telemetry_integration.py`:

```python
"""Integration tests for PPO telemetry."""

import torch
import pytest

from esper.simic.ppo import PPOAgent
from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel


class TestPPOTelemetryIntegration:
    """Tests for telemetry integration in PPO."""

    @pytest.fixture
    def agent_with_telemetry(self):
        """Create PPO agent with telemetry config."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_buffer(self, agent_with_telemetry):
        """Fill buffer with transitions."""
        agent = agent_with_telemetry
        for _ in range(20):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            action, log_prob, value, _ = agent.get_action(state, action_mask)
            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0.1,
                done=False,
                action_mask=action_mask,
            )
        return agent.buffer

    def test_update_returns_ppo_health_telemetry(
        self, agent_with_telemetry, filled_buffer
    ):
        """PPO update returns structured telemetry."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should have all Ops Normal metrics
        assert "ratio_mean" in metrics
        assert "ratio_max" in metrics
        assert "explained_variance" in metrics
        assert "policy_loss" in metrics

    def test_debug_level_adds_extra_diagnostics(
        self, agent_with_telemetry, filled_buffer
    ):
        """DEBUG level adds extra diagnostic info."""
        config = TelemetryConfig(level=TelemetryLevel.DEBUG)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should have debug-level metrics
        assert "value_function_telemetry" in metrics or "explained_variance" in metrics
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_ppo_telemetry_integration.py -v`

Expected: FAIL (telemetry_config parameter not accepted)

**Step 3: Modify PPO update to accept telemetry config**

In `src/esper/simic/ppo.py`, modify `update()` signature to accept optional telemetry config:

```python
def update(
    self,
    last_value: float = 0.0,
    clear_buffer: bool = True,
    telemetry_config: "TelemetryConfig | None" = None,
) -> dict:
    """Perform PPO update.

    Args:
        last_value: Value estimate for bootstrapping
        clear_buffer: Whether to clear buffer after update
        telemetry_config: Optional telemetry configuration

    Returns:
        Dictionary of training metrics
    """
    from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel

    if telemetry_config is None:
        telemetry_config = TelemetryConfig(level=TelemetryLevel.NORMAL)

    # ... rest of update logic with telemetry collection ...
```

(Full implementation details follow the pattern established in earlier tasks)

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_ppo_telemetry_integration.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_telemetry_integration.py
git commit -m "feat(simic): integrate telemetry config into PPO update"
```

---

### Task 12: Add Action Distribution Telemetry

**Files:**
- Create: `src/esper/simic/action_telemetry.py`
- Test: `tests/simic/test_action_telemetry.py`

**Step 1: Write the failing test**

Create `tests/simic/test_action_telemetry.py`:

```python
"""Tests for action distribution telemetry."""

import pytest

from esper.simic.action_telemetry import ActionTelemetry


class TestActionTelemetry:
    """Tests for ActionTelemetry tracking."""

    def test_record_action(self):
        """Can record actions and compute stats."""
        telemetry = ActionTelemetry()
        telemetry.record_action("WAIT", success=True)
        telemetry.record_action("WAIT", success=True)
        telemetry.record_action("GERMINATE_CONV", success=True)
        telemetry.record_action("CULL", success=False)

        stats = telemetry.get_stats()
        assert stats["action_counts"]["WAIT"] == 2
        assert stats["action_counts"]["GERMINATE_CONV"] == 1
        assert stats["successful_action_counts"]["WAIT"] == 2
        assert stats["action_success_rate"]["CULL"] == 0.0

    def test_reset(self):
        """Can reset telemetry for new batch."""
        telemetry = ActionTelemetry()
        telemetry.record_action("WAIT", success=True)
        telemetry.reset()

        stats = telemetry.get_stats()
        assert stats["action_counts"] == {}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_action_telemetry.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement ActionTelemetry**

Create `src/esper/simic/action_telemetry.py`:

```python
"""Action Distribution Telemetry.

Tracks action selection patterns for detecting policy issues
like over-conservative behavior or action spam.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ActionTelemetry:
    """Tracks action distribution and success rates.

    Collects per-batch statistics about which actions are taken
    and whether they succeed, to detect policy pathologies.
    """

    _action_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _success_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_action(self, action_name: str, success: bool = True) -> None:
        """Record an action taken.

        Args:
            action_name: Name of action (e.g., "WAIT", "GERMINATE_CONV")
            success: Whether the action succeeded
        """
        self._action_counts[action_name] += 1
        if success:
            self._success_counts[action_name] += 1

    def get_stats(self) -> dict:
        """Get action distribution statistics.

        Returns:
            Dict with action_counts, successful_action_counts, action_success_rate
        """
        action_counts = dict(self._action_counts)
        success_counts = dict(self._success_counts)

        success_rates = {}
        for action, count in action_counts.items():
            if count > 0:
                success_rates[action] = success_counts.get(action, 0) / count
            else:
                success_rates[action] = 0.0

        return {
            "action_counts": action_counts,
            "successful_action_counts": success_counts,
            "action_success_rate": success_rates,
        }

    def reset(self) -> None:
        """Reset counters for new batch/episode."""
        self._action_counts.clear()
        self._success_counts.clear()


__all__ = ["ActionTelemetry"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_action_telemetry.py -v`

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/action_telemetry.py tests/simic/test_action_telemetry.py
git commit -m "feat(simic): add action distribution telemetry"
```

---

### Task 13: Update simic __init__.py Exports

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Check current exports**

Read the file to see current state.

**Step 2: Add new telemetry exports**

Update imports to include all new telemetry modules.

**Step 3: Verify imports work**

Run: `uv run python -c "from esper.simic import TelemetryConfig, PPOHealthTelemetry; print('OK')"`

Expected: Prints `OK`

**Step 4: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "feat(simic): export telemetry modules from package"
```

---

### Task 14: Add CLI Flag for Telemetry Level

**Files:**
- Modify: `src/esper/scripts/train.py`

**Step 1: Add --telemetry-level argument**

Add to the global options section:

```python
parser.add_argument(
    "--telemetry-level",
    type=str,
    choices=["off", "minimal", "normal", "debug"],
    default="normal",
    help="Telemetry verbosity level (default: normal)",
)
```

**Step 2: Wire telemetry level to training**

Create TelemetryConfig from CLI argument and pass to training functions.

**Step 3: Verify with --help**

Run: `uv run python -m esper.scripts.train --help`

Expected: Shows `--telemetry-level` in help

**Step 4: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "feat(cli): add --telemetry-level flag for verbosity control"
```

---

## Summary

After completing all 14 tasks:

**New Files:**
- `src/esper/simic/telemetry_config.py` - TelemetryLevel enum and config
- `src/esper/simic/ppo_telemetry.py` - PPOHealthTelemetry, ValueFunctionTelemetry
- `src/esper/simic/reward_telemetry.py` - RewardComponentsTelemetry
- `src/esper/simic/memory_telemetry.py` - MemoryMetrics
- `src/esper/simic/debug_telemetry.py` - LayerGradientStats, NumericalStabilityReport, RatioExplosionDiagnostic
- `src/esper/simic/action_telemetry.py` - ActionTelemetry

**Modified Files:**
- `src/esper/leyline/telemetry.py` - New event types
- `src/esper/simic/ppo.py` - Ratio stats, explained variance, telemetry integration
- `src/esper/simic/rewards.py` - return_components parameter
- `src/esper/simic/gradient_collector.py` - Enhanced metrics
- `src/esper/scripts/train.py` - --telemetry-level flag

**Usage:**
```bash
# Normal monitoring
PYTHONPATH=src python -m esper.scripts.train ppo \
    --telemetry-level normal \
    --telemetry-dir ./telemetry

# Debug mode (full diagnostics)
PYTHONPATH=src python -m esper.scripts.train ppo \
    --telemetry-level debug \
    --telemetry-dir ./telemetry
```

**Telemetry automatically escalates to DEBUG when anomalies detected:**
- Ratio explosion (ratio > 5.0 or < 0.1)
- Value function collapse (explained_variance < 0.1)
- Memory warning (headroom < 200MB)
- Numerical instability (NaN/Inf detected)
