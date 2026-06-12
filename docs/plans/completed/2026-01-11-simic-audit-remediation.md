# Simic Audit Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Address all actionable findings from the Simic subsystem deep audit, improving testability, maintainability, and observability.

**Architecture:** This plan addresses 7 remediation items in priority order (P0→P2). P0 items are safety-critical tests. P1 items are major refactors to god modules. P2 items are optimizations and observability improvements.

**Tech Stack:** Python 3.11+, PyTorch, pytest, Hypothesis (property tests)

---

## Overview of Remediation Items

| Priority | Item | Category | LOE |
|----------|------|----------|-----|
| **P0** | Governor rollback integration tests | Testing | 1 hour |
| **P1** | Extract action handlers (strategy pattern) | Refactor | 2 hours |
| **P1** | Split vectorized_trainer.py | Refactor | 2 hours |
| **P2** | Normalize optional params at init | Performance | 30 min |
| **P2** | Add Governor rollback telemetry | Observability | 30 min |
| **P2** | Promote LSTM health to normal level | Observability | 15 min |
| **P2** | Add LSTM state recovery tests | Testing | 45 min |

---

## Task 1: Governor Rollback Integration Tests (P0)

**Why:** The Governor's panic→rollback→recovery cycle is critical for training stability but has zero test coverage. A regression could corrupt training state silently.

**Files:**
- Create: `tests/simic/training/test_governor_integration.py`
- Reference: `src/esper/tolaria/governor.py`
- Reference: `src/esper/simic/training/action_execution.py:381-404`

### Step 1.1: Write test scaffold

Create the test file with imports and fixtures:

```python
"""Integration tests for Governor rollback mechanism.

Tests the full panic → rollback → optimizer clear → training continues cycle.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from esper.tolaria.governor import TolariaGovernor
from esper.leyline import SeedStage, DEFAULT_GOVERNOR_DEATH_PENALTY


class SimpleModel(nn.Module):
    """Minimal model for Governor testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.seed_slots = {}  # Empty slots for filtering test

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def model() -> SimpleModel:
    return SimpleModel()


@pytest.fixture
def governor(model: SimpleModel) -> TolariaGovernor:
    return TolariaGovernor(
        model=model,
        sensitivity=3.0,  # Lower for faster triggering in tests
        history_window=10,
        min_panics_before_rollback=2,
    )


@pytest.fixture
def optimizer(model: SimpleModel) -> torch.optim.SGD:
    return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Step 1.2: Write test for NaN detection and panic state

```python
class TestGovernorPanicDetection:
    """Tests for Governor panic trigger conditions."""

    def test_nan_loss_triggers_immediate_panic(self, governor: TolariaGovernor):
        """NaN loss should trigger panic without history buildup."""
        # Warm up history with valid losses
        for _ in range(5):
            governor.check_vital_signs(1.0)

        # NaN should trigger panic
        result = governor.check_vital_signs(float('nan'))

        assert result is True, "NaN loss should trigger panic"
        assert governor._pending_panic is True
        assert governor._panic_reason == "nan_loss"

    def test_inf_loss_triggers_immediate_panic(self, governor: TolariaGovernor):
        """Inf loss should trigger panic without history buildup."""
        for _ in range(5):
            governor.check_vital_signs(1.0)

        result = governor.check_vital_signs(float('inf'))

        assert result is True
        assert governor._pending_panic is True

    def test_normal_loss_no_panic(self, governor: TolariaGovernor):
        """Normal training losses should not trigger panic."""
        for _ in range(10):
            result = governor.check_vital_signs(1.0 + 0.1 * _)

        assert result is False
        assert governor._pending_panic is False
```

### Step 1.3: Run tests to verify they pass

Run: `uv run pytest tests/simic/training/test_governor_integration.py::TestGovernorPanicDetection -v`

Expected: PASS (testing existing Governor functionality)

### Step 1.4: Write test for rollback and state restoration

```python
class TestGovernorRollback:
    """Tests for Governor rollback mechanism."""

    def test_rollback_restores_model_weights(
        self, model: SimpleModel, governor: TolariaGovernor
    ):
        """Rollback should restore model to last known good state."""
        # Capture original weights
        original_weight = model.linear.weight.clone()

        # Take snapshot
        governor.snapshot()

        # Corrupt weights
        with torch.no_grad():
            model.linear.weight.fill_(999.0)

        assert not torch.allclose(model.linear.weight, original_weight)

        # Execute rollback
        governor.execute_rollback()

        # Weights should be restored
        assert torch.allclose(model.linear.weight, original_weight)

    def test_rollback_clears_optimizer_momentum(
        self, model: SimpleModel, governor: TolariaGovernor, optimizer: torch.optim.SGD
    ):
        """Rollback should trigger optimizer state clear (caller responsibility)."""
        # Do a training step to build momentum
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Verify momentum exists
        param = list(model.parameters())[0]
        assert 'momentum_buffer' in optimizer.state[param]

        # Simulate rollback cycle (as done in action_execution.py:381-404)
        governor.snapshot()
        governor._pending_panic = True
        governor.execute_rollback()

        # Caller must clear optimizer state
        optimizer.state.clear()

        # Verify momentum cleared
        assert param not in optimizer.state or 'momentum_buffer' not in optimizer.state.get(param, {})
```

### Step 1.5: Write test for full panic→rollback→continue cycle

```python
class TestGovernorFullCycle:
    """Integration tests for complete panic→rollback→recovery cycle."""

    def test_full_panic_rollback_recovery_cycle(
        self, model: SimpleModel, governor: TolariaGovernor, optimizer: torch.optim.SGD
    ):
        """Full cycle: train → panic → rollback → resume training."""
        # Phase 1: Normal training
        for step in range(5):
            x = torch.randn(2, 10)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            governor.check_vital_signs(loss.item())

        # Snapshot known good state
        governor.snapshot()
        good_weight = model.linear.weight.clone()

        # Phase 2: Corrupt training (simulate catastrophic failure)
        with torch.no_grad():
            model.linear.weight.fill_(float('nan'))

        # Phase 3: Detect panic
        is_panic = governor.check_vital_signs(float('nan'))
        assert is_panic is True

        # Phase 4: Execute rollback (as in action_execution.py)
        governor.execute_rollback()
        optimizer.state.clear()

        # Phase 5: Verify recovery
        assert torch.allclose(model.linear.weight, good_weight)
        assert governor._pending_panic is False

        # Phase 6: Resume training (should work normally)
        for step in range(3):
            x = torch.randn(2, 10)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            is_panic = governor.check_vital_signs(loss.item())
            assert is_panic is False

    def test_death_penalty_returned_on_rollback(
        self, model: SimpleModel, governor: TolariaGovernor
    ):
        """Rollback should return death penalty for PPO buffer injection."""
        governor.snapshot()

        # Trigger panic
        governor.check_vital_signs(float('nan'))

        # Get report (includes penalty)
        report = governor.execute_rollback()

        assert report.penalty == DEFAULT_GOVERNOR_DEATH_PENALTY
        assert report.rolled_back is True
```

### Step 1.6: Run all Governor tests

Run: `uv run pytest tests/simic/training/test_governor_integration.py -v`

Expected: All PASS

### Step 1.7: Commit

```bash
git add tests/simic/training/test_governor_integration.py
git commit -m "test(simic): add Governor rollback integration tests

Addresses P0 finding from Simic audit - the panic→rollback→recovery
cycle was untested. Tests verify:
- NaN/Inf detection triggers panic
- Rollback restores model weights
- Optimizer momentum must be cleared by caller
- Full cycle: train → panic → rollback → resume

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Extract Action Handlers (P1)

**Why:** `action_execution.py` (1,262 lines) handles 7 lifecycle operations in deep conditionals. Strategy pattern improves testability and reduces cognitive load.

**Files:**
- Create: `src/esper/simic/training/handlers/__init__.py`
- Create: `src/esper/simic/training/handlers/base.py`
- Create: `src/esper/simic/training/handlers/germinate.py`
- Create: `src/esper/simic/training/handlers/advance.py`
- Create: `src/esper/simic/training/handlers/fossilize.py`
- Create: `src/esper/simic/training/handlers/prune.py`
- Create: `src/esper/simic/training/handlers/wait.py`
- Create: `src/esper/simic/training/handlers/alpha.py`
- Modify: `src/esper/simic/training/action_execution.py`
- Create: `tests/simic/training/handlers/test_germinate_handler.py`

### Step 2.1: Create handler base protocol

```python
# src/esper/simic/training/handlers/base.py
"""Base protocol for lifecycle operation handlers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from esper.leyline import SeedSlotProtocol, SlottedHostProtocol
    from esper.simic.training.parallel_env_state import ParallelEnvState


@dataclass
class HandlerContext:
    """Shared context for all lifecycle handlers."""
    env_idx: int
    slot_id: str
    env_state: "ParallelEnvState"
    model: "SlottedHostProtocol"
    slot: "SeedSlotProtocol"
    epoch: int
    max_epochs: int


@dataclass
class HandlerResult:
    """Result from executing a lifecycle handler."""
    success: bool
    reward_modifier: float = 0.0
    needs_snapshot: bool = False
    telemetry: dict[str, Any] | None = None
    error: str | None = None


class LifecycleHandler(Protocol):
    """Protocol for lifecycle operation handlers."""

    def can_execute(self, ctx: HandlerContext) -> bool:
        """Check if this handler can execute given current state."""
        ...

    def execute(self, ctx: HandlerContext, **kwargs: Any) -> HandlerResult:
        """Execute the lifecycle operation."""
        ...
```

### Step 2.2: Create handlers __init__.py

```python
# src/esper/simic/training/handlers/__init__.py
"""Lifecycle operation handlers for action execution.

Each handler encapsulates one lifecycle operation (GERMINATE, ADVANCE, etc.)
following the Strategy pattern for testability and maintainability.
"""
from esper.simic.training.handlers.base import (
    HandlerContext,
    HandlerResult,
    LifecycleHandler,
)

__all__ = [
    "HandlerContext",
    "HandlerResult",
    "LifecycleHandler",
]
```

### Step 2.3: Create germinate handler (example)

```python
# src/esper/simic/training/handlers/germinate.py
"""Handler for GERMINATE lifecycle operation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from esper.leyline import BLUEPRINT_IDS, SeedStage
from esper.simic.training.handlers.base import HandlerContext, HandlerResult

if TYPE_CHECKING:
    from esper.leyline import SlotConfig


def can_germinate(ctx: HandlerContext) -> bool:
    """Check if germination is valid for this slot."""
    return ctx.slot.state is None


def execute_germinate(
    ctx: HandlerContext,
    *,
    blueprint_idx: int,
    style_idx: int,
    tempo_idx: int,
    alpha_target_idx: int,
    alpha_speed_idx: int,
    alpha_curve_idx: int,
    slot_config: "SlotConfig",
) -> HandlerResult:
    """Execute GERMINATE operation on a slot.

    Creates a new seed module in the specified slot with the given
    blueprint and alpha blending configuration.
    """
    if not can_germinate(ctx):
        return HandlerResult(
            success=False,
            error=f"Slot {ctx.slot_id} already occupied"
        )

    blueprint_id = BLUEPRINT_IDS[blueprint_idx]

    # Get blueprint-specific config
    blueprint_config = slot_config.get_blueprint_config(blueprint_id)

    # Create seed via slot interface
    ctx.slot.germinate(
        blueprint_id=blueprint_id,
        host=ctx.model.host,
        blueprint_config=blueprint_config,
    )

    return HandlerResult(
        success=True,
        telemetry={
            "action": "germinate",
            "slot_id": ctx.slot_id,
            "blueprint_id": blueprint_id,
            "stage": SeedStage.GERMINATED.name,
        }
    )
```

### Step 2.4: Write test for germinate handler

```python
# tests/simic/training/handlers/test_germinate_handler.py
"""Tests for germinate lifecycle handler."""
from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from esper.simic.training.handlers.base import HandlerContext
from esper.simic.training.handlers.germinate import can_germinate, execute_germinate


@pytest.fixture
def empty_slot() -> MagicMock:
    """Slot with no active seed."""
    slot = MagicMock()
    slot.state = None
    slot.germinate = MagicMock()
    return slot


@pytest.fixture
def occupied_slot() -> MagicMock:
    """Slot with active seed."""
    slot = MagicMock()
    slot.state = MagicMock()  # Non-None = occupied
    return slot


@pytest.fixture
def ctx_factory():
    """Factory for creating handler contexts."""
    def _make_ctx(slot: MagicMock) -> HandlerContext:
        model = MagicMock()
        model.host = MagicMock()
        return HandlerContext(
            env_idx=0,
            slot_id="r0c0",
            env_state=MagicMock(),
            model=model,
            slot=slot,
            epoch=5,
            max_epochs=100,
        )
    return _make_ctx


class TestCanGerminate:
    def test_empty_slot_can_germinate(self, empty_slot, ctx_factory):
        ctx = ctx_factory(empty_slot)
        assert can_germinate(ctx) is True

    def test_occupied_slot_cannot_germinate(self, occupied_slot, ctx_factory):
        ctx = ctx_factory(occupied_slot)
        assert can_germinate(ctx) is False


class TestExecuteGerminate:
    def test_germinate_empty_slot_succeeds(self, empty_slot, ctx_factory):
        ctx = ctx_factory(empty_slot)
        slot_config = MagicMock()
        slot_config.get_blueprint_config = MagicMock(return_value={})

        result = execute_germinate(
            ctx,
            blueprint_idx=0,
            style_idx=0,
            tempo_idx=0,
            alpha_target_idx=0,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            slot_config=slot_config,
        )

        assert result.success is True
        assert result.error is None
        empty_slot.germinate.assert_called_once()

    def test_germinate_occupied_slot_fails(self, occupied_slot, ctx_factory):
        ctx = ctx_factory(occupied_slot)
        slot_config = MagicMock()

        result = execute_germinate(
            ctx,
            blueprint_idx=0,
            style_idx=0,
            tempo_idx=0,
            alpha_target_idx=0,
            alpha_speed_idx=0,
            alpha_curve_idx=0,
            slot_config=slot_config,
        )

        assert result.success is False
        assert "occupied" in result.error.lower()
```

### Step 2.5: Run handler tests

Run: `uv run pytest tests/simic/training/handlers/test_germinate_handler.py -v`

Expected: PASS

### Step 2.6: Create remaining handlers

Repeat the pattern for:
- `advance.py` - ADVANCE operation (G1→G2→G3 transitions)
- `fossilize.py` - FOSSILIZE operation (HOLDING→FOSSILIZED)
- `prune.py` - PRUNE operation (any stage→PRUNED)
- `wait.py` - WAIT operation (no-op)
- `alpha.py` - SET_ALPHA_TARGET operation (alpha curve adjustment)

Each handler follows the same structure:
1. `can_<operation>(ctx: HandlerContext) -> bool`
2. `execute_<operation>(ctx: HandlerContext, **kwargs) -> HandlerResult`

### Step 2.7: Create handler registry

```python
# src/esper/simic/training/handlers/registry.py
"""Registry mapping LifecycleOp to handlers."""
from __future__ import annotations

from esper.leyline import LifecycleOp, OP_GERMINATE, OP_ADVANCE, OP_FOSSILIZE, OP_PRUNE, OP_WAIT, OP_SET_ALPHA_TARGET

from esper.simic.training.handlers.germinate import execute_germinate
from esper.simic.training.handlers.advance import execute_advance
from esper.simic.training.handlers.fossilize import execute_fossilize
from esper.simic.training.handlers.prune import execute_prune
from esper.simic.training.handlers.wait import execute_wait
from esper.simic.training.handlers.alpha import execute_set_alpha_target

HANDLER_REGISTRY = {
    OP_GERMINATE: execute_germinate,
    OP_ADVANCE: execute_advance,
    OP_FOSSILIZE: execute_fossilize,
    OP_PRUNE: execute_prune,
    OP_WAIT: execute_wait,
    OP_SET_ALPHA_TARGET: execute_set_alpha_target,
}
```

### Step 2.8: Update action_execution.py to use handlers

Modify `execute_action_on_slot()` to dispatch to handlers:

```python
# In action_execution.py, replace the large if/elif chain with:
from esper.simic.training.handlers.registry import HANDLER_REGISTRY

def execute_action_on_slot(
    # ... existing params ...
) -> ActionOutcome:
    ctx = HandlerContext(
        env_idx=env_idx,
        slot_id=target_slot,
        env_state=env_state,
        model=model,
        slot=slot,
        epoch=epoch,
        max_epochs=max_epochs,
    )

    handler = HANDLER_REGISTRY.get(op_idx)
    if handler is None:
        raise ValueError(f"Unknown operation: {op_idx}")

    result = handler(ctx, **handler_kwargs)

    # Convert HandlerResult to ActionOutcome
    return ActionOutcome(
        success=result.success,
        reward_modifier=result.reward_modifier,
        # ... map other fields ...
    )
```

### Step 2.9: Commit

```bash
git add src/esper/simic/training/handlers/
git add tests/simic/training/handlers/
git add src/esper/simic/training/action_execution.py
git commit -m "refactor(simic): extract action handlers using strategy pattern

Addresses P1 finding from Simic audit - action_execution.py (1262 lines)
had 7 lifecycle operations in deep conditionals.

- Create handlers/ subpackage with one module per operation
- Define HandlerContext and HandlerResult for consistent interface
- Add handler registry for op dispatch
- Update action_execution.py to use registry
- Add tests for germinate handler

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Split vectorized_trainer.py (P1)

**Why:** At 1,856 lines with a 500+ line `run()` method, this god module is difficult to debug and test. Extract into focused components.

**Files:**
- Create: `src/esper/simic/training/epoch_runner.py`
- Create: `src/esper/simic/training/ppo_coordinator.py`
- Modify: `src/esper/simic/training/vectorized_trainer.py`

### Step 3.1: Extract EpochRunner

Create a class that handles the inner epoch loop:

```python
# src/esper/simic/training/epoch_runner.py
"""Epoch execution logic extracted from VectorizedPPOTrainer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from esper.simic.training.parallel_env_state import ParallelEnvState
    from esper.simic.training.batch_ops import BatchSummary


@dataclass
class EpochResult:
    """Result from executing one epoch across all environments."""
    batch_summaries: list["BatchSummary"]
    epoch_metrics: dict[str, float]
    any_episode_done: bool
    any_rollback: bool


class EpochRunner:
    """Executes training epochs across vectorized environments.

    Extracted from VectorizedPPOTrainer.run() to reduce method size
    and improve testability.
    """

    def __init__(
        self,
        env_states: list["ParallelEnvState"],
        action_context: Any,  # ActionExecutionContext
        batch_processor: Any,  # Callable for process_train_batch
        emitter: Any,
    ):
        self.env_states = env_states
        self.action_context = action_context
        self.batch_processor = batch_processor
        self.emitter = emitter

    def run_epoch(
        self,
        epoch: int,
        batch_idx: int,
        shared_batch: tuple[torch.Tensor, torch.Tensor],
    ) -> EpochResult:
        """Execute one epoch of training across all environments.

        This is the inner loop extracted from VectorizedPPOTrainer.run().
        """
        batch_summaries = []
        any_episode_done = False
        any_rollback = False

        for env_idx, env_state in enumerate(self.env_states):
            # Process batch for this environment
            summary = self.batch_processor(
                env_idx=env_idx,
                env_state=env_state,
                batch=shared_batch,
                epoch=epoch,
            )
            batch_summaries.append(summary)

            if summary.episode_done:
                any_episode_done = True
            if summary.rollback_occurred:
                any_rollback = True

        return EpochResult(
            batch_summaries=batch_summaries,
            epoch_metrics=self._aggregate_metrics(batch_summaries),
            any_episode_done=any_episode_done,
            any_rollback=any_rollback,
        )

    def _aggregate_metrics(
        self, summaries: list["BatchSummary"]
    ) -> dict[str, float]:
        """Aggregate metrics across environments."""
        if not summaries:
            return {}

        return {
            "mean_loss": sum(s.loss for s in summaries) / len(summaries),
            "mean_accuracy": sum(s.accuracy for s in summaries) / len(summaries),
        }
```

### Step 3.2: Extract PPOCoordinator

Create a class that handles PPO update orchestration:

```python
# src/esper/simic/training/ppo_coordinator.py
"""PPO update coordination extracted from VectorizedPPOTrainer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from esper.simic.agent import PPOAgent


@dataclass
class PPOUpdateResult:
    """Result from PPO update phase."""
    metrics: dict[str, Any]
    early_stopped: bool
    epochs_completed: int


class PPOCoordinator:
    """Coordinates PPO updates after rollout collection.

    Extracted from VectorizedPPOTrainer to separate concerns:
    - EpochRunner: environment stepping
    - PPOCoordinator: policy optimization
    """

    def __init__(
        self,
        agent: "PPOAgent",
        ppo_epochs: int,
        telemetry_config: Any,
    ):
        self.agent = agent
        self.ppo_epochs = ppo_epochs
        self.telemetry_config = telemetry_config

    def run_ppo_updates(
        self,
        batch_idx: int,
    ) -> PPOUpdateResult:
        """Execute PPO updates on collected rollout data.

        Returns early if KL divergence exceeds threshold.
        """
        all_metrics = []
        early_stopped = False
        epochs_completed = 0

        for ppo_epoch in range(self.ppo_epochs):
            metrics = self.agent.update()
            all_metrics.append(metrics)
            epochs_completed += 1

            if metrics.get("early_stop", False):
                early_stopped = True
                break

        return PPOUpdateResult(
            metrics=self._aggregate_ppo_metrics(all_metrics),
            early_stopped=early_stopped,
            epochs_completed=epochs_completed,
        )

    def _aggregate_ppo_metrics(
        self, metrics_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate metrics across PPO epochs."""
        if not metrics_list:
            return {}

        # Average scalar metrics
        aggregated = {}
        for key in metrics_list[0]:
            values = [m[key] for m in metrics_list if key in m]
            if values and isinstance(values[0], (int, float)):
                aggregated[key] = sum(values) / len(values)
            else:
                aggregated[key] = values[-1]  # Take last for non-scalar

        return aggregated
```

### Step 3.3: Update vectorized_trainer.py to use extracted components

Modify `VectorizedPPOTrainer.run()` to delegate to `EpochRunner` and `PPOCoordinator`:

```python
# At top of vectorized_trainer.py
from esper.simic.training.epoch_runner import EpochRunner, EpochResult
from esper.simic.training.ppo_coordinator import PPOCoordinator, PPOUpdateResult

# In VectorizedPPOTrainer.__post_init__():
self.epoch_runner = EpochRunner(
    env_states=self.env_states,
    action_context=self.action_execution_context,
    batch_processor=self._process_train_batch,
    emitter=self.batch_emitter,
)
self.ppo_coordinator = PPOCoordinator(
    agent=self.agent,
    ppo_epochs=self.ppo_updates_per_batch,
    telemetry_config=self.telemetry_config,
)

# In run() method, replace inline epoch loop with:
epoch_result = self.epoch_runner.run_epoch(
    epoch=epoch,
    batch_idx=batch_idx,
    shared_batch=(x_batch, y_batch),
)

# Replace inline PPO update with:
if should_update_ppo:
    ppo_result = self.ppo_coordinator.run_ppo_updates(batch_idx)
```

### Step 3.4: Run existing vectorized tests

Run: `uv run pytest tests/simic/test_vectorized.py -v`

Expected: PASS (refactor should not change behavior)

### Step 3.5: Commit

```bash
git add src/esper/simic/training/epoch_runner.py
git add src/esper/simic/training/ppo_coordinator.py
git add src/esper/simic/training/vectorized_trainer.py
git commit -m "refactor(simic): split vectorized_trainer.py into components

Addresses P1 finding from Simic audit - vectorized_trainer.py (1856 lines)
had a 500+ line run() method.

- Extract EpochRunner for inner epoch loop
- Extract PPOCoordinator for PPO update orchestration
- VectorizedPPOTrainer now delegates to these components
- No behavioral changes, existing tests pass

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Normalize Optional Params at Init (P2)

**Why:** PyTorch Engineer identified that conditional branches (`if forced_mask is not None`) cause torch.compile graph specialization. Normalizing at init time creates consistent graph shapes.

**Files:**
- Modify: `src/esper/simic/agent/ppo_agent.py`
- Modify: `src/esper/simic/agent/ppo_update.py`

### Step 4.1: Normalize entropy_floor at PPOAgent init

In `ppo_agent.py`, ensure entropy_floor is always a dict:

```python
# In PPOAgent.__init__(), after receiving entropy_floor parameter:
# Normalize entropy_floor to dict (avoid isinstance in hot path)
if entropy_floor is None:
    self._entropy_floor: dict[str, float] = {}
elif isinstance(entropy_floor, dict):
    self._entropy_floor = entropy_floor
else:
    # Scalar applies to all heads
    self._entropy_floor = {head: entropy_floor for head in HEAD_NAMES}
```

### Step 4.2: Normalize entropy_floor_penalty_coef

```python
# Similarly for penalty coefficient:
if entropy_floor_penalty_coef is None:
    self._entropy_floor_penalty_coef: dict[str, float] = {}
elif isinstance(entropy_floor_penalty_coef, dict):
    self._entropy_floor_penalty_coef = entropy_floor_penalty_coef
else:
    self._entropy_floor_penalty_coef = {head: entropy_floor_penalty_coef for head in HEAD_NAMES}
```

### Step 4.3: Update ppo_update.py to expect dict

Remove isinstance checks in `compute_entropy_floor_penalty()`:

```python
# In ppo_update.py compute_entropy_floor_penalty():
# Remove this block:
# if isinstance(entropy_floor_penalty_coef, dict):
#     penalty_coef_dict = entropy_floor_penalty_coef
# else:
#     penalty_coef_dict = {head: entropy_floor_penalty_coef for head in entropy_floor}

# Replace with direct usage (caller guarantees dict):
penalty_coef_dict = penalty_coef  # Already a dict from PPOAgent
```

### Step 4.4: Run PPO tests

Run: `uv run pytest tests/simic/agent/ -v`

Expected: PASS

### Step 4.5: Commit

```bash
git add src/esper/simic/agent/ppo_agent.py
git add src/esper/simic/agent/ppo_update.py
git commit -m "perf(simic): normalize optional params at init for torch.compile

Addresses P2 finding from Simic audit - isinstance() checks in hot path
cause torch.compile graph specialization.

- Normalize entropy_floor to dict in PPOAgent.__init__()
- Normalize entropy_floor_penalty_coef to dict at init
- Remove isinstance() checks from compute_entropy_floor_penalty()
- Consistent graph shape for torch.compile

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Governor Rollback Telemetry (P2)

**Why:** Governor rollbacks are functional but not observable in Karn analytics. Explicit telemetry events enable debugging.

**Files:**
- Modify: `src/esper/simic/training/action_execution.py`
- Reference: `src/esper/leyline/telemetry.py` (GovernorRollbackPayload already exists)

### Step 5.1: Emit telemetry on rollback

In `action_execution.py`, after rollback execution:

```python
# After line 404 (governor.execute_rollback()):
from esper.leyline import TelemetryEvent, TelemetryEventType, GovernorRollbackPayload

# Emit telemetry event for observability
rollback_event = TelemetryEvent(
    event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
    data=GovernorRollbackPayload(
        env_idx=env_idx,
        panic_reason=env_state.governor._panic_reason or "unknown",
        panic_loss=env_state.governor._panic_loss,
        penalty=report.penalty,
        epoch=epoch,
    ),
)
emitter.emit(rollback_event)
```

### Step 5.2: Run tests

Run: `uv run pytest tests/simic/training/ -v`

Expected: PASS

### Step 5.3: Commit

```bash
git add src/esper/simic/training/action_execution.py
git commit -m "feat(simic): emit telemetry on Governor rollback

Addresses P2 finding from Simic audit - Governor rollbacks were
functional but not observable in Karn analytics.

- Emit GovernorRollbackPayload when rollback executes
- Includes panic_reason, panic_loss, penalty, epoch
- Enables debugging via Karn dashboard

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Promote LSTM Health to Normal Level (P2)

**Why:** LSTM health metrics (h_rms, c_rms bounds) are only available at `debug` telemetry level but are useful for detecting training issues early.

**Files:**
- Modify: `src/esper/simic/telemetry/telemetry_config.py`
- Modify: `src/esper/simic/training/vectorized_trainer.py`

### Step 6.1: Add lstm_health to normal level

In `telemetry_config.py`, update the level definitions:

```python
# In should_collect() or similar:
NORMAL_LEVEL_METRICS = {
    "epoch_metrics",
    "ppo_update",
    "seed_lifecycle",
    "lstm_health",  # PROMOTED from debug
}
```

### Step 6.2: Update vectorized_trainer to collect at normal

In `vectorized_trainer.py`, change the lstm_health collection guard:

```python
# Change from:
if telemetry_config.level == TelemetryLevel.DEBUG:
    lstm_metrics = compute_lstm_health(...)

# To:
if telemetry_config.level >= TelemetryLevel.NORMAL:
    lstm_metrics = compute_lstm_health(...)
```

### Step 6.3: Commit

```bash
git add src/esper/simic/telemetry/telemetry_config.py
git add src/esper/simic/training/vectorized_trainer.py
git commit -m "feat(simic): promote LSTM health to normal telemetry level

Addresses P2 finding from Simic audit - LSTM health metrics were
only at debug level but useful for early detection of issues.

- Add lstm_health to NORMAL_LEVEL_METRICS
- Collect compute_lstm_health() at normal level
- Enables earlier detection of hidden state saturation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add LSTM State Recovery Tests (P2)

**Why:** LSTM hidden state recovery after rollback is untested. A bug here would cause subtle policy divergence.

**Files:**
- Create: `tests/simic/agent/test_lstm_state_recovery.py`

### Step 7.1: Write LSTM recovery test

```python
"""Tests for LSTM hidden state recovery after rollback."""
from __future__ import annotations

import pytest
import torch

from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic


@pytest.fixture
def policy() -> FactoredRecurrentActorCritic:
    """Create policy for testing."""
    return FactoredRecurrentActorCritic(
        obs_dim=128,
        hidden_dim=64,
        n_slots=3,
    )


class TestLSTMStateRecovery:
    """Tests for LSTM hidden state management during rollback."""

    def test_hidden_state_reset_clears_memory(self, policy):
        """Resetting hidden state should clear LSTM memory."""
        # Run forward to build hidden state
        obs = torch.randn(1, 128)
        _, hidden = policy(obs, hidden=None)

        assert hidden[0].abs().sum() > 0, "Hidden state should be non-zero"

        # Reset
        policy.reset_hidden()
        fresh_hidden = policy.get_initial_hidden(batch_size=1)

        # Fresh hidden should be zeros
        assert fresh_hidden[0].abs().sum() == 0

    def test_hidden_state_detach_breaks_gradient(self, policy):
        """Detaching hidden state should break gradient graph."""
        obs = torch.randn(1, 128)
        _, hidden = policy(obs, hidden=None)

        # Detach
        detached_h = hidden[0].detach()
        detached_c = hidden[1].detach()

        assert not detached_h.requires_grad
        assert not detached_c.requires_grad

    def test_hidden_state_after_rollback_scenario(self, policy):
        """Simulate rollback: hidden state should be reset, not restored."""
        # This tests the contract: after rollback, we reset hidden state
        # rather than trying to restore old LSTM memory

        # Build up hidden state
        obs = torch.randn(1, 128)
        for _ in range(10):
            _, hidden = policy(obs, hidden=hidden if 'hidden' in dir() else None)

        pre_rollback_h = hidden[0].clone()

        # Simulate rollback (reset hidden state)
        fresh_hidden = policy.get_initial_hidden(batch_size=1)

        # Verify reset, not restore
        assert not torch.allclose(fresh_hidden[0], pre_rollback_h)
        assert fresh_hidden[0].abs().sum() == 0
```

### Step 7.2: Run tests

Run: `uv run pytest tests/simic/agent/test_lstm_state_recovery.py -v`

Expected: PASS

### Step 7.3: Commit

```bash
git add tests/simic/agent/test_lstm_state_recovery.py
git commit -m "test(simic): add LSTM state recovery tests

Addresses P2 finding from Simic audit - LSTM hidden state
recovery after rollback was untested.

- Test hidden state reset clears memory
- Test detach breaks gradient graph
- Test rollback scenario uses reset, not restore

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

After completing all tasks, run the full test suite:

```bash
# Run all simic tests
uv run pytest tests/simic/ -v --tb=short

# Run property tests
uv run pytest tests/simic/properties/ -v

# Run integration tests
uv run pytest tests/simic/agent/test_ppo_policy_integration.py -v
```

All tests should pass. If any fail, debug before marking complete.

---

## Summary

This plan addresses 7 remediation items from the Simic audit:

| # | Task | Status |
|---|------|--------|
| 1 | Governor rollback integration tests | P0 |
| 2 | Extract action handlers | P1 |
| 3 | Split vectorized_trainer.py | P1 |
| 4 | Normalize optional params | P2 |
| 5 | Governor rollback telemetry | P2 |
| 6 | Promote LSTM health level | P2 |
| 7 | LSTM state recovery tests | P2 |

Total estimated time: ~7 hours

Each task is designed to be independently commitable and testable.
