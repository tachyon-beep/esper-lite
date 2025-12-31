# Vectorized.py Refactoring - Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `vectorized.py` from 2,740 lines to ~2,121 lines (-23%) by extracting three cohesive modules with zero/low risk.

**Architecture:** Extract pure helper functions and data structures that have no circular dependencies on the main training loop. The main `train_ppo_vectorized()` function remains in place but becomes easier to read.

**Tech Stack:** Python dataclasses, PyTorch

---

## Background

`vectorized.py` is currently 2,740 lines containing:
- 390 lines of telemetry emission helpers (13 functions)
- 145 lines of seed management helpers (7 functions)
- 84 lines of `ParallelEnvState` dataclass
- 1,959 lines of `train_ppo_vectorized()` function

Phase 1 extracts the first three clusters, which are:
- **Stateless** (telemetry emitters are pure functions)
- **Self-contained** (ParallelEnvState has no logic dependencies)
- **One-way dependent** (seed management helpers are called, never call back)

---

## Task 1: Extract Telemetry Emitters

**Files:**
- Create: `src/esper/simic/telemetry/emitters.py`
- Modify: `src/esper/simic/vectorized.py`
- Modify: `src/esper/simic/telemetry/__init__.py` (if exists, or create)

**Step 1: Create the telemetry subdirectory if needed**

Check if `src/esper/simic/telemetry/` exists. If not, create it with `__init__.py`.

**Step 2: Create emitters.py with all telemetry functions**

Move these functions from `vectorized.py` (lines ~98-490):

```python
"""Telemetry emission helpers for vectorized PPO training.

These are pure functions that format and emit telemetry events.
They do not modify training state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.simic.debug_telemetry import LayerGradientStats
from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel

if TYPE_CHECKING:
    from esper.nissa import TelemetryHub
    from esper.simic.anomaly_detector import AnomalyReport
    from esper.simic.reward_telemetry import RewardComponentsTelemetry


def emit_with_env_context(
    hub: "TelemetryHub",
    event: TelemetryEvent,
    env_id: int,
    device: str,
) -> None:
    """Inject environment context into telemetry event and emit."""
    # ... move implementation from vectorized.py


def emit_batch_completed(
    hub: "TelemetryHub",
    batch_idx: int,
    episodes_completed: int,
    avg_reward: float,
    avg_accuracy: float,
    avg_val_accuracy: float,
    learning_rate: float,
    entropy_coef: float,
    seed_stage: str,
    active_slot: str | None,
    telemetry_level: TelemetryLevel,
) -> None:
    """Emit batch completion snapshot."""
    # ... move implementation


def emit_last_action(
    hub: "TelemetryHub",
    env_id: int,
    device: str,
    action_type: str,
    target_slot: str | None,
    reward: float,
    cumulative_reward: float,
    action_mask: dict[str, bool],
    telemetry_level: TelemetryLevel,
) -> None:
    """Emit per-step action debug telemetry."""
    # ... move implementation


def compute_grad_norm_surrogate(model: nn.Module) -> float:
    """Compute gradient norm without expensive all-reduce."""
    # ... move implementation


def aggregate_layer_gradient_health(
    layer_stats: list[LayerGradientStats],
) -> dict[str, float]:
    """Summarize per-layer gradient statistics."""
    # ... move implementation


def emit_ppo_update_event(
    hub: "TelemetryHub",
    ppo_metrics: dict,
    layer_stats: list[LayerGradientStats] | None,
    telemetry_level: TelemetryLevel,
) -> None:
    """Emit PPO update metrics."""
    # ... move implementation


def emit_action_distribution(
    hub: "TelemetryHub",
    action_counts: dict[str, int],
    action_successes: dict[str, int],
) -> None:
    """Emit action success distribution."""
    # ... move implementation


def emit_cf_unavailable(
    hub: "TelemetryHub",
    env_id: int,
    device: str,
    reason: str,
) -> None:
    """Emit counterfactual baseline unavailable marker."""
    # ... move implementation


def emit_throughput(
    hub: "TelemetryHub",
    env_id: int,
    device: str,
    batches_per_second: float,
    samples_per_second: float,
    gpu_utilization: float | None,
) -> None:
    """Emit per-env throughput metrics."""
    # ... move implementation


def emit_reward_summary(
    hub: "TelemetryHub",
    env_id: int,
    device: str,
    reward_components: "RewardComponentsTelemetry",
    total_reward: float,
) -> None:
    """Emit per-env reward aggregation."""
    # ... move implementation


def emit_mask_hit_rates(
    hub: "TelemetryHub",
    mask_stats: dict[str, dict[str, int]],
) -> None:
    """Emit action mask statistics."""
    # ... move implementation


def check_performance_degradation(
    current_accuracy: float,
    baseline_accuracy: float,
    threshold: float = 0.05,
) -> tuple[bool, float]:
    """Check if accuracy has degraded beyond threshold.

    Returns:
        (is_degraded, delta)
    """
    # ... move implementation


def apply_slot_telemetry(
    slot_config: dict,
    telemetry_config: TelemetryConfig,
    inner_epoch: int,
    global_epoch: int,
) -> None:
    """Apply telemetry configuration to slot."""
    # ... move implementation


__all__ = [
    "emit_with_env_context",
    "emit_batch_completed",
    "emit_last_action",
    "compute_grad_norm_surrogate",
    "aggregate_layer_gradient_health",
    "emit_ppo_update_event",
    "emit_action_distribution",
    "emit_cf_unavailable",
    "emit_throughput",
    "emit_reward_summary",
    "emit_mask_hit_rates",
    "check_performance_degradation",
    "apply_slot_telemetry",
]
```

**Step 3: Update vectorized.py imports**

At the top of `vectorized.py`, add:

```python
from esper.simic.telemetry.emitters import (
    emit_with_env_context,
    emit_batch_completed,
    emit_last_action,
    compute_grad_norm_surrogate,
    aggregate_layer_gradient_health,
    emit_ppo_update_event,
    emit_action_distribution,
    emit_cf_unavailable,
    emit_throughput,
    emit_reward_summary,
    emit_mask_hit_rates,
    check_performance_degradation,
    apply_slot_telemetry,
)
```

**Step 4: Remove the function definitions from vectorized.py**

Delete lines ~98-490 (the telemetry helper section).

**Step 5: Update function calls to remove underscore prefix**

The extracted functions no longer need the `_` prefix since they're now in a separate module. Update all call sites:

- `_emit_with_env_context(...)` → `emit_with_env_context(...)`
- `_emit_batch_completed(...)` → `emit_batch_completed(...)`
- etc.

**Step 6: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x
```

**Step 7: Commit**

```bash
git add src/esper/simic/telemetry/
git add src/esper/simic/vectorized.py
git commit -m "refactor(simic): extract telemetry emitters from vectorized.py

Move 13 telemetry emission functions (~390 lines) to simic/telemetry/emitters.py.
These are pure functions with no state dependencies.

vectorized.py: 2,740 → 2,350 lines (-14%)"
```

---

## Task 2: Extract ParallelEnvState

**Files:**
- Create: `src/esper/simic/parallel_env_state.py`
- Modify: `src/esper/simic/vectorized.py`
- Modify: `src/esper/simic/__init__.py`

**Step 1: Create parallel_env_state.py**

Move the `ParallelEnvState` dataclass (lines ~491-574 after Task 1 deletions):

```python
"""Parallel environment state container for vectorized PPO training.

ParallelEnvState holds all per-environment state needed during training,
including model references, optimizers, CUDA streams, and accumulators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from esper.leyline import SeedStage
    from esper.tamiyo import SignalTracker
    from esper.tolaria import TolariaGovernor


@dataclass
class ParallelEnvState:
    """Per-environment training state container.

    Attributes:
        model: The neural network model for this environment
        host_optimizer: Optimizer for host network parameters
        seed_optimizer: Optimizer for seed parameters (None if no active seed)
        scaler: AMP gradient scaler for mixed precision
        stream: CUDA stream for async execution (None on CPU)
        device: Device string (e.g., "cuda:0")
        env_idx: Environment index in the batch

        # Lifecycle state
        governor: Tolaria governor for model management
        signal_tracker: Tamiyo signal tracker for seed decisions
        seed_stage: Current seed lifecycle stage
        active_slot: Currently active slot ID (None if dormant)

        # Accumulators (reset each epoch)
        epoch_rewards: Accumulated rewards per slot
        epoch_counts: Action counts per slot
        epoch_successes: Successful actions per slot

        # Metrics
        train_loss: Running training loss
        train_correct: Running correct predictions
        train_total: Total training samples
        val_loss: Running validation loss
        val_correct: Running correct predictions
        val_total: Total validation samples
    """

    # Core training components
    model: nn.Module
    host_optimizer: torch.optim.Optimizer
    seed_optimizer: torch.optim.Optimizer | None
    scaler: torch.amp.GradScaler
    stream: torch.cuda.Stream | None
    device: str
    env_idx: int

    # Lifecycle management
    governor: "TolariaGovernor"
    signal_tracker: "SignalTracker"
    seed_stage: "SeedStage"
    active_slot: str | None

    # Per-slot accumulators
    epoch_rewards: dict[str, float] = field(default_factory=dict)
    epoch_counts: dict[str, int] = field(default_factory=dict)
    epoch_successes: dict[str, int] = field(default_factory=dict)

    # Training metrics
    train_loss: float = 0.0
    train_correct: int = 0
    train_total: int = 0
    val_loss: float = 0.0
    val_correct: int = 0
    val_total: int = 0

    # LSTM hidden state (if using recurrent policy)
    lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        """Validate state after initialization."""
        if self.stream is not None and not self.device.startswith("cuda"):
            raise ValueError(f"CUDA stream provided but device is {self.device}")

    def init_accumulators(self, slots: list[str]) -> None:
        """Initialize per-slot accumulators."""
        self.epoch_rewards = {slot: 0.0 for slot in slots}
        self.epoch_counts = {slot: 0 for slot in slots}
        self.epoch_successes = {slot: 0 for slot in slots}

    def zero_accumulators(self) -> None:
        """Reset accumulators for new epoch."""
        for slot in self.epoch_rewards:
            self.epoch_rewards[slot] = 0.0
            self.epoch_counts[slot] = 0
            self.epoch_successes[slot] = 0

        self.train_loss = 0.0
        self.train_correct = 0
        self.train_total = 0
        self.val_loss = 0.0
        self.val_correct = 0
        self.val_total = 0

    def reset_lstm_hidden(self) -> None:
        """Reset LSTM hidden state (call at episode boundaries)."""
        self.lstm_hidden = None


__all__ = ["ParallelEnvState"]
```

**Step 2: Update vectorized.py imports**

Add at top of `vectorized.py`:

```python
from esper.simic.parallel_env_state import ParallelEnvState
```

**Step 3: Remove ParallelEnvState from vectorized.py**

Delete the dataclass definition (now ~100-184 after Task 1).

**Step 4: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x
```

**Step 5: Commit**

```bash
git add src/esper/simic/parallel_env_state.py
git add src/esper/simic/vectorized.py
git commit -m "refactor(simic): extract ParallelEnvState from vectorized.py

Move ParallelEnvState dataclass (~84 lines) to dedicated module.
Self-contained state container with no logic dependencies.

vectorized.py: 2,350 → 2,266 lines (-4%)"
```

---

## Task 3: Extract Seed Management Helpers

**Files:**
- Create: `src/esper/simic/seed_management.py`
- Modify: `src/esper/simic/vectorized.py`
- Modify: `src/esper/simic/__init__.py`

**Step 1: Create seed_management.py**

Move these functions (lines ~185-330 after previous tasks):

```python
"""Seed lifecycle management helpers for vectorized PPO training.

These functions handle seed stage transitions, slot resolution,
and PPO update orchestration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from esper.leyline import SeedStage, LifecycleOp
from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel

if TYPE_CHECKING:
    from esper.kasmina import MorphogeneticModel
    from esper.simic.anomaly_detector import AnomalyReport
    from esper.simic.normalization import RunningMeanStd
    from esper.simic.ppo import PPOAgent
    from esper.nissa import TelemetryHub


def advance_active_seed(
    model: "MorphogeneticModel",
    slot_id: str,
) -> bool:
    """Attempt to advance seed to next lifecycle stage.

    Args:
        model: MorphogeneticModel containing the seed
        slot_id: Slot ID of the seed to advance

    Returns:
        True if seed was advanced (fossilized), False otherwise
    """
    # ... move implementation from vectorized.py _advance_active_seed


def resolve_target_slot(
    slot_idx: int,
    *,
    enabled_slots: list[str],
    slot_config: dict,
) -> tuple[str, str]:
    """Map action index to slot ID and position.

    Args:
        slot_idx: Index from action selection
        enabled_slots: List of enabled slot IDs
        slot_config: Slot configuration dict

    Returns:
        (slot_id, position) tuple
    """
    # ... move implementation from vectorized.py _resolve_target_slot


def calculate_entropy_anneal_steps(
    total_episodes: int,
    batches_per_episode: int,
    epochs_per_batch: int,
    anneal_fraction: float,
) -> int:
    """Convert episode count to entropy annealing steps.

    Args:
        total_episodes: Total training episodes
        batches_per_episode: Batches per episode
        epochs_per_batch: Epochs per batch
        anneal_fraction: Fraction of training for annealing

    Returns:
        Number of steps for entropy annealing
    """
    # ... move implementation from vectorized.py _calculate_entropy_anneal_steps


def aggregate_ppo_metrics(update_metrics: list[dict]) -> dict:
    """Merge metrics from multiple PPO updates.

    Args:
        update_metrics: List of metric dicts from each update

    Returns:
        Aggregated metrics dict with means
    """
    # ... move implementation from vectorized.py _aggregate_ppo_metrics


def run_ppo_updates(
    agent: "PPOAgent",
    ppo_updates_per_batch: int,
    raw_states: list,
    obs_normalizer: "RunningMeanStd | None",
    use_amp: bool,
) -> dict:
    """Execute one or more PPO update passes.

    Args:
        agent: PPOAgent instance
        ppo_updates_per_batch: Number of update passes
        raw_states: Raw observation states for normalization
        obs_normalizer: Optional observation normalizer
        use_amp: Whether to use automatic mixed precision

    Returns:
        Aggregated PPO metrics
    """
    # ... move implementation from vectorized.py _run_ppo_updates


def handle_telemetry_escalation(
    anomaly_report: "AnomalyReport",
    telemetry_config: TelemetryConfig,
) -> TelemetryLevel:
    """Escalate telemetry level based on anomaly report.

    Args:
        anomaly_report: Report from anomaly detector
        telemetry_config: Current telemetry configuration

    Returns:
        New telemetry level (may be escalated)
    """
    # ... move implementation from vectorized.py _handle_telemetry_escalation


def emit_anomaly_diagnostics(
    hub: "TelemetryHub",
    anomaly_report: "AnomalyReport",
    batch_idx: int,
    epoch_idx: int,
) -> None:
    """Emit detailed anomaly diagnostic telemetry.

    Args:
        hub: Telemetry hub
        anomaly_report: Report from anomaly detector
        batch_idx: Current batch index
        epoch_idx: Current epoch index
    """
    # ... move implementation from vectorized.py _emit_anomaly_diagnostics


__all__ = [
    "advance_active_seed",
    "resolve_target_slot",
    "calculate_entropy_anneal_steps",
    "aggregate_ppo_metrics",
    "run_ppo_updates",
    "handle_telemetry_escalation",
    "emit_anomaly_diagnostics",
]
```

**Step 2: Update vectorized.py imports**

Add at top of `vectorized.py`:

```python
from esper.simic.seed_management import (
    advance_active_seed,
    resolve_target_slot,
    calculate_entropy_anneal_steps,
    aggregate_ppo_metrics,
    run_ppo_updates,
    handle_telemetry_escalation,
    emit_anomaly_diagnostics,
)
```

**Step 3: Remove helper functions from vectorized.py**

Delete the seed management helper section (now ~185-330).

**Step 4: Update function calls to remove underscore prefix**

- `_advance_active_seed(...)` → `advance_active_seed(...)`
- `_resolve_target_slot(...)` → `resolve_target_slot(...)`
- etc.

**Step 5: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x
```

**Step 6: Commit**

```bash
git add src/esper/simic/seed_management.py
git add src/esper/simic/vectorized.py
git commit -m "refactor(simic): extract seed management from vectorized.py

Move 7 seed lifecycle helper functions (~145 lines) to dedicated module.
One-way dependencies - called from main loop, never call back.

vectorized.py: 2,266 → 2,121 lines (-6%)"
```

---

## Task 4: Update simic/__init__.py

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Add new module exports**

Update `__init__.py` to re-export from new modules:

```python
# Add to imports section
from esper.simic.parallel_env_state import ParallelEnvState
from esper.simic.seed_management import (
    advance_active_seed,
    resolve_target_slot,
    calculate_entropy_anneal_steps,
    aggregate_ppo_metrics,
    run_ppo_updates,
)
# Note: telemetry emitters stay internal (prefixed functions)

# Add to __all__
__all__ = [
    # ... existing exports ...
    "ParallelEnvState",
    # seed_management exports if needed publicly
]
```

**Step 2: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "refactor(simic): update __init__.py with new module exports

Re-export ParallelEnvState for external consumers."
```

---

## Task 5: Final Verification

**Step 1: Run full simic test suite**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v
```

**Step 2: Run integration tests**

```bash
PYTHONPATH=src uv run pytest tests/integration/ -v -k simic
```

**Step 3: Verify imports work from scripts**

```python
# Quick smoke test
from esper.simic import train_ppo_vectorized, ParallelEnvState
from esper.simic.telemetry.emitters import emit_batch_completed
from esper.simic.seed_management import advance_active_seed

print("All imports successful!")
```

**Step 4: Verify line count reduction**

```bash
wc -l src/esper/simic/vectorized.py
# Expected: ~2,121 lines (down from 2,740)
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "refactor(simic): complete Phase 1 vectorized.py refactoring

Summary:
- Extracted telemetry emitters to simic/telemetry/emitters.py (~390 lines)
- Extracted ParallelEnvState to simic/parallel_env_state.py (~84 lines)
- Extracted seed management to simic/seed_management.py (~145 lines)

vectorized.py reduced from 2,740 → 2,121 lines (-23%)

All extracted modules are:
- Pure functions or self-contained dataclasses
- One-way dependencies (no circular imports)
- Independently testable"
```

---

## Summary of Changes

| File | Before | After | Change |
|------|--------|-------|--------|
| `vectorized.py` | 2,740 | 2,121 | -619 (-23%) |
| `telemetry/emitters.py` | 0 | ~390 | +390 (new) |
| `parallel_env_state.py` | 0 | ~84 | +84 (new) |
| `seed_management.py` | 0 | ~145 | +145 (new) |

**Risk Assessment:**
- Telemetry emitters: 🟢 ZERO RISK (pure functions, stateless)
- ParallelEnvState: 🟢 ZERO RISK (self-contained dataclass)
- Seed management: 🟢 LOW RISK (one-way dependencies)

**Estimated Time:** 5-6 hours total

---

## Future Work (Phase 2)

After Phase 1 is stable, consider:

1. **Extract batch_processing.py** (~240 lines)
   - `create_env_state()`, `process_train_batch()`, `process_val_batch()`
   - Requires converting closures to explicit parameters
   - Risk: 🟡 MEDIUM

2. **Extract device_validation.py** (~62 lines)
   - `_parse_device()`, `_validate_cuda_device()`
   - Risk: 🟢 LOW

This would further reduce `vectorized.py` to ~1,219 lines (pure training loop).
