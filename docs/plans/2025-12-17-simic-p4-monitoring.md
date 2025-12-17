# Simic P4 Monitoring & Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU-accurate profiling, per-head gradient monitoring, LSTM health tracking, and gradient drift detection to enable early identification of training instability.

**Architecture:** Extend existing telemetry infrastructure with new collectors and metrics. CUDA events for GPU timing, per-head gradient norms via `_foreach_norm`, LSTM hidden state monitoring in network forward pass, and EMA-based drift detection in anomaly detector.

**Tech Stack:** PyTorch 2.4+, torch.cuda.Event, torch.profiler, dataclasses

---

## Task 1: Add CUDA Event-Based Timing (P4-1)

**Files:**
- Modify: `src/esper/simic/training/vectorized.py`
- Modify: `src/esper/simic/telemetry/emitters.py`

**Why:** Current `time.perf_counter()` measures CPU time, not GPU kernel execution. For async CUDA, this gives misleading results. CUDA events measure actual GPU work.

**Step 1: Add CUDATimer utility class to vectorized.py**

Add after imports (around line 50):

```python
class CUDATimer:
    """GPU-accurate timing using CUDA events.

    Falls back to CPU timing when CUDA unavailable.
    """

    def __init__(self, device: str = "cuda"):
        self.use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = 0.0

    def start(self) -> None:
        """Record start time."""
        if self.use_cuda:
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Record end time and return elapsed milliseconds."""
        if self.use_cuda:
            self.end_event.record()
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        else:
            return (time.perf_counter() - self.start_time) * 1000.0
```

**Step 2: Replace step timing with CUDATimer**

In `train_ppo_vectorized()`, around line 1061, change:

```python
# Before
throughput_step_time_ms_sum = 0.0
# ... later ...
step_start = time.perf_counter()
# ... later ...
throughput_step_time_ms_sum += (time.perf_counter() - step_start) * 1000.0
```

To:

```python
# After
throughput_step_time_ms_sum = 0.0
step_timer = CUDATimer(device)
# ... later (line ~1081) ...
step_timer.start()
# ... later (line ~1952) ...
throughput_step_time_ms_sum += step_timer.stop()
```

**Step 3: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_vectorized.py -v -x -q --tb=short
```
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "perf(simic): add CUDA event-based timing for GPU-accurate profiling (P4-1)

Replaces time.perf_counter() with torch.cuda.Event for step timing.
CUDA events measure actual GPU kernel execution, not CPU-side overhead.
Falls back to CPU timing when CUDA unavailable."
```

---

## Task 2: Add Per-Head Gradient Norm Logging (P4-6)

**Files:**
- Modify: `src/esper/simic/agent/ppo.py`
- Modify: `src/esper/simic/agent/types.py`
- Modify: `src/esper/simic/telemetry/emitters.py`

**Why:** Per-head entropy (P3-1) shows exploration collapse. Per-head gradient norms show if one head dominates learning. Together they diagnose head-specific issues.

**Step 1: Add HeadGradientNorms to types.py**

Add to `src/esper/simic/agent/types.py`:

```python
class HeadGradientNorms(TypedDict):
    """Per-head gradient norms from factored policy."""
    slot: float
    blueprint: float
    blend: float
    op: float
    value: float
```

Add to `__all__` and update `PPOUpdateMetrics`:

```python
class PPOUpdateMetrics(TypedDict):
    # ... existing fields ...
    head_entropies: dict[str, list[float]]
    head_grad_norms: dict[str, list[float]]  # NEW: per-head gradient norms
```

**Step 2: Collect per-head gradient norms in ppo.py update()**

After `loss.backward()` (around line 580), add:

```python
# Collect per-head gradient norms (P4-6)
head_grad_norms: dict[str, list[float]] = {head: [] for head in HEAD_NAMES + ("value",)}

# ... inside epoch loop, after loss.backward() ...
with torch.inference_mode():
    for head_name, head_module in [
        ("slot", self.network.slot_head),
        ("blueprint", self.network.blueprint_head),
        ("blend", self.network.blend_head),
        ("op", self.network.op_head),
        ("value", self.network.value_head),
    ]:
        params_with_grad = [p for p in head_module.parameters() if p.grad is not None]
        if params_with_grad:
            grad_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad) for p in params_with_grad])
            ).item()
        else:
            grad_norm = 0.0
        head_grad_norms[head_name].append(grad_norm)
```

Add to returned metrics:

```python
return {
    # ... existing fields ...
    "head_entropies": head_entropy_history,
    "head_grad_norms": head_grad_norms,  # NEW
}
```

**Step 3: Emit per-head gradient norms in telemetry**

In `src/esper/simic/telemetry/emitters.py`, update `emit_ppo_update_event()` to include:

```python
# Per-head gradient norms (P4-6)
if "head_grad_norms" in update_info:
    for head, values in update_info["head_grad_norms"].items():
        if values:
            data[f"head_{head}_grad_norm"] = sum(values) / len(values)
```

**Step 4: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/agent/ -v -x -q --tb=short
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/agent/ppo.py src/esper/simic/agent/types.py src/esper/simic/telemetry/emitters.py
git commit -m "feat(simic): add per-head gradient norm logging (P4-6)

Tracks gradient norm per action head (slot, blueprint, blend, op, value).
Complements P3-1 entropy tracking - together diagnose head-specific issues.
Emitted in PPO_UPDATE telemetry events."
```

---

## Task 3: Add LSTM Hidden State Health Monitoring (P4-8)

**Files:**
- Create: `src/esper/simic/telemetry/lstm_health.py`
- Modify: `src/esper/simic/agent/tamiyo_network.py`
- Modify: `src/esper/simic/telemetry/emitters.py`
- Modify: `src/esper/simic/telemetry/__init__.py`

**Why:** LSTM hidden states can experience magnitude drift or NaN propagation. Early detection prevents catastrophic training failures.

**Step 1: Create lstm_health.py**

```python
"""LSTM Hidden State Health Monitoring.

Tracks hidden state statistics to detect:
- Magnitude explosion (norm > threshold)
- Magnitude vanishing (norm < threshold)
- NaN/Inf propagation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class LSTMHealthMetrics:
    """Health metrics for LSTM hidden state."""

    h_norm: float  # L2 norm of hidden state
    c_norm: float  # L2 norm of cell state
    h_max: float   # Max absolute value in h
    c_max: float   # Max absolute value in c
    has_nan: bool
    has_inf: bool

    def is_healthy(
        self,
        max_norm: float = 100.0,
        min_norm: float = 1e-6,
    ) -> bool:
        """Check if LSTM state is healthy."""
        return (
            not self.has_nan
            and not self.has_inf
            and self.h_norm < max_norm
            and self.c_norm < max_norm
            and self.h_norm > min_norm
            and self.c_norm > min_norm
        )

    def to_dict(self) -> dict[str, float | bool]:
        """Convert to dict for telemetry."""
        return {
            "lstm_h_norm": self.h_norm,
            "lstm_c_norm": self.c_norm,
            "lstm_h_max": self.h_max,
            "lstm_c_max": self.c_max,
            "lstm_has_nan": self.has_nan,
            "lstm_has_inf": self.has_inf,
        }


def compute_lstm_health(
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
) -> LSTMHealthMetrics | None:
    """Compute health metrics for LSTM hidden state.

    Args:
        hidden: Tuple of (h, c) tensors, or None if no hidden state

    Returns:
        LSTMHealthMetrics or None if no hidden state
    """
    if hidden is None:
        return None

    h, c = hidden

    with torch.inference_mode():
        h_norm = torch.linalg.vector_norm(h).item()
        c_norm = torch.linalg.vector_norm(c).item()
        h_max = h.abs().max().item()
        c_max = c.abs().max().item()
        has_nan = torch.isnan(h).any().item() or torch.isnan(c).any().item()
        has_inf = torch.isinf(h).any().item() or torch.isinf(c).any().item()

    return LSTMHealthMetrics(
        h_norm=h_norm,
        c_norm=c_norm,
        h_max=h_max,
        c_max=c_max,
        has_nan=has_nan,
        has_inf=has_inf,
    )


__all__ = ["LSTMHealthMetrics", "compute_lstm_health"]
```

**Step 2: Export from telemetry/__init__.py**

Add to `src/esper/simic/telemetry/__init__.py`:

```python
from esper.simic.telemetry.lstm_health import LSTMHealthMetrics, compute_lstm_health
```

And add to `__all__`:

```python
"LSTMHealthMetrics",
"compute_lstm_health",
```

**Step 3: Add health check to tamiyo_network.py get_action()**

In `get_action()` method, after LSTM forward pass (around line 200), add:

```python
# Optionally return health metrics for monitoring
# (caller can check via compute_lstm_health(new_hidden))
```

The health check will be called from vectorized.py when storing transitions.

**Step 4: Add LSTM health to step telemetry in vectorized.py**

In the step storage section (around line 1920), add:

```python
# Check LSTM health (P4-8)
if use_telemetry and env_state.lstm_hidden is not None:
    from esper.simic.telemetry import compute_lstm_health
    lstm_health = compute_lstm_health(env_state.lstm_hidden)
    if lstm_health and not lstm_health.is_healthy():
        logger.warning(
            f"Env {env_idx}: LSTM unhealthy - h_norm={lstm_health.h_norm:.2f}, "
            f"c_norm={lstm_health.c_norm:.2f}, nan={lstm_health.has_nan}"
        )
```

**Step 5: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x -q --tb=short
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/telemetry/lstm_health.py src/esper/simic/telemetry/__init__.py \
        src/esper/simic/training/vectorized.py
git commit -m "feat(simic): add LSTM hidden state health monitoring (P4-8)

New LSTMHealthMetrics tracks h/c norms, max values, NaN/Inf presence.
Warns when hidden state magnitude drifts or becomes numerically unstable.
Critical for catching recurrent policy failures early."
```

---

## Task 4: Add Gradient EMA Drift Detection (P4-9)

**Files:**
- Create: `src/esper/simic/telemetry/gradient_ema.py`
- Modify: `src/esper/simic/telemetry/anomaly_detector.py`
- Modify: `src/esper/simic/telemetry/__init__.py`

**Why:** Gradual gradient drift indicates training instability before it becomes catastrophic. EMA tracking detects slow degradation that single-step checks miss.

**Step 1: Create gradient_ema.py**

```python
"""Gradient EMA Tracking for Drift Detection.

Uses exponential moving average to track gradient statistics over time.
Detects gradual drift that single-step anomaly checks miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GradientEMATracker:
    """Tracks gradient statistics with EMA for drift detection.

    Drift indicator = |current - ema| / (ema + epsilon)
    High drift suggests training instability.
    """

    momentum: float = 0.99
    epsilon: float = 1e-8

    # EMA state (initialized on first update)
    ema_norm: float = field(default=0.0, init=False)
    ema_health: float = field(default=1.0, init=False)
    _initialized: bool = field(default=False, init=False)
    _update_count: int = field(default=0, init=False)

    def update(self, grad_norm: float, grad_health: float) -> dict[str, float]:
        """Update EMA and return drift indicators.

        Args:
            grad_norm: Current gradient norm
            grad_health: Current gradient health (0-1)

        Returns:
            Dict with ema values and drift indicators
        """
        self._update_count += 1

        if not self._initialized:
            # First update: initialize to current values
            self.ema_norm = grad_norm
            self.ema_health = grad_health
            self._initialized = True
            return {
                "ema_grad_norm": self.ema_norm,
                "ema_grad_health": self.ema_health,
                "norm_drift": 0.0,
                "health_drift": 0.0,
            }

        # Compute drift before updating EMA
        norm_drift = abs(grad_norm - self.ema_norm) / (self.ema_norm + self.epsilon)
        health_drift = abs(grad_health - self.ema_health) / (self.ema_health + self.epsilon)

        # Update EMA
        self.ema_norm = self.momentum * self.ema_norm + (1 - self.momentum) * grad_norm
        self.ema_health = self.momentum * self.ema_health + (1 - self.momentum) * grad_health

        return {
            "ema_grad_norm": self.ema_norm,
            "ema_grad_health": self.ema_health,
            "norm_drift": norm_drift,
            "health_drift": health_drift,
        }

    def check_drift(
        self,
        grad_norm: float,
        grad_health: float,
        drift_threshold: float = 0.5,
    ) -> tuple[bool, dict[str, float]]:
        """Update and check if drift exceeds threshold.

        Args:
            grad_norm: Current gradient norm
            grad_health: Current gradient health
            drift_threshold: Threshold for drift warning

        Returns:
            Tuple of (has_drift, metrics_dict)
        """
        metrics = self.update(grad_norm, grad_health)
        has_drift = (
            metrics["norm_drift"] > drift_threshold
            or metrics["health_drift"] > drift_threshold
        )
        return has_drift, metrics

    def state_dict(self) -> dict[str, float | bool | int]:
        """Return state for checkpointing."""
        return {
            "ema_norm": self.ema_norm,
            "ema_health": self.ema_health,
            "initialized": self._initialized,
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict[str, float | bool | int]) -> None:
        """Load state from checkpoint."""
        self.ema_norm = state["ema_norm"]
        self.ema_health = state["ema_health"]
        self._initialized = state["initialized"]
        self._update_count = state["update_count"]


__all__ = ["GradientEMATracker"]
```

**Step 2: Add drift check to AnomalyDetector**

In `src/esper/simic/telemetry/anomaly_detector.py`, add method:

```python
def check_gradient_drift(
    self,
    norm_drift: float,
    health_drift: float,
    drift_threshold: float = 0.5,
) -> AnomalyReport:
    """Check for gradient drift anomaly.

    Args:
        norm_drift: Gradient norm drift indicator
        health_drift: Gradient health drift indicator
        drift_threshold: Threshold for drift warning

    Returns:
        AnomalyReport with any detected drift
    """
    report = AnomalyReport()

    if norm_drift > drift_threshold:
        report.add_anomaly(
            "gradient_norm_drift",
            f"norm_drift={norm_drift:.3f} > {drift_threshold}",
        )

    if health_drift > drift_threshold:
        report.add_anomaly(
            "gradient_health_drift",
            f"health_drift={health_drift:.3f} > {drift_threshold}",
        )

    return report
```

Update `check_all()` to accept drift parameters (optional, for backwards compatibility).

**Step 3: Export from telemetry/__init__.py**

Add to imports and `__all__`:

```python
from esper.simic.telemetry.gradient_ema import GradientEMATracker
# ...
"GradientEMATracker",
```

**Step 4: Integrate into vectorized.py PPO update**

In the PPO update section (around line 2030), after collecting gradient stats:

```python
# Track gradient drift (P4-9)
if not hasattr(train_ppo_vectorized, '_grad_ema_tracker'):
    from esper.simic.telemetry import GradientEMATracker
    train_ppo_vectorized._grad_ema_tracker = GradientEMATracker()

if gradient_stats:
    has_drift, drift_metrics = train_ppo_vectorized._grad_ema_tracker.check_drift(
        gradient_stats.gradient_norm,
        gradient_stats.gradient_health,
    )
    if has_drift:
        logger.warning(f"Gradient drift detected: {drift_metrics}")
```

**Step 5: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v -x -q --tb=short
```
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/telemetry/gradient_ema.py src/esper/simic/telemetry/anomaly_detector.py \
        src/esper/simic/telemetry/__init__.py src/esper/simic/training/vectorized.py
git commit -m "feat(simic): add gradient EMA drift detection (P4-9)

New GradientEMATracker uses momentum=0.99 to track gradient statistics.
Drift indicator detects gradual instability that single-step checks miss.
Integrated with AnomalyDetector for unified anomaly reporting."
```

---

## Task 5: Add torch.profiler Integration Points (P4-5)

**Files:**
- Create: `src/esper/simic/telemetry/profiler.py`
- Modify: `src/esper/simic/training/vectorized.py`

**Why:** torch.profiler provides detailed GPU trace for identifying bottlenecks. Integration points allow on-demand profiling without code changes.

**Step 1: Create profiler.py helper**

```python
"""torch.profiler Integration for Simic Training.

Provides context manager for on-demand profiling of training loops.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def training_profiler(
    output_dir: str = "./profiler_traces",
    enabled: bool = True,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    repeat: int = 1,
) -> Iterator[torch.profiler.profile | None]:
    """Context manager for profiling training steps.

    Args:
        output_dir: Directory for trace output
        enabled: Whether profiling is enabled
        wait: Steps to wait before warmup
        warmup: Warmup steps before active profiling
        active: Steps to actively profile
        repeat: Number of profiling cycles

    Yields:
        Profiler instance or None if disabled

    Usage:
        with training_profiler(enabled=args.profile) as prof:
            for step in training_loop:
                train_step()
                if prof:
                    prof.step()
    """
    if not enabled:
        yield None
        return

    os.makedirs(output_dir, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        yield prof


__all__ = ["training_profiler"]
```

**Step 2: Export from telemetry/__init__.py**

Add to imports and `__all__`:

```python
from esper.simic.telemetry.profiler import training_profiler
# ...
"training_profiler",
```

**Step 3: Add profile flag to train_ppo_vectorized**

Add parameter to function signature:

```python
def train_ppo_vectorized(
    # ... existing params ...
    profile: bool = False,
    profile_dir: str = "./profiler_traces",
) -> tuple[PPOAgent, list[dict]]:
```

Wrap main training loop:

```python
from esper.simic.telemetry import training_profiler

with training_profiler(output_dir=profile_dir, enabled=profile) as prof:
    while episodes_completed < total_episodes:
        # ... existing training loop ...

        # At end of each batch
        if prof:
            prof.step()
```

**Step 4: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_vectorized.py -v -x -q --tb=short
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/telemetry/profiler.py src/esper/simic/telemetry/__init__.py \
        src/esper/simic/training/vectorized.py
git commit -m "feat(simic): add torch.profiler integration points (P4-5)

New training_profiler() context manager for on-demand GPU profiling.
Outputs TensorBoard-compatible traces for bottleneck analysis.
Enabled via profile=True flag in train_ppo_vectorized()."
```

---

## Verification

After all tasks complete, run the full test suite:

```bash
PYTHONPATH=src uv run pytest tests/simic/ tests/leyline/ -v --tb=short
```

Expected: All tests pass.

---

## Summary

| Task | Issue | Type | Est. Time |
|------|-------|------|-----------|
| 1 | P4-1 | Performance | 10 min |
| 2 | P4-6 | Observability | 20 min |
| 3 | P4-8 | Stability | 30 min |
| 4 | P4-9 | Stability | 30 min |
| 5 | P4-5 | Performance | 20 min |

**Total estimated time:** ~110 minutes

**New files created:**
- `src/esper/simic/telemetry/lstm_health.py`
- `src/esper/simic/telemetry/gradient_ema.py`
- `src/esper/simic/telemetry/profiler.py`

**Key integration points:**
- CUDA events: Step timing in vectorized.py
- Per-head gradients: PPO update() method
- LSTM health: Step storage in vectorized.py
- Gradient EMA: PPO update in vectorized.py
- Profiler: Training loop wrapper
