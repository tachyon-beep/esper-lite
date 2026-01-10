# Value Function Metrics Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the 9 advanced value function metrics (TELE-220 to TELE-228) from PPO update loop through telemetry to the ValueDiagnosticsPanel.

**Architecture:** Compute TD error statistics and return distribution metrics from rollout buffer data after GAE computation. Add fields to PPOUpdatePayload, wire through emitter, and update ValueFunctionMetrics in aggregator. The data sources already exist (buffer.values, buffer.returns, delta from GAE), we just need to extract and aggregate them.

**Tech Stack:** PyTorch tensors, dataclasses, existing telemetry pipeline (NissaHub → Aggregator → Snapshot)

---

## Overview

The 9 metrics to wire:

| TELE | Metric | Data Source | Computation |
|------|--------|-------------|-------------|
| TELE-220 | `v_return_correlation` | buffer.values, buffer.returns | Pearson correlation |
| TELE-221 | `td_error_mean` | delta from GAE | mean(δ) |
| TELE-222 | `td_error_std` | delta from GAE | std(δ) |
| TELE-223 | `bellman_error` | delta from GAE | mean(δ²) |
| TELE-224 | `return_p10` | buffer.returns | 10th percentile |
| TELE-225 | `return_p50` | buffer.returns | 50th percentile |
| TELE-226 | `return_p90` | buffer.returns | 90th percentile |
| TELE-227 | `return_variance` | buffer.returns | variance |
| TELE-228 | `return_skewness` | buffer.returns | skewness |

---

## Task 1: Store TD Errors in Rollout Buffer

**Files:**
- Modify: `src/esper/simic/agent/rollout_buffer.py:164-165` (add field)
- Modify: `src/esper/simic/agent/rollout_buffer.py:262-263` (initialize)
- Modify: `src/esper/simic/agent/rollout_buffer.py:446-451` (store delta)
- Test: `tests/simic/test_rollout_buffer.py`

**Step 1: Add td_errors field to TamiyoRolloutBuffer**

After line 165 (`returns: torch.Tensor`), add:

```python
    td_errors: torch.Tensor = field(init=False)
```

**Step 2: Initialize td_errors tensor**

After line 263 (`self.returns = ...`), add:

```python
        self.td_errors = torch.zeros(n, m, device=device)
```

**Step 3: Store delta during GAE computation**

After line 448 (`advantages[t] = last_gae`), add:

```python
                # Store TD error for telemetry (TELE-221/222/223)
                td_errors[t] = delta
```

Also need to initialize td_errors before the loop (after line 422):

```python
            td_errors = torch.zeros(num_steps, device=self.device)
```

And store it after the loop (after line 451):

```python
            self.td_errors[env_id, :num_steps] = td_errors
```

**Step 4: Add td_errors to get_batch**

In `get_batch()` method (around line 547), add to the returned dict:

```python
            "td_errors": self.td_errors.to(device, non_blocking=nb),
```

**Step 5: Write test**

```python
def test_td_errors_stored_during_gae(self):
    """TD errors should be stored during GAE computation."""
    buffer = TamiyoRolloutBuffer(
        num_envs=2, max_steps_per_env=4, num_slots=4,
        device="cpu", lstm_hidden_dim=64, num_lstm_layers=1
    )
    # ... fill buffer with test data
    buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)

    # TD errors should be populated
    assert buffer.td_errors.shape == (2, 4)
    # At least some non-zero values
    assert buffer.td_errors[:, :2].abs().sum() > 0
```

**Step 6: Run test**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_rollout_buffer.py -v -k td_errors`
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/simic/agent/rollout_buffer.py tests/simic/test_rollout_buffer.py
git commit -m "feat(simic): store TD errors during GAE computation

TELE-221/222/223: Store delta (TD error) alongside advantages during
compute_advantages_and_returns(). This enables td_error_mean, td_error_std,
and bellman_error telemetry without recomputing the values."
```

---

## Task 2: Create Value Function Metrics Computation Module

**Files:**
- Create: `src/esper/simic/telemetry/value_metrics.py`
- Test: `tests/simic/telemetry/test_value_metrics.py`

**Step 1: Write failing test**

```python
"""Tests for value function metrics computation."""

import pytest
import torch

from esper.simic.telemetry.value_metrics import compute_value_function_metrics


class TestComputeValueFunctionMetrics:
    """Tests for compute_value_function_metrics()."""

    def test_computes_td_error_stats(self):
        """Should compute TD error mean, std, and bellman error."""
        # Simple TD errors: [1, 2, 3, 4]
        td_errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([10.0, 20.0, 30.0, 40.0])
        returns = torch.tensor([11.0, 22.0, 33.0, 44.0])

        metrics = compute_value_function_metrics(td_errors, values, returns)

        assert abs(metrics["td_error_mean"] - 2.5) < 0.01
        assert metrics["td_error_std"] > 0
        # Bellman = mean(delta²) = mean([1, 4, 9, 16]) = 7.5
        assert abs(metrics["bellman_error"] - 7.5) < 0.01

    def test_computes_v_return_correlation(self):
        """Should compute Pearson correlation between values and returns."""
        # Perfect positive correlation
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        returns = torch.tensor([10.0, 20.0, 30.0, 40.0])
        td_errors = torch.zeros(4)

        metrics = compute_value_function_metrics(td_errors, values, returns)

        # Perfect correlation = 1.0
        assert abs(metrics["v_return_correlation"] - 1.0) < 0.01

    def test_computes_return_percentiles(self):
        """Should compute p10, p50, p90 of returns."""
        # Returns from 0 to 99
        returns = torch.arange(100, dtype=torch.float32)
        values = torch.zeros(100)
        td_errors = torch.zeros(100)

        metrics = compute_value_function_metrics(td_errors, values, returns)

        # p10 ≈ 10, p50 ≈ 50, p90 ≈ 90
        assert 8 < metrics["return_p10"] < 12
        assert 48 < metrics["return_p50"] < 52
        assert 88 < metrics["return_p90"] < 92

    def test_computes_return_variance_and_skewness(self):
        """Should compute variance and skewness of returns."""
        returns = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        values = torch.zeros(5)
        td_errors = torch.zeros(5)

        metrics = compute_value_function_metrics(td_errors, values, returns)

        assert metrics["return_variance"] > 0
        # Symmetric distribution has ~0 skewness
        assert abs(metrics["return_skewness"]) < 0.5

    def test_handles_empty_input(self):
        """Should return zeros for empty tensors."""
        td_errors = torch.tensor([])
        values = torch.tensor([])
        returns = torch.tensor([])

        metrics = compute_value_function_metrics(td_errors, values, returns)

        assert metrics["td_error_mean"] == 0.0
        assert metrics["v_return_correlation"] == 0.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_value_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write implementation**

```python
"""Value function metrics computation for telemetry.

Computes TELE-220 to TELE-228 metrics from rollout buffer data.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class ValueFunctionMetricsDict(TypedDict):
    """Computed value function metrics for telemetry."""

    # TELE-220: V-Return Correlation
    v_return_correlation: float

    # TELE-221/222/223: TD Error Statistics
    td_error_mean: float
    td_error_std: float
    bellman_error: float

    # TELE-224/225/226: Return Percentiles
    return_p10: float
    return_p50: float
    return_p90: float

    # TELE-227/228: Return Distribution
    return_variance: float
    return_skewness: float


@torch.inference_mode()
def compute_value_function_metrics(
    td_errors: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
) -> ValueFunctionMetricsDict:
    """Compute value function quality metrics from buffer data.

    Args:
        td_errors: TD errors from GAE computation [N]
        values: Value predictions V(s) [N]
        returns: Computed returns [N]

    Returns:
        Dictionary with all 9 value function metrics.

    PERF: Uses single .tolist() call at end to minimize GPU-CPU syncs.
    """
    if td_errors.numel() == 0:
        return ValueFunctionMetricsDict(
            v_return_correlation=0.0,
            td_error_mean=0.0,
            td_error_std=0.0,
            bellman_error=0.0,
            return_p10=0.0,
            return_p50=0.0,
            return_p90=0.0,
            return_variance=0.0,
            return_skewness=0.0,
        )

    # TD Error Statistics (TELE-221/222/223)
    td_mean = td_errors.mean()
    td_std = td_errors.std(correction=0)
    bellman = (td_errors ** 2).mean()  # Mean squared TD error

    # V-Return Correlation (TELE-220)
    # Pearson: cov(X,Y) / (std(X) * std(Y))
    v_mean = values.mean()
    r_mean = returns.mean()
    v_std = values.std(correction=0)
    r_std = returns.std(correction=0)

    if v_std > 1e-8 and r_std > 1e-8:
        covariance = ((values - v_mean) * (returns - r_mean)).mean()
        correlation = covariance / (v_std * r_std)
        correlation = correlation.clamp(-1.0, 1.0)  # Numerical stability
    else:
        correlation = torch.tensor(0.0, device=td_errors.device)

    # Return Percentiles (TELE-224/225/226)
    # quantile() requires float tensor on same device
    p10 = torch.quantile(returns.float(), 0.1)
    p50 = torch.quantile(returns.float(), 0.5)
    p90 = torch.quantile(returns.float(), 0.9)

    # Return Variance (TELE-227)
    ret_var = returns.var(correction=0)

    # Return Skewness (TELE-228)
    # skewness = E[(X-μ)³] / σ³
    if r_std > 1e-8:
        centered = returns - r_mean
        skewness = (centered ** 3).mean() / (r_std ** 3)
    else:
        skewness = torch.tensor(0.0, device=td_errors.device)

    # PERF: Single GPU→CPU sync via stacking all scalars
    results = torch.stack([
        correlation, td_mean, td_std, bellman,
        p10, p50, p90, ret_var, skewness
    ]).tolist()

    return ValueFunctionMetricsDict(
        v_return_correlation=results[0],
        td_error_mean=results[1],
        td_error_std=results[2],
        bellman_error=results[3],
        return_p10=results[4],
        return_p50=results[5],
        return_p90=results[6],
        return_variance=results[7],
        return_skewness=results[8],
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_value_metrics.py -v`
Expected: PASS

**Step 5: Add to __init__.py**

In `src/esper/simic/telemetry/__init__.py`, add:

```python
from .value_metrics import compute_value_function_metrics, ValueFunctionMetricsDict
```

**Step 6: Commit**

```bash
git add src/esper/simic/telemetry/value_metrics.py tests/simic/telemetry/test_value_metrics.py src/esper/simic/telemetry/__init__.py
git commit -m "feat(telemetry): add value function metrics computation

TELE-220 to TELE-228: Add compute_value_function_metrics() to compute:
- v_return_correlation (Pearson between V(s) and returns)
- td_error_mean, td_error_std, bellman_error
- return_p10, return_p50, return_p90
- return_variance, return_skewness

Uses single GPU→CPU sync via torch.stack().tolist() for performance."
```

---

## Task 3: Add Value Function Metrics to PPOUpdatePayload

**Files:**
- Modify: `src/esper/leyline/telemetry.py:648-660` (add fields to PPOUpdatePayload)
- Test: Existing tests should still pass

**Step 1: Add fields to PPOUpdatePayload**

After `return_std: float = 0.0` (around line 649), add:

```python
    # Value function quality metrics (TELE-220 to TELE-228)
    # These measure value network calibration and return distribution shape
    v_return_correlation: float = 0.0
    td_error_mean: float = 0.0
    td_error_std: float = 0.0
    bellman_error: float = 0.0
    return_p10: float = 0.0
    return_p50: float = 0.0
    return_p90: float = 0.0
    return_variance: float = 0.0
    return_skewness: float = 0.0
```

**Step 2: Verify existing tests still pass**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_telemetry.py -v --tb=short`
Expected: PASS (dataclass defaults are backwards-compatible)

**Step 3: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "feat(telemetry): add value function metrics to PPOUpdatePayload

TELE-220 to TELE-228: Add 9 optional fields for value function quality:
- v_return_correlation: Pearson correlation V(s) vs returns
- td_error_mean/std/bellman_error: TD error statistics
- return_p10/p50/p90: Return distribution percentiles
- return_variance/skewness: Return distribution shape

All default to 0.0 for backwards compatibility."
```

---

## Task 4: Compute and Emit Value Function Metrics in PPO Update

**Files:**
- Modify: `src/esper/simic/agent/ppo.py:20-25` (add import)
- Modify: `src/esper/simic/agent/ppo.py:370-380` (compute after GAE)
- Modify: `src/esper/simic/agent/ppo.py:1000-1040` (add to metrics dict)
- Test: `tests/simic/test_ppo_value_metrics.py`

**Step 1: Write failing test**

```python
"""Tests for value function metrics in PPO update."""

import pytest
import torch

from esper.simic.agent.ppo import PPOAgent
from esper.tamiyo.policy.factory import create_policy
from esper.leyline import SlotConfig


@pytest.fixture
def ppo_agent():
    """Create a minimal PPO agent for testing."""
    slot_config = SlotConfig(
        num_slots=4,
        blueprints=["A", "B", "C", "D"],
        default_blueprint="A",
    )
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        state_dim=32,
        lstm_hidden_dim=64,
        device="cpu",
        compile_mode="off",
    )
    return PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device="cpu",
        num_envs=2,
        max_steps_per_env=8,
        state_dim=32,
    )


class TestTELE220to228ValueFunctionMetrics:
    """TELE-220 to TELE-228: Value function metrics in PPO update."""

    def test_value_function_metrics_in_update_result(self, ppo_agent):
        """update() should return all 9 value function metrics."""
        # Collect minimal rollout
        for _ in range(8):
            obs = torch.randn(2, 32)
            blueprints = torch.zeros(2, 4, dtype=torch.long)
            actions, hidden = ppo_agent.get_action(obs, blueprints, hidden=None)
            rewards = torch.ones(2)
            dones = torch.zeros(2, dtype=torch.bool)
            ppo_agent.buffer.store(
                obs, blueprints, actions, rewards, dones, hidden
            )
        ppo_agent.buffer.finalize(
            torch.randn(2, 32), torch.zeros(2, 4, dtype=torch.long), None
        )

        metrics = ppo_agent.update()

        # All 9 metrics should be present
        assert "v_return_correlation" in metrics
        assert "td_error_mean" in metrics
        assert "td_error_std" in metrics
        assert "bellman_error" in metrics
        assert "return_p10" in metrics
        assert "return_p50" in metrics
        assert "return_p90" in metrics
        assert "return_variance" in metrics
        assert "return_skewness" in metrics

    def test_v_return_correlation_in_valid_range(self, ppo_agent):
        """V-return correlation should be in [-1, 1]."""
        for _ in range(8):
            obs = torch.randn(2, 32)
            blueprints = torch.zeros(2, 4, dtype=torch.long)
            actions, hidden = ppo_agent.get_action(obs, blueprints, hidden=None)
            rewards = torch.randn(2)  # Varied rewards
            dones = torch.zeros(2, dtype=torch.bool)
            ppo_agent.buffer.store(
                obs, blueprints, actions, rewards, dones, hidden
            )
        ppo_agent.buffer.finalize(
            torch.randn(2, 32), torch.zeros(2, 4, dtype=torch.long), None
        )

        metrics = ppo_agent.update()

        assert -1.0 <= metrics["v_return_correlation"] <= 1.0

    def test_return_percentiles_ordered(self, ppo_agent):
        """Return percentiles should be ordered: p10 <= p50 <= p90."""
        for _ in range(8):
            obs = torch.randn(2, 32)
            blueprints = torch.zeros(2, 4, dtype=torch.long)
            actions, hidden = ppo_agent.get_action(obs, blueprints, hidden=None)
            rewards = torch.randn(2).abs()  # Positive varied rewards
            dones = torch.zeros(2, dtype=torch.bool)
            ppo_agent.buffer.store(
                obs, blueprints, actions, rewards, dones, hidden
            )
        ppo_agent.buffer.finalize(
            torch.randn(2, 32), torch.zeros(2, 4, dtype=torch.long), None
        )

        metrics = ppo_agent.update()

        assert metrics["return_p10"] <= metrics["return_p50"]
        assert metrics["return_p50"] <= metrics["return_p90"]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ppo_value_metrics.py -v`
Expected: FAIL with `KeyError: 'v_return_correlation'`

**Step 3: Add import in ppo.py**

After line 23 (the compute_lstm_health import), add:

```python
from esper.simic.telemetry.value_metrics import compute_value_function_metrics
```

**Step 4: Compute value function metrics after GAE**

After line 372 (`pre_norm_adv_mean, pre_norm_adv_std = self.buffer.normalize_advantages()`), add:

```python
        # Compute value function metrics (TELE-220 to TELE-228)
        # Uses buffer data: td_errors, values, returns after GAE computation
        value_func_metrics = self._compute_value_function_metrics()
```

**Step 5: Add helper method to PPOAgent**

Add this method to the PPOAgent class (after the existing helper methods):

```python
    def _compute_value_function_metrics(self) -> dict[str, float]:
        """Compute value function metrics from buffer data.

        Called after compute_advantages_and_returns() to extract
        TELE-220 to TELE-228 metrics.
        """
        # Collect valid data from all environments
        all_td_errors = []
        all_values = []
        all_returns = []

        for env_id in range(self.buffer.num_envs):
            num_steps = self.buffer.step_counts[env_id]
            if num_steps > 0:
                all_td_errors.append(self.buffer.td_errors[env_id, :num_steps])
                all_values.append(self.buffer.values[env_id, :num_steps])
                all_returns.append(self.buffer.returns[env_id, :num_steps])

        if not all_td_errors:
            return {
                "v_return_correlation": 0.0,
                "td_error_mean": 0.0,
                "td_error_std": 0.0,
                "bellman_error": 0.0,
                "return_p10": 0.0,
                "return_p50": 0.0,
                "return_p90": 0.0,
                "return_variance": 0.0,
                "return_skewness": 0.0,
            }

        td_errors = torch.cat(all_td_errors)
        values = torch.cat(all_values)
        returns = torch.cat(all_returns)

        return compute_value_function_metrics(td_errors, values, returns)
```

**Step 6: Add to aggregated_result before return**

After the existing LSTM health metrics aggregation (around line 1037), add:

```python
        # Add value function metrics (TELE-220 to TELE-228)
        aggregated_result.update(value_func_metrics)
```

**Step 7: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ppo_value_metrics.py -v`
Expected: PASS

**Step 8: Run full PPO test suite**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_ppo.py -v --tb=short`
Expected: All tests PASS

**Step 9: Commit**

```bash
git add src/esper/simic/agent/ppo.py tests/simic/test_ppo_value_metrics.py
git commit -m "feat(simic): compute value function metrics in PPO update

TELE-220 to TELE-228: Add _compute_value_function_metrics() helper that:
- Collects td_errors, values, returns from buffer after GAE
- Calls compute_value_function_metrics() for statistical computation
- Adds all 9 metrics to PPOUpdateMetrics return value

Enables v_return_correlation, td_error_*, bellman_error, return_* metrics."
```

---

## Task 5: Wire Emitter to Include Value Function Metrics

**Files:**
- Modify: `src/esper/simic/telemetry/emitters.py:790-830` (emit_ppo_update_event)
- Test: Run existing telemetry tests

**Step 1: Add value function metrics to emit_ppo_update_event**

In `emit_ppo_update_event()`, after the existing metrics extraction (around line 820), add:

```python
            # Value function metrics (TELE-220 to TELE-228)
            v_return_correlation=metrics.get("v_return_correlation", 0.0),
            td_error_mean=metrics.get("td_error_mean", 0.0),
            td_error_std=metrics.get("td_error_std", 0.0),
            bellman_error=metrics.get("bellman_error", 0.0),
            return_p10=metrics.get("return_p10", 0.0),
            return_p50=metrics.get("return_p50", 0.0),
            return_p90=metrics.get("return_p90", 0.0),
            return_variance=metrics.get("return_variance", 0.0),
            return_skewness=metrics.get("return_skewness", 0.0),
```

**Step 2: Run emitter tests**

Run: `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_emitters.py -v --tb=short`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/simic/telemetry/emitters.py
git commit -m "feat(telemetry): emit value function metrics in PPO update event

TELE-220 to TELE-228: Extract 9 value function metrics from PPO update
result and include in PPOUpdatePayload:
- v_return_correlation, td_error_mean/std, bellman_error
- return_p10/p50/p90, return_variance, return_skewness"
```

---

## Task 6: Wire Aggregator to Update ValueFunctionMetrics

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py:940-950` (in _handle_ppo_update)
- Test: Run aggregator tests

**Step 1: Add value function metrics update to _handle_ppo_update**

After the existing value_mean/value_std updates (around line 943), add:

```python
        # Value function quality metrics (TELE-220 to TELE-228)
        # Update the nested ValueFunctionMetrics dataclass
        vf = self._tamiyo.value_function
        vf.v_return_correlation = payload.v_return_correlation
        vf.td_error_mean = payload.td_error_mean
        vf.td_error_std = payload.td_error_std
        vf.bellman_error = payload.bellman_error
        vf.return_p10 = payload.return_p10
        vf.return_p50 = payload.return_p50
        vf.return_p90 = payload.return_p90
        vf.return_variance = payload.return_variance
        vf.return_skewness = payload.return_skewness
```

**Step 2: Run aggregator tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_aggregator.py -v --tb=short`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): wire value function metrics in aggregator

TELE-220 to TELE-228: Update ValueFunctionMetrics from PPOUpdatePayload:
- v_return_correlation, td_error_mean/std, bellman_error
- return_p10/p50/p90, return_variance, return_skewness

ValueDiagnosticsPanel now displays real data instead of zeros."
```

---

## Task 7: Remove xfails from Telemetry Tests

**Files:**
- Modify: `tests/telemetry/test_tele_value_metrics.py` (9 xfail tests)

**Step 1: Find and update the xfail tests**

Search for `@pytest.mark.xfail` in the file and remove the decorator from all 9 wiring tests:
- TestTELE220WiringVReturnCorrelation
- TestTELE221WiringTDErrorMean
- TestTELE222WiringTDErrorStd
- TestTELE223WiringBellmanError
- TestTELE224WiringReturnP10
- TestTELE225WiringReturnP50
- TestTELE226WiringReturnP90
- TestTELE227WiringReturnVariance
- TestTELE228WiringReturnSkewness

Also update the test docstrings to indicate the wiring is complete.

**Step 2: Run the tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/telemetry/test_tele_value_metrics.py -v`
Expected: 82 passed, 0 xfailed

**Step 3: Commit**

```bash
git add tests/telemetry/test_tele_value_metrics.py
git commit -m "test(telemetry): remove xfails for TELE-220 to TELE-228

Value function metrics now fully wired:
- TELE-220: v_return_correlation
- TELE-221: td_error_mean
- TELE-222: td_error_std
- TELE-223: bellman_error
- TELE-224-226: return_p10/p50/p90
- TELE-227: return_variance
- TELE-228: return_skewness"
```

---

## Task 8: Update WIRING_GAPS.md

**Files:**
- Modify: `docs/telemetry/WIRING_GAPS.md`

**Step 1: Update the document**

1. Change "Latest fix" to TELE-228 (return_skewness)
2. Update total count from 25 to 16 xfails (25 - 9 = 16)
3. Mark TELE-220 to TELE-228 section as "ALL FIXED ✓"
4. Update Summary table
5. Update Quick Reference table

**Step 2: Verify changes look correct**

```bash
cat docs/telemetry/WIRING_GAPS.md | head -30
```

**Step 3: Commit**

```bash
git add docs/telemetry/WIRING_GAPS.md
git commit -m "docs(telemetry): TELE-220 to TELE-228 fixed - update WIRING_GAPS.md

Value Function Metrics section now has 0 xfails.
Total xfails: 25 -> 16"
```

---

## Task 9: Run Full Telemetry Test Suite

**Files:** None (verification only)

**Step 1: Run all telemetry tests**

Run: `PYTHONPATH=src uv run pytest tests/telemetry/ -v --tb=short`
Expected: All tests PASS except documented xfails (should be 16 xfails now)

**Step 2: Verify value metrics specifically**

Run: `PYTHONPATH=src uv run pytest tests/telemetry/test_value_metrics.py tests/telemetry/test_tele_value_metrics.py -v`
Expected: All tests PASS (0 xfails in these files)

---

## PyTorch Considerations

**GPU-CPU Sync Points:**
- `compute_value_function_metrics()` uses single `torch.stack().tolist()` sync
- Called once per PPO update (not per sample), so sync overhead is minimal
- All tensor ops are batched before the sync

**Memory:**
- `td_errors` tensor adds O(num_envs × max_steps) memory to buffer
- Same size as `advantages` tensor - negligible overhead
- Metrics are Python floats after computation (no GPU memory growth)

**torch.compile Compatibility:**
- `compute_value_function_metrics()` uses `@torch.inference_mode()` decorator
- No dynamic control flow that would cause graph breaks
- `compute_advantages_and_returns()` already has `@torch.compiler.disable`

---

## DRL Considerations

**V-Return Correlation:**
- Most diagnostic value metric - low correlation means advantages are noise
- Should trend upward during healthy training (0.3 → 0.7+)
- Sudden drops indicate value network divergence

**TD Error Statistics:**
- High mean = biased value estimates (check learning rate, target staleness)
- High std = noisy targets (normal early, concerning late)
- Bellman error spikes often precede NaN losses

**Return Percentiles:**
- Large p90-p10 spread = bimodal policy (some episodes great, some terrible)
- All negative = policy hasn't found any good trajectories
- Converging percentiles = policy stabilizing

---

## Success Criteria

1. `PYTHONPATH=src uv run pytest tests/simic/test_rollout_buffer.py -v -k td_errors` - PASS
2. `PYTHONPATH=src uv run pytest tests/simic/telemetry/test_value_metrics.py -v` - PASS
3. `PYTHONPATH=src uv run pytest tests/simic/test_ppo_value_metrics.py -v` - PASS
4. `PYTHONPATH=src uv run pytest tests/simic/test_ppo.py -v` - PASS (no regression)
5. `PYTHONPATH=src uv run pytest tests/telemetry/test_tele_value_metrics.py -v` - 82 PASS, 0 xfail
6. Total xfails in telemetry suite: 16 (down from 25)
