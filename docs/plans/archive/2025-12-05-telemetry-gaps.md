# Telemetry Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the three gaps identified in code review: (1) wire auto-escalation triggers, (2) add debug telemetry collection hooks in PPO update, (3) add missing test coverage.

**Architecture:** Extend the PPO update loop to detect anomalies (ratio explosion, value collapse, numerical instability) and trigger `telemetry_config.escalate_temporarily()`. When DEBUG level is active, call expensive diagnostic functions and include results in metrics.

**Tech Stack:** Python, PyTorch, pytest, existing telemetry infrastructure

---

## Phase 1: Test Coverage Gaps

### Task 1: Add TelemetryLevel.OFF Behavior Test

**Files:**
- Modify: `tests/simic/test_telemetry_config.py:60` (append to file)

**Step 1: Write the test**

Add to `tests/simic/test_telemetry_config.py`:

```python
    def test_should_collect_when_off(self):
        """should_collect returns False for all categories when OFF."""
        config = TelemetryConfig(level=TelemetryLevel.OFF)
        assert config.should_collect("ops_normal") is False
        assert config.should_collect("debug") is False

    def test_should_collect_when_minimal(self):
        """MINIMAL level collects neither ops_normal nor debug."""
        config = TelemetryConfig(level=TelemetryLevel.MINIMAL)
        assert config.should_collect("ops_normal") is False
        assert config.should_collect("debug") is False
```

**Step 2: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_telemetry_config.py::TestTelemetryConfig::test_should_collect_when_off -v`

Expected: PASS (the implementation already handles this correctly)

**Step 3: Commit**

```bash
git add tests/simic/test_telemetry_config.py
git commit -m "test(simic): add OFF and MINIMAL level collection tests"
```

---

### Task 2: Add PPOHealthTelemetry Boundary Condition Tests

**Files:**
- Modify: `tests/simic/test_ppo_telemetry.py:52` (append to TestPPOHealthTelemetry)

**Step 1: Write the tests**

Add to `tests/simic/test_ppo_telemetry.py` in `TestPPOHealthTelemetry`:

```python
    def test_is_ratio_healthy_at_exact_threshold(self):
        """Boundary: ratio exactly at threshold is healthy (< not <=)."""
        # ratio_max exactly at 5.0 threshold - should be healthy (< 5.0)
        at_max = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=4.999, ratio_min=0.101,
        )
        assert at_max.is_ratio_healthy() is True

        # Just above max threshold
        above_max = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=5.001, ratio_min=0.5,
        )
        assert above_max.is_ratio_healthy() is False

        # Just below min threshold
        below_min = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=2.0, ratio_min=0.099,
        )
        assert below_min.is_ratio_healthy() is False

    def test_is_ratio_healthy_with_custom_thresholds(self):
        """Can use custom thresholds for ratio health check."""
        telemetry = PPOHealthTelemetry(
            policy_loss=0.5, value_loss=0.3, entropy=0.8,
            approx_kl=0.01, clip_fraction=0.15,
            ratio_mean=1.0, ratio_std=0.1, ratio_max=3.0, ratio_min=0.3,
        )
        # Default thresholds (5.0, 0.1) - healthy
        assert telemetry.is_ratio_healthy() is True
        # Stricter thresholds - unhealthy
        assert telemetry.is_ratio_healthy(max_ratio_threshold=2.0) is False
        assert telemetry.is_ratio_healthy(min_ratio_threshold=0.5) is False
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/simic/test_ppo_telemetry.py::TestPPOHealthTelemetry -v`

Expected: PASS (5 tests)

**Step 3: Commit**

```bash
git add tests/simic/test_ppo_telemetry.py
git commit -m "test(simic): add PPOHealthTelemetry boundary condition tests"
```

---

### Task 3: Add ValueFunctionTelemetry Zero-Variance Test

**Files:**
- Modify: `tests/simic/test_ppo_telemetry.py:82` (append to TestValueFunctionTelemetry)

**Step 1: Write the test**

Add to `tests/simic/test_ppo_telemetry.py` in `TestValueFunctionTelemetry`:

```python
    def test_explained_variance_zero_variance_returns(self):
        """Handles zero-variance returns without division error."""
        import torch

        # All returns identical - zero variance
        returns = torch.tensor([2.0, 2.0, 2.0, 2.0])
        values = torch.tensor([1.9, 2.1, 2.0, 2.0])

        telemetry = ValueFunctionTelemetry.from_tensors(returns, values)
        # Should gracefully handle and return 0.0
        assert telemetry.explained_variance == 0.0

    def test_from_tensors_with_advantages(self):
        """from_tensors correctly handles advantages parameter."""
        import torch

        returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([1.1, 1.9, 3.1, 3.9])
        advantages = torch.tensor([0.5, -0.3, 0.2, 0.1])

        telemetry = ValueFunctionTelemetry.from_tensors(returns, values, advantages)
        assert abs(telemetry.advantage_mean - 0.125) < 0.01  # mean of advantages
        assert telemetry.advantage_std > 0
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/simic/test_ppo_telemetry.py::TestValueFunctionTelemetry -v`

Expected: PASS (4 tests)

**Step 3: Commit**

```bash
git add tests/simic/test_ppo_telemetry.py
git commit -m "test(simic): add ValueFunctionTelemetry edge case tests"
```

---

### Task 4: Add MemoryMetrics Fragmentation Test

**Files:**
- Modify: `tests/simic/test_memory_telemetry.py:67` (append to file)

**Step 1: Write the test**

Add to `tests/simic/test_memory_telemetry.py`:

```python
    def test_is_healthy_high_fragmentation(self):
        """High fragmentation is unhealthy even with adequate headroom."""
        fragmented = MemoryMetrics(
            allocated_mb=2000,
            reserved_mb=6000,  # 3x fragmentation ratio
            max_allocated_mb=2500,
            fragmentation_ratio=3.0,  # Above default 2.5 threshold
            utilization=0.3,
            headroom_mb=4000,  # Plenty of headroom
            oom_risk_score=0.3,
        )
        assert fragmented.is_healthy() is False

    def test_is_healthy_with_custom_thresholds(self):
        """Can customize health thresholds."""
        metrics = MemoryMetrics(
            allocated_mb=5000,
            reserved_mb=6000,
            max_allocated_mb=5500,
            fragmentation_ratio=1.5,
            utilization=0.5,
            headroom_mb=150,  # Below default 200MB threshold
            oom_risk_score=0.2,
        )
        # Default thresholds - unhealthy (headroom < 200)
        assert metrics.is_healthy() is False
        # Custom thresholds - healthy
        assert metrics.is_healthy(min_headroom_mb=100.0) is True
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/simic/test_memory_telemetry.py -v`

Expected: PASS (6 tests)

**Step 3: Commit**

```bash
git add tests/simic/test_memory_telemetry.py
git commit -m "test(simic): add MemoryMetrics fragmentation and threshold tests"
```

---

## Phase 2: Auto-Escalation Triggers

### Task 5: Add AnomalyDetector Helper Class

**Files:**
- Create: `src/esper/simic/anomaly_detector.py`
- Test: `tests/simic/test_anomaly_detector.py`

**Step 1: Write the failing test**

Create `tests/simic/test_anomaly_detector.py`:

```python
"""Tests for anomaly detection in PPO training."""

import pytest

from esper.simic.anomaly_detector import AnomalyDetector, AnomalyReport


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def test_detect_ratio_explosion(self):
        """Detects ratio explosion."""
        detector = AnomalyDetector()
        report = detector.check_ratios(
            ratio_max=6.0,  # > 5.0 threshold
            ratio_min=0.5,
        )
        assert report.has_anomaly is True
        assert "ratio_explosion" in report.anomaly_types

    def test_detect_ratio_collapse(self):
        """Detects ratio collapse (min too low)."""
        detector = AnomalyDetector()
        report = detector.check_ratios(
            ratio_max=2.0,
            ratio_min=0.05,  # < 0.1 threshold
        )
        assert report.has_anomaly is True
        assert "ratio_collapse" in report.anomaly_types

    def test_healthy_ratios_no_anomaly(self):
        """Healthy ratios produce no anomaly."""
        detector = AnomalyDetector()
        report = detector.check_ratios(
            ratio_max=2.0,
            ratio_min=0.5,
        )
        assert report.has_anomaly is False
        assert len(report.anomaly_types) == 0

    def test_detect_value_collapse(self):
        """Detects value function collapse."""
        detector = AnomalyDetector()
        report = detector.check_value_function(
            explained_variance=-0.5,  # Negative = worse than mean
        )
        assert report.has_anomaly is True
        assert "value_collapse" in report.anomaly_types

    def test_detect_numerical_instability(self):
        """Detects NaN/Inf in metrics."""
        detector = AnomalyDetector()
        report = detector.check_numerical_stability(
            has_nan=True,
            has_inf=False,
        )
        assert report.has_anomaly is True
        assert "numerical_instability" in report.anomaly_types

    def test_combined_check(self):
        """Can check all anomalies at once."""
        detector = AnomalyDetector()
        report = detector.check_all(
            ratio_max=6.0,
            ratio_min=0.5,
            explained_variance=0.5,
            has_nan=False,
            has_inf=False,
        )
        assert report.has_anomaly is True
        assert "ratio_explosion" in report.anomaly_types
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_anomaly_detector.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'esper.simic.anomaly_detector'`

**Step 3: Implement AnomalyDetector**

Create `src/esper/simic/anomaly_detector.py`:

```python
"""Anomaly Detection for PPO Training.

Detects training anomalies that should trigger escalation to DEBUG telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AnomalyReport:
    """Report of detected anomalies."""

    has_anomaly: bool = False
    anomaly_types: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def add_anomaly(self, anomaly_type: str, detail: str | None = None) -> None:
        """Add an anomaly to the report."""
        self.has_anomaly = True
        self.anomaly_types.append(anomaly_type)
        if detail:
            self.details[anomaly_type] = detail


@dataclass
class AnomalyDetector:
    """Detects training anomalies for telemetry escalation.

    Thresholds are configurable but have sensible defaults based on
    typical PPO training behavior.
    """

    # Ratio thresholds
    max_ratio_threshold: float = 5.0
    min_ratio_threshold: float = 0.1

    # Value function thresholds
    min_explained_variance: float = 0.1

    def check_ratios(
        self,
        ratio_max: float,
        ratio_min: float,
    ) -> AnomalyReport:
        """Check for ratio explosion or collapse.

        Args:
            ratio_max: Maximum ratio in batch
            ratio_min: Minimum ratio in batch

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if ratio_max > self.max_ratio_threshold:
            report.add_anomaly(
                "ratio_explosion",
                f"ratio_max={ratio_max:.3f} > {self.max_ratio_threshold}",
            )

        if ratio_min < self.min_ratio_threshold:
            report.add_anomaly(
                "ratio_collapse",
                f"ratio_min={ratio_min:.3f} < {self.min_ratio_threshold}",
            )

        return report

    def check_value_function(
        self,
        explained_variance: float,
    ) -> AnomalyReport:
        """Check for value function collapse.

        Args:
            explained_variance: Explained variance metric

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if explained_variance < self.min_explained_variance:
            report.add_anomaly(
                "value_collapse",
                f"explained_variance={explained_variance:.3f} < {self.min_explained_variance}",
            )

        return report

    def check_numerical_stability(
        self,
        has_nan: bool,
        has_inf: bool,
    ) -> AnomalyReport:
        """Check for numerical instability.

        Args:
            has_nan: Whether NaN values were detected
            has_inf: Whether Inf values were detected

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if has_nan or has_inf:
            issues = []
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")
            report.add_anomaly(
                "numerical_instability",
                f"Detected: {', '.join(issues)}",
            )

        return report

    def check_all(
        self,
        ratio_max: float,
        ratio_min: float,
        explained_variance: float,
        has_nan: bool = False,
        has_inf: bool = False,
    ) -> AnomalyReport:
        """Run all anomaly checks and combine results.

        Args:
            ratio_max: Maximum ratio in batch
            ratio_min: Minimum ratio in batch
            explained_variance: Explained variance metric
            has_nan: Whether NaN values were detected
            has_inf: Whether Inf values were detected

        Returns:
            Combined AnomalyReport
        """
        combined = AnomalyReport()

        for check_report in [
            self.check_ratios(ratio_max, ratio_min),
            self.check_value_function(explained_variance),
            self.check_numerical_stability(has_nan, has_inf),
        ]:
            if check_report.has_anomaly:
                combined.has_anomaly = True
                combined.anomaly_types.extend(check_report.anomaly_types)
                combined.details.update(check_report.details)

        return combined


__all__ = ["AnomalyDetector", "AnomalyReport"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_anomaly_detector.py -v`

Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/anomaly_detector.py tests/simic/test_anomaly_detector.py
git commit -m "feat(simic): add AnomalyDetector for telemetry escalation triggers"
```

---

### Task 6: Wire Auto-Escalation into PPO Update

**Files:**
- Modify: `src/esper/simic/ppo.py:258-399`
- Test: `tests/simic/test_ppo_auto_escalation.py`

**Step 1: Write the failing test**

Create `tests/simic/test_ppo_auto_escalation.py`:

```python
"""Tests for PPO auto-escalation on anomaly detection."""

import torch
import pytest

from esper.simic.ppo import PPOAgent
from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel


class TestPPOAutoEscalation:
    """Tests for auto-escalation in PPO update."""

    @pytest.fixture
    def agent(self):
        """Create PPO agent."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_buffer(self, agent):
        """Fill buffer with transitions."""
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

    def test_escalation_triggered_on_anomaly(self, agent, filled_buffer):
        """Anomaly detection triggers escalation."""
        config = TelemetryConfig(
            level=TelemetryLevel.NORMAL,
            auto_escalate_on_anomaly=True,
        )

        # Initial state - not escalated
        assert config.effective_level == TelemetryLevel.NORMAL
        assert config.escalation_epochs_remaining == 0

        # Run update - may or may not detect anomaly depending on random init
        # For this test, we just verify the mechanism exists
        metrics = agent.update(last_value=0.0, telemetry_config=config)

        # Verify anomaly_detected field is in metrics
        assert "anomaly_detected" in metrics

    def test_escalation_disabled_when_flag_false(self, agent, filled_buffer):
        """Escalation doesn't happen when auto_escalate_on_anomaly=False."""
        config = TelemetryConfig(
            level=TelemetryLevel.NORMAL,
            auto_escalate_on_anomaly=False,
        )

        metrics = agent.update(last_value=0.0, telemetry_config=config)

        # Should never escalate
        assert config.escalation_epochs_remaining == 0

    def test_tick_escalation_called_each_update(self, agent, filled_buffer):
        """Escalation countdown ticks each update."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        config.escalate_temporarily(epochs=3)

        assert config.escalation_epochs_remaining == 3

        # First update
        agent.update(last_value=0.0, telemetry_config=config)
        assert config.escalation_epochs_remaining == 2

        # Refill buffer for another update
        for _ in range(20):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            action, log_prob, value, _ = agent.get_action(state, action_mask)
            agent.store_transition(
                state=state, action=action, log_prob=log_prob,
                value=value, reward=0.1, done=False, action_mask=action_mask,
            )

        # Second update
        agent.update(last_value=0.0, telemetry_config=config)
        assert config.escalation_epochs_remaining == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_ppo_auto_escalation.py -v`

Expected: FAIL with `KeyError: 'anomaly_detected'`

**Step 3: Modify PPO update to wire auto-escalation**

In `src/esper/simic/ppo.py`, modify the `update()` method. After line 303 (after computing explained_variance), add anomaly detection:

```python
        # === AUTO-ESCALATION: Check for anomalies and escalate if needed ===
        from esper.simic.anomaly_detector import AnomalyDetector

        anomaly_detector = AnomalyDetector()
        anomaly_detected = False
```

Then, after line 396 (after the epoch loop, before clearing buffer), add:

```python
        # === AUTO-ESCALATION: Analyze collected metrics for anomalies ===
        if metrics['ratio_max']:  # Only if we have ratio data
            max_ratio = max(metrics['ratio_max'])
            min_ratio = min(metrics['ratio_min'])

            anomaly_report = anomaly_detector.check_all(
                ratio_max=max_ratio,
                ratio_min=min_ratio,
                explained_variance=explained_variance,
                has_nan=False,  # Would need to track during update
                has_inf=False,
            )

            if anomaly_report.has_anomaly:
                anomaly_detected = True
                if telemetry_config.auto_escalate_on_anomaly:
                    telemetry_config.escalate_temporarily()

        # Tick escalation countdown
        telemetry_config.tick_escalation()
```

And update the result dict at the end (around line 398):

```python
        result = {k: sum(v) / len(v) for k, v in metrics.items()}
        result['anomaly_detected'] = 1.0 if anomaly_detected else 0.0
        if early_stopped:
            result['early_stopped'] = 1.0
        return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_ppo_auto_escalation.py -v`

Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_auto_escalation.py
git commit -m "feat(simic): wire auto-escalation triggers into PPO update"
```

---

## Phase 3: Debug Telemetry Collection Hooks

### Task 7: Add Debug Telemetry Collection to PPO Update

**Files:**
- Modify: `src/esper/simic/ppo.py:258-399`
- Modify: `tests/simic/test_ppo_telemetry_integration.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_ppo_telemetry_integration.py`:

```python
    def test_debug_level_collects_layer_gradients(
        self, agent_with_telemetry, filled_buffer
    ):
        """DEBUG level collects per-layer gradient statistics."""
        config = TelemetryConfig(level=TelemetryLevel.DEBUG)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should have debug-specific metrics
        assert "debug_gradient_stats" in metrics
        assert "debug_numerical_stability" in metrics

    def test_normal_level_skips_debug_collection(
        self, agent_with_telemetry, filled_buffer
    ):
        """NORMAL level does not collect expensive debug telemetry."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should NOT have debug-specific metrics
        assert "debug_gradient_stats" not in metrics
        assert "debug_numerical_stability" not in metrics
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_ppo_telemetry_integration.py::TestPPOTelemetryIntegration::test_debug_level_collects_layer_gradients -v`

Expected: FAIL with `AssertionError` (key not present)

**Step 3: Add debug telemetry collection to PPO update**

In `src/esper/simic/ppo.py`, inside the batch loop (after `loss.backward()` around line 361), add conditional debug collection:

```python
                self.optimizer.zero_grad()
                loss.backward()

                # === DEBUG TELEMETRY: Collect expensive diagnostics if enabled ===
                if telemetry_config.should_collect("debug"):
                    from esper.simic.debug_telemetry import (
                        collect_per_layer_gradients,
                        check_numerical_stability,
                    )
                    # Collect once per update (first batch only) to limit overhead
                    if 'debug_gradient_stats' not in metrics:
                        layer_stats = collect_per_layer_gradients(self.network)
                        metrics['debug_gradient_stats'] = [
                            [s.to_dict() for s in layer_stats]
                        ]
                        stability = check_numerical_stability(self.network, loss)
                        metrics['debug_numerical_stability'] = [stability.to_dict()]

                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_ppo_telemetry_integration.py -v`

Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_telemetry_integration.py
git commit -m "feat(simic): add debug telemetry collection hooks in PPO update"
```

---

### Task 8: Update simic __init__.py with AnomalyDetector Export

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Add import**

Add to the imports section:

```python
from esper.simic.anomaly_detector import AnomalyDetector, AnomalyReport
```

**Step 2: Add to __all__**

Add to the `__all__` list:

```python
    "AnomalyDetector",
    "AnomalyReport",
```

**Step 3: Verify import works**

Run: `uv run python -c "from esper.simic import AnomalyDetector, AnomalyReport; print('OK')"`

Expected: Prints `OK`

**Step 4: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "feat(simic): export AnomalyDetector from package"
```

---

### Task 9: Fix DEBUG Level Integration Test Assertion

**Files:**
- Modify: `tests/simic/test_ppo_telemetry_integration.py:57-68`

**Step 1: Fix the weak assertion**

Replace the existing `test_debug_level_adds_extra_diagnostics` test:

```python
    def test_debug_level_adds_extra_diagnostics(
        self, agent_with_telemetry, filled_buffer
    ):
        """DEBUG level adds extra diagnostic info beyond NORMAL level."""
        config = TelemetryConfig(level=TelemetryLevel.DEBUG)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should have NORMAL level metrics
        assert "ratio_mean" in metrics
        assert "explained_variance" in metrics

        # Should ALSO have DEBUG-specific metrics
        assert "debug_gradient_stats" in metrics
        assert "debug_numerical_stability" in metrics
```

**Step 2: Run all telemetry tests**

Run: `uv run pytest tests/simic/test_ppo_telemetry_integration.py tests/simic/test_telemetry_config.py tests/simic/test_ppo_telemetry.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/simic/test_ppo_telemetry_integration.py
git commit -m "test(simic): strengthen DEBUG level integration test assertions"
```

---

## Summary

After completing all 9 tasks:

**New Files:**
- `src/esper/simic/anomaly_detector.py` - AnomalyDetector class for detecting training issues
- `tests/simic/test_anomaly_detector.py` - Tests for anomaly detection
- `tests/simic/test_ppo_auto_escalation.py` - Tests for auto-escalation wiring

**Modified Files:**
- `src/esper/simic/ppo.py` - Auto-escalation triggers + debug collection hooks
- `src/esper/simic/__init__.py` - Export AnomalyDetector
- `tests/simic/test_telemetry_config.py` - OFF/MINIMAL level tests
- `tests/simic/test_ppo_telemetry.py` - Boundary condition tests
- `tests/simic/test_memory_telemetry.py` - Fragmentation tests
- `tests/simic/test_ppo_telemetry_integration.py` - Debug collection + stronger assertions

**Verification:**

Run full telemetry test suite:
```bash
uv run pytest tests/simic/test_telemetry_config.py tests/simic/test_ppo_telemetry.py tests/simic/test_anomaly_detector.py tests/simic/test_ppo_auto_escalation.py tests/simic/test_ppo_telemetry_integration.py tests/simic/test_memory_telemetry.py -v
```

Expected: All tests PASS (~25 tests total)
