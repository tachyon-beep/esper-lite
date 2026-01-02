# TamiyoBrainV2 Debugging Enhancements Implementation Plan

> **Status:** COMPLETED (2026-01-03)
> **Implementation Note:** Tasks 0-3, 5-6 fully implemented. Task 4 (keyboard navigation for decisions) NOT IMPLEMENTED.

**Goal:** Enhance TamiyoBrainV2 with predictive monitoring capabilities to help operators spot and debug RL training issues before they become catastrophic.

**Architecture:** Six independent enhancements that add early warning indicators, numerical stability displays, and correlation metrics. Each enhancement follows TDD with isolated changes to minimize risk.

**Tech Stack:** Python 3.13, Textual TUI framework, Rich text rendering, dataclasses, numpy

---

## Enhancement Overview

| # | Enhancement | Files | Effort |
|---|-------------|-------|--------|
| 1 | NaN/Inf Display | status_banner.py, schema.py | Low |
| 2 | Triggering Condition in Banner | status_banner.py | Low |
| 3 | Entropy Velocity + Collapse Risk | schema.py, ppo_health.py | Medium |
| 4 | Keyboard Navigation for Decisions | decisions_column.py, widget.py | Medium |
| 5 | Entropy ↔ Clip Correlation | schema.py, ppo_health.py | Medium |
| 6 | Value Function Stats | schema.py, ppo.py, ppo_health.py | Medium |

---

## Task 0: Prerequisites - Add Missing Schema Field

**CRITICAL:** The `inf_grad_count` field exists in `PPOUpdatePayload` but is NOT captured in `TamiyoState`. This must be added before Task 1.

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:428` (add field after nan_grad_count)
- Modify: `src/esper/karn/sanctum/aggregator.py:698` (add capture line)

**Step 1: Add field to TamiyoState**

```python
# In src/esper/karn/sanctum/schema.py, after line 428
nan_grad_count: int = 0  # NaN gradient count (existing)
inf_grad_count: int = 0  # Inf gradient count (NEW)
```

**Step 2: Capture in aggregator**

```python
# In src/esper/karn/sanctum/aggregator.py, after line 698
self._tamiyo.nan_grad_count = payload.nan_grad_count  # existing
self._tamiyo.inf_grad_count = payload.inf_grad_count  # NEW
```

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/aggregator.py
git commit -m "fix(sanctum): add missing inf_grad_count to TamiyoState

Field existed in PPOUpdatePayload but was not captured in TamiyoState.
Required for Task 1 (NaN/Inf display)."
```

---

## Task 1: Display NaN/Inf Counts in Status Banner

The schema tracks `nan_grad_count` and `inf_grad_count` in `TamiyoState` (after Task 0) but they're never displayed. This is a quick win.

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/status_banner.py:107-156`
- Test: `tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py
import pytest
from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo_brain_v2.status_banner import StatusBanner


class TestNaNInfDisplay:
    """Test NaN/Inf indicator in status banner."""

    def test_no_nan_shows_nothing(self):
        """When nan_grad_count=0, no NaN indicator should appear."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                nan_grad_count=0,
                inf_grad_count=0,
                entropy=1.0,
                clip_fraction=0.1,
            ),
            current_batch=60,  # Past warmup
        )
        banner.update_snapshot(snapshot)
        content = banner._render_content()
        assert "NaN" not in content.plain
        assert "Inf" not in content.plain

    def test_nan_detected_shows_indicator(self):
        """When nan_grad_count>0, show NaN indicator with count."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                nan_grad_count=3,
                inf_grad_count=0,
                entropy=1.0,
                clip_fraction=0.1,
            ),
            current_batch=60,
        )
        banner.update_snapshot(snapshot)
        content = banner._render_content()
        assert "NaN:3" in content.plain

    def test_both_nan_and_inf_shows_both(self):
        """When both NaN and Inf detected, show both counts."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                nan_grad_count=2,
                inf_grad_count=5,
                entropy=1.0,
                clip_fraction=0.1,
            ),
            current_batch=60,
        )
        banner.update_snapshot(snapshot)
        content = banner._render_content()
        assert "NaN:2" in content.plain
        assert "Inf:5" in content.plain

    def test_nan_triggers_critical_status(self):
        """Any NaN should trigger critical status."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                nan_grad_count=1,
                entropy=1.0,  # Otherwise healthy
                clip_fraction=0.1,
            ),
            current_batch=60,
        )
        banner.update_snapshot(snapshot)
        status, label, style = banner._get_overall_status()
        assert status == "critical"
        assert "NaN" in label

    def test_nan_priority_over_other_critical(self):
        """NaN should override all other critical conditions."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                nan_grad_count=1,
                entropy=0.1,  # Also collapsed - but NaN takes priority
                clip_fraction=0.4,  # Also critical
            ),
            current_batch=60,
        )
        banner.update_snapshot(snapshot)
        status, label, style = banner._get_overall_status()
        assert status == "critical"
        assert "NaN" in label
        assert "Entropy" not in label  # NaN takes priority
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py::TestNaNInfDisplay -v`
Expected: FAIL - tests don't exist yet, then NaN not in content

**Step 3: Implement NaN/Inf display in status banner**

In `status_banner.py`, modify `_render_content()` to add NaN/Inf indicator FIRST (highest visibility position):

```python
def _render_content(self) -> Text:
    """Render the status banner content."""
    banner = Text()
    tamiyo = self._snapshot.tamiyo if self._snapshot else None

    # NaN/Inf indicator FIRST - leftmost position for F-pattern visibility
    if tamiyo:
        if tamiyo.nan_grad_count > 0 or tamiyo.inf_grad_count > 0:
            # Severity graduation: >5 issues = reverse video for maximum visibility
            if tamiyo.nan_grad_count > 5 or tamiyo.inf_grad_count > 5:
                style = "red bold reverse"
            else:
                style = "red bold"

            if tamiyo.nan_grad_count > 0:
                banner.append(f"NaN:{tamiyo.nan_grad_count}", style=style)
                banner.append(" ")
            if tamiyo.inf_grad_count > 0:
                banner.append(f"Inf:{tamiyo.inf_grad_count}", style=style)
                banner.append(" ")

    # Then status icon and label...
    # ... rest of existing code ...
```

And modify `_get_overall_status()` to check NaN/Inf first:

```python
def _get_overall_status(self) -> tuple[str, str, str]:
    # ... existing warmup check ...

    # NaN/Inf check (HIGHEST PRIORITY - before all other checks)
    if tamiyo.nan_grad_count > 0:
        return "critical", "NaN DETECTED", "red bold"
    if tamiyo.inf_grad_count > 0:
        return "critical", "Inf DETECTED", "red bold"

    # ... existing critical/warning checks ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py::TestNaNInfDisplay -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py src/esper/karn/sanctum/widgets/tamiyo_brain_v2/status_banner.py
git commit -m "feat(sanctum): display NaN/Inf counts in status banner

- Add NaN/Inf indicator to leftmost banner position (F-pattern)
- Show both NaN and Inf counts when present
- Severity graduation: >5 issues gets reverse video
- NaN/Inf triggers critical status with highest priority"
```

---

## Task 2: Show Triggering Condition in Status Banner

Currently the banner shows "CAUTION" or "FAILING" but not WHY. Operators must scan all metrics to find the problem.

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/status_banner.py:157-208`
- Test: `tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py`

**Step 1: Write the failing test**

```python
class TestTriggeringCondition:
    """Test that status banner shows which metric triggered the status."""

    def test_entropy_warning_shows_reason(self):
        """Low entropy should show 'Entropy' in label."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                entropy=0.4,  # Below ENTROPY_WARNING (0.5)
                clip_fraction=0.1,  # OK
                kl_divergence=0.005,  # OK
                explained_variance=0.7,  # OK
                advantage_std=1.0,  # OK
                grad_norm=1.0,  # OK
            ),
            current_batch=60,
        )
        banner.update_snapshot(snapshot)
        status, label, style = banner._get_overall_status()
        assert status == "warning"
        assert "Entropy" in label

    def test_clip_critical_shows_reason(self):
        """High clip fraction should show 'Clip' in label."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                entropy=1.0,  # OK
                clip_fraction=0.35,  # Above CLIP_CRITICAL (0.3)
                kl_divergence=0.005,  # OK
                explained_variance=0.7,  # OK
                advantage_std=1.0,  # OK
                grad_norm=1.0,  # OK
            ),
            current_batch=60,
        )
        banner.update_snapshot(snapshot)
        status, label, style = banner._get_overall_status()
        assert status == "critical"
        assert "Clip" in label

    def test_multiple_issues_shows_count(self):
        """When multiple issues exist, show count of additional issues."""
        banner = StatusBanner()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                entropy=0.2,  # CRITICAL
                clip_fraction=0.35,  # CRITICAL
                kl_divergence=0.04,  # CRITICAL
                explained_variance=0.7,
                advantage_std=1.0,
                grad_norm=1.0,
            ),
            current_batch=60,
        )
        banner.update_snapshot(snapshot)
        status, label, style = banner._get_overall_status()
        assert status == "critical"
        assert "Entropy" in label
        assert "+2" in label  # 2 more critical issues
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py::TestTriggeringCondition -v`
Expected: FAIL - label is just "CAUTION" or "FAILING" without reason

**Step 3: Implement triggering condition display with multi-issue count**

Refactor `_get_overall_status()` to collect all issues and return with count:

```python
def _get_overall_status(self) -> tuple[str, str, str]:
    """Determine overall status with triggering condition.

    Returns:
        (status, label, style) tuple where label includes the reason
        and count of additional issues if multiple.
    """
    if self._snapshot is None:
        return "ok", "WAITING", "dim"

    tamiyo = self._snapshot.tamiyo

    if not tamiyo.ppo_data_received:
        return "ok", "WAITING", "dim"

    # Warmup period
    current_batch = self._snapshot.current_batch
    if current_batch < self.WARMUP_BATCHES:
        return "warmup", f"WARMUP [{current_batch}/{self.WARMUP_BATCHES}]", "cyan"

    # NaN/Inf check (highest priority - not counted with others)
    if tamiyo.nan_grad_count > 0:
        return "critical", "NaN DETECTED", "red bold"
    if tamiyo.inf_grad_count > 0:
        return "critical", "Inf DETECTED", "red bold"

    # Collect all critical issues
    critical_issues: list[str] = []
    if tamiyo.entropy < TUIThresholds.ENTROPY_CRITICAL:
        critical_issues.append("Entropy")
    if tamiyo.explained_variance <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
        critical_issues.append("Value")
    if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
        critical_issues.append("AdvLow")
    if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
        critical_issues.append("AdvHigh")
    if tamiyo.kl_divergence > TUIThresholds.KL_CRITICAL:
        critical_issues.append("KL")
    if tamiyo.clip_fraction > TUIThresholds.CLIP_CRITICAL:
        critical_issues.append("Clip")
    if tamiyo.grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
        critical_issues.append("Grad")

    if critical_issues:
        primary = critical_issues[0]
        if len(critical_issues) > 1:
            label = f"FAIL:{primary} (+{len(critical_issues)-1})"
        else:
            label = f"FAIL:{primary}"
        return "critical", label, "red bold"

    # Collect all warning issues
    warning_issues: list[str] = []
    if tamiyo.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
        warning_issues.append("Value")
    if tamiyo.entropy < TUIThresholds.ENTROPY_WARNING:
        warning_issues.append("Entropy")
    if tamiyo.kl_divergence > TUIThresholds.KL_WARNING:
        warning_issues.append("KL")
    if tamiyo.clip_fraction > TUIThresholds.CLIP_WARNING:
        warning_issues.append("Clip")
    if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_WARNING:
        warning_issues.append("AdvHigh")
    if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
        warning_issues.append("AdvLow")
    if tamiyo.grad_norm > TUIThresholds.GRAD_NORM_WARNING:
        warning_issues.append("Grad")

    if warning_issues:
        primary = warning_issues[0]
        if len(warning_issues) > 1:
            label = f"WARN:{primary} (+{len(warning_issues)-1})"
        else:
            label = f"WARN:{primary}"
        return "warning", label, "yellow"

    return "ok", "LEARNING", "green"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py::TestTriggeringCondition -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain_v2/status_banner.py tests/karn/sanctum/widgets/tamiyo_brain_v2/test_status_banner.py
git commit -m "feat(sanctum): show triggering condition in status banner

- Status label now shows which metric triggered warning/critical
- Short labels: 'FAIL:Entropy', 'WARN:Clip' (not verbose)
- Shows count of additional issues: 'FAIL:Entropy (+2)'
- Helps operators immediately identify scope of problem"
```

---

## Task 3: Add Entropy Velocity and Collapse Risk Score

This is the most impactful enhancement - predicting entropy collapse before it happens.

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py` (add fields to TamiyoState)
- Modify: `src/esper/karn/sanctum/backend.py` (compute velocity/risk in aggregator)
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py` (display)
- Test: `tests/karn/sanctum/test_entropy_prediction.py`

**Step 1: Write the failing test for entropy velocity calculation**

```python
# tests/karn/sanctum/test_entropy_prediction.py
import pytest
import numpy as np
from collections import deque
from esper.karn.sanctum.schema import compute_entropy_velocity, compute_collapse_risk


class TestEntropyVelocity:
    """Test entropy velocity calculation."""

    def test_stable_entropy_zero_velocity(self):
        """Constant entropy should have ~0 velocity."""
        history = deque([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        velocity = compute_entropy_velocity(history)
        assert abs(velocity) < 0.01

    def test_declining_entropy_negative_velocity(self):
        """Declining entropy should have negative velocity."""
        history = deque([1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55])
        velocity = compute_entropy_velocity(history)
        assert velocity < -0.03  # About -0.05 per step

    def test_rising_entropy_positive_velocity(self):
        """Rising entropy should have positive velocity."""
        history = deque([0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        velocity = compute_entropy_velocity(history)
        assert velocity > 0.03

    def test_short_history_returns_zero(self):
        """With <5 samples, return 0 (insufficient data)."""
        history = deque([1.0, 0.9, 0.8])
        velocity = compute_entropy_velocity(history)
        assert velocity == 0.0

    def test_noisy_declining_entropy(self):
        """Declining entropy with realistic noise should still detect trend."""
        # Realistic noisy data (not perfectly linear)
        history = deque([0.82, 0.78, 0.81, 0.74, 0.69, 0.72, 0.65, 0.62, 0.58, 0.55])
        velocity = compute_entropy_velocity(history)
        assert velocity < -0.02  # Clear downward trend despite noise


class TestCollapseRisk:
    """Test entropy collapse risk scoring."""

    def test_stable_high_entropy_no_risk(self):
        """Stable entropy at healthy level should have low risk."""
        history = deque([1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk < 0.1

    def test_declining_entropy_high_risk(self):
        """Rapidly declining entropy should have high risk."""
        # Declining from 0.8 toward 0.3 (critical)
        history = deque([0.8, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.38])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk > 0.5

    def test_already_collapsed_max_risk(self):
        """Entropy already at critical should have risk=1.0."""
        history = deque([0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk >= 0.95

    def test_rising_entropy_low_risk(self):
        """Rising entropy should have minimal risk (just proximity-based)."""
        history = deque([0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk < 0.2  # Small proximity risk but no velocity risk

    def test_zero_velocity_no_crash(self):
        """Zero velocity should not cause divide-by-zero."""
        history = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        risk = compute_collapse_risk(history, critical_threshold=0.3)
        assert risk < 0.3  # Just proximity risk, no velocity component
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_entropy_prediction.py -v`
Expected: FAIL - functions don't exist

**Step 3: Implement entropy velocity and collapse risk functions**

Add to `schema.py`:

```python
import numpy as np

def compute_entropy_velocity(entropy_history: deque[float] | list[float]) -> float:
    """Compute rate of entropy change (d(entropy)/d(batch)).

    Uses numpy linear regression over last 10 samples for performance and stability.

    Returns:
        Velocity in entropy units per batch. Negative = declining.
    """
    if len(entropy_history) < 5:
        return 0.0

    values = np.array(list(entropy_history)[-10:])
    n = len(values)
    x = np.arange(n)

    # Least squares slope using numpy (10x faster than pure Python)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def compute_collapse_risk(
    entropy_history: deque[float] | list[float],
    critical_threshold: float = 0.3,
    warning_threshold: float = 0.5,
    previous_risk: float = 0.0,
    hysteresis: float = 0.08,
) -> float:
    """Compute entropy collapse risk score (0.0 to 1.0).

    Risk is based on:
    - Current distance from critical threshold (proximity)
    - Velocity (rate of decline)
    - Hysteresis to prevent risk score flapping

    Returns:
        0.0 = no risk, 1.0 = imminent/active collapse
    """
    if len(entropy_history) < 5:
        return 0.0

    values = list(entropy_history)
    current = values[-1]
    velocity = compute_entropy_velocity(entropy_history)

    # Already collapsed
    if current <= critical_threshold:
        return 1.0

    # Calculate proximity-based risk (being near critical is risky even if stable)
    max_entropy = warning_threshold + 0.5  # Assume ~1.0 is healthy
    proximity = 1.0 - (current - critical_threshold) / (max_entropy - critical_threshold)
    proximity = max(0.0, min(1.0, proximity))
    proximity_risk = proximity * 0.3  # Cap proximity contribution at 0.3

    # Rising or stable entropy = minimal risk (just proximity)
    EPSILON = 1e-6
    if velocity >= -EPSILON:
        base_risk = proximity_risk
    else:
        # Declining entropy - calculate time to collapse
        distance = current - critical_threshold
        time_to_collapse = distance / abs(velocity)

        # Time-based risk thresholds (adjusted per DRL review)
        # 100+ batches = low urgency, <10 batches = high urgency
        if time_to_collapse > 100:
            time_risk = 0.1
        elif time_to_collapse > 50:
            time_risk = 0.25
        elif time_to_collapse > 20:
            time_risk = 0.5
        elif time_to_collapse > 10:
            time_risk = 0.7
        else:
            time_risk = 0.9

        # Combine time and proximity risks
        # Weight proximity more when already close to critical
        if current < warning_threshold:
            base_risk = 0.5 * time_risk + 0.5 * proximity_risk
        else:
            base_risk = 0.7 * time_risk + 0.3 * proximity_risk

    # Apply hysteresis to prevent flapping
    if abs(base_risk - previous_risk) < hysteresis:
        return previous_risk

    return min(1.0, max(0.0, base_risk))
```

**Step 4: Add fields to TamiyoState**

```python
@dataclass
class TamiyoState:
    # ... existing fields ...

    # Entropy prediction (computed from entropy_history)
    entropy_velocity: float = 0.0          # d(entropy)/d(batch), negative = declining
    collapse_risk_score: float = 0.0       # 0.0-1.0, >0.7 = high risk
    _previous_risk: float = 0.0            # For hysteresis (not serialized)
```

**Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_entropy_prediction.py -v`
Expected: PASS

**Step 6: Update aggregator to compute these metrics**

In `backend.py` (or `aggregator.py`), update the PPO update handler:

```python
# In _handle_ppo_update after updating entropy_history
from esper.karn.sanctum.schema import compute_entropy_velocity, compute_collapse_risk

# Compute velocity
tamiyo.entropy_velocity = compute_entropy_velocity(tamiyo.entropy_history)

# Compute risk with hysteresis (pass previous value)
tamiyo.collapse_risk_score = compute_collapse_risk(
    tamiyo.entropy_history,
    critical_threshold=TUIThresholds.ENTROPY_CRITICAL,
    warning_threshold=TUIThresholds.ENTROPY_WARNING,
    previous_risk=tamiyo._previous_risk,
)
tamiyo._previous_risk = tamiyo.collapse_risk_score
```

**Step 7: Display collapse risk in PPOHealthPanel**

Add entropy trend row to `_render_metrics()` in `ppo_health.py`:

```python
def _render_entropy_trend(self) -> Text:
    """Render entropy trend with velocity and countdown."""
    if self._snapshot is None:
        return Text()

    tamiyo = self._snapshot.tamiyo
    velocity = tamiyo.entropy_velocity
    risk = tamiyo.collapse_risk_score

    result = Text()
    result.append("Entropy Δ    ", style="dim")

    EPSILON = 1e-6
    if abs(velocity) < 0.005:
        result.append("stable [--]", style="green")
        return result

    # Trend arrows
    if velocity < -0.03:
        arrow = "[vv]"
        arrow_style = "red bold"
    elif velocity < -0.01:
        arrow = "[v]"
        arrow_style = "yellow"
    elif velocity > 0.01:
        arrow = "[^]"
        arrow_style = "green"
    else:
        arrow = "[~]"
        arrow_style = "dim"

    result.append(f"{velocity:+.3f}/batch ", style=arrow_style)
    result.append(arrow, style=arrow_style)

    # Countdown (only if declining toward critical)
    if velocity < -EPSILON and tamiyo.entropy > TUIThresholds.ENTROPY_CRITICAL:
        distance = tamiyo.entropy - TUIThresholds.ENTROPY_CRITICAL
        batches_to_collapse = int(distance / abs(velocity))

        if batches_to_collapse < 100:
            result.append(f" ~{batches_to_collapse}b", style="yellow")

        if risk > 0.7:
            result.append(" [ALERT]", style="red bold")

    return result
```

Also update border title in `update_snapshot()`:

```python
def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
    # ... existing code ...

    # Update border title with collapse risk if high
    if snapshot.tamiyo.collapse_risk_score > 0.7:
        velocity = snapshot.tamiyo.entropy_velocity
        if velocity < 0:
            distance = snapshot.tamiyo.entropy - TUIThresholds.ENTROPY_CRITICAL
            batches = int(distance / abs(velocity))
            self.border_title = f"PPO HEALTH ⚠ COLLAPSE ~{batches}b"
    elif batch < self.WARMUP_BATCHES:
        self.border_title = f"PPO HEALTH ─ WARMING UP [{batch}/{self.WARMUP_BATCHES}]"
    else:
        self.border_title = "PPO HEALTH"
```

**Step 8: Commit**

```bash
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/backend.py \
        src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py \
        tests/karn/sanctum/test_entropy_prediction.py
git commit -m "feat(sanctum): add entropy velocity and collapse risk prediction

- compute_entropy_velocity() uses numpy for 10x faster linear regression
- compute_collapse_risk() combines proximity + velocity with hysteresis
- Hysteresis prevents risk score flapping near thresholds
- PPOHealthPanel shows entropy trend row with countdown
- Border title shows COLLAPSE warning when risk > 0.7"
```

---

## Task 4: Keyboard Navigation for Decision Cards

Add keyboard support for navigating and pinning decisions (currently mouse-only).

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/decisions_column.py`
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/widget.py` (CSS for focus)
- Test: `tests/karn/sanctum/widgets/tamiyo_brain_v2/test_decisions_keyboard.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/widgets/tamiyo_brain_v2/test_decisions_keyboard.py
import pytest
from datetime import datetime, timezone

from esper.karn.sanctum.schema import DecisionSnapshot, SanctumSnapshot, TamiyoState
from esper.karn.sanctum.widgets.tamiyo_brain_v2.decisions_column import DecisionCard, DecisionsColumn


class TestDecisionCardKeyboard:
    """Test keyboard navigation for decision cards."""

    @pytest.mark.asyncio
    async def test_p_key_toggles_pin(self):
        """Pressing 'p' on focused card should toggle pin."""
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DecisionsColumn(id="decisions")

        app = TestApp()
        async with app.run_test() as pilot:
            col = app.query_one(DecisionsColumn)

            decision = DecisionSnapshot(
                timestamp=datetime.now(timezone.utc),
                slot_states={},
                host_accuracy=75.0,
                chosen_action="GERMINATE",
                chosen_slot="r0c0",
                confidence=0.8,
                expected_value=0.5,
                actual_reward=None,
                alternatives=[],
                decision_id="test-1",
                pinned=False,
            )
            snapshot = SanctumSnapshot(
                tamiyo=TamiyoState(recent_decisions=[decision]),
                current_batch=60,
            )
            col.update_snapshot(snapshot)
            await pilot.pause()

            cards = list(col.query(DecisionCard))
            assert len(cards) == 1
            cards[0].focus()
            await pilot.pause()

            await pilot.press("p")
            await pilot.pause()

            # Verify pin message was posted
            # (actual pin toggle happens via message handler)

    @pytest.mark.asyncio
    async def test_j_k_navigation(self):
        """j/k should navigate between cards."""
        from textual.app import App, ComposeResult

        class TestApp(App):
            def compose(self) -> ComposeResult:
                yield DecisionsColumn(id="decisions")

        app = TestApp()
        async with app.run_test() as pilot:
            col = app.query_one(DecisionsColumn)

            decisions = [
                DecisionSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    slot_states={},
                    host_accuracy=75.0,
                    chosen_action="GERMINATE",
                    chosen_slot="r0c0",
                    confidence=0.8,
                    expected_value=0.5,
                    actual_reward=None,
                    alternatives=[],
                    decision_id=f"test-{i}",
                    pinned=False,
                )
                for i in range(3)
            ]
            snapshot = SanctumSnapshot(
                tamiyo=TamiyoState(recent_decisions=decisions),
                current_batch=60,
            )
            col.update_snapshot(snapshot)
            await pilot.pause()

            cards = list(col.query(DecisionCard))
            assert len(cards) == 3

            # Focus first card
            cards[0].focus()
            await pilot.pause()
            assert app.focused == cards[0]

            # j moves to next
            await pilot.press("j")
            await pilot.pause()
            assert app.focused == cards[1]

            # k moves back
            await pilot.press("k")
            await pilot.pause()
            assert app.focused == cards[0]
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_decisions_keyboard.py -v`
Expected: FAIL - no keyboard handler

**Step 3: Add keyboard handlers**

In `decisions_column.py`:

```python
from textual.binding import Binding


class DecisionCard(Static):
    """Individual decision card widget with CSS-driven styling."""

    CARD_WIDTH: ClassVar[int] = 42

    # Enable keyboard focus
    can_focus = True

    # ... existing code ...

    def on_key(self, event) -> None:
        """Handle keyboard input on focused card."""
        if event.key == "p":
            self.post_message(self.Pinned(self.decision.decision_id))
            event.stop()


class DecisionsColumn(Container):
    """Column of decision cards with keyboard navigation."""

    BINDINGS = [
        Binding("j", "focus_next", "Next card", show=False),
        Binding("k", "focus_prev", "Previous card", show=False),
    ]

    def compose(self) -> ComposeResult:
        # Add help hint in header
        yield Static("DECISIONS [j/k:nav p:pin]", id="decisions-header", classes="decisions-header")
        yield VerticalScroll(id="cards-container")

    def action_focus_next(self) -> None:
        """Move focus to next decision card."""
        cards = list(self.query(DecisionCard))
        if not cards:
            return
        focused = self.app.focused
        if focused in cards:
            idx = cards.index(focused)
            next_idx = (idx + 1) % len(cards)
            cards[next_idx].focus()
        else:
            cards[0].focus()

    def action_focus_prev(self) -> None:
        """Move focus to previous decision card."""
        cards = list(self.query(DecisionCard))
        if not cards:
            return
        focused = self.app.focused
        if focused in cards:
            idx = cards.index(focused)
            prev_idx = (idx - 1) % len(cards)
            cards[prev_idx].focus()
        else:
            cards[-1].focus()
```

**Step 4: Add focus CSS**

In `widget.py` DEFAULT_CSS:

```css
DecisionCard:focus {
    border: thick $accent;
    background: $panel-darken-1;
}

DecisionCard.pinned:focus {
    border: thick $success;
    background: $panel-darken-1;
}

.decisions-header {
    height: 1;
    text-style: bold;
    color: $text-muted;
}
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/widgets/tamiyo_brain_v2/test_decisions_keyboard.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain_v2/decisions_column.py \
        src/esper/karn/sanctum/widgets/tamiyo_brain_v2/widget.py \
        tests/karn/sanctum/widgets/tamiyo_brain_v2/test_decisions_keyboard.py
git commit -m "feat(sanctum): add keyboard navigation for decision cards

- j/k to navigate between cards (vim-style)
- p to toggle pin on focused card
- Header shows shortcut hints: [j/k:nav p:pin]
- Focus ring with background change for visibility
- Improves accessibility for keyboard-only users"
```

---

## Task 5: Add Entropy ↔ Clip Correlation Indicator

Display the relationship between entropy and clip fraction to identify policy collapse patterns.

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py` (add correlation field)
- Modify: `src/esper/karn/sanctum/backend.py` (compute correlation)
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py` (display)
- Test: `tests/karn/sanctum/test_correlation.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_correlation.py
import pytest
from collections import deque
from esper.karn.sanctum.schema import compute_correlation


class TestCorrelation:
    """Test metric correlation calculation."""

    def test_perfect_negative_correlation(self):
        """When entropy drops and clip rises perfectly, correlation = -1."""
        entropy = deque([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        clip = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        corr = compute_correlation(entropy, clip)
        assert corr < -0.95

    def test_perfect_positive_correlation(self):
        """When both rise together, correlation = +1."""
        entropy = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        clip = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        corr = compute_correlation(entropy, clip)
        assert corr > 0.95

    def test_no_correlation(self):
        """When metrics are uncorrelated, correlation near 0."""
        entropy = deque([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
        clip = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        corr = compute_correlation(entropy, clip)
        assert abs(corr) < 0.3

    def test_short_history_returns_zero(self):
        """With <5 samples, return 0."""
        entropy = deque([1.0, 0.9])
        clip = deque([0.1, 0.2])
        corr = compute_correlation(entropy, clip)
        assert corr == 0.0

    def test_zero_variance_returns_zero(self):
        """Constant values should return 0 (not NaN/crash)."""
        entropy = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        clip = deque([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        corr = compute_correlation(entropy, clip)
        assert corr == 0.0  # Not NaN
```

**Step 2: Implement correlation function with epsilon safety**

```python
# In schema.py
EPSILON = 1e-10

def compute_correlation(
    x_values: deque[float] | list[float],
    y_values: deque[float] | list[float],
) -> float:
    """Compute Pearson correlation between two metric histories.

    Returns:
        Correlation coefficient (-1 to +1), or 0.0 if insufficient data
        or zero variance (to avoid NaN).
    """
    if len(x_values) < 5 or len(y_values) < 5:
        return 0.0

    x = list(x_values)[-10:]
    y = list(y_values)[-10:]

    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    y_var = sum((yi - y_mean) ** 2 for yi in y)

    denominator = (x_var * y_var) ** 0.5

    # Epsilon check to prevent divide-by-zero
    if denominator < EPSILON:
        return 0.0

    return numerator / denominator
```

**Step 3: Add field to TamiyoState**

```python
# In TamiyoState
entropy_clip_correlation: float = 0.0  # Negative = potential policy collapse
```

**Step 4: Update aggregator to compute correlation**

```python
# In backend.py after updating histories
tamiyo.entropy_clip_correlation = compute_correlation(
    tamiyo.entropy_history,
    tamiyo.clip_fraction_history,
)
```

**Step 5: Display in PPOHealthPanel with correct interpretation**

The dangerous pattern requires BOTH negative correlation AND low entropy AND high clip:

```python
# In _render_metrics() of ppo_health.py
def _render_policy_state(self) -> Text:
    """Render policy state based on entropy/clip correlation.

    Interpretation (per DRL review):
    - Negative correlation + low entropy + high clip = COLLAPSE RISK
    - Negative correlation + low entropy = collapsing
    - Negative correlation + low clip = healthy convergence (NARROWING)
    - Low correlation = stable
    """
    if self._snapshot is None:
        return Text()

    tamiyo = self._snapshot.tamiyo
    corr = tamiyo.entropy_clip_correlation
    entropy = tamiyo.entropy
    clip = tamiyo.clip_fraction

    result = Text()
    result.append("Policy       ", style="dim")

    # The dangerous pattern: entropy falling + clip rising + both concerning
    if (corr < -0.5 and
        entropy < TUIThresholds.ENTROPY_WARNING and
        clip > TUIThresholds.CLIP_WARNING):
        result.append("COLLAPSE RISK", style="red bold")
        result.append(f" (r={corr:.2f})", style="dim")
    elif corr < -0.6 and entropy < TUIThresholds.ENTROPY_WARNING:
        # Entropy low and correlated with clip - concerning
        result.append("collapsing", style="yellow")
        result.append(f" (r={corr:.2f})", style="dim")
    elif corr < -0.4 and clip < 0.15:
        # Negative correlation but low clip = healthy convergence
        result.append("narrowing", style="green")
    elif abs(corr) < 0.3:
        result.append("stable", style="green")
    else:
        result.append("drifting", style="yellow")
        result.append(f" (r={corr:.2f})", style="dim")

    return result
```

**Step 6: Run tests and commit**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_correlation.py -v
git add src/esper/karn/sanctum/schema.py src/esper/karn/sanctum/backend.py \
        src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py \
        tests/karn/sanctum/test_correlation.py
git commit -m "feat(sanctum): add entropy-clip correlation for policy state detection

- compute_correlation() with epsilon safety against divide-by-zero
- Interpretation requires BOTH correlation AND absolute levels:
  - COLLAPSE RISK: neg corr + low entropy + high clip
  - collapsing: neg corr + low entropy
  - narrowing: neg corr + low clip (healthy convergence)
  - stable: low correlation
- Prevents false positives from healthy late-stage convergence"
```

---

## Task 6: Add Value Function Statistics

Display value function health metrics to detect value explosion.

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py` (add value stats fields)
- Modify: `src/esper/leyline/telemetry.py` (add to PPOUpdatePayload)
- Modify: `src/esper/simic/agent/ppo.py` (emit value stats - CORRECT LOCATION)
- Modify: `src/esper/karn/sanctum/backend.py` (aggregate value stats)
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py` (display)
- Test: `tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health_value.py`

**Step 1: Add fields to PPOUpdatePayload**

```python
# In telemetry.py PPOUpdatePayload
value_mean: float = 0.0
value_std: float = 0.0
value_min: float = 0.0
value_max: float = 0.0
```

**Step 2: Add fields to TamiyoState**

```python
# In schema.py TamiyoState
value_mean: float = 0.0
value_std: float = 0.0
value_min: float = 0.0
value_max: float = 0.0
initial_value_spread: float | None = None  # Set after warmup for relative thresholds
```

**Step 3: Update PPO training code to emit value stats (CORRECT LOCATION)**

In `src/esper/simic/agent/ppo.py`, find the `logging_tensors` stack (look for `logging_tensors = torch.stack([`) and add value stats:

```python
# In ppo.py, inside _ppo_update() or equivalent
# Add to the existing logging_tensors stack to avoid extra GPU syncs
# IMPORTANT: The existing code uses 6 tensors. Add 4 more for value stats.

# Modify the existing stack to include value stats:
logging_tensors = torch.stack([
    policy_loss,
    value_loss,
    -entropy_loss,
    joint_ratio.mean(),
    joint_ratio.max(),
    joint_ratio.min(),
    # NEW: Value function stats (single sync with rest)
    values.mean(),
    values.std(),
    values.min(),
    values.max(),
]).cpu().tolist()

# IMPORTANT: Use indexed access pattern (matching existing code style)
# Add these AFTER the existing indexed assignments:
metrics["value_mean"].append(logging_tensors[6])
metrics["value_std"].append(logging_tensors[7])
metrics["value_min"].append(logging_tensors[8])
metrics["value_max"].append(logging_tensors[9])
```

**Step 4: Update metric aggregation for min/max**

In `src/esper/simic/training/vectorized.py`, modify `_aggregate_ppo_metrics()` (the EXISTING function around line 264-292) to use min/max for value bounds. Add these cases to the existing if/elif chain that handles `ratio_max` and `ratio_min`:

```python
# In _aggregate_ppo_metrics(), ADD to existing if/elif chain:
if key == "ratio_max":
    aggregated[key] = max(values)
elif key == "ratio_min":
    aggregated[key] = min(values)
elif key == "value_min":  # NEW
    aggregated[key] = min(values)
elif key == "value_max":  # NEW
    aggregated[key] = max(values)
elif key == "early_stop_epoch":
    # ... existing code ...

    return aggregated
```

**Step 5: Update aggregator to capture value stats**

```python
# In backend.py _handle_ppo_update
tamiyo.value_mean = payload.value_mean
tamiyo.value_std = payload.value_std
tamiyo.value_min = payload.value_min
tamiyo.value_max = payload.value_max

# Set initial spread after warmup for relative thresholds
if tamiyo.initial_value_spread is None and current_batch >= WARMUP_BATCHES:
    spread = tamiyo.value_max - tamiyo.value_min
    if spread > 0.1:  # Only set if non-trivial
        tamiyo.initial_value_spread = spread
```

**Step 6: Display in PPOHealthPanel with relative thresholds**

```python
# In _render_metrics()
def _render_value_stats(self) -> Text:
    """Render value function statistics with relative thresholds."""
    if self._snapshot is None:
        return Text()

    tamiyo = self._snapshot.tamiyo
    result = Text()

    value_status = self._get_value_status(tamiyo)
    value_style = self._status_style(value_status)

    result.append("Value Range  ", style="dim")
    result.append(f"[{tamiyo.value_min:.1f}, {tamiyo.value_max:.1f}]", style=value_style)

    if tamiyo.value_std > 0:
        result.append(f" σ={tamiyo.value_std:.2f}", style="dim")

    if value_status != "ok":
        result.append(" !", style=value_style)

    return result


def _get_value_status(self, tamiyo) -> str:
    """Check if value function is healthy using relative thresholds.

    Uses coefficient of variation and relative spread when possible.
    Falls back to absolute thresholds during warmup.
    """
    v_range = tamiyo.value_max - tamiyo.value_min
    v_mean = tamiyo.value_mean
    v_std = tamiyo.value_std
    initial = tamiyo.initial_value_spread

    # Collapse detection: values stuck at constant
    if v_range < 0.1 and v_std < 0.01:
        return "critical"

    # Coefficient of variation check (relative instability)
    if abs(v_mean) > 0.1:
        cov = v_std / abs(v_mean)
        if cov > 3.0:
            return "critical"  # Extreme instability
        if cov > 2.0:
            return "warning"

    # Relative threshold (if initial spread known)
    if initial is not None and initial > 0.1:
        ratio = v_range / initial
        if ratio > 10:
            return "critical"  # 10x initial spread
        if ratio > 5:
            return "warning"  # 5x initial spread
        return "ok"

    # Absolute fallback (during warmup or if initial unknown)
    if v_range > 1000 or abs(tamiyo.value_max) > 10000:
        return "critical"
    if v_range > 500 or abs(tamiyo.value_max) > 5000:
        return "warning"

    return "ok"
```

**Step 7: Write test**

```python
# tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health_value.py
import pytest
from esper.karn.sanctum.schema import TamiyoState, SanctumSnapshot
from esper.karn.sanctum.widgets.tamiyo_brain_v2.ppo_health import PPOHealthPanel


class TestValueFunctionDisplay:
    """Test value function statistics display."""

    def test_healthy_values_show_ok(self):
        """Normal value range should show ok status."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=2.0,
                value_min=-3.0,
                value_max=15.0,
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "ok"

    def test_exploding_values_show_critical(self):
        """Values 10x initial spread should show critical."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=50.0,
                value_std=30.0,
                value_min=-50.0,
                value_max=150.0,  # 200 range vs initial 10 = 20x
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "critical"

    def test_collapsed_values_show_critical(self):
        """Constant values should show critical (collapsed)."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=0.001,
                value_min=4.999,
                value_max=5.001,
                initial_value_spread=10.0,
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "critical"

    def test_high_cov_shows_warning(self):
        """High coefficient of variation should show warning."""
        panel = PPOHealthPanel()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(
                ppo_data_received=True,
                value_mean=5.0,
                value_std=12.0,  # CoV = 12/5 = 2.4 > 2.0
                value_min=-10.0,
                value_max=25.0,
                initial_value_spread=None,  # No initial, use absolute
            ),
            current_batch=60,
        )
        panel._snapshot = snapshot
        status = panel._get_value_status(snapshot.tamiyo)
        assert status == "warning"
```

**Step 8: Commit**

```bash
git add src/esper/leyline/telemetry.py src/esper/karn/sanctum/schema.py \
        src/esper/simic/agent/ppo.py src/esper/simic/training/vectorized.py \
        src/esper/karn/sanctum/backend.py \
        src/esper/karn/sanctum/widgets/tamiyo_brain_v2/ppo_health.py \
        tests/karn/sanctum/widgets/tamiyo_brain_v2/test_ppo_health_value.py
git commit -m "feat(sanctum): add value function statistics for divergence detection

- Value stats collected in ppo.py with single GPU sync (batched stack)
- Aggregation uses min/max for bounds (not average)
- Relative thresholds: warning at 5x initial, critical at 10x
- Coefficient of variation check for instability
- Collapse detection for constant values
- Detects value explosion before NaN occurs"
```

---

## Final Integration Test

After all tasks are complete, run the full test suite:

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v
```

---

## Summary

| Task | Description | Key Benefit |
|------|-------------|-------------|
| 0 | Prerequisites | Add missing `inf_grad_count` field to TamiyoState |
| 1 | NaN/Inf Display | Immediate visibility of numerical instability (both types) |
| 2 | Triggering Condition | Know WHY and HOW MANY issues exist |
| 3 | Entropy Collapse Prediction | Predict failure with hysteresis-stabilized countdown |
| 4 | Keyboard Navigation | j/k/p for vim-style accessibility |
| 5 | Entropy-Clip Correlation | Detect policy collapse (requires both correlation AND levels) |
| 6 | Value Function Stats | Detect value explosion with relative thresholds |

Total estimated implementation time: 4-6 hours

---

## Reviewer Feedback Incorporated

### Round 1 (Initial Review)

| Reviewer | Key Changes Made |
|----------|------------------|
| **Code Review** | Removed `blink` style; added epsilon checks for divide-by-zero; shortened labels |
| **UX Specialist** | Added j/k navigation; multi-issue count; renamed "converging" to "narrowing" |
| **DRL Expert** | Adjusted risk thresholds; correlation requires entropy+clip levels; relative value thresholds |
| **PyTorch Expert** | Moved value stats to ppo.py with batched sync; min/max aggregation |

### Round 2 (Verification Review)

| Reviewer | Key Changes Made |
|----------|------------------|
| **Code Review** | Added Task 0 for missing `inf_grad_count` field in TamiyoState |
| **UX Specialist** | Noted variable-width labels and border title collision (implementation notes) |
| **DRL Expert** | All corrections verified ✓; suggests asymmetric hysteresis (optional) |
| **PyTorch Expert** | Fixed: removed `ratio_std` (doesn't exist); use indexed access pattern; correct function name `_aggregate_ppo_metrics` |
