# Fix Per-Head Entropy Collapse Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent individual action heads (especially blueprint) from collapsing to deterministic behavior while total policy entropy appears healthy.

**Architecture:** Add per-head entropy floor penalties to the PPO loss function, plus per-head collapse detection in anomaly monitoring. The floor penalty creates a soft constraint that pushes entropy back up when a head drops below threshold, while detection provides early warning in telemetry.

**Tech Stack:** Python, PyTorch, pytest, Esper PPO agent (simic domain)

---

## Background

### The Problem

Telemetry from run `2026-01-08_191448` shows:
- **Blueprint entropy collapsed by batch 16**: Dropped from 0.12 to 0.008
- **`entropy_collapsed` flag stayed False**: Total policy entropy was still 7.0
- **conv_heavy dominated**: 55% of germinations (2808/5364) vs 6-15% for others

### Root Cause

1. **Sparse activation**: Blueprint head only fires during GERMINATE (~18% of steps), receiving 5x less gradient signal than the op head
2. **Joint entropy masking**: The single `entropy_collapsed` flag checks total entropy, missing per-head collapse
3. **Current coefficients insufficient**: `blueprint: 1.3` is not enough for a head active only 18% of the time

---

## Peer Review Amendments

**Reviewed by:** Gemini, ChatGPT Pro, DRL Expert, PyTorch Expert

### Critical Fixes Applied:
1. **Zero valid steps bug**: Skip floor penalty when head has no activations in batch
2. **Per-head penalty coefficients**: Different penalty strength per head (sparse heads need stronger signal)
3. **Redundant tensor creation**: Use Python scalars instead of `torch.tensor(floor)` in loop
4. **Collapse detection hysteresis**: Require N=3 consecutive collapses, 1.5x recovery threshold
5. **Telemetry visibility**: Return `entropy_floor_loss` separately in `LossMetrics`

### Confirmed Non-Issues:
- **Entropy units**: Already normalized to [0, 1] per head in `action_masks.py:546-578`
- **Floor values**: Correctly calibrated for normalized entropy

---

## Task 1: Add Per-Head Entropy Floor Constants to Leyline

**Files:**
- Modify: `src/esper/leyline/__init__.py:182-183`

**Step 1: Add entropy floor constants and penalty coefficients**

After the existing entropy thresholds (line 183), add:

```python
# Per-head entropy floor thresholds (normalized 0-1 scale)
# Higher floors for sparse heads that receive fewer gradient signals
# These are SOFT floors enforced via quadratic penalty in PPO loss
ENTROPY_FLOOR_PER_HEAD: dict[str, float] = {
    "op": 0.15,           # Always active (100% of steps) - can exploit more
    "slot": 0.20,         # Usually active (~60%)
    "blueprint": 0.40,    # GERMINATE only (~18%) - CRITICAL: needs high floor
    "style": 0.30,        # GERMINATE + SET_ALPHA_TARGET (~22%)
    "tempo": 0.40,        # GERMINATE only (~18%) - needs high floor
    "alpha_target": 0.25, # GERMINATE + SET_ALPHA_TARGET (~22%)
    "alpha_speed": 0.20,  # SET_ALPHA_TARGET + PRUNE (~19%)
    "alpha_curve": 0.20,  # SET_ALPHA_TARGET + PRUNE (~19%)
}

# Per-head entropy collapse thresholds (stricter than floor for detection)
# Entropy below this triggers anomaly detection
ENTROPY_COLLAPSE_PER_HEAD: dict[str, float] = {
    "op": 0.08,
    "slot": 0.10,
    "blueprint": 0.05,  # Lower threshold but still detect collapse
    "style": 0.08,
    "tempo": 0.05,
    "alpha_target": 0.08,
    "alpha_speed": 0.08,
    "alpha_curve": 0.08,
}

# Per-head entropy floor penalty coefficients (DRL Expert recommendation)
# Sparse heads need stronger penalty signal to compensate for fewer gradients
ENTROPY_FLOOR_PENALTY_COEF: dict[str, float] = {
    "op": 0.05,           # Always active - minimal penalty needed
    "slot": 0.10,         # Usually active
    "blueprint": 0.20,    # GERMINATE only - needs stronger signal
    "style": 0.15,        # GERMINATE + SET_ALPHA_TARGET
    "tempo": 0.20,        # GERMINATE only - needs stronger signal
    "alpha_target": 0.12, # GERMINATE + SET_ALPHA_TARGET
    "alpha_speed": 0.10,  # SET_ALPHA_TARGET + PRUNE
    "alpha_curve": 0.10,  # SET_ALPHA_TARGET + PRUNE
}
```

**Step 2: Add to `__all__` export list**

Find the `__all__` list and add:

```python
    "ENTROPY_FLOOR_PER_HEAD",
    "ENTROPY_COLLAPSE_PER_HEAD",
    "ENTROPY_FLOOR_PENALTY_COEF",
```

**Step 3: Run import test**

Run: `uv run python -c "from esper.leyline import ENTROPY_FLOOR_PER_HEAD, ENTROPY_COLLAPSE_PER_HEAD, ENTROPY_FLOOR_PENALTY_COEF; print(ENTROPY_FLOOR_PENALTY_COEF)"`

Expected: Dict prints successfully

**Step 4: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "feat(leyline): add per-head entropy floor and collapse thresholds

Sparse heads (blueprint, tempo) receive fewer gradient signals due to
causal masking, making them prone to premature convergence. These
thresholds define soft floors for the entropy penalty and hard floors
for anomaly detection.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Entropy Floor Penalty Function

**Files:**
- Modify: `src/esper/simic/agent/ppo_update.py:197-212`
- Test: `tests/simic/agent/test_ppo_entropy_floor.py`

**Step 1: Write the failing test**

Create `tests/simic/agent/test_ppo_entropy_floor.py`:

```python
"""Tests for per-head entropy floor penalty in PPO loss."""

import pytest
import torch

from esper.simic.agent.ppo_update import compute_entropy_floor_penalty


class TestEntropyFloorPenalty:
    """Tests for per-head entropy floor penalty function."""

    def test_no_penalty_when_above_floor(self) -> None:
        """Entropy above floor should incur no penalty."""
        entropy = {"blueprint": torch.tensor(0.5)}  # Above 0.4 floor
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor)

        assert penalty.item() == pytest.approx(0.0)

    def test_penalty_when_below_floor(self) -> None:
        """Entropy below floor should incur quadratic penalty."""
        entropy = {"blueprint": torch.tensor(0.1)}  # Below 0.4 floor
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor)

        # Shortfall = 0.4 - 0.1 = 0.3, penalty = 0.1 * 0.3^2 = 0.009
        assert penalty.item() > 0
        assert penalty.item() == pytest.approx(0.1 * (0.3 ** 2), rel=0.01)

    def test_penalty_scales_with_shortfall(self) -> None:
        """Larger shortfall should incur larger penalty."""
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}

        # Small shortfall
        entropy_small = {"blueprint": torch.tensor(0.35)}
        penalty_small = compute_entropy_floor_penalty(entropy_small, head_masks, floor)

        # Large shortfall
        entropy_large = {"blueprint": torch.tensor(0.1)}
        penalty_large = compute_entropy_floor_penalty(entropy_large, head_masks, floor)

        assert penalty_large > penalty_small

    def test_penalty_respects_mask(self) -> None:
        """Penalty should only consider masked (active) timesteps."""
        # Half the timesteps are masked (inactive)
        mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        head_masks = {"blueprint": mask}
        floor = {"blueprint": 0.4}

        # Per-step entropy: first 5 have 0.5, last 5 have 0.0
        # Masked mean should be 0.5 (above floor) -> no penalty
        per_step_entropy = torch.tensor([0.5] * 5 + [0.0] * 5)
        entropy = {"blueprint": per_step_entropy}

        # Note: compute_entropy_floor_penalty expects pre-computed mean entropy per head
        # Actually, we need to check how it handles per-step vs scalar entropy
        # For now, test with scalar (mean over masked steps)
        mean_ent = (per_step_entropy * mask).sum() / mask.sum()
        entropy_scalar = {"blueprint": mean_ent}

        penalty = compute_entropy_floor_penalty(entropy_scalar, head_masks, floor)
        assert penalty.item() == pytest.approx(0.0)

    def test_multiple_heads(self) -> None:
        """Penalty should sum across all heads."""
        entropy = {
            "blueprint": torch.tensor(0.1),  # Below 0.4 -> penalty
            "op": torch.tensor(0.5),         # Above 0.15 -> no penalty
        }
        head_masks = {
            "blueprint": torch.ones(10),
            "op": torch.ones(10),
        }
        floor = {"blueprint": 0.4, "op": 0.15}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor)

        # Only blueprint should contribute
        expected = 0.1 * (0.3 ** 2)  # shortfall 0.3
        assert penalty.item() == pytest.approx(expected, rel=0.01)


class TestEntropyFloorEdgeCases:
    """Edge case tests added per expert review."""

    def test_no_penalty_when_head_inactive(self) -> None:
        """Heads with no valid steps should NOT incur penalty (critical fix)."""
        # Blueprint head is completely masked (no valid steps)
        entropy = {"blueprint": torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])}
        head_masks = {"blueprint": torch.zeros(5)}  # All masked!
        floor = {"blueprint": 0.4}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor)

        # Should be 0 - head was never active
        assert penalty.item() == pytest.approx(0.0), \
            f"Expected no penalty for inactive head, got {penalty.item()}"

    def test_per_head_penalty_coefficients(self) -> None:
        """Different heads should use different penalty strengths."""
        entropy = {
            "op": torch.tensor(0.1),       # Below floor 0.15, shortfall 0.05
            "blueprint": torch.tensor(0.1) # Below floor 0.40, shortfall 0.30
        }
        head_masks = {k: torch.ones(10) for k in entropy}
        floor = {"op": 0.15, "blueprint": 0.40}
        coefs = {"op": 0.05, "blueprint": 0.20}

        penalty = compute_entropy_floor_penalty(entropy, head_masks, floor, coefs)

        # op: 0.05 * 0.05^2 = 0.000125
        # bp: 0.20 * 0.30^2 = 0.018
        expected = 0.000125 + 0.018
        assert penalty.item() == pytest.approx(expected, rel=0.01)

    def test_entropy_floor_gradient_direction(self) -> None:
        """Entropy below floor should have gradient pushing it up."""
        entropy_param = torch.tensor(0.1, requires_grad=True)
        entropy_dict = {"blueprint": entropy_param}
        head_masks = {"blueprint": torch.ones(10)}
        floor = {"blueprint": 0.4}

        penalty = compute_entropy_floor_penalty(entropy_dict, head_masks, floor)
        penalty.backward()

        # Gradient should be negative (pushes entropy up to reduce penalty)
        assert entropy_param.grad < 0, f"Expected negative gradient, got {entropy_param.grad}"

    def test_empty_entropy_dict_returns_zero(self) -> None:
        """Empty entropy dict should return zero penalty."""
        penalty = compute_entropy_floor_penalty({}, {}, {"blueprint": 0.4})
        assert penalty.item() == pytest.approx(0.0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/agent/test_ppo_entropy_floor.py -v`

Expected: FAIL with "ImportError: cannot import name 'compute_entropy_floor_penalty'"

**Step 3: Implement the entropy floor penalty function (AMENDED per expert review)**

In `src/esper/simic/agent/ppo_update.py`, add before `compute_losses()`:

```python
def compute_entropy_floor_penalty(
    entropy: dict[str, torch.Tensor],
    head_masks: dict[str, torch.Tensor],
    entropy_floor: dict[str, float],
    penalty_coef: dict[str, float] | float = 0.1,
) -> torch.Tensor:
    """Compute penalty for heads whose entropy falls below floor.

    Uses quadratic penalty: loss += coef * max(0, floor - entropy)^2
    This creates smooth gradient pressure to maintain minimum entropy.

    CRITICAL: Skips heads with no valid steps (n_valid < 1) to avoid
    penalizing inactive heads that had no opportunity to maintain entropy.

    Args:
        entropy: Dict of head_name -> entropy tensor (per-step or scalar, normalized 0-1)
        head_masks: Dict of head_name -> mask tensor (for device inference and masking)
        entropy_floor: Dict of head_name -> minimum acceptable entropy (float)
        penalty_coef: Per-head coefficients dict OR global scalar (default 0.1)

    Returns:
        Scalar penalty to add to total loss (larger = more penalty)
    """
    if not entropy:
        # Early return if no entropy provided (defensive)
        return torch.tensor(0.0)

    # Get device from first entropy tensor
    first_entropy = next(iter(entropy.values()))
    device = first_entropy.device
    penalty = torch.tensor(0.0, device=device)

    for head, floor in entropy_floor.items():
        if head not in entropy:
            continue

        head_ent = entropy[head]

        # If per-step entropy, compute mean over valid steps
        if head_ent.ndim > 0:
            mask = head_masks.get(head)
            if mask is not None:
                n_valid = mask.sum()
                if n_valid < 1:
                    # CRITICAL FIX: Skip heads with no valid steps
                    # (no opportunity to maintain entropy)
                    continue
                head_ent = (head_ent * mask).sum() / n_valid
            else:
                head_ent = head_ent.mean()

        # Get per-head penalty coefficient
        if isinstance(penalty_coef, dict):
            head_coef = penalty_coef.get(head, 0.1)
        else:
            head_coef = penalty_coef

        # Quadratic penalty for entropy below floor
        # Use Python scalar for floor - PyTorch broadcasts efficiently
        shortfall = torch.clamp(floor - head_ent, min=0.0)
        penalty = penalty + head_coef * (shortfall ** 2)

    return penalty
```

**Step 4: Add to `__all__` export**

Update `__all__` at the bottom of `ppo_update.py`:

```python
__all__ = [
    "RatioMetrics",
    "LossMetrics",
    "PPOUpdateResult",
    "compute_ratio_metrics",
    "compute_losses",
    "compute_entropy_floor_penalty",
]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/simic/agent/test_ppo_entropy_floor.py -v`

Expected: All 9 tests PASS (5 basic + 4 edge cases)

**Step 6: Commit**

```bash
git add src/esper/simic/agent/ppo_update.py tests/simic/agent/test_ppo_entropy_floor.py
git commit -m "feat(ppo): add compute_entropy_floor_penalty function

Quadratic penalty for heads whose entropy falls below floor threshold.
This prevents per-head entropy collapse while allowing natural
convergence above the floor.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Integrate Entropy Floor into PPO Loss

**Files:**
- Modify: `src/esper/simic/agent/ppo_update.py:212` (compute_losses function)
- Test: `tests/simic/agent/test_ppo_entropy_floor.py`

**Step 1: Add integration test**

Append to `tests/simic/agent/test_ppo_entropy_floor.py`:

```python
from esper.simic.agent.ppo_update import compute_losses, LossMetrics


class TestEntropyFloorIntegration:
    """Integration tests for entropy floor in compute_losses."""

    def test_compute_losses_includes_entropy_floor_penalty(self) -> None:
        """compute_losses should include entropy floor penalty when enabled."""
        device = torch.device("cpu")
        batch_size = 32

        # Create minimal inputs for compute_losses
        log_probs = torch.randn(batch_size, device=device)
        old_log_probs = torch.randn(batch_size, device=device)
        advantages = torch.randn(batch_size, device=device)
        values = torch.randn(batch_size, device=device)
        returns = torch.randn(batch_size, device=device)

        # Create entropy dict with collapsed blueprint head
        entropy = {
            "op": torch.full((batch_size,), 0.5),      # Healthy
            "blueprint": torch.full((batch_size,), 0.05),  # Collapsed!
            "slot": torch.full((batch_size,), 0.4),
            "style": torch.full((batch_size,), 0.4),
            "tempo": torch.full((batch_size,), 0.4),
            "alpha_target": torch.full((batch_size,), 0.4),
            "alpha_speed": torch.full((batch_size,), 0.4),
            "alpha_curve": torch.full((batch_size,), 0.4),
        }

        # All heads active for simplicity
        head_masks = {k: torch.ones(batch_size) for k in entropy}

        # Without floor penalty
        losses_no_floor = compute_losses(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            returns=returns,
            entropy=entropy,
            head_masks=head_masks,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            entropy_coef_per_head={"op": 1.0, "blueprint": 1.3, "slot": 1.0,
                                   "style": 1.2, "tempo": 1.3, "alpha_target": 1.2,
                                   "alpha_speed": 1.2, "alpha_curve": 1.2},
            entropy_floor=None,  # Disabled
        )

        # With floor penalty
        entropy_floor = {"blueprint": 0.4}  # Blueprint is at 0.05, well below 0.4
        losses_with_floor = compute_losses(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            returns=returns,
            entropy=entropy,
            head_masks=head_masks,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            entropy_coef_per_head={"op": 1.0, "blueprint": 1.3, "slot": 1.0,
                                   "style": 1.2, "tempo": 1.3, "alpha_target": 1.2,
                                   "alpha_speed": 1.2, "alpha_curve": 1.2},
            entropy_floor=entropy_floor,
        )

        # Total loss should be higher with floor penalty
        assert losses_with_floor.total_loss > losses_no_floor.total_loss
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/agent/test_ppo_entropy_floor.py::TestEntropyFloorIntegration -v`

Expected: FAIL (entropy_floor parameter doesn't exist yet)

**Step 3: Modify compute_losses signature**

In `src/esper/simic/agent/ppo_update.py`, update the `compute_losses` function signature (around line 130) to add the `entropy_floor` parameter:

```python
def compute_losses(
    *,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: dict[str, torch.Tensor],
    head_masks: dict[str, torch.Tensor],
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    entropy_coef_per_head: dict[str, float],
    value_clip_range: float | None = None,
    old_values: torch.Tensor | None = None,
    actor_weight: torch.Tensor | None = None,
    entropy_floor: dict[str, float] | None = None,  # NEW: per-head entropy floors
) -> LossMetrics:
```

**Step 4: Add floor penalty to loss computation**

After the entropy loss loop (around line 210), add:

```python
    # Per-head entropy floor penalty (prevents sparse heads from collapsing)
    entropy_floor_penalty = torch.tensor(0.0, device=values.device)
    if entropy_floor is not None:
        # Compute mean entropy per head for floor comparison
        mean_entropy: dict[str, torch.Tensor] = {}
        for key in entropy:
            mask = head_masks[key]
            if actor_weight is not None:
                effective_mask = mask * actor_weight
            else:
                effective_mask = mask
            n_valid = effective_mask.sum().clamp(min=1)
            mean_entropy[key] = (entropy[key] * effective_mask).sum() / n_valid

        entropy_floor_penalty = compute_entropy_floor_penalty(
            entropy=mean_entropy,
            head_masks=head_masks,
            entropy_floor=entropy_floor,
        )

    total_loss = (
        policy_loss
        + value_coef * value_loss
        + entropy_coef * entropy_loss
        + entropy_floor_penalty  # NEW
    )
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/simic/agent/test_ppo_entropy_floor.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/agent/ppo_update.py tests/simic/agent/test_ppo_entropy_floor.py
git commit -m "feat(ppo): integrate entropy floor penalty into compute_losses

Adds optional entropy_floor parameter to compute_losses(). When enabled,
heads with entropy below their floor incur a quadratic penalty, pushing
the policy back toward exploration.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Wire Entropy Floor Through PPOAgent

**Files:**
- Modify: `src/esper/simic/agent/ppo_agent.py`

**Step 1: Read the current PPOAgent.update() method**

Find the call to `compute_losses()` in `ppo_agent.py` and identify where to pass `entropy_floor`.

**Step 2: Add entropy_floor to PPOAgent config**

Add to PPOAgent's `__init__` parameters and store as instance attribute:

```python
from esper.leyline import ENTROPY_FLOOR_PER_HEAD

# In __init__:
self.entropy_floor = entropy_floor if entropy_floor is not None else ENTROPY_FLOOR_PER_HEAD
```

**Step 3: Pass entropy_floor to compute_losses**

In the `update()` method, add `entropy_floor=self.entropy_floor` to the `compute_losses()` call.

**Step 4: Run existing PPO tests**

Run: `uv run pytest tests/simic/agent/ -v --tb=short`

Expected: All tests PASS (no regressions)

**Step 5: Commit**

```bash
git add src/esper/simic/agent/ppo_agent.py
git commit -m "feat(ppo): wire entropy floor through PPOAgent

PPOAgent now passes ENTROPY_FLOOR_PER_HEAD to compute_losses() by default.
This enables per-head entropy floor penalties for all training runs.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Per-Head Entropy Collapse Detection

**Files:**
- Modify: `src/esper/simic/telemetry/anomaly_detector.py:213-245`
- Test: `tests/simic/telemetry/test_anomaly_detector.py`

**Step 1: Write the failing test**

Add to or create `tests/simic/telemetry/test_anomaly_detector.py`:

```python
"""Tests for per-head entropy collapse detection."""

import pytest

from esper.simic.telemetry.anomaly_detector import AnomalyDetector


class TestPerHeadEntropyCollapse:
    """Tests for per-head entropy collapse detection with hysteresis."""

    def test_no_warning_on_single_collapse(self) -> None:
        """Single collapse should not trigger warning (hysteresis)."""
        detector = AnomalyDetector()
        head_entropies = {"blueprint": 0.01}  # Below 0.05

        report = detector.check_per_head_entropy_collapse(head_entropies)

        # First collapse - no warning yet (need N consecutive)
        assert not report.has_any_anomalies()

    def test_warning_after_consecutive_collapses(self) -> None:
        """Warning should fire after N=3 consecutive collapses."""
        detector = AnomalyDetector()
        head_entropies = {"blueprint": 0.01}

        # First two - no warning
        detector.check_per_head_entropy_collapse(head_entropies)
        detector.check_per_head_entropy_collapse(head_entropies)

        # Third - warning fires
        report = detector.check_per_head_entropy_collapse(head_entropies)
        assert report.has_anomaly("entropy_collapse_blueprint")

    def test_no_detection_when_healthy(self) -> None:
        """Should not flag healthy heads."""
        detector = AnomalyDetector()

        head_entropies = {
            "op": 0.5,
            "blueprint": 0.3,
            "slot": 0.4,
        }

        report = detector.check_per_head_entropy_collapse(head_entropies)

        assert not report.has_any_anomalies()

    def test_recovery_requires_margin(self) -> None:
        """Recovery should require entropy above threshold * 1.5."""
        detector = AnomalyDetector()

        # Build up streak of 2 collapses
        detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        detector.check_per_head_entropy_collapse({"blueprint": 0.01})

        # Barely above threshold (0.06 > 0.05) - should NOT clear streak
        detector.check_per_head_entropy_collapse({"blueprint": 0.06})

        # Back to collapse - should continue streak and fire (now 3rd)
        report = detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        assert report.has_anomaly("entropy_collapse_blueprint")

        # Well above threshold (0.10 > 0.075) - should clear
        detector.check_per_head_entropy_collapse({"blueprint": 0.10})

        # New collapse - should start fresh (only 1 in streak, no warning)
        report = detector.check_per_head_entropy_collapse({"blueprint": 0.01})
        assert not report.has_any_anomalies()

    def test_uses_per_head_thresholds(self) -> None:
        """Different heads should have different collapse thresholds."""
        detector = AnomalyDetector()

        # op threshold is 0.08, blueprint is 0.05
        head_entropies = {
            "op": 0.07,        # Below 0.08 -> collapse streak
            "blueprint": 0.06, # Above 0.05 -> OK
        }

        # Need 3 consecutive to trigger
        for _ in range(3):
            report = detector.check_per_head_entropy_collapse(head_entropies)

        assert report.has_anomaly("entropy_collapse_op")
        assert not report.has_anomaly("entropy_collapse_blueprint")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/telemetry/test_anomaly_detector.py::TestPerHeadEntropyCollapse -v`

Expected: FAIL with "AttributeError: 'AnomalyDetector' object has no attribute 'check_per_head_entropy_collapse'"

**Step 3: Add hysteresis tracking to AnomalyDetector class**

In `src/esper/simic/telemetry/anomaly_detector.py`, add instance variables for hysteresis tracking:

```python
from dataclasses import field

@dataclass(slots=True)
class AnomalyDetector:
    # ... existing fields ...

    # Per-head collapse tracking (for hysteresis)
    _head_collapse_streak: dict[str, int] = field(default_factory=dict)
    collapse_streak_threshold: int = 3  # Consecutive updates before warning
    recovery_multiplier: float = 1.5    # Entropy must exceed threshold * 1.5 to clear
```

**Step 4: Implement check_per_head_entropy_collapse with hysteresis**

Add after `check_entropy_collapse()`:

```python
from esper.leyline import ENTROPY_COLLAPSE_PER_HEAD

def check_per_head_entropy_collapse(
    self,
    head_entropies: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> AnomalyReport:
    """Check for per-head entropy collapse with hysteresis.

    Each head has its own threshold based on activation frequency.
    Uses hysteresis to prevent warning spam:
    - Requires N consecutive collapses before warning
    - Requires entropy > threshold * 1.5 to clear the streak

    Args:
        head_entropies: Dict of head_name -> mean entropy (normalized 0-1)
        thresholds: Optional custom thresholds per head (defaults to ENTROPY_COLLAPSE_PER_HEAD)

    Returns:
        AnomalyReport with any detected per-head collapse issues
    """
    report = AnomalyReport()
    thresholds = thresholds or ENTROPY_COLLAPSE_PER_HEAD

    for head, entropy in head_entropies.items():
        threshold = thresholds.get(head, 0.1)  # Default fallback
        current_streak = self._head_collapse_streak.get(head, 0)

        if entropy < threshold:
            # Increment streak
            current_streak += 1
            self._head_collapse_streak[head] = current_streak

            # Only report after sustained collapse (N consecutive)
            if current_streak >= self.collapse_streak_threshold:
                report.add_anomaly(
                    f"entropy_collapse_{head}",
                    f"{head} entropy={entropy:.4f} < {threshold} "
                    f"({current_streak} consecutive updates)",
                )
        elif entropy > threshold * self.recovery_multiplier:
            # Clear streak only when entropy is well above threshold
            self._head_collapse_streak[head] = 0

    return report
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/simic/telemetry/test_anomaly_detector.py::TestPerHeadEntropyCollapse -v`

Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/telemetry/anomaly_detector.py tests/simic/telemetry/test_anomaly_detector.py
git commit -m "feat(telemetry): add per-head entropy collapse detection

Detects when individual action heads collapse while total policy entropy
remains healthy. Uses per-head thresholds from leyline that account for
sparse heads receiving fewer gradient signals.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Wire Per-Head Detection into Training Loop

**Files:**
- Modify: `src/esper/simic/training/vectorized_trainer.py` (or wherever anomaly detection is called)

**Step 1: Find where anomaly detection is called**

Search for `check_entropy_collapse` calls in the training code.

**Step 2: Add per-head check after existing entropy check**

After the existing `check_entropy_collapse()` call, add:

```python
# Per-head entropy collapse detection
if ppo_metrics.get("head_entropies"):
    # Convert per-epoch lists to mean per head
    mean_head_entropies = {
        head: sum(values) / len(values) if values else 0.0
        for head, values in ppo_metrics["head_entropies"].items()
    }
    per_head_report = self.anomaly_detector.check_per_head_entropy_collapse(
        mean_head_entropies
    )
    if per_head_report.has_any_anomalies():
        # Log but don't halt - this is a warning
        for anomaly in per_head_report.anomalies:
            logger.warning(f"Per-head entropy anomaly: {anomaly}")
```

**Step 3: Run training loop tests**

Run: `uv run pytest tests/simic/training/ -v --tb=short -k "not slow"`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/simic/training/vectorized_trainer.py
git commit -m "feat(training): wire per-head entropy collapse detection into loop

Logs warnings when individual action heads collapse, providing early
warning even when total policy entropy appears healthy.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Run Full Test Suite and Type Check

**Step 1: Run all simic tests**

Run: `uv run pytest tests/simic/ -v --tb=short`

Expected: All tests PASS

**Step 2: Run type check**

Run: `uv run mypy src/esper/simic/agent/ppo_update.py src/esper/simic/telemetry/anomaly_detector.py --ignore-missing-imports`

Expected: No errors

**Step 3: Final commit if any cleanup needed**

---

## Verification Checklist

After implementation, verify:

- [ ] `test_no_penalty_when_above_floor` passes
- [ ] `test_penalty_when_below_floor` passes
- [ ] `test_compute_losses_includes_entropy_floor_penalty` passes
- [ ] `test_detects_blueprint_collapse` passes
- [ ] All existing PPO tests still pass
- [ ] All existing training tests still pass
- [ ] No mypy errors
- [ ] Running a short training shows entropy floor in loss components

---

## Expected Behavior After Fix

With these changes:

1. **Blueprint entropy will be maintained**: Quadratic penalty pushes entropy back up when it drops below 0.4
2. **Early warning in telemetry**: `entropy_collapse_blueprint` anomaly will fire if entropy drops below 0.05
3. **Other heads protected**: tempo (0.4 floor) and other sparse heads also get protection
4. **Smooth intervention**: Quadratic penalty creates gradual pressure, not hard constraints

---

## Follow-Up (Optional Enhancements)

These are lower-priority enhancements if the primary fix is insufficient:

1. **Early training entropy boost**: 2x multiplier for sparse heads in first 25% of training
2. **PBRS diversity bonus**: Potential-based shaping for underused blueprints
3. **Adaptive entropy coefficient**: Increase entropy_coef when heads collapse
