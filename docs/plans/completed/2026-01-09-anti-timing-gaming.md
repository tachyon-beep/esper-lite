# Anti-Timing-Gaming Reward Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the reward function anti-pattern where seeds are incentivized to germinate immediately to maximize bounded_attribution by claiming credit for host drift.

**Architecture:** Two complementary fixes:
1. **D3-Timing:** Germination timing discount - seeds germinated before a warmup period receive discounted attribution
2. **D3-Attribution:** Harmonic mean formula - replace geometric mean with harmonic mean to better bound attribution when progress >> contribution

**Tech Stack:** Python, pytest, hypothesis (property tests)

---

## Background

The policy has learned to germinate ALL seeds in epochs 1-5 because:
- `bounded_attribution = sqrt(progress × seed_contribution)` where `progress = val_acc - acc_at_germination`
- Earlier germination at lower accuracy → more headroom → higher cumulative attribution
- A seed germinated at 22% can claim credit for 29pp of "improvement" even if the host would have improved anyway

**Evidence:** 100% of completed episodes have first germination in epochs 1-5.

---

## Task 1: Add Germination Timing Discount Config Parameters

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:51-220` (ContributionRewardConfig)
- Test: `tests/simic/rewards/shaped/test_shaped_attribution.py`

**Step 1: Write the failing test for new config parameters**

Add to `tests/simic/rewards/shaped/test_shaped_attribution.py`:

```python
def test_timing_discount_config_defaults() -> None:
    """D3-Timing: Config should have timing discount parameters with sensible defaults."""
    from esper.simic.rewards import ContributionRewardConfig

    config = ContributionRewardConfig()

    # Default warmup period: 10 epochs before full credit
    assert config.germination_warmup_epochs == 10
    # Default floor: epoch-1 germination gets 40% credit
    assert config.germination_discount_floor == 0.4
    # Default: timing discount enabled
    assert config.disable_timing_discount is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py::test_timing_discount_config_defaults -v`
Expected: FAIL with "AttributeError: 'ContributionRewardConfig' object has no attribute 'germination_warmup_epochs'"

**Step 3: Add config parameters to ContributionRewardConfig**

Add to `src/esper/simic/rewards/contribution.py` after line 213 (after `first_germinate_bonus`):

```python
    # === D3: Anti-Timing-Gaming (early germination discount) ===
    # Seeds germinated before warmup period receive discounted attribution.
    # This prevents "germinate early to claim host drift" gaming pattern.
    # Linear discount: epoch 1 = discount_floor, epoch warmup = 1.0
    germination_warmup_epochs: int = 10
    germination_discount_floor: float = 0.4
    disable_timing_discount: bool = False
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py::test_timing_discount_config_defaults -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/shaped/test_shaped_attribution.py
git commit -m "$(cat <<'EOF'
feat(rewards): add D3 timing discount config parameters

Add germination_warmup_epochs, germination_discount_floor, and
disable_timing_discount to ContributionRewardConfig. Seeds germinated
before the warmup period will receive discounted attribution.

This addresses the anti-pattern where policy learns to germinate
immediately to maximize bounded_attribution headroom.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement Timing Discount Helper Function

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py`
- Test: `tests/simic/rewards/shaped/test_shaped_attribution.py`

**Step 1: Write tests for the timing discount helper**

Add to `tests/simic/rewards/shaped/test_shaped_attribution.py`:

```python
def test_timing_discount_epoch_1_gets_floor() -> None:
    """D3-Timing: Epoch 1 germination gets discount_floor credit."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    discount = _compute_timing_discount(
        germination_epoch=1,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(0.4)


def test_timing_discount_at_warmup_gets_full_credit() -> None:
    """D3-Timing: Germination at warmup epoch gets full credit."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    discount = _compute_timing_discount(
        germination_epoch=10,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(1.0)


def test_timing_discount_after_warmup_gets_full_credit() -> None:
    """D3-Timing: Germination after warmup gets full credit."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    discount = _compute_timing_discount(
        germination_epoch=50,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(1.0)


def test_timing_discount_mid_warmup_interpolates() -> None:
    """D3-Timing: Mid-warmup germination gets linearly interpolated discount."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    # Epoch 5 out of 10 warmup, floor=0.4
    # discount = 0.4 + (1 - 0.4) * (5 / 10) = 0.4 + 0.3 = 0.7
    discount = _compute_timing_discount(
        germination_epoch=5,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(0.7)


def test_timing_discount_epoch_0_gets_floor() -> None:
    """D3-Timing: Edge case - epoch 0 gets discount_floor."""
    from esper.simic.rewards.contribution import _compute_timing_discount

    discount = _compute_timing_discount(
        germination_epoch=0,
        warmup_epochs=10,
        discount_floor=0.4,
    )
    assert discount == pytest.approx(0.4)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py -k "timing_discount" -v`
Expected: FAIL with "cannot import name '_compute_timing_discount'"

**Step 3: Implement the helper function**

Add to `src/esper/simic/rewards/contribution.py` after `_contribution_prune_shaping` (around line 848):

```python
def _compute_timing_discount(
    germination_epoch: int,
    warmup_epochs: int,
    discount_floor: float,
) -> float:
    """Compute timing discount for early germination.

    Seeds germinated before warmup_epochs receive discounted attribution.
    Linear interpolation from discount_floor (epoch 0) to 1.0 (epoch >= warmup).

    Args:
        germination_epoch: Epoch when seed was germinated
        warmup_epochs: Number of epochs before full credit
        discount_floor: Minimum discount (applied at epoch 0)

    Returns:
        Discount factor in [discount_floor, 1.0]
    """
    if germination_epoch >= warmup_epochs:
        return 1.0

    # Linear interpolation: epoch 0 = floor, epoch warmup = 1.0
    progress = germination_epoch / warmup_epochs
    return discount_floor + (1.0 - discount_floor) * progress
```

Also add to `__all__` list at the bottom:

```python
    "_compute_timing_discount",
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py -k "timing_discount" -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/shaped/test_shaped_attribution.py
git commit -m "$(cat <<'EOF'
feat(rewards): implement _compute_timing_discount helper

Linear interpolation from discount_floor at epoch 0 to 1.0 at
warmup_epochs. Seeds germinated after warmup get full credit.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire Timing Discount into bounded_attribution

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:390-420`
- Test: `tests/simic/rewards/shaped/test_shaped_attribution.py`

**Step 1: Write integration test for timing discount on bounded_attribution**

Add to `tests/simic/rewards/shaped/test_shaped_attribution.py`:

```python
def test_shaped_timing_discount_reduces_early_germination_attribution() -> None:
    """D3-Timing: Early germination receives discounted bounded_attribution."""
    from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig
    from esper.leyline import LifecycleOp, SeedStage

    # Config with timing discount enabled
    config = ContributionRewardConfig(
        contribution_weight=1.0,
        germination_warmup_epochs=10,
        germination_discount_floor=0.4,
        disable_timing_discount=False,
        # Disable other components for isolation
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        first_germinate_bonus=0.0,
    )

    # Seed germinated at epoch 2, now at epoch 20
    # germination_epoch = 20 - 18 = 2
    seed = SeedInfo(
        stage=SeedStage.TRAINING.value,
        total_improvement=5.0,
        improvement_since_stage_start=0.0,
        epochs_in_stage=1,
        seed_params=0,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=0,
        seed_age_epochs=18,  # epoch 20 - age 18 = germinated at epoch 2
        interaction_sum=0.0,
        boost_received=0.0,
    )

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=3.0,
        val_acc=75.0,
        seed_info=seed,
        epoch=20,
        max_epochs=150,
        acc_at_germination=65.0,  # progress = 10.0
        config=config,
        return_components=True,
    )

    # Without discount: attributed = 3.0 (contribution < progress)
    # With discount (epoch 2, warmup 10, floor 0.4): 0.4 + 0.6 * (2/10) = 0.52
    # Expected attribution = 3.0 * 0.52 = 1.56
    expected_discount = 0.4 + 0.6 * (2 / 10)
    expected_attribution = 3.0 * expected_discount

    assert components.bounded_attribution == pytest.approx(expected_attribution, rel=0.01)


def test_shaped_timing_discount_disabled_gives_full_credit() -> None:
    """D3-Timing: When disabled, early germination gets full credit."""
    from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig(
        contribution_weight=1.0,
        germination_warmup_epochs=10,
        germination_discount_floor=0.4,
        disable_timing_discount=True,  # DISABLED
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        first_germinate_bonus=0.0,
    )

    seed = SeedInfo(
        stage=SeedStage.TRAINING.value,
        total_improvement=5.0,
        improvement_since_stage_start=0.0,
        epochs_in_stage=1,
        seed_params=0,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=0,
        seed_age_epochs=18,
        interaction_sum=0.0,
        boost_received=0.0,
    )

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=3.0,
        val_acc=75.0,
        seed_info=seed,
        epoch=20,
        max_epochs=150,
        acc_at_germination=65.0,
        config=config,
        return_components=True,
    )

    # No discount: attributed = 3.0
    assert components.bounded_attribution == pytest.approx(3.0)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py -k "timing_discount_reduces or timing_discount_disabled" -v`
Expected: FAIL (timing discount not yet applied)

**Step 3: Wire timing discount into compute_contribution_reward**

In `src/esper/simic/rewards/contribution.py`, modify the bounded_attribution calculation (around line 407).

After line 407 (`bounded_attribution = (config.contribution_weight * attributed) + ratio_penalty`), add:

```python
                # D3: Apply timing discount for early germination
                if not config.disable_timing_discount and seed_info is not None:
                    germination_epoch = epoch - seed_info.seed_age_epochs
                    timing_discount = _compute_timing_discount(
                        germination_epoch=germination_epoch,
                        warmup_epochs=config.germination_warmup_epochs,
                        discount_floor=config.germination_discount_floor,
                    )
                    bounded_attribution *= timing_discount
```

Also add `timing_discount` to components. After line 431, add:

```python
        components.timing_discount = timing_discount if not config.disable_timing_discount else 1.0
```

And add the field to `RewardComponentsTelemetry` in `src/esper/simic/rewards/reward_telemetry.py`:

```python
    timing_discount: float | None = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py -k "timing_discount" -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py src/esper/simic/rewards/reward_telemetry.py tests/simic/rewards/shaped/test_shaped_attribution.py
git commit -m "$(cat <<'EOF'
feat(rewards): wire D3 timing discount into bounded_attribution

Apply germination timing discount after computing base attribution.
Seeds germinated before warmup_epochs receive reduced credit.

Adds timing_discount field to RewardComponentsTelemetry for observability.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add Attribution Formula Config Parameter

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py`
- Test: `tests/simic/rewards/shaped/test_shaped_attribution.py`

**Step 1: Write test for attribution formula config**

Add to `tests/simic/rewards/shaped/test_shaped_attribution.py`:

```python
def test_attribution_formula_config_default() -> None:
    """D3-Attribution: Config should support attribution formula selection."""
    from esper.simic.rewards import ContributionRewardConfig

    config = ContributionRewardConfig()

    # Default: geometric mean (current behavior)
    assert config.attribution_formula == "geometric"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py::test_attribution_formula_config_default -v`
Expected: FAIL with "AttributeError"

**Step 3: Add config parameter**

Add to `src/esper/simic/rewards/contribution.py` after the timing discount parameters:

```python
    # === D3: Attribution formula variant ===
    # Controls how progress and seed_contribution combine into attributed value.
    # - "geometric": sqrt(progress * contribution) - current default, rewards host drift
    # - "harmonic": 2*p*c/(p+c) - dominated by smaller value, conservative
    # - "minimum": min(progress, contribution) - very conservative
    attribution_formula: str = "geometric"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py::test_attribution_formula_config_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/shaped/test_shaped_attribution.py
git commit -m "$(cat <<'EOF'
feat(rewards): add attribution_formula config parameter

Supports "geometric" (current), "harmonic", and "minimum" formulas
for combining progress and seed_contribution.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Implement Harmonic Mean Attribution Helper

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py`
- Test: `tests/simic/rewards/shaped/test_shaped_attribution.py`

**Step 1: Write tests for attribution formula helpers**

Add to `tests/simic/rewards/shaped/test_shaped_attribution.py`:

```python
def test_compute_attributed_geometric_mean() -> None:
    """D3-Attribution: Geometric mean formula (current behavior)."""
    from esper.simic.rewards.contribution import _compute_attributed_value

    # sqrt(4 * 9) = 6.0
    result = _compute_attributed_value(
        progress=4.0,
        seed_contribution=9.0,
        formula="geometric",
    )
    assert result == pytest.approx(6.0)


def test_compute_attributed_harmonic_mean() -> None:
    """D3-Attribution: Harmonic mean dominated by smaller value."""
    from esper.simic.rewards.contribution import _compute_attributed_value

    # 2 * 4 * 9 / (4 + 9) = 72 / 13 ≈ 5.54
    result = _compute_attributed_value(
        progress=4.0,
        seed_contribution=9.0,
        formula="harmonic",
    )
    assert result == pytest.approx(72 / 13)


def test_compute_attributed_harmonic_handles_zero() -> None:
    """D3-Attribution: Harmonic mean returns 0 when either input is 0."""
    from esper.simic.rewards.contribution import _compute_attributed_value

    result = _compute_attributed_value(
        progress=0.0,
        seed_contribution=9.0,
        formula="harmonic",
    )
    assert result == pytest.approx(0.0)


def test_compute_attributed_minimum() -> None:
    """D3-Attribution: Minimum formula is very conservative."""
    from esper.simic.rewards.contribution import _compute_attributed_value

    result = _compute_attributed_value(
        progress=4.0,
        seed_contribution=9.0,
        formula="minimum",
    )
    assert result == pytest.approx(4.0)


def test_compute_attributed_harmonic_vs_geometric_with_large_progress() -> None:
    """D3-Attribution: Harmonic much lower than geometric when progress >> contribution.

    This is the key anti-gaming property: when a seed claims credit for
    massive host drift (progress=29) with small actual contribution (0.5),
    harmonic mean gives much less credit than geometric mean.
    """
    from esper.simic.rewards.contribution import _compute_attributed_value

    progress = 29.0  # Host improved 29pp since germination
    contribution = 0.5  # Seed only contributed 0.5pp

    geometric = _compute_attributed_value(progress, contribution, "geometric")
    harmonic = _compute_attributed_value(progress, contribution, "harmonic")

    # Geometric: sqrt(29 * 0.5) ≈ 3.81
    assert geometric == pytest.approx(3.807, rel=0.01)

    # Harmonic: 2 * 29 * 0.5 / (29 + 0.5) = 29 / 29.5 ≈ 0.98
    assert harmonic == pytest.approx(0.983, rel=0.01)

    # Harmonic is ~4x lower for this gaming scenario
    assert harmonic < geometric * 0.3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py -k "compute_attributed" -v`
Expected: FAIL with "cannot import name '_compute_attributed_value'"

**Step 3: Implement the helper function**

Add to `src/esper/simic/rewards/contribution.py` after `_compute_timing_discount`:

```python
def _compute_attributed_value(
    progress: float,
    seed_contribution: float,
    formula: str,
) -> float:
    """Compute attributed value using the specified formula.

    Args:
        progress: Accuracy improvement since germination (val_acc - acc_at_germination)
        seed_contribution: Counterfactual contribution of the seed
        formula: One of "geometric", "harmonic", "minimum"

    Returns:
        Attributed value combining progress and contribution

    Formulas:
        - geometric: sqrt(progress * contribution) - rewards host drift
        - harmonic: 2*p*c/(p+c) - dominated by smaller value, anti-gaming
        - minimum: min(progress, contribution) - very conservative
    """
    if progress <= 0 or seed_contribution <= 0:
        return 0.0

    if formula == "geometric":
        return math.sqrt(progress * seed_contribution)

    elif formula == "harmonic":
        # Harmonic mean: 2ab/(a+b), dominated by smaller value
        return 2 * progress * seed_contribution / (progress + seed_contribution)

    elif formula == "minimum":
        return min(progress, seed_contribution)

    else:
        raise ValueError(f"Unknown attribution formula: {formula}")
```

Add to `__all__`:

```python
    "_compute_attributed_value",
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py -k "compute_attributed" -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/shaped/test_shaped_attribution.py
git commit -m "$(cat <<'EOF'
feat(rewards): implement _compute_attributed_value with formula variants

Supports geometric (current), harmonic, and minimum formulas.
Harmonic mean is dominated by smaller value, reducing credit when
progress >> contribution (anti-timing-gaming property).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire Attribution Formula into bounded_attribution

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:395-407`
- Test: `tests/simic/rewards/shaped/test_shaped_attribution.py`

**Step 1: Write integration test for harmonic formula**

Add to `tests/simic/rewards/shaped/test_shaped_attribution.py`:

```python
def test_shaped_harmonic_formula_reduces_drift_attribution() -> None:
    """D3-Attribution: Harmonic formula gives less credit for host drift."""
    from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig
    from esper.leyline import LifecycleOp, SeedStage

    # Config with harmonic formula
    config_harmonic = ContributionRewardConfig(
        contribution_weight=1.0,
        attribution_formula="harmonic",
        disable_timing_discount=True,  # Isolate formula test
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        first_germinate_bonus=0.0,
    )

    # Config with geometric formula (default)
    config_geometric = ContributionRewardConfig(
        contribution_weight=1.0,
        attribution_formula="geometric",
        disable_timing_discount=True,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        first_germinate_bonus=0.0,
    )

    seed = SeedInfo(
        stage=SeedStage.TRAINING.value,
        total_improvement=5.0,
        improvement_since_stage_start=0.0,
        epochs_in_stage=1,
        seed_params=0,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=0,
        seed_age_epochs=50,
        interaction_sum=0.0,
        boost_received=0.0,
    )

    # Gaming scenario: large progress (host drift), small contribution
    _, components_geo = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,  # Small actual contribution
        val_acc=51.0,
        seed_info=seed,
        epoch=55,
        max_epochs=150,
        acc_at_germination=22.0,  # progress = 29.0 (large host drift)
        config=config_geometric,
        return_components=True,
    )

    _, components_harm = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.5,
        val_acc=51.0,
        seed_info=seed,
        epoch=55,
        max_epochs=150,
        acc_at_germination=22.0,
        config=config_harmonic,
        return_components=True,
    )

    # Geometric: sqrt(29 * 0.5) ≈ 3.81
    # Harmonic: 2 * 29 * 0.5 / 29.5 ≈ 0.98
    assert components_geo.bounded_attribution == pytest.approx(3.807, rel=0.01)
    assert components_harm.bounded_attribution == pytest.approx(0.983, rel=0.01)

    # Harmonic is ~4x less for this gaming scenario
    assert components_harm.bounded_attribution < components_geo.bounded_attribution * 0.3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py::test_shaped_harmonic_formula_reduces_drift_attribution -v`
Expected: FAIL (formula not yet wired)

**Step 3: Wire formula into compute_contribution_reward**

In `src/esper/simic/rewards/contribution.py`, replace the attribution calculation (lines ~395-407).

Replace:

```python
                if progress is not None:
                    if progress <= 0:
                        attributed = 0.0
                    elif seed_contribution >= progress:
                        attributed = math.sqrt(progress * seed_contribution)
                    else:
                        attributed = seed_contribution
                else:
                    attributed = seed_contribution * 0.5
```

With:

```python
                if progress is not None:
                    if progress <= 0:
                        attributed = 0.0
                    elif seed_contribution >= progress:
                        # Use configurable formula when contribution exceeds progress
                        attributed = _compute_attributed_value(
                            progress=progress,
                            seed_contribution=seed_contribution,
                            formula=config.attribution_formula,
                        )
                    else:
                        # contribution < progress: cap at contribution (unchanged)
                        attributed = seed_contribution
                else:
                    attributed = seed_contribution * 0.5
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/rewards/shaped/test_shaped_attribution.py::test_shaped_harmonic_formula_reduces_drift_attribution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/shaped/test_shaped_attribution.py
git commit -m "$(cat <<'EOF'
feat(rewards): wire attribution_formula into bounded_attribution calculation

Use configurable formula (geometric/harmonic/minimum) when computing
attributed value. Harmonic mean gives ~4x less credit when seeds
claim large host drift with small actual contribution.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add Property Tests for Anti-Timing-Gaming

**Files:**
- Create: `tests/simic/properties/test_timing_gaming.py`

**Step 1: Write property tests for timing discount**

Create `tests/simic/properties/test_timing_gaming.py`:

```python
"""Property-based tests for anti-timing-gaming reward behavior.

These tests verify that the D3 timing discount and harmonic attribution
formula correctly discourage early germination gaming patterns.

Tier 3: Anti-Gaming Properties
- Early germination receives discounted attribution
- Harmonic mean bounds attribution when progress >> contribution
- Combined fixes reduce incentive for timing gaming
"""

import math

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)
from esper.simic.rewards.contribution import (
    _compute_timing_discount,
    _compute_attributed_value,
)

pytestmark = pytest.mark.property


@st.composite
def early_germination_inputs(draw: st.DrawFn):
    """Generate inputs representing early germination scenarios."""
    warmup = draw(st.integers(min_value=5, max_value=20))
    germination_epoch = draw(st.integers(min_value=0, max_value=warmup - 1))
    current_epoch = draw(st.integers(min_value=germination_epoch + 10, max_value=150))
    seed_age = current_epoch - germination_epoch

    acc_at_germination = draw(st.floats(min_value=10.0, max_value=40.0))
    val_acc = draw(st.floats(min_value=acc_at_germination + 5.0, max_value=70.0))
    seed_contribution = draw(st.floats(min_value=0.1, max_value=5.0))

    return {
        "warmup_epochs": warmup,
        "germination_epoch": germination_epoch,
        "current_epoch": current_epoch,
        "seed_age": seed_age,
        "acc_at_germination": acc_at_germination,
        "val_acc": val_acc,
        "seed_contribution": seed_contribution,
    }


@given(inputs=early_germination_inputs())
@settings(max_examples=200)
def test_timing_discount_always_reduces_early_germination_reward(inputs):
    """D3-Timing: Early germination ALWAYS receives less attribution than late."""
    warmup = inputs["warmup_epochs"]
    germ_epoch = inputs["germination_epoch"]
    floor = 0.4

    discount = _compute_timing_discount(germ_epoch, warmup, floor)

    # Early germination should get discount < 1.0
    assert discount < 1.0, f"Epoch {germ_epoch} before warmup {warmup} got discount {discount}"
    assert discount >= floor, f"Discount {discount} below floor {floor}"


@given(
    progress=st.floats(min_value=10.0, max_value=50.0),
    contribution=st.floats(min_value=0.1, max_value=2.0),
)
@settings(max_examples=200)
def test_harmonic_always_less_than_geometric_when_progress_dominates(
    progress: float, contribution: float
):
    """D3-Attribution: Harmonic <= Geometric when progress > contribution."""
    assume(progress > contribution * 2)  # Progress dominates

    geometric = _compute_attributed_value(progress, contribution, "geometric")
    harmonic = _compute_attributed_value(progress, contribution, "harmonic")

    assert harmonic <= geometric, (
        f"Harmonic {harmonic} > Geometric {geometric} "
        f"for progress={progress}, contribution={contribution}"
    )


@given(
    progress=st.floats(min_value=0.1, max_value=50.0),
    contribution=st.floats(min_value=0.1, max_value=50.0),
)
@settings(max_examples=200)
def test_harmonic_bounded_by_minimum(progress: float, contribution: float):
    """D3-Attribution: Harmonic mean is always <= min(progress, contribution) * 2."""
    harmonic = _compute_attributed_value(progress, contribution, "harmonic")
    minimum = _compute_attributed_value(progress, contribution, "minimum")

    # Harmonic mean <= 2 * min(a, b) for positive a, b
    # Actually: harmonic <= min(a, b) when a != b
    assert harmonic <= minimum * 2.1, (
        f"Harmonic {harmonic} > 2*minimum {minimum * 2} "
        f"for progress={progress}, contribution={contribution}"
    )


@given(inputs=early_germination_inputs())
@settings(max_examples=100)
def test_combined_fixes_reduce_gaming_incentive(inputs):
    """D3 Combined: Timing discount + harmonic formula together reduce gaming."""
    config = ContributionRewardConfig(
        contribution_weight=1.0,
        germination_warmup_epochs=inputs["warmup_epochs"],
        germination_discount_floor=0.4,
        attribution_formula="harmonic",
        disable_timing_discount=False,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        first_germinate_bonus=0.0,
    )

    config_baseline = ContributionRewardConfig(
        contribution_weight=1.0,
        attribution_formula="geometric",
        disable_timing_discount=True,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        first_germinate_bonus=0.0,
    )

    seed = SeedInfo(
        stage=SeedStage.TRAINING.value,
        total_improvement=5.0,
        improvement_since_stage_start=0.0,
        epochs_in_stage=1,
        seed_params=0,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=0,
        seed_age_epochs=inputs["seed_age"],
        interaction_sum=0.0,
        boost_received=0.0,
    )

    # Progress >> contribution scenario (gaming)
    progress = inputs["val_acc"] - inputs["acc_at_germination"]
    assume(progress > inputs["seed_contribution"] * 5)

    _, components_fixed = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=inputs["seed_contribution"],
        val_acc=inputs["val_acc"],
        seed_info=seed,
        epoch=inputs["current_epoch"],
        max_epochs=150,
        acc_at_germination=inputs["acc_at_germination"],
        config=config,
        return_components=True,
    )

    _, components_baseline = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=inputs["seed_contribution"],
        val_acc=inputs["val_acc"],
        seed_info=seed,
        epoch=inputs["current_epoch"],
        max_epochs=150,
        acc_at_germination=inputs["acc_at_germination"],
        config=config_baseline,
        return_components=True,
    )

    # Combined fixes should significantly reduce attribution for gaming scenarios
    assert components_fixed.bounded_attribution <= components_baseline.bounded_attribution, (
        f"Fixed attribution {components_fixed.bounded_attribution} > "
        f"baseline {components_baseline.bounded_attribution}"
    )
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/simic/properties/test_timing_gaming.py -v`
Expected: PASS (4 property tests)

**Step 3: Commit**

```bash
git add tests/simic/properties/test_timing_gaming.py
git commit -m "$(cat <<'EOF'
test(rewards): add property tests for D3 anti-timing-gaming

Hypothesis-based tests verify:
- Timing discount always reduces early germination reward
- Harmonic mean <= geometric when progress dominates
- Combined fixes reduce gaming incentive

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update Test Harness and Run Full Test Suite

**Files:**
- Modify: `tests/simic/rewards/shaped/harness.py`

**Step 1: Update harness with new config options**

Add new parameters to `shaped_config()` in `tests/simic/rewards/shaped/harness.py`:

```python
def shaped_config(
    *,
    contribution_weight: float = 1.0,
    proxy_confidence_factor: float = 0.3,
    disable_pbrs: bool = True,
    disable_terminal_reward: bool = True,
    disable_anti_gaming: bool = True,
    disable_timing_discount: bool = True,  # NEW: default disabled for isolated tests
    rent_weight: float = 0.0,
    rent_host_params_floor: int = 1,
    alpha_shock_coef: float = 0.0,
    pbrs_weight: float = 0.0,
    terminal_acc_weight: float = 0.0,
    fossilize_terminal_scale: float = 0.0,
    prune_good_seed_penalty: float = 0.0,
    prune_hurting_bonus: float = 0.0,
    prune_acceptable_bonus: float = 0.0,
    prune_cost: float = 0.0,
    fossilize_cost: float = 0.0,
    germinate_cost: float = 0.0,
    set_alpha_target_cost: float = 0.0,
    attribution_formula: str = "geometric",  # NEW: default geometric for backwards compat
    germination_warmup_epochs: int = 10,  # NEW
    germination_discount_floor: float = 0.4,  # NEW
) -> ContributionRewardConfig:
```

And add to the return statement:

```python
        disable_timing_discount=disable_timing_discount,
        attribution_formula=attribution_formula,
        germination_warmup_epochs=germination_warmup_epochs,
        germination_discount_floor=germination_discount_floor,
```

**Step 2: Run full test suite**

Run: `uv run pytest tests/simic/rewards/ tests/simic/properties/ -v --tb=short`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/simic/rewards/shaped/harness.py
git commit -m "$(cat <<'EOF'
test(rewards): update shaped harness with D3 config options

Add disable_timing_discount, attribution_formula, germination_warmup_epochs,
and germination_discount_floor to shaped_config harness.

Defaults maintain backwards compatibility (timing discount disabled,
geometric formula) for existing isolated tests.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Final Integration Test and Documentation

**Files:**
- Create: `tests/simic/rewards/test_anti_timing_gaming.py`

**Step 1: Write end-to-end integration test**

Create `tests/simic/rewards/test_anti_timing_gaming.py`:

```python
"""Integration tests for D3 anti-timing-gaming reward fixes.

These tests verify the complete behavior of timing discount + harmonic
attribution working together to discourage early germination gaming.
"""

import pytest

from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)


class TestAntiTimingGamingIntegration:
    """End-to-end tests for D3 anti-timing-gaming fixes."""

    def test_early_germination_scenario_with_all_fixes(self) -> None:
        """Simulate the exact scenario that caused the anti-pattern.

        Before D3: Seed germinated at epoch 2 (~22% acc) riding host to 51%
        accumulated massive bounded_attribution (~1680 over episode).

        After D3: Same scenario should yield significantly less reward.
        """
        # Configuration with all D3 fixes enabled
        config_d3 = ContributionRewardConfig(
            contribution_weight=1.0,
            germination_warmup_epochs=10,
            germination_discount_floor=0.4,
            attribution_formula="harmonic",
            disable_timing_discount=False,
            # Keep other defaults for realistic scenario
        )

        # Baseline configuration (pre-D3 behavior)
        config_baseline = ContributionRewardConfig(
            contribution_weight=1.0,
            attribution_formula="geometric",
            disable_timing_discount=True,
        )

        # Simulate seed germinated at epoch 2, now at epoch 100
        seed = SeedInfo(
            stage=SeedStage.TRAINING.value,
            total_improvement=25.0,  # Host improved significantly
            improvement_since_stage_start=0.5,
            epochs_in_stage=1,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            previous_epochs_in_stage=0,
            seed_age_epochs=98,  # epoch 100 - 98 = germinated at epoch 2
            interaction_sum=0.0,
            boost_received=0.0,
        )

        # Gaming scenario: seed_contribution=0.5, progress=29
        _, comp_d3 = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.5,  # Small actual contribution
            val_acc=51.0,
            seed_info=seed,
            epoch=100,
            max_epochs=150,
            acc_at_germination=22.0,  # progress = 29.0
            config=config_d3,
            return_components=True,
        )

        _, comp_baseline = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=0.5,
            val_acc=51.0,
            seed_info=seed,
            epoch=100,
            max_epochs=150,
            acc_at_germination=22.0,
            config=config_baseline,
            return_components=True,
        )

        # D3 should give significantly less credit
        # Baseline: sqrt(29 * 0.5) ≈ 3.81
        # D3 Harmonic: 2 * 29 * 0.5 / 29.5 ≈ 0.98
        # D3 Timing: epoch 2, warmup 10, floor 0.4 → discount = 0.52
        # D3 Combined: 0.98 * 0.52 ≈ 0.51

        assert comp_baseline.bounded_attribution == pytest.approx(3.81, rel=0.1)
        assert comp_d3.bounded_attribution == pytest.approx(0.51, rel=0.1)

        # D3 reduces attribution by ~7x for this gaming scenario
        reduction_factor = comp_baseline.bounded_attribution / comp_d3.bounded_attribution
        assert reduction_factor > 5, f"Expected >5x reduction, got {reduction_factor}x"

    def test_legitimate_late_germination_not_penalized(self) -> None:
        """Seeds germinated after warmup with real contribution get full credit."""
        config_d3 = ContributionRewardConfig(
            contribution_weight=1.0,
            germination_warmup_epochs=10,
            germination_discount_floor=0.4,
            attribution_formula="harmonic",
            disable_timing_discount=False,
        )

        # Seed germinated at epoch 50, now at epoch 100
        seed = SeedInfo(
            stage=SeedStage.TRAINING.value,
            total_improvement=8.0,
            improvement_since_stage_start=0.5,
            epochs_in_stage=1,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            previous_epochs_in_stage=0,
            seed_age_epochs=50,  # germinated at epoch 50 (after warmup)
            interaction_sum=0.0,
            boost_received=0.0,
        )

        # Legitimate scenario: contribution roughly matches progress
        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,  # Significant contribution
            val_acc=51.0,
            seed_info=seed,
            epoch=100,
            max_epochs=150,
            acc_at_germination=43.0,  # progress = 8.0 (similar to contribution)
            config=config_d3,
            return_components=True,
        )

        # When contribution ≈ progress, harmonic ≈ geometric
        # Harmonic: 2 * 8 * 5 / 13 ≈ 6.15
        # Timing discount: epoch 50 >= warmup 10 → discount = 1.0
        # Total: 6.15 * 1.0 = 6.15

        assert components.bounded_attribution == pytest.approx(6.15, rel=0.1)
        assert components.timing_discount == pytest.approx(1.0)
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/simic/rewards/test_anti_timing_gaming.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/simic/rewards/test_anti_timing_gaming.py
git commit -m "$(cat <<'EOF'
test(rewards): add D3 anti-timing-gaming integration tests

End-to-end tests verify:
- Early germination gaming scenario gets ~7x less reward with D3 fixes
- Legitimate late germination with real contribution not penalized

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan implements two complementary fixes for the early germination gaming anti-pattern:

1. **D3-Timing (Tasks 1-3):** Germination timing discount
   - Seeds germinated before `warmup_epochs` receive `[discount_floor, 1.0)` credit
   - Default: epoch 1 = 40%, epoch 10+ = 100%

2. **D3-Attribution (Tasks 4-6):** Harmonic mean formula
   - Replaces `sqrt(progress × contribution)` with `2×p×c/(p+c)`
   - Reduces credit when progress >> contribution (host drift scenario)

3. **Property Tests (Task 7):** Hypothesis-based verification
4. **Integration Tests (Task 9):** End-to-end gaming scenario verification

**Expected Impact:**
- Early germination (epoch 1-5) attribution reduced by ~7x for gaming scenarios
- Legitimate late germination with real contribution unchanged
- Policy should learn to wait and observe before germinating
