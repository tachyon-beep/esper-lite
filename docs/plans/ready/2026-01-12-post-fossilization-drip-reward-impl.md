# Post-Fossilization Drip Reward Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `BASIC_PLUS` reward mode with 30% immediate + 70% drip to create post-fossilization accountability, as a parallel mode to unchanged `BASIC`.

**Architecture:** Add `BASIC_PLUS` to `RewardMode` enum. Add `FossilizedSeedDripState` dataclass to track drip state per fossilized seed. Modify `compute_basic_reward` to accept drip states and return new drip state on FOSSILIZE. Drip is computed each epoch as `drip_scale * contribution` with asymmetric clipping. Per-epoch counterfactual is the correct signal for drip (not terminal Shapley).

**Tech Stack:** Python dataclasses, existing reward infrastructure, pytest with hypothesis for property tests.

**Design Decision: Parallel Mode (BASIC_PLUS)**
- `BASIC` remains unchanged (drip_fraction=0.0, 100% immediate)
- `BASIC_PLUS` enables drip (drip_fraction=0.7, 30% immediate + 70% drip)
- Enables clean A/B testing and instant rollback
- Same `compute_basic_reward` function, different config defaults

---

## Plan Metadata

```yaml
id: post-fossilization-drip-reward-impl
title: Post-Fossilization Drip Reward Implementation
type: ready
created: 2026-01-12
updated: 2026-01-12
owner: Claude

reviewed_by:
  - reviewer: drl-expert
    date: 2026-01-12
    verdict: approved_with_modifications
    notes: |
      Core drip design approved. 70/30 split provides adequate credit assignment.
      Epoch normalization correctly prevents early-fossilization gaming.

      REQUIRED CHANGES (incorporated below):
      1. Remove diminishing returns - replace with contribution threshold
      2. Add asymmetric negative clipping (-0.05 vs +0.1)
      3. Use per-epoch counterfactual only (no Shapley reconciliation)

      Effective expected value is ~85% of nominal due to gamma discounting.
      This is desirable - creates slight pessimism toward marginal fossilizations.
```

---

## Design Summary

### Core Mechanism

| Component | Value | Rationale |
|-----------|-------|-----------|
| Immediate fraction | 30% | Sufficient for PPO credit assignment with PBRS support |
| Drip fraction | 70% | Strong accountability signal |
| Drip signal | Per-epoch counterfactual | Markovian, correct causal question |
| Positive clip | +0.1 | Variance control |
| Negative clip | -0.05 | Asymmetric to prevent death spirals |
| Contribution threshold | 0.0 | Only positive contribution generates drip |

### Key Formula

```python
remaining_epochs = max(max_epochs - fossilize_epoch, 5)  # Floor at 5
drip_scale = drip_total / remaining_epochs
raw_drip = drip_scale * current_contribution

# Asymmetric clipping
if raw_drip >= 0:
    clipped_drip = min(raw_drip, max_drip_per_epoch)  # +0.1
else:
    clipped_drip = max(raw_drip, -0.5 * max_drip_per_epoch)  # -0.05
```

### What Was Removed from Original Plan

1. **Diminishing returns (D5)** - Removed per DRL expert review. Replaced with contribution threshold.
2. **Shapley reconciliation** - Not needed. Per-epoch counterfactual is the correct signal.

---

## Task 0: Add BASIC_PLUS to RewardMode Enum

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:32-48`
- Test: `tests/simic/test_config.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_config.py

def test_basic_plus_reward_mode_exists() -> None:
    """BASIC_PLUS reward mode is available."""
    from esper.simic.rewards.contribution import RewardMode

    assert hasattr(RewardMode, "BASIC_PLUS")
    assert RewardMode.BASIC_PLUS.value == "basic_plus"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_config.py::test_basic_plus_reward_mode_exists -v`
Expected: FAIL with "AttributeError: BASIC_PLUS"

**Step 3: Write minimal implementation**

Update `RewardMode` enum in `contribution.py`:

```python
class RewardMode(Enum):
    """Reward function variant for experimentation.

    SHAPED: Current dense shaping with PBRS, attribution, warnings (default)
    ESCROW: Dense, reversible attribution (anti-peak / anti-thrash)
    BASIC: Accuracy improvement minus parameter rent (minimal, no lifecycle shaping)
    BASIC_PLUS: BASIC + post-fossilization drip reward (accountability for fossilized seeds)
    SPARSE: Terminal-only ground truth (accuracy - param_cost)
    MINIMAL: Sparse + early-prune penalty only
    SIMPLIFIED: DRL Expert recommended - PBRS + intervention cost + terminal only
    """

    SHAPED = "shaped"
    ESCROW = "escrow"
    BASIC = "basic"
    BASIC_PLUS = "basic_plus"  # NEW: BASIC with drip accountability
    SPARSE = "sparse"
    MINIMAL = "minimal"
    SIMPLIFIED = "simplified"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_config.py::test_basic_plus_reward_mode_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/test_config.py
git commit -m "feat(simic): add BASIC_PLUS reward mode enum

BASIC_PLUS = BASIC + post-fossilization drip accountability.
Parallel mode to enable A/B testing against unchanged BASIC.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 1: Add Drip Configuration Fields to ContributionRewardConfig

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:52-250`
- Test: `tests/simic/test_config.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_config.py

def test_drip_config_fields_exist() -> None:
    """ContributionRewardConfig has drip-related fields with correct defaults."""
    from esper.simic.rewards.contribution import ContributionRewardConfig

    config = ContributionRewardConfig()

    # Core drip fields - default is 0.0 (disabled for BASIC mode)
    assert hasattr(config, "drip_fraction")
    assert config.drip_fraction == 0.0  # BASIC mode default: drip disabled

    assert hasattr(config, "max_drip_per_epoch")
    assert config.max_drip_per_epoch == 0.1

    assert hasattr(config, "min_drip_epochs")
    assert config.min_drip_epochs == 5

    # Asymmetric clipping ratio
    assert hasattr(config, "negative_drip_ratio")
    assert config.negative_drip_ratio == 0.5  # -0.05 = -0.5 * 0.1


def test_drip_config_validation() -> None:
    """Drip configuration validates bounds."""
    from esper.simic.rewards.contribution import ContributionRewardConfig
    import pytest

    # drip_fraction must be in [0, 1]
    with pytest.raises(ValueError, match="drip_fraction"):
        ContributionRewardConfig(drip_fraction=1.5)

    with pytest.raises(ValueError, match="drip_fraction"):
        ContributionRewardConfig(drip_fraction=-0.1)

    # max_drip_per_epoch must be positive
    with pytest.raises(ValueError, match="max_drip_per_epoch"):
        ContributionRewardConfig(max_drip_per_epoch=0.0)


def test_basic_plus_mode_enables_drip_by_default() -> None:
    """BASIC_PLUS mode automatically sets drip_fraction=0.7."""
    from esper.simic.rewards.contribution import ContributionRewardConfig, RewardMode

    # BASIC mode: drip disabled
    basic_config = ContributionRewardConfig(reward_mode=RewardMode.BASIC)
    assert basic_config.drip_fraction == 0.0

    # BASIC_PLUS mode: drip enabled automatically
    basic_plus_config = ContributionRewardConfig(reward_mode=RewardMode.BASIC_PLUS)
    assert basic_plus_config.drip_fraction == 0.7

    # BASIC_PLUS with explicit drip_fraction: honor the explicit value
    explicit_config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC_PLUS,
        drip_fraction=0.5,
    )
    assert explicit_config.drip_fraction == 0.5
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_config.py::test_drip_config_fields_exist -v`
Expected: FAIL with "AttributeError: drip_fraction"

**Step 3: Write minimal implementation**

Add to `ContributionRewardConfig` dataclass in `contribution.py` (after line ~230):

```python
    # === Drip Reward Configuration (BASIC_PLUS mode) ===
    # Post-fossilization accountability: drip reward paid over remaining epochs
    # based on continued seed contribution. DRL Expert review 2026-01-12.

    # Fraction of fossilize reward paid as drip (vs immediate)
    # 0.0 = disable drip (BASIC mode default)
    # 0.7 = 70% drip, 30% immediate (BASIC_PLUS mode)
    drip_fraction: float = 0.0  # Default: disabled (BASIC mode unchanged)

    # Maximum drip per epoch (prevents variance explosion)
    max_drip_per_epoch: float = 0.1

    # Minimum remaining epochs for drip calculation (floor for epoch normalization)
    # Prevents division by near-zero for late fossilization
    min_drip_epochs: int = 5

    # Ratio for asymmetric negative clipping (negative_clip = -ratio * max_drip)
    # DRL Expert: asymmetric clipping prevents death spirals while allowing full positive signal
    negative_drip_ratio: float = 0.5

    def __post_init__(self) -> None:
        """Validate drip configuration and set mode-specific defaults."""
        # BASIC_PLUS mode: enable drip by default if not explicitly set
        if self.reward_mode == RewardMode.BASIC_PLUS and self.drip_fraction == 0.0:
            object.__setattr__(self, 'drip_fraction', 0.7)

        if self.drip_fraction < 0.0 or self.drip_fraction > 1.0:
            raise ValueError("drip_fraction must be in [0.0, 1.0]")
        if self.max_drip_per_epoch <= 0:
            raise ValueError("max_drip_per_epoch must be positive")
        if self.min_drip_epochs < 1:
            raise ValueError("min_drip_epochs must be >= 1")
        if self.negative_drip_ratio < 0 or self.negative_drip_ratio > 1.0:
            raise ValueError("negative_drip_ratio must be in [0.0, 1.0]")
```

Note: The `object.__setattr__` is needed because dataclasses with `frozen=True` or `slots=True` need this pattern to modify attributes in `__post_init__`. If the dataclass is not frozen, you can use `self.drip_fraction = 0.7` directly.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_config.py::test_drip_config_fields_exist tests/simic/test_config.py::test_drip_config_validation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/test_config.py
git commit -m "feat(simic): add drip reward configuration fields

Add drip_fraction, max_drip_per_epoch, min_drip_epochs, and
negative_drip_ratio to ContributionRewardConfig for post-fossilization
accountability in BASIC mode.

DRL Expert review: 70/30 split approved with asymmetric clipping.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add FossilizedSeedDripState Dataclass

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py`
- Test: `tests/simic/rewards/test_drip_state.py` (new file)

**Step 1: Write the failing test**

```python
# tests/simic/rewards/test_drip_state.py
"""Tests for FossilizedSeedDripState drip calculation."""

import pytest
from esper.simic.rewards.contribution import FossilizedSeedDripState


class TestFossilizedSeedDripState:
    """Unit tests for drip state calculation."""

    def test_remaining_epochs_property(self) -> None:
        """remaining_epochs computed correctly from max_epochs - fossilize_epoch."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=20,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=1.96 / 130,
        )
        assert state.remaining_epochs == 130

    def test_compute_epoch_drip_positive_contribution(self) -> None:
        """Positive contribution yields positive drip, capped at max_drip."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=20,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.015,  # ~1.96 / 130
        )

        drip = state.compute_epoch_drip(
            current_contribution=3.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.015 * 3.0 = 0.045, below cap
        assert drip == pytest.approx(0.045, abs=0.001)

    def test_compute_epoch_drip_positive_clipping(self) -> None:
        """Large positive drip is clipped to max_drip."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=140,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.196,  # 1.96 / 10 (late fossilization)
        )

        drip = state.compute_epoch_drip(
            current_contribution=5.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.196 * 5.0 = 0.98, clipped to 0.1
        assert drip == pytest.approx(0.1, abs=0.001)

    def test_compute_epoch_drip_negative_contribution(self) -> None:
        """Negative contribution yields negative drip (penalty)."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=20,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.015,
        )

        drip = state.compute_epoch_drip(
            current_contribution=-2.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.015 * (-2.0) = -0.03, below asymmetric cap
        assert drip == pytest.approx(-0.03, abs=0.001)
        assert drip < 0, "Negative contribution should produce negative drip"

    def test_compute_epoch_drip_asymmetric_clipping(self) -> None:
        """Large negative drip is clipped asymmetrically (tighter than positive)."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=140,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.196,
        )

        drip = state.compute_epoch_drip(
            current_contribution=-5.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.196 * (-5.0) = -0.98, clipped to -0.05 (asymmetric)
        assert drip == pytest.approx(-0.05, abs=0.001)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/rewards/test_drip_state.py -v`
Expected: FAIL with "ImportError: cannot import name 'FossilizedSeedDripState'"

**Step 3: Write minimal implementation**

Add to `contribution.py` (after the imports, before ContributionRewardConfig):

```python
@dataclass(slots=True)
class FossilizedSeedDripState:
    """Tracks drip reward state for a fossilized seed.

    Created at fossilization, updated each epoch until episode end.
    Drip provides post-fossilization accountability: if the seed degrades
    after fossilization, drip becomes negative (penalty).

    DRL Expert review 2026-01-12: Per-epoch counterfactual is the correct
    signal for drip (Markovian, answers "is this seed helping right now?").
    Terminal Shapley is used for telemetry only, not reward calculation.
    """

    seed_id: str
    slot_id: str
    fossilize_epoch: int
    max_epochs: int

    # Total drip pool (70% of original fossilize bonus)
    drip_total: float

    # Per-epoch drip scale (normalized by remaining epochs)
    drip_scale: float

    # Accumulated drip paid so far (for telemetry)
    drip_paid: float = 0.0

    @property
    def remaining_epochs(self) -> int:
        """Epochs remaining when seed was fossilized."""
        return self.max_epochs - self.fossilize_epoch

    def compute_epoch_drip(
        self,
        current_contribution: float,
        max_drip: float,
        negative_drip_ratio: float,
    ) -> float:
        """Compute drip for this epoch with asymmetric clipping.

        Args:
            current_contribution: Counterfactual contribution this epoch
            max_drip: Maximum positive drip per epoch
            negative_drip_ratio: Ratio for negative clip (neg_clip = -ratio * max_drip)

        Returns:
            Clipped drip amount for this epoch
        """
        # Base drip from contribution
        raw_drip = self.drip_scale * current_contribution

        # Asymmetric clipping (DRL Expert: prevents death spirals)
        if raw_drip >= 0:
            return min(raw_drip, max_drip)
        else:
            negative_clip = -negative_drip_ratio * max_drip
            return max(raw_drip, negative_clip)
```

Also add to `__all__`:
```python
"FossilizedSeedDripState",
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/rewards/test_drip_state.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/test_drip_state.py
git commit -m "feat(simic): add FossilizedSeedDripState for drip tracking

Dataclass tracks per-seed drip state with asymmetric clipping.
Positive capped at +0.1, negative at -0.05 to prevent death spirals.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Modify compute_basic_reward for Drip Support

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:753-897`
- Test: `tests/simic/rewards/test_reward_golden.py`

**Step 1: Write the failing test**

```python
# tests/simic/rewards/test_reward_golden.py

def test_basic_mode_fossilize_drip_split() -> None:
    """FOSSILIZE in BASIC mode splits reward into immediate and drip pool.

    With drip_fraction=0.7, only 30% of the bonus is paid immediately.
    """
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.simic.rewards.types import SeedInfo
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
    )

    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,  # Full legitimacy
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
    )

    result = compute_basic_reward(
        acc_delta=5.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=20,
        max_epochs=150,
        seed_info=seed_info,
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    # Result now includes drip state
    reward, rent, growth, pbrs, foss_bonus, new_drip, drip_epoch = result

    # Full bonus = (0.3 + 0.5 * harmonic(5,5)) * 1.0 = 0.3 + 0.5*5 = 2.8
    # Immediate = 30% = 0.84
    assert foss_bonus == pytest.approx(0.84, abs=0.05)

    # Drip state should be created
    assert new_drip is not None
    assert isinstance(new_drip, FossilizedSeedDripState)
    assert new_drip.drip_total == pytest.approx(1.96, abs=0.05)  # 70% of 2.8
    assert new_drip.remaining_epochs == 130  # 150 - 20
    assert new_drip.seed_id == "test-seed"
    assert new_drip.slot_id == "r0c1"

    # No drip this epoch (first epoch after fossilize)
    assert drip_epoch == 0.0


def test_basic_mode_drip_disabled_when_fraction_zero() -> None:
    """When drip_fraction=0, full bonus is immediate and no drip state created."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
    )
    from esper.simic.rewards.types import SeedInfo
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig(
        drip_fraction=0.0,  # Disabled
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
    )

    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
    )

    result = compute_basic_reward(
        acc_delta=5.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=20,
        max_epochs=150,
        seed_info=seed_info,
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    reward, rent, growth, pbrs, foss_bonus, new_drip, drip_epoch = result

    # Full bonus = 2.8 (100% immediate)
    assert foss_bonus == pytest.approx(2.8, abs=0.1)

    # No drip state created
    assert new_drip is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/rewards/test_reward_golden.py::test_basic_mode_fossilize_drip_split -v`
Expected: FAIL with "TypeError: compute_basic_reward() got unexpected keyword arguments"

**Step 3: Write minimal implementation**

Modify `compute_basic_reward` signature and implementation (in `contribution.py`):

```python
def compute_basic_reward(
    *,
    acc_delta: float,
    effective_seed_params: float | None,
    total_params: int,
    host_params: int,
    config: ContributionRewardConfig,
    epoch: int,
    max_epochs: int,
    seed_info: SeedInfo | None = None,
    action: LifecycleOp = LifecycleOp.WAIT,
    seed_contribution: float | None = None,
    # NEW: Drip-related parameters
    seed_id: str | None = None,
    slot_id: str | None = None,
    fossilized_drip_states: list[FossilizedSeedDripState] | None = None,
    fossilized_contributions: dict[str, float] | None = None,
) -> tuple[float, float, float, float, float, FossilizedSeedDripState | None, float]:
    """Compute BASIC reward with drip support.

    Returns:
        tuple of (total_reward, rent_penalty, growth_ratio, pbrs_bonus,
                  fossilize_bonus, new_drip_state, drip_this_epoch)

        new_drip_state: FossilizedSeedDripState if FOSSILIZE created one, else None
        drip_this_epoch: Total drip reward from all fossilized seeds this epoch
    """
    # ... existing PBRS and rent code unchanged ...

    # Component 3: Contribution-Scaled Fossilization with Drip Split
    fossilize_bonus = 0.0
    new_drip_state: FossilizedSeedDripState | None = None

    if action == LifecycleOp.FOSSILIZE and seed_info is not None:
        is_valid = seed_info.stage == STAGE_HOLDING
        if is_valid:
            improvement = seed_info.total_improvement
            meets_threshold = (
                seed_contribution is not None
                and seed_contribution >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
            )

            if improvement > 0 and meets_threshold:
                # Calculate full bonus using harmonic mean
                combined = _compute_attributed_value(
                    progress=improvement,
                    seed_contribution=seed_contribution,
                    formula=config.attribution_formula,
                )
                legitimacy_discount = min(1.0, seed_info.epochs_in_stage / MIN_HOLDING_EPOCHS)
                full_bonus = (
                    config.basic_fossilize_base_bonus
                    + config.basic_contribution_scale * combined
                ) * legitimacy_discount

                # Split into immediate and drip
                immediate_fraction = 1.0 - config.drip_fraction
                fossilize_bonus = full_bonus * immediate_fraction

                # Create drip state for the drip portion
                remaining_epochs = max_epochs - epoch
                if config.drip_fraction > 0 and remaining_epochs > 0:
                    effective_remaining = max(remaining_epochs, config.min_drip_epochs)
                    drip_total = full_bonus * config.drip_fraction
                    drip_scale = drip_total / effective_remaining

                    new_drip_state = FossilizedSeedDripState(
                        seed_id=seed_id or "unknown",
                        slot_id=slot_id or "unknown",
                        fossilize_epoch=epoch,
                        max_epochs=max_epochs,
                        drip_total=drip_total,
                        drip_scale=drip_scale,
                    )
            elif improvement <= 0 and seed_contribution is not None and seed_contribution > 0.1:
                fossilize_bonus = config.basic_fossilize_invalid_penalty
            else:
                fossilize_bonus = config.basic_fossilize_noncontributing_penalty
        else:
            fossilize_bonus = config.basic_fossilize_invalid_penalty

    reward += fossilize_bonus

    # Component 5: Drip Payout for Existing Fossilized Seeds
    drip_this_epoch = 0.0
    if fossilized_drip_states and fossilized_contributions:
        for drip_state in fossilized_drip_states:
            contribution = fossilized_contributions.get(drip_state.seed_id, 0.0)
            # Only pay drip for positive contribution (DRL Expert: replaces diminishing returns)
            if contribution > 0:
                epoch_drip = drip_state.compute_epoch_drip(
                    current_contribution=contribution,
                    max_drip=config.max_drip_per_epoch,
                    negative_drip_ratio=config.negative_drip_ratio,
                )
                drip_this_epoch += epoch_drip
                drip_state.drip_paid += epoch_drip
            elif contribution < 0:
                # Negative contribution: apply penalty (asymmetrically clipped)
                epoch_drip = drip_state.compute_epoch_drip(
                    current_contribution=contribution,
                    max_drip=config.max_drip_per_epoch,
                    negative_drip_ratio=config.negative_drip_ratio,
                )
                drip_this_epoch += epoch_drip
                drip_state.drip_paid += epoch_drip

    reward += drip_this_epoch

    # ... existing terminal bonus code ...

    return reward, rent_penalty, growth_ratio, pbrs_bonus, fossilize_bonus, new_drip_state, drip_this_epoch
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/rewards/test_reward_golden.py::test_basic_mode_fossilize_drip_split tests/simic/rewards/test_reward_golden.py::test_basic_mode_drip_disabled_when_fraction_zero -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/test_reward_golden.py
git commit -m "feat(simic): implement drip split in compute_basic_reward

FOSSILIZE now pays 30% immediate + creates 70% drip pool.
Per-epoch drip calculated from counterfactual contribution.
Asymmetric clipping: +0.1 positive, -0.05 negative.

DRL Expert: epoch-normalized drip prevents early-fossilization gaming.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Drip Golden Tests for Payout Scenarios

**Files:**
- Modify: `tests/simic/rewards/test_reward_golden.py`

**Step 1: Write the failing tests**

```python
def test_basic_mode_drip_payout_positive() -> None:
    """Drip payout rewards continued positive contribution."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.leyline import LifecycleOp

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
        negative_drip_ratio=0.5,
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=20,
        max_epochs=150,
        drip_total=1.96,
        drip_scale=1.96 / 130,  # ~0.015 per epoch
    )

    result = compute_basic_reward(
        acc_delta=0.1,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=25,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": 3.0},
    )

    reward, _, _, _, _, _, drip_epoch = result

    # Drip = drip_scale * contribution = 0.015 * 3.0 = 0.045
    assert drip_epoch == pytest.approx(0.045, abs=0.005)
    assert drip_epoch > 0


def test_basic_mode_drip_payout_negative() -> None:
    """Drip payout penalizes negative contribution (seed now hurting)."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.leyline import LifecycleOp

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
        negative_drip_ratio=0.5,
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=20,
        max_epochs=150,
        drip_total=1.96,
        drip_scale=0.015,
    )

    result = compute_basic_reward(
        acc_delta=-1.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=25,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": -2.0},
    )

    reward, _, _, _, _, _, drip_epoch = result

    # Drip = 0.015 * (-2.0) = -0.03
    assert drip_epoch == pytest.approx(-0.03, abs=0.005)
    assert drip_epoch < 0, "Negative contribution should produce negative drip"


def test_basic_mode_drip_asymmetric_clipping() -> None:
    """Large negative drip is clipped more aggressively than positive."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.leyline import LifecycleOp

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
        negative_drip_ratio=0.5,  # -0.05 cap
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=140,  # Late fossilization = high drip_scale
        max_epochs=150,
        drip_total=1.96,
        drip_scale=1.96 / 10,  # ~0.196 per epoch
    )

    # Test positive clipping
    result_pos = compute_basic_reward(
        acc_delta=0.1,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=145,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": 5.0},
    )
    _, _, _, _, _, _, drip_pos = result_pos
    assert drip_pos == pytest.approx(0.1, abs=0.001), "Positive clipped to +0.1"

    # Reset drip_paid for next test
    drip_state.drip_paid = 0.0

    # Test negative clipping (asymmetric - tighter)
    result_neg = compute_basic_reward(
        acc_delta=-1.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=145,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": -5.0},
    )
    _, _, _, _, _, _, drip_neg = result_neg
    assert drip_neg == pytest.approx(-0.05, abs=0.001), "Negative clipped to -0.05 (asymmetric)"
```

**Step 2-5: Run, verify, commit**

Run: `uv run pytest tests/simic/rewards/test_reward_golden.py -v -k drip`
Expected: PASS

```bash
git add tests/simic/rewards/test_reward_golden.py
git commit -m "test(simic): add drip payout golden tests

Cover positive, negative, and asymmetric clipping scenarios.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Add Property Tests for Drip Anti-Gaming

**Files:**
- Modify: `tests/simic/properties/test_reward_antigaming.py`

**Step 1: Write the property tests**

```python
# tests/simic/properties/test_reward_antigaming.py

import pytest
from hypothesis import given, settings, strategies as st

from esper.simic.rewards.contribution import (
    ContributionRewardConfig,
    FossilizedSeedDripState,
)


class TestDripAntiGaming:
    """Property tests for drip reward anti-gaming guarantees."""

    @given(
        fossilize_epoch=st.integers(min_value=10, max_value=140),
        contribution_at_foss=st.floats(min_value=1.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_epoch_normalization_prevents_early_gaming(
        self, fossilize_epoch: int, contribution_at_foss: float
    ) -> None:
        """Expected total drip is roughly equal regardless of fossilization timing.

        Without epoch normalization, early fossilization would capture more
        total drip. Normalization by remaining_epochs should equalize this.
        """
        max_epochs = 150
        config = ContributionRewardConfig(
            drip_fraction=0.7,
            max_drip_per_epoch=0.1,
            min_drip_epochs=5,
        )

        remaining = max(max_epochs - fossilize_epoch, config.min_drip_epochs)

        # Simulate drip with constant contribution
        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=fossilize_epoch,
            max_epochs=max_epochs,
            drip_total=2.0,
            drip_scale=2.0 / remaining,
        )

        # Sum drip over remaining epochs with constant contribution
        total_drip = 0.0
        actual_remaining = max_epochs - fossilize_epoch
        for _ in range(actual_remaining):
            epoch_drip = drip_state.compute_epoch_drip(
                current_contribution=contribution_at_foss,
                max_drip=config.max_drip_per_epoch,
                negative_drip_ratio=0.5,
            )
            total_drip += epoch_drip

        # Expected total (without clipping): drip_total * contribution
        expected_max = 2.0 * contribution_at_foss

        # Total should be bounded regardless of timing
        assert total_drip <= expected_max + 0.1, (
            f"Total drip {total_drip} exceeds expected max {expected_max} "
            f"for fossilize_epoch={fossilize_epoch}"
        )

    @given(
        contribution_sequence=st.lists(
            st.floats(min_value=-5.0, max_value=5.0),
            min_size=10,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_negative_drip_for_degrading_seeds(
        self, contribution_sequence: list[float]
    ) -> None:
        """Seeds that degrade post-fossilization receive negative drip (penalty)."""
        config = ContributionRewardConfig(
            drip_fraction=0.7,
            max_drip_per_epoch=0.1,
            negative_drip_ratio=0.5,
        )

        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=100,
            max_epochs=150,
            drip_total=2.0,
            drip_scale=2.0 / 50,
        )

        total_drip = 0.0
        negative_epochs = 0
        for contrib in contribution_sequence:
            epoch_drip = drip_state.compute_epoch_drip(
                current_contribution=contrib,
                max_drip=config.max_drip_per_epoch,
                negative_drip_ratio=0.5,
            )
            total_drip += epoch_drip
            if contrib < 0:
                negative_epochs += 1
                # Negative contribution should produce negative or zero drip
                assert epoch_drip <= 0, (
                    f"Negative contribution {contrib} produced positive drip {epoch_drip}"
                )

        # If mostly negative contributions, total drip should be negative
        if negative_epochs > len(contribution_sequence) * 0.7:
            assert total_drip < 0, (
                f"Mostly negative contributions ({negative_epochs}/{len(contribution_sequence)}) "
                f"should produce negative total drip, got {total_drip}"
            )

    @given(
        drip_scale=st.floats(min_value=0.01, max_value=1.0),
        contribution=st.floats(min_value=-10.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_asymmetric_clipping_invariant(
        self, drip_scale: float, contribution: float
    ) -> None:
        """Asymmetric clipping: positive cap is 2x the negative cap."""
        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=100,
            max_epochs=150,
            drip_total=2.0,
            drip_scale=drip_scale,
        )

        max_drip = 0.1
        negative_ratio = 0.5

        drip = drip_state.compute_epoch_drip(
            current_contribution=contribution,
            max_drip=max_drip,
            negative_drip_ratio=negative_ratio,
        )

        # Invariant: drip is always in [-0.05, +0.1]
        assert drip >= -max_drip * negative_ratio - 1e-9
        assert drip <= max_drip + 1e-9
```

**Step 2-5: Run, verify, commit**

Run: `uv run pytest tests/simic/properties/test_reward_antigaming.py::TestDripAntiGaming -v`
Expected: PASS

```bash
git add tests/simic/properties/test_reward_antigaming.py
git commit -m "test(simic): add property tests for drip anti-gaming

Verify epoch normalization, negative drip, and asymmetric clipping.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update reward dispatcher for BASIC_PLUS mode

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py:117-143`
- Test: `tests/simic/test_reward_modes.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_reward_modes.py

def test_basic_plus_mode_dispatches_correctly() -> None:
    """BASIC_PLUS mode uses compute_basic_reward with drip enabled."""
    from esper.simic.rewards.contribution import ContributionRewardConfig, RewardMode
    from esper.simic.rewards.rewards import compute_reward
    from esper.simic.rewards.types import ContributionRewardInputs, SeedInfo
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig(reward_mode=RewardMode.BASIC_PLUS)

    # BASIC_PLUS should have drip enabled by default
    assert config.drip_fraction == 0.7

    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
    )

    inputs = ContributionRewardInputs(
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
        val_acc=0.85,
        seed_info=seed_info,
        epoch=20,
        max_epochs=150,
        total_params=110_000,
        host_params=100_000,
        acc_at_germination=0.70,
        acc_delta=5.0,
        config=config,
        return_components=True,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    result = compute_reward(inputs)
    assert isinstance(result, tuple)
    reward, components = result

    # Should have drip state created (30% immediate, 70% drip pool)
    assert components.new_drip_state is not None
```

**Step 2: Update dispatcher**

Add BASIC_PLUS branch that creates a config with drip enabled:

```python
# In rewards.py, add BASIC_PLUS branch (after BASIC):

elif config.reward_mode in (RewardMode.BASIC, RewardMode.BASIC_PLUS):
    # BASIC_PLUS: Override drip_fraction to 0.7 (drip enabled)
    # BASIC: Uses config default (drip_fraction=0.0, drip disabled)
    effective_config = config
    if config.reward_mode == RewardMode.BASIC_PLUS and config.drip_fraction == 0.0:
        # Create a config with drip enabled for BASIC_PLUS
        effective_config = dataclasses.replace(config, drip_fraction=0.7)

    result = compute_basic_reward(
        acc_delta=inputs.acc_delta,
        effective_seed_params=inputs.effective_seed_params,
        total_params=inputs.total_params,
        host_params=inputs.host_params,
        config=effective_config,
        epoch=inputs.epoch,
        max_epochs=inputs.max_epochs,
        seed_info=inputs.seed_info,
        action=inputs.action,
        seed_contribution=inputs.seed_contribution,
        seed_id=inputs.seed_id,
        slot_id=inputs.slot_id,
        fossilized_drip_states=inputs.fossilized_drip_states,
        fossilized_contributions=inputs.fossilized_contributions,
    )
    reward, rent_penalty, growth_ratio, pbrs_bonus, fossilize_bonus, new_drip_state, drip_epoch = result

    if inputs.return_components:
        components = RewardComponentsTelemetry()
        components.total_reward = reward
        components.action_name = inputs.action.name
        components.epoch = inputs.epoch
        components.seed_stage = inputs.seed_info.stage if inputs.seed_info else None
        components.val_acc = inputs.val_acc
        components.base_acc_delta = inputs.acc_delta
        components.compute_rent = -rent_penalty
        components.growth_ratio = growth_ratio
        components.pbrs_bonus = pbrs_bonus
        components.fossilize_terminal_bonus = fossilize_bonus
        components.drip_this_epoch = drip_epoch
        components.new_drip_state = new_drip_state
        return reward, components
```

**Step 2: Update ContributionRewardInputs**

Add drip-related fields to `ContributionRewardInputs` in `types.py`:

```python
# In types.py, add to ContributionRewardInputs:
    # Drip reward fields (BASIC mode post-fossilization accountability)
    fossilized_drip_states: list[Any] | None = None  # list[FossilizedSeedDripState]
    fossilized_contributions: dict[str, float] | None = None
```

**Step 3: Commit**

```bash
git add src/esper/simic/rewards/rewards.py src/esper/simic/rewards/types.py
git commit -m "feat(simic): update reward dispatcher for drip support

Pass drip parameters through ContributionRewardInputs and handle
new compute_basic_reward return signature.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add Drip Telemetry Fields

**Files:**
- Modify: `src/esper/simic/rewards/reward_telemetry.py`
- Modify: `src/esper/leyline/telemetry.py`

**Step 1: Add to RewardComponentsTelemetry**

```python
# In reward_telemetry.py, add fields:
    # Drip reward fields (BASIC mode post-fossilization accountability)
    drip_this_epoch: float = 0.0
    drip_immediate_bonus: float = 0.0  # 30% portion
    drip_deferred_total: float = 0.0  # 70% pool created this epoch
    num_drip_sources: int = 0  # Number of fossilized seeds contributing drip
    new_drip_state: Any = None  # FossilizedSeedDripState if created
```

**Step 2: Update to_dict()**

```python
def to_dict(self) -> dict[str, Any]:
    # ... existing fields ...
    d["drip_this_epoch"] = self.drip_this_epoch
    d["drip_immediate_bonus"] = self.drip_immediate_bonus
    d["drip_deferred_total"] = self.drip_deferred_total
    d["num_drip_sources"] = self.num_drip_sources
    return d
```

**Step 3: Commit**

```bash
git add src/esper/simic/rewards/reward_telemetry.py
git commit -m "feat(simic): add drip telemetry fields

Track drip_this_epoch, drip_immediate_bonus, drip_deferred_total,
and num_drip_sources for BASIC mode monitoring.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integrate Drip State Management in Vectorized Trainer

**Files:**
- Modify: `src/esper/simic/training/vectorized_trainer.py`
- Test: Integration test via manual training run

This is the final integration task. The vectorized trainer needs to:

1. Track `_drip_states: list[list[FossilizedSeedDripState]]` per environment
2. Pass drip states to `compute_basic_reward`
3. Collect `new_drip_state` from fossilize actions
4. Reset drip states at episode boundaries
5. Compute `fossilized_contributions` from counterfactual validation

**Implementation sketch:**

```python
# In VectorizedPPOTrainer.__init__:
self._drip_states: list[list[FossilizedSeedDripState]] = [[] for _ in range(n_envs)]

# In reward computation (when RewardMode.BASIC):
result = compute_basic_reward(
    # ... existing args ...
    fossilized_drip_states=self._drip_states[env_id],
    fossilized_contributions=self._get_fossilized_contributions(env_id),
)
reward, rent, growth, pbrs, foss_bonus, new_drip, drip_epoch = result

if new_drip is not None:
    self._drip_states[env_id].append(new_drip)

# In _reset_env:
self._drip_states[env_id].clear()
```

**Commit:**

```bash
git add src/esper/simic/training/vectorized_trainer.py
git commit -m "feat(simic): integrate drip state in vectorized trainer

Track FossilizedSeedDripState per environment.
Collect drip on fossilize, pay drip each epoch, reset on episode end.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] `uv run pytest tests/simic/ -v` - All tests pass
- [ ] `uv run pytest tests/simic/properties/ -v` - Property tests pass
- [ ] `uv run python -m esper.scripts.train ppo --preset cifar10 --reward-mode basic --rounds 5` - BASIC mode unchanged
- [ ] `uv run python -m esper.scripts.train ppo --preset cifar10 --reward-mode basic_plus --rounds 5` - BASIC_PLUS with drip
- [ ] Check telemetry: `drip_this_epoch` field appears in BASIC_PLUS reward telemetry
- [ ] A/B comparison: Run BASIC vs BASIC_PLUS on same seed to compare behavior

---

## Rollback Plan

If drip introduces training instability:

1. Switch from `--reward-mode basic_plus` to `--reward-mode basic` (instant rollback)
2. Or set `drip_fraction=0.0` in config to disable drip within BASIC_PLUS
3. Investigate via telemetry which component caused issues

**Key advantage of parallel mode**: BASIC is completely unchanged, so you can always fall back to it.

---

## Summary of DRL Expert Modifications

| Original Plan | Final Implementation |
|---------------|---------------------|
| Diminishing returns for multiple fossils | **Removed** - replaced with contribution threshold |
| Symmetric +/-0.1 clipping | **Asymmetric**: +0.1 positive, -0.05 negative |
| Shapley reconciliation considered | **Not implemented** - per-epoch counterfactual is correct signal |
| ~100% effective value assumed | **Documented**: ~85% effective due to γ discounting |
