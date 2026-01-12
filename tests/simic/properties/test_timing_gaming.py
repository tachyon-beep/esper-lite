"""Property-based tests for anti-timing-gaming reward behavior.

These tests verify that the D3 timing discount and harmonic attribution
formula correctly discourage early germination gaming patterns.

Tier 3: Anti-Gaming Properties
- Early germination receives discounted attribution
- Harmonic mean bounds attribution when progress >> contribution
- Combined fixes reduce incentive for timing gaming
"""

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


# ============================================================================
# Expert-Recommended Property Tests (DRL Expert / PyTorch Expert Reviews)
# ============================================================================


@given(
    e1=st.integers(min_value=0, max_value=100),
    e2=st.integers(min_value=0, max_value=100),
    warmup=st.integers(min_value=5, max_value=50),
    floor=st.floats(min_value=0.1, max_value=0.9),
)
@settings(max_examples=200)
def test_timing_discount_monotonicity(e1: int, e2: int, warmup: int, floor: float):
    """D3-Timing: Discount is monotonically non-decreasing with germination epoch.

    Later germination (higher epoch) should yield higher or equal discount,
    incentivizing waiting for stable training before germinating.
    """
    d1 = _compute_timing_discount(e1, warmup, floor)
    d2 = _compute_timing_discount(e2, warmup, floor)

    if e1 >= e2:
        assert d1 >= d2, (
            f"Monotonicity violated: epoch {e1} got discount {d1} < "
            f"epoch {e2} got {d2} (warmup={warmup}, floor={floor})"
        )


@given(value=st.floats(min_value=0.01, max_value=100.0))
@settings(max_examples=200)
def test_formula_symmetry_when_equal(value: float):
    """D3-Attribution: All formulas yield identical values when progress == contribution.

    This is a mathematical identity for geometric/harmonic/minimum means:
    - geometric: sqrt(a*a) = a
    - harmonic: 2*a*a/(a+a) = a
    - minimum: min(a, a) = a
    """
    geometric = _compute_attributed_value(value, value, "geometric")
    harmonic = _compute_attributed_value(value, value, "harmonic")
    minimum = _compute_attributed_value(value, value, "minimum")

    assert abs(geometric - value) < 1e-6, f"Geometric({value}, {value}) = {geometric} != {value}"
    assert abs(harmonic - value) < 1e-6, f"Harmonic({value}, {value}) = {harmonic} != {value}"
    assert abs(minimum - value) < 1e-6, f"Minimum({value}, {value}) = {minimum} != {value}"


@given(
    progress=st.floats(min_value=0.01, max_value=100.0),
    contribution=st.floats(min_value=0.01, max_value=100.0),
)
@settings(max_examples=300)
def test_attribution_ordering_invariant(progress: float, contribution: float):
    """D3-Attribution: minimum <= harmonic <= geometric always holds.

    This ordering is a mathematical property of these means for positive values:
    - Minimum is the most conservative (smallest)
    - Harmonic is dominated by the smaller value
    - Geometric is the middle ground
    The equality holds when progress == contribution.
    """
    geometric = _compute_attributed_value(progress, contribution, "geometric")
    harmonic = _compute_attributed_value(progress, contribution, "harmonic")
    minimum = _compute_attributed_value(progress, contribution, "minimum")

    # Allow small floating point tolerance
    eps = 1e-6

    assert minimum <= harmonic + eps, (
        f"Ordering violated: minimum {minimum} > harmonic {harmonic} "
        f"for progress={progress}, contribution={contribution}"
    )
    assert harmonic <= geometric + eps, (
        f"Ordering violated: harmonic {harmonic} > geometric {geometric} "
        f"for progress={progress}, contribution={contribution}"
    )
