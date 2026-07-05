"""Phase 0: the J integrand (alpha-weighted counterfactual residency) + offline J.

J = Σ_seed (1/params) Σ_t c_t·α_t over on-output-path steps (stage ≥ BLENDING, α>0),
where c_t = the PER-SEED LOO marginal (``seed_contribution``, PIN A) — NOT the env-joint
counterfactual. ``SeedResidencyAccumulator`` is the per-seed accumulator the live EnvState
hook uses; testing it here pins the live accumulator's math.

Reviewer-mandated properties:
- (test b) the §1.6 worked example reproduces the hand-computed value;
- (test d) the freeloader guard: a high-α, ZERO-counterfactual seed scores ≤ no-op (0),
  whereas the REJECTED raw-residency form (Σα) would score it positive;
- the on-output-path gate (stage ≥ BLENDING, α>0) and the None-LOO → contribute-0 fallback.
"""

from __future__ import annotations

import pytest

from esper.simic.rewards.residency import (
    SeedResidencyAccumulator,
    compute_residency_j,
)
from esper.simic.rewards.types import (
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
    STAGE_HOLDING,
    STAGE_TRAINING,
)


def test_worked_example_reproduces_hand_value() -> None:
    """§1.6: params=2000; (α, c) = (.25,4),(.50,6),(1.0,8) over BLENDING/HOLDING.
    Integral = .25*4 + .50*6 + 1.0*8 = 1 + 3 + 8 = 12.0; J = 12/2000 = 6.0e-3."""
    acc = SeedResidencyAccumulator(params=2000)
    acc.add_step(stage=STAGE_BLENDING, alpha=0.25, seed_contribution=4.0)
    acc.add_step(stage=STAGE_BLENDING, alpha=0.50, seed_contribution=6.0)
    acc.add_step(stage=STAGE_HOLDING, alpha=1.0, seed_contribution=8.0)

    assert acc.cf_weighted_integral == pytest.approx(12.0)
    assert acc.j() == pytest.approx(6.0e-3)
    assert compute_residency_j([acc]) == pytest.approx(6.0e-3)


def test_freeloader_scores_at_most_no_op() -> None:
    """A high-α, zero-counterfactual parked seed must score ≤ a no-op (0). The cf-weight
    zeroes it; the REJECTED raw-residency form (Σα) would credit it positively."""
    free = SeedResidencyAccumulator(params=2000)
    free.add_step(stage=STAGE_BLENDING, alpha=0.25, seed_contribution=0.0)
    free.add_step(stage=STAGE_HOLDING, alpha=0.50, seed_contribution=0.0)
    free.add_step(stage=STAGE_HOLDING, alpha=1.0, seed_contribution=0.0)

    assert free.cf_weighted_integral == pytest.approx(0.0)
    assert free.j() == pytest.approx(0.0)
    assert free.j() <= 0.0 + 1e-12  # ≤ no-op
    # The rejected raw-residency form would have credited it:
    assert free.raw_alpha_integral == pytest.approx(1.75)


def test_negative_counterfactual_scores_below_no_op() -> None:
    """A harmful seed (negative LOO marginal) scores < no-op — strictly penalized."""
    harmful = SeedResidencyAccumulator(params=1000)
    harmful.add_step(stage=STAGE_BLENDING, alpha=1.0, seed_contribution=-5.0)
    assert harmful.j() < 0.0


def test_pre_blending_steps_excluded() -> None:
    """A seed in TRAINING (stage < BLENDING) is NOT on the output path: contributes 0
    even with positive contribution and alpha."""
    acc = SeedResidencyAccumulator(params=1000)
    acc.add_step(stage=STAGE_TRAINING, alpha=0.9, seed_contribution=7.0)
    assert acc.cf_weighted_integral == pytest.approx(0.0)
    assert acc.n_on_path_steps == 0


def test_zero_alpha_excluded() -> None:
    """alpha == 0 (not yet on the output path) contributes nothing."""
    acc = SeedResidencyAccumulator(params=1000)
    acc.add_step(stage=STAGE_BLENDING, alpha=0.0, seed_contribution=7.0)
    assert acc.n_on_path_steps == 0
    assert acc.cf_weighted_integral == pytest.approx(0.0)


def test_none_loo_contributes_zero_and_is_counted() -> None:
    """A step with no measured LOO (seed_contribution=None) contributes 0 to the cf
    integral but IS an on-path step (counted, and tracked in n_none_steps) and accrues
    raw alpha residency (so the None-fraction is reportable)."""
    acc = SeedResidencyAccumulator(params=1000)
    acc.add_step(stage=STAGE_BLENDING, alpha=0.5, seed_contribution=None)
    acc.add_step(stage=STAGE_BLENDING, alpha=0.5, seed_contribution=4.0)
    assert acc.cf_weighted_integral == pytest.approx(2.0)  # only the measured step
    assert acc.n_on_path_steps == 2
    assert acc.n_none_steps == 1
    assert acc.raw_alpha_integral == pytest.approx(1.0)


def test_committed_uncommitted_split() -> None:
    """Residency accrued while FOSSILIZED is 'committed'; BLENDING/HOLDING is
    'uncommitted'. Surfaces survival-time accrual (reviewer headline)."""
    acc = SeedResidencyAccumulator(params=1000)
    acc.add_step(stage=STAGE_BLENDING, alpha=1.0, seed_contribution=2.0)   # uncommitted
    acc.add_step(stage=STAGE_HOLDING, alpha=1.0, seed_contribution=3.0)    # uncommitted
    acc.add_step(stage=STAGE_FOSSILIZED, alpha=1.0, seed_contribution=4.0) # committed
    assert acc.cf_weighted_integral == pytest.approx(9.0)
    assert acc.cf_weighted_integral_committed == pytest.approx(4.0)
    assert acc.cf_weighted_integral_uncommitted == pytest.approx(5.0)


def test_j_zero_params_is_safe() -> None:
    """A seed with no measured params must not divide-by-zero (defensive only against a
    genuine 0; production passes effective_seed_params > 0)."""
    acc = SeedResidencyAccumulator(params=0)
    acc.add_step(stage=STAGE_BLENDING, alpha=1.0, seed_contribution=4.0)
    assert acc.j() == 0.0
