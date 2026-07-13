"""Stage-0 instrumentation: the per-term SIGNED additend decomposition (ev-stab).

``decompose_additends(reward_raw, components)`` returns each CONTRIBUTION-family
(SHAPED/ESCROW) reward term as its *signed contribution to ``reward_raw``* plus a
``residual`` key, such that ``sum(values()) == reward_raw`` exhaustively by
construction. This is the per-step feed the value-free per-RETURN variance-share
decomposition (``compute_return_variance_shares``) reconciles over; its ``residual``
share is the ``~0`` reconciliation diagnostic (attributable to telemetry-less
terminal corrections — germination forfeit / pending auto-prune penalty).

Component fields are stored in MIXED sign conventions — ``compute_rent`` /
``escrow_forfeit`` are stored already-negative, while ``occupancy_rent`` /
``fossilized_rent`` are positive magnitudes the reward SUBTRACTS — so naive
summation of the raw fields flips two signs. ``ratio_penalty`` is folded into
``bounded_attribution`` (NOT a separate additend) and ``fossilize_terminal_bonus``
is a dead 0.0 field; both are excluded from the map.

NOTE (branch divergence, ev-stab): the counterfactual-scaffolding term is named
``synergy_bonus`` here — the Shapley line renamed it ``interaction_bonus`` in the
Phase-0 reward redesign (commit 5910c5e6), which ev-stab predates. The two
``ADDITEND_SIGN_MAP`` copies therefore diverge on that one key; reconcile at the
owner-gated branch unification (per the product workspace).
"""

from __future__ import annotations

import dataclasses

import pytest

from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.rewards.partition import ADDITEND_SIGN_MAP, decompose_additends


def _components(**overrides: float | None) -> RewardComponentsTelemetry:
    return RewardComponentsTelemetry(**overrides)


def test_additend_sign_map_keys_are_component_fields() -> None:
    """Pin the map<->dataclass pairing the decomposition's typed field map relies on.

    ``decompose_additends`` maps each ADDITEND_SIGN_MAP key from the typed
    RewardComponentsTelemetry instance. This makes a drift failure (e.g. porting a
    key like ``interaction_bonus`` that does not exist on this branch) a NAMED
    contract violation rather than an incidental runtime error mid-rollout.
    """
    field_names = {f.name for f in dataclasses.fields(RewardComponentsTelemetry)}
    missing = set(ADDITEND_SIGN_MAP) - field_names
    assert not missing, (
        f"ADDITEND_SIGN_MAP keys must be RewardComponentsTelemetry fields: {sorted(missing)}"
    )


def test_decompose_reconciles_to_reward_raw_exactly() -> None:
    """sum(signed additends + residual) == reward_raw, and residual == 0 when the
    named terms fully account for the reward."""
    c = _components(
        bounded_attribution=5.0,
        blending_warning=-0.2,
        holding_warning=-0.1,
        pbrs_bonus=1.5,
        synergy_bonus=0.05,
        compute_rent=-0.3,       # already negative
        alpha_shock=-0.04,
        occupancy_rent=0.2,      # positive magnitude, reward subtracts
        fossilized_rent=0.1,     # positive magnitude, reward subtracts
        action_shaping=0.8,
        terminal_bonus=2.0,
        hindsight_credit=0.0,
    )
    # Signed sum: 5.0 - 0.2 - 0.1 + 1.5 + 0.05 - 0.3 - 0.04 - 0.2 - 0.1 + 0.8 + 2.0 = 8.41
    reward_raw = 8.41
    decomposed = decompose_additends(reward_raw, c)

    assert decomposed["residual"] == pytest.approx(0.0, abs=1e-9)
    assert sum(decomposed.values()) == pytest.approx(reward_raw, abs=1e-9)


def test_synergy_bonus_is_the_scaffolding_additend() -> None:
    """The +1 scaffolding term is ``synergy_bonus`` on ev-stab (NOT interaction_bonus)."""
    assert "synergy_bonus" in ADDITEND_SIGN_MAP
    assert "interaction_bonus" not in ADDITEND_SIGN_MAP
    c = _components(synergy_bonus=0.3)
    decomposed = decompose_additends(0.3, c)
    assert decomposed["synergy_bonus"] == pytest.approx(0.3)
    assert decomposed["residual"] == pytest.approx(0.0)


def test_rents_enter_negated() -> None:
    """occupancy_rent / fossilized_rent are stored as positive magnitudes the reward
    subtracts, so their signed contribution is the NEGATED field."""
    c = _components(occupancy_rent=0.2, fossilized_rent=0.1)
    decomposed = decompose_additends(-0.3, c)
    assert decomposed["occupancy_rent"] == pytest.approx(-0.2)
    assert decomposed["fossilized_rent"] == pytest.approx(-0.1)


def test_already_signed_terms_pass_through() -> None:
    """compute_rent / escrow_forfeit are stored already-signed (negative); their
    signed contribution is the field verbatim, NOT negated."""
    c = _components(compute_rent=-0.3, escrow_forfeit=-0.5)
    decomposed = decompose_additends(-0.8, c)
    assert decomposed["compute_rent"] == pytest.approx(-0.3)
    assert decomposed["escrow_forfeit"] == pytest.approx(-0.5)


def test_residual_captures_untracked_penalty() -> None:
    """A telemetry-less terminal correction (no component field, e.g. pending
    auto-prune penalty) is captured by residual = reward_raw - sum(named signed)."""
    c = _components(bounded_attribution=1.0, pbrs_bonus=0.5)
    # named signed sum = 1.5; reward_raw carries an extra -1.0 auto-prune penalty.
    decomposed = decompose_additends(0.5, c)
    assert decomposed["residual"] == pytest.approx(-1.0)
    assert sum(decomposed.values()) == pytest.approx(0.5)


def test_bounded_attribution_none_is_zero() -> None:
    """None bounded_attribution means 'no attribution this step' = 0.0 contribution
    (matches split_reward_streams)."""
    c = _components(bounded_attribution=None, pbrs_bonus=0.5)
    decomposed = decompose_additends(0.5, c)
    assert decomposed["bounded_attribution"] == pytest.approx(0.0)
    assert decomposed["residual"] == pytest.approx(0.0)


def test_ratio_penalty_and_dead_fields_excluded() -> None:
    """ratio_penalty is folded into bounded_attribution (mirror, not an additend);
    fossilize_terminal_bonus is always 0.0 (dead field). Neither is a key — summing
    them would double-count."""
    assert "ratio_penalty" not in ADDITEND_SIGN_MAP
    assert "fossilize_terminal_bonus" not in ADDITEND_SIGN_MAP
    c = _components(bounded_attribution=2.0, ratio_penalty=-0.7, fossilize_terminal_bonus=0.0)
    decomposed = decompose_additends(2.0, c)
    assert "ratio_penalty" not in decomposed
    assert "fossilize_terminal_bonus" not in decomposed
    # ratio_penalty already lives inside bounded_attribution -> residual stays 0.
    assert decomposed["residual"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Keystone (§7): completeness against the REAL reward composition. The direct
# output of compute_contribution_reward carries ONLY the 11 internal additive
# terms (the terminal corrections — escrow_forfeit / hindsight_credit — are
# applied downstream in action_execution), so its residual must be EXACTLY 0.
# An unmapped ``reward +=`` term in contribution.py surfaces here as a non-zero
# residual, naming the gap. This is the test that makes ADDITEND_SIGN_MAP's
# completeness (not just its mechanics) load-bearing.
# ---------------------------------------------------------------------------

from esper.leyline import LifecycleOp, SeedStage  # noqa: E402
from esper.simic.rewards import (  # noqa: E402
    ContributionRewardConfig,
    RewardMode,
    compute_contribution_reward,
)
from esper.simic.rewards.types import SeedInfo  # noqa: E402


def _training_seed(**overrides: object) -> SeedInfo:
    base: dict[str, object] = dict(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=1.0,
        total_improvement=5.0,
        epochs_in_stage=3,
        seed_params=10_000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=2,
        seed_age_epochs=8,
        counterfactual_total_improvement=4.0,
    )
    base.update(overrides)
    return SeedInfo(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "label, reward_mode, action, seed_contribution, seed_info, extra",
    [
        (
            "shaped_wait_positive_contribution",
            RewardMode.SHAPED,
            LifecycleOp.WAIT,
            5.0,
            _training_seed(),
            {},
        ),
        (
            "shaped_holding_wait_warning",
            RewardMode.SHAPED,
            LifecycleOp.WAIT,
            5.0,
            _training_seed(stage=SeedStage.HOLDING.value, previous_stage=SeedStage.BLENDING.value),
            {},
        ),
        (
            "shaped_fossilize_terminal",
            RewardMode.SHAPED,
            LifecycleOp.FOSSILIZE,
            3.0,
            _training_seed(stage=SeedStage.HOLDING.value, previous_stage=SeedStage.BLENDING.value),
            {"num_fossilized_seeds": 1, "num_contributing_fossilized": 1},
        ),
        (
            "escrow_wait_positive",
            RewardMode.ESCROW,
            LifecycleOp.WAIT,
            5.0,
            _training_seed(),
            {"stable_val_acc": 55.0, "escrow_credit_prev": 0.5},
        ),
        (
            "shaped_capacity_economics_active_seeds",
            RewardMode.SHAPED,
            LifecycleOp.WAIT,
            2.0,
            _training_seed(),
            {"n_active_seeds": 4, "num_fossilized_seeds": 2},
        ),
    ],
)
def test_decompose_reconciles_real_contribution_reward_exactly(
    label: str,
    reward_mode: RewardMode,
    action: LifecycleOp,
    seed_contribution: float,
    seed_info: SeedInfo,
    extra: dict[str, object],
) -> None:
    """residual == 0 across the real reward composition: every ``reward +=`` term
    inside compute_contribution_reward is a named additend (completeness)."""
    # escrow_fossil_settlement is harmless for non-ESCROW modes and opts the ESCROW
    # parametrization past the fail-closed permanence guard (isolated additend test).
    config = ContributionRewardConfig(
        reward_mode=reward_mode, escrow_fossil_settlement="configured"
    )
    reward, components = compute_contribution_reward(
        action=action,
        seed_contribution=seed_contribution,
        val_acc=52.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=100,
        total_params=120_000,
        host_params=100_000,
        acc_at_germination=48.0,
        acc_delta=0.2,
        config=config,
        return_components=True,
        **extra,  # type: ignore[arg-type]
    )

    decomposed = decompose_additends(reward, components)
    assert decomposed["residual"] == pytest.approx(0.0, abs=1e-9), (
        f"[{label}] non-zero residual {decomposed['residual']:.6g} => an unmapped "
        f"reward-additive term in contribution.py; decomposition = {decomposed}"
    )
    assert sum(decomposed.values()) == pytest.approx(reward, abs=1e-9)


def test_return_components_does_not_change_reward_scalar_shaped() -> None:
    """Byte-identity foundation: forcing return_components (which the Stage-0 flag does on
    the control run) must NOT change the reward scalar — it only ALSO returns the telemetry
    struct. If this held only approximately, the OFF-leg byte-identity claim would be false.

    SHAPED is the only family where this is a real question: ESCROW *raises* without
    return_components (state update mandates it — so forcing it is a structural no-op), and
    non-CONTRIBUTION families (BASIC/LOSS) are rejected by the flag guard entirely.
    """
    kwargs = dict(
        action=LifecycleOp.WAIT,
        seed_contribution=5.0,
        val_acc=52.0,
        seed_info=_training_seed(),
        epoch=10,
        max_epochs=100,
        total_params=120_000,
        host_params=100_000,
        acc_at_germination=48.0,
        acc_delta=0.2,
        config=ContributionRewardConfig(reward_mode=RewardMode.SHAPED),
    )
    scalar = compute_contribution_reward(**kwargs, return_components=False)  # type: ignore[arg-type]
    with_components = compute_contribution_reward(**kwargs, return_components=True)  # type: ignore[arg-type]

    assert isinstance(scalar, float)
    # Exact bit-for-bit equality, not approx: the whole byte-identity argument rests on this.
    assert with_components[0] == scalar


def test_escrow_mandates_return_components() -> None:
    """Documents WHY the invariance test is SHAPED-only: ESCROW cannot run without
    return_components, so the Stage-0 flag forcing it there is a guaranteed no-op."""
    with pytest.raises(ValueError, match="ESCROW requires return_components"):
        compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,
            val_acc=52.0,
            seed_info=_training_seed(),
            epoch=10,
            max_epochs=100,
            config=ContributionRewardConfig(
                reward_mode=RewardMode.ESCROW, escrow_fossil_settlement="configured"
            ),
            stable_val_acc=55.0,  # satisfy the earlier ESCROW guard to reach the one under test
            escrow_credit_prev=0.5,
            return_components=False,
        )
