"""Phase 0 (Stage-0 instrumentation): the per-term SIGNED additend decomposition.

``decompose_additends(reward_raw, components)`` returns each SHAPED/ESCROW reward
term as its *signed contribution to ``reward_raw``* plus a ``residual`` key, such
that ``sum(values()) == reward_raw`` exhaustively by construction. This is the
foundation the per-term variance-share decomposition reconciles over (GATE 0:
``residual`` ~ 0, attributable to the telemetry-less ``pending_auto_prune_penalty``).

The signed map is reviewer-mandated (Phase 0 evidence packet §4): component fields
are stored in MIXED sign conventions — ``compute_rent``/``escrow_forfeit`` are stored
already-negative, while ``occupancy_rent``/``fossilized_rent`` are positive magnitudes
the reward SUBTRACTS — so naïve summation of the raw fields flips two signs.
``ratio_penalty`` is folded into ``bounded_attribution`` (NOT a separate additend) and
``fossilize_terminal_bonus`` is always 0.0 (a dead field); both are excluded.
"""

from __future__ import annotations

import pytest

from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.rewards.partition import ADDITEND_SIGN_MAP, decompose_additends


def _components(**overrides: float | None) -> RewardComponentsTelemetry:
    return RewardComponentsTelemetry(**overrides)


def test_decompose_reconciles_to_reward_raw_exactly() -> None:
    """sum(signed additends + residual) == reward_raw, and residual == 0 when the
    named terms fully account for the reward."""
    c = _components(
        bounded_attribution=5.0,
        blending_warning=-0.2,
        holding_warning=-0.1,
        pbrs_bonus=1.5,
        interaction_bonus=0.05,
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
    """The telemetry-less pending_auto_prune_penalty (no component field) is captured
    by residual = reward_raw - sum(named signed)."""
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
