"""Regression tests for the 2nd-order interaction index (esper-lite-90bc132573).

The reward-bearing synergy bonus and scaffold hindsight credit are driven by the
interaction index I_ij = f({i,j}) - f({i}) - f({j}) + f(empty). All four terms must
share single-coalition-ON semantics: the solo terms must be f({i}) ("only this slot
enabled" = solo_on_accs), NOT leave-one-out f(N\\{i}) ("everyone but i" = baseline_accs).
The two diverge whenever seeds interact, so mixing them corrupts the synergy signal.
"""
import pytest

from esper.simic.training.vectorized_trainer import _pair_interaction_index


def test_interaction_index_zero_for_additive_seeds():
    """Non-interacting (additive) seeds yield interaction == 0.

    Additive means f({i,j}) - f(empty) == (f({i}) - f(empty)) + (f({j}) - f(empty)).
    With host f(empty)=0.50 and each seed adding +0.10 alone, together exactly +0.20.
    """
    all_off = 0.50
    solo_on_a = 0.60  # +0.10 marginal
    solo_on_b = 0.60  # +0.10 marginal
    pair = 0.70  # +0.20 == sum of marginals -> no interaction
    assert _pair_interaction_index(pair, solo_on_a, solo_on_b, all_off) == pytest.approx(0.0)


def test_interaction_index_positive_for_synergy():
    """Synergy: the coalition beats the sum of marginals -> positive interaction."""
    all_off = 0.50
    solo_on_a = 0.60  # +0.10
    solo_on_b = 0.60  # +0.10
    pair = 0.85  # +0.35 > 0.20 -> synergy of +0.15
    assert _pair_interaction_index(pair, solo_on_a, solo_on_b, all_off) == pytest.approx(0.15)


def test_interaction_index_negative_for_antagonism():
    """Antagonism: the coalition underperforms the sum of marginals -> negative."""
    all_off = 0.50
    solo_on_a = 0.60
    solo_on_b = 0.60
    pair = 0.62  # +0.12 < 0.20 -> antagonism of -0.08
    assert _pair_interaction_index(pair, solo_on_a, solo_on_b, all_off) == pytest.approx(-0.08)


def test_solo_on_vs_leave_one_out_diverge_under_interaction():
    """Pin WHY the solo source matters.

    With a genuine interaction, plugging leave-one-out accuracies (f(N\\{i}),
    other seeds still ON) into the solo slots yields a DIFFERENT (and meaningless)
    index than the correct solo-ON values (f({i}), only that slot ON). This is the
    exact mistake the fix corrects: the reward path previously read baseline_accs
    (leave-one-out) for the solo terms.
    """
    all_off = 0.50
    pair = 0.85
    # Correct: single-seed-ON marginals.
    correct = _pair_interaction_index(pair, 0.60, 0.60, all_off)
    # Wrong: leave-one-out accuracies are inflated by the other (still-ON) seeds.
    wrong = _pair_interaction_index(pair, 0.80, 0.82, all_off)
    assert correct == pytest.approx(0.15)
    assert wrong != pytest.approx(correct)
