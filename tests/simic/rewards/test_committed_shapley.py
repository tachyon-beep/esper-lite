"""Committed-Shapley top-up math (WI-1): exact 2^k coalition Shapley + guards.

All expectations hand-computed. Units are percentage points end-to-end.
Design: docs/plans/ready/2026-07-02-committed-shapley-topup-build.md §2.
"""

from __future__ import annotations

import pytest

from esper.simic.rewards.committed_shapley import (
    CommittedShapleyResult,
    compute_committed_shapley_topup,
)


def _accs_2slot(v0=50.0, va=60.0, vb=55.0, vab=70.0):
    return {
        frozenset(): v0,
        frozenset({"a"}): va,
        frozenset({"b"}): vb,
        frozenset({"a", "b"}): vab,
    }


def test_two_slot_hand_computed_case():
    # phi(a) = 1/2[(60-50) + (70-55)] = 12.5 ; phi(b) = 1/2[(55-50) + (70-60)] = 7.5
    # c_paid(a) = 10, c_paid(b) = 5 ; tau=0 -> gap = 2.5 each
    # scale=1, cap=10 -> raw = 2.5 each ; G = 20 ; sum_raw=5 <= G -> top_up = raw
    res = compute_committed_shapley_topup(
        _accs_2slot(), ("a", "b"), scale=1.0, cap=10.0, tau=0.0
    )
    assert isinstance(res, CommittedShapleyResult)
    assert res.k == 2
    assert res.per_slot["a"].phi == pytest.approx(12.5)
    assert res.per_slot["b"].phi == pytest.approx(7.5)
    assert res.per_slot["a"].c_paid == pytest.approx(10.0)
    assert res.per_slot["b"].c_paid == pytest.approx(5.0)
    assert res.per_slot["a"].gap == pytest.approx(2.5)
    assert res.per_slot["b"].gap == pytest.approx(2.5)
    assert res.per_slot["a"].top_up == pytest.approx(2.5)
    assert res.per_slot["b"].top_up == pytest.approx(2.5)
    assert res.g == pytest.approx(20.0)
    assert res.clamp_binding is False


def test_three_slot_hand_computed_and_efficiency():
    accs = {
        frozenset(): 50.0,
        frozenset({"a"}): 60.0,
        frozenset({"b"}): 55.0,
        frozenset({"c"}): 50.0,
        frozenset({"a", "b"}): 70.0,
        frozenset({"a", "c"}): 62.0,
        frozenset({"b", "c"}): 56.0,
        frozenset({"a", "b", "c"}): 75.0,
    }
    res = compute_committed_shapley_topup(
        accs, ("a", "b", "c"), scale=1.0, cap=100.0, tau=0.0
    )
    # Exact fractions: phi(a)=85/6, phi(b)=52/6, phi(c)=13/6.
    assert res.per_slot["a"].phi == pytest.approx(85.0 / 6.0)
    assert res.per_slot["b"].phi == pytest.approx(52.0 / 6.0)
    assert res.per_slot["c"].phi == pytest.approx(13.0 / 6.0)
    # Shapley efficiency: sum(phi) == v(C) - v(empty) == 25.
    total_phi = sum(s.phi for s in res.per_slot.values())
    assert total_phi == pytest.approx(25.0)


def test_null_player_gets_exactly_zero():
    # b contributes nothing to ANY coalition: v({b})=v(0), v({a,b})=v({a}).
    accs = _accs_2slot(v0=50.0, va=60.0, vb=50.0, vab=60.0)
    res = compute_committed_shapley_topup(
        accs, ("a", "b"), scale=1.0, cap=10.0, tau=0.0
    )
    assert res.per_slot["b"].phi == 0.0  # exact, not approx (identical leave-outs)
    assert res.per_slot["b"].top_up == 0.0


def test_tau_deadband_subtracts_before_gap():
    res = compute_committed_shapley_topup(
        _accs_2slot(), ("a", "b"), scale=1.0, cap=10.0, tau=2.0
    )
    assert res.per_slot["a"].gap == pytest.approx(0.5)  # 12.5 - 10 - 2
    res_all = compute_committed_shapley_topup(
        _accs_2slot(), ("a", "b"), scale=1.0, cap=10.0, tau=5.0
    )
    assert res_all.per_slot["a"].gap == 0.0  # max(0, ...) floors at zero
    assert res_all.per_slot["a"].top_up == 0.0


def test_per_seed_cap_bounds_raw():
    res = compute_committed_shapley_topup(
        _accs_2slot(), ("a", "b"), scale=4.0, cap=1.0, tau=0.0
    )
    # scale*gap = 10 each, capped to 1.0.
    assert res.per_slot["a"].raw == pytest.approx(1.0)
    assert res.per_slot["b"].raw == pytest.approx(1.0)


def test_g_clamp_binds_and_conserves_g():
    # scale=10 -> raw = 25 each, sum_raw = 50 > G = 20 -> factor 0.4.
    res = compute_committed_shapley_topup(
        _accs_2slot(), ("a", "b"), scale=10.0, cap=100.0, tau=0.0
    )
    assert res.clamp_binding is True
    assert res.per_slot["a"].top_up == pytest.approx(10.0)
    assert res.per_slot["b"].top_up == pytest.approx(10.0)
    total = sum(s.top_up for s in res.per_slot.values())
    assert total == pytest.approx(res.g)


def test_negative_system_gain_zeroes_all_top_ups():
    # Committed set makes things WORSE: v(C) < v(empty) -> G = 0 -> factor 0.
    accs = _accs_2slot(v0=50.0, va=60.0, vb=55.0, vab=45.0)
    res = compute_committed_shapley_topup(
        accs, ("a", "b"), scale=1.0, cap=10.0, tau=0.0
    )
    assert res.g == 0.0
    assert all(s.top_up == 0.0 for s in res.per_slot.values())


def test_zero_sum_raw_yields_zero_factor_not_nan():
    # All gaps zero (phi == c_paid at k=1-like symmetry) -> sum_raw == 0.
    accs = _accs_2slot(v0=50.0, va=60.0, vb=50.0, vab=60.0)  # additive, no synergy
    res = compute_committed_shapley_topup(
        accs, ("a", "b"), scale=1.0, cap=10.0, tau=10.0
    )
    assert all(s.top_up == 0.0 for s in res.per_slot.values())
    assert res.clamp_binding is False


def test_k_zero_returns_empty_result():
    res = compute_committed_shapley_topup(
        {frozenset(): 50.0}, (), scale=1.0, cap=10.0, tau=0.0
    )
    assert res.k == 0
    assert res.per_slot == {}
    assert res.g == 0.0


def test_k_one_top_up_is_structurally_zero():
    # k=1: phi == v({s}) - v(empty) == c_paid exactly -> gap 0 -> top_up 0.
    accs = {frozenset(): 50.0, frozenset({"a"}): 62.0}
    res = compute_committed_shapley_topup(accs, ("a",), scale=1.0, cap=10.0, tau=0.0)
    assert res.per_slot["a"].phi == pytest.approx(12.0)
    assert res.per_slot["a"].c_paid == pytest.approx(12.0)
    assert res.per_slot["a"].top_up == 0.0


def test_k_above_three_refused():
    slots = ("a", "b", "c", "d")
    accs = {frozenset(): 50.0}
    with pytest.raises(ValueError, match="exact"):
        compute_committed_shapley_topup(accs, slots, scale=1.0, cap=10.0, tau=0.0)


def test_missing_coalition_fails_loud():
    accs = _accs_2slot()
    del accs[frozenset({"b"})]
    with pytest.raises(ValueError, match="coalition"):
        compute_committed_shapley_topup(accs, ("a", "b"), scale=1.0, cap=10.0, tau=0.0)


def test_negative_parameters_refused():
    for kwargs in (
        {"scale": -1.0, "cap": 10.0, "tau": 0.0},
        {"scale": 1.0, "cap": -1.0, "tau": 0.0},
        {"scale": 1.0, "cap": 10.0, "tau": -0.5},
    ):
        with pytest.raises(ValueError):
            compute_committed_shapley_topup(_accs_2slot(), ("a", "b"), **kwargs)
