"""Tests for the value-free per-RETURN reward-variance decomposition (EV-stab Stage-0).

``compute_return_variance_shares`` accumulates each per-step signed reward component
into a discounted return-to-go (value-free, lambda=1 within segments, raw scale), then
returns the per-term return-variance share ``share_i = Cov(R_i, R) / Var(R)``. By
covariance linearity ``sum_i Cov(R_i, R) = Var(R)``, so the shares sum to 1 exactly
whenever the components sum to the total reward per step.
"""

import pytest

from esper.simic.telemetry.reward_variance import compute_return_variance_shares


def test_return_variance_shares_match_hand_computed_covariance() -> None:
    # Single 3-step segment, gamma=0.5, two components summing to reward=[1,0,1].
    # Value-free discounted return-to-go:
    #   R_total = [1.25, 0.5, 1.0], R_a = [1.0, 0.0, 0.0], R_b = [0.25, 0.5, 1.0]
    #   Var(R_total) = 7/24 ; Cov(R_a, R_total) = 8/24 -> share_a = 8/7
    #   share_b = 1 - 8/7 = -1/7  (correlated components -> shares may leave [0, 1])
    report = compute_return_variance_shares(
        component_rewards={"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]},
        dones=[False, False, True],
        gamma=0.5,
    )

    assert report["shares"]["a"] == pytest.approx(8.0 / 7.0, rel=1e-9)
    assert report["shares"]["b"] == pytest.approx(-1.0 / 7.0, rel=1e-9)
    assert report["shares_sum"] == pytest.approx(1.0, abs=1e-9)


def test_gate_outputs_attribution_and_main_variance_share() -> None:
    # Same arithmetic, relabelled so the cf stream is `bounded_attribution`:
    #   R_cf = [0.25, 0.5, 1.0], R_main = R_total - R_cf = [1.0, 0.0, 0.0]
    #   Var(R_total) = 7/72 ; Var(R_main) = 2/9 = 16/72
    #   share_attribution = Cov(R_cf, R)/Var(R) = -1/7  (the >0.40 gate read)
    #   residual_share    = Cov(R_residual, R)/Var(R) = 8/7  (the ~0 reconciliation term)
    #   r_main_var_share  = Var(R_main)/Var(R) = (16/72)/(7/72) = 16/7  (the smoothness leg)
    # r_main_var_share is a DISTINCT quantity from any covariance share (it is NOT bounded
    # by [0, 1] — here R_main is anti-correlated with R_cf so it carries MORE return
    # variance than the total): the gate's "is R_main the smoother stream" leg reads this
    # variance-share, not the fragile CoV.
    report = compute_return_variance_shares(
        component_rewards={"bounded_attribution": [0.0, 0.0, 1.0], "residual": [1.0, 0.0, 0.0]},
        dones=[False, False, True],
        gamma=0.5,
    )

    assert report["share_attribution"] == pytest.approx(-1.0 / 7.0, rel=1e-9)
    assert report["residual_share"] == pytest.approx(8.0 / 7.0, rel=1e-9)
    assert report["r_main_var_share"] == pytest.approx(16.0 / 7.0, rel=1e-9)


def test_degenerate_zero_variance_batch_returns_finite_zero_shares() -> None:
    # Two single-step episodes with identical returns -> Var(R_total) = 0. A real PPO
    # update can hit this; the decomposition must return finite 0.0 shares, never nan.
    import math

    report = compute_return_variance_shares(
        component_rewards={"bounded_attribution": [1.0, 1.0], "residual": [0.0, 0.0]},
        dones=[True, True],
        gamma=0.5,
    )

    assert report["shares"]["bounded_attribution"] == 0.0
    assert report["share_attribution"] == 0.0
    assert report["r_main_var_share"] == 0.0
    assert all(math.isfinite(v) for v in report["shares"].values())


def test_return_to_go_resets_at_episode_boundaries() -> None:
    # Two 2-step episodes (done at steps 1 and 3). If the reset were ignored (one
    # 4-step segment), the returns and therefore share_a would differ. With the reset:
    #   R_a = [1, 0, 1, 0], R_b = [0.5, 1, 0.5, 1], R_total = [1.5, 1, 1.5, 1]
    #   share_a = Cov(R_a, R)/Var(R) = 0.125 / 0.0625 = 2.0
    report = compute_return_variance_shares(
        component_rewards={"a": [1.0, 0.0, 1.0, 0.0], "b": [0.0, 1.0, 0.0, 1.0]},
        dones=[False, True, False, True],
        gamma=0.5,
    )

    assert report["shares"]["a"] == pytest.approx(2.0, rel=1e-9)
    assert report["shares_sum"] == pytest.approx(1.0, abs=1e-9)
