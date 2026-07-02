"""Committed-Shapley build (WI-10): the MANDATED GAE retro-write unit test.

GATE 2 (docs/analysis/2026-07-02-gate2-learnability-result.md) mandated that the
terminal credit be delivered by writing into ``buffer.rewards[env, t_f]`` BEFORE
GAE runs. That delivery relies on GAE being linear in rewards: a reward delta
``d`` at step ``t_f`` must move advantages by exactly

    dA(t) = d * (gamma*lambda)^(t_f - t)   for t <= t_f (no true-done in between)
    dA(t) = 0                              for t >  t_f

These tests pin that property against hand-computed advantages, plus the
credit-assignment edge matrix from the drl-expert plan review (F-B): true-done
between t and t_f stops propagation; t_f at 0 and at the last step; truncated
vs true-done terminals; episodes shorter than the buffer width; and the GATE 2
terminal-flush decay identity (gamma*lambda)^(T-1-t_f).
"""

from __future__ import annotations

import torch

from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer

GAMMA = 0.9
LAM = 0.5
GL = GAMMA * LAM


def _buf(n, values, rewards, bootstrap, dones, trunc, *, width_pad=2):
    b = TamiyoRolloutBuffer(num_envs=1, max_steps_per_env=n + width_pad, state_dim=8)
    b.step_counts[0] = n
    b.values[0, :n] = torch.tensor(values)
    b.rewards[0, :n] = torch.tensor(rewards)
    b.bootstrap_values[0, :n] = torch.tensor(bootstrap)
    b.dones[0, :n] = torch.tensor(dones, dtype=torch.bool)
    b.truncated[0, :n] = torch.tensor(trunc, dtype=torch.bool)
    return b


def _paired_delta(n, t_f, delta, values, rewards, bootstrap, dones, trunc):
    """Run GAE on a control buffer and a retro-credited twin; return dA, dRet."""
    control = _buf(n, values, rewards, bootstrap, dones, trunc)
    credited = _buf(n, values, rewards, bootstrap, dones, trunc)
    credited.rewards[0, t_f] += delta

    control.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM)
    credited.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM)

    d_adv = credited.advantages[0, :n] - control.advantages[0, :n]
    d_ret = credited.returns[0, :n] - control.returns[0, :n]
    return d_adv, d_ret


# Shared fixture values: nothing degenerate, terminal done at the last step
# (the production shape: one episode per env, true terminal at T-1).
N = 6
VALUES = [0.5, -0.25, 1.0, 0.375, -0.125, 0.75]
REWARDS = [1.0, 0.5, -0.5, 0.25, 0.625, -0.75]
BOOTSTRAP = [0.0] * N
DONES = [False] * (N - 1) + [True]
TRUNC = [False] * N
DELTA = 2.0


def test_retro_write_moves_advantage_at_t_f_exactly():
    t_f = 3
    d_adv, d_ret = _paired_delta(N, t_f, DELTA, VALUES, REWARDS, BOOTSTRAP, DONES, TRUNC)

    # The decision step itself moves 1:1 with the credit.
    assert torch.allclose(d_adv[t_f], torch.tensor(DELTA), atol=1e-6)
    # Earlier steps decay by (gamma*lambda)^(t_f - t).
    for t in range(t_f):
        expected = DELTA * GL ** (t_f - t)
        assert torch.allclose(d_adv[t], torch.tensor(expected), atol=1e-6), t
    # Later steps are bitwise untouched (backward recursion never sees t_f).
    assert torch.all(d_adv[t_f + 1 :] == 0.0)
    # Values are unchanged, so returns move exactly with advantages.
    assert torch.allclose(d_ret, d_adv, atol=1e-7)


def test_full_gae_matches_hand_computed_advantages_with_credit():
    """The literal mandate: credited-buffer GAE vs a hand-computed expectation."""
    n, t_f, delta = 4, 1, 2.0
    values = [0.5, -0.2, 1.0, 0.3]
    rewards = [1.0, 0.5, -0.5, 0.2]
    dones = [False, False, False, True]

    credited = _buf(n, values, rewards, [0.0] * n, dones, [False] * n)
    credited.rewards[0, t_f] += delta
    credited.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM)

    # Hand computation on the credited reward vector r = [1.0, 2.5, -0.5, 0.2]:
    # t=3 (terminal): delta3 = 0.2 + 0 - 0.3                      = -0.1 ; A3 = -0.1
    # t=2: delta2 = -0.5 + 0.9*0.3 - 1.0 = -1.23 ; A2 = -1.23 + 0.45*(-0.1) = -1.275
    # t=1: delta1 =  2.5 + 0.9*1.0 - (-0.2) = 3.6 ; A1 = 3.6 + 0.45*(-1.275) = 3.026250
    # t=0: delta0 =  1.0 + 0.9*(-0.2) - 0.5 = 0.32 ; A0 = 0.32 + 0.45*3.026250 = 1.681813
    expected = torch.tensor([1.6818125, 3.02625, -1.275, -0.1])
    assert torch.allclose(credited.advantages[0, :n], expected, atol=1e-5)
    assert torch.allclose(
        credited.returns[0, :n], expected + torch.tensor(values), atol=1e-5
    )


def test_propagation_stops_at_true_done_between_t_and_t_f():
    """drl F-B: a true terminal STRICTLY between t and t_f must stop the credit."""
    t_f, t_d = 4, 2  # done at step 2, credit at step 4
    dones = [False, False, True, False, False, True]
    d_adv, _ = _paired_delta(N, t_f, DELTA, VALUES, REWARDS, BOOTSTRAP, dones, TRUNC)

    # At and before the intervening terminal: zero credit leakage across episodes.
    assert torch.all(d_adv[: t_d + 1] == 0.0)
    # Between the terminal and t_f the decay identity still holds.
    assert torch.allclose(d_adv[t_f], torch.tensor(DELTA), atol=1e-6)
    assert torch.allclose(d_adv[t_f - 1], torch.tensor(DELTA * GL), atol=1e-6)


def test_t_f_at_step_zero():
    d_adv, _ = _paired_delta(N, 0, DELTA, VALUES, REWARDS, BOOTSTRAP, DONES, TRUNC)
    assert torch.allclose(d_adv[0], torch.tensor(DELTA), atol=1e-6)
    assert torch.all(d_adv[1:] == 0.0)


def test_t_f_at_last_step_done_and_truncated_agree():
    """drl F-B: credit at the final step behaves identically under true-done vs
    truncation (the credit enters via rewards[t_f]; terminal type only affects
    the bootstrap, which is identical between the paired buffers)."""
    t_f = N - 1
    trunc_last = [False] * (N - 1) + [True]
    bootstrap_trunc = [0.0] * (N - 1) + [0.7]

    d_done, _ = _paired_delta(N, t_f, DELTA, VALUES, REWARDS, BOOTSTRAP, DONES, TRUNC)
    d_trunc, _ = _paired_delta(
        N, t_f, DELTA, VALUES, REWARDS, bootstrap_trunc, [False] * N, trunc_last
    )

    for d_adv in (d_done, d_trunc):
        assert torch.allclose(d_adv[t_f], torch.tensor(DELTA), atol=1e-6)
        for t in range(t_f):
            expected = DELTA * GL ** (t_f - t)
            assert torch.allclose(d_adv[t], torch.tensor(expected), atol=1e-6), t
    assert torch.allclose(d_done, d_trunc, atol=1e-6)


def test_short_episode_leaves_padding_untouched():
    """drl F-B: step_counts < max_steps_per_env — GAE reads only the valid
    slice; the padded tail stays zero with or without the credit."""
    n, t_f = 3, 1
    credited = _buf(
        n, VALUES[:n], REWARDS[:n], [0.0] * n, [False, False, True], [False] * n,
        width_pad=4,
    )
    credited.rewards[0, t_f] += DELTA
    credited.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM)

    assert torch.all(credited.advantages[0, n:] == 0.0)
    assert torch.all(credited.returns[0, n:] == 0.0)
    # And the in-slice identity still holds vs a control.
    control = _buf(
        n, VALUES[:n], REWARDS[:n], [0.0] * n, [False, False, True], [False] * n,
        width_pad=4,
    )
    control.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM)
    d_adv = credited.advantages[0, :n] - control.advantages[0, :n]
    assert torch.allclose(d_adv[t_f], torch.tensor(DELTA), atol=1e-6)


def test_terminal_flush_decay_identity_reproduced():
    """GATE 2's analytic kill of terminal-flush: a credit at the LAST step
    reaches an earlier decision step t_f only as delta*(gamma*lambda)^(T-1-t_f).
    (This is why retro-write is mandated and terminal-flush is dead.)"""
    t_last = N - 1
    t_decision = 1
    d_adv, _ = _paired_delta(
        N, t_last, DELTA, VALUES, REWARDS, BOOTSTRAP, DONES, TRUNC
    )
    expected = DELTA * GL ** (t_last - t_decision)
    assert torch.allclose(d_adv[t_decision], torch.tensor(expected), atol=1e-6)
    # Retro-write at the decision step delivers the full credit instead.
    d_retro, _ = _paired_delta(
        N, t_decision, DELTA, VALUES, REWARDS, BOOTSTRAP, DONES, TRUNC
    )
    assert torch.allclose(d_retro[t_decision], torch.tensor(DELTA), atol=1e-6)
