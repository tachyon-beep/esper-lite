"""EV-stab Stage 2: per-stream GAE in the rollout buffer (§4.4/§5).

The decomposition `returns_main := returns_total - returns_cf` is exact-by-definition
(tautological). The MEANINGFUL test (the one the review panel insisted on) verifies the
linearity claim itself: each per-stream return must equal an INDEPENDENT single-stream
GAE on that stream alone, using the buffer's own proven OFF-leg path as the reference.
"""

from __future__ import annotations

import torch

from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
from esper.simic.control import ValueNormalizer

GAMMA = 0.99
LAM = 0.95


def _buf(n, values, rewards, bootstrap, dones, trunc, *, cf_values=None, cf_bootstrap=None, r_cf=None):
    b = TamiyoRolloutBuffer(num_envs=1, max_steps_per_env=n + 2, state_dim=8)
    b.step_counts[0] = n
    b.values[0, :n] = torch.tensor(values)
    b.rewards[0, :n] = torch.tensor(rewards)
    b.bootstrap_values[0, :n] = torch.tensor(bootstrap)
    b.dones[0, :n] = torch.tensor(dones, dtype=torch.bool)
    b.truncated[0, :n] = torch.tensor(trunc, dtype=torch.bool)
    if cf_values is not None:
        b.cf_values[0, :n] = torch.tensor(cf_values)
        b.cf_bootstrap_values[0, :n] = torch.tensor(cf_bootstrap)
        b.r_cf_norm[0, :n] = torch.tensor(r_cf)
    return b


def test_per_stream_returns_match_independent_single_stream_gae():
    n = 5
    v_main = [0.5, -0.2, 1.0, 0.3, -0.1]
    v_cf = [0.1, 0.4, -0.3, 0.2, 0.0]
    r_total = [1.0, 0.5, -0.5, 0.2, 0.7]
    r_cf = [0.3, -0.1, 0.2, 0.05, 0.4]
    boot_main = [0.0, 0.0, 0.0, 0.0, 0.9]
    boot_cf = [0.0, 0.0, 0.0, 0.0, 0.2]
    dones = [False] * n
    trunc = [False, False, False, False, True]  # truncated last -> bootstrap engaged
    r_main = [t - c for t, c in zip(r_total, r_cf)]

    on = _buf(n, v_main, r_total, boot_main, dones, trunc,
              cf_values=v_cf, cf_bootstrap=boot_cf, r_cf=r_cf)
    on.compute_advantages_and_returns(
        gamma=GAMMA, gae_lambda=LAM, value_normalizer=None, cf_value_normalizer=ValueNormalizer()
    )

    # Independent single-stream references via the buffer's own OFF-leg path.
    main_ref = _buf(n, v_main, r_main, boot_main, dones, trunc)
    main_ref.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM, value_normalizer=None)
    cf_ref = _buf(n, v_cf, r_cf, boot_cf, dones, trunc)
    cf_ref.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM, value_normalizer=None)

    # Non-tautological: per-stream returns == the independent single-stream GAEs.
    assert torch.allclose(on.returns_main[0, :n], main_ref.returns[0, :n], atol=1e-5)
    assert torch.allclose(on.returns_cf[0, :n], cf_ref.returns[0, :n], atol=1e-5)
    # Total return is the sum of the two independent streams.
    assert torch.allclose(
        on.returns[0, :n], main_ref.returns[0, :n] + cf_ref.returns[0, :n], atol=1e-5
    )
    # Buffer-level identity.
    assert torch.allclose(
        on.returns[0, :n], on.returns_main[0, :n] + on.returns_cf[0, :n], atol=1e-6
    )


def test_off_leg_is_byte_identical_and_leaves_cf_returns_zero():
    n = 4
    v = [0.5, -0.2, 1.0, 0.3]
    r = [1.0, 0.5, -0.5, 0.2]
    boot = [0.0, 0.0, 0.0, 0.7]
    dones = [False, False, False, False]
    trunc = [False, False, False, True]

    off = _buf(n, v, r, boot, dones, trunc)
    off.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM, value_normalizer=None)

    baseline = _buf(n, v, r, boot, dones, trunc)
    baseline.compute_advantages_and_returns(gamma=GAMMA, gae_lambda=LAM, value_normalizer=None)

    # OFF leg (cf_value_normalizer=None) is byte-identical to the single-head path...
    assert torch.equal(off.returns[0, :n], baseline.returns[0, :n])
    assert torch.equal(off.advantages[0, :n], baseline.advantages[0, :n])
    # ...and the per-stream arrays are untouched (zero).
    assert torch.all(off.returns_main[0, :n] == 0.0)
    assert torch.all(off.returns_cf[0, :n] == 0.0)
