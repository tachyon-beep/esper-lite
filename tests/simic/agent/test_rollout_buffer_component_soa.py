"""EV-stab Stage 0: per-component signed-additend SoA in the rollout buffer.

The buffer collects, per step, the signed additend decomposition
(``decompose_additends``) so ``compute_return_variance_shares`` can read the
value-free Stage-0 gate off a finished rollout. This is pure telemetry — the
SoA is never read by the loss/GAE/value path — so the OFF (unset) path must
leave it as untouched zeros.

The read accessor pools per-step components ACROSS envs, forcing an
episode-boundary ``done`` at each env's last stored step: the return-to-go in
``compute_return_variance_shares`` resets only on ``done``, so without the
boundary reset env N+1's early rewards would bleed backwards into env N's tail
(silent wrong covariance).
"""

from __future__ import annotations

import torch

from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
from esper.simic.rewards.partition import (
    ADDITEND_SIGN_MAP,
    COMPONENT_TERMS,
    RESIDUAL_KEY,
)
from esper.simic.telemetry.reward_variance import (
    compute_return_variance_metrics,
    compute_return_variance_shares,
)


def test_component_terms_ordering() -> None:
    """COMPONENT_TERMS = the signed-additend keys then residual, in a fixed order."""
    assert COMPONENT_TERMS == (*ADDITEND_SIGN_MAP.keys(), RESIDUAL_KEY)
    assert COMPONENT_TERMS[0] == "bounded_attribution"
    assert COMPONENT_TERMS[-1] == RESIDUAL_KEY
    assert len(COMPONENT_TERMS) == len(ADDITEND_SIGN_MAP) + 1


def _buf(num_envs: int = 1, steps: int = 6) -> TamiyoRolloutBuffer:
    # Tiny LSTM dims so the [1, 1, 2] hidden below matches [lstm_layers, 1, hidden_dim].
    return TamiyoRolloutBuffer(
        num_envs=num_envs,
        max_steps_per_env=steps + 2,
        state_dim=8,
        lstm_layers=1,
        lstm_hidden_dim=2,
    )


def test_add_writes_component_additends_in_soa() -> None:
    """add(component_additends=...) stores each term in COMPONENT_TERMS order."""
    b = _buf()
    additends = {t: float(i) for i, t in enumerate(COMPONENT_TERMS)}
    _add_wait_step(b, env_id=0, reward=sum(additends.values()), component_additends=additends)

    row = b.component_additends[0, 0]
    expected = torch.tensor([additends[t] for t in COMPONENT_TERMS])
    assert torch.allclose(row, expected)


def test_add_without_component_additends_leaves_zeros() -> None:
    """OFF path (no component_additends): the SoA row stays zeros (byte-identical)."""
    b = _buf()
    _add_wait_step(b, env_id=0, reward=1.0, component_additends=None)
    assert torch.count_nonzero(b.component_additends[0, 0]) == 0


def test_collect_component_rewards_pools_with_env_boundary_reset() -> None:
    """The accessor pools per-step components across envs and forces a done at each
    env's last stored step, so return-to-go cannot bleed across the env boundary."""
    b = _buf(num_envs=2)
    # env 0: two steps, neither terminal in-episode.
    _add_wait_step(b, 0, reward=1.0, component_additends=_terms(bounded_attribution=1.0), done=False)
    _add_wait_step(b, 0, reward=2.0, component_additends=_terms(bounded_attribution=2.0), done=False)
    # env 1: two steps, last one a real episode end.
    _add_wait_step(b, 1, reward=3.0, component_additends=_terms(bounded_attribution=3.0), done=False)
    _add_wait_step(b, 1, reward=4.0, component_additends=_terms(bounded_attribution=4.0), done=True)

    component_rewards, dones = b.collect_component_rewards()

    # Pooled in env-major order: env0(2) then env1(2).
    assert component_rewards["bounded_attribution"] == [1.0, 2.0, 3.0, 4.0]
    # env0's last step (index 1) forced True despite done=False; env1's already True.
    assert dones == [False, True, False, True]


def test_compute_return_variance_metrics_reads_gate_from_buffer() -> None:
    """compute_return_variance_metrics reads the value-free gate off a finished buffer,
    matching compute_return_variance_shares on the pooled component data."""
    b = _buf(num_envs=1, steps=4)
    for r_cf in (1.0, 0.0, 0.0):
        _add_wait_step(
            b, 0, reward=r_cf, component_additends=_terms(bounded_attribution=r_cf), done=False
        )

    metrics = compute_return_variance_metrics(b, gamma=0.5)
    assert set(metrics) == {
        "return_var_cf_share",
        "return_var_main_share",
        "return_var_residual_share",
    }

    # Cross-check: identical to reading the shares directly off the pooled data.
    component_rewards, dones = b.collect_component_rewards()
    ref = compute_return_variance_shares(component_rewards, dones, 0.5)
    assert metrics["return_var_cf_share"] == ref["share_attribution"]
    assert metrics["return_var_main_share"] == ref["r_main_var_share"]
    assert metrics["return_var_residual_share"] == ref["residual_share"]


def test_compute_return_variance_metrics_empty_buffer_returns_empty() -> None:
    """No collected steps -> empty dict (compute_return_variance_shares raises on an empty
    batch; a flag-on update with no CONTRIBUTION steps must not crash)."""
    assert compute_return_variance_metrics(_buf(), gamma=0.5) == {}


# --------------------------------------------------------------------------- helpers


def _terms(**overrides: float) -> dict[str, float]:
    d = dict.fromkeys(COMPONENT_TERMS, 0.0)
    d.update(overrides)
    return d


def _add_wait_step(
    b: TamiyoRolloutBuffer,
    env_id: int,
    *,
    reward: float,
    component_additends: dict[str, float] | None,
    done: bool = False,
) -> None:
    """Minimal buffer.add for a WAIT-like transition (only the fields we assert on)."""
    num_slots = b.blueprint_indices.shape[2]
    zeros_mask = torch.ones(1, dtype=torch.bool)
    b.add(
        env_id=env_id,
        state=torch.zeros(8),
        blueprint_indices=torch.zeros(num_slots, dtype=torch.long),
        slot_action=0,
        blueprint_action=0,
        style_action=0,
        tempo_action=0,
        alpha_target_action=0,
        alpha_speed_action=0,
        alpha_curve_action=0,
        op_action=0,
        effective_op_action=0,
        slot_log_prob=0.0,
        blueprint_log_prob=0.0,
        style_log_prob=0.0,
        tempo_log_prob=0.0,
        alpha_target_log_prob=0.0,
        alpha_speed_log_prob=0.0,
        alpha_curve_log_prob=0.0,
        op_log_prob=0.0,
        value=0.0,
        reward=reward,
        done=done,
        slot_mask=zeros_mask,
        blueprint_mask=zeros_mask,
        style_mask=zeros_mask,
        tempo_mask=zeros_mask,
        alpha_target_mask=zeros_mask,
        alpha_speed_mask=zeros_mask,
        alpha_curve_mask=zeros_mask,
        op_mask=zeros_mask,
        hidden_h=torch.zeros(1, 1, 2),
        hidden_c=torch.zeros(1, 1, 2),
        component_additends=component_additends,
    )
