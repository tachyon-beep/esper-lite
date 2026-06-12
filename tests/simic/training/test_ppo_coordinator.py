"""Tests for PPO update coordination edge cases."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

from esper.simic.training.ppo_coordinator import PPOCoordinator, PPOCoordinatorConfig


class _StubBuffer:
    def __len__(self) -> int:
        return 5


class _StubAgent:
    def __init__(self) -> None:
        self.buffer = _StubBuffer()
        self.entropy_coef = 0.01


def _make_coordinator(*, run_ppo_updates_fn):
    return PPOCoordinator(
        agent=_StubAgent(),
        config=PPOCoordinatorConfig(
            ppo_updates_per_batch=1,
            max_epochs=4,
            total_env_episodes=16,
            amp_enabled=False,
            resolved_amp_dtype=None,
        ),
        reward_normalizer=SimpleNamespace(),
        anomaly_detector=SimpleNamespace(),
        env_reward_configs=[SimpleNamespace(reward_mode=SimpleNamespace(value="basic"))],
        reward_family_enum=SimpleNamespace(value="dense"),
        hub=None,
        telemetry_config=None,
        group_id="test-group",
        run_ppo_updates_fn=run_ppo_updates_fn,
        handle_telemetry_escalation_fn=lambda **_: None,
        emit_anomaly_diagnostics_fn=lambda **_: None,
        logger=logging.getLogger(__name__),
    )


def test_run_update_treats_epoch0_kl_abort_as_skipped_update():
    """Epoch-0 KL abort should not require gradient metrics from the coordinator."""

    def _run_ppo_updates_fn(**_kwargs):
        return {
            "ppo_update_performed": False,
            "finiteness_gate_skip_count": 0,
            "early_stop_epoch": 0,
            "approx_kl": 0.2,
        }

    coordinator = _make_coordinator(run_ppo_updates_fn=_run_ppo_updates_fn)

    metrics, update_skipped, ppo_update_time_ms = coordinator.run_update(
        raw_states_for_normalizer_update=[torch.ones(1, 3)],
        obs_normalizer=SimpleNamespace(update=lambda _tensor: None),
        envs_this_batch=2,
        throughput_step_time_ms_sum=10.0,
        throughput_dataloader_wait_ms_sum=1.0,
    )

    assert update_skipped is True
    assert metrics["ppo_update_performed"] is False
    assert "ppo_grad_norm" not in metrics
    assert ppo_update_time_ms is not None


def test_check_finiteness_gate_ignores_non_finiteness_skip_without_gradient_step():
    """KL early-stop before backward should not increment the finiteness failure counter."""

    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {})

    consecutive_failures, should_continue = coordinator.check_finiteness_gate(
        metrics={
            "ppo_update_performed": False,
            "finiteness_gate_skip_count": 0,
            "early_stop_epoch": 0,
        },
        consecutive_finiteness_failures=2,
    )

    assert consecutive_failures == 0
    assert should_continue is True
