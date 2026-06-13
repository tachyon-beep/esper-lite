"""Tests for PPO update coordination edge cases."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from esper.leyline import EpisodeOutcome, TelemetryEventType
from esper.simic.training.ppo_coordinator import PPOCoordinator, PPOCoordinatorConfig


class _StubBuffer:
    def __init__(self, length: int = 5) -> None:
        self.length = length
        self.penalty_calls: list[tuple[int, float, float, str]] = []
        self.action_ids = (
            ("morph-b0-e1-env0-r0c0-op3",),
            ("morph-b0-e1-env1-r0c0-op4",),
        )

    def __len__(self) -> int:
        return self.length

    def last_action_id(self, env_id: int) -> str:
        return self.action_ids[env_id][0]

    def mark_terminal_with_penalty(
        self,
        env_id: int,
        penalty: float,
        *,
        severity: float,
        triggering_action_id: str,
        watch_window_evidence: float,
    ) -> bool:
        self.penalty_calls.append((env_id, penalty, severity, triggering_action_id))
        return True


class _StubAgent:
    def __init__(self) -> None:
        self.buffer = _StubBuffer()
        self.entropy_coef = 0.01


class _CaptureHub:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


def _make_coordinator(*, run_ppo_updates_fn, hub=None):
    return PPOCoordinator(
        agent=_StubAgent(),
        config=PPOCoordinatorConfig(
            ppo_updates_per_batch=1,
            max_epochs=4,
            total_env_episodes=16,
            amp_enabled=False,
            resolved_amp_dtype=None,
        ),
        reward_normalizer=SimpleNamespace(normalize_only=lambda reward: reward),
        anomaly_detector=SimpleNamespace(),
        env_reward_configs=[SimpleNamespace(reward_mode=SimpleNamespace(value="basic"))],
        reward_family_enum=SimpleNamespace(value="dense"),
        hub=hub,
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


def _make_outcome(env_id: int, episode_idx: int, reward: float) -> EpisodeOutcome:
    return EpisodeOutcome(
        env_id=env_id,
        episode_idx=episode_idx,
        final_accuracy=0.5,
        param_ratio=1.0,
        num_fossilized=0,
        num_contributing_fossilized=0,
        episode_reward=reward,
        stability_score=1.0,
        reward_mode="basic",
    )


def test_handle_rollbacks_corrects_latest_episode_outcome_for_env():
    """Rollback correction must update the most recent outcome for the env."""
    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {})
    emitted: list[object] = []
    env_states = [
        SimpleNamespace(
            governor=SimpleNamespace(get_punishment_reward=lambda: -10.0),
            episode_rewards=[1.0, 2.0],
            telemetry_cb=emitted.append,
            val_acc=12.5,
            action_counts={"GERMINATE": 2, "PRUNE": 1, "FOSSILIZE": 0},
        )
    ]
    env_total_rewards = [3.0]
    episode_history = [
        SimpleNamespace(env_id=0, episode_reward=100.0),
        SimpleNamespace(env_id=0, episode_reward=3.0),
    ]
    episode_outcomes = [
        _make_outcome(env_id=0, episode_idx=1, reward=100.0),
        _make_outcome(env_id=0, episode_idx=2, reward=3.0),
    ]

    rollback_envs = coordinator.handle_rollbacks(
        env_states=env_states,
        env_rollback_occurred=[True],
        env_total_rewards=env_total_rewards,
        episode_history=episode_history,
        episode_outcomes=episode_outcomes,
    )

    assert rollback_envs == [0]
    assert coordinator.agent.buffer.penalty_calls == [
        (0, -10.0, 10.0, "morph-b0-e1-env0-r0c0-op3")
    ]
    assert env_total_rewards == [-9.0]
    assert episode_history[0].episode_reward == 100.0
    assert episode_history[1].episode_reward == -9.0
    assert episode_outcomes[0].episode_reward == 100.0
    assert episode_outcomes[1].episode_reward == -9.0


def test_handle_rollbacks_emits_exactly_one_corrected_episode_outcome():
    """SIMIC-PROD-001: a rollback episode emits exactly one penalty-adjusted EPISODE_OUTCOME."""
    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {})
    emitted: list[object] = []
    # Two episode rewards with variance so stability is recomputed (not the default 1.0).
    env_states = [
        SimpleNamespace(
            governor=SimpleNamespace(get_punishment_reward=lambda: -10.0),
            episode_rewards=[1.0, 2.0],
            telemetry_cb=emitted.append,
            val_acc=12.5,
            action_counts={"GERMINATE": 2, "PRUNE": 1, "FOSSILIZE": 0},
        )
    ]
    env_total_rewards = [3.0]
    episode_history = [SimpleNamespace(env_id=0, episode_reward=3.0)]
    episode_outcomes = [_make_outcome(env_id=0, episode_idx=7, reward=3.0)]

    coordinator.handle_rollbacks(
        env_states=env_states,
        env_rollback_occurred=[True],
        env_total_rewards=env_total_rewards,
        episode_history=episode_history,
        episode_outcomes=episode_outcomes,
    )

    # EXACTLY ONE corrected outcome is emitted (no uncorrected + corrected double-emit).
    outcome_events = [
        e for e in emitted
        if e.event_type == TelemetryEventType.EPISODE_OUTCOME
    ]
    assert len(outcome_events) == 1

    payload = outcome_events[0].data
    # Penalty-adjusted reward (3.0 episode reward overwritten to penalty -10.0 -> sum -9.0)
    # MUST match the corrected in-memory outcome, not the pre-penalty value.
    assert payload.episode_reward == -9.0
    assert payload.episode_reward == episode_outcomes[0].episode_reward
    # Recomputed stability from post-penalty variance of [1.0, -10.0], not the default 1.0.
    import numpy as np

    expected_stability = 1.0 / (1.0 + float(np.var([1.0, -10.0])))
    assert payload.stability_score == pytest.approx(expected_stability)
    assert payload.stability_score == pytest.approx(episode_outcomes[0].stability_score)
    # Real post-episode diagnostics taken from env_state (not placeholders).
    assert payload.episode_idx == 7
    assert payload.episode_length == coordinator.config.max_epochs
    assert payload.outcome_type == "timeout"  # val_acc 12.5 below success threshold
    assert payload.germinate_count == 2
    assert payload.prune_count == 1
    assert payload.fossilize_count == 0


def test_run_update_reports_rollout_total_steps_before_buffer_clear():
    """rollout_total_steps should reflect the rollout size before PPO clears it."""

    coordinator = None

    def _run_ppo_updates_fn(**_kwargs):
        assert coordinator is not None
        coordinator.agent.buffer.length = 0
        return {
            "ppo_update_performed": True,
            "pre_clip_grad_norm": 1.0,
        }

    coordinator = _make_coordinator(run_ppo_updates_fn=_run_ppo_updates_fn)

    metrics, update_skipped, ppo_update_time_ms = coordinator.run_update(
        raw_states_for_normalizer_update=[torch.ones(1, 3)],
        obs_normalizer=SimpleNamespace(update=lambda _tensor: None),
        envs_this_batch=2,
        throughput_step_time_ms_sum=10.0,
        throughput_dataloader_wait_ms_sum=1.0,
    )

    assert update_skipped is False
    assert metrics["rollout_total_steps"] == 5
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


def test_check_finiteness_gate_marks_batch_degraded_but_allows_telemetry():
    """First finiteness-gate skip should advance through degraded batch telemetry."""

    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {})
    metrics = {
        "ppo_update_performed": False,
        "finiteness_gate_skip_count": 1,
        "finiteness_gate_failures": [
            {
                "epoch": 0,
                "sources": ["log_probs[op]: NaN detected"],
            }
        ],
    }

    consecutive_failures, should_continue = coordinator.check_finiteness_gate(
        metrics=metrics,
        consecutive_finiteness_failures=0,
    )

    assert consecutive_failures == 1
    assert should_continue is True
    assert metrics["run_governor_signal"] == "ppo_finiteness_failure"
    assert metrics["run_governor_status"] == "degraded"


def test_check_finiteness_gate_emits_proof_blocking_anomaly_on_repeated_failures():
    """Repeated PPO finiteness failures should surface as a run-level governor signal."""
    hub = _CaptureHub()
    coordinator = _make_coordinator(
        run_ppo_updates_fn=lambda **_kwargs: {},
        hub=hub,
    )
    metrics = {
        "ppo_update_performed": False,
        "finiteness_gate_skip_count": 2,
        "finiteness_gate_failures": [
            {
                "epoch": 0,
                "sources": [
                    "log_probs[op]: NaN detected",
                    "values: Inf detected",
                ],
            }
        ],
    }

    with pytest.raises(RuntimeError, match="consecutive updates skipped"):
        coordinator.check_finiteness_gate(
            metrics=metrics,
            consecutive_finiteness_failures=2,
        )

    assert metrics["run_governor_signal"] == "ppo_finiteness_failure"
    assert metrics["run_governor_status"] == "halted"
    assert len(hub.events) == 1
    event = hub.events[0]
    assert event.event_type == TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED
    assert event.severity == "error"
    assert event.data.anomaly_type == "ppo_finiteness_failure"
    assert "log_probs[op]: NaN detected" in event.data.detail
    assert "values: Inf detected" in event.data.detail


@pytest.mark.parametrize("bad_norm", [float("nan"), float("inf"), float("-inf")])
def test_check_gradient_drift_rejects_non_finite_norm_without_poisoning_tracker(bad_norm):
    """Non-finite grad norms must fail before entering EMA state."""
    from esper.simic.telemetry import GradientEMATracker

    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {})
    tracker = GradientEMATracker()
    tracker.update(grad_norm=10.0, grad_health=1.0)
    state_before = tracker.state_dict()

    with pytest.raises(ValueError, match="ppo_grad_norm must be finite"):
        coordinator.check_gradient_drift(tracker, bad_norm)

    assert tracker.state_dict() == state_before
