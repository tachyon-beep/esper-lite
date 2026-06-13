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
        triggering_action_id: str | None = None,
        watch_window_evidence: float,
    ) -> bool:
        # Mirror the real buffer: when the caller does not supply a triggering
        # action id, resolve it from the most recent (prior, executed)
        # transition. The coordinator passes None so attribution lands on the
        # action that actually caused the panic, not a phantom rollback row.
        if triggering_action_id is None:
            triggering_action_id = self.last_action_id(env_id)
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
        reward_normalizer=SimpleNamespace(clip=10.0),
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


def test_handle_rollbacks_clamps_death_penalty_std_independently():
    """The rollback death penalty is clamped to the reward clip, std-INDEPENDENTLY.

    Regression for esper-lite-1e3d4e8fa6: the penalty used to be passed through
    normalize_only (raw_penalty / running_std, then clipped), which saturated at -clip
    only by accident when std < 1 and would silently shrink if std rose. The fix clamps
    the raw catastrophe penalty directly to the clip. This test's reward_normalizer mock
    exposes ONLY `clip` (no normalize_only / no std) -- so it passes only if the code path
    never consults the running std. A penalty larger than clip clamps to -clip; a penalty
    within clip is preserved exactly (not std-scaled).
    """
    for raw_penalty, expected_injected in ((-50.0, -10.0), (-3.0, -3.0)):
        coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {})
        env_states = [
            SimpleNamespace(
                governor=SimpleNamespace(get_punishment_reward=lambda p=raw_penalty: p),
                episode_rewards=[1.0, 2.0],
                telemetry_cb=lambda _e: None,
                val_acc=12.5,
                action_counts={"GERMINATE": 2, "PRUNE": 1, "FOSSILIZE": 0},
            )
        ]
        episode_history = [SimpleNamespace(env_id=0, episode_reward=3.0)]
        episode_outcomes = [_make_outcome(env_id=0, episode_idx=1, reward=3.0)]

        coordinator.handle_rollbacks(
            env_states=env_states,
            env_rollback_occurred=[True],
            env_total_rewards=[3.0],
            episode_history=episode_history,
            episode_outcomes=episode_outcomes,
        )

        env_idx, injected, severity, _action_id = coordinator.agent.buffer.penalty_calls[0]
        assert env_idx == 0
        assert injected == expected_injected  # clamped to clip, NOT raw/std
        assert severity == abs(raw_penalty)  # severity still records the true magnitude


def test_handle_rollbacks_drops_penalty_when_no_executed_transition(caplog):
    """First-step panic (no prior executed transition) must not crash.

    P2 fix: the coordinator must not eagerly resolve ``last_action_id`` (which
    raises when the buffer is empty). When ``mark_terminal_with_penalty``
    reports no transition was modified, the dropped penalty is surfaced via a
    warning rather than silently swallowed.
    """

    class _EmptyBuffer:
        def __len__(self) -> int:
            return 0

        def last_action_id(self, env_id: int) -> str:
            raise ValueError(f"env_id {env_id} has no transitions")

        def mark_terminal_with_penalty(
            self,
            env_id: int,
            penalty: float,
            *,
            severity: float,
            triggering_action_id: str | None = None,
            watch_window_evidence: float,
        ) -> bool:
            # The coordinator must NOT have eagerly resolved an action id from
            # an empty buffer (that would have raised before reaching here).
            assert triggering_action_id is None
            return False

    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {})
    coordinator.agent.buffer = _EmptyBuffer()
    env_states = [
        SimpleNamespace(
            governor=SimpleNamespace(get_punishment_reward=lambda: -10.0),
            episode_rewards=[],
            telemetry_cb=lambda event: None,
            val_acc=12.5,
            action_counts={},
        )
    ]

    with caplog.at_level(logging.WARNING):
        rollback_envs = coordinator.handle_rollbacks(
            env_states=env_states,
            env_rollback_occurred=[True],
            env_total_rewards=[0.0],
            episode_history=[],
            episode_outcomes=[],
        )

    assert rollback_envs == [0]
    assert any(
        "penalt" in record.getMessage().lower()
        for record in caplog.records
    )


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


def _finiteness_skip_metrics(skip_count: int):
    return {
        "ppo_update_performed": False,
        "finiteness_gate_skip_count": skip_count,
        "finiteness_gate_failures": [
            {"epoch": 0, "sources": ["log_probs[op]: NaN detected"]}
        ],
    }


def test_check_finiteness_gate_emits_degraded_skip_telemetry_on_first_skip():
    """SIMIC-PROD-004: the first (non-halting) finiteness skip emits auditable telemetry."""
    hub = _CaptureHub()
    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {}, hub=hub)
    metrics = _finiteness_skip_metrics(skip_count=4)

    consecutive_failures, should_continue = coordinator.check_finiteness_gate(
        metrics=metrics,
        consecutive_finiteness_failures=0,
    )

    assert consecutive_failures == 1
    assert should_continue is True
    assert metrics["run_governor_status"] == "degraded"
    # Exactly one degraded anomaly, severity below "error", non-halting type.
    assert len(hub.events) == 1
    event = hub.events[0]
    assert event.event_type == TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED
    assert event.severity == "warning"
    assert event.data.anomaly_type == "ppo_finiteness_skip"
    assert "1/3" in event.data.detail
    assert "all 4 epochs" in event.data.detail
    assert "log_probs[op]: NaN detected" in event.data.detail


def test_check_finiteness_gate_emits_degraded_skip_on_second_skip_without_halt():
    """SIMIC-PROD-004: the second consecutive skip also emits degraded telemetry, no halt."""
    hub = _CaptureHub()
    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {}, hub=hub)
    metrics = _finiteness_skip_metrics(skip_count=4)

    consecutive_failures, should_continue = coordinator.check_finiteness_gate(
        metrics=metrics,
        consecutive_finiteness_failures=1,
    )

    assert consecutive_failures == 2
    assert should_continue is True
    assert metrics["run_governor_status"] == "degraded"
    assert metrics["run_governor_finiteness_streak"] == 2
    assert len(hub.events) == 1
    assert hub.events[0].data.anomaly_type == "ppo_finiteness_skip"
    assert "2/3" in hub.events[0].data.detail


def test_check_finiteness_gate_does_not_double_emit_on_halting_third_skip():
    """SIMIC-PROD-004: the halting 3rd skip emits only the error anomaly, not the degraded one."""
    hub = _CaptureHub()
    coordinator = _make_coordinator(run_ppo_updates_fn=lambda **_kwargs: {}, hub=hub)
    metrics = _finiteness_skip_metrics(skip_count=4)

    with pytest.raises(RuntimeError, match="consecutive updates skipped"):
        coordinator.check_finiteness_gate(
            metrics=metrics,
            consecutive_finiteness_failures=2,
        )

    # Exactly one event, and it is the halting error anomaly (no degraded ppo_finiteness_skip).
    assert len(hub.events) == 1
    event = hub.events[0]
    assert event.severity == "error"
    assert event.data.anomaly_type == "ppo_finiteness_failure"


def test_run_anomaly_detection_threads_ratio_diagnostic_into_emitter():
    """SIMIC-PROD-002: PPO's ratio_diagnostic must reach the emitted anomaly payload."""
    captured: dict[str, object] = {}

    def _emit_anomaly_diagnostics_fn(*_args, **kwargs):
        captured.update(kwargs)

    report = SimpleNamespace(
        has_anomaly=True,
        anomaly_types=["ratio_explosion"],
        details={"ratio_explosion": "ratio max 12.0"},
    )
    detector = SimpleNamespace(
        check_all=lambda **_kw: report,
        check_lstm_health=lambda **_kw: SimpleNamespace(
            has_anomaly=False, anomaly_types=[], details={}
        ),
    )
    coordinator = PPOCoordinator(
        agent=_StubAgent(),
        config=PPOCoordinatorConfig(
            ppo_updates_per_batch=1,
            max_epochs=4,
            total_env_episodes=16,
            amp_enabled=False,
            resolved_amp_dtype=None,
        ),
        reward_normalizer=SimpleNamespace(clip=10.0),
        anomaly_detector=detector,
        env_reward_configs=[SimpleNamespace(reward_mode=SimpleNamespace(value="basic"))],
        reward_family_enum=SimpleNamespace(value="dense"),
        hub=_CaptureHub(),
        telemetry_config=None,
        group_id="test-group",
        run_ppo_updates_fn=lambda **_kw: {},
        handle_telemetry_escalation_fn=lambda *a, **k: None,
        emit_anomaly_diagnostics_fn=_emit_anomaly_diagnostics_fn,
        logger=logging.getLogger(__name__),
    )

    ratio_diag = {"head": "op", "ratio": 12.0, "where": "epoch3"}
    coordinator.run_anomaly_detection(
        metrics={
            "ratio_max": 12.0,
            "ratio_min": 0.9,
            "ratio_diagnostic": ratio_diag,
        },
        drift_metrics=None,
        batched_lstm_hidden=None,
        batch_epoch_id=3,
        batch_idx=1,
    )

    assert captured["ratio_diagnostic"] == ratio_diag


def test_run_anomaly_detection_preserves_update_lstm_and_adds_rollout_lstm():
    """SIMIC-PROD-003: update-time lstm_* survives; rollout health lands under rollout_lstm_*."""
    report = SimpleNamespace(has_anomaly=False, anomaly_types=[], details={})
    detector = SimpleNamespace(
        check_all=lambda **_kw: report,
        check_lstm_health=lambda **_kw: SimpleNamespace(
            has_anomaly=False, anomaly_types=[], details={}
        ),
    )
    coordinator = PPOCoordinator(
        agent=_StubAgent(),
        config=PPOCoordinatorConfig(
            ppo_updates_per_batch=1,
            max_epochs=4,
            total_env_episodes=16,
            amp_enabled=False,
            resolved_amp_dtype=None,
        ),
        reward_normalizer=SimpleNamespace(clip=10.0),
        anomaly_detector=detector,
        env_reward_configs=[SimpleNamespace(reward_mode=SimpleNamespace(value="basic"))],
        reward_family_enum=SimpleNamespace(value="dense"),
        hub=_CaptureHub(),
        telemetry_config=None,
        group_id="test-group",
        run_ppo_updates_fn=lambda **_kw: {},
        handle_telemetry_escalation_fn=lambda *a, **k: None,
        emit_anomaly_diagnostics_fn=lambda *a, **k: None,
        logger=logging.getLogger(__name__),
    )

    # Distinct rollout hidden state so rollout RMS differs from the update-time value.
    rollout_hidden = (torch.full((1, 2, 4), 3.0), torch.full((1, 2, 4), 5.0))
    metrics = {
        "ratio_max": 1.1,
        "ratio_min": 0.9,
        # Update-time health from finalize() (must NOT be clobbered).
        "lstm_h_rms": 0.5,
        "lstm_c_rms": 0.7,
        "lstm_h_max": 1.0,
    }

    coordinator.run_anomaly_detection(
        metrics=metrics,
        drift_metrics=None,
        batched_lstm_hidden=rollout_hidden,
        batch_epoch_id=3,
        batch_idx=1,
    )

    # Update-time health preserved.
    assert metrics["lstm_h_rms"] == 0.5
    assert metrics["lstm_c_rms"] == 0.7
    assert metrics["lstm_h_max"] == 1.0
    # Rollout health surfaced under distinct keys and reflects rollout_hidden (RMS 3.0 / 5.0).
    assert metrics["rollout_lstm_h_rms"] == pytest.approx(3.0)
    assert metrics["rollout_lstm_c_rms"] == pytest.approx(5.0)
    assert metrics["rollout_lstm_h_rms"] != metrics["lstm_h_rms"]


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
