"""Tests for TUI state management."""

from collections import deque

import pytest

from esper.karn.tui import EnvState, TUIState
from esper.leyline import TelemetryEvent, TelemetryEventType


class TestEnvState:
    """Tests for per-environment state tracking."""

    def test_env_state_has_reward_history(self):
        """EnvState tracks per-env reward history."""
        env = EnvState(env_id=0)
        assert hasattr(env, "reward_history")
        assert isinstance(env.reward_history, deque)
        assert env.reward_history.maxlen == 50

    def test_env_state_has_accuracy_history(self):
        """EnvState tracks per-env accuracy history."""
        env = EnvState(env_id=0)
        assert hasattr(env, "accuracy_history")
        assert isinstance(env.accuracy_history, deque)

    def test_env_state_has_action_history(self):
        """EnvState tracks recent actions."""
        env = EnvState(env_id=0)
        assert hasattr(env, "action_history")
        assert isinstance(env.action_history, deque)
        assert env.action_history.maxlen == 10

    def test_env_state_current_reward_from_history(self):
        """EnvState.current_reward returns last reward in history."""
        env = EnvState(env_id=0)
        assert env.current_reward == 0.0  # default when empty

        env.reward_history.append(0.5)
        env.reward_history.append(0.3)
        assert env.current_reward == 0.3

    def test_env_state_reward_sparkline(self):
        """EnvState generates sparkline from reward history."""
        env = EnvState(env_id=0)
        env.reward_history.extend([0.1, 0.3, 0.5, 0.7, 0.9])

        sparkline = env.reward_sparkline
        assert isinstance(sparkline, str)
        assert len(sparkline) > 0
        assert any(c in sparkline for c in "▁▂▃▄▅▆▇█")


class TestTUIStateAggregation:
    """Tests for TUIState aggregate calculations."""

    def test_aggregate_mean_reward_from_envs(self):
        """TUIState.aggregate_mean_reward averages across all envs."""
        state = TUIState()
        state.n_envs = 3

        for i in range(3):
            env = state.get_or_create_env(i)
            env.reward_history.append(float(i + 1) * 0.1)

        assert abs(state.aggregate_mean_reward - 0.2) < 0.001

    def test_aggregate_best_accuracy_tracks_source_env(self):
        """TUIState.aggregate_best_accuracy returns best across envs with source."""
        state = TUIState()
        state.n_envs = 3

        env0 = state.get_or_create_env(0)
        env0.best_accuracy = 75.0
        env0.best_accuracy_epoch = 10

        env1 = state.get_or_create_env(1)
        env1.best_accuracy = 85.0
        env1.best_accuracy_epoch = 15

        env2 = state.get_or_create_env(2)
        env2.best_accuracy = 70.0

        best_acc, best_env, best_epoch = state.aggregate_best_accuracy
        assert best_acc == 85.0
        assert best_env == 1
        assert best_epoch == 15

    def test_aggregate_action_counts(self):
        """TUIState.aggregate_action_counts sums across all envs."""
        state = TUIState()

        env0 = state.get_or_create_env(0)
        env0.action_counts = {"WAIT": 10, "GERMINATE": 2, "CULL": 1, "FOSSILIZE": 0}

        env1 = state.get_or_create_env(1)
        env1.action_counts = {"WAIT": 5, "GERMINATE": 3, "CULL": 0, "FOSSILIZE": 2}

        counts = state.aggregate_action_counts
        assert counts["WAIT"] == 15
        assert counts["GERMINATE"] == 5


class TestTUIOutputEventHandlers:
    """Tests for TUIOutput event routing."""

    def test_reward_computed_routes_to_correct_env(self):
        """REWARD_COMPUTED updates the correct env's reward history."""
        from esper.karn.tui import TUIOutput

        tui = TUIOutput()
        tui.state.n_envs = 4
        for i in range(4):
            tui.state.get_or_create_env(i)

        event = TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            epoch=10,
            data={
                "env_id": 2,
                "total_reward": 0.75,
                "action_name": "GERMINATE",
                "val_acc": 82.5,
            }
        )
        tui._handle_reward_computed(event)

        env2 = tui.state.env_states[2]
        assert env2.current_reward == 0.75
        assert env2.action_history[-1] == "GERMINATE"
        assert env2.host_accuracy == 82.5

        assert tui.state.env_states[0].current_reward == 0.0
        assert tui.state.env_states[1].current_reward == 0.0

    def test_action_distribution_per_env(self):
        """Actions are tracked per-env, not globally."""
        from esper.karn.tui import TUIOutput

        tui = TUIOutput()
        tui.state.n_envs = 2
        for i in range(2):
            tui.state.get_or_create_env(i)

        tui._handle_reward_computed(TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            data={"env_id": 0, "total_reward": 0.1, "action_name": "GERMINATE"}
        ))
        tui._handle_reward_computed(TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            data={"env_id": 1, "total_reward": 0.1, "action_name": "WAIT"}
        ))

        assert tui.state.env_states[0].action_counts["GERMINATE"] == 1
        assert tui.state.env_states[1].action_counts["WAIT"] == 1
        assert tui.state.aggregate_action_counts["GERMINATE"] == 1
        assert tui.state.aggregate_action_counts["WAIT"] == 1


class TestEventLogFormatting:
    """Tests for event log formatting with env IDs."""

    def test_event_log_includes_env_id(self):
        """Formatted event log entries include env prefix."""
        from esper.karn.tui import TUIOutput

        tui = TUIOutput()

        event = TelemetryEvent(
            event_type=TelemetryEventType.REWARD_COMPUTED,
            data={
                "env_id": 3,
                "action_name": "GERMINATE",
                "total_reward": 0.5,
            }
        )

        formatted = tui._format_event_for_log(event)
        assert formatted is not None
        timestamp, event_type, msg = formatted
        assert "[3]" in msg or "3" in msg
