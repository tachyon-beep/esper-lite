"""Tests for TUI state management."""

from collections import deque

import pytest

from esper.karn.tui import EnvState, TUIState


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
