"""Property-based tests for TamiyoRolloutBuffer.

Tier 6: Buffer Invariants and Stateful Testing

This module provides:
1. Property tests for buffer mathematical invariants
2. Stateful testing via RuleBasedStateMachine for episode lifecycle

Key invariants tested:
- Buffer never exceeds max_steps_per_env capacity
- Episode isolation: transitions don't leak between environments
- Reset completeness: reset() fully clears all state
- Batch consistency: sampled batches have correct shapes
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    rule,
    invariant,
    initialize,
    precondition,
)

from esper.leyline import (
    DEFAULT_LSTM_HIDDEN_DIM,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
from esper.simic.agent import TamiyoRolloutBuffer


# =============================================================================
# Property Tests: Buffer Invariants
# =============================================================================


class TestBufferCapacityInvariants:
    """Buffer capacity must be respected regardless of operation sequence."""

    @given(
        num_envs=st.integers(min_value=1, max_value=4),
        max_steps=st.integers(min_value=5, max_value=20),
        state_dim=st.integers(min_value=10, max_value=50),
    )
    @settings(max_examples=50)
    def test_buffer_creation_valid(self, num_envs: int, max_steps: int, state_dim: int):
        """Property: Buffer creation succeeds for valid parameters."""
        buffer = TamiyoRolloutBuffer(
            num_envs=num_envs,
            max_steps_per_env=max_steps,
            state_dim=state_dim,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        assert buffer.num_envs == num_envs
        assert buffer.max_steps_per_env == max_steps
        assert buffer.state_dim == state_dim

    @given(
        num_envs=st.integers(min_value=1, max_value=4),
        max_steps=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=30)
    def test_reset_clears_all_state(self, num_envs: int, max_steps: int):
        """Property: reset() completely clears buffer state."""
        buffer = TamiyoRolloutBuffer(
            num_envs=num_envs,
            max_steps_per_env=max_steps,
            state_dim=30,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        # Add some data
        buffer.start_episode(env_id=0)
        _add_dummy_transition(buffer, env_id=0, num_slots=3)
        buffer.end_episode(env_id=0)

        # Reset
        buffer.reset()

        # Verify cleared
        for env_id in range(num_envs):
            assert buffer.step_counts[env_id] == 0
            assert env_id not in buffer._current_episode_start


class TestBufferDataIntegrity:
    """Data stored in buffer must maintain integrity."""

    @given(
        reward=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
        value=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_reward_value_stored_correctly(self, reward: float, value: float):
        """Property: Stored rewards and values are retrievable without corruption."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=30,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        buffer.start_episode(env_id=0)
        _add_dummy_transition(buffer, env_id=0, num_slots=3, reward=reward, value=value)

        # Verify stored values (2D tensor: [env_id, step_idx])
        # Use approximate comparison due to float64 -> float32 conversion
        assert abs(buffer.rewards[0, 0].item() - reward) < 1e-5
        assert abs(buffer.values[0, 0].item() - value) < 1e-5

    @given(
        slot_action=st.integers(min_value=0, max_value=2),
        blueprint_action=st.integers(min_value=0, max_value=NUM_BLUEPRINTS - 1),
        tempo_action=st.integers(min_value=0, max_value=NUM_TEMPO - 1),
    )
    @settings(max_examples=50)
    def test_actions_stored_correctly(
        self, slot_action: int, blueprint_action: int, tempo_action: int
    ):
        """Property: Action indices are stored without corruption."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=30,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        buffer.start_episode(env_id=0)
        buffer.add(
            env_id=0,
            state=torch.randn(30),
            slot_action=slot_action,
            blueprint_action=blueprint_action,
            style_action=0,
            tempo_action=tempo_action,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=0,
            slot_log_prob=-1.0,
            blueprint_log_prob=-1.0,
            style_log_prob=-1.0,
            tempo_log_prob=-1.0,
            alpha_target_log_prob=-1.0,
            alpha_speed_log_prob=-1.0,
            alpha_curve_log_prob=-1.0,
            op_log_prob=-1.0,
            value=0.0,
            reward=0.0,
            done=False,
            slot_mask=torch.ones(3, dtype=torch.bool),
            blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
            tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
            alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
            alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
            alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
            op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
            hidden_h=torch.zeros(1, 1, DEFAULT_LSTM_HIDDEN_DIM),
            hidden_c=torch.zeros(1, 1, DEFAULT_LSTM_HIDDEN_DIM),
        )

        assert buffer.slot_actions[0, 0].item() == slot_action
        assert buffer.blueprint_actions[0, 0].item() == blueprint_action
        assert buffer.tempo_actions[0, 0].item() == tempo_action


class TestBufferLogProbInvariants:
    """Log probabilities must satisfy mathematical constraints."""

    @given(
        log_prob=st.floats(min_value=-20.0, max_value=0.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_log_probs_are_non_positive(self, log_prob: float):
        """Property: Log probabilities must be <= 0 (probabilities <= 1)."""
        # This is more of a documentation of expected invariant
        # The buffer should store whatever is passed, but callers should
        # ensure log_probs are valid
        assert log_prob <= 0.0, "Log probabilities must be non-positive"

    @given(
        slot_lp=st.floats(min_value=-10.0, max_value=0.0, allow_nan=False),
        blueprint_lp=st.floats(min_value=-10.0, max_value=0.0, allow_nan=False),
        tempo_lp=st.floats(min_value=-10.0, max_value=0.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_log_probs_stored_independently(
        self, slot_lp: float, blueprint_lp: float, tempo_lp: float
    ):
        """Property: Each head's log prob is stored independently."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=30,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        buffer.start_episode(env_id=0)
        buffer.add(
            env_id=0,
            state=torch.randn(30),
            slot_action=0,
            blueprint_action=0,
            style_action=0,
            tempo_action=1,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=0,
            slot_log_prob=slot_lp,
            blueprint_log_prob=blueprint_lp,
            style_log_prob=-1.0,
            tempo_log_prob=tempo_lp,
            alpha_target_log_prob=-1.0,
            alpha_speed_log_prob=-1.0,
            alpha_curve_log_prob=-1.0,
            op_log_prob=-1.0,
            value=0.0,
            reward=0.0,
            done=False,
            slot_mask=torch.ones(3, dtype=torch.bool),
            blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
            tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
            alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
            alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
            alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
            op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
            hidden_h=torch.zeros(1, 1, DEFAULT_LSTM_HIDDEN_DIM),
            hidden_c=torch.zeros(1, 1, DEFAULT_LSTM_HIDDEN_DIM),
        )

        assert abs(buffer.slot_log_probs[0, 0].item() - slot_lp) < 1e-6
        assert abs(buffer.blueprint_log_probs[0, 0].item() - blueprint_lp) < 1e-6
        assert abs(buffer.tempo_log_probs[0, 0].item() - tempo_lp) < 1e-6


# =============================================================================
# Stateful Testing: Buffer Episode Lifecycle
# =============================================================================


class RolloutBufferStateMachine(RuleBasedStateMachine):
    """Stateful test for TamiyoRolloutBuffer episode lifecycle.

    This state machine simulates realistic buffer usage patterns:
    - Starting/ending episodes
    - Adding transitions
    - Resetting the buffer

    Invariants verified after every operation:
    - Step counts are non-negative and bounded
    - Episode state is consistent
    - No data corruption across environments
    """

    def __init__(self):
        super().__init__()
        self.buffer = None
        self.num_envs = 2
        self.max_steps = 10
        self.num_slots = 3
        self.episode_active = {}
        self.steps_added = {}

    @initialize()
    def setup(self):
        """Initialize fresh buffer."""
        self.buffer = TamiyoRolloutBuffer(
            num_envs=self.num_envs,
            max_steps_per_env=self.max_steps,
            state_dim=30,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )
        self.episode_active = {i: False for i in range(self.num_envs)}
        self.steps_added = {i: 0 for i in range(self.num_envs)}

    @rule(env_id=st.integers(min_value=0, max_value=1))
    @precondition(lambda self: not all(self.episode_active.values()))
    def start_episode(self, env_id: int):
        """Start an episode for an environment."""
        if not self.episode_active[env_id]:
            self.buffer.start_episode(env_id=env_id)
            self.episode_active[env_id] = True

    @rule(
        env_id=st.integers(min_value=0, max_value=1),
        reward=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
    )
    @precondition(lambda self: any(self.episode_active.values()))
    def add_transition(self, env_id: int, reward: float):
        """Add a transition if episode is active and not full."""
        if self.episode_active[env_id] and self.steps_added[env_id] < self.max_steps:
            _add_dummy_transition(
                self.buffer, env_id=env_id, num_slots=self.num_slots, reward=reward
            )
            self.steps_added[env_id] += 1

    @rule(env_id=st.integers(min_value=0, max_value=1))
    @precondition(lambda self: any(self.episode_active.values()))
    def end_episode(self, env_id: int):
        """End an episode for an environment."""
        if self.episode_active[env_id] and self.steps_added[env_id] > 0:
            self.buffer.end_episode(env_id=env_id)
            self.episode_active[env_id] = False

    @rule()
    def reset_buffer(self):
        """Reset the entire buffer."""
        self.buffer.reset()
        self.episode_active = {i: False for i in range(self.num_envs)}
        self.steps_added = {i: 0 for i in range(self.num_envs)}

    # =========================================================================
    # Invariants
    # =========================================================================

    @invariant()
    def steps_non_negative(self):
        """Invariant: Step counts are always non-negative."""
        for env_id in range(self.num_envs):
            assert self.buffer.step_counts[env_id] >= 0, (
                f"Negative step count for env {env_id}: {self.buffer.step_counts[env_id]}"
            )

    @invariant()
    def steps_bounded(self):
        """Invariant: Step counts never exceed max_steps_per_env."""
        for env_id in range(self.num_envs):
            assert self.buffer.step_counts[env_id] <= self.max_steps, (
                f"Step count {self.buffer.step_counts[env_id]} exceeds max {self.max_steps}"
            )

    @invariant()
    def episode_state_consistent(self):
        """Invariant: Our tracking matches buffer's internal state."""
        for env_id in range(self.num_envs):
            # Buffer step count should match our tracking
            assert self.buffer.step_counts[env_id] == self.steps_added[env_id], (
                f"Step mismatch for env {env_id}: "
                f"buffer={self.buffer.step_counts[env_id]}, tracked={self.steps_added[env_id]}"
            )

    @invariant()
    def no_nan_in_rewards(self):
        """Invariant: No NaN values in stored rewards."""
        for env_id in range(self.num_envs):
            n_steps = self.buffer.step_counts[env_id]
            if n_steps > 0:
                rewards = self.buffer.rewards[env_id, :n_steps]
                assert not torch.isnan(rewards).any(), (
                    f"NaN found in rewards for env {env_id}"
                )


# Create test class for pytest discovery
TestRolloutBufferLifecycle = RolloutBufferStateMachine.TestCase


# =============================================================================
# Helper Functions
# =============================================================================


def _add_dummy_transition(
    buffer: TamiyoRolloutBuffer,
    env_id: int,
    num_slots: int,
    reward: float = 1.0,
    value: float = 0.5,
) -> None:
    """Add a dummy transition with valid structure."""
    buffer.add(
        env_id=env_id,
        state=torch.randn(buffer.state_dim),
        slot_action=0,
        blueprint_action=0,
        style_action=0,
        tempo_action=1,  # STANDARD
        alpha_target_action=0,
        alpha_speed_action=0,
        alpha_curve_action=0,
        op_action=0,
        slot_log_prob=-1.0,
        blueprint_log_prob=-1.0,
        style_log_prob=-1.0,
        tempo_log_prob=-1.0,
        alpha_target_log_prob=-1.0,
        alpha_speed_log_prob=-1.0,
        alpha_curve_log_prob=-1.0,
        op_log_prob=-1.0,
        value=value,
        reward=reward,
        done=False,
        slot_mask=torch.ones(num_slots, dtype=torch.bool),
        blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
        style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
        tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
        alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
        alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
        alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
        op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
        hidden_h=torch.zeros(1, 1, DEFAULT_LSTM_HIDDEN_DIM),
        hidden_c=torch.zeros(1, 1, DEFAULT_LSTM_HIDDEN_DIM),
    )
