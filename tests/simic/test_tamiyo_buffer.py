"""Tests for TamiyoRolloutBuffer - unified factored recurrent buffer."""

import pytest
import torch

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


class TestTamiyoRolloutBuffer:
    """Tests for the unified Tamiyo rollout buffer."""

    def test_per_env_storage_isolation(self):
        """Transitions from different envs must not contaminate each other's GAE.

        This is the P0 bug fix: interleaved storage caused GAE to use values
        from wrong environments.
        """
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        # Env 0: low rewards (1.0)
        buffer.start_episode(env_id=0)
        for i in range(3):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=0,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
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
                value=1.0,
                reward=1.0,
                done=(i == 2),
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 512),
                hidden_c=torch.zeros(1, 1, 512),
                blueprint_indices=torch.zeros(3, dtype=torch.long),
            )
        buffer.end_episode(env_id=0)

        # Env 1: HIGH rewards (100.0) - should NOT affect env 0's GAE
        buffer.start_episode(env_id=1)
        for i in range(3):
            buffer.add(
                env_id=1,
                state=torch.randn(50),
                slot_action=0,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
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
                value=50.0,
                reward=100.0,  # HIGH reward
                done=(i == 2),
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 512),
                hidden_c=torch.zeros(1, 1, 512),
                blueprint_indices=torch.zeros(3, dtype=torch.long),
            )
        buffer.end_episode(env_id=1)

        buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)

        # Env 0's advantages should be based ONLY on env 0's rewards (1.0)
        # They should be small, NOT contaminated by env 1's high rewards
        env0_advantages = buffer.advantages[0, :3]
        env1_advantages = buffer.advantages[1, :3]

        # Env 0 advantages should be MUCH smaller than env 1
        assert env0_advantages.abs().max() < 10.0, (
            f"Env 0 advantages {env0_advantages} contaminated by env 1 high rewards"
        )
        assert env1_advantages.abs().max() > 10.0, (
            f"Env 1 advantages {env1_advantages} should be large from high rewards"
        )

    def test_per_head_log_probs_stored(self):
        """Buffer must store per-head log probs, not just joint."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        buffer.start_episode(env_id=0)
        buffer.add(
            env_id=0,
            state=torch.randn(50),
            slot_action=1,
            blueprint_action=2,
            style_action=0,
            tempo_action=1,
            alpha_target_action=0,
            alpha_speed_action=0,
            alpha_curve_action=0,
            op_action=1,
            slot_log_prob=-0.5,
            blueprint_log_prob=-1.2,
            style_log_prob=-0.3,
            tempo_log_prob=-0.8,
            alpha_target_log_prob=-0.8,
            alpha_speed_log_prob=-0.8,
            alpha_curve_log_prob=-0.8,
            op_log_prob=-0.8,
            value=1.0,
            reward=1.0,
            done=True,
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
            blueprint_indices=torch.zeros(3, dtype=torch.long),
        )
        buffer.end_episode(env_id=0)

        # Check per-head log probs are stored correctly
        assert buffer.slot_log_probs[0, 0].item() == pytest.approx(-0.5)
        assert buffer.blueprint_log_probs[0, 0].item() == pytest.approx(-1.2)
        assert buffer.style_log_probs[0, 0].item() == pytest.approx(-0.3)
        assert buffer.tempo_log_probs[0, 0].item() == pytest.approx(-0.8)
        assert buffer.op_log_probs[0, 0].item() == pytest.approx(-0.8)

    def test_lstm_hidden_states_stored(self):
        """Buffer must store LSTM hidden states for sequence reconstruction."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        hidden_h = torch.randn(1, 1, 512)
        hidden_c = torch.randn(1, 1, 512)

        buffer.start_episode(env_id=0)
        buffer.add(
            env_id=0,
            state=torch.randn(50),
            slot_action=0,
            blueprint_action=0,
            style_action=0,
            tempo_action=0,
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
            value=1.0,
            reward=1.0,
            done=True,
            slot_mask=torch.ones(3, dtype=torch.bool),
            blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
            style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
            tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
            alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
            alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
            alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
            op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
            hidden_h=hidden_h,
            hidden_c=hidden_c,
            blueprint_indices=torch.zeros(3, dtype=torch.long),
        )
        buffer.end_episode(env_id=0)

        # Hidden states should be stored (squeezed from [1, 1, 512] to [512])
        stored_h = buffer.hidden_h[0, 0]
        assert stored_h.shape == (1, 512)
        assert torch.allclose(stored_h.squeeze(0), hidden_h.squeeze())

    def test_empty_buffer_update(self):
        """update with empty buffer should return empty dict, not crash."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        # Should not crash - returns without computing anything
        buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)
        assert len(buffer) == 0

    def test_normalize_advantages_single_transition_no_nan(self):
        """normalize_advantages must not produce NaNs for batch size 1.

        torch.std defaults to Bessel-corrected (correction=1), which yields NaN
        for a single element. Advantage normalization runs every PPO update, so
        this edge case must be numerically safe.
        """
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=1,
            state_dim=10,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        buffer.start_episode(env_id=0)
        buffer.add(
            env_id=0,
            state=torch.randn(10),
            slot_action=0,
            blueprint_action=0,
            style_action=0,
            tempo_action=0,
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
            reward=1.0,
            done=True,
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
            blueprint_indices=torch.zeros(3, dtype=torch.long),
        )
        buffer.end_episode(env_id=0)

        buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)
        buffer.normalize_advantages()

        assert torch.isfinite(buffer.advantages[0, 0]).item() is True

    def test_buffer_overflow_raises(self):
        """Exceeding max_steps_per_env should raise RuntimeError."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=2,  # Very small
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        buffer.start_episode(env_id=0)
        # Add 2 transitions (at limit)
        for i in range(2):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=0,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
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
                value=1.0,
                reward=1.0,
                done=False,
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 512),
                hidden_c=torch.zeros(1, 1, 512),
                blueprint_indices=torch.zeros(3, dtype=torch.long),
            )

        # Third add should raise
        with pytest.raises(RuntimeError, match="exceeded max_steps"):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=0,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
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
                value=1.0,
                reward=1.0,
                done=False,
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 512),
                hidden_c=torch.zeros(1, 1, 512),
                blueprint_indices=torch.zeros(3, dtype=torch.long),
            )

    def test_dynamic_slot_config_3_slots(self):
        """Buffer with 3-slot config should allocate [n, max_steps, 3] slot_masks."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.default()  # 3 slots (r0c0, r0c1, r0c2)

        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
            slot_config=slot_config,
        )

        # slot_masks should have shape [num_envs, max_steps, 3]
        assert buffer.slot_masks.shape == (2, 5, 3)
        assert buffer.num_slots == 3

    def test_dynamic_slot_config_5_slots(self):
        """Buffer with 5-slot config should allocate [n, max_steps, 5] slot_masks."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.for_grid(rows=1, cols=5)  # 5 slots

        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
            slot_config=slot_config,
        )

        # slot_masks should have shape [num_envs, max_steps, 5]
        assert buffer.slot_masks.shape == (2, 5, 5)
        assert buffer.num_slots == 5

    def test_dynamic_slot_config_full_cycle(self):
        """Full PPO cycle with non-default slot_config should work end-to-end."""
        from esper.leyline.slot_config import SlotConfig

        slot_config = SlotConfig.for_grid(rows=1, cols=5)  # 5 slots

        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=3,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
            slot_config=slot_config,
        )

        # Add transitions with 5-slot masks
        buffer.start_episode(env_id=0)
        for i in range(3):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=i % 5,  # Use slots 0-4
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
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
                value=1.0,
                reward=1.0,
                done=(i == 2),
                slot_mask=torch.ones(5, dtype=torch.bool),  # 5 slots
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 512),
                hidden_c=torch.zeros(1, 1, 512),
                blueprint_indices=torch.zeros(5, dtype=torch.long),  # 5 slots
            )
        buffer.end_episode(env_id=0)

        # Compute advantages and returns
        buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)

        # Get batched data - slot_masks should have correct shape
        data = buffer.get_batched_sequences()
        assert data["slot_masks"].shape == (1, 3, 5)
        assert data["slot_actions"].shape == (1, 3)

    def test_mark_terminal_with_penalty(self):
        """mark_terminal_with_penalty should inject death penalty and mark done.

        B1-DRL-01 fix: When governor rollback occurs, the agent needs a negative
        reward signal to learn to avoid catastrophic actions.
        """
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        # Add some transitions with positive rewards
        buffer.start_episode(env_id=0)
        for i in range(3):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=0,
                blueprint_action=0,
                style_action=0,
                tempo_action=0,
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
                value=1.0,
                reward=1.0,  # Positive reward
                done=False,
                slot_mask=torch.ones(buffer.num_slots, dtype=torch.bool),
                blueprint_mask=torch.ones(NUM_BLUEPRINTS, dtype=torch.bool),
                style_mask=torch.ones(NUM_STYLES, dtype=torch.bool),
                tempo_mask=torch.ones(NUM_TEMPO, dtype=torch.bool),
                alpha_target_mask=torch.ones(NUM_ALPHA_TARGETS, dtype=torch.bool),
                alpha_speed_mask=torch.ones(NUM_ALPHA_SPEEDS, dtype=torch.bool),
                alpha_curve_mask=torch.ones(NUM_ALPHA_CURVES, dtype=torch.bool),
                op_mask=torch.ones(NUM_OPS, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 512),
                hidden_c=torch.zeros(1, 1, 512),
                blueprint_indices=torch.zeros(buffer.num_slots, dtype=torch.long),
            )

        # Simulate governor rollback with death penalty
        death_penalty = -10.0
        modified = buffer.mark_terminal_with_penalty(env_id=0, penalty=death_penalty)

        assert modified is True
        # Last transition should now have death penalty
        assert buffer.rewards[0, 2].item() == death_penalty
        # Last transition should be marked as terminal
        assert buffer.dones[0, 2].item() is True
        # Not truncated - true terminal
        assert buffer.truncated[0, 2].item() is False
        # Transitions are preserved (not cleared)
        assert buffer.step_counts[0] == 3

    def test_mark_terminal_with_penalty_empty_env(self):
        """mark_terminal_with_penalty on empty env returns False."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        # No transitions added - should return False
        modified = buffer.mark_terminal_with_penalty(env_id=0, penalty=-10.0)
        assert modified is False

    def test_mark_terminal_with_penalty_invalid_env(self):
        """mark_terminal_with_penalty with invalid env_id raises ValueError."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=DEFAULT_LSTM_HIDDEN_DIM,
        )

        with pytest.raises(ValueError, match="out of range"):
            buffer.mark_terminal_with_penalty(env_id=5, penalty=-10.0)
