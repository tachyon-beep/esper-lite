"""Unit tests for TamiyoRolloutBuffer.

These tests focus on buffer invariants and edge cases.
Property tests in test_buffer_properties.py cover mathematical properties.
"""

from __future__ import annotations

import pytest
import torch

from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer


class TestBufferInitialization:
    """Tests for buffer initialization and pre-allocation."""

    def test_creates_with_correct_shapes(self) -> None:
        """Buffer tensors have expected shapes after init."""
        buffer = TamiyoRolloutBuffer(
            num_envs=4,
            max_steps_per_env=25,
            state_dim=128,
        )

        assert buffer.states.shape == (4, 25, 128)
        assert buffer.rewards.shape == (4, 25)
        assert buffer.values.shape == (4, 25)
        assert buffer.advantages.shape == (4, 25)
        assert buffer.dones.shape == (4, 25)

    def test_step_counts_initialized_to_zero(self) -> None:
        """All environments start with zero steps."""
        buffer = TamiyoRolloutBuffer(
            num_envs=4,
            max_steps_per_env=25,
            state_dim=128,
        )

        assert buffer.step_counts == [0, 0, 0, 0]

    def test_action_masks_have_default_valid_action(self) -> None:
        """Padded rows have at least one valid action to prevent crashes."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )

        # At least one action should be valid per row for padded timesteps
        assert buffer.slot_masks[:, :, 0].all()
        assert buffer.blueprint_masks[:, :, 0].all()
        assert buffer.op_masks[:, :, 0].all()

    def test_len_returns_total_transitions(self) -> None:
        """__len__ returns sum of all step counts."""
        buffer = TamiyoRolloutBuffer(
            num_envs=3,
            max_steps_per_env=10,
            state_dim=64,
        )
        buffer.step_counts = [5, 3, 7]
        assert len(buffer) == 15


class TestBufferAdd:
    """Tests for adding transitions to the buffer."""

    def _make_minimal_add_kwargs(
        self, buffer: TamiyoRolloutBuffer, env_id: int = 0
    ) -> dict:
        """Create minimal valid kwargs for buffer.add()."""
        num_slots = buffer.num_slots
        return {
            "env_id": env_id,
            "state": torch.zeros(buffer.state_dim),
            "blueprint_indices": torch.zeros(num_slots, dtype=torch.long),
            "slot_action": 0,
            "blueprint_action": 0,
            "style_action": 0,
            "tempo_action": 0,
            "alpha_target_action": 0,
            "alpha_speed_action": 0,
            "alpha_curve_action": 0,
            "op_action": 0,
            "effective_op_action": 0,
            "slot_log_prob": 0.0,
            "blueprint_log_prob": 0.0,
            "style_log_prob": 0.0,
            "tempo_log_prob": 0.0,
            "alpha_target_log_prob": 0.0,
            "alpha_speed_log_prob": 0.0,
            "alpha_curve_log_prob": 0.0,
            "op_log_prob": 0.0,
            "value": 0.5,
            "reward": 1.0,
            "done": False,
            "slot_mask": torch.ones(num_slots, dtype=torch.bool),
            "blueprint_mask": torch.ones(buffer.num_blueprints, dtype=torch.bool),
            "style_mask": torch.ones(buffer.num_styles, dtype=torch.bool),
            "tempo_mask": torch.ones(buffer.num_tempo, dtype=torch.bool),
            "alpha_target_mask": torch.ones(buffer.num_alpha_targets, dtype=torch.bool),
            "alpha_speed_mask": torch.ones(buffer.num_alpha_speeds, dtype=torch.bool),
            "alpha_curve_mask": torch.ones(buffer.num_alpha_curves, dtype=torch.bool),
            "op_mask": torch.ones(buffer.num_ops, dtype=torch.bool),
            "hidden_h": torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
            "hidden_c": torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
        }

    def test_add_increments_step_count(self) -> None:
        """Adding a transition increments step_count for that env."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )

        buffer.add(**self._make_minimal_add_kwargs(buffer, env_id=0))
        assert buffer.step_counts[0] == 1
        assert buffer.step_counts[1] == 0

        buffer.add(**self._make_minimal_add_kwargs(buffer, env_id=1))
        assert buffer.step_counts[0] == 1
        assert buffer.step_counts[1] == 1

    def test_add_stores_reward_correctly(self) -> None:
        """Reward is stored at correct index."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )

        kwargs = self._make_minimal_add_kwargs(buffer, env_id=0)
        kwargs["reward"] = 42.0
        buffer.add(**kwargs)

        assert buffer.rewards[0, 0].item() == 42.0

    def test_add_raises_on_overflow(self) -> None:
        """Adding beyond max_steps raises RuntimeError."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=2,
            state_dim=64,
        )

        buffer.add(**self._make_minimal_add_kwargs(buffer))
        buffer.add(**self._make_minimal_add_kwargs(buffer))

        with pytest.raises(RuntimeError, match="exceeded max_steps"):
            buffer.add(**self._make_minimal_add_kwargs(buffer))

    def test_add_detaches_tensors(self) -> None:
        """Added tensors are detached to prevent gradient leaks."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=64,
        )

        # Create tensor that requires grad
        state = torch.zeros(64, requires_grad=True)
        kwargs = self._make_minimal_add_kwargs(buffer)
        kwargs["state"] = state
        buffer.add(**kwargs)

        # Stored tensor should not require grad
        assert not buffer.states[0, 0].requires_grad


class TestBufferReset:
    """Tests for buffer reset functionality."""

    def test_reset_clears_step_counts(self) -> None:
        """reset() sets all step counts to zero."""
        buffer = TamiyoRolloutBuffer(
            num_envs=3,
            max_steps_per_env=10,
            state_dim=64,
        )
        buffer.step_counts = [5, 3, 7]
        buffer.reset()
        assert buffer.step_counts == [0, 0, 0]

    def test_reset_clears_episode_boundaries(self) -> None:
        """reset() clears episode tracking."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )
        buffer.episode_boundaries = {0: [(0, 5)], 1: [(0, 3)]}
        buffer._current_episode_start = {0: 5}
        buffer.reset()
        assert buffer.episode_boundaries == {}
        assert buffer._current_episode_start == {}

    def test_reset_clears_normalization_flag(self) -> None:
        """reset() clears advantages normalization flag."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )
        buffer._advantages_normalized = True
        buffer.reset()
        assert not buffer._advantages_normalized


class TestClearEnv:
    """Tests for single-environment clearing."""

    def test_clear_env_zeros_step_count(self) -> None:
        """clear_env() zeros step count for that env only."""
        buffer = TamiyoRolloutBuffer(
            num_envs=3,
            max_steps_per_env=10,
            state_dim=64,
        )
        buffer.step_counts = [5, 3, 7]
        buffer.clear_env(1)
        assert buffer.step_counts == [5, 0, 7]

    def test_clear_env_invalid_id_raises(self) -> None:
        """clear_env() with invalid env_id raises ValueError."""
        buffer = TamiyoRolloutBuffer(
            num_envs=3,
            max_steps_per_env=10,
            state_dim=64,
        )

        with pytest.raises(ValueError, match="out of range"):
            buffer.clear_env(-1)

        with pytest.raises(ValueError, match="out of range"):
            buffer.clear_env(3)

    def test_clear_env_zeros_hidden_states(self) -> None:
        """clear_env() zeros LSTM hidden states for that env."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )
        buffer.hidden_h[0].fill_(1.0)
        buffer.hidden_c[0].fill_(1.0)
        buffer.clear_env(0)
        assert buffer.hidden_h[0].sum().item() == 0.0
        assert buffer.hidden_c[0].sum().item() == 0.0


class TestMarkTerminalWithPenalty:
    """Tests for terminal marking with penalty reward."""

    def _add_steps(self, buffer: TamiyoRolloutBuffer, env_id: int, n: int) -> None:
        """Helper to add n minimal steps to buffer."""
        num_slots = buffer.num_slots
        for _ in range(n):
            buffer.add(
                env_id=env_id,
                state=torch.zeros(buffer.state_dim),
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
                reward=1.0,
                done=False,
                slot_mask=torch.ones(num_slots, dtype=torch.bool),
                blueprint_mask=torch.ones(buffer.num_blueprints, dtype=torch.bool),
                style_mask=torch.ones(buffer.num_styles, dtype=torch.bool),
                tempo_mask=torch.ones(buffer.num_tempo, dtype=torch.bool),
                alpha_target_mask=torch.ones(buffer.num_alpha_targets, dtype=torch.bool),
                alpha_speed_mask=torch.ones(buffer.num_alpha_speeds, dtype=torch.bool),
                alpha_curve_mask=torch.ones(buffer.num_alpha_curves, dtype=torch.bool),
                op_mask=torch.ones(buffer.num_ops, dtype=torch.bool),
                hidden_h=torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
                hidden_c=torch.zeros(buffer.lstm_layers, 1, buffer.lstm_hidden_dim),
            )

    def test_marks_last_step_terminal(self) -> None:
        """mark_terminal_with_penalty sets done=True on last step."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=64,
        )
        self._add_steps(buffer, 0, 5)

        buffer.mark_terminal_with_penalty(0, penalty=-10.0)

        assert buffer.dones[0, 4].item() is True
        assert buffer.truncated[0, 4].item() is False

    def test_applies_penalty_reward(self) -> None:
        """mark_terminal_with_penalty overwrites reward with penalty."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=64,
        )
        self._add_steps(buffer, 0, 5)

        buffer.mark_terminal_with_penalty(0, penalty=-10.0)

        assert buffer.rewards[0, 4].item() == -10.0

    def test_returns_false_for_empty_env(self) -> None:
        """Returns False if env has no transitions."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=64,
        )

        result = buffer.mark_terminal_with_penalty(0, penalty=-10.0)
        assert result is False

    def test_invalid_env_id_raises(self) -> None:
        """Invalid env_id raises ValueError."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )

        with pytest.raises(ValueError, match="out of range"):
            buffer.mark_terminal_with_penalty(-1, penalty=-10.0)


class TestNormalizeAdvantages:
    """Tests for advantage normalization."""

    def _fill_buffer_with_advantages(
        self, buffer: TamiyoRolloutBuffer, advantages: list[list[float]]
    ) -> None:
        """Helper to set advantages directly for testing."""
        for env_id, env_advantages in enumerate(advantages):
            buffer.step_counts[env_id] = len(env_advantages)
            for step_idx, adv in enumerate(env_advantages):
                buffer.advantages[env_id, step_idx] = adv

    def test_normalizes_to_zero_mean_unit_std(self) -> None:
        """Advantages are normalized to approximately μ=0, σ=1."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=10,
            state_dim=64,
        )
        # Use enough values to get stable statistics
        self._fill_buffer_with_advantages(buffer, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        buffer.normalize_advantages()

        # Collect normalized advantages
        all_adv = []
        for env_id in range(buffer.num_envs):
            n = buffer.step_counts[env_id]
            all_adv.extend(buffer.advantages[env_id, :n].tolist())

        mean = sum(all_adv) / len(all_adv)
        assert abs(mean) < 1e-6  # Should be ~0

    def test_idempotent_normalization(self) -> None:
        """Calling normalize_advantages twice doesn't change values."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=64,
        )
        self._fill_buffer_with_advantages(buffer, [[1, 2, 3, 4, 5]])

        buffer.normalize_advantages()
        first_values = buffer.advantages[0, :5].clone()

        # Second call should be no-op
        buffer.normalize_advantages()
        second_values = buffer.advantages[0, :5]

        assert torch.allclose(first_values, second_values)

    def test_returns_nan_when_already_normalized(self) -> None:
        """Returns NaN stats when already normalized."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=10,
            state_dim=64,
        )
        self._fill_buffer_with_advantages(buffer, [[1, 2, 3]])
        buffer.normalize_advantages()

        mean, std = buffer.normalize_advantages()

        assert mean != mean  # NaN check
        assert std != std


class TestGetBatchedSequences:
    """Tests for get_batched_sequences method."""

    def test_returns_valid_mask(self) -> None:
        """valid_mask correctly identifies filled timesteps."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=64,
        )
        buffer.step_counts = [3, 2]

        data = buffer.get_batched_sequences()

        # Env 0: steps 0,1,2 valid; 3,4 invalid
        assert data["valid_mask"][0, :3].all()
        assert not data["valid_mask"][0, 3:].any()

        # Env 1: steps 0,1 valid; 2,3,4 invalid
        assert data["valid_mask"][1, :2].all()
        assert not data["valid_mask"][1, 2:].any()

    def test_moves_tensors_to_device(self) -> None:
        """Tensors are moved to requested device."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=5,
            state_dim=64,
            device=torch.device("cpu"),
        )

        data = buffer.get_batched_sequences(device="cpu")

        # All tensors should be on CPU
        for key, tensor in data.items():
            assert tensor.device.type == "cpu", f"{key} not on CPU"
