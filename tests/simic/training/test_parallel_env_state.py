"""Unit tests for ParallelEnvState dataclass.

Tests state container initialization, accumulator management, and reset behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from esper.leyline import LifecycleOp
from esper.simic.training.parallel_env_state import ParallelEnvState


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock MorphogeneticModel."""
    return MagicMock()


@pytest.fixture
def mock_optimizer() -> MagicMock:
    """Create a mock optimizer."""
    return MagicMock()


@pytest.fixture
def mock_signal_tracker() -> MagicMock:
    """Create a mock SignalTracker."""
    tracker = MagicMock()
    tracker.reset = MagicMock()
    return tracker


@pytest.fixture
def mock_governor() -> MagicMock:
    """Create a mock TolariaGovernor."""
    governor = MagicMock()
    governor.reset = MagicMock()
    return governor


@pytest.fixture
def env_state(
    mock_model: MagicMock,
    mock_optimizer: MagicMock,
    mock_signal_tracker: MagicMock,
    mock_governor: MagicMock,
) -> ParallelEnvState:
    """Create a ParallelEnvState for testing."""
    return ParallelEnvState(
        model=mock_model,
        host_optimizer=mock_optimizer,
        signal_tracker=mock_signal_tracker,
        governor=mock_governor,
        env_device="cpu",
    )


class TestParallelEnvStateInit:
    """Tests for ParallelEnvState initialization."""

    def test_action_counts_initialized(self, env_state: ParallelEnvState) -> None:
        """action_counts initialized with LifecycleOp names."""
        for op in LifecycleOp:
            assert op.name in env_state.action_counts
            assert env_state.action_counts[op.name] == 0

    def test_successful_action_counts_initialized(
        self, env_state: ParallelEnvState
    ) -> None:
        """successful_action_counts initialized with LifecycleOp names."""
        for op in LifecycleOp:
            assert op.name in env_state.successful_action_counts
            assert env_state.successful_action_counts[op.name] == 0

    def test_last_action_defaults(self, env_state: ParallelEnvState) -> None:
        """Obs V3 action feedback starts with neutral values."""
        assert env_state.last_action_success is True
        assert env_state.last_action_op == LifecycleOp.WAIT.value


class TestInitAccumulators:
    """Tests for accumulator initialization."""

    def test_creates_train_accumulators(self, env_state: ParallelEnvState) -> None:
        """init_accumulators creates train loss/correct accumulators."""
        env_state.init_accumulators(["slot_0", "slot_1"])

        assert env_state.train_loss_accum is not None
        assert env_state.train_correct_accum is not None
        assert env_state.train_loss_accum.device.type == "cpu"

    def test_creates_val_accumulators(self, env_state: ParallelEnvState) -> None:
        """init_accumulators creates val loss/correct accumulators."""
        env_state.init_accumulators(["slot_0"])

        assert env_state.val_loss_accum is not None
        assert env_state.val_correct_accum is not None

    def test_creates_per_slot_counterfactual_accumulators(
        self, env_state: ParallelEnvState
    ) -> None:
        """init_accumulators creates per-slot counterfactual tracking."""
        slots = ["slot_0", "slot_1", "slot_2"]
        env_state.init_accumulators(slots)

        for slot_id in slots:
            assert slot_id in env_state.cf_correct_accums
            assert slot_id in env_state.cf_totals
            assert env_state.cf_totals[slot_id] == 0

    def test_creates_pair_accumulators_for_3_slots(
        self, env_state: ParallelEnvState
    ) -> None:
        """init_accumulators creates pair accumulators for 3 slots."""
        slots = ["slot_0", "slot_1", "slot_2"]
        env_state.init_accumulators(slots)

        # Should have C(3,2) = 3 pairs
        expected_pairs = [(0, 1), (0, 2), (1, 2)]
        for pair in expected_pairs:
            assert pair in env_state.cf_pair_accums
            assert pair in env_state.cf_pair_totals

    def test_creates_pair_accumulators_for_4_slots(
        self, env_state: ParallelEnvState
    ) -> None:
        """init_accumulators creates pair accumulators for 4 slots."""
        slots = ["slot_0", "slot_1", "slot_2", "slot_3"]
        env_state.init_accumulators(slots)

        # Should have C(4,2) = 6 pairs
        assert len(env_state.cf_pair_accums) == 6

    def test_no_pair_accumulators_for_2_slots(
        self, env_state: ParallelEnvState
    ) -> None:
        """init_accumulators doesn't create pair accumulators for 2 slots."""
        slots = ["slot_0", "slot_1"]
        env_state.init_accumulators(slots)

        # Only create pairs for 3-4 slots
        assert len(env_state.cf_pair_accums) == 0


class TestZeroAccumulators:
    """Tests for zeroing accumulators."""

    def test_zeros_train_accumulators(self, env_state: ParallelEnvState) -> None:
        """zero_accumulators zeros train loss/correct."""
        env_state.init_accumulators(["slot_0"])
        env_state.train_loss_accum.fill_(5.0)  # type: ignore
        env_state.train_correct_accum.fill_(10.0)  # type: ignore

        env_state.zero_accumulators()

        assert env_state.train_loss_accum.item() == 0.0  # type: ignore
        assert env_state.train_correct_accum.item() == 0.0  # type: ignore

    def test_zeros_val_accumulators(self, env_state: ParallelEnvState) -> None:
        """zero_accumulators zeros val loss/correct."""
        env_state.init_accumulators(["slot_0"])
        env_state.val_loss_accum.fill_(5.0)  # type: ignore

        env_state.zero_accumulators()

        assert env_state.val_loss_accum.item() == 0.0  # type: ignore

    def test_zeros_counterfactual_totals(self, env_state: ParallelEnvState) -> None:
        """zero_accumulators zeros counterfactual sample counts."""
        slots = ["slot_0", "slot_1"]
        env_state.init_accumulators(slots)
        env_state.cf_totals["slot_0"] = 100
        env_state.cf_totals["slot_1"] = 50

        env_state.zero_accumulators()

        assert env_state.cf_totals["slot_0"] == 0
        assert env_state.cf_totals["slot_1"] == 0


class TestResetEpisodeState:
    """Tests for episode state reset."""

    def test_resets_seed_counters(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state zeros seed creation/fossilization counts."""
        env_state.seeds_created = 5
        env_state.seeds_fossilized = 3
        env_state.contributing_fossilized = 2

        env_state.reset_episode_state(["slot_0"])

        assert env_state.seeds_created == 0
        assert env_state.seeds_fossilized == 0
        assert env_state.contributing_fossilized == 0

    def test_clears_episode_rewards(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state clears episode reward history."""
        env_state.episode_rewards = [1.0, 2.0, 3.0]

        env_state.reset_episode_state(["slot_0"])

        assert env_state.episode_rewards == []

    def test_resets_action_counts(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state resets all action counts to zero."""
        env_state.action_counts["GERMINATE"] = 10
        env_state.successful_action_counts["PRUNE"] = 5

        env_state.reset_episode_state(["slot_0"])

        for op in LifecycleOp:
            assert env_state.action_counts[op.name] == 0
            assert env_state.successful_action_counts[op.name] == 0

    def test_clears_seed_optimizers(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state clears seed optimizer dict."""
        env_state.seed_optimizers["slot_0"] = MagicMock()

        env_state.reset_episode_state(["slot_0"])

        assert env_state.seed_optimizers == {}

    def test_resets_obs_v3_tracking(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state resets Obs V3 action feedback."""
        env_state.last_action_success = False
        env_state.last_action_op = LifecycleOp.PRUNE.value

        env_state.reset_episode_state(["slot_0"])

        assert env_state.last_action_success is True
        assert env_state.last_action_op == LifecycleOp.WAIT.value

    def test_clears_gradient_health_prev(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state clears per-slot gradient health tracking."""
        env_state.gradient_health_prev = {"slot_0": 0.5, "slot_1": 0.8}

        env_state.reset_episode_state(["slot_0", "slot_1"])

        assert env_state.gradient_health_prev == {}

    def test_calls_tracker_reset(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state calls signal_tracker.reset()."""
        env_state.reset_episode_state(["slot_0"])

        env_state.signal_tracker.reset.assert_called_once()

    def test_calls_governor_reset(self, env_state: ParallelEnvState) -> None:
        """reset_episode_state calls governor.reset()."""
        env_state.reset_episode_state(["slot_0"])

        env_state.governor.reset.assert_called_once()

    def test_initializes_accumulators_if_missing(
        self, env_state: ParallelEnvState
    ) -> None:
        """reset_episode_state creates accumulators if not present."""
        assert env_state.train_loss_accum is None

        env_state.reset_episode_state(["slot_0"])

        assert env_state.train_loss_accum is not None


class TestObsV3SlotTracking:
    """Tests for Obs V3 per-slot tracking methods."""

    def test_init_obs_v3_slot_tracking(self, env_state: ParallelEnvState) -> None:
        """init_obs_v3_slot_tracking sets default values."""
        env_state.init_obs_v3_slot_tracking("slot_0")

        assert env_state.gradient_health_prev["slot_0"] == 1.0
        assert env_state.epochs_since_counterfactual["slot_0"] == 0

    def test_clear_obs_v3_slot_tracking(self, env_state: ParallelEnvState) -> None:
        """clear_obs_v3_slot_tracking removes slot entries."""
        env_state.gradient_health_prev = {"slot_0": 0.5}
        env_state.epochs_since_counterfactual = {"slot_0": 3}

        env_state.clear_obs_v3_slot_tracking("slot_0")

        assert "slot_0" not in env_state.gradient_health_prev
        assert "slot_0" not in env_state.epochs_since_counterfactual

    def test_clear_obs_v3_slot_tracking_missing_key(
        self, env_state: ParallelEnvState
    ) -> None:
        """clear_obs_v3_slot_tracking handles missing keys gracefully."""
        # Should not raise
        env_state.clear_obs_v3_slot_tracking("nonexistent_slot")
