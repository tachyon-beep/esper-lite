"""End-to-end tests for seed lifecycle telemetry metrics (TELE-500 to TELE-599).

Verifies seed germination, fossilization, and pruning events flow correctly
from SeedSlot/aggregator through to schema. Tests cover:

- TELE-500: total_slots
- TELE-501: slot_stage_counts
- TELE-502: active_count
- TELE-503: fossilize_count
- TELE-504: prune_count
- TELE-505: germination_count
- TELE-506: germination_rate
- TELE-507: prune_rate
- TELE-508: fossilize_rate
- TELE-509: germination_trend
- TELE-510: prune_trend
- TELE-511: fossilize_trend
- TELE-512: blend_success_rate
- TELE-513: avg_lifespan_epochs
- TELE-514: has_exploding
- TELE-515: has_vanishing

These test the typed payload dataclasses:
- SeedGerminatedPayload
- SeedFossilizedPayload
- SeedPrunedPayload
- SeedStageChangedPayload
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from esper.leyline import (
    SeedStage,
    TelemetryEvent,
    TelemetryEventType,
    SeedGerminatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    SeedStageChangedPayload,
)
from esper.kasmina.slot import SeedSlot
from esper.karn.sanctum.aggregator import SanctumAggregator, detect_rate_trend
from esper.karn.sanctum.schema import SeedLifecycleStats


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def capture_events():
    """Capture telemetry events from a SeedSlot."""
    events: list[TelemetryEvent] = []

    def capture(event: TelemetryEvent):
        events.append(event)

    return events, capture


@pytest.fixture
def slot_with_capture(capture_events):
    """Create a SeedSlot that captures telemetry."""
    events, capture = capture_events
    slot = SeedSlot(
        slot_id="test_slot_0",
        channels=64,
        device="cpu",
        on_telemetry=capture,
        fast_mode=False,
    )
    return slot, events


@pytest.fixture
def aggregator():
    """Create a fresh SanctumAggregator."""
    return SanctumAggregator(num_envs=4)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def make_germinated_event(
    slot_id: str = "r0c0",
    env_id: int = 0,
    blueprint_id: str = "depthwise",
    params: int = 1000,
    has_vanishing: bool = False,
    has_exploding: bool = False,
    grad_ratio: float = 0.0,
) -> MagicMock:
    """Create a mock SEED_GERMINATED event."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "SEED_GERMINATED"
    event.timestamp = datetime.now(timezone.utc)
    event.slot_id = slot_id
    event.data = SeedGerminatedPayload(
        slot_id=slot_id,
        env_id=env_id,
        blueprint_id=blueprint_id,
        params=params,
        alpha=0.0,
        blend_tempo_epochs=5,
        alpha_curve="LINEAR",
        grad_ratio=grad_ratio,
        has_vanishing=has_vanishing,
        has_exploding=has_exploding,
        epochs_in_stage=0,
    )
    return event


def make_fossilized_event(
    slot_id: str = "r0c0",
    env_id: int = 0,
    blueprint_id: str = "depthwise",
    improvement: float = 5.0,
    params_added: int = 5000,
    epochs_total: int = 10,
    counterfactual: float = 2.5,
) -> MagicMock:
    """Create a mock SEED_FOSSILIZED event."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "SEED_FOSSILIZED"
    event.timestamp = datetime.now(timezone.utc)
    event.slot_id = slot_id
    event.data = SeedFossilizedPayload(
        slot_id=slot_id,
        env_id=env_id,
        blueprint_id=blueprint_id,
        improvement=improvement,
        params_added=params_added,
        alpha=1.0,
        epochs_total=epochs_total,
        counterfactual=counterfactual,
    )
    return event


def make_pruned_event(
    slot_id: str = "r0c0",
    env_id: int = 0,
    blueprint_id: str = "depthwise",
    reason: str = "no_improvement",
    improvement: float = -0.5,
    auto_pruned: bool = False,
    epochs_total: int = 5,
    initiator: str = "policy",
) -> MagicMock:
    """Create a mock SEED_PRUNED event."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "SEED_PRUNED"
    event.timestamp = datetime.now(timezone.utc)
    event.slot_id = slot_id
    event.data = SeedPrunedPayload(
        slot_id=slot_id,
        env_id=env_id,
        reason=reason,
        blueprint_id=blueprint_id,
        improvement=improvement,
        auto_pruned=auto_pruned,
        epochs_total=epochs_total,
        counterfactual=0.0,
        initiator=initiator,
    )
    return event


def make_stage_changed_event(
    slot_id: str = "r0c0",
    env_id: int = 0,
    from_stage: str = "GERMINATED",
    to_stage: str = "TRAINING",
    has_vanishing: bool = False,
    has_exploding: bool = False,
    grad_ratio: float = 0.0,
    epochs_in_stage: int = 0,
) -> MagicMock:
    """Create a mock SEED_STAGE_CHANGED event."""
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "SEED_STAGE_CHANGED"
    event.timestamp = datetime.now(timezone.utc)
    event.slot_id = slot_id
    event.data = SeedStageChangedPayload(
        slot_id=slot_id,
        env_id=env_id,
        from_stage=from_stage,
        to_stage=to_stage,
        alpha=0.0,
        accuracy_delta=0.0,
        epochs_in_stage=epochs_in_stage,
        alpha_curve="LINEAR",
        grad_ratio=grad_ratio,
        has_vanishing=has_vanishing,
        has_exploding=has_exploding,
    )
    return event


# -----------------------------------------------------------------------------
# TELE-500: total_slots
# -----------------------------------------------------------------------------


class TestTELE500TotalSlots:
    """TELE-500: Total number of seed slots in the system."""

    def test_total_slots_initialized_on_training_started(self, aggregator):
        """TELE-500: total_slots is set from TRAINING_STARTED event."""
        from esper.leyline.telemetry import TrainingStartedPayload

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = TrainingStartedPayload(
            episode_id="test-run",
            task="mnist",
            max_epochs=50,
            max_batches=100,
            n_envs=4,
            host_params=1000000,
            slot_ids=("r0c0", "r0c1", "r1c0", "r1c1"),  # 4 slots per env
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=100000,
            policy_device="cpu",
            env_devices=("cpu",),
            reward_mode="shaped",
        )

        aggregator.process_event(event)
        snapshot = aggregator.get_snapshot()

        # total_slots = n_envs * len(slot_ids) = 4 * 4 = 16
        # (each environment has its own instance of each slot)
        assert snapshot.seed_lifecycle.total_slots == 16

    def test_total_slots_inferred_from_germination_events(self, aggregator):
        """TELE-500: total_slots tracks distinct slot_ids from germination events."""
        # Germinate in different slots
        for i, slot_id in enumerate(["r0c0", "r0c1", "r1c0"]):
            event = make_germinated_event(slot_id=slot_id, env_id=0)
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        # Should have discovered 3 slots
        assert len(snapshot.slot_ids) >= 3


# -----------------------------------------------------------------------------
# TELE-501: slot_stage_counts
# -----------------------------------------------------------------------------


class TestTELE501SlotStageCounts:
    """TELE-501: Count of slots in each lifecycle stage."""

    def test_stage_counts_after_germination(self, aggregator):
        """TELE-501: Germinated seeds are counted correctly."""
        event = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        env = snapshot.envs[0]

        assert "r0c0" in env.seeds
        assert env.seeds["r0c0"].stage == "GERMINATED"

    def test_stage_counts_after_fossilization(self, aggregator):
        """TELE-501: Fossilized seeds show correct stage."""
        # First germinate
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        # Then fossilize
        foss = make_fossilized_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(foss)

        snapshot = aggregator.get_snapshot()
        env = snapshot.envs[0]

        assert env.seeds["r0c0"].stage == "FOSSILIZED"


# -----------------------------------------------------------------------------
# TELE-502: active_count
# -----------------------------------------------------------------------------


class TestTELE502ActiveCount:
    """TELE-502: Count of currently active (non-dormant, non-terminal) seeds."""

    def test_active_count_increments_on_germination(self, aggregator):
        """TELE-502: active_count increases when seeds germinate."""
        event = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        # Environment-level active count
        assert snapshot.envs[0].active_seed_count == 1

    def test_active_count_decrements_on_fossilize(self, aggregator):
        """TELE-502: active_count decreases when seeds fossilize."""
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        foss = make_fossilized_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(foss)

        snapshot = aggregator.get_snapshot()
        assert snapshot.envs[0].active_seed_count == 0

    def test_active_count_decrements_on_prune(self, aggregator):
        """TELE-502: active_count decreases when seeds are pruned."""
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        prune = make_pruned_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(prune)

        snapshot = aggregator.get_snapshot()
        assert snapshot.envs[0].active_seed_count == 0


# -----------------------------------------------------------------------------
# TELE-503: fossilize_count
# -----------------------------------------------------------------------------


class TestTELE503FossilizeCount:
    """TELE-503: Cumulative count of fossilized seeds."""

    def test_fossilize_count_increments(self, aggregator):
        """TELE-503: fossilize_count increments on SEED_FOSSILIZED."""
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        foss = make_fossilized_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(foss)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.fossilize_count == 1
        # Also check env-level
        assert snapshot.envs[0].fossilized_count == 1

    def test_fossilize_count_multiple_seeds(self, aggregator):
        """TELE-503: fossilize_count accumulates across multiple seeds."""
        for i, slot_id in enumerate(["r0c0", "r0c1", "r1c0"]):
            germ = make_germinated_event(slot_id=slot_id, env_id=0)
            aggregator.process_event(germ)
            foss = make_fossilized_event(slot_id=slot_id, env_id=0)
            aggregator.process_event(foss)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.fossilize_count == 3


# -----------------------------------------------------------------------------
# TELE-504: prune_count
# -----------------------------------------------------------------------------


class TestTELE504PruneCount:
    """TELE-504: Cumulative count of pruned seeds."""

    def test_prune_count_increments(self, aggregator):
        """TELE-504: prune_count increments on SEED_PRUNED."""
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        prune = make_pruned_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(prune)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.prune_count == 1

    def test_prune_count_tracks_reason(self, aggregator):
        """TELE-504: Pruned seeds include reason in SeedState."""
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        prune = make_pruned_event(
            slot_id="r0c0",
            env_id=0,
            reason="gradient_explosion",
            auto_pruned=True,
        )
        aggregator.process_event(prune)

        snapshot = aggregator.get_snapshot()
        seed_state = snapshot.envs[0].seeds["r0c0"]
        assert seed_state.prune_reason == "gradient_explosion"
        assert seed_state.auto_pruned is True


# -----------------------------------------------------------------------------
# TELE-505: germination_count
# -----------------------------------------------------------------------------


class TestTELE505GerminationCount:
    """TELE-505: Cumulative count of germinated seeds."""

    def test_germination_count_increments(self, aggregator):
        """TELE-505: germination_count increments on SEED_GERMINATED."""
        event = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.germination_count == 1

    def test_germination_count_multiple(self, aggregator):
        """TELE-505: germination_count accumulates correctly."""
        for i, slot_id in enumerate(["r0c0", "r0c1", "r1c0", "r1c1"]):
            event = make_germinated_event(slot_id=slot_id, env_id=i % 4)
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.germination_count == 4


# -----------------------------------------------------------------------------
# TELE-506/507/508: Rate calculations
# -----------------------------------------------------------------------------


class TestTELE506GerminationRate:
    """TELE-506: Germination rate (germinations per episode)."""

    def test_germination_rate_computed_correctly(self, aggregator):
        """TELE-506: germination_rate = germination_count / current_episode."""
        # Simulate some episodes
        aggregator._current_episode = 10

        # Germinate 5 seeds
        for i in range(5):
            event = make_germinated_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        # 5 germinations / 10 episodes = 0.5
        assert snapshot.seed_lifecycle.germination_rate == pytest.approx(0.5)


class TestTELE507PruneRate:
    """TELE-507: Prune rate (prunes per episode)."""

    def test_prune_rate_computed_correctly(self, aggregator):
        """TELE-507: prune_rate = prune_count / current_episode."""
        aggregator._current_episode = 20

        # Germinate and prune 4 seeds
        for i in range(4):
            germ = make_germinated_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(germ)
            prune = make_pruned_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(prune)

        snapshot = aggregator.get_snapshot()
        # 4 prunes / 20 episodes = 0.2
        assert snapshot.seed_lifecycle.prune_rate == pytest.approx(0.2)


class TestTELE508FossilizeRate:
    """TELE-508: Fossilize rate (fossilizations per episode)."""

    def test_fossilize_rate_computed_correctly(self, aggregator):
        """TELE-508: fossilize_rate = fossilize_count / current_episode."""
        aggregator._current_episode = 50

        # Germinate and fossilize 10 seeds
        for i in range(10):
            germ = make_germinated_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(germ)
            foss = make_fossilized_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(foss)

        snapshot = aggregator.get_snapshot()
        # 10 fossilizations / 50 episodes = 0.2
        assert snapshot.seed_lifecycle.fossilize_rate == pytest.approx(0.2)


# -----------------------------------------------------------------------------
# TELE-509/510/511: Trend detection
# -----------------------------------------------------------------------------


class TestTELE509GerminationTrend:
    """TELE-509: Germination trend (rising/stable/falling)."""

    def test_detect_rate_trend_stable_insufficient_samples(self):
        """TELE-509: Trend is stable with insufficient samples."""
        from collections import deque

        history: deque[float] = deque(maxlen=20)
        # Only 3 samples - not enough for trend
        for val in [0.5, 0.5, 0.5]:
            history.append(val)

        assert detect_rate_trend(history) == "stable"

    def test_detect_rate_trend_rising(self):
        """TELE-509: Trend is rising when recent mean > older mean by 20%."""
        from collections import deque

        history: deque[float] = deque(maxlen=20)
        # Older samples (5): 1.0 average
        for _ in range(5):
            history.append(1.0)
        # Recent samples (5): 1.5 average (50% increase)
        for _ in range(5):
            history.append(1.5)

        assert detect_rate_trend(history) == "rising"

    def test_detect_rate_trend_falling(self):
        """TELE-509: Trend is falling when recent mean < older mean by 20%."""
        from collections import deque

        history: deque[float] = deque(maxlen=20)
        # Older samples (5): 1.0 average
        for _ in range(5):
            history.append(1.0)
        # Recent samples (5): 0.5 average (50% decrease)
        for _ in range(5):
            history.append(0.5)

        assert detect_rate_trend(history) == "falling"

    def test_detect_rate_trend_stable_within_threshold(self):
        """TELE-509: Trend is stable when change is within 20%."""
        from collections import deque

        history: deque[float] = deque(maxlen=20)
        # Older samples (5): 1.0 average
        for _ in range(5):
            history.append(1.0)
        # Recent samples (5): 1.1 average (10% increase - within threshold)
        for _ in range(5):
            history.append(1.1)

        assert detect_rate_trend(history) == "stable"


class TestTELE510PruneTrend:
    """TELE-510: Prune trend (rising/stable/falling)."""

    def test_prune_trend_uses_same_algorithm(self):
        """TELE-510: Prune trend uses detect_rate_trend()."""
        # This is verified by implementation - prune_trend uses
        # detect_rate_trend(self._prune_rate_history)
        from collections import deque

        history: deque[float] = deque(maxlen=20)
        # Rising pattern
        for _ in range(5):
            history.append(0.1)
        for _ in range(5):
            history.append(0.3)

        assert detect_rate_trend(history) == "rising"


class TestTELE511FossilizeTrend:
    """TELE-511: Fossilize trend (rising/stable/falling)."""

    def test_fossilize_trend_uses_same_algorithm(self):
        """TELE-511: Fossilize trend uses detect_rate_trend()."""
        from collections import deque

        history: deque[float] = deque(maxlen=20)
        # Falling pattern
        for _ in range(5):
            history.append(0.5)
        for _ in range(5):
            history.append(0.2)

        assert detect_rate_trend(history) == "falling"


# -----------------------------------------------------------------------------
# TELE-512: blend_success_rate
# -----------------------------------------------------------------------------


class TestTELE512BlendSuccessRate:
    """TELE-512: Blend success rate (fossilized / (fossilized + pruned))."""

    def test_blend_success_rate_zero_when_no_terminal_outcomes(self, aggregator):
        """TELE-512: blend_success_rate is 0.0 when no seeds have terminated."""
        # Only germinate, don't terminate
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.blend_success_rate == 0.0

    def test_blend_success_rate_100_percent_all_fossilized(self, aggregator):
        """TELE-512: blend_success_rate is 1.0 when all seeds fossilize."""
        for i in range(5):
            germ = make_germinated_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(germ)
            foss = make_fossilized_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(foss)

        snapshot = aggregator.get_snapshot()
        # 5 fossilized / (5 fossilized + 0 pruned) = 1.0
        assert snapshot.seed_lifecycle.blend_success_rate == pytest.approx(1.0)

    def test_blend_success_rate_zero_percent_all_pruned(self, aggregator):
        """TELE-512: blend_success_rate is 0.0 when all seeds are pruned."""
        for i in range(5):
            germ = make_germinated_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(germ)
            prune = make_pruned_event(slot_id=f"r{i}c0", env_id=i % 4)
            aggregator.process_event(prune)

        snapshot = aggregator.get_snapshot()
        # 0 fossilized / (0 fossilized + 5 pruned) = 0.0
        assert snapshot.seed_lifecycle.blend_success_rate == pytest.approx(0.0)

    def test_blend_success_rate_mixed_outcomes(self, aggregator):
        """TELE-512: blend_success_rate is correct with mixed outcomes."""
        # 3 fossilized
        for i in range(3):
            germ = make_germinated_event(slot_id=f"r{i}c0", env_id=0)
            aggregator.process_event(germ)
            foss = make_fossilized_event(slot_id=f"r{i}c0", env_id=0)
            aggregator.process_event(foss)

        # 2 pruned
        for i in range(3, 5):
            germ = make_germinated_event(slot_id=f"r{i}c0", env_id=0)
            aggregator.process_event(germ)
            prune = make_pruned_event(slot_id=f"r{i}c0", env_id=0)
            aggregator.process_event(prune)

        snapshot = aggregator.get_snapshot()
        # 3 fossilized / (3 fossilized + 2 pruned) = 0.6
        assert snapshot.seed_lifecycle.blend_success_rate == pytest.approx(0.6)


# -----------------------------------------------------------------------------
# TELE-513: avg_lifespan_epochs
# -----------------------------------------------------------------------------


class TestTELE513AvgLifespanEpochs:
    """TELE-513: Average lifespan in epochs for terminated seeds."""

    def test_avg_lifespan_zero_before_terminations(self, aggregator):
        """TELE-513: avg_lifespan_epochs is 0.0 before any seeds terminate."""
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.avg_lifespan_epochs == 0.0

    def test_avg_lifespan_from_fossilized_seeds(self, aggregator):
        """TELE-513: avg_lifespan_epochs includes fossilized seed lifespans."""
        # Fossilize with epochs_total=10
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)
        foss = make_fossilized_event(slot_id="r0c0", env_id=0, epochs_total=10)
        aggregator.process_event(foss)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.avg_lifespan_epochs == pytest.approx(10.0)

    def test_avg_lifespan_from_pruned_seeds(self, aggregator):
        """TELE-513: avg_lifespan_epochs includes pruned seed lifespans."""
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)
        prune = make_pruned_event(slot_id="r0c0", env_id=0, epochs_total=5)
        aggregator.process_event(prune)

        snapshot = aggregator.get_snapshot()
        assert snapshot.seed_lifecycle.avg_lifespan_epochs == pytest.approx(5.0)

    def test_avg_lifespan_averages_multiple_seeds(self, aggregator):
        """TELE-513: avg_lifespan_epochs averages across multiple terminated seeds."""
        # Seed 1: 10 epochs then fossilize
        germ1 = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ1)
        foss1 = make_fossilized_event(slot_id="r0c0", env_id=0, epochs_total=10)
        aggregator.process_event(foss1)

        # Seed 2: 5 epochs then prune
        germ2 = make_germinated_event(slot_id="r0c1", env_id=0)
        aggregator.process_event(germ2)
        prune2 = make_pruned_event(slot_id="r0c1", env_id=0, epochs_total=5)
        aggregator.process_event(prune2)

        # Seed 3: 15 epochs then fossilize
        germ3 = make_germinated_event(slot_id="r1c0", env_id=0)
        aggregator.process_event(germ3)
        foss3 = make_fossilized_event(slot_id="r1c0", env_id=0, epochs_total=15)
        aggregator.process_event(foss3)

        snapshot = aggregator.get_snapshot()
        # (10 + 5 + 15) / 3 = 10.0
        assert snapshot.seed_lifecycle.avg_lifespan_epochs == pytest.approx(10.0)


# -----------------------------------------------------------------------------
# TELE-514: has_exploding
# -----------------------------------------------------------------------------


class TestTELE514HasExploding:
    """TELE-514: Flag indicating seed has exploding gradients."""

    def test_has_exploding_in_germinated_payload(self):
        """TELE-514: SeedGerminatedPayload includes has_exploding field."""
        payload = SeedGerminatedPayload(
            slot_id="r0c0",
            env_id=0,
            blueprint_id="depthwise",
            params=1000,
            alpha=0.0,
            blend_tempo_epochs=5,
            alpha_curve="LINEAR",
            grad_ratio=0.0,
            has_vanishing=False,
            has_exploding=True,  # Testing this field
            epochs_in_stage=0,
        )
        assert payload.has_exploding is True

    def test_has_exploding_in_stage_changed_payload(self):
        """TELE-514: SeedStageChangedPayload includes has_exploding field."""
        payload = SeedStageChangedPayload(
            slot_id="r0c0",
            env_id=0,
            from_stage="TRAINING",
            to_stage="BLENDING",
            alpha=0.5,
            accuracy_delta=2.0,
            epochs_in_stage=10,
            alpha_curve="LINEAR",
            grad_ratio=0.8,
            has_vanishing=False,
            has_exploding=True,  # Testing this field
        )
        assert payload.has_exploding is True

    def test_has_exploding_propagates_to_seed_state(self, aggregator):
        """TELE-514: has_exploding is captured in SeedState via stage change event."""
        # Germinate
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        # Stage change with has_exploding=True
        stage = make_stage_changed_event(
            slot_id="r0c0",
            env_id=0,
            from_stage="GERMINATED",
            to_stage="TRAINING",
            has_exploding=True,
        )
        aggregator.process_event(stage)

        snapshot = aggregator.get_snapshot()
        seed_state = snapshot.envs[0].seeds["r0c0"]
        assert seed_state.has_exploding is True


# -----------------------------------------------------------------------------
# TELE-515: has_vanishing
# -----------------------------------------------------------------------------


class TestTELE515HasVanishing:
    """TELE-515: Flag indicating seed has vanishing gradients."""

    def test_has_vanishing_in_germinated_payload(self):
        """TELE-515: SeedGerminatedPayload includes has_vanishing field."""
        payload = SeedGerminatedPayload(
            slot_id="r0c0",
            env_id=0,
            blueprint_id="depthwise",
            params=1000,
            alpha=0.0,
            blend_tempo_epochs=5,
            alpha_curve="LINEAR",
            grad_ratio=0.0,
            has_vanishing=True,  # Testing this field
            has_exploding=False,
            epochs_in_stage=0,
        )
        assert payload.has_vanishing is True

    def test_has_vanishing_in_stage_changed_payload(self):
        """TELE-515: SeedStageChangedPayload includes has_vanishing field."""
        payload = SeedStageChangedPayload(
            slot_id="r0c0",
            env_id=0,
            from_stage="TRAINING",
            to_stage="BLENDING",
            alpha=0.5,
            accuracy_delta=2.0,
            epochs_in_stage=10,
            alpha_curve="LINEAR",
            grad_ratio=0.8,
            has_vanishing=True,  # Testing this field
            has_exploding=False,
        )
        assert payload.has_vanishing is True

    def test_has_vanishing_propagates_to_seed_state(self, aggregator):
        """TELE-515: has_vanishing is captured in SeedState via stage change event."""
        # Germinate
        germ = make_germinated_event(slot_id="r0c0", env_id=0)
        aggregator.process_event(germ)

        # Stage change with has_vanishing=True
        stage = make_stage_changed_event(
            slot_id="r0c0",
            env_id=0,
            from_stage="GERMINATED",
            to_stage="TRAINING",
            has_vanishing=True,
        )
        aggregator.process_event(stage)

        snapshot = aggregator.get_snapshot()
        seed_state = snapshot.envs[0].seeds["r0c0"]
        assert seed_state.has_vanishing is True


# -----------------------------------------------------------------------------
# Integration tests - real SeedSlot emits events
# -----------------------------------------------------------------------------


class TestSeedSlotEmitsTypedPayloads:
    """Integration tests verifying SeedSlot emits correctly typed payloads."""

    def test_germinate_emits_seed_germinated_payload(self, slot_with_capture):
        """SeedSlot.germinate() emits SeedGerminatedPayload."""
        slot, events = slot_with_capture

        slot.germinate("depthwise", "test-seed-001")

        assert len(events) == 1
        event = events[0]
        assert event.event_type == TelemetryEventType.SEED_GERMINATED
        assert isinstance(event.data, SeedGerminatedPayload)
        assert event.data.blueprint_id == "depthwise"
        assert event.data.params > 0
        assert event.data.has_vanishing is False
        assert event.data.has_exploding is False

    def test_prune_emits_seed_pruned_payload(self, slot_with_capture):
        """SeedSlot.prune() emits SeedPrunedPayload."""
        slot, events = slot_with_capture

        slot.germinate("depthwise", "test-seed-001")
        events.clear()

        slot.prune("test_reason", initiator="policy")

        # Should emit SEED_STAGE_CHANGED and SEED_PRUNED
        prune_events = [
            e for e in events if e.event_type == TelemetryEventType.SEED_PRUNED
        ]
        assert len(prune_events) == 1

        event = prune_events[0]
        assert isinstance(event.data, SeedPrunedPayload)
        assert event.data.reason == "test_reason"
        assert event.data.blueprint_id == "depthwise"
        assert event.data.initiator == "policy"

    def test_fossilize_emits_seed_fossilized_payload(self, slot_with_capture):
        """SeedSlot.advance_stage() to FOSSILIZED emits SeedFossilizedPayload."""
        slot, events = slot_with_capture

        slot.germinate("depthwise", "test-seed-001")
        events.clear()

        # Set up state for fossilization
        slot.state.metrics.initial_val_accuracy = 70.0
        slot.state.metrics.current_val_accuracy = 75.0
        slot.state.metrics.counterfactual_contribution = 5.0
        slot.state.stage = SeedStage.HOLDING
        slot.state.is_healthy = True

        slot.advance_stage(SeedStage.FOSSILIZED)

        foss_events = [
            e for e in events if e.event_type == TelemetryEventType.SEED_FOSSILIZED
        ]
        assert len(foss_events) == 1

        event = foss_events[0]
        assert isinstance(event.data, SeedFossilizedPayload)
        assert event.data.blueprint_id == "depthwise"
        assert event.data.improvement == 5.0
        assert event.data.params_added > 0


# -----------------------------------------------------------------------------
# SeedLifecycleStats schema tests
# -----------------------------------------------------------------------------


class TestSeedLifecycleStatsSchema:
    """Verify SeedLifecycleStats schema is correct."""

    def test_default_values(self):
        """SeedLifecycleStats has correct defaults."""
        stats = SeedLifecycleStats()

        assert stats.germination_count == 0
        assert stats.prune_count == 0
        assert stats.fossilize_count == 0
        assert stats.active_count == 0
        assert stats.total_slots == 0
        assert stats.germination_rate == 0.0
        assert stats.prune_rate == 0.0
        assert stats.fossilize_rate == 0.0
        assert stats.blend_success_rate == 0.0
        assert stats.avg_lifespan_epochs == 0.0
        assert stats.germination_trend == "stable"
        assert stats.prune_trend == "stable"
        assert stats.fossilize_trend == "stable"

    def test_all_fields_settable(self):
        """SeedLifecycleStats fields can be set."""
        stats = SeedLifecycleStats(
            germination_count=10,
            prune_count=3,
            fossilize_count=7,
            active_count=2,
            total_slots=4,
            germination_rate=0.5,
            prune_rate=0.15,
            fossilize_rate=0.35,
            blend_success_rate=0.7,
            avg_lifespan_epochs=12.5,
            germination_trend="rising",
            prune_trend="falling",
            fossilize_trend="stable",
        )

        assert stats.germination_count == 10
        assert stats.prune_count == 3
        assert stats.fossilize_count == 7
        assert stats.active_count == 2
        assert stats.total_slots == 4
        assert stats.germination_rate == 0.5
        assert stats.prune_rate == 0.15
        assert stats.fossilize_rate == 0.35
        assert stats.blend_success_rate == 0.7
        assert stats.avg_lifespan_epochs == 12.5
        assert stats.germination_trend == "rising"
        assert stats.prune_trend == "falling"
        assert stats.fossilize_trend == "stable"
