"""Tests for alpha_curve propagation through aggregator."""

import pytest
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline.telemetry import (
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    TelemetryEvent,
)


class TestAggregatorAlphaCurve:
    """Test alpha_curve flows from payloads to SeedState."""

    def test_germinated_payload_sets_alpha_curve(self):
        """SEED_GERMINATED should copy alpha_curve to SeedState."""
        aggregator = SanctumAggregator()

        event = TelemetryEvent(
            event_type="SEED_GERMINATED",
            data=SeedGerminatedPayload(
                slot_id="slot_0",
                env_id=0,
                blueprint_id="conv_l",
                params=1000,
                alpha_curve="SIGMOID",
            ),
        )
        aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        env = snapshot.envs[0]
        assert env is not None
        seed = env.seeds.get("slot_0")
        assert seed is not None
        assert seed.alpha_curve == "SIGMOID"

    def test_stage_changed_payload_updates_alpha_curve(self):
        """SEED_STAGE_CHANGED should update alpha_curve."""
        aggregator = SanctumAggregator()

        # First germinate the seed
        germinate_event = TelemetryEvent(
            event_type="SEED_GERMINATED",
            data=SeedGerminatedPayload(
                slot_id="slot_0",
                env_id=0,
                blueprint_id="conv_l",
                params=1000,
                alpha_curve="LINEAR",
            ),
        )
        aggregator.process_event(germinate_event)

        # Then change stage with new curve
        stage_event = TelemetryEvent(
            event_type="SEED_STAGE_CHANGED",
            data=SeedStageChangedPayload(
                slot_id="slot_0",
                env_id=0,
                from_stage="TRAINING",
                to_stage="BLENDING",
                alpha_curve="SIGMOID_SHARP",
            ),
        )
        aggregator.process_event(stage_event)

        snapshot = aggregator.get_snapshot()
        env = snapshot.envs[0]
        seed = env.seeds.get("slot_0")
        assert seed.alpha_curve == "SIGMOID_SHARP"
