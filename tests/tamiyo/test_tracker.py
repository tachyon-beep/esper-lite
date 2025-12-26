"""Tests for TAMIYO_INITIATED telemetry event emission.

When host training stabilizes (germination becomes allowed), the tracker should
emit a TAMIYO_INITIATED event with relevant context about the stabilization.
"""


from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.tamiyo.tracker import SignalTracker


class TestTamiyoInitiatedEvent:
    """Tests for TAMIYO_INITIATED event emission on stabilization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.events: list[TelemetryEvent] = []

        def capture_event(event: TelemetryEvent):
            self.events.append(event)

        # Mock the global hub
        from esper.nissa import output

        class MockHub:
            def __init__(self, capture_fn):
                self.capture_fn = capture_fn

            def emit(self, event: TelemetryEvent):
                self.capture_fn(event)

        # Store original global hub
        self._original_global_hub = output._global_hub
        # Replace with mock
        output._global_hub = MockHub(capture_event)

    def teardown_method(self):
        """Restore original hub."""
        from esper.nissa import output
        output._global_hub = self._original_global_hub

    def test_no_event_during_explosive_growth(self):
        """Should not emit TAMIYO_INITIATED during explosive growth phase."""
        tracker = SignalTracker(env_id=42)

        # Explosive growth: 20% improvement per epoch (way above 3% threshold)
        losses = [2.0, 1.6, 1.28, 1.02, 0.82]

        for i, loss in enumerate(losses):
            tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=loss,
                train_accuracy=50.0 + i * 5,
                val_loss=loss,
                val_accuracy=50.0 + i * 5,
                active_seeds=[],
            )

        # Should NOT have emitted TAMIYO_INITIATED
        tamiyo_events = [e for e in self.events
                         if e.event_type == TelemetryEventType.TAMIYO_INITIATED]
        assert len(tamiyo_events) == 0

    def test_emits_event_on_stabilization(self):
        """Should emit TAMIYO_INITIATED when stabilization is triggered."""
        tracker = SignalTracker(env_id=42, stabilization_epochs=3)

        # First, a few epochs of moderate improvement (above threshold)
        losses = [2.0, 1.8, 1.6]
        for i, loss in enumerate(losses):
            tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=loss,
                train_accuracy=50.0,
                val_loss=loss,
                val_accuracy=50.0,
                active_seeds=[],
            )

        assert not tracker.is_stabilized
        assert len(self.events) == 0

        # Now consecutive stable epochs (< 3% improvement)
        # From 1.6, improvements of 2%, 1.5%, 1%
        stable_losses = [1.568, 1.544, 1.529]
        for i, loss in enumerate(stable_losses):
            tracker.update(
                epoch=len(losses) + i,
                global_step=(len(losses) + i) * 100,
                train_loss=loss,
                train_accuracy=55.0,
                val_loss=loss,
                val_accuracy=55.0,
                active_seeds=[],
            )

        # Should have emitted exactly ONE TAMIYO_INITIATED event
        tamiyo_events = [e for e in self.events
                         if e.event_type == TelemetryEventType.TAMIYO_INITIATED]
        assert len(tamiyo_events) == 1

    def test_event_contains_correct_data(self):
        """TAMIYO_INITIATED event should contain all required fields."""
        tracker = SignalTracker(env_id=42, stabilization_epochs=3)

        # Fast path to stabilization - set up 2 stable epochs
        tracker._prev_loss = 1.0
        tracker._stable_count = 2

        # One more stable epoch triggers stabilization at epoch 10
        final_loss = 0.98
        tracker.update(
            epoch=10,
            global_step=1000,
            train_loss=final_loss,
            train_accuracy=60.0,
            val_loss=final_loss,
            val_accuracy=60.0,
            active_seeds=[],
        )

        # Find the TAMIYO_INITIATED event
        tamiyo_events = [e for e in self.events
                         if e.event_type == TelemetryEventType.TAMIYO_INITIATED]
        assert len(tamiyo_events) == 1

        event = tamiyo_events[0]

        # Check event type
        assert event.event_type == TelemetryEventType.TAMIYO_INITIATED

        # Check epoch field
        assert event.epoch == 10

        # Check data fields (typed payload: TamiyoInitiatedPayload)
        assert event.data.env_id == 42
        assert event.data.epoch == 10
        assert event.data.stable_count == 3  # After update, should be 3
        assert event.data.stabilization_epochs == 3
        assert event.data.val_loss == final_loss

        # Check message
        assert "Host stabilized" in event.message
        assert "germination now allowed" in event.message

    def test_event_emitted_only_once(self):
        """Should only emit TAMIYO_INITIATED once due to latch behavior."""
        tracker = SignalTracker(env_id=42, stabilization_epochs=3)

        # Fast path to stabilization
        tracker._prev_loss = 1.0
        tracker._stable_count = 2

        # Trigger stabilization
        tracker.update(
            epoch=10, global_step=1000,
            train_loss=0.98, train_accuracy=60.0,
            val_loss=0.98, val_accuracy=60.0,
            active_seeds=[],
        )

        # Should have emitted event
        tamiyo_events = [e for e in self.events
                         if e.event_type == TelemetryEventType.TAMIYO_INITIATED]
        assert len(tamiyo_events) == 1

        # Additional epochs should NOT emit more events
        for i in range(5):
            tracker.update(
                epoch=11 + i, global_step=1100 + i * 100,
                train_loss=0.9, train_accuracy=65.0,
                val_loss=0.9, val_accuracy=65.0,
                active_seeds=[],
            )

        # Still only one event
        tamiyo_events = [e for e in self.events
                         if e.event_type == TelemetryEventType.TAMIYO_INITIATED]
        assert len(tamiyo_events) == 1

    def test_no_env_id_skips_event(self):
        """Tracker without env_id should NOT emit event (typed payload requires env_id)."""
        tracker = SignalTracker(stabilization_epochs=3)

        # Fast path to stabilization
        tracker._prev_loss = 1.0
        tracker._stable_count = 2

        tracker.update(
            epoch=5, global_step=500,
            train_loss=0.98, train_accuracy=60.0,
            val_loss=0.98, val_accuracy=60.0,
            active_seeds=[],
        )

        # Should NOT emit event when env_id is None (typed payload requires it)
        tamiyo_events = [e for e in self.events
                         if e.event_type == TelemetryEventType.TAMIYO_INITIATED]
        assert len(tamiyo_events) == 0

        # But stabilization should still be recorded internally
        assert tracker._is_stabilized is True
