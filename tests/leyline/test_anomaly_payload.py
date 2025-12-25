"""Tests for AnomalyDetectedPayload typed payload."""

from esper.leyline import TelemetryEvent, TelemetryEventType


class TestAnomalyDetectedPayload:
    """Test suite for AnomalyDetectedPayload typed payload."""

    def test_payload_exists(self):
        """AnomalyDetectedPayload class exists in leyline."""
        from esper.leyline import AnomalyDetectedPayload

        assert AnomalyDetectedPayload is not None

    def test_minimal_payload_construction(self):
        """Can construct payload with only required fields."""
        from esper.leyline import AnomalyDetectedPayload

        payload = AnomalyDetectedPayload(
            anomaly_type="ratio_explosion",
            episode=10,
            batch=5,
            inner_epoch=25,
            total_episodes=100,
        )

        assert payload.anomaly_type == "ratio_explosion"
        assert payload.episode == 10
        assert payload.batch == 5
        assert payload.inner_epoch == 25
        assert payload.total_episodes == 100
        assert payload.detail == ""
        assert payload.gradient_stats is None
        assert payload.stability is None
        assert payload.ratio_diagnostic is None

    def test_full_payload_construction(self):
        """Can construct payload with all optional fields."""
        from esper.leyline import AnomalyDetectedPayload

        gradient_stats = (
            {"layer": "fc1", "norm": 1.5},
            {"layer": "fc2", "norm": 2.3},
        )
        stability = {"has_nan": False, "has_inf": False}
        ratio_diagnostic = {"mean": 1.2, "std": 0.3}

        payload = AnomalyDetectedPayload(
            anomaly_type="ratio_collapse",
            episode=15,
            batch=8,
            inner_epoch=25,
            total_episodes=100,
            detail="Ratio collapsed below 0.1 threshold",
            gradient_stats=gradient_stats,
            stability=stability,
            ratio_diagnostic=ratio_diagnostic,
        )

        assert payload.anomaly_type == "ratio_collapse"
        assert payload.detail == "Ratio collapsed below 0.1 threshold"
        assert payload.gradient_stats == gradient_stats
        assert payload.stability == stability
        assert payload.ratio_diagnostic == ratio_diagnostic

    def test_payload_is_frozen(self):
        """Payload is immutable (frozen=True)."""
        from esper.leyline import AnomalyDetectedPayload
        import pytest

        payload = AnomalyDetectedPayload(
            anomaly_type="value_collapse",
            episode=10,
            batch=5,
            inner_epoch=25,
            total_episodes=100,
        )

        with pytest.raises(AttributeError):
            payload.episode = 20  # type: ignore

    def test_from_dict_minimal(self):
        """from_dict works with minimal required fields."""
        from esper.leyline import AnomalyDetectedPayload

        data = {
            "anomaly_type": "numerical_instability",
            "episode": 12,
            "batch": 3,
            "inner_epoch": 25,
            "total_episodes": 100,
        }

        payload = AnomalyDetectedPayload.from_dict(data)

        assert payload.anomaly_type == "numerical_instability"
        assert payload.episode == 12
        assert payload.batch == 3
        assert payload.inner_epoch == 25
        assert payload.total_episodes == 100
        assert payload.detail == ""
        assert payload.gradient_stats is None

    def test_from_dict_full(self):
        """from_dict works with all fields."""
        from esper.leyline import AnomalyDetectedPayload

        data = {
            "anomaly_type": "gradient_pathology",
            "episode": 18,
            "batch": 7,
            "inner_epoch": 25,
            "total_episodes": 100,
            "detail": "Dead gradients detected in layer fc3",
            "gradient_stats": [
                {"layer": "fc1", "norm": 1.2},
                {"layer": "fc2", "norm": 0.8},
            ],
            "stability": {"has_nan": True, "has_inf": False},
            "ratio_diagnostic": {"mean": 0.05, "std": 0.01},
        }

        payload = AnomalyDetectedPayload.from_dict(data)

        assert payload.anomaly_type == "gradient_pathology"
        assert payload.detail == "Dead gradients detected in layer fc3"
        assert len(payload.gradient_stats) == 2  # type: ignore
        assert payload.gradient_stats[0]["layer"] == "fc1"  # type: ignore
        assert payload.stability["has_nan"] is True  # type: ignore

    def test_from_dict_tuple_conversion(self):
        """from_dict converts list gradient_stats to tuple."""
        from esper.leyline import AnomalyDetectedPayload

        data = {
            "anomaly_type": "ratio_explosion",
            "episode": 10,
            "batch": 5,
            "inner_epoch": 25,
            "total_episodes": 100,
            "gradient_stats": [{"layer": "fc1", "norm": 1.5}],  # list in JSON
        }

        payload = AnomalyDetectedPayload.from_dict(data)

        # Should be converted to tuple
        assert isinstance(payload.gradient_stats, tuple)
        assert len(payload.gradient_stats) == 1

    def test_from_dict_missing_required_field_raises(self):
        """from_dict raises KeyError on missing required field."""
        from esper.leyline import AnomalyDetectedPayload
        import pytest

        data = {
            "anomaly_type": "ratio_explosion",
            "episode": 10,
            # Missing: batch, inner_epoch, total_episodes
        }

        with pytest.raises(KeyError):
            AnomalyDetectedPayload.from_dict(data)

    def test_ratio_explosion_event(self):
        """RATIO_EXPLOSION_DETECTED event with typed payload."""
        from esper.leyline import AnomalyDetectedPayload

        payload = AnomalyDetectedPayload(
            anomaly_type="ratio_explosion",
            episode=10,
            batch=5,
            inner_epoch=25,
            total_episodes=100,
            detail="PPO ratio exceeded 5.0 threshold",
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.RATIO_EXPLOSION_DETECTED,
            epoch=10,
            data=payload,
            severity="warning",
        )

        assert event.event_type == TelemetryEventType.RATIO_EXPLOSION_DETECTED
        assert isinstance(event.data, AnomalyDetectedPayload)
        assert event.data.anomaly_type == "ratio_explosion"
        assert event.severity == "warning"

    def test_ratio_collapse_event(self):
        """RATIO_COLLAPSE_DETECTED event with typed payload."""
        from esper.leyline import AnomalyDetectedPayload

        payload = AnomalyDetectedPayload(
            anomaly_type="ratio_collapse",
            episode=15,
            batch=8,
            inner_epoch=25,
            total_episodes=100,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.RATIO_COLLAPSE_DETECTED,
            epoch=15,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.RATIO_COLLAPSE_DETECTED
        assert isinstance(event.data, AnomalyDetectedPayload)

    def test_value_collapse_event(self):
        """VALUE_COLLAPSE_DETECTED event with typed payload."""
        from esper.leyline import AnomalyDetectedPayload

        payload = AnomalyDetectedPayload(
            anomaly_type="value_collapse",
            episode=20,
            batch=10,
            inner_epoch=25,
            total_episodes=100,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.VALUE_COLLAPSE_DETECTED,
            epoch=20,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.VALUE_COLLAPSE_DETECTED

    def test_numerical_instability_event(self):
        """NUMERICAL_INSTABILITY_DETECTED event with typed payload."""
        from esper.leyline import AnomalyDetectedPayload

        payload = AnomalyDetectedPayload(
            anomaly_type="numerical_instability",
            episode=25,
            batch=12,
            inner_epoch=25,
            total_episodes=100,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
            epoch=25,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED

    def test_gradient_anomaly_event(self):
        """GRADIENT_ANOMALY event with typed payload."""
        from esper.leyline import AnomalyDetectedPayload

        payload = AnomalyDetectedPayload(
            anomaly_type="gradient_anomaly",
            episode=30,
            batch=15,
            inner_epoch=25,
            total_episodes=100,
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.GRADIENT_ANOMALY,
            epoch=30,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.GRADIENT_ANOMALY

    def test_gradient_pathology_event(self):
        """GRADIENT_PATHOLOGY_DETECTED event with typed payload."""
        from esper.leyline import AnomalyDetectedPayload

        payload = AnomalyDetectedPayload(
            anomaly_type="gradient_pathology",
            episode=35,
            batch=18,
            inner_epoch=25,
            total_episodes=100,
            detail="Vanishing gradients in LSTM layers",
        )

        event = TelemetryEvent(
            event_type=TelemetryEventType.GRADIENT_PATHOLOGY_DETECTED,
            epoch=35,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.GRADIENT_PATHOLOGY_DETECTED
        assert event.data.detail == "Vanishing gradients in LSTM layers"

    def test_all_anomaly_types_use_same_payload(self):
        """All 6 anomaly event types can use AnomalyDetectedPayload."""
        from esper.leyline import AnomalyDetectedPayload

        anomaly_events = [
            TelemetryEventType.RATIO_EXPLOSION_DETECTED,
            TelemetryEventType.RATIO_COLLAPSE_DETECTED,
            TelemetryEventType.VALUE_COLLAPSE_DETECTED,
            TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
            TelemetryEventType.GRADIENT_ANOMALY,
            TelemetryEventType.GRADIENT_PATHOLOGY_DETECTED,
        ]

        for idx, event_type in enumerate(anomaly_events):
            payload = AnomalyDetectedPayload(
                anomaly_type=event_type.name.lower(),
                episode=idx * 5,
                batch=idx + 1,
                inner_epoch=25,
                total_episodes=100,
            )

            event = TelemetryEvent(
                event_type=event_type,
                epoch=idx * 5,
                data=payload,
            )

            assert event.event_type == event_type
            assert isinstance(event.data, AnomalyDetectedPayload)

    def test_debug_payload_with_gradient_stats(self):
        """Payload with debug gradient stats (collect_debug=True)."""
        from esper.leyline import AnomalyDetectedPayload

        gradient_stats = tuple(
            {"layer": f"layer_{i}", "norm": i * 0.5, "mean": i * 0.1}
            for i in range(5)
        )

        payload = AnomalyDetectedPayload(
            anomaly_type="ratio_explosion",
            episode=10,
            batch=5,
            inner_epoch=25,
            total_episodes=100,
            gradient_stats=gradient_stats,
            stability={"has_nan": False, "has_inf": False, "max_abs": 100.0},
        )

        assert len(payload.gradient_stats) == 5  # type: ignore
        assert payload.gradient_stats[0]["layer"] == "layer_0"  # type: ignore
        assert payload.stability["max_abs"] == 100.0  # type: ignore
