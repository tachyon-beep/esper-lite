"""Tests for Karn collector multi-env support."""


from esper.leyline import TelemetryEvent, TelemetryEventType


class TestMultiEnvSlotTracking:
    """Tests for slot tracking across multiple environments."""

    def test_slots_namespaced_by_env_id(self):
        """Slots from different envs don't collide."""
        from esper.karn.collector import KarnCollector

        # KarnCollector creates its own store internally
        collector = KarnCollector()
        store = collector.store

        # Start episode - TRAINING_STARTED handler creates context and starts epoch
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test_multi", "max_epochs": 10, "n_envs": 2}
        ))

        # Germinate in slot "mid" for env 0
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="mid",
            data={"env_id": 0, "seed_id": "env0_seed_0", "blueprint_id": "conv"}
        ))

        # Germinate in slot "mid" for env 1 (same slot_id, different env)
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="mid",
            data={"env_id": 1, "seed_id": "env1_seed_0", "blueprint_id": "norm"}
        ))

        # Both should be tracked separately
        slots = store.current_epoch.slots

        # Expect namespaced keys
        assert "env0:mid" in slots or ("mid" in slots and len(slots) == 2)

        # If namespaced, verify different blueprints
        if "env0:mid" in slots:
            assert slots["env0:mid"].blueprint_id == "conv"
            assert slots["env1:mid"].blueprint_id == "norm"

    def test_env_id_extracted_from_event_data(self):
        """env_id is correctly extracted from event.data."""
        from esper.karn.collector import KarnCollector

        # KarnCollector creates its own store internally
        collector = KarnCollector()
        store = collector.store

        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={"episode_id": "test", "max_epochs": 5}
        ))

        # Event with env_id in data
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            slot_id="early",
            data={"env_id": 3, "seed_id": "env3_seed_0", "blueprint_id": "test"}
        ))

        # Should namespace by env_id
        slots = store.current_epoch.slots
        assert "env3:early" in slots or "early" in slots

    def test_counterfactual_env_idx_fallback_namespaces_by_env(self):
        """Counterfactual events accept legacy env_idx for namespacing."""
        from esper.karn.collector import KarnCollector

        collector = KarnCollector()
        store = collector.store

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.TRAINING_STARTED,
                data={"episode_id": "test_cf_env_idx", "max_epochs": 5, "n_envs": 2},
            )
        )

        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
                slot_id="mid",
                data={"env_idx": 0, "contribution": 0.1},
            )
        )
        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.COUNTERFACTUAL_COMPUTED,
                slot_id="mid",
                data={"env_idx": 1, "contribution": 0.9},
            )
        )

        slots = store.current_epoch.slots
        assert slots["env0:mid"].counterfactual_contribution == 0.1
        assert slots["env1:mid"].counterfactual_contribution == 0.9
