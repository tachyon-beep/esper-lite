"""Tests for heuristic training telemetry emission."""


from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import TrainingStartedPayload


class TestHeuristicTelemetry:
    """Tests for telemetry emission in heuristic training."""

    def test_training_started_event_contract(self):
        """Document expected TRAINING_STARTED event structure."""
        # This test documents the contract that heuristic training should emit
        # TRAINING_STARTED to activate Karn.

        payload = TrainingStartedPayload(
            n_envs=1,
            max_epochs=75,
            max_batches=100,  # Total episodes in run
            task="cifar_baseline",
            host_params=1000000,
            slot_ids=("r0c0",),
            seed=42,
            n_episodes=100,
            lr=0.0003,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=5000000,
            policy_device="cpu",
            env_devices=("cpu",),
            episode_id="heur_42",
            reward_mode="shaped",
        )
        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=payload,
        )

        assert event.event_type == TelemetryEventType.TRAINING_STARTED
        assert event.data.episode_id == "heur_42"
        assert event.data.max_epochs == 75
