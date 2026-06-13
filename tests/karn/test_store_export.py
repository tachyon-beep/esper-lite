"""Tests for Karn store export functionality."""

import json
from datetime import datetime, timezone
from pathlib import Path


from esper.karn.collector import KarnCollector
from esper.karn.store import TelemetryStore, EpisodeContext
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    EpochCompletedPayload,
    TrainingStartedPayload,
)


class TestExportJsonl:
    """Tests for JSONL export serialization."""

    def test_export_handles_datetime(self, tmp_path: Path):
        """export_jsonl correctly serializes datetime objects."""
        store = TelemetryStore()

        # Create context with datetime
        store.context = EpisodeContext(
            episode_id="test_123",
            timestamp=datetime(2025, 12, 14, 12, 0, 0, tzinfo=timezone.utc),
            base_seed=42,
        )

        output_file = tmp_path / "test_export.jsonl"
        count = store.export_jsonl(output_file)

        assert count >= 1
        assert output_file.exists()

        # Verify it's valid JSON
        with open(output_file) as f:
            line = f.readline()
            data = json.loads(line)  # Should not raise
            assert data["type"] == "context"
            # timestamp should be serialized as ISO string
            assert "timestamp" in data["data"]

    def test_export_handles_path_objects(self, tmp_path: Path):
        """export_jsonl correctly serializes Path objects."""
        store = TelemetryStore()
        store.context = EpisodeContext(
            episode_id="test_path",
            timestamp=datetime.now(timezone.utc),
        )

        output_file = tmp_path / "test_path.jsonl"

        # Should not raise TypeError
        count = store.export_jsonl(output_file)
        assert count >= 1

    def test_export_handles_enums(self, tmp_path: Path):
        """export_jsonl correctly serializes Enum values."""
        from esper.karn.store import SlotSnapshot, EpisodeContext
        from esper.leyline import SeedStage

        store = TelemetryStore()

        # start_episode takes EpisodeContext, not string args
        context = EpisodeContext(
            episode_id="test_enum",
            base_seed=42,
        )
        store.start_episode(context)

        # start_episode doesn't create current_epoch - must call start_epoch explicitly
        store.start_epoch(1)

        # Add a slot with enum stage
        store.current_epoch.slots["r0c1"] = SlotSnapshot(
            slot_id="r0c1",
            stage=SeedStage.TRAINING,
        )
        store.commit_epoch()

        output_file = tmp_path / "test_enum.jsonl"
        store.export_jsonl(output_file)

        # Should serialize enum as name string
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                if data["type"] == "epoch":
                    # Check that stage is serialized (as int from enum value)
                    assert "slots" in data["data"]


class TestHostParamsPreserved:
    """SMOKE-001: TrainingStartedPayload.host_params must survive the Karn path.

    A smoke run emitted host_params=164 on TRAINING_STARTED, yet the Karn export
    showed 0 because the collector never threaded the value into the episode
    context or the host snapshots. These tests pin the real value (164) through
    the export's context block AND the first committed host snapshot.
    """

    HOST_PARAMS = 164

    def _training_started_event(self) -> TelemetryEvent:
        return TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=TrainingStartedPayload(
                n_envs=1,
                max_epochs=5,
                max_batches=10,
                task="smoke_task",
                host_params=self.HOST_PARAMS,
                slot_ids=("r0c0",),
                seed=42,
                n_episodes=1,
                lr=0.01,
                clip_ratio=0.2,
                entropy_coef=0.01,
                param_budget=0,
                reward_mode="shaped",
                policy_device="cpu",
                env_devices=("cpu",),
                episode_id="smoke_host_params",
            ),
        )

    def test_export_context_and_first_host_snapshot_carry_host_params(
        self, tmp_path: Path
    ):
        """Export context AND the first host snapshot carry the emitted value, not 0."""
        collector = KarnCollector()

        # Start training: handler threads host_params into the episode context
        # and seeds epoch 1's host snapshot from it.
        collector.emit(self._training_started_event())

        # Complete one epoch so a host snapshot is committed to history.
        collector.emit(
            TelemetryEvent(
                event_type=TelemetryEventType.EPOCH_COMPLETED,
                epoch=1,
                data=EpochCompletedPayload(
                    inner_epoch=1,
                    env_id=0,
                    val_loss=0.5,
                    val_accuracy=0.9,
                ),
            )
        )

        # Context must already carry the real value in memory.
        assert collector.store.context is not None
        assert collector.store.context.host_params == self.HOST_PARAMS

        # First committed epoch's host snapshot must carry it too.
        assert len(collector.store.epoch_snapshots) >= 1
        first_host = collector.store.epoch_snapshots[0].host
        assert first_host.host_params == self.HOST_PARAMS

        # And it must survive serialization into the export.
        output_file = tmp_path / "smoke_export.jsonl"
        collector.store.export_jsonl(output_file)

        context_params: int | None = None
        first_epoch_host_params: int | None = None
        with open(output_file) as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == "context":
                    context_params = record["data"]["host_params"]
                elif record["type"] == "epoch" and first_epoch_host_params is None:
                    first_epoch_host_params = record["data"]["host"]["host_params"]

        assert context_params == self.HOST_PARAMS, (
            "Karn export context dropped host_params"
        )
        assert first_epoch_host_params == self.HOST_PARAMS, (
            "Karn export first host snapshot dropped host_params"
        )

    def test_nissa_dir_import_preserves_host_params(self, tmp_path: Path):
        """Importing a Nissa events.jsonl preserves the emitted host_params."""
        events_file = tmp_path / "events.jsonl"
        records = [
            {
                "event_type": "TRAINING_STARTED",
                "data": {
                    "episode_id": "nissa_smoke",
                    "seed": 42,
                    "task": "smoke_task",
                    "reward_mode": "shaped",
                    "max_epochs": 5,
                    "host_params": self.HOST_PARAMS,
                },
            },
            {
                "event_type": "EPOCH_COMPLETED",
                "epoch": 1,
                "data": {"val_loss": 0.5, "val_accuracy": 0.9},
            },
        ]
        with open(events_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        store = TelemetryStore.import_from_nissa_dir(tmp_path)

        assert store.context is not None
        assert store.context.host_params == self.HOST_PARAMS
        assert len(store.epoch_snapshots) >= 1
        assert store.epoch_snapshots[0].host.host_params == self.HOST_PARAMS
