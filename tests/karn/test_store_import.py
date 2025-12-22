"""Tests for Karn store import_jsonl typing and validation."""

from datetime import datetime, timezone
from pathlib import Path

from esper.karn.store import (
    BatchMetrics,
    DenseTrace,
    EpisodeContext,
    GateEvaluationTrace,
    HostBaseline,
    RewardComponents,
    SlotSnapshot,
    TelemetryStore,
)
from esper.leyline import SeedStage


class TestStoreImportJsonl:
    """Ensure import_jsonl restores key types (datetime, enums, nested dataclasses)."""

    def test_import_jsonl_restores_types(self, tmp_path: Path) -> None:
        store = TelemetryStore()
        store.context = EpisodeContext(
            episode_id="episode_1",
            timestamp=datetime(2025, 12, 20, 12, 0, 0, tzinfo=timezone.utc),
            base_seed=123,
            slot_config=(("r0c0", 64),),
            hyperparameters=(("lr", 0.001),),
        )
        store.baseline = HostBaseline(initial_checkpoint_path=Path("checkpoints/host.pt"))

        store.start_epoch(1)
        assert store.current_epoch is not None
        store.current_epoch.host.val_accuracy = 55.5
        store.current_epoch.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.TRAINING,
            seed_id="seed_1",
            blueprint_id="conv_light",
            seed_params=123,
            alpha=0.5,
        )
        # Policy snapshot with nested RewardComponents (covers import reconstruction)
        from esper.karn.store import PolicySnapshot

        store.current_epoch.policy = PolicySnapshot(
            action_op="WAIT",
            reward_total=1.0,
            reward_components=RewardComponents(total=1.0, accuracy_delta=0.5),
            kl_divergence=0.1,
        )
        store.commit_epoch()

        # Dense trace with nested batch metrics + gate evaluation
        store.dense_traces.append(DenseTrace(
            trigger_reason="test",
            window_start_epoch=1,
            window_end_epoch=2,
            timestamp=datetime(2025, 12, 20, 12, 0, 1, tzinfo=timezone.utc),
            batch_metrics=[
                BatchMetrics(
                    epoch=1,
                    batch_idx=0,
                    loss=0.123,
                    accuracy=0.5,
                    seed_grad_norms={"r0c0": 1.0},
                )
            ],
            gate_evaluation_details=GateEvaluationTrace(
                gate_id="G2",
                slot_id="r0c0",
                passed=True,
                reason="ok",
                metrics_at_evaluation={"val_accuracy": 55.5},
                thresholds_used={"min_accuracy": 50.0},
            ),
        ))

        output = tmp_path / "store.jsonl"
        store.export_jsonl(output)

        imported = TelemetryStore.import_jsonl(output)

        assert imported.context is not None
        assert isinstance(imported.context.timestamp, datetime)
        assert imported.baseline is not None
        assert isinstance(imported.baseline.initial_checkpoint_path, Path)

        assert len(imported.epoch_snapshots) == 1
        epoch = imported.epoch_snapshots[0]
        assert isinstance(epoch.timestamp, datetime)
        assert isinstance(epoch.host.val_accuracy, float)
        assert "r0c0" in epoch.slots
        slot = epoch.slots["r0c0"]
        assert isinstance(slot.stage, SeedStage)
        assert slot.stage == SeedStage.TRAINING
        assert slot.seed_params == 123

        assert epoch.policy is not None
        assert isinstance(epoch.policy.reward_components, RewardComponents)
        assert epoch.policy.reward_components.total == 1.0

        assert len(imported.dense_traces) == 1
        trace = imported.dense_traces[0]
        assert isinstance(trace.timestamp, datetime)
        assert trace.batch_metrics
        assert isinstance(trace.batch_metrics[0], BatchMetrics)
        assert trace.gate_evaluation_details is not None
        assert isinstance(trace.gate_evaluation_details, GateEvaluationTrace)
