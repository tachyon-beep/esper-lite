"""Tests for Karn store import_jsonl typing and validation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from esper.karn.store import (
    PROOF_CRITICAL_EVENT_TYPES,
    BatchMetrics,
    DenseTrace,
    EpisodeContext,
    GateEvaluationTrace,
    HostBaseline,
    RewardComponents,
    SlotSnapshot,
    TelemetryStore,
    UnsupportedProofImportError,
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


def _write_nissa_events(dir_path: Path, records: list[dict]) -> None:
    events_file = dir_path / "events.jsonl"
    with open(events_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class TestProofGradeAccounting:
    """KARN-PROOF-006 / TT-003: import is honest about dropped proof families."""

    def _training_started(self) -> dict:
        return {
            "event_type": "TRAINING_STARTED",
            "data": {
                "episode_id": "proof_run",
                "seed": 42,
                "task": "classification",
                "reward_mode": "shaped",
                "max_epochs": 5,
                "host_params": 100,
            },
        }

    def test_unsupported_families_counted_and_store_not_proof_grade(
        self, tmp_path: Path
    ) -> None:
        records = [
            self._training_started(),
            {"event_type": "EPOCH_COMPLETED", "epoch": 1,
             "data": {"val_loss": 0.5, "val_accuracy": 0.9}},
            {"event_type": "EPISODE_OUTCOME", "data": {"final_accuracy": 0.9}},
            {"event_type": "GOVERNOR_ROLLBACK", "data": {"reason": "nan"}},
            {"event_type": "GOVERNOR_ROLLBACK", "data": {"reason": "divergence"}},
            {"event_type": "MORPHOLOGY_CAUSAL_LOG", "data": {"phase": "proposal"}},
            {
                "event_type": "TOPOLOGY_MANIFEST_RECORDED",
                "data": {"manifest_role": "source_final"},
            },
            {"event_type": "SEED_GERMINATED", "data": {}},
            {"event_type": "PPO_UPDATE_COMPLETED", "data": {}},
        ]
        _write_nissa_events(tmp_path, records)

        store = TelemetryStore.import_from_nissa_dir(tmp_path)

        # Every non-modeled family was counted.
        assert store.unsupported_event_counts["EPISODE_OUTCOME"] == 1
        assert store.unsupported_event_counts["GOVERNOR_ROLLBACK"] == 2
        assert store.unsupported_event_counts["MORPHOLOGY_CAUSAL_LOG"] == 1
        assert store.unsupported_event_counts["TOPOLOGY_MANIFEST_RECORDED"] == 1
        assert store.unsupported_event_counts["SEED_GERMINATED"] == 1
        assert store.unsupported_event_counts["PPO_UPDATE_COMPLETED"] == 1
        # Modeled families are NOT counted as unsupported.
        assert "EPOCH_COMPLETED" not in store.unsupported_event_counts
        assert "TRAINING_STARTED" not in store.unsupported_event_counts

        # Proof-critical families observed-but-dropped are flagged.
        assert store.dropped_proof_families == set(PROOF_CRITICAL_EVENT_TYPES)
        assert store.proof_grade is False

    def test_assert_proof_grade_raises_typed_error_with_families(
        self, tmp_path: Path
    ) -> None:
        records = [
            self._training_started(),
            {"event_type": "EPISODE_OUTCOME", "data": {}},
            {"event_type": "GOVERNOR_ROLLBACK", "data": {}},
        ]
        _write_nissa_events(tmp_path, records)
        store = TelemetryStore.import_from_nissa_dir(tmp_path)

        with pytest.raises(UnsupportedProofImportError) as excinfo:
            store.assert_proof_grade()

        err = excinfo.value
        assert "EPISODE_OUTCOME" in err.dropped_proof_families
        assert "GOVERNOR_ROLLBACK" in err.dropped_proof_families
        assert err.unsupported_event_counts["EPISODE_OUTCOME"] == 1
        assert "EPISODE_OUTCOME" in str(err)

    def test_proof_grade_false_even_with_no_unsupported_events(
        self, tmp_path: Path
    ) -> None:
        """A pristine analytics-only import is still NOT a proof archive."""
        records = [
            self._training_started(),
            {"event_type": "EPOCH_COMPLETED", "epoch": 1,
             "data": {"val_loss": 0.5, "val_accuracy": 0.9}},
        ]
        _write_nissa_events(tmp_path, records)
        store = TelemetryStore.import_from_nissa_dir(tmp_path)

        assert store.unsupported_event_counts == {}
        assert store.dropped_proof_families == set()
        assert store.proof_grade is False
        with pytest.raises(UnsupportedProofImportError):
            store.assert_proof_grade()

    def test_absent_metrics_not_fabricated_as_zero(self, tmp_path: Path) -> None:
        """Missing val_loss/val_accuracy must NOT be defaulted to 0.0."""
        records = [
            self._training_started(),
            # EPOCH_COMPLETED with NO val_loss / val_accuracy keys.
            {"event_type": "EPOCH_COMPLETED", "epoch": 1, "data": {}},
        ]
        _write_nissa_events(tmp_path, records)
        store = TelemetryStore.import_from_nissa_dir(tmp_path)

        assert len(store.epoch_snapshots) == 1
        host = store.epoch_snapshots[0].host
        # val_loss stays None (the dataclass default), not fabricated 0.0.
        assert host.val_loss is None
        # val_accuracy also stays None; absent telemetry is not measured-zero accuracy.
        assert host.val_accuracy is None

    def test_explicit_zero_accuracy_preserved(self, tmp_path: Path) -> None:
        """A real measured 0.0 accuracy must remain distinguishable from absence."""
        records = [
            self._training_started(),
            {
                "event_type": "EPOCH_COMPLETED",
                "epoch": 1,
                "data": {"val_loss": 0.0, "val_accuracy": 0.0},
            },
        ]
        _write_nissa_events(tmp_path, records)
        store = TelemetryStore.import_from_nissa_dir(tmp_path)

        host = store.epoch_snapshots[0].host
        assert host.val_loss == 0.0
        assert host.val_accuracy == 0.0

    def test_present_metrics_preserved(self, tmp_path: Path) -> None:
        records = [
            self._training_started(),
            {"event_type": "EPOCH_COMPLETED", "epoch": 1,
             "data": {"val_loss": 0.25, "val_accuracy": 0.88}},
        ]
        _write_nissa_events(tmp_path, records)
        store = TelemetryStore.import_from_nissa_dir(tmp_path)

        host = store.epoch_snapshots[0].host
        assert host.val_loss == 0.25
        assert host.val_accuracy == 0.88


class TestExportSelfDeclaresProofStatus:
    """KARN-PROOF-005: the export artifact declares its own proof status."""

    def test_export_writes_proof_status_header(self, tmp_path: Path) -> None:
        store = TelemetryStore()
        store.record_unsupported_event("EPISODE_OUTCOME")
        store.record_unsupported_event("GOVERNOR_ROLLBACK")

        output = tmp_path / "store.jsonl"
        store.export_jsonl(output)

        with open(output) as f:
            header = json.loads(f.readline())
        assert header["type"] == "proof_status"
        assert header["data"]["proof_grade"] is False
        assert header["data"]["unsupported_event_counts"]["EPISODE_OUTCOME"] == 1
        assert "EPISODE_OUTCOME" in header["data"]["dropped_proof_families"]
        assert "GOVERNOR_ROLLBACK" in header["data"]["dropped_proof_families"]

    def test_proof_status_roundtrips_through_import(self, tmp_path: Path) -> None:
        store = TelemetryStore()
        store.record_unsupported_event("EPISODE_OUTCOME")
        store.record_unsupported_event("MORPHOLOGY_CAUSAL_LOG")

        output = tmp_path / "store.jsonl"
        store.export_jsonl(output)
        imported = TelemetryStore.import_jsonl(output)

        assert imported.unsupported_event_counts["EPISODE_OUTCOME"] == 1
        assert imported.unsupported_event_counts["MORPHOLOGY_CAUSAL_LOG"] == 1
        assert imported.dropped_proof_families == {
            "EPISODE_OUTCOME",
            "MORPHOLOGY_CAUSAL_LOG",
        }
        assert imported.proof_grade is False
