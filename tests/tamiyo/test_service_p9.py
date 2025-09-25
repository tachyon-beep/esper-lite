import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext
from esper.tamiyo import FieldReportStore, FieldReportStoreConfig, TamiyoPolicy, TamiyoService
from esper.urza import UrzaLibrary

_SIGN = SignatureContext(secret=b"tamiyo-p9-test")


def _packet(
    epoch: int, run_id: str = "run-p9", *, loss_delta: float = -0.1
) -> leyline_pb2.SystemStatePacket:
    pkt = leyline_pb2.SystemStatePacket(
        version=1,
        current_epoch=epoch,
        training_run_id=run_id,
        training_metrics={"loss_delta": loss_delta},
    )
    # Provide a seed so policy can select SEED/OPTIMIZER paths as needed
    seed = pkt.seed_states.add()
    seed.seed_id = f"seed-{epoch}"
    seed.stage = leyline_pb2.SeedLifecycleStage.SEED_STAGE_TRAINING
    return pkt


def test_observation_window_synthesises_after_n_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAMIYO_FR_OBS_WINDOW_EPOCHS", "2")
    store_cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(), store_config=store_cfg, urza=urza, signature_context=_SIGN
    )

    service.evaluate_step(_packet(1))
    # After first step: only per-step report should exist; no synthesis yet
    reports1 = service.field_reports
    assert any(r.observation_window_epochs == 1 for r in reports1)
    assert not any(r.report_id.startswith("fr-synth-") for r in reports1)

    service.evaluate_step(_packet(2))
    reports2 = service.field_reports
    # Expect a synthesised report with observation_window_epochs=2
    synth = [r for r in reports2 if r.report_id.startswith("fr-synth-")]
    assert synth, "expected a synthesised field report after 2 epochs"
    assert all(r.observation_window_epochs == 2 for r in synth)


class _FlakyOona:
    def __init__(self) -> None:
        self._attempt = 0

    async def publish_field_report(self, _report: leyline_pb2.FieldReport) -> bool:
        self._attempt += 1
        if self._attempt == 1:
            raise RuntimeError("temporary failure")
        return True

    async def publish_telemetry(
        self, _pkt: leyline_pb2.TelemetryPacket, *, priority=None
    ) -> bool:  # noqa: ARG002
        return True


@pytest.mark.asyncio
async def test_publish_history_ack_retry_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAMIYO_FR_RETRY_BACKOFF_MS", "0")
    store_cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(), store_config=store_cfg, urza=urza, signature_context=_SIGN
    )
    service.evaluate_step(_packet(1))

    oona = _FlakyOona()
    ok1 = await service.publish_history(oona)
    assert not ok1
    # Inspect retry index on disk
    index_path = store_cfg.path.parent / "field_reports.index.json"
    assert index_path.exists()
    content = index_path.read_text(encoding="utf-8")
    assert "retry_count" in content and "published" in content

    ok2 = await service.publish_history(oona)
    assert ok2
    # Field reports drained from memory; index marks published
    assert service.field_reports == []
    idx = index_path.read_text(encoding="utf-8")
    assert '"published":true' in idx.replace(" ", "").lower()


class _AlwaysFailOona:
    async def publish_field_report(self, _report: leyline_pb2.FieldReport) -> bool:  # noqa: ARG002
        raise RuntimeError("fail")

    async def publish_telemetry(
        self, _pkt: leyline_pb2.TelemetryPacket, *, priority=None
    ) -> bool:  # noqa: ARG002
        return True


@pytest.mark.asyncio
async def test_retry_cap_drops_from_memory_preserves_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAMIYO_FIELD_REPORT_MAX_RETRIES", "1")
    monkeypatch.setenv("TAMIYO_FR_RETRY_BACKOFF_MS", "0")
    store_cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(), store_config=store_cfg, urza=urza, signature_context=_SIGN
    )
    service.evaluate_step(_packet(1))
    oona = _AlwaysFailOona()
    # Two attempts → should drop from memory
    ok1 = await service.publish_history(oona)
    ok2 = await service.publish_history(oona)
    assert not ok1 and not ok2
    assert service.field_reports == []
    # WAL still contains the report
    store = FieldReportStore(store_cfg)
    assert len(store.reports()) >= 1
    # Index marks dropped
    idx_path = store_cfg.path.parent / "field_reports.index.json"
    assert "dropped" in idx_path.read_text(encoding="utf-8").lower()


def test_restart_restores_window_and_retry_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAMIYO_FR_OBS_WINDOW_EPOCHS", "2")
    store_cfg = FieldReportStoreConfig(path=tmp_path / "field_reports.log")
    urza = UrzaLibrary(root=tmp_path / "urza")
    service = TamiyoService(
        policy=TamiyoPolicy(), store_config=store_cfg, urza=urza, signature_context=_SIGN
    )
    service.evaluate_step(_packet(1))
    # Ensure a window file is present
    win_path = store_cfg.path.parent / "field_reports.windows.json"
    assert win_path.exists()

    # Simulate restart: new instance should load prior window
    service2 = TamiyoService(
        policy=TamiyoPolicy(), store_config=store_cfg, urza=urza, signature_context=_SIGN
    )
    service2.evaluate_step(_packet(2))
    # Synthesised report should be present
    synth = [r for r in service2.field_reports if r.report_id.startswith("fr-synth-")]
    assert synth, "expected synthesis after restart and another step"
