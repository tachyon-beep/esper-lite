from __future__ import annotations

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis
import os

from esper.core import EsperSettings
from esper.oona import OonaClient, StreamConfig
from esper.weatherlight.service_runner import WeatherlightService
from esper.tolaria import TolariaTrainer, TrainingLoopConfig
from esper.tolaria.rollback import SharedDeadlineSignal
from esper.leyline import leyline_pb2
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _TamiyoTimeout:
    def evaluate_epoch(self, state: leyline_pb2.SystemStatePacket) -> leyline_pb2.AdaptationCommand:
        raise TimeoutError("simulated-timeout")


class _KasminaStub:
    def apply_command(self, command: leyline_pb2.AdaptationCommand) -> None:
        pass

    def export_seed_states(self):
        return []

    def advance_alpha(self, seed_id: str, *, steps: int = 1) -> float:
        return 0.0


async def _start_weatherlight(redis: FakeRedis, signal_name: str) -> WeatherlightService:
    settings = EsperSettings(tolaria_rollback_signal_name=signal_name)
    svc = WeatherlightService(settings=settings)
    # Inject FakeRedis into OonaClient by patching build method
    async def _build(self):
        cfg = StreamConfig(
            normal_stream=settings.oona_normal_stream,
            emergency_stream=settings.oona_emergency_stream,
            telemetry_stream=settings.oona_telemetry_stream,
            policy_stream=settings.oona_policy_stream,
            group="weatherlight",
            consumer="weatherlight-test",
            dead_letter_stream="oona.deadletter",
        )
        client = OonaClient("redis://localhost", config=cfg, redis_client=redis)
        await client.ensure_consumer_group()
        return client

    WeatherlightService._build_oona_client = _build  # type: ignore[assignment]
    # Prevent worker registration to avoid consuming streams in tests
    WeatherlightService._register_worker = lambda self, state: None  # type: ignore[assignment]
    # Weatherlight requires a secret; provide a dummy one for tests
    monkey_secret = os.environ.get("ESPER_LEYLINE_SECRET")
    os.environ["ESPER_LEYLINE_SECRET"] = "test-secret"
    await svc.start()
    # restore secret if needed
    if monkey_secret is not None:
        os.environ["ESPER_LEYLINE_SECRET"] = monkey_secret
    return svc


@pytest.mark.asyncio
async def test_shared_rollback_signal_weatherlight_bridge(monkeypatch) -> None:
    redis = FakeRedis()
    signal_name = "tolaria-test-shared"
    # Build a trainer that uses the same shared signal name and triggers deadline
    model = nn.Sequential(nn.Linear(8, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(8, 8)
    targets = torch.randint(0, 4, (8,))
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    # Ensure rollback path enabled via env to align with BaseSettings
    os.environ["TOLARIA_ROLLBACK_ENABLED"] = "true"
    settings = EsperSettings(
        tolaria_rollback_signal_name=signal_name,
        tolaria_rollback_deadline_ms=1,
        tolaria_emergency_enabled=True,
        tolaria_emergency_l4_failed_epochs_threshold=999,
    )
    trainer = TolariaTrainer(
        model=model,
        optimizer=opt,
        dataloader=loader,
        tamiyo=_TamiyoTimeout(),
        kasmina=_KasminaStub(),
        config=TrainingLoopConfig(max_epochs=1, gradient_accumulation_steps=1, device=torch.device("cpu")),
        settings=settings,
    )
    trainer.set_shared_rollback_signal(signal_name)
    # Start Weatherlight after the shared segment exists
    svc = await _start_weatherlight(redis, signal_name)
    # Prevent fast cache hits; force slow restore via monkeypatch
    import types
    if getattr(trainer, "_fast_cache", None) is not None:
        trainer._fast_cache.get_nearest = types.MethodType(lambda self, step: None, trainer._fast_cache)  # type: ignore[attr-defined]
    import time
    monkeypatch.setattr(TolariaTrainer, "rollback_to_last_checkpoint", lambda self: (time.sleep(0.05) or False), raising=True)
    # Ensure shared signal used by trainer
    sig_local = trainer.get_rollback_signal()
    assert sig_local is not None
    list(trainer.run())
    # Trigger the shared signal manually to validate Weatherlight bridge
    sig = SharedDeadlineSignal.attach(signal_name)
    sig.trigger()
    # Verify Weatherlight detected the signal via internal telemetry snapshot
    ok = False
    for _ in range(30):
        packet = await svc._build_telemetry_packet()  # type: ignore[attr-defined]
        metrics = {m.name: m.value for m in packet.metrics}
        if metrics.get("weatherlight.rollback.detections_total", 0.0) >= 1.0:
            ok = True
            break
        await asyncio.sleep(0.1)
    assert ok
    # Cleanup shared memory
    try:
        sig.close()
        sig.unlink()
    except Exception:
        pass
    await svc.shutdown()
