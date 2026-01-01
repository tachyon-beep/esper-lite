"""Wiring tests for heuristic training telemetry.

These tests lock the contract that heuristic training emits:
- TRAINING_STARTED with env_devices reflecting the actual training device
- EPOCH_COMPLETED with optional seed gradient telemetry gated by gradient_telemetry_stride
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from esper.leyline import TelemetryEventType
from esper.leyline.actions import build_action_enum
from esper.leyline.telemetry import EpochCompletedPayload, TrainingStartedPayload
from esper.nissa import get_hub, reset_hub
from esper.runtime import get_task_spec
from esper.simic.training.helpers import run_heuristic_episode
from esper.tamiyo.decisions import TamiyoDecision


class _ScriptedPolicy:
    """Deterministic policy that germinates and activates exactly one seed."""

    def __init__(self, topology: str) -> None:
        self._Action = build_action_enum(topology)

    def reset(self) -> None:  # pragma: no cover - exercised implicitly
        pass

    def decide(self, signals: Any, active_seeds: list[Any]) -> TamiyoDecision:
        epoch = int(signals.metrics.epoch)
        if epoch == 1:
            return TamiyoDecision(
                action=self._Action["GERMINATE_CONV_LIGHT"],
                reason="test germinate",
            )
        if epoch == 2:
            assert active_seeds, "Expected germinated seed to be present by epoch 2"
            return TamiyoDecision(
                action=self._Action.ADVANCE,
                target_seed_id=active_seeds[0].seed_id,
                reason="test advance to TRAINING",
            )
        return TamiyoDecision(action=self._Action.WAIT, reason="test wait")


def _single_batch_loader(*, batch_size: int, num_classes: int) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, num_classes, (batch_size,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def test_heuristic_gradient_telemetry_stride_wires_into_epoch_completed_payload() -> None:
    reset_hub()
    hub = get_hub()

    training_started: list[Any] = []
    epoch_completed: list[Any] = []

    class CaptureBackend:
        def start(self) -> None:
            pass

        def emit(self, event: Any) -> None:
            if event.event_type == TelemetryEventType.TRAINING_STARTED:
                training_started.append(event)
            if event.event_type == TelemetryEventType.EPOCH_COMPLETED:
                epoch_completed.append(event)

        def close(self) -> None:
            pass

    hub.add_backend(CaptureBackend())

    try:
        task_spec = get_task_spec("cifar10")
        policy = _ScriptedPolicy(task_spec.topology)
        trainloader = _single_batch_loader(batch_size=2, num_classes=10)
        testloader = _single_batch_loader(batch_size=2, num_classes=10)

        run_heuristic_episode(
            policy=policy,
            trainloader=trainloader,
            testloader=testloader,
            max_epochs=4,
            max_batches=1,
            base_seed=123,
            device="cpu:0",
            task_spec=task_spec,
            slots=["r0c0"],
            telemetry_config=None,
            telemetry_lifecycle_only=False,
            gradient_telemetry_stride=3,
        )
        hub.flush()

        assert len(training_started) == 1
        started_event = training_started[0]
        assert isinstance(started_event.data, TrainingStartedPayload)
        assert started_event.data.env_devices == ("cpu:0",)

        by_epoch = {int(e.epoch): e for e in epoch_completed}
        assert set(by_epoch.keys()) == {1, 2, 3, 4}

        epoch3 = by_epoch[3]
        epoch4 = by_epoch[4]
        assert isinstance(epoch3.data, EpochCompletedPayload)
        assert isinstance(epoch4.data, EpochCompletedPayload)

        # Seed becomes active at epoch 3 (after ADVANCE at end of epoch 2).
        # With stride=3, gradient telemetry should be present on epoch 3 but not epoch 4.
        assert epoch3.data.seeds is not None
        assert "__aggregate__" in epoch3.data.seeds
        assert epoch4.data.seeds is None
    finally:
        reset_hub()
