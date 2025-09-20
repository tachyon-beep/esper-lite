from __future__ import annotations

import pytest
from torch import nn

from esper.kasmina import KasminaLifecycle, KasminaSeedManager, LifecycleEvent, LifecycleState
from esper.leyline import leyline_pb2


class _RuntimeStub:
    def __init__(self) -> None:
        self.loaded = []

    def load_kernel(self, blueprint_id: str) -> nn.Module:
        self.loaded.append(blueprint_id)
        return nn.Identity()


def test_lifecycle_transitions_follow_order() -> None:
    lifecycle = KasminaLifecycle()
    lifecycle.apply(LifecycleEvent.REGISTER)
    lifecycle.apply(LifecycleEvent.GERMINATE)
    lifecycle.apply(LifecycleEvent.ACTIVATE)
    assert lifecycle.state is LifecycleState.ACTIVE


def test_lifecycle_rejects_invalid_transition() -> None:
    lifecycle = KasminaLifecycle()
    with pytest.raises(ValueError):
        lifecycle.apply(LifecycleEvent.ACTIVATE)


def test_seed_manager_grafts_and_retires_seed() -> None:
    runtime = _RuntimeStub()
    manager = KasminaSeedManager(runtime=runtime)
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-1",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "bp-1"
    manager.handle_command(command)
    assert "seed-1" in manager.seeds()
    assert runtime.loaded == ["bp-1"]

    retire = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-2",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    retire.seed_operation.operation = leyline_pb2.SEED_OP_CULL
    manager.handle_command(retire)
    assert "seed-1" not in manager.seeds()
