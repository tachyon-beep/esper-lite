from __future__ import annotations

from dataclasses import dataclass

import torch

from esper.kasmina.seed_manager import KasminaSeedManager, SeedContext
from esper.leyline import leyline_pb2 as pb


@dataclass
class _Runtime:
    def fetch_kernel(self, *_args, **_kwargs):  # pragma: no cover - stub
        return torch.nn.Identity(), 0.0


def test_command_context_defaults() -> None:
    command = pb.AdaptationCommand()
    context = KasminaSeedManager.CommandContext(command)
    assert context.seed_id == ""
    assert context.blueprint_id == ""
    assert context.training_run_id == ""
    assert context.annotations == {}
    assert context.seed_context is None
    assert context.operation is None
    assert context.parameters == {}
    assert context.expected_stage is None
    assert context.resume is False
    assert context.include_teacher is False
    assert context.optimizer_id == ""


def test_command_outcome_defaults() -> None:
    outcome = KasminaSeedManager.CommandOutcome()
    assert outcome.events == []
    assert outcome.handled is False
    assert outcome.seed_id is None


def test_dispatcher_returns_outcome_for_pause() -> None:
    manager = KasminaSeedManager(runtime=_Runtime())
    command = pb.AdaptationCommand(command_type=pb.COMMAND_PAUSE)
    manager.handle_command(command)
    # No exception means dispatcher path executed; ensure outcome reflects missing seed
    outcome = manager._dispatch_command(manager._build_command_context(command))
    assert isinstance(outcome, KasminaSeedManager.CommandOutcome)
    assert outcome.handled is False
    assert outcome.events == []


def test_build_command_context_seed_fields() -> None:
    manager = KasminaSeedManager(runtime=_Runtime())
    manager._seeds["seed-1"] = SeedContext("seed-1")
    command = pb.AdaptationCommand(
        command_type=pb.COMMAND_SEED,
        target_seed_id="seed-1",
    )
    command.seed_operation.operation = pb.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "bp-1"
    command.seed_operation.parameters["alpha"] = 0.5
    command.annotations["training_run_id"] = " run-123  "

    ctx = manager._build_command_context(command)
    assert ctx.seed_id == "seed-1"
    assert ctx.blueprint_id == "bp-1"
    assert ctx.training_run_id == "run-123"
    assert ctx.operation == pb.SEED_OP_GERMINATE
    assert ctx.parameters.get("alpha") == 0.5
    assert ctx.seed_context is manager._seeds["seed-1"]


def test_build_command_context_pause_resume() -> None:
    manager = KasminaSeedManager(runtime=_Runtime())
    manager._seeds["seed-2"] = SeedContext("seed-2")
    command = pb.AdaptationCommand(
        command_type=pb.COMMAND_PAUSE,
        target_seed_id="seed-2",
    )
    command.annotations["resume"] = "true"

    ctx = manager._build_command_context(command)
    assert ctx.seed_id == "seed-2"
    assert ctx.resume is True
    assert ctx.seed_context is manager._seeds["seed-2"]


def test_dispatcher_handles_seed_command() -> None:
    manager = KasminaSeedManager(runtime=_Runtime())
    manager.register_host_model(torch.nn.Linear(1, 1))

    command = pb.AdaptationCommand(
        command_type=pb.COMMAND_SEED,
        target_seed_id="seed-3",
    )
    command.seed_operation.operation = pb.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "bp-3"
    command.annotations["training_run_id"] = "run-3"

    manager.handle_command(command)
    manager.finalize_step()
    # Dispatcher path should queue events for the seed
    packets = manager.drain_telemetry_packets()
    assert packets  # at least one packet exists
