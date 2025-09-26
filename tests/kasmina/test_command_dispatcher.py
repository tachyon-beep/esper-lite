from __future__ import annotations

from dataclasses import dataclass

import torch

from esper.kasmina.seed_manager import KasminaSeedManager
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
    assert context.legacy_mode is True


def test_command_outcome_defaults() -> None:
    outcome = KasminaSeedManager.CommandOutcome()
    assert outcome.events == []
    assert outcome.handled is False
    assert outcome.seed_id is None


def test_dispatcher_stub_returns_outcome(monkeypatch) -> None:
    monkeypatch.setattr("esper.kasmina.seed_manager._DISPATCHER_EXPERIMENTAL", True)
    manager = KasminaSeedManager(runtime=_Runtime())
    command = pb.AdaptationCommand(command_type=pb.COMMAND_PAUSE)
    manager.handle_command(command)
    # No exception means stub path executed; ensure dispatcher returns default outcome
    outcome = manager._dispatch_command(manager._build_command_context(command))
    assert isinstance(outcome, KasminaSeedManager.CommandOutcome)
    assert outcome.handled is False
    assert outcome.events == []
