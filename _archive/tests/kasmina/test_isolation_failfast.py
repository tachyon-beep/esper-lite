from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign

_SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-isolation-test")


def _make_seed_command() -> leyline_pb2.AdaptationCommand:
    command = leyline_pb2.AdaptationCommand(
        version=1,
        command_id="cmd-seed-iso",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id="seed-iso",
    )
    command.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    command.seed_operation.blueprint_id = "BP-ISO"
    command.annotations["training_run_id"] = "iso-run"
    command.annotations["mesh_host_layers"] = json.dumps(["weight", "bias"])
    command.issued_at.GetCurrentTime()
    command.annotations["signature"] = sign(
        command.SerializeToString(deterministic=True), _SIGNING_CONTEXT
    )
    return command


def test_isolation_violation_triggers_failfast() -> None:
    runtime = type(
        "_Runtime",
        (),
        {"fetch_kernel": staticmethod(lambda *_: (nn.Identity(), 0.0))},
    )()
    kasmina = KasminaSeedManager(
        runtime=runtime,
        signing_context=_SIGNING_CONTEXT,
        fail_fast_isolation=True,
    )

    host = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(host.parameters(), lr=0.01)

    kasmina.register_host_model(host)
    kasmina.register_optimizer(optimizer)
    kasmina.handle_command(_make_seed_command())

    with pytest.raises(RuntimeError):
        kasmina.record_isolation_violation("seed-iso")

    assert "seed-iso" in kasmina.seeds()
