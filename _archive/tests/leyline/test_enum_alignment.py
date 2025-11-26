from __future__ import annotations

import inspect
import json

import torch
from torch import nn

from esper.kasmina import KasminaLifecycle, KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign

_SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-leyline-test")


def _sign_command(command: leyline_pb2.AdaptationCommand) -> None:
    if "signature" in command.annotations:
        del command.annotations["signature"]
    command.issued_at.GetCurrentTime()
    command.annotations["signature"] = sign(
        command.SerializeToString(deterministic=True),
        _SIGNING_CONTEXT,
    )


def test_kasmina_uses_leyline_enums_exclusively() -> None:
    lc = KasminaLifecycle()
    # Unknown should only promote into canonical Leyline enums
    allowed = set(lc.allowed_next(leyline_pb2.SEED_STAGE_UNKNOWN))
    assert allowed == {leyline_pb2.SEED_STAGE_DORMANT}
    assert isinstance(lc.state, int)
    assert lc.state == leyline_pb2.SEED_STAGE_DORMANT


def test_seed_state_export_uses_leyline_enum() -> None:
    class _Runt:
        def fetch_kernel(self, blueprint_id: str):
            import torch.nn as nn

            return nn.Identity(), 1.0

    mgr = KasminaSeedManager(runtime=_Runt(), signing_context=_SIGNING_CONTEXT)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mgr.register_host_model(model)
    mgr.register_optimizer(optimizer)
    # Create a seed path to a known state (germinated → training → blending)
    from esper.leyline import leyline_pb2 as pb

    cmd = pb.AdaptationCommand(
        version=1, command_id="c", command_type=pb.COMMAND_SEED, target_seed_id="seed-a"
    )
    cmd.seed_operation.operation = pb.SEED_OP_GERMINATE
    cmd.seed_operation.blueprint_id = "bp-1"
    cmd.annotations["training_run_id"] = "enum-test"
    cmd.annotations["feature_coverage"] = "0.5"
    cmd.annotations["coverage_types"] = '{"node.seed":1}'
    mesh_layers = sorted(model.state_dict().keys()) or ["__root__"]
    cmd.annotations["mesh_host_layers"] = json.dumps(mesh_layers)
    _sign_command(cmd)
    mgr.handle_command(cmd)
    mgr.finalize_step(step_index=0)
    exported = mgr.export_seed_states()
    assert exported
    assert all(isinstance(s.stage, int) for s in exported)
    valid_numbers = {v.number for v in leyline_pb2.SeedLifecycleStage.DESCRIPTOR.values}
    assert all(s.stage in valid_numbers for s in exported)
