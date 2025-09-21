from __future__ import annotations

import inspect

from esper.kasmina import KasminaLifecycle, KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign

_SIGNING_CONTEXT = SignatureContext(secret=b"kasmina-leyline-test")


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
    # Create a seed path to a known state (germinated → training → blending)
    from esper.leyline import leyline_pb2 as pb

    cmd = pb.AdaptationCommand(version=1, command_id="c", command_type=pb.COMMAND_SEED, target_seed_id="seed-a")
    cmd.seed_operation.operation = pb.SEED_OP_GERMINATE
    cmd.seed_operation.blueprint_id = "bp-1"
    cmd.issued_at.GetCurrentTime()
    cmd.annotations["signature"] = sign(cmd.SerializeToString(), _SIGNING_CONTEXT)
    mgr.handle_command(cmd)
    exported = mgr.export_seed_states()
    assert exported
    assert all(isinstance(s.stage, int) for s in exported)
    valid_numbers = {v.number for v in leyline_pb2.SeedLifecycleStage.DESCRIPTOR.values}
    assert all(s.stage in valid_numbers for s in exported)
