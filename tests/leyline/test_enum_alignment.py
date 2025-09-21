from __future__ import annotations

import inspect

from esper.kasmina import KasminaLifecycle, KasminaSeedManager
from esper.leyline import leyline_pb2


def test_kasmina_uses_leyline_enums_exclusively() -> None:
    lc = KasminaLifecycle()
    # Allowed next from UNKNOWN includes GERMINATING and CANCELLED only
    allowed = set(lc.allowed_next(leyline_pb2.SEED_STAGE_UNKNOWN))
    assert leyline_pb2.SEED_STAGE_GERMINATING in allowed
    assert leyline_pb2.SEED_STAGE_CANCELLED in allowed
    # Transition must use Leyline numbers
    lc.transition(leyline_pb2.SEED_STAGE_GERMINATING)
    assert isinstance(lc.state, int)
    assert lc.state == leyline_pb2.SEED_STAGE_GERMINATING


def test_seed_state_export_uses_leyline_enum() -> None:
    class _Runt:
        def fetch_kernel(self, blueprint_id: str):
            import torch.nn as nn

            return nn.Identity(), 1.0

    mgr = KasminaSeedManager(runtime=_Runt())
    # Create a seed path to a known state (germinating → graft → stabilize → active)
    from esper.leyline import leyline_pb2 as pb

    cmd = pb.AdaptationCommand(version=1, command_id="c", command_type=pb.COMMAND_SEED, target_seed_id="seed-a")
    cmd.seed_operation.operation = pb.SEED_OP_GERMINATE
    cmd.seed_operation.blueprint_id = "bp-1"
    mgr.handle_command(cmd)
    exported = mgr.export_seed_states()
    assert exported
    assert all(isinstance(s.stage, int) for s in exported)
    valid_numbers = {v.number for v in leyline_pb2.SeedLifecycleStage.DESCRIPTOR.values}
    assert all(s.stage in valid_numbers for s in exported)
