from __future__ import annotations

import torch
from torch import nn

from esper.kasmina import KasminaSeedManager
from esper.leyline import leyline_pb2
from esper.security.signing import SignatureContext, sign


SIGN = SignatureContext(secret=b"kasmina-prewarm-test")


class _Runtime:
    def fetch_kernel(self, blueprint_id: str):
        return nn.Linear(4, 4), 0.0

    def get_prewarm_batch(self, blueprint_id: str):  # representative batch
        return torch.randn(2, 4)


def _cmd(seed_id: str, blueprint_id: str) -> leyline_pb2.AdaptationCommand:
    cmd = leyline_pb2.AdaptationCommand(
        version=1,
        command_id=f"cmd-{seed_id}",
        command_type=leyline_pb2.COMMAND_SEED,
        target_seed_id=seed_id,
    )
    cmd.seed_operation.operation = leyline_pb2.SEED_OP_GERMINATE
    cmd.seed_operation.blueprint_id = blueprint_id
    cmd.issued_at.GetCurrentTime()
    cmd.annotations["signature"] = sign(cmd.SerializeToString(deterministic=True), SIGN)
    return cmd


def test_prewarm_emits_latency_metric_and_metadata() -> None:
    mgr = KasminaSeedManager(_Runtime(), signing_context=SIGN)
    mgr.register_host_model(nn.Linear(4, 4))
    mgr.handle_command(_cmd("seed-prewarm", "BP-PW"))
    # Flush to emit telemetry
    mgr.finalize_step(step_index=1)
    pkt = mgr.telemetry_packets[-1]
    # Check metric presence
    vals = {m.name: m.value for m in pkt.metrics}
    assert "kasmina.prewarm.latency_ms" in vals
    assert vals["kasmina.prewarm.latency_ms"] >= 0.0
    # Check per-seed metadata
    ctx = mgr.seeds().get("seed-prewarm")
    assert ctx is not None
    assert "prewarm_ms" in ctx.metadata

