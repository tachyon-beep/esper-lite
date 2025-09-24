from __future__ import annotations

import torch
from torch import nn

from esper.kasmina import KasminaSeedManager, SeedContext
from esper.leyline import leyline_pb2 as pb


class _Runtime:
    def fetch_kernel(self, *_args, **_kwargs):
        return nn.Identity(), 0.0


def _advance_to(mgr: KasminaSeedManager, seed_id: str, stage: int) -> None:
    ctx = mgr._seeds.setdefault(seed_id, SeedContext(seed_id))
    # Drive transitions through legal path
    lc = ctx.lifecycle
    path = [
        pb.SEED_STAGE_GERMINATED,
        pb.SEED_STAGE_TRAINING,
        pb.SEED_STAGE_BLENDING,
        pb.SEED_STAGE_SHADOWING,
        pb.SEED_STAGE_PROBATIONARY,
    ]
    for s in path:
        if lc.state == stage:
            break
        if s not in mgr._lifecycle.allowed_next(lc.state) if hasattr(mgr, "_lifecycle") else ():
            # Fallback: use lifecycle instance directly
            pass
        try:
            lc.transition(s)
        except Exception:
            continue
        if s == stage:
            break


def test_run_probe_disables_autograd_in_shadowing() -> None:
    mgr = KasminaSeedManager(_Runtime(), fallback_blueprint_id=None)
    seed_id = "seed-probe"
    _advance_to(mgr, seed_id, pb.SEED_STAGE_SHADOWING)

    # Function that would normally produce a tensor requiring grad
    def fn() -> torch.Tensor:
        x = torch.randn(2, 2, requires_grad=True)
        y = x * 2.0
        return y

    out = mgr.run_probe(seed_id, fn)
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad is False


def test_run_probe_respects_grad_in_training() -> None:
    mgr = KasminaSeedManager(_Runtime(), fallback_blueprint_id=None)
    seed_id = "seed-train"
    _advance_to(mgr, seed_id, pb.SEED_STAGE_TRAINING)

    def fn() -> torch.Tensor:
        x = torch.randn(2, 2, requires_grad=True)
        return (x * 3.0).sum()

    out = mgr.run_probe(seed_id, fn)
    assert out.requires_grad is True

