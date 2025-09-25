from __future__ import annotations

import torch
from torch import nn

from esper.kasmina import KasminaSeedManager, SeedContext
from esper.leyline import leyline_pb2 as pb


class _Runtime:
    def fetch_kernel(self, blueprint_id: str):
        return nn.Linear(4, 2), 0.0


def _backward_once(host: nn.Module, seed: nn.Module, mgr: KasminaSeedManager, seed_id: str) -> None:
    x = torch.randn(3, 4)
    host_out = host(x)
    seed_out = seed(x)
    y = mgr.blend(host_out, seed_out, seed_id=seed_id)
    y.sum().backward()


def test_isolation_collection_enabled_only_in_training_and_blending() -> None:
    mgr = KasminaSeedManager(_Runtime(), fallback_blueprint_id=None)
    host = nn.Linear(4, 2)
    mgr.register_host_model(host)

    seed_id = "seed-scope"
    seed_module, _ = _Runtime().fetch_kernel("BP-scope")
    # Attach kernel to create isolation session
    mgr._seeds[seed_id] = SeedContext(seed_id)
    mgr._attach_kernel(seed_id, seed_module)

    # Enable collection via BLENDING stage
    ctx = mgr._seeds[seed_id]
    mgr._handle_post_transition(ctx, pb.SEED_STAGE_BLENDING)
    before = mgr.isolation_stats(seed_id)
    _backward_once(host, seed_module, mgr, seed_id)
    after = mgr.isolation_stats(seed_id)
    assert before is not None and after is not None
    # Seed norm should increase when collection is enabled
    assert after.seed_norm >= before.seed_norm

    # Disable collection via SHADOWING stage
    mgr._handle_post_transition(ctx, pb.SEED_STAGE_SHADOWING)
    frozen = mgr.isolation_stats(seed_id)
    _backward_once(host, seed_module, mgr, seed_id)
    again = mgr.isolation_stats(seed_id)
    # Stats should not change when collection is disabled
    assert frozen is not None and again is not None
    assert again.seed_norm == frozen.seed_norm
