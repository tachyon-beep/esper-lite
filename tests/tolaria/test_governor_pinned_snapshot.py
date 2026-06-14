"""P3-SNAP: pinned + non_blocking governor snapshot offload.

Verifies the snapshot stores into persistent PINNED host buffers (fast async DtoH), the
buffers are reused across snapshots, and snapshot->restore is bitwise-equal (the post-copy/
pre-assignment sync guarantees no half-copied buffer is observable).
"""

import pytest
import torch
import torch.nn as nn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pinned snapshot")
def test_snapshot_uses_pinned_buffers_and_round_trips_bitwise():
    from esper.tolaria import TolariaGovernor

    device = torch.device("cuda:0")
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).to(device)
    governor = TolariaGovernor(model)

    # Capture pre-snapshot weights (the good state).
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}

    governor.snapshot()
    assert governor.last_good_state is not None

    # Every tensor in last_good_state is on pinned CPU memory.
    for k, v in governor.last_good_state.items():
        if isinstance(v, torch.Tensor):
            assert v.device.type == "cpu", f"{k} not on CPU"
            assert v.is_pinned(), f"{k} not pinned"

    # The pinned buffer pool is populated and is what last_good_state aliases.
    assert len(governor._pinned_snapshot) > 0
    buf_ids_1 = {k: id(v) for k, v in governor._pinned_snapshot.items()}

    # Corrupt the live weights, then restore -> must match the pre-snapshot weights bitwise.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    for _ in range(5):
        governor.loss_history.append(1.0)
    governor.execute_rollback(env_id=0)

    post = model.state_dict()
    for k in pre:
        torch.testing.assert_close(post[k].float(), pre[k].float(), rtol=0, atol=0)

    # A second snapshot REUSES the same pinned buffers (no realloc; same shapes/dtypes).
    governor.snapshot()
    buf_ids_2 = {k: id(v) for k, v in governor._pinned_snapshot.items()}
    assert buf_ids_1 == buf_ids_2, "pinned buffers were reallocated instead of reused"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_snapshot_stream_contract_still_fires():
    """The default-stream contract (torn-snapshot guard) must still raise under a
    non-default stream, even with the pinned-offload rewrite."""
    from esper.tolaria import TolariaGovernor

    device = torch.device("cuda:0")
    model = nn.Linear(16, 4).to(device)
    governor = TolariaGovernor(model)

    secondary = torch.cuda.Stream(device)
    with torch.cuda.stream(secondary):
        with pytest.raises(RuntimeError, match="non-default CUDA stream"):
            governor.snapshot()
