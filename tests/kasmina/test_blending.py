from __future__ import annotations

import pytest
import torch

from esper.kasmina.blending import AlphaBlender, AlphaSchedule


def test_alpha_schedule_sigmoid_progression() -> None:
    schedule = AlphaSchedule(total_steps=10, temperature=2.0)
    start = schedule.value(0)
    mid = schedule.value(5)
    end = schedule.value(10)
    assert start < mid < end <= 1.0


def test_alpha_blender_detaches_host_branch() -> None:
    host = torch.tensor([1.0], requires_grad=True)
    seed = torch.tensor([2.0], requires_grad=True)
    blended = AlphaBlender().blend(host, seed, alpha=0.3)
    blended.backward()
    assert host.grad is None or torch.allclose(host.grad, torch.zeros_like(host.grad))
    assert pytest.approx(seed.grad.item(), rel=1e-5) == 0.3
