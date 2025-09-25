from __future__ import annotations

import torch

from esper.tolaria.lr_controller import build_controller


class _DummyOpt:
    def __init__(self, lr: float = 0.1) -> None:
        self.param_groups = [{"lr": lr}]


def test_lr_controller_warmup_and_cosine() -> None:
    opt = _DummyOpt(0.1)
    ctrl = build_controller(opt, policy="cosine", warmup_steps=2, t_max=10)
    assert ctrl is not None
    # Warmup: step 0,1 increase up to base
    lr0 = ctrl.apply(0, 0)
    lr1 = ctrl.apply(1, 0)
    assert lr1 >= lr0
    # After warmup: cosine decay
    lr2 = ctrl.apply(2, 0)
    lr3 = ctrl.apply(3, 0)
    assert lr3 <= lr2
