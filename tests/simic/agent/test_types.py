"""Tests for Simic agent public type contracts."""

from typing import get_type_hints

import torch

from esper.simic.agent.types import ActionDict


def test_action_dict_action_heads_are_tensors() -> None:
    """ActionDict should match runtime batched action tensors."""
    hints = get_type_hints(ActionDict)

    assert hints["slot"] is torch.Tensor
    assert hints["blueprint"] is torch.Tensor
    assert hints["style"] is torch.Tensor
    assert hints["tempo"] is torch.Tensor
    assert hints["alpha_target"] is torch.Tensor
    assert hints["alpha_speed"] is torch.Tensor
    assert hints["alpha_curve"] is torch.Tensor
    assert hints["op"] is torch.Tensor
