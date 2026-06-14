"""Regression tests for recurrent PPO state handling."""

from __future__ import annotations

import torch

from esper.simic.training.vectorized_trainer import _reset_hidden_for_terminal_envs


def test_reset_hidden_for_terminal_envs_zeroes_only_rolled_back_envs() -> None:
    """Terminal rollback must not leak stale recurrent context into later steps."""

    h = torch.ones(2, 3, 4)
    c = torch.full((2, 3, 4), 2.0)

    reset_h, reset_c = _reset_hidden_for_terminal_envs((h, c), terminal_envs=[1])

    assert torch.equal(reset_h[:, 0, :], torch.ones(2, 4))
    assert torch.equal(reset_c[:, 0, :], torch.full((2, 4), 2.0))
    assert torch.equal(reset_h[:, 1, :], torch.zeros(2, 4))
    assert torch.equal(reset_c[:, 1, :], torch.zeros(2, 4))
    assert torch.equal(reset_h[:, 2, :], torch.ones(2, 4))
    assert torch.equal(reset_c[:, 2, :], torch.full((2, 4), 2.0))

