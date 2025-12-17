"""Type definitions for PolicyBundle interface."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Result from policy action selection.

    Attributes:
        action: Dict mapping head names to selected action indices
        log_prob: Dict mapping head names to log probabilities
        value: State value estimate
        hidden: New hidden state tuple (h, c) or None for stateless policies
    """
    action: dict[str, int]
    log_prob: dict[str, torch.Tensor]
    value: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor] | None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result from policy action evaluation (for PPO training).

    Attributes:
        log_prob: Dict mapping head names to log probabilities
        value: State value estimate
        entropy: Dict mapping head names to entropy values
        hidden: New hidden state or None
    """
    log_prob: dict[str, torch.Tensor]
    value: torch.Tensor
    entropy: dict[str, torch.Tensor]
    hidden: tuple[torch.Tensor, torch.Tensor] | None


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Result from policy forward pass (distribution params without sampling).

    Used by off-policy algorithms (SAC) that need to compute log_prob
    of sampled actions.

    Attributes:
        logits: Dict mapping head names to raw logits
        value: State value estimate
        hidden: New hidden state or None
    """
    logits: dict[str, torch.Tensor]
    value: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor] | None


__all__ = [
    "ActionResult",
    "EvalResult",
    "ForwardResult",
]
