"""Heuristic PolicyBundle for ablations and debugging.

Wraps HeuristicTamiyo as a PolicyBundle for the registry.
"""

from __future__ import annotations

from typing import Any

import torch

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.registry import register_policy
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig


@register_policy("heuristic")
class HeuristicPolicyBundle:
    """Rule-based heuristic policy for ablations and debugging.

    This PolicyBundle wraps HeuristicTamiyo, providing the standard
    PolicyBundle interface for a non-learning baseline.

    Note: This policy does not have learnable parameters. Methods like
    evaluate_actions() and get_value() return dummy values.
    """

    def __init__(
        self,
        config: HeuristicPolicyConfig | None = None,
        topology: str = "cnn",
    ):
        """Initialize heuristic policy bundle.

        Args:
            config: Heuristic policy configuration
            topology: Model topology ("cnn" or "transformer")
        """
        self._heuristic = HeuristicTamiyo(config, topology)
        self._topology = topology

    # === Action Selection ===

    def get_action(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> ActionResult:
        """Not directly usable - heuristic needs TrainingSignals.

        Use decide_from_signals() instead.
        """
        raise NotImplementedError(
            "HeuristicPolicyBundle.get_action() is not supported. "
            "Use the training loop's heuristic path instead."
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ForwardResult:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no forward pass")

    # === On-Policy ===

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> EvalResult:
        """Not supported for heuristic (no learnable parameters)."""
        raise NotImplementedError("Heuristic has no learnable parameters")

    # === Off-Policy ===

    def get_q_values(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no Q-values")

    def sync_from(self, source: "PolicyBundle", tau: float = 0.005) -> None:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no target network")

    # === Value Estimation ===

    def get_value(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Not supported for heuristic."""
        raise NotImplementedError("Heuristic has no value function")

    # === Recurrent State ===

    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Heuristic is stateless - returns None (no hidden state)."""
        return None

    # === Serialization ===

    def state_dict(self) -> dict[str, Any]:
        """Heuristic has no learnable state."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Heuristic has no learnable state."""
        pass

    # === Device Management ===

    @property
    def device(self) -> torch.device:
        """Heuristic runs on CPU."""
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "HeuristicPolicyBundle":
        """No-op for heuristic."""
        return self

    # === Introspection ===

    @property
    def is_recurrent(self) -> bool:
        """Heuristic is stateless."""
        return False

    @property
    def supports_off_policy(self) -> bool:
        """Heuristic doesn't support any training."""
        return False

    @property
    def dtype(self) -> torch.dtype:
        """Return float32 for compatibility."""
        return torch.float32

    # === Optional ===

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """No-op for heuristic."""
        pass

    # === Heuristic-specific ===

    @property
    def heuristic(self) -> HeuristicTamiyo:
        """Access underlying heuristic for direct decision-making."""
        return self._heuristic

    def reset(self) -> None:
        """Reset heuristic state."""
        self._heuristic.reset()


__all__ = ["HeuristicPolicyBundle"]
