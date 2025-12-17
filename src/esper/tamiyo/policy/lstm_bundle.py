"""LSTM-based PolicyBundle implementation.

Wraps FactoredRecurrentActorCritic as a PolicyBundle for the registry.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch import nn

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.registry import register_policy
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.leyline.slot_config import SlotConfig
from esper.leyline import HEAD_NAMES

if TYPE_CHECKING:
    from esper.simic.agent.network import FactoredRecurrentActorCritic


@register_policy("lstm")
class LSTMPolicyBundle:
    """LSTM-based recurrent policy for seed lifecycle control.

    This PolicyBundle wraps FactoredRecurrentActorCritic, providing the
    standard PolicyBundle interface while delegating to the existing
    well-tested network implementation.

    Attributes:
        feature_dim: Input feature dimension
        hidden_dim: LSTM hidden state dimension
        slot_config: Slot configuration for action masking
    """

    def __init__(
        self,
        feature_dim: int = 50,
        hidden_dim: int = 256,
        num_lstm_layers: int = 1,
        slot_config: SlotConfig | None = None,
        dropout: float = 0.0,
    ):
        """Initialize LSTM policy bundle.

        Args:
            feature_dim: Observation feature dimension (state_dim for the network)
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            slot_config: Slot configuration (defaults to SlotConfig.default())
            dropout: Dropout rate for LSTM (currently unused by network)
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.slot_config = slot_config or SlotConfig.default()

        # Lazy import to avoid circular dependency
        from esper.simic.agent.network import FactoredRecurrentActorCritic

        # Create the network
        # Note: The network's first parameter is "state_dim" which is our feature_dim
        self._network = FactoredRecurrentActorCritic(
            state_dim=feature_dim,
            num_slots=self.slot_config.num_slots,
            lstm_hidden_dim=hidden_dim,
            lstm_layers=num_lstm_layers,
        )

    # === Action Selection ===

    def get_action(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> ActionResult:
        """Select action using the LSTM network.

        This method wraps the network's get_action method, converting from
        the dict-based mask format to individual mask parameters.
        """
        # Convert dict masks to individual parameters
        result = self._network.get_action(
            features,
            hidden,
            slot_mask=masks.get("slot"),
            blueprint_mask=masks.get("blueprint"),
            blend_mask=masks.get("blend"),
            op_mask=masks.get("op"),
            deterministic=deterministic,
        )

        # Network returns (actions, log_probs, value, hidden)
        actions, log_probs, value, new_hidden = result

        return ActionResult(
            action=actions,
            log_prob=log_probs,
            value=value,
            hidden=new_hidden,
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ForwardResult:
        """Forward pass returning distribution parameters.

        For off-policy algorithms that need to compute log_prob separately.
        """
        output = self._network.forward(
            features,
            hidden,
            slot_mask=masks.get("slot"),
            blueprint_mask=masks.get("blueprint"),
            blend_mask=masks.get("blend"),
            op_mask=masks.get("op"),
        )

        return ForwardResult(
            logits={head: output[f"{head}_logits"] for head in HEAD_NAMES},
            value=output["value"],
            hidden=output["hidden"],
        )

    # === On-Policy (PPO) ===

    def evaluate_actions(
        self,
        features: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> EvalResult:
        """Evaluate actions for PPO training.

        This method wraps the network's evaluate_actions method.
        """
        log_probs, values, entropies, new_hidden = self._network.evaluate_actions(
            features,
            actions,
            slot_mask=masks.get("slot"),
            blueprint_mask=masks.get("blueprint"),
            blend_mask=masks.get("blend"),
            op_mask=masks.get("op"),
            hidden=hidden,
        )

        return EvalResult(
            log_prob=log_probs,
            value=values,
            entropy=entropies,
            hidden=new_hidden,
        )

    # === Off-Policy (not supported for LSTM) ===

    def get_q_values(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Not supported for LSTM policy."""
        raise NotImplementedError(
            "LSTMPolicyBundle does not support off-policy algorithms. "
            "Off-policy support requires a future MLP-based PolicyBundle."
        )

    def sync_from(self, source: "PolicyBundle", tau: float = 0.005) -> None:
        """Not supported for LSTM policy."""
        raise NotImplementedError(
            "LSTMPolicyBundle does not support target network updates. "
            "Off-policy support requires a future MLP-based PolicyBundle."
        )

    # === Value Estimation ===

    def get_value(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Get state value estimate.

        Note: The network doesn't have a standalone get_value method,
        so we'll do a forward pass and extract the value.
        """
        # Need to add seq_len dimension if not present
        if features.dim() == 2:
            features = features.unsqueeze(1)

        output = self._network.forward(features, hidden)
        # value is [batch, seq_len], return [batch]
        return output["value"][:, 0] if output["value"].dim() > 1 else output["value"]

    # === Recurrent State ===

    @torch.inference_mode()
    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get initial LSTM hidden state."""
        return self._network.get_initial_hidden(batch_size, self.device)

    # === Serialization ===

    def state_dict(self) -> dict[str, Any]:
        """Return network state dict."""
        # Handle torch.compile wrapper
        base = getattr(self._network, '_orig_mod', self._network)
        return base.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load network state dict."""
        base = getattr(self._network, '_orig_mod', self._network)
        base.load_state_dict(state_dict, strict=strict)

    # === Device Management ===

    @property
    def device(self) -> torch.device:
        """Get device of network parameters."""
        return next(self._network.parameters()).device

    def to(self, device: torch.device | str) -> "LSTMPolicyBundle":
        """Move network to device."""
        self._network = self._network.to(device)
        return self

    # === Introspection ===

    @property
    def is_recurrent(self) -> bool:
        """LSTM is recurrent."""
        return True

    @property
    def supports_off_policy(self) -> bool:
        """LSTM does not support off-policy (needs R2D2 machinery)."""
        return False

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of network parameters."""
        return next(self._network.parameters()).dtype

    # === Optional: Gradient Checkpointing ===

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """No-op for LSTM (gradient checkpointing not beneficial)."""
        pass

    # === Network Access (for Simic's torch.compile) ===

    @property
    def network(self) -> nn.Module:
        """Access underlying network for torch.compile."""
        return self._network


__all__ = ["LSTMPolicyBundle"]
