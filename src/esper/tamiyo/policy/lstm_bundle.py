"""LSTM-based PolicyBundle implementation.

Wraps FactoredRecurrentActorCritic as a PolicyBundle for the registry.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from esper.leyline import PolicyBundle, ActionResult, EvalResult, ForwardResult
from esper.tamiyo.policy.registry import register_policy
from esper.leyline.slot_config import SlotConfig
from esper.leyline import DEFAULT_LSTM_HIDDEN_DIM


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
        feature_dim: int | None = None,
        hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
        num_lstm_layers: int = 1,
        slot_config: SlotConfig | None = None,
        dropout: float = 0.0,
    ):
        """Initialize LSTM policy bundle.

        Args:
            feature_dim: Observation feature dimension. If None, auto-computed from
                slot_config using get_feature_size(). This ensures the network input
                dimension matches the feature extractor output.
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            slot_config: Slot configuration (defaults to SlotConfig.default())
            dropout: Dropout rate for LSTM (currently unused by network)
        """
        self._hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self._slot_config = slot_config or SlotConfig.default()

        # Auto-compute feature_dim from slot_config if not provided
        # This prevents drift between feature extraction and network input dimensions
        if feature_dim is None:
            from esper.tamiyo.policy.features import get_feature_size
            feature_dim = get_feature_size(self._slot_config)

        self._feature_dim = feature_dim

        # Lazy import to avoid circular dependency
        from esper.tamiyo.networks import FactoredRecurrentActorCritic

        # Create the network
        # Note: The network's first parameter is "state_dim" which is our feature_dim
        self._network = FactoredRecurrentActorCritic(
            state_dim=feature_dim,
            slot_config=self.slot_config,
            lstm_hidden_dim=hidden_dim,
            lstm_layers=num_lstm_layers,
        )

    # === Action Selection ===

    def get_action(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
        probability_floor: dict[str, float] | None = None,
    ) -> ActionResult:
        """Select action using the LSTM network.

        This method wraps the network's get_action method, converting from
        the dict-based mask format to individual mask parameters.

        Args:
            features: Input features [batch, feature_dim]
            blueprint_indices: Blueprint indices per slot [batch, num_slots]
            masks: Dict of boolean masks for each action head
            hidden: Optional LSTM hidden state
            deterministic: If True, use argmax instead of sampling
            probability_floor: Optional per-head minimum probability. Must match
                what is passed to evaluate_actions() for consistent log_probs.

        Returns:
            ActionResult with actions, log_probs, value, hidden, and op_logits
        """
        # Convert dict masks to individual parameters
        # Request op_logits for telemetry (decision snapshots need action probabilities)
        # NOTE: Direct dict access (not .get()) to fail fast if caller provides incomplete masks
        result = self._network.get_action(
            features,
            blueprint_indices,
            hidden,
            slot_mask=masks["slot"],
            slot_by_op_mask=masks["slot_by_op"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            op_mask=masks["op"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            deterministic=deterministic,
            return_op_logits=True,
            probability_floor=probability_floor,
        )

        # Network returns GetActionResult dataclass
        return ActionResult(
            action=result.actions,
            log_prob=result.log_probs,
            value=result.values,
            hidden=result.hidden,
            op_logits=result.op_logits,
        )

    def forward(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ForwardResult:
        """Forward pass returning distribution parameters.

        For off-policy algorithms that need to compute log_prob separately.

        Args:
            features: Input features [batch, seq_len, feature_dim] or [batch, feature_dim].
                2D inputs are automatically expanded to 3D with seq_len=1.
            blueprint_indices: Blueprint indices [batch, seq_len, num_slots] or [batch, num_slots].
                2D inputs are automatically expanded to 3D.
            masks: Dict of boolean masks for each action head. Masks should be
                [batch, action_dim] or [batch, seq_len, action_dim]. 2D masks are
                automatically expanded to match features.
            hidden: Optional LSTM hidden state.

        Returns:
            ForwardResult with logits, value, and new hidden state.
        """
        # Normalize 2D inputs to 3D (add seq_len=1 dimension)
        # Network expects [batch, seq_len, state_dim]
        need_expand = features.dim() == 2
        if need_expand:
            features = features.unsqueeze(1)

        # Expand blueprint_indices to 3D if needed
        if blueprint_indices.dim() == 2:
            blueprint_indices = blueprint_indices.unsqueeze(1)

        # Normalize masks to 3D based on mask rank, not features rank
        # FIX: Previously gated on `need_expand` (features was 2D), but if features
        # is already 3D [B, 1, F] and masks are 2D [B, A], the mask wasn't expanded,
        # causing PyTorch broadcasting to align [B, A] against [B, 1, A] incorrectly.
        def expand_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
            if mask is None:
                return None
            if mask.dim() == 2:
                return mask.unsqueeze(1)  # [B, A] -> [B, 1, A]
            return mask

        output = self._network.forward(
            features,
            blueprint_indices,
            hidden,
            # NOTE: Direct dict access (not .get()) to fail fast if caller provides incomplete masks
            slot_mask=expand_mask(masks["slot"]),
            blueprint_mask=expand_mask(masks["blueprint"]),
            style_mask=expand_mask(masks["style"]),
            tempo_mask=expand_mask(masks["tempo"]),
            op_mask=expand_mask(masks["op"]),
            alpha_target_mask=expand_mask(masks["alpha_target"]),
            alpha_speed_mask=expand_mask(masks["alpha_speed"]),
            alpha_curve_mask=expand_mask(masks["alpha_curve"]),
        )

        # Build logits dict with explicit key access to satisfy mypy TypedDict constraints
        logits_dict = {
            "slot": output["slot_logits"],
            "blueprint": output["blueprint_logits"],
            "style": output["style_logits"],
            "tempo": output["tempo_logits"],
            "alpha_target": output["alpha_target_logits"],
            "alpha_speed": output["alpha_speed_logits"],
            "alpha_curve": output["alpha_curve_logits"],
            "op": output["op_logits"],
        }

        return ForwardResult(
            logits=logits_dict,
            value=output["value"],
            hidden=output["hidden"],
        )

    # === On-Policy (PPO) ===

    def evaluate_actions(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        probability_floor: dict[str, float] | None = None,
    ) -> EvalResult:
        """Evaluate actions for PPO training.

        This method wraps the network's evaluate_actions method.

        Args:
            features: Input features [batch, seq_len, feature_dim]
            blueprint_indices: Blueprint indices [batch, seq_len, num_slots]
            actions: Stored actions from buffer
            masks: Dict of boolean masks for each action head
            hidden: Optional initial LSTM hidden state
            probability_floor: Optional dict mapping head names to minimum probability
                values. Passed through to network's evaluate_actions.

        Returns:
            EvalResult with log_probs, value, entropy, and new hidden state
        """
        log_probs, values, entropies, new_hidden = self._network.evaluate_actions(
            features,
            blueprint_indices,
            actions,
            # NOTE: Direct dict access (not .get()) to fail fast if caller provides incomplete masks
            slot_mask=masks["slot"],
            blueprint_mask=masks["blueprint"],
            style_mask=masks["style"],
            tempo_mask=masks["tempo"],
            op_mask=masks["op"],
            alpha_target_mask=masks["alpha_target"],
            alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"],
            hidden=hidden,
            probability_floor=probability_floor,
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

    @torch.inference_mode()
    def get_value(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Get state value estimate.

        Note: Uses inference_mode() since this is only called for bootstrap
        value computation during rollout collection, not during PPO training.
        The network's forward pass for training uses evaluate_actions() instead.

        Args:
            features: Input features [batch, feature_dim]
            blueprint_indices: Blueprint indices [batch, num_slots]
            hidden: Optional LSTM hidden state

        Returns:
            Value estimates [batch]
        """
        # Need to add seq_len dimension if not present
        if features.dim() == 2:
            features = features.unsqueeze(1)

        # Expand blueprint_indices to 3D
        if blueprint_indices.dim() == 2:
            blueprint_indices = blueprint_indices.unsqueeze(1)

        output = self._network.forward(features, blueprint_indices, hidden)
        # value is [batch, seq_len], return [batch]
        return output["value"][:, 0] if output["value"].dim() > 1 else output["value"]

    # === Recurrent State ===

    @torch.inference_mode()
    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get initial LSTM hidden state for rollout collection.

        Returns inference-mode tensors (is_inference()=True) for performance.
        These CANNOT be used directly in autograd - would cause RuntimeError.

        The RolloutBuffer pattern makes this safe:
        1. Buffer pre-allocates regular tensors at initialization
        2. add() copies VALUES to buffer (inference flag not transferred)
        3. get_batched_sequences() returns regular tensors
        4. evaluate_actions() receives gradient-compatible hidden states

        Do NOT bypass the buffer by passing these tensors directly to
        evaluate_actions() - this would cause RuntimeError.
        """
        return self._network.get_initial_hidden(batch_size, self.device)

    # === Serialization ===

    def state_dict(self) -> dict[str, Any]:
        """Return network state dict."""
        # Handle torch.compile wrapper
        # getattr AUTHORIZED by Code Review 2025-12-17
        # Justification: torch.compile wraps modules - must unwrap to access actual state_dict
        base = getattr(self._network, '_orig_mod', self._network)
        return base.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load network state dict."""
        # getattr AUTHORIZED by Code Review 2025-12-17
        # Justification: torch.compile wraps modules - must unwrap to access actual load_state_dict
        base = getattr(self._network, '_orig_mod', self._network)
        base.load_state_dict(state_dict, strict=strict)

    # === Device Management ===

    @property
    def device(self) -> torch.device:
        """Get device of network parameters."""
        return next(self._network.parameters()).device

    def _check_not_compiled(self, method_name: str) -> None:
        """Raise if policy is compiled (device/dtype mutations break compiled graphs)."""
        if self.is_compiled:
            raise RuntimeError(
                f"Cannot call .{method_name}() on a compiled policy. "
                "Create a new policy on the target device and compile after placement."
            )

    def to(self, device: torch.device | str) -> "LSTMPolicyBundle":
        """Move network to device.

        Compiled policies cannot be moved; create a new policy on the target
        device and compile after placement.
        """
        self._check_not_compiled("to")
        self._network = self._network.to(device)
        return self

    def cuda(self, device: int | torch.device | None = None) -> "LSTMPolicyBundle":
        """Move network to CUDA device. Raises if compiled."""
        self._check_not_compiled("cuda")
        self._network = self._network.cuda(device)
        return self

    def cpu(self) -> "LSTMPolicyBundle":
        """Move network to CPU. Raises if compiled."""
        self._check_not_compiled("cpu")
        self._network = self._network.cpu()
        return self

    def half(self) -> "LSTMPolicyBundle":
        """Convert network to float16. Raises if compiled."""
        self._check_not_compiled("half")
        self._network = self._network.half()
        return self

    def float(self) -> "LSTMPolicyBundle":
        """Convert network to float32. Raises if compiled."""
        self._check_not_compiled("float")
        self._network = self._network.float()
        return self

    def double(self) -> "LSTMPolicyBundle":
        """Convert network to float64. Raises if compiled."""
        self._check_not_compiled("double")
        self._network = self._network.double()
        return self

    def bfloat16(self) -> "LSTMPolicyBundle":
        """Convert network to bfloat16. Raises if compiled."""
        self._check_not_compiled("bfloat16")
        self._network = self._network.bfloat16()
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

    # === Configuration Access ===

    @property
    def slot_config(self) -> SlotConfig:
        """Slot configuration for action masking."""
        return self._slot_config

    @property
    def feature_dim(self) -> int:
        """Input feature dimension."""
        return self._feature_dim

    @property
    def hidden_dim(self) -> int:
        """LSTM hidden state dimension."""
        return self._hidden_dim

    # === Optional: Gradient Checkpointing ===

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """No-op for LSTM (gradient checkpointing not beneficial)."""
        pass

    # === Network Access (for Simic's torch.compile) ===

    @property
    def network(self) -> nn.Module:
        """Access underlying network for torch.compile."""
        return self._network

    # === torch.compile Integration ===

    def compile(
        self,
        mode: str = "default",
        dynamic: bool = True,
    ) -> None:
        """Compile the underlying network with torch.compile.

        Must be called AFTER device placement (.to(device)).
        Compilation is idempotent - calling twice is safe.

        Args:
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune", "off")
            dynamic: Enable dynamic shapes for varying batch/sequence lengths
        """
        if mode == "off" or self.is_compiled:
            return
        # torch.compile returns a compiled version but mypy sees it as generic callable
        # Type is actually the same as input (FactoredRecurrentActorCritic)
        compiled_net = torch.compile(self._network, mode=mode, dynamic=dynamic)
        # Safe to reassign since torch.compile preserves the module interface
        self._network = compiled_net  # type: ignore[assignment]

    @property
    def is_compiled(self) -> bool:
        """True if the network has been compiled with torch.compile."""
        # hasattr AUTHORIZED by John on 2025-12-25 00:00:00 UTC
        # Justification: torch.compile detection - OptimizedModule has _orig_mod attr
        return hasattr(self._network, '_orig_mod')


__all__ = ["LSTMPolicyBundle"]
