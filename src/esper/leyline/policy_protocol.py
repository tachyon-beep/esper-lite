"""Leyline Policy Protocol - Contract for swappable Tamiyo policies.

This protocol defines the interface for Tamiyo policy implementations.
Tamiyo is the "brain" of Esper - she makes strategic decisions about
seed lifecycle (germinate, fossilize, prune, wait).

Different PolicyBundle implementations provide different strategies:
- LSTMPolicyBundle: Recurrent neural policy with temporal memory
- MLPPolicyBundle: Stateless feedforward policy (simpler baseline)
- HeuristicPolicyBundle: Rule-based expert system (for ablations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from torch import nn

    from esper.leyline.slot_config import SlotConfig


# =============================================================================
# Result Types (returned by PolicyBundle methods)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Result from policy action selection.

    Attributes:
        action: Dict mapping head names to selected action tensors [batch] or [batch, seq]
        log_prob: Dict mapping head names to log probabilities
        value: State value estimate
        hidden: New hidden state tuple (h, c) or None for stateless policies
        op_logits: Raw masked logits for op head [batch, num_ops].
            Used by telemetry for decision snapshots (action confidence, alternatives).
            Only populated if policy bundle supports it, otherwise None.
    """

    action: dict[str, torch.Tensor]
    log_prob: dict[str, torch.Tensor]
    value: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor] | None
    op_logits: torch.Tensor | None = None


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


# =============================================================================
# PolicyBundle Protocol
# =============================================================================


@runtime_checkable
class PolicyBundle(Protocol):
    """Interface for swappable Tamiyo policy implementations.

    Tamiyo is the "brain" of Esper - she makes strategic decisions about
    seed lifecycle (germinate, fossilize, prune, wait). Different PolicyBundle
    implementations provide different decision-making strategies:

    - LSTMPolicyBundle: Recurrent neural policy with temporal memory
    - MLPPolicyBundle: Stateless feedforward policy (simpler baseline)
    - HeuristicPolicyBundle: Rule-based expert system (for ablations)

    ## Design Rationale (from expert review)

    - Protocol over ABC: Avoids MRO conflicts with nn.Module inheritance
    - runtime_checkable: Enables validation at policy registration time
    - Explicit state_dict: Required for checkpoint compatibility
    - Device management: Essential for multi-GPU and distributed training

    ## Adding a New Policy

    1. Create `tamiyo/policy/my_bundle.py`
    2. Implement the PolicyBundle protocol
    3. Decorate with @register_policy("my_policy")
    4. Add to config: {"tamiyo": {"policy": "my_policy", ...}}

    ## On-Policy vs Off-Policy

    Policies declare their capabilities via `supports_off_policy`.
    On-policy algorithms (PPO) use `evaluate_actions()`.
    Off-policy algorithms (SAC) use `get_q_values()` and `forward()`.

    ## Recurrent Policies

    Recurrent policies (LSTM) maintain hidden state across steps.
    They must implement `initial_hidden()` and set `is_recurrent = True`.
    Simic handles hidden state threading during rollout collection.

    ## torch.compile Guidance

    Compile the inner nn.Module, NOT the PolicyBundle wrapper.
    Keep torch.compile() calls in Simic (training infrastructure).
    """

    # === Action Selection (both paradigms) ===
    #
    # Note: Feature extraction (signals → features) is handled by Simic's
    # signals_to_features() which requires training context (slot_reports,
    # telemetry settings, max_epochs, etc.) that the PolicyBundle doesn't have.
    # PolicyBundle receives pre-computed tensor features + blueprint indices,
    # not raw TrainingSignals.
    def get_action(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
        probability_floor: dict[str, float] | None = None,
    ) -> ActionResult:
        """Select action given observations.

        Uses inference_mode internally - returned tensors are non-differentiable.

        Args:
            features: State features [batch, feature_dim]
            blueprint_indices: Blueprint indices [batch, num_slots]
            masks: Dict of boolean masks for action heads
            hidden: Optional recurrent hidden state
            deterministic: If True, use argmax instead of sampling
            probability_floor: Optional per-head minimum probability. Must match
                what is passed to evaluate_actions() for consistent log_probs.
        """
        ...

    # === Forward (for off-policy) ===
    def forward(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> ForwardResult:
        """Compute action distribution parameters without sampling.

        Required for:
        - SAC: Computing log_prob of sampled actions for entropy bonus
        - TD3: Getting deterministic action for target policy
        - Offline RL: Computing action distribution for OOD detection

        Returns:
            ForwardResult with logits per head, value, new_hidden
        """
        ...

    # === On-Policy (PPO/A2C) ===
    def evaluate_actions(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        probability_floor: dict[str, float] | None = None,
    ) -> EvalResult:
        """Evaluate actions for PPO update.

        Args:
            features: State features [batch, seq_len, feature_dim]
            blueprint_indices: Blueprint indices per slot [batch, seq_len, num_slots]
            actions: Dict mapping head names to action tensors [batch, seq_len]
            masks: Dict mapping head names to boolean masks
            hidden: Optional recurrent hidden state
            probability_floor: Optional dict mapping head names to minimum probability
                values. When provided, all valid actions for that head are guaranteed
                at least this probability, ensuring gradient flow even when entropy
                would otherwise collapse. Typical: {"blueprint": 0.10, "tempo": 0.10}

        Must be called with gradient tracking enabled (not in inference_mode).
        """
        ...

    # === Off-Policy (SAC/TD3) ===
    def get_q_values(
        self,
        features: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Twin Q-values for off-policy critic.

        Returns (Q1, Q2) for clipped double-Q learning.
        Raises NotImplementedError if supports_off_policy is False.
        """
        ...

    def sync_from(self, source: "PolicyBundle", tau: float = 0.005) -> None:
        """Polyak averaging update from source policy (for target networks).

        target = tau * source + (1 - tau) * target

        Required for SAC/TD3 target network updates.
        """
        ...

    # === Value Estimation ===
    def get_value(
        self,
        features: torch.Tensor,
        blueprint_indices: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """State value estimate for baseline."""
        ...

    # === Recurrent State ===
    def initial_hidden(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Initial hidden state for recurrent policies (None if stateless).

        Note:
            This method returns **inference-mode tensors** that are NOT suitable
            for gradient computation. For training:

            - Rollout collection: Use initial_hidden() - no gradients needed
            - PPO update: Pass hidden=None to evaluate_actions() - the network
              creates gradient-compatible hidden states internally

            This design prevents accidental gradient tracking during rollout
            while ensuring correct autograd behavior during training.
        """
        ...

    # === Serialization ===
    def state_dict(self) -> dict[str, Any]:
        """Return policy state for checkpointing."""
        ...

    def load_state_dict(
        self, state_dict: dict[str, Any], strict: bool = True
    ) -> None:
        """Load policy state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
            strict: If True, keys must match exactly. If False, allows partial loading.
        """
        ...

    # === Device Management ===
    @property
    def device(self) -> torch.device:
        """Device where policy parameters reside."""
        ...

    def to(self, device: torch.device | str) -> "PolicyBundle":
        """Move policy to specified device. Returns self for chaining."""
        ...

    # === Introspection ===
    @property
    def is_recurrent(self) -> bool:
        """True if policy maintains hidden state across steps."""
        ...

    @property
    def supports_off_policy(self) -> bool:
        """True if policy supports off-policy algorithms (SAC/TD3).

        If False, get_q_values() and sync_from() raise NotImplementedError.
        """
        ...

    @property
    def dtype(self) -> torch.dtype:
        """Data type of policy parameters (for AMP compatibility)."""
        ...

    # === Configuration Access ===
    @property
    def slot_config(self) -> "SlotConfig":
        """Slot configuration for action masking.

        Required by PPOAgent for buffer construction and action mask validation.
        """
        ...

    @property
    def feature_dim(self) -> int:
        """Input feature dimension.

        This is the observation feature size that the policy expects.
        Used for buffer construction and network dimension validation.
        """
        ...

    @property
    def hidden_dim(self) -> int:
        """Hidden state dimension (for recurrent policies).

        For non-recurrent policies, return 0 or raise NotImplementedError.
        Used by PPOAgent for buffer construction.
        """
        ...

    # === Network Access (for training infrastructure) ===
    @property
    def network(self) -> "nn.Module":
        """Access underlying nn.Module for training.

        **Why exposed**: PPOAgent needs direct module access for:
        - Creating optimizer: optimizer = Adam(policy.network.parameters())
        - Gradient clipping: clip_grad_norm_(policy.network.parameters())
        - torch.compile: torch.compile(policy.network)

        **Warning**: This is an intentional abstraction leak. Training
        infrastructure (Simic) may access the network directly, but
        higher-level code (Tamiyo decision logic) should not.

        Implementation note: If using torch.compile, the compiled module
        wraps the original. Use getattr(network, '_orig_mod', network)
        to access the original module when needed.
        """
        ...

    # === Optional: Gradient Checkpointing ===
    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable/disable gradient checkpointing for memory efficiency.

        Optional - policies that don't support this should no-op.
        Primarily useful for Transformer-based policies.
        """
        ...

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
        ...

    @property
    def is_compiled(self) -> bool:
        """True if the network has been compiled with torch.compile."""
        ...


__all__ = [
    "ActionResult",
    "EvalResult",
    "ForwardResult",
    "PolicyBundle",
]
