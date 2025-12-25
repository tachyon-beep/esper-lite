"""PolicyBundle protocol for swappable Tamiyo policy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult


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
    # PolicyBundle receives pre-computed features, not raw TrainingSignals.
    def get_action(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> "ActionResult":
        """Select action given observations.

        Uses inference_mode internally - returned tensors are non-differentiable.
        """
        ...

    # === Forward (for off-policy) ===
    def forward(
        self,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> "ForwardResult":
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
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> "EvalResult":
        """Evaluate actions for PPO update.

        Args:
            features: State features [batch, seq_len, feature_dim]
            actions: Dict mapping head names to action tensors [batch, seq_len]
            masks: Dict mapping head names to boolean masks
            hidden: Optional recurrent hidden state

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
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """State value estimate for baseline."""
        ...

    # === Recurrent State ===
    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
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

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
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

    # === Optional: Gradient Checkpointing ===
    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable/disable gradient checkpointing for memory efficiency.

        Optional - policies that don't support this should no-op.
        Primarily useful for Transformer-based policies.
        """
        ...


__all__ = ["PolicyBundle"]
