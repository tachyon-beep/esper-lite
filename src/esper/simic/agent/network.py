"""Factored Recurrent Actor-Critic for Tamiyo.

Architecture:
    state -> feature_net -> LSTM -> shared_repr
    shared_repr -> slot_head -> slot_logits
    shared_repr -> blueprint_head -> blueprint_logits
    shared_repr -> blend_head -> blend_logits
    shared_repr -> tempo_head -> tempo_logits
    shared_repr -> op_head -> op_logits
    shared_repr -> value_head -> value

Design rationale (DRL expert):
    - Feature extraction reduces state_dim before LSTM
    - LayerNorm before LSTM stabilizes training
    - LSTM learns temporal patterns for 10-20 epoch seed learning
    - All heads share temporal context but specialize on their action space
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn

from esper.tamiyo.policy.action_masks import MaskedCategorical


@dataclass(frozen=True, slots=True)
class GetActionResult:
    """Result from get_action() method.

    Attributes:
        actions: Dict of action indices per head [batch]
        log_probs: Dict of log probs per head [batch] (NON-DIFFERENTIABLE)
        values: Value estimates [batch]
        hidden: Updated hidden state (h, c)
        op_logits: Raw masked logits for op head [batch, num_ops].
            Only populated if return_op_logits=True, otherwise None.
            Use F.softmax(op_logits, dim=-1) to get action probabilities.
    """

    actions: dict[str, torch.Tensor]
    log_probs: dict[str, torch.Tensor]
    values: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]
    op_logits: torch.Tensor | None = None


class _ForwardOutput(TypedDict):
    """Typed dict for forward() return value - enables mypy to track per-key types."""

    slot_logits: torch.Tensor
    blueprint_logits: torch.Tensor
    blend_logits: torch.Tensor
    tempo_logits: torch.Tensor
    op_logits: torch.Tensor
    value: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]

from esper.leyline import (
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_FEATURE_DIM,
    HEAD_NAMES,
    MASKED_LOGIT_VALUE,
)
from esper.leyline.factored_actions import (
    NUM_BLUEPRINTS,
    NUM_BLENDS,
    NUM_OPS,
    NUM_TEMPO,
)
from esper.leyline.slot_config import SlotConfig


class FactoredRecurrentActorCritic(nn.Module):
    """Recurrent actor-critic with factored action heads.

    Uses LSTM for temporal reasoning over 10-20 epoch seed learning cycles.
    All 5 action heads share the same temporal context from the LSTM.
    """

    def __init__(
        self,
        state_dim: int,
        num_slots: int = SlotConfig.default().num_slots,
        num_blueprints: int = NUM_BLUEPRINTS,
        num_blends: int = NUM_BLENDS,
        num_tempo: int = NUM_TEMPO,
        num_ops: int = NUM_OPS,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
        lstm_layers: int = 1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_slots = num_slots
        self.num_blueprints = num_blueprints
        self.num_blends = num_blends
        self.num_tempo = num_tempo
        self.num_ops = num_ops
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # Feature extraction before LSTM (reduces dimensionality)
        # M7: Pre-LSTM LayerNorm stabilizes input distribution to LSTM
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.LayerNorm(feature_dim),  # Normalize BEFORE LSTM
            nn.ReLU(),
        )

        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # M7: Post-LSTM LayerNorm (CRITICAL for training stability)
        # Why TWO LayerNorms (pre + post LSTM)?
        # 1. Pre-LSTM LN: Stabilizes input distribution, helps LSTM gates
        # 2. Post-LSTM LN: Prevents hidden state magnitude drift over 25-epoch sequences
        #
        # This is intentional and follows the "LN everywhere" pattern from transformer
        # literature (Ba et al., 2016). LSTMs particularly benefit from post-output LN
        # because hidden state magnitude can drift in long sequences without it.
        self.lstm_ln = nn.LayerNorm(lstm_hidden_dim)

        # H7: Removed unused max_entropies dict.
        # MaskedCategorical.entropy() already returns normalized entropy internally,
        # so per-head max entropy tracking is unnecessary.

        # Factored action heads (feedforward on LSTM output)
        head_hidden = lstm_hidden_dim // 2
        self.slot_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_slots),
        )
        self.blueprint_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_blueprints),
        )
        self.blend_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_blends),
        )
        self.tempo_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_tempo),
        )
        self.op_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_ops),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller init for output layers (policy stability)
        for head in [self.slot_head, self.blueprint_head, self.blend_head, self.tempo_head, self.op_head]:
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

        # LSTM-specific initialization
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # M9: Set forget gate bias to 1 (helps with long-term memory)
                #
                # PyTorch LSTM packs 4 gate biases concatenated: [input, forget, cell, output]
                # Each gate gets n/4 elements, so the forget gate is at indices n//4 : n//2.
                #
                # Why bias=1 for forget gate? (Gers et al., 2000 "Learning to Forget")
                # - Forget gate controls how much of the previous cell state to retain
                # - Sigmoid(1) ≈ 0.73, so default behavior is "mostly remember"
                # - Without this, LSTM initially forgets too aggressively, hurting long sequences
                # - Critical for our 25-epoch seed learning trajectories
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def get_initial_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized hidden state."""
        h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        return h, c

    def forward(
        self,
        state: torch.Tensor,  # [batch, seq_len, state_dim]
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        slot_mask: torch.Tensor | None = None,
        blueprint_mask: torch.Tensor | None = None,
        blend_mask: torch.Tensor | None = None,
        tempo_mask: torch.Tensor | None = None,
        op_mask: torch.Tensor | None = None,
    ) -> _ForwardOutput:
        """Forward pass returning logits, value, and new hidden state.

        Args:
            state: Input state [batch, seq_len, state_dim]
            hidden: (h, c) tuple, each [num_layers, batch, hidden_dim]
            *_mask: Boolean masks [batch, seq_len, action_dim], True = valid

        Returns:
            _ForwardOutput with slot_logits, blueprint_logits, blend_logits,
            tempo_logits, op_logits, value, and hidden tuple
        """
        batch_size = state.size(0)
        device = state.device

        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, device)

        # Feature extraction
        features = self.feature_net(state)  # [batch, seq_len, feature_dim]

        # LSTM forward
        lstm_out, new_hidden = self.lstm(features, hidden)
        # lstm_out: [batch, seq_len, hidden_dim]

        # LayerNorm on LSTM output (prevents magnitude drift)
        lstm_out = self.lstm_ln(lstm_out)

        # Compute logits for each head
        slot_logits = self.slot_head(lstm_out)
        blueprint_logits = self.blueprint_head(lstm_out)
        blend_logits = self.blend_head(lstm_out)
        tempo_logits = self.tempo_head(lstm_out)
        op_logits = self.op_head(lstm_out)

        # Apply masks using canonical MASKED_LOGIT_VALUE from leyline
        # (See leyline/__init__.py for rationale on value choice for FP16 safety)
        if slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~slot_mask, MASKED_LOGIT_VALUE)
        if blueprint_mask is not None:
            blueprint_logits = blueprint_logits.masked_fill(~blueprint_mask, MASKED_LOGIT_VALUE)
        if blend_mask is not None:
            blend_logits = blend_logits.masked_fill(~blend_mask, MASKED_LOGIT_VALUE)
        if tempo_mask is not None:
            tempo_logits = tempo_logits.masked_fill(~tempo_mask, MASKED_LOGIT_VALUE)
        if op_mask is not None:
            op_logits = op_logits.masked_fill(~op_mask, MASKED_LOGIT_VALUE)

        # Value prediction
        value = self.value_head(lstm_out).squeeze(-1)  # [batch, seq_len]

        return {
            "slot_logits": slot_logits,
            "blueprint_logits": blueprint_logits,
            "blend_logits": blend_logits,
            "tempo_logits": tempo_logits,
            "op_logits": op_logits,
            "value": value,
            "hidden": new_hidden,
        }

    def get_action(
        self,
        state: torch.Tensor,  # [batch, state_dim] or [batch, 1, state_dim]
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        slot_mask: torch.Tensor | None = None,
        blueprint_mask: torch.Tensor | None = None,
        blend_mask: torch.Tensor | None = None,
        tempo_mask: torch.Tensor | None = None,
        op_mask: torch.Tensor | None = None,
        deterministic: bool = False,
        return_op_logits: bool = False,
    ) -> GetActionResult:
        """Sample actions from all heads (inference mode).

        WARNING: This method runs under torch.inference_mode(). The returned
        log_probs are NOT differentiable and cannot be used for backpropagation.
        Use evaluate_actions() for training - it recomputes differentiable log_probs.

        The log_probs returned here are stored as old_log_probs for PPO ratio
        computation, but the actual gradient flows through evaluate_actions().

        Args:
            state: Input state tensor [batch, state_dim] or [batch, 1, state_dim]
            hidden: LSTM hidden state (h, c) or None for initial state
            slot_mask: Boolean mask for slot actions [batch, num_slots]
            blueprint_mask: Boolean mask for blueprint actions [batch, num_blueprints]
            blend_mask: Boolean mask for blend actions [batch, num_blends]
            tempo_mask: Boolean mask for tempo actions [batch, num_tempo]
            op_mask: Boolean mask for op actions [batch, num_ops]
            deterministic: If True, use argmax instead of sampling
            return_op_logits: If True, include raw masked op logits in result
                for telemetry/decision snapshot. Default False for performance.

        Returns:
            GetActionResult with actions, log_probs, values, hidden, and
            optionally op_logits if return_op_logits=True.
        """
        # Ensure 3D input
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        # Reshape masks to [batch, 1, dim] if provided as [batch, dim]
        if slot_mask is not None and slot_mask.dim() == 2:
            slot_mask = slot_mask.unsqueeze(1)
        if blueprint_mask is not None and blueprint_mask.dim() == 2:
            blueprint_mask = blueprint_mask.unsqueeze(1)
        if blend_mask is not None and blend_mask.dim() == 2:
            blend_mask = blend_mask.unsqueeze(1)
        if tempo_mask is not None and tempo_mask.dim() == 2:
            tempo_mask = tempo_mask.unsqueeze(1)
        if op_mask is not None and op_mask.dim() == 2:
            op_mask = op_mask.unsqueeze(1)

        with torch.inference_mode():
            output = self.forward(
                state, hidden, slot_mask, blueprint_mask, blend_mask, tempo_mask, op_mask
            )

            # Sample from each head using MaskedCategorical for safety
            actions: dict[str, torch.Tensor] = {}
            log_probs: dict[str, torch.Tensor] = {}

            masks = {
                "slot": slot_mask[:, 0, :] if slot_mask is not None else None,
                "blueprint": blueprint_mask[:, 0, :] if blueprint_mask is not None else None,
                "blend": blend_mask[:, 0, :] if blend_mask is not None else None,
                "tempo": tempo_mask[:, 0, :] if tempo_mask is not None else None,
                "op": op_mask[:, 0, :] if op_mask is not None else None,
            }

            # Map head names to logits (TypedDict keys must be literals)
            head_logits: dict[str, torch.Tensor] = {
                "slot": output["slot_logits"][:, 0, :],
                "blueprint": output["blueprint_logits"][:, 0, :],
                "blend": output["blend_logits"][:, 0, :],
                "tempo": output["tempo_logits"][:, 0, :],
                "op": output["op_logits"][:, 0, :],
            }

            for key in HEAD_NAMES:
                logits = head_logits[key]  # [batch, action_dim]
                mask = masks[key]
                if mask is None:
                    mask = torch.ones_like(logits, dtype=torch.bool)
                dist = MaskedCategorical(logits=logits, mask=mask)

                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()

                actions[key] = action
                log_probs[key] = dist.log_prob(action)

            value = output["value"][:, 0]  # [batch]
            new_hidden = output["hidden"]

            # Conditionally capture op_logits for telemetry (Decision Snapshot)
            op_logits_out = head_logits["op"] if return_op_logits else None

            return GetActionResult(
                actions=actions,
                log_probs=log_probs,
                values=value,
                hidden=new_hidden,
                op_logits=op_logits_out,
            )

    def evaluate_actions(
        self,
        states: torch.Tensor,  # [batch, seq_len, state_dim]
        actions: dict[str, torch.Tensor],  # Each [batch, seq_len]
        slot_mask: torch.Tensor | None = None,
        blueprint_mask: torch.Tensor | None = None,
        blend_mask: torch.Tensor | None = None,
        tempo_mask: torch.Tensor | None = None,
        op_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], tuple]:
        """Evaluate actions for PPO update.

        Returns:
            log_probs: Dict of per-head log probs [batch, seq_len]
            values: Value estimates [batch, seq_len]
            entropy: Dict of per-head entropies [batch, seq_len]
            hidden: Final hidden state
        """
        output = self.forward(
            states, hidden, slot_mask, blueprint_mask, blend_mask, tempo_mask, op_mask
        )

        log_probs: dict[str, torch.Tensor] = {}
        entropy: dict[str, torch.Tensor] = {}

        masks = {
            "slot": slot_mask,
            "blueprint": blueprint_mask,
            "blend": blend_mask,
            "tempo": tempo_mask,
            "op": op_mask,
        }

        # Map head names to logits (TypedDict keys must be literals)
        head_logits: dict[str, torch.Tensor] = {
            "slot": output["slot_logits"],
            "blueprint": output["blueprint_logits"],
            "blend": output["blend_logits"],
            "tempo": output["tempo_logits"],
            "op": output["op_logits"],
        }

        for key in HEAD_NAMES:
            logits = head_logits[key]  # [batch, seq_len, action_dim]
            action = actions[key]  # [batch, seq_len]

            # Reshape for distribution
            batch, seq, action_dim = logits.shape
            logits_flat = logits.reshape(-1, action_dim)
            action_flat = action.reshape(-1)

            mask = masks[key]
            if mask is None:
                mask_flat = torch.ones_like(logits_flat, dtype=torch.bool)
            else:
                mask_flat = mask.reshape(-1, action_dim)

            dist = MaskedCategorical(logits=logits_flat, mask=mask_flat)
            log_probs[key] = dist.log_prob(action_flat).reshape(batch, seq)
            entropy[key] = dist.entropy().reshape(batch, seq)

        return log_probs, output["value"], entropy, output["hidden"]


__all__ = ["FactoredRecurrentActorCritic"]
