"""Factored Recurrent Actor-Critic for Tamiyo.

Architecture:
    state -> feature_net -> LSTM -> shared_repr
    shared_repr -> slot_head -> slot_logits
    shared_repr -> blueprint_head -> blueprint_logits
    shared_repr -> blend_head -> blend_logits
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

import torch
import torch.nn as nn
from torch.distributions import Categorical

from esper.leyline.factored_actions import (
    NUM_SLOTS,
    NUM_BLUEPRINTS,
    NUM_BLENDS,
    NUM_OPS,
)

# Mask value for invalid actions. Use -1e4 (not -inf or dtype.min) because:
# 1. float("-inf") causes FP16 saturation issues
# 2. torch.finfo(dtype).min can cause softmax overflow after max-subtraction
# 3. -1e4 is large enough to zero out softmax but small enough to avoid overflow
# This is the standard practice in HuggingFace Transformers and PyTorch attention.
_MASK_VALUE = -1e4


class FactoredRecurrentActorCritic(nn.Module):
    """Recurrent actor-critic with factored action heads.

    Uses LSTM for temporal reasoning over 10-20 epoch seed learning cycles.
    All 4 action heads share the same temporal context from the LSTM.
    """

    def __init__(
        self,
        state_dim: int,
        num_slots: int = NUM_SLOTS,
        num_blueprints: int = NUM_BLUEPRINTS,
        num_blends: int = NUM_BLENDS,
        num_ops: int = NUM_OPS,
        feature_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.num_slots = num_slots
        self.num_blueprints = num_blueprints
        self.num_blends = num_blends
        self.num_ops = num_ops
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # Feature extraction before LSTM (reduces dimensionality)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        # LSTM for temporal reasoning
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # LayerNorm on LSTM output (CRITICAL for training stability)
        # Prevents hidden state magnitude drift over long 25-epoch sequences
        self.lstm_ln = nn.LayerNorm(lstm_hidden_dim)

        # Max entropy for per-head normalization (different cardinalities)
        self.max_entropies = {
            "slot": math.log(num_slots),
            "blueprint": math.log(num_blueprints),
            "blend": math.log(num_blends),
            "op": math.log(num_ops),
        }

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
        for head in [self.slot_head, self.blueprint_head, self.blend_head, self.op_head]:
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
                # Set forget gate bias to 1 (helps with long-term memory)
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
        op_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning logits, value, and new hidden state.

        Args:
            state: Input state [batch, seq_len, state_dim]
            hidden: (h, c) tuple, each [num_layers, batch, hidden_dim]
            *_mask: Boolean masks [batch, seq_len, action_dim], True = valid

        Returns:
            Dict with slot_logits, blueprint_logits, blend_logits,
            op_logits, value, and hidden tuple
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
        op_logits = self.op_head(lstm_out)

        # Apply masks (set invalid actions to large negative for softmax zeroing)
        # Using -1e4 instead of -inf or dtype.min to avoid FP16 overflow issues
        # (PyTorch expert review: dtype.min can cause NaN after softmax normalization)
        if slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~slot_mask, _MASK_VALUE)
        if blueprint_mask is not None:
            blueprint_logits = blueprint_logits.masked_fill(~blueprint_mask, _MASK_VALUE)
        if blend_mask is not None:
            blend_logits = blend_logits.masked_fill(~blend_mask, _MASK_VALUE)
        if op_mask is not None:
            op_logits = op_logits.masked_fill(~op_mask, _MASK_VALUE)

        # Value prediction
        value = self.value_head(lstm_out).squeeze(-1)  # [batch, seq_len]

        return {
            "slot_logits": slot_logits,
            "blueprint_logits": blueprint_logits,
            "blend_logits": blend_logits,
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
        op_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, tuple]:
        """Sample actions from all heads (inference mode).

        Returns:
            actions: Dict of action indices per head [batch]
            log_probs: Dict of log probs per head [batch]
            values: Value estimates [batch]
            hidden: Updated hidden state
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
        if op_mask is not None and op_mask.dim() == 2:
            op_mask = op_mask.unsqueeze(1)

        with torch.inference_mode():
            output = self.forward(
                state, hidden, slot_mask, blueprint_mask, blend_mask, op_mask
            )

            # Sample from each head
            actions = {}
            log_probs = {}

            for key in ["slot", "blueprint", "blend", "op"]:
                logits = output[f"{key}_logits"][:, 0, :]  # [batch, action_dim]
                dist = Categorical(logits=logits)

                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()

                actions[key] = action
                log_probs[key] = dist.log_prob(action)

            value = output["value"][:, 0]  # [batch]
            new_hidden = output["hidden"]

            return actions, log_probs, value, new_hidden

    def evaluate_actions(
        self,
        states: torch.Tensor,  # [batch, seq_len, state_dim]
        actions: dict[str, torch.Tensor],  # Each [batch, seq_len]
        slot_mask: torch.Tensor | None = None,
        blueprint_mask: torch.Tensor | None = None,
        blend_mask: torch.Tensor | None = None,
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
            states, hidden, slot_mask, blueprint_mask, blend_mask, op_mask
        )

        log_probs = {}
        entropy = {}

        for key in ["slot", "blueprint", "blend", "op"]:
            logits = output[f"{key}_logits"]  # [batch, seq_len, action_dim]
            action = actions[key]  # [batch, seq_len]

            # Reshape for Categorical
            batch, seq, action_dim = logits.shape
            logits_flat = logits.reshape(-1, action_dim)
            action_flat = action.reshape(-1)

            dist = Categorical(logits=logits_flat)
            log_probs[key] = dist.log_prob(action_flat).reshape(batch, seq)
            # Normalize entropy by max possible (different head cardinalities)
            raw_entropy = dist.entropy().reshape(batch, seq)
            entropy[key] = raw_entropy / max(self.max_entropies[key], 1e-8)

        return log_probs, output["value"], entropy, output["hidden"]


__all__ = ["FactoredRecurrentActorCritic"]
