"""Factored Recurrent Actor-Critic for Tamiyo.

Architecture:
    state -> feature_net -> LSTM -> shared_repr
    shared_repr -> slot_head -> slot_logits
    shared_repr -> blueprint_head -> blueprint_logits
    shared_repr -> style_head -> style_logits
    shared_repr -> tempo_head -> tempo_logits
    shared_repr -> alpha_target/speed/curve heads -> alpha logits
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
from typing import TypedDict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.leyline import (
    BLUEPRINT_NULL_INDEX,
    DEFAULT_BLUEPRINT_EMBED_DIM,
    DEFAULT_LSTM_HIDDEN_DIM,
    DEFAULT_FEATURE_DIM,
    GerminationStyle,
    HEAD_NAMES,
    LifecycleOp,
    MASKED_LOGIT_VALUE,
    NUM_BLUEPRINTS,
    NUM_OPS,
    get_action_head_sizes,
)
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.policy.action_masks import (
    InvalidStateMachineError,
    MaskedCategorical,
    _validate_action_mask,
    _validate_logits,
)


@dataclass(frozen=True, slots=True)
class GetActionResult:
    """Result from get_action() method.

    Attributes:
        actions: Dict of action indices per head [batch]
        log_probs: Dict of log probs per head [batch] (NON-DIFFERENTIABLE)
        values: Value estimates [batch] - Q(s, sampled_op)
        hidden: Updated hidden state (h, c)
        sampled_op: The op action sampled/selected [batch] - used for value conditioning
        op_logits: Raw masked logits for op head [batch, num_ops].
            Only populated if return_op_logits=True, otherwise None.
            Use F.softmax(op_logits, dim=-1) to get action probabilities.
    """

    actions: dict[str, torch.Tensor]
    log_probs: dict[str, torch.Tensor]
    values: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]
    sampled_op: torch.Tensor
    op_logits: torch.Tensor | None = None


class _ForwardOutput(TypedDict):
    """Typed dict for forward() return value - enables mypy to track per-key types.

    Note: value is Q(s, sampled_op) - conditioned on the sampled op action.
    """

    slot_logits: torch.Tensor
    blueprint_logits: torch.Tensor
    style_logits: torch.Tensor
    tempo_logits: torch.Tensor
    alpha_target_logits: torch.Tensor
    alpha_speed_logits: torch.Tensor
    alpha_curve_logits: torch.Tensor
    op_logits: torch.Tensor
    value: torch.Tensor
    lstm_out: torch.Tensor  # NEW: [batch, seq, hidden_dim] for value recomputation
    sampled_op: torch.Tensor  # NEW: Op used for value conditioning
    hidden: tuple[torch.Tensor, torch.Tensor]


class BlueprintEmbedding(nn.Module):
    """Learned blueprint embeddings for Obs V3.

    Converts blueprint indices (0-12) to learned vector representations.
    Inactive slots (blueprint_index = -1) map to a trainable null embedding.

    Architecture:
        - Embedding table: (NUM_BLUEPRINTS + 1) x embed_dim
        - Initialization: N(0, 0.02) for stable early training
        - Null handling: register_buffer for device-safe -1 → 13 mapping

    Args:
        num_blueprints: Number of valid blueprints (default: 13)
        embed_dim: Dimension of embedding vectors (default: 4)
    """

    def __init__(
        self,
        num_blueprints: int = NUM_BLUEPRINTS,
        embed_dim: int = DEFAULT_BLUEPRINT_EMBED_DIM,
    ):
        super().__init__()
        # Index 13 = null embedding for inactive slots (from leyline)
        self.embedding = nn.Embedding(num_blueprints + 1, embed_dim)

        # Small initialization per DRL expert recommendation
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Register null index as buffer: moves with module.to(device), no grad, in state_dict
        # This avoids per-forward-call tensor allocation that torch.tensor() would cause
        self.register_buffer(
            "_null_idx",
            torch.tensor(BLUEPRINT_NULL_INDEX, dtype=torch.int64),
        )

    def forward(self, blueprint_indices: torch.Tensor) -> torch.Tensor:
        """Convert blueprint indices to embeddings.

        Args:
            blueprint_indices: Int64 tensor [batch, num_slots], -1 for inactive

        Returns:
            Float tensor [batch, num_slots, embed_dim]
        """
        # _null_idx is already on correct device via module.to(device)
        safe_idx = torch.where(blueprint_indices < 0, self._null_idx, blueprint_indices)
        return cast(torch.Tensor, self.embedding(safe_idx))


class FactoredRecurrentActorCritic(nn.Module):
    """Recurrent actor-critic with factored action heads.

    Uses LSTM for temporal reasoning over 10-20 epoch seed learning cycles.
    All action heads share the same temporal context from the LSTM.
    """

    def __init__(
        self,
        state_dim: int,
        num_slots: int | None = None,
        num_blueprints: int | None = None,
        num_styles: int | None = None,
        num_tempo: int | None = None,
        num_alpha_targets: int | None = None,
        num_alpha_speeds: int | None = None,
        num_alpha_curves: int | None = None,
        num_ops: int | None = None,
        feature_dim: int = DEFAULT_FEATURE_DIM,
        lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM,
        lstm_layers: int = 1,
        slot_config: SlotConfig | None = None,
    ):
        super().__init__()

        if slot_config is None:
            slot_config = SlotConfig.default()

        head_sizes = get_action_head_sizes(slot_config)

        self.state_dim = state_dim
        self.num_slots = head_sizes["slot"] if num_slots is None else num_slots
        self.num_blueprints = head_sizes["blueprint"] if num_blueprints is None else num_blueprints
        self.num_styles = head_sizes["style"] if num_styles is None else num_styles
        self.num_tempo = head_sizes["tempo"] if num_tempo is None else num_tempo
        self.num_alpha_targets = (
            head_sizes["alpha_target"] if num_alpha_targets is None else num_alpha_targets
        )
        self.num_alpha_speeds = (
            head_sizes["alpha_speed"] if num_alpha_speeds is None else num_alpha_speeds
        )
        self.num_alpha_curves = (
            head_sizes["alpha_curve"] if num_alpha_curves is None else num_alpha_curves
        )
        self.num_ops = head_sizes["op"] if num_ops is None else num_ops
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # Feature extraction before LSTM (reduces dimensionality)
        # M7: Pre-LSTM LayerNorm stabilizes input distribution to LSTM
        # Input: state_dim (114) + blueprint embeddings (num_slots * embed_dim = 3 * 4 = 12) = 126 total
        blueprint_embed_size = self.num_slots * DEFAULT_BLUEPRINT_EMBED_DIM
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + blueprint_embed_size, feature_dim),
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
            nn.Linear(head_hidden, self.num_slots),
        )
        # NOTE: blueprint_head defined below with 3-layer architecture (Phase 4)
        self.style_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_styles),
        )
        self.tempo_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_tempo),
        )
        self.alpha_target_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_alpha_targets),
        )
        self.alpha_speed_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_alpha_speeds),
        )
        self.alpha_curve_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_alpha_curves),
        )
        self.op_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_ops),
        )

        # Blueprint embedding for Obs V3 (Phase 3)
        self.blueprint_embedding = BlueprintEmbedding(
            num_blueprints=NUM_BLUEPRINTS,
            embed_dim=DEFAULT_BLUEPRINT_EMBED_DIM,
        )
        # Total embedding contribution: num_slots * embed_dim
        self._blueprint_embed_total_dim = self.num_slots * DEFAULT_BLUEPRINT_EMBED_DIM

        # 3-layer blueprint head (Phase 4 - deeper for blueprint selection complexity)
        self.blueprint_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),  # 512 -> 512
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, head_hidden),  # 512 -> 256
            nn.ReLU(),
            nn.Linear(head_hidden, self.num_blueprints),  # 256 -> 13
        )

        # Op-conditioned value head (Phase 4): Q(s, op) instead of V(s)
        # Input: lstm_out (lstm_hidden_dim) + op_one_hot (NUM_OPS)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim + NUM_OPS, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller init for output layers (policy stability)
        for head in [
            self.slot_head,
            self.blueprint_head,
            self.style_head,
            self.tempo_head,
            self.alpha_target_head,
            self.alpha_speed_head,
            self.alpha_curve_head,
            self.op_head,
        ]:
            # head[-1] is a Linear layer, access .weight.data to get Tensor
            last_layer = head[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.orthogonal_(last_layer.weight.data, gain=0.01)
        # value_head[-1] is also a Linear layer
        last_value_layer = self.value_head[-1]
        if isinstance(last_value_layer, nn.Linear):
            nn.init.orthogonal_(last_value_layer.weight.data, gain=1.0)

        # LSTM-specific initialization
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                # param is a Parameter, access .data to get the Tensor
                weight_tensor: torch.Tensor = param.data
                nn.init.orthogonal_(weight_tensor)
            elif "weight_hh" in name:
                weight_tensor = param.data
                nn.init.orthogonal_(weight_tensor)
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
        """Return zero-initialized hidden state.

        MEMORY MANAGEMENT - Hidden State Detachment:
        --------------------------------------------
        LSTM hidden states carry gradient graphs. To prevent memory leaks during
        training, callers MUST detach hidden states at episode boundaries:

            hidden = (h.detach(), c.detach())  # Break gradient graph

        Failure to detach causes:
        1. Unbounded BPTT across episode boundaries (exploding memory)
        2. Gradient graph accumulation proportional to total training steps
        3. OOM after ~100-1000 episodes on typical GPU memory

        The training loop (vectorized.py) handles this automatically when
        resetting environments. If using this network directly, ensure you
        call .detach() on hidden states when starting new episodes.

        Returns:
            Tuple of (h, c) zero tensors, each [num_layers, batch, hidden_dim]
        """
        h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dim, device=device)
        return h, c

    def _compute_value(self, lstm_out: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, op) value conditioned on operation.

        Shared helper used by both forward() (with sampled op) and
        evaluate_actions() (with stored op from buffer).

        Args:
            lstm_out: LSTM output [batch, seq_len, lstm_hidden_dim]
            op: Operation indices [batch, seq_len]

        Returns:
            Value estimates [batch, seq_len]
        """
        op_one_hot = F.one_hot(op, num_classes=NUM_OPS).float()
        value_input = torch.cat([lstm_out, op_one_hot], dim=-1)
        value = cast(torch.Tensor, self.value_head(value_input))
        return value.squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,  # [batch, seq_len, state_dim]
        blueprint_indices: torch.Tensor,  # [batch, seq_len, num_slots]
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        slot_mask: torch.Tensor | None = None,
        blueprint_mask: torch.Tensor | None = None,
        style_mask: torch.Tensor | None = None,
        tempo_mask: torch.Tensor | None = None,
        alpha_target_mask: torch.Tensor | None = None,
        alpha_speed_mask: torch.Tensor | None = None,
        alpha_curve_mask: torch.Tensor | None = None,
        op_mask: torch.Tensor | None = None,
    ) -> _ForwardOutput:
        """Forward pass returning logits, value, and new hidden state.

        Args:
            state: Input state [batch, seq_len, state_dim]
            blueprint_indices: Blueprint indices per slot [batch, seq_len, num_slots]
                Values 0-12 for active blueprints, -1 for inactive slots.
            hidden: (h, c) tuple, each [num_layers, batch, hidden_dim]
            *_mask: Boolean masks [batch, seq_len, action_dim], True = valid

        Returns:
            _ForwardOutput with logits, Q(s, sampled_op) value, sampled_op, and hidden
        """
        batch_size = state.size(0)
        seq_len = state.size(1)
        device = state.device

        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, device)

        # Blueprint embedding (Phase 3) - embed active blueprint indices
        bp_emb = self.blueprint_embedding(blueprint_indices)  # [batch, seq, num_slots, embed_dim]
        bp_emb_flat = bp_emb.flatten(start_dim=2)  # [batch, seq, num_slots * embed_dim] = [batch, seq, 12]
        state_with_bp = torch.cat([state, bp_emb_flat], dim=-1)  # [batch, seq, state_dim + 12]

        # Feature extraction
        features = self.feature_net(state_with_bp)  # [batch, seq_len, feature_dim]

        # LSTM forward
        lstm_out, new_hidden = self.lstm(features, hidden)
        # lstm_out: [batch, seq_len, hidden_dim]

        # LayerNorm on LSTM output (prevents magnitude drift)
        lstm_out = self.lstm_ln(lstm_out)

        # Compute logits for each head
        slot_logits = self.slot_head(lstm_out)
        blueprint_logits = self.blueprint_head(lstm_out)
        style_logits = self.style_head(lstm_out)
        tempo_logits = self.tempo_head(lstm_out)
        alpha_target_logits = self.alpha_target_head(lstm_out)
        alpha_speed_logits = self.alpha_speed_head(lstm_out)
        alpha_curve_logits = self.alpha_curve_head(lstm_out)
        op_logits = self.op_head(lstm_out)

        # Apply masks using canonical MASKED_LOGIT_VALUE from leyline
        # (See leyline/__init__.py for rationale on value choice for FP16 safety)
        if slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~slot_mask, MASKED_LOGIT_VALUE)
        if blueprint_mask is not None:
            blueprint_logits = blueprint_logits.masked_fill(~blueprint_mask, MASKED_LOGIT_VALUE)
        if style_mask is not None:
            style_logits = style_logits.masked_fill(~style_mask, MASKED_LOGIT_VALUE)
        if tempo_mask is not None:
            tempo_logits = tempo_logits.masked_fill(~tempo_mask, MASKED_LOGIT_VALUE)
        if alpha_target_mask is not None:
            alpha_target_logits = alpha_target_logits.masked_fill(~alpha_target_mask, MASKED_LOGIT_VALUE)
        if alpha_speed_mask is not None:
            alpha_speed_logits = alpha_speed_logits.masked_fill(~alpha_speed_mask, MASKED_LOGIT_VALUE)
        if alpha_curve_mask is not None:
            alpha_curve_logits = alpha_curve_logits.masked_fill(~alpha_curve_mask, MASKED_LOGIT_VALUE)
        if op_mask is not None:
            op_logits = op_logits.masked_fill(~op_mask, MASKED_LOGIT_VALUE)

        # Validate op mask and logits before sampling (matches MaskedCategorical behavior)
        # CRITICAL: Catches state machine bugs (empty masks) and network instability
        # (inf/nan logits) during forward pass. Only runs when validation is enabled.
        if MaskedCategorical.validate:
            # Validate mask if provided (check all batch×seq elements have valid actions)
            if op_mask is not None:
                # op_mask: [batch, seq_len, num_ops]
                valid_count = op_mask.sum(dim=-1)  # [batch, seq_len]
                if (valid_count == 0).any():
                    raise InvalidStateMachineError(
                        f"No valid op actions available in forward(). Mask: {op_mask}. "
                        "This indicates a bug in the Kasmina state machine."
                    )
            # Validate logits for inf/nan
            _validate_logits(op_logits)

        # Sample op from policy for op-conditioned value
        # PERFORMANCE OPTIMIZATION: Use direct multinomial sampling instead of
        # MaskedCategorical. PyTorch's Categorical.sample() triggers 2 CPU-GPU syncs
        # per call via internal validation (aten::item, aten::is_nonzero). Direct
        # multinomial avoids this overhead while maintaining correctness.
        # Note: op_logits is already masked (see above), so we can
        # compute softmax directly without re-masking.
        op_probs = F.softmax(op_logits, dim=-1)  # [batch, seq_len, num_ops]
        op_probs_flat = op_probs.reshape(-1, self.num_ops)  # [batch*seq_len, num_ops]
        sampled_op_flat = torch.multinomial(op_probs_flat, num_samples=1).squeeze(-1)
        sampled_op = sampled_op_flat.reshape(batch_size, seq_len)

        # Op-conditioned value: Q(s, sampled_op)
        value = self._compute_value(lstm_out, sampled_op)

        return {
            "slot_logits": slot_logits,
            "blueprint_logits": blueprint_logits,
            "style_logits": style_logits,
            "tempo_logits": tempo_logits,
            "alpha_target_logits": alpha_target_logits,
            "alpha_speed_logits": alpha_speed_logits,
            "alpha_curve_logits": alpha_curve_logits,
            "op_logits": op_logits,
            "value": value,
            "lstm_out": lstm_out,  # NEW: Expose for value recomputation in get_action()
            "sampled_op": sampled_op,
            "hidden": new_hidden,
        }

    def get_action(
        self,
        state: torch.Tensor,  # [batch, state_dim] or [batch, 1, state_dim]
        blueprint_indices: torch.Tensor,  # [batch, num_slots] or [batch, 1, num_slots]
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        slot_mask: torch.Tensor | None = None,
        blueprint_mask: torch.Tensor | None = None,
        style_mask: torch.Tensor | None = None,
        tempo_mask: torch.Tensor | None = None,
        alpha_target_mask: torch.Tensor | None = None,
        alpha_speed_mask: torch.Tensor | None = None,
        alpha_curve_mask: torch.Tensor | None = None,
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
            blueprint_indices: Blueprint indices per slot [batch, num_slots] or
                [batch, 1, num_slots]. Values 0-12 for active blueprints, -1 for inactive.
            hidden: LSTM hidden state (h, c) or None for initial state
            slot_mask: Boolean mask for slot actions [batch, num_slots]
            blueprint_mask: Boolean mask for blueprint actions [batch, num_blueprints]
            style_mask: Boolean mask for germination style actions [batch, num_styles]
            tempo_mask: Boolean mask for tempo actions [batch, num_tempo]
            op_mask: Boolean mask for op actions [batch, num_ops]
            deterministic: If True, use argmax instead of sampling
            return_op_logits: If True, include raw masked op logits in result
                for telemetry/decision snapshot. Default False for performance.

        Returns:
            GetActionResult with actions, log_probs, values, hidden, sampled_op,
            and optionally op_logits if return_op_logits=True.
        """
        # Ensure 3D input
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        # Ensure blueprint_indices is 3D [batch, seq, num_slots]
        if blueprint_indices.dim() == 2:
            blueprint_indices = blueprint_indices.unsqueeze(1)

        # Reshape masks to [batch, 1, dim] if provided as [batch, dim]
        if slot_mask is not None and slot_mask.dim() == 2:
            slot_mask = slot_mask.unsqueeze(1)
        if blueprint_mask is not None and blueprint_mask.dim() == 2:
            blueprint_mask = blueprint_mask.unsqueeze(1)
        if style_mask is not None and style_mask.dim() == 2:
            style_mask = style_mask.unsqueeze(1)
        if tempo_mask is not None and tempo_mask.dim() == 2:
            tempo_mask = tempo_mask.unsqueeze(1)
        if alpha_target_mask is not None and alpha_target_mask.dim() == 2:
            alpha_target_mask = alpha_target_mask.unsqueeze(1)
        if alpha_speed_mask is not None and alpha_speed_mask.dim() == 2:
            alpha_speed_mask = alpha_speed_mask.unsqueeze(1)
        if alpha_curve_mask is not None and alpha_curve_mask.dim() == 2:
            alpha_curve_mask = alpha_curve_mask.unsqueeze(1)
        if op_mask is not None and op_mask.dim() == 2:
            op_mask = op_mask.unsqueeze(1)

        with torch.inference_mode():
            output = self.forward(
                state,
                blueprint_indices,
                hidden,
                slot_mask,
                blueprint_mask,
                style_mask,
                tempo_mask,
                alpha_target_mask,
                alpha_speed_mask,
                alpha_curve_mask,
                op_mask,
            )

            # Sample from each head using MaskedCategorical for safety
            actions: dict[str, torch.Tensor] = {}
            log_probs: dict[str, torch.Tensor] = {}

            masks = {
                "slot": slot_mask[:, 0, :] if slot_mask is not None else None,
                "blueprint": blueprint_mask[:, 0, :] if blueprint_mask is not None else None,
                "style": style_mask[:, 0, :] if style_mask is not None else None,
                "tempo": tempo_mask[:, 0, :] if tempo_mask is not None else None,
                "alpha_target": alpha_target_mask[:, 0, :] if alpha_target_mask is not None else None,
                "alpha_speed": alpha_speed_mask[:, 0, :] if alpha_speed_mask is not None else None,
                "alpha_curve": alpha_curve_mask[:, 0, :] if alpha_curve_mask is not None else None,
                "op": op_mask[:, 0, :] if op_mask is not None else None,
            }

            # Map head names to logits (TypedDict keys must be literals)
            head_logits: dict[str, torch.Tensor] = {
                "slot": output["slot_logits"][:, 0, :],
                "blueprint": output["blueprint_logits"][:, 0, :],
                "style": output["style_logits"][:, 0, :],
                "tempo": output["tempo_logits"][:, 0, :],
                "alpha_target": output["alpha_target_logits"][:, 0, :],
                "alpha_speed": output["alpha_speed_logits"][:, 0, :],
                "alpha_curve": output["alpha_curve_logits"][:, 0, :],
                "op": output["op_logits"][:, 0, :],
            }

            def _sample_head(
                key: str,
                *,
                mask_override: torch.Tensor | None = None,
            ) -> None:
                """Sample action and compute log_prob without MaskedCategorical.

                PERFORMANCE OPTIMIZATION: PyTorch's Categorical triggers 2+ CPU-GPU
                syncs per instantiation. Using direct multinomial + log_softmax avoids
                this overhead (0.3ms -> 0.1ms per head on RTX 4060 Ti).
                """
                logits = head_logits[key]  # [batch, action_dim]
                mask = mask_override if mask_override is not None else masks[key]
                if mask is None:
                    mask = torch.ones_like(logits, dtype=torch.bool)

                # Validate mask and logits (matches MaskedCategorical behavior)
                # CRITICAL: Catches state machine bugs (empty masks) and network
                # instability (inf/nan logits) early instead of silently proceeding.
                if MaskedCategorical.validate:
                    _validate_action_mask(mask)
                    _validate_logits(logits)

                # Apply mask directly (same as MaskedCategorical)
                masked_logits = logits.masked_fill(~mask, MASKED_LOGIT_VALUE)

                if deterministic:
                    action = masked_logits.argmax(dim=-1)
                else:
                    # Direct multinomial sampling (no Categorical overhead)
                    probs = F.softmax(masked_logits, dim=-1)
                    action = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # Compute log_prob directly: log_softmax(logits)[action]
                all_log_probs = F.log_softmax(masked_logits, dim=-1)
                log_prob = all_log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

                actions[key] = action
                log_probs[key] = log_prob

            # === OP HEAD: CRITICAL FIX - Must use same op for value and action ===
            # Bug: Previously _sample_head("op") sampled independently from forward(),
            # causing value/action mismatch. This created biased advantages and
            # bootstrap corruption. Fix: Reuse sampled_op from forward() (stochastic)
            # or recompute value with argmax_op (deterministic).
            #
            # PERFORMANCE OPTIMIZATION: Avoid creating MaskedCategorical for op here.
            # MaskedCategorical uses Categorical internally, which triggers 2 CPU-GPU
            # syncs per instantiation via aten::item/is_nonzero. Instead, compute
            # masked_logits and log_prob directly using tensor ops (no sync).
            op_logits = head_logits["op"]
            op_mask = masks["op"]
            if op_mask is None:
                op_mask = torch.ones_like(op_logits, dtype=torch.bool)

            # Validate mask and logits (matches MaskedCategorical behavior)
            # CRITICAL: This validation catches state machine bugs early rather than
            # silently selecting invalid actions. Only runs when validation is enabled.
            if MaskedCategorical.validate:
                _validate_action_mask(op_mask)
                _validate_logits(op_logits)

            # Compute masked logits directly (same as MaskedCategorical)
            op_masked_logits = op_logits.masked_fill(~op_mask, MASKED_LOGIT_VALUE)

            if deterministic:
                # Deterministic mode (bootstrap/eval): use argmax
                selected_op = op_masked_logits.argmax(dim=-1)
                # CRITICAL: Recompute value with argmax op for consistency
                # Value from forward() used sampled op - we need Q(s, argmax_op)
                lstm_out = output["lstm_out"][:, 0, :]  # [batch, hidden_dim]
                value = self._compute_value(
                    lstm_out.unsqueeze(1),  # [batch, 1, hidden_dim]
                    selected_op.unsqueeze(1)  # [batch, 1]
                ).squeeze(1)  # [batch]
            else:
                # Stochastic mode (rollout): reuse sampled op from forward()
                # This ensures value = Q(s, op) where op is the action we'll take
                selected_op = output["sampled_op"][:, 0]
                value = output["value"][:, 0]  # Already Q(s, sampled_op)

            actions["op"] = selected_op
            # Compute log_prob directly without creating Categorical (avoids 2 GPU syncs)
            # log_prob = log_softmax(logits)[action]
            op_log_probs = F.log_softmax(op_masked_logits, dim=-1)
            log_probs["op"] = op_log_probs.gather(-1, selected_op.unsqueeze(-1)).squeeze(-1)
            sampled_op = selected_op  # For return value consistency

            # Style mask override based on selected_op (not from independent sample)
            style_mask_override = masks["style"]
            if style_mask_override is None:
                style_mask_override = torch.ones_like(head_logits["style"], dtype=torch.bool)
            # Avoid `.any()` (CPU sync) by applying the override unconditionally.
            style_mask_override = style_mask_override.clone()
            style_irrelevant = (selected_op != LifecycleOp.GERMINATE) & (
                selected_op != LifecycleOp.SET_ALPHA_TARGET
            )
            style_mask_override[style_irrelevant] = False
            style_mask_override[style_irrelevant, int(GerminationStyle.SIGMOID_ADD)] = True
            _sample_head("style", mask_override=style_mask_override)

            # Sample remaining heads (unchanged)
            for key in [
                "slot",
                "blueprint",
                "tempo",
                "alpha_target",
                "alpha_speed",
                "alpha_curve",
            ]:
                _sample_head(key)

            # Value and sampled_op are already set above based on deterministic mode
            new_hidden = output["hidden"]

            # Conditionally capture op_logits for telemetry (Decision Snapshot)
            op_logits_out = head_logits["op"] if return_op_logits else None

            return GetActionResult(
                actions=actions,
                log_probs=log_probs,
                values=value,
                hidden=new_hidden,
                sampled_op=sampled_op,
                op_logits=op_logits_out,
            )

    def evaluate_actions(
        self,
        states: torch.Tensor,  # [batch, seq_len, state_dim]
        blueprint_indices: torch.Tensor,  # [batch, seq_len, num_slots]
        actions: dict[str, torch.Tensor],  # Each [batch, seq_len]
        slot_mask: torch.Tensor | None = None,
        blueprint_mask: torch.Tensor | None = None,
        style_mask: torch.Tensor | None = None,
        tempo_mask: torch.Tensor | None = None,
        alpha_target_mask: torch.Tensor | None = None,
        alpha_speed_mask: torch.Tensor | None = None,
        alpha_curve_mask: torch.Tensor | None = None,
        op_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Evaluate actions for PPO update.

        Uses stored op from actions dict for value conditioning (Q(s, stored_op)),
        ensuring consistency with what was stored during rollout collection.

        Args:
            states: Input states [batch, seq_len, state_dim]
            blueprint_indices: Blueprint indices [batch, seq_len, num_slots]
            actions: Stored actions from buffer, including 'op' key
            *_mask: Boolean masks [batch, seq_len, action_dim]
            hidden: Initial LSTM hidden state

        Returns:
            log_probs: Dict of per-head log probs [batch, seq_len]
            values: Value estimates Q(s, stored_op) [batch, seq_len]
            entropy: Dict of per-head entropies [batch, seq_len]
            hidden: Final hidden state
        """
        batch_size = states.size(0)
        device = states.device

        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, device)

        # Blueprint embedding (Phase 3) - embed active blueprint indices
        bp_emb = self.blueprint_embedding(blueprint_indices)  # [batch, seq, num_slots, embed_dim]
        bp_emb_flat = bp_emb.flatten(start_dim=2)  # [batch, seq, num_slots * embed_dim] = [batch, seq, 12]
        state_with_bp = torch.cat([states, bp_emb_flat], dim=-1)  # [batch, seq, state_dim + 12]

        # Feature extraction and LSTM (same as forward)
        features = self.feature_net(state_with_bp)
        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = self.lstm_ln(lstm_out)

        # Compute logits for each head
        slot_logits = self.slot_head(lstm_out)
        blueprint_logits = self.blueprint_head(lstm_out)
        style_logits = self.style_head(lstm_out)
        tempo_logits = self.tempo_head(lstm_out)
        alpha_target_logits = self.alpha_target_head(lstm_out)
        alpha_speed_logits = self.alpha_speed_head(lstm_out)
        alpha_curve_logits = self.alpha_curve_head(lstm_out)
        op_logits = self.op_head(lstm_out)

        # Apply masks
        if slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~slot_mask, MASKED_LOGIT_VALUE)
        if blueprint_mask is not None:
            blueprint_logits = blueprint_logits.masked_fill(~blueprint_mask, MASKED_LOGIT_VALUE)
        if style_mask is not None:
            style_logits = style_logits.masked_fill(~style_mask, MASKED_LOGIT_VALUE)
        if tempo_mask is not None:
            tempo_logits = tempo_logits.masked_fill(~tempo_mask, MASKED_LOGIT_VALUE)
        if alpha_target_mask is not None:
            alpha_target_logits = alpha_target_logits.masked_fill(~alpha_target_mask, MASKED_LOGIT_VALUE)
        if alpha_speed_mask is not None:
            alpha_speed_logits = alpha_speed_logits.masked_fill(~alpha_speed_mask, MASKED_LOGIT_VALUE)
        if alpha_curve_mask is not None:
            alpha_curve_logits = alpha_curve_logits.masked_fill(~alpha_curve_mask, MASKED_LOGIT_VALUE)
        if op_mask is not None:
            op_logits = op_logits.masked_fill(~op_mask, MASKED_LOGIT_VALUE)

        # Use STORED op for value conditioning (not freshly sampled)
        stored_op = actions["op"]
        value = self._compute_value(lstm_out, stored_op)

        log_probs: dict[str, torch.Tensor] = {}
        entropy: dict[str, torch.Tensor] = {}

        op_actions = actions["op"]

        masks = {
            "slot": slot_mask,
            "blueprint": blueprint_mask,
            "style": style_mask,
            "tempo": tempo_mask,
            "alpha_target": alpha_target_mask,
            "alpha_speed": alpha_speed_mask,
            "alpha_curve": alpha_curve_mask,
            "op": op_mask,
        }

        head_logits: dict[str, torch.Tensor] = {
            "slot": slot_logits,
            "blueprint": blueprint_logits,
            "style": style_logits,
            "tempo": tempo_logits,
            "alpha_target": alpha_target_logits,
            "alpha_speed": alpha_speed_logits,
            "alpha_curve": alpha_curve_logits,
            "op": op_logits,
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
                mask = torch.ones_like(logits, dtype=torch.bool)
            if key == "style":
                # When op is not GERMINATE or SET_ALPHA_TARGET, style is irrelevant.
                # Force selection of SIGMOID_ADD (the default/no-op style).
                style_irrelevant = (op_actions != LifecycleOp.GERMINATE) & (
                    op_actions != LifecycleOp.SET_ALPHA_TARGET
                )
                # For 3D mask [batch, seq_len, num_styles], use masked_fill pattern
                # to avoid incorrect advanced indexing. Expand style_irrelevant to match.
                expanded_irrelevant = style_irrelevant.unsqueeze(-1).expand_as(mask)
                mask = mask.masked_fill(expanded_irrelevant, False)
                # Set SIGMOID_ADD column to True for irrelevant rows
                sigmoid_add_idx = int(GerminationStyle.SIGMOID_ADD)
                mask[..., sigmoid_add_idx] = mask[..., sigmoid_add_idx] | style_irrelevant
            mask_flat = mask.reshape(-1, action_dim)

            dist = MaskedCategorical(logits=logits_flat, mask=mask_flat)
            log_probs[key] = dist.log_prob(action_flat).reshape(batch, seq)
            entropy[key] = dist.entropy().reshape(batch, seq)

        return log_probs, value, entropy, new_hidden


__all__ = ["BlueprintEmbedding", "FactoredRecurrentActorCritic", "GetActionResult"]
