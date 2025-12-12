"""Factored Multi-Head Policy Network for Multi-Slot Control.

The policy outputs separate distributions for each action dimension,
then samples them jointly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical

from esper.leyline.factored_actions import NUM_SLOTS, NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS
from esper.simic.networks import InvalidStateMachineError


@torch.compiler.disable
def _validate_factored_masks(masks: dict[str, torch.Tensor]) -> None:
    """Validate factored action masks have at least one valid action per head per env.

    Isolated from torch.compile to prevent graph breaks in the main forward path.
    The .any() call forces CPU sync, but this safety check is worth the cost.
    """
    for key, mask in masks.items():
        valid_per_env = mask.any(dim=-1)
        if not valid_per_env.all():
            invalid_envs = (~valid_per_env).nonzero(as_tuple=True)[0]
            raise InvalidStateMachineError(
                f"All actions masked for '{key}' head in env {invalid_envs[0].item()}. "
                f"This indicates a bug in mask computation."
            )


class FactoredActorCritic(nn.Module):
    """Actor-Critic with factored action heads.

    Instead of one output for all actions, we have separate heads:
    - slot_head: which slot to target
    - blueprint_head: which blueprint to germinate
    - blend_head: which blending algorithm
    - op_head: which lifecycle operation
    """

    def __init__(
        self,
        state_dim: int,
        num_slots: int = NUM_SLOTS,
        num_blueprints: int = NUM_BLUEPRINTS,
        num_blends: int = NUM_BLENDS,
        num_ops: int = NUM_OPS,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.num_blueprints = num_blueprints
        self.num_blends = num_blends
        self.num_ops = num_ops

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Factored action heads
        head_hidden = hidden_dim // 2
        self.slot_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_slots),
        )
        self.blueprint_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_blueprints),
        )
        self.blend_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_blends),
        )
        self.op_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, num_ops),
        )

        # Critic head (single value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller init for output layers
        for head in [self.slot_head, self.blueprint_head, self.blend_head, self.op_head]:
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, Categorical], torch.Tensor]:
        """Forward pass returning distributions for each head and value.

        Args:
            state: Observation tensor (batch, state_dim)
            masks: Optional dict of action masks per head

        Returns:
            dists: Dict of Categorical distributions
            value: Value estimates (batch,)
        """
        features = self.shared(state)

        # Get logits from each head
        slot_logits = self.slot_head(features)
        blueprint_logits = self.blueprint_head(features)
        blend_logits = self.blend_head(features)
        op_logits = self.op_head(features)

        # Apply masks if provided (use dtype-safe min value for mixed precision compatibility)
        if masks:
            mask_value = torch.finfo(slot_logits.dtype).min
            if "slot" in masks:
                slot_logits = slot_logits.masked_fill(~masks["slot"], mask_value)
            if "blueprint" in masks:
                blueprint_logits = blueprint_logits.masked_fill(~masks["blueprint"], mask_value)
            if "blend" in masks:
                blend_logits = blend_logits.masked_fill(~masks["blend"], mask_value)
            if "op" in masks:
                op_logits = op_logits.masked_fill(~masks["op"], mask_value)

            # Validate masks - at least one action must be valid per head per env
            _validate_factored_masks(masks)

        dists = {
            "slot": Categorical(logits=slot_logits),
            "blueprint": Categorical(logits=blueprint_logits),
            "blend": Categorical(logits=blend_logits),
            "op": Categorical(logits=op_logits),
        }

        value = self.critic(features).squeeze(-1)
        return dists, value

    def get_action_batch(
        self,
        states: torch.Tensor,
        masks: dict[str, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Sample actions from all heads.

        Returns:
            actions: Dict of action indices per head
            log_probs: Sum of log probs across heads
            values: Value estimates
        """
        with torch.no_grad():
            dists, values = self.forward(states, masks)

            actions = {}
            log_probs_list = []

            for key, dist in dists.items():
                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()
                actions[key] = action
                log_probs_list.append(dist.log_prob(action))

            # Sum log probs across heads (joint probability)
            log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)

            return actions, log_probs, values

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            log_probs: Sum of log probs across heads
            values: Value estimates
            entropy: Sum of NORMALIZED entropies across heads (each in [0, 1])

        Note on entropy normalization:
            We normalize by FULL action space (log(num_actions)), not by valid
            actions under the mask. This means:
            - "Uniform over 2 valid actions" reports ~50% entropy (not 100%)
            - entropy_coef has consistent meaning across mask states
            - Exploration bonus is weaker when fewer actions available

            This is intentional: we want consistent regularization strength
            regardless of how restrictive the mask is.
        """
        dists, values = self.forward(states, masks)

        log_probs_list = []
        entropy_list = []

        # Max entropies for normalization (full action space, not masked)
        # See docstring for design rationale
        max_entropies = {
            "slot": math.log(self.num_slots),
            "blueprint": math.log(self.num_blueprints),
            "blend": math.log(self.num_blends),
            "op": math.log(self.num_ops),
        }

        for key, dist in dists.items():
            log_probs_list.append(dist.log_prob(actions[key]))

            # Normalize entropy to [0, 1] by full action space
            raw_entropy = dist.entropy()
            max_ent = max_entropies[key]
            normalized_entropy = raw_entropy / max(max_ent, 1e-8)
            entropy_list.append(normalized_entropy)

        log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)
        # Sum normalized entropies (not mean) - each head contributes equally
        # to the exploration budget. Range: [0, num_heads] (here [0, 4])
        entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)

        return log_probs, values, entropy


__all__ = ["FactoredActorCritic"]
