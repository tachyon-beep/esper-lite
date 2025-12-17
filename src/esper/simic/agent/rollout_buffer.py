"""Unified Rollout Buffer for Tamiyo - Factored Recurrent Actor-Critic.

This buffer combines:
- Per-environment storage (fixes GAE interleaving bug)
- Factored action space (4 heads: slot, blueprint, blend, op)
- Per-head log probs (enables head-specific credit assignment)
- LSTM hidden state tracking (enables temporal reasoning)
- Pre-allocated tensors (memory efficiency)

Design rationale:
    Seeds take 10-20 epochs to learn. Without temporal memory, Tamiyo
    cannot distinguish between "seed germinated 5 epochs ago" vs
    "seed germinated 18 epochs ago" - both look identical in the
    current observation. LSTM hidden state provides this memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from esper.leyline import DEFAULT_GAMMA, DEFAULT_LSTM_HIDDEN_DIM
from esper.leyline.factored_actions import (
    NUM_BLUEPRINTS,
    NUM_BLENDS,
    NUM_OPS,
)
from esper.leyline.slot_config import SlotConfig


class TamiyoRolloutStep(NamedTuple):
    """Single transition for factored recurrent actor-critic.

    Uses flat NamedTuple (not nested) for torch.compile compatibility.
    Per-head log probs enable head-specific gradient attribution.
    """

    # Core state
    state: torch.Tensor  # [state_dim]

    # Factored actions (4 heads)
    slot_action: int
    blueprint_action: int
    blend_action: int
    op_action: int

    # Per-head log probs (NOT joint) - see PyTorch expert rationale
    slot_log_prob: float
    blueprint_log_prob: float
    blend_log_prob: float
    op_log_prob: float

    # Value and reward
    value: float
    reward: float

    # Episode boundaries
    done: bool
    truncated: bool
    bootstrap_value: float

    # Action masks (4 heads)
    slot_mask: torch.Tensor
    blueprint_mask: torch.Tensor
    blend_mask: torch.Tensor
    op_mask: torch.Tensor

    # LSTM hidden state at this step
    hidden_h: torch.Tensor  # [num_layers, hidden_dim]
    hidden_c: torch.Tensor  # [num_layers, hidden_dim]


@dataclass(slots=True)
class TamiyoRolloutBuffer:
    """Per-environment rollout buffer with pre-allocated tensors.

    Designed for:
    - N parallel environments
    - 25 epochs per episode (max_steps_per_env)
    - Factored action space (4 heads)
    - LSTM hidden state tracking

    Pre-allocation rationale (PyTorch expert):
    - Fixed episode length (25 epochs) - no dynamic sizing needed
    - torch.compile compatible (direct indexing, not list append)
    - Predictable memory (~100KB for 4 envs x 25 steps)
    - No GC pressure from intermediate objects
    """

    num_envs: int
    max_steps_per_env: int
    state_dim: int
    lstm_hidden_dim: int = DEFAULT_LSTM_HIDDEN_DIM
    lstm_layers: int = 1
    slot_config: SlotConfig = field(default_factory=SlotConfig.default)
    num_blueprints: int = NUM_BLUEPRINTS
    num_blends: int = NUM_BLENDS
    num_ops: int = NUM_OPS
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Initialized in __post_init__
    step_counts: list[int] = field(default_factory=list, init=False)

    # Pre-allocated tensors (set in __post_init__)
    states: torch.Tensor = field(init=False)
    slot_actions: torch.Tensor = field(init=False)
    blueprint_actions: torch.Tensor = field(init=False)
    blend_actions: torch.Tensor = field(init=False)
    op_actions: torch.Tensor = field(init=False)
    slot_log_probs: torch.Tensor = field(init=False)
    blueprint_log_probs: torch.Tensor = field(init=False)
    blend_log_probs: torch.Tensor = field(init=False)
    op_log_probs: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    rewards: torch.Tensor = field(init=False)
    dones: torch.Tensor = field(init=False)
    truncated: torch.Tensor = field(init=False)
    bootstrap_values: torch.Tensor = field(init=False)
    slot_masks: torch.Tensor = field(init=False)
    blueprint_masks: torch.Tensor = field(init=False)
    blend_masks: torch.Tensor = field(init=False)
    op_masks: torch.Tensor = field(init=False)
    hidden_h: torch.Tensor = field(init=False)
    hidden_c: torch.Tensor = field(init=False)
    advantages: torch.Tensor = field(init=False)
    returns: torch.Tensor = field(init=False)

    # Episode boundary tracking
    _current_episode_start: dict[int, int] = field(default_factory=dict, init=False)
    episode_boundaries: dict[int, list[tuple[int, int]]] = field(
        default_factory=dict, init=False
    )

    @property
    def num_slots(self) -> int:
        """Number of slots from slot_config."""
        return self.slot_config.num_slots

    def __post_init__(self):
        """Allocate all tensors upfront."""
        self.step_counts = [0] * self.num_envs
        device = self.device
        n = self.num_envs
        m = self.max_steps_per_env

        # Core state
        self.states = torch.zeros(n, m, self.state_dim, device=device)

        # Factored actions
        self.slot_actions = torch.zeros(n, m, dtype=torch.long, device=device)
        self.blueprint_actions = torch.zeros(n, m, dtype=torch.long, device=device)
        self.blend_actions = torch.zeros(n, m, dtype=torch.long, device=device)
        self.op_actions = torch.zeros(n, m, dtype=torch.long, device=device)

        # Per-head log probs
        self.slot_log_probs = torch.zeros(n, m, device=device)
        self.blueprint_log_probs = torch.zeros(n, m, device=device)
        self.blend_log_probs = torch.zeros(n, m, device=device)
        self.op_log_probs = torch.zeros(n, m, device=device)

        # Values, rewards, dones
        self.values = torch.zeros(n, m, device=device)
        self.rewards = torch.zeros(n, m, device=device)
        self.dones = torch.zeros(n, m, dtype=torch.bool, device=device)
        self.truncated = torch.zeros(n, m, dtype=torch.bool, device=device)
        self.bootstrap_values = torch.zeros(n, m, device=device)

        # Action masks
        self.slot_masks = torch.zeros(n, m, self.num_slots, dtype=torch.bool, device=device)
        self.blueprint_masks = torch.zeros(n, m, self.num_blueprints, dtype=torch.bool, device=device)
        self.blend_masks = torch.zeros(n, m, self.num_blends, dtype=torch.bool, device=device)
        self.op_masks = torch.zeros(n, m, self.num_ops, dtype=torch.bool, device=device)

        # LSTM hidden states: [num_envs, max_steps, lstm_layers, hidden_dim]
        self.hidden_h = torch.zeros(n, m, self.lstm_layers, self.lstm_hidden_dim, device=device)
        self.hidden_c = torch.zeros(n, m, self.lstm_layers, self.lstm_hidden_dim, device=device)

        # Computed during finalization
        self.advantages = torch.zeros(n, m, device=device)
        self.returns = torch.zeros(n, m, device=device)

    def start_episode(self, env_id: int) -> None:
        """Mark start of a new episode for an environment."""
        self._current_episode_start[env_id] = self.step_counts[env_id]

    def end_episode(self, env_id: int) -> None:
        """Mark end of episode, recording boundary."""
        if env_id not in self._current_episode_start:
            return
        start = self._current_episode_start[env_id]
        end = self.step_counts[env_id]
        if env_id not in self.episode_boundaries:
            self.episode_boundaries[env_id] = []
        self.episode_boundaries[env_id].append((start, end))
        del self._current_episode_start[env_id]

    def add(
        self,
        env_id: int,
        state: torch.Tensor,
        slot_action: int,
        blueprint_action: int,
        blend_action: int,
        op_action: int,
        slot_log_prob: float,
        blueprint_log_prob: float,
        blend_log_prob: float,
        op_log_prob: float,
        value: float,
        reward: float,
        done: bool,
        slot_mask: torch.Tensor,
        blueprint_mask: torch.Tensor,
        blend_mask: torch.Tensor,
        op_mask: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Add a transition for a specific environment.

        Direct tensor assignment - no list append overhead.
        """
        step_idx = self.step_counts[env_id]

        if step_idx >= self.max_steps_per_env:
            raise RuntimeError(
                f"Environment {env_id} exceeded max_steps ({self.max_steps_per_env}). "
                f"Call reset() or compute_advantages_and_returns() first."
            )

        # Direct tensor assignment
        self.states[env_id, step_idx] = state.detach()
        self.slot_actions[env_id, step_idx] = slot_action
        self.blueprint_actions[env_id, step_idx] = blueprint_action
        self.blend_actions[env_id, step_idx] = blend_action
        self.op_actions[env_id, step_idx] = op_action
        self.slot_log_probs[env_id, step_idx] = slot_log_prob
        self.blueprint_log_probs[env_id, step_idx] = blueprint_log_prob
        self.blend_log_probs[env_id, step_idx] = blend_log_prob
        self.op_log_probs[env_id, step_idx] = op_log_prob
        self.values[env_id, step_idx] = value
        self.rewards[env_id, step_idx] = reward
        self.dones[env_id, step_idx] = done
        self.truncated[env_id, step_idx] = truncated
        self.bootstrap_values[env_id, step_idx] = bootstrap_value
        self.slot_masks[env_id, step_idx] = slot_mask.detach().bool()
        self.blueprint_masks[env_id, step_idx] = blueprint_mask.detach().bool()
        self.blend_masks[env_id, step_idx] = blend_mask.detach().bool()
        self.op_masks[env_id, step_idx] = op_mask.detach().bool()
        # Hidden state: LSTM returns [num_layers, batch, hidden_dim]
        # Squeeze batch dim (dim=1) to get [num_layers, hidden_dim]
        self.hidden_h[env_id, step_idx] = hidden_h.detach().squeeze(1)
        self.hidden_c[env_id, step_idx] = hidden_c.detach().squeeze(1)

        self.step_counts[env_id] = step_idx + 1

    @torch.compiler.disable  # Python loops cause graph breaks; runs once per rollout
    def compute_advantages_and_returns(
        self,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages per-environment (no cross-contamination).

        This is the P0 bug fix: each environment's GAE is computed
        independently using only that environment's values and rewards.
        """
        for env_id in range(self.num_envs):
            num_steps = self.step_counts[env_id]
            if num_steps == 0:
                continue

            # Work with valid slice only
            values = self.values[env_id, :num_steps]
            rewards = self.rewards[env_id, :num_steps]
            dones = self.dones[env_id, :num_steps]
            truncated = self.truncated[env_id, :num_steps]
            bootstrap_values = self.bootstrap_values[env_id, :num_steps]

            advantages = torch.zeros(num_steps, device=self.device)
            last_gae: torch.Tensor = torch.tensor(0.0, device=self.device)

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    # Last step: use bootstrap value if truncated
                    if truncated[t]:
                        next_value = bootstrap_values[t]
                        # Truncation is NOT a true terminal - the episode was cut off
                        # by time limit. We MUST use next_non_terminal=1.0 so the
                        # bootstrap value contributes to delta and GAE propagates.
                        next_non_terminal = 1.0
                    else:
                        next_value = torch.tensor(0.0, device=self.device)
                        next_non_terminal = 1.0 - float(dones[t])
                else:
                    next_value = values[t + 1]
                    # For non-terminal steps, only reset on TRUE terminal (not truncation)
                    next_non_terminal = 1.0 - float(dones[t] and not truncated[t])

                # Reset GAE at true terminal (not truncation)
                if dones[t] and not truncated[t]:
                    last_gae = torch.tensor(0.0, device=self.device)

                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            self.advantages[env_id, :num_steps] = advantages
            self.returns[env_id, :num_steps] = advantages + values

    def normalize_advantages(self) -> None:
        """Normalize advantages globally across all environments."""
        # Collect all valid advantages
        all_advantages = []
        for env_id in range(self.num_envs):
            num_steps = self.step_counts[env_id]
            if num_steps > 0:
                all_advantages.append(self.advantages[env_id, :num_steps])

        if not all_advantages:
            return

        # Compute global stats
        all_adv = torch.cat(all_advantages)
        mean = all_adv.mean()
        std = all_adv.std()

        # Normalize in-place
        for env_id in range(self.num_envs):
            num_steps = self.step_counts[env_id]
            if num_steps > 0:
                self.advantages[env_id, :num_steps] = (
                    self.advantages[env_id, :num_steps] - mean
                ) / (std + 1e-8)

    def get_batched_sequences(
        self,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Get all data as batched sequences [num_envs, max_steps, ...].

        Returns dict with all tensors moved to target device.
        Includes valid_mask to indicate which timesteps have real data.
        """
        device = torch.device(device) if isinstance(device, str) else device

        # Build valid mask - vectorized (avoids Python loop)
        # step_idx < step_counts[env_id] using broadcasting
        step_counts_tensor = torch.tensor(self.step_counts, device=device).unsqueeze(1)
        step_indices = torch.arange(self.max_steps_per_env, device=device).unsqueeze(0)
        valid_mask = step_indices < step_counts_tensor  # [num_envs, max_steps]

        # Use non_blocking=True for async CPU->GPU transfer (overlaps with computation)
        # This only helps when transferring TO CUDA - CPU transfers ignore it
        nb = device.type == "cuda"

        return {
            "states": self.states.to(device, non_blocking=nb),
            "slot_actions": self.slot_actions.to(device, non_blocking=nb),
            "blueprint_actions": self.blueprint_actions.to(device, non_blocking=nb),
            "blend_actions": self.blend_actions.to(device, non_blocking=nb),
            "op_actions": self.op_actions.to(device, non_blocking=nb),
            "slot_log_probs": self.slot_log_probs.to(device, non_blocking=nb),
            "blueprint_log_probs": self.blueprint_log_probs.to(device, non_blocking=nb),
            "blend_log_probs": self.blend_log_probs.to(device, non_blocking=nb),
            "op_log_probs": self.op_log_probs.to(device, non_blocking=nb),
            "values": self.values.to(device, non_blocking=nb),
            "rewards": self.rewards.to(device, non_blocking=nb),
            "advantages": self.advantages.to(device, non_blocking=nb),
            "returns": self.returns.to(device, non_blocking=nb),
            "slot_masks": self.slot_masks.to(device, non_blocking=nb),
            "blueprint_masks": self.blueprint_masks.to(device, non_blocking=nb),
            "blend_masks": self.blend_masks.to(device, non_blocking=nb),
            "op_masks": self.op_masks.to(device, non_blocking=nb),
            "hidden_h": self.hidden_h.to(device, non_blocking=nb),
            "hidden_c": self.hidden_c.to(device, non_blocking=nb),
            "valid_mask": valid_mask,
            # Initial hidden states for each env (first timestep)
            # Buffer stores [num_envs, max_steps, lstm_layers, hidden_dim]
            # LSTM expects [lstm_layers, batch, hidden_dim], so permute after slicing
            "initial_hidden_h": self.hidden_h[:, 0, :, :].permute(1, 0, 2).contiguous().to(device, non_blocking=nb),
            "initial_hidden_c": self.hidden_c[:, 0, :, :].permute(1, 0, 2).contiguous().to(device, non_blocking=nb),
        }

    def reset(self) -> None:
        """Reset buffer for new episode collection."""
        self.step_counts = [0] * self.num_envs
        self._current_episode_start = {}
        self.episode_boundaries = {}
        # Tensors don't need zeroing - step_counts controls valid range

    def clear_env(self, env_id: int) -> None:
        """Clear transitions for a single environment.

        Used for per-environment rollback when only one env panics.
        More sample-efficient than clearing the entire buffer.

        Args:
            env_id: Index of the environment to clear

        Raises:
            ValueError: If env_id is out of range [0, num_envs)
        """
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id {env_id} out of range [0, {self.num_envs})")
        self.step_counts[env_id] = 0
        # Clear episode tracking for this env
        if env_id in self._current_episode_start:
            del self._current_episode_start[env_id]
        if env_id in self.episode_boundaries:
            del self.episode_boundaries[env_id]
        # Zero LSTM hidden states for this env to prevent stale state leakage.
        # When new transitions are added, they'll start with fresh hidden states.
        self.hidden_h[env_id].zero_()
        self.hidden_c[env_id].zero_()

    def __len__(self) -> int:
        """Total transitions across all environments."""
        return sum(self.step_counts)


__all__ = ["TamiyoRolloutStep", "TamiyoRolloutBuffer"]
