# Tamiyo Unified Buffer & Recurrent Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Expert Review (2025-12-13)

This plan was reviewed by three specialist agents. All critical issues have been addressed:

| Reviewer | Verdict | Key Fixes Applied |
|----------|---------|-------------------|
| Code Reviewer | Approved | Buffer init guards, weight_decay optimizer logic, test coverage gaps |
| DRL Expert | Approved | Entropy normalization, anneal steps fix, GAE logic verified correct |
| PyTorch Expert | Approved | LSTM LayerNorm added, `@torch.compiler.disable` on GAE, `inference_mode()` |

**Final Review Pass (2025-12-13):**

| Issue | Resolution |
|-------|------------|
| `recurrent_n_epochs` not stored in `__init__` | Added to Task 4 Step 3 parameter list and init body |
| `_base_network` vs `network` inconsistency | Fixed to use `self.network` consistently |
| Tasks 7-8 coupling (broken state between tasks) | Added warning; combined into single commit |
| Task 8 test audit | Added Step 3.5 to grep for deleted method usages |

---

**Goal:** Fix P0 GAE interleaving bug, restore LSTM for temporal reasoning over 10-20 epoch seed learning cycles, and unify factored actions with recurrence.

**Architecture:** Create `TamiyoRolloutBuffer` (per-env storage + LSTM states + factored actions) and `FactoredRecurrentActorCritic` (feature extraction → LSTM → factored heads). This replaces the mutually exclusive `FactoredRolloutBuffer`/`RecurrentRolloutBuffer` split.

**Tech Stack:** Python 3.13, PyTorch 2.9, pytest

**Critical Context:**
- Seeds take 10-20 epochs to learn; episodes are 25 epochs
- Current 5-epoch history window is insufficient for temporal reasoning
- FactoredActorCritic is pure feedforward - LSTM was lost in refactor
- FactoredRolloutBuffer stores interleaved across envs → GAE uses wrong env values (P0 bug)

---

## Task 1: TamiyoRolloutStep and TamiyoRolloutBuffer

**Problem:** FactoredRolloutBuffer stores transitions interleaved across environments. GAE computes sequentially over this list, using values from wrong environments.

**Solution:** Create unified buffer with:
- Per-environment storage (like RecurrentRolloutBuffer)
- Factored actions (4 heads)
- Per-head log probs (not joint)
- LSTM hidden states per-step
- Pre-allocated tensors for memory efficiency

**Files:**
- Create: `src/esper/simic/tamiyo_buffer.py`
- Create: `tests/simic/test_tamiyo_buffer.py`

**Step 1: Write failing test for per-env storage**

Create `tests/simic/test_tamiyo_buffer.py`:

```python
"""Tests for TamiyoRolloutBuffer - unified factored recurrent buffer."""

import pytest
import torch

from esper.simic.tamiyo_buffer import TamiyoRolloutBuffer, TamiyoRolloutStep


class TestTamiyoRolloutBuffer:
    """Tests for the unified Tamiyo rollout buffer."""

    def test_per_env_storage_isolation(self):
        """Transitions from different envs must not contaminate each other's GAE.

        This is the P0 bug fix: interleaved storage caused GAE to use values
        from wrong environments.
        """
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=128,
        )

        # Env 0: low rewards (1.0)
        buffer.start_episode(env_id=0)
        for i in range(3):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=0,
                blueprint_action=0,
                blend_action=0,
                op_action=0,
                slot_log_prob=-1.0,
                blueprint_log_prob=-1.0,
                blend_log_prob=-1.0,
                op_log_prob=-1.0,
                value=1.0,
                reward=1.0,
                done=(i == 2),
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(5, dtype=torch.bool),
                blend_mask=torch.ones(3, dtype=torch.bool),
                op_mask=torch.ones(4, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 128),
                hidden_c=torch.zeros(1, 1, 128),
            )
        buffer.end_episode(env_id=0)

        # Env 1: HIGH rewards (100.0) - should NOT affect env 0's GAE
        buffer.start_episode(env_id=1)
        for i in range(3):
            buffer.add(
                env_id=1,
                state=torch.randn(50),
                slot_action=0,
                blueprint_action=0,
                blend_action=0,
                op_action=0,
                slot_log_prob=-1.0,
                blueprint_log_prob=-1.0,
                blend_log_prob=-1.0,
                op_log_prob=-1.0,
                value=50.0,
                reward=100.0,  # HIGH reward
                done=(i == 2),
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(5, dtype=torch.bool),
                blend_mask=torch.ones(3, dtype=torch.bool),
                op_mask=torch.ones(4, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 128),
                hidden_c=torch.zeros(1, 1, 128),
            )
        buffer.end_episode(env_id=1)

        buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)

        # Env 0's advantages should be based ONLY on env 0's rewards (1.0)
        # They should be small, NOT contaminated by env 1's high rewards
        env0_advantages = buffer.advantages[0, :3]
        env1_advantages = buffer.advantages[1, :3]

        # Env 0 advantages should be MUCH smaller than env 1
        assert env0_advantages.abs().max() < 10.0, (
            f"Env 0 advantages {env0_advantages} contaminated by env 1 high rewards"
        )
        assert env1_advantages.abs().max() > 10.0, (
            f"Env 1 advantages {env1_advantages} should be large from high rewards"
        )

    def test_per_head_log_probs_stored(self):
        """Buffer must store per-head log probs, not just joint."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=128,
        )

        buffer.start_episode(env_id=0)
        buffer.add(
            env_id=0,
            state=torch.randn(50),
            slot_action=1,
            blueprint_action=2,
            blend_action=0,
            op_action=1,
            slot_log_prob=-0.5,
            blueprint_log_prob=-1.2,
            blend_log_prob=-0.3,
            op_log_prob=-0.8,
            value=1.0,
            reward=1.0,
            done=True,
            slot_mask=torch.ones(3, dtype=torch.bool),
            blueprint_mask=torch.ones(5, dtype=torch.bool),
            blend_mask=torch.ones(3, dtype=torch.bool),
            op_mask=torch.ones(4, dtype=torch.bool),
            hidden_h=torch.zeros(1, 1, 128),
            hidden_c=torch.zeros(1, 1, 128),
        )
        buffer.end_episode(env_id=0)

        # Check per-head log probs are stored correctly
        assert buffer.slot_log_probs[0, 0].item() == pytest.approx(-0.5)
        assert buffer.blueprint_log_probs[0, 0].item() == pytest.approx(-1.2)
        assert buffer.blend_log_probs[0, 0].item() == pytest.approx(-0.3)
        assert buffer.op_log_probs[0, 0].item() == pytest.approx(-0.8)

    def test_lstm_hidden_states_stored(self):
        """Buffer must store LSTM hidden states for sequence reconstruction."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=128,
        )

        hidden_h = torch.randn(1, 1, 128)
        hidden_c = torch.randn(1, 1, 128)

        buffer.start_episode(env_id=0)
        buffer.add(
            env_id=0,
            state=torch.randn(50),
            slot_action=0,
            blueprint_action=0,
            blend_action=0,
            op_action=0,
            slot_log_prob=-1.0,
            blueprint_log_prob=-1.0,
            blend_log_prob=-1.0,
            op_log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=True,
            slot_mask=torch.ones(3, dtype=torch.bool),
            blueprint_mask=torch.ones(5, dtype=torch.bool),
            blend_mask=torch.ones(3, dtype=torch.bool),
            op_mask=torch.ones(4, dtype=torch.bool),
            hidden_h=hidden_h,
            hidden_c=hidden_c,
        )
        buffer.end_episode(env_id=0)

        # Hidden states should be stored (squeezed from [1, 1, 128] to [128])
        stored_h = buffer.hidden_h[0, 0]
        assert stored_h.shape == (1, 128)
        assert torch.allclose(stored_h.squeeze(0), hidden_h.squeeze())

    def test_empty_buffer_update(self):
        """update with empty buffer should return empty dict, not crash."""
        buffer = TamiyoRolloutBuffer(
            num_envs=2,
            max_steps_per_env=5,
            state_dim=50,
            lstm_hidden_dim=128,
        )

        # Should not crash - returns without computing anything
        buffer.compute_advantages_and_returns(gamma=0.99, gae_lambda=0.95)
        assert len(buffer) == 0

    def test_buffer_overflow_raises(self):
        """Exceeding max_steps_per_env should raise RuntimeError."""
        buffer = TamiyoRolloutBuffer(
            num_envs=1,
            max_steps_per_env=2,  # Very small
            state_dim=50,
            lstm_hidden_dim=128,
        )

        buffer.start_episode(env_id=0)
        # Add 2 transitions (at limit)
        for i in range(2):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=0, blueprint_action=0, blend_action=0, op_action=0,
                slot_log_prob=-1.0, blueprint_log_prob=-1.0,
                blend_log_prob=-1.0, op_log_prob=-1.0,
                value=1.0, reward=1.0, done=False,
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(5, dtype=torch.bool),
                blend_mask=torch.ones(3, dtype=torch.bool),
                op_mask=torch.ones(4, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 128),
                hidden_c=torch.zeros(1, 1, 128),
            )

        # Third add should raise
        with pytest.raises(RuntimeError, match="exceeded max_steps"):
            buffer.add(
                env_id=0,
                state=torch.randn(50),
                slot_action=0, blueprint_action=0, blend_action=0, op_action=0,
                slot_log_prob=-1.0, blueprint_log_prob=-1.0,
                blend_log_prob=-1.0, op_log_prob=-1.0,
                value=1.0, reward=1.0, done=False,
                slot_mask=torch.ones(3, dtype=torch.bool),
                blueprint_mask=torch.ones(5, dtype=torch.bool),
                blend_mask=torch.ones(3, dtype=torch.bool),
                op_mask=torch.ones(4, dtype=torch.bool),
                hidden_h=torch.zeros(1, 1, 128),
                hidden_c=torch.zeros(1, 1, 128),
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_tamiyo_buffer.py -v`
Expected: FAIL with ImportError (module doesn't exist)

**Step 3: Create TamiyoRolloutBuffer implementation**

Create `src/esper/simic/tamiyo_buffer.py`:

```python
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
from typing import Iterator, NamedTuple

import torch

from esper.leyline.factored_actions import (
    NUM_SLOTS,
    NUM_BLUEPRINTS,
    NUM_BLENDS,
    NUM_OPS,
    LifecycleOp,
)


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


@dataclass
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
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    num_slots: int = NUM_SLOTS
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
        gamma: float = 0.99,
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
            last_gae = 0.0

            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    # Last step: use bootstrap value if truncated
                    if truncated[t]:
                        next_value = bootstrap_values[t]
                    else:
                        next_value = 0.0
                    next_non_terminal = 1.0 - float(dones[t])
                else:
                    next_value = values[t + 1]
                    next_non_terminal = 1.0 - float(dones[t])

                # Reset GAE at true terminal (not truncation)
                if dones[t] and not truncated[t]:
                    last_gae = 0.0

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

        # Build valid mask
        valid_mask = torch.zeros(
            self.num_envs, self.max_steps_per_env, dtype=torch.bool, device=device
        )
        for env_id in range(self.num_envs):
            valid_mask[env_id, : self.step_counts[env_id]] = True

        return {
            "states": self.states.to(device),
            "slot_actions": self.slot_actions.to(device),
            "blueprint_actions": self.blueprint_actions.to(device),
            "blend_actions": self.blend_actions.to(device),
            "op_actions": self.op_actions.to(device),
            "slot_log_probs": self.slot_log_probs.to(device),
            "blueprint_log_probs": self.blueprint_log_probs.to(device),
            "blend_log_probs": self.blend_log_probs.to(device),
            "op_log_probs": self.op_log_probs.to(device),
            "values": self.values.to(device),
            "rewards": self.rewards.to(device),
            "advantages": self.advantages.to(device),
            "returns": self.returns.to(device),
            "slot_masks": self.slot_masks.to(device),
            "blueprint_masks": self.blueprint_masks.to(device),
            "blend_masks": self.blend_masks.to(device),
            "op_masks": self.op_masks.to(device),
            "hidden_h": self.hidden_h.to(device),
            "hidden_c": self.hidden_c.to(device),
            "valid_mask": valid_mask,
            # Initial hidden states for each env (first timestep)
            "initial_hidden_h": self.hidden_h[:, 0, :, :].to(device),
            "initial_hidden_c": self.hidden_c[:, 0, :, :].to(device),
        }

    def reset(self) -> None:
        """Reset buffer for new episode collection."""
        self.step_counts = [0] * self.num_envs
        self._current_episode_start = {}
        self.episode_boundaries = {}
        # Tensors don't need zeroing - step_counts controls valid range

    def __len__(self) -> int:
        """Total transitions across all environments."""
        return sum(self.step_counts)


__all__ = ["TamiyoRolloutStep", "TamiyoRolloutBuffer"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/simic/test_tamiyo_buffer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/esper/simic/tamiyo_buffer.py tests/simic/test_tamiyo_buffer.py
git commit -m "$(cat <<'EOF'
feat(simic): add TamiyoRolloutBuffer with per-env storage

Fixes P0 bug where FactoredRolloutBuffer stored transitions interleaved
across environments, causing GAE to use values from wrong envs.

New TamiyoRolloutBuffer provides:
- Per-environment storage (GAE isolation)
- Factored actions (4 heads)
- Per-head log probs (credit attribution)
- LSTM hidden state tracking (temporal memory)
- Pre-allocated tensors (memory efficiency)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: FactoredRecurrentActorCritic Network

**Problem:** FactoredActorCritic is pure feedforward. Seeds take 10-20 epochs to learn, but feedforward policy can't distinguish "germinated 5 epochs ago" from "germinated 18 epochs ago".

**Solution:** Create `FactoredRecurrentActorCritic` with:
- Feature extraction → LSTM → factored heads
- All heads share temporal context from LSTM
- Per-head entropy for exploration control

**Files:**
- Create: `src/esper/simic/tamiyo_network.py`
- Create: `tests/simic/test_tamiyo_network.py`

**Step 1: Write failing test**

Create `tests/simic/test_tamiyo_network.py`:

```python
"""Tests for FactoredRecurrentActorCritic - Tamiyo's neural network."""

import pytest
import torch

from esper.simic.tamiyo_network import FactoredRecurrentActorCritic


class TestFactoredRecurrentActorCritic:
    """Tests for the factored recurrent network."""

    def test_forward_returns_all_heads(self):
        """Forward pass must return logits for all 4 factored heads."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 1, 50)  # [batch, seq, state_dim]

        output = net(state)

        assert "slot_logits" in output
        assert "blueprint_logits" in output
        assert "blend_logits" in output
        assert "op_logits" in output
        assert "value" in output
        assert "hidden" in output

        # Check shapes
        assert output["slot_logits"].shape == (2, 1, 3)  # NUM_SLOTS=3
        assert output["blueprint_logits"].shape == (2, 1, 5)  # NUM_BLUEPRINTS=5
        assert output["blend_logits"].shape == (2, 1, 3)  # NUM_BLENDS=3
        assert output["op_logits"].shape == (2, 1, 4)  # NUM_OPS=4
        assert output["value"].shape == (2, 1)

    def test_hidden_state_propagates(self):
        """LSTM hidden state must propagate across time steps."""
        net = FactoredRecurrentActorCritic(state_dim=50, lstm_hidden_dim=64)

        batch_size = 2
        hidden = net.get_initial_hidden(batch_size, torch.device("cpu"))

        # First step
        state1 = torch.randn(batch_size, 1, 50)
        output1 = net(state1, hidden=hidden)

        # Second step with updated hidden
        state2 = torch.randn(batch_size, 1, 50)
        output2 = net(state2, hidden=output1["hidden"])

        # Hidden states should be different after processing different inputs
        h1, c1 = output1["hidden"]
        h2, c2 = output2["hidden"]
        assert not torch.allclose(h1, h2), "Hidden states should change between steps"

    def test_masks_applied_correctly(self):
        """Invalid actions must have -inf logits after masking."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(1, 1, 50)

        # Mask out slot 0 and 2, only slot 1 valid
        slot_mask = torch.tensor([[[False, True, False]]])
        op_mask = torch.tensor([[[True, True, True, True]]])

        output = net(
            state,
            slot_mask=slot_mask,
            blueprint_mask=None,
            blend_mask=None,
            op_mask=op_mask,
        )

        slot_logits = output["slot_logits"][0, 0]
        assert slot_logits[0] == float("-inf")
        assert slot_logits[1] != float("-inf")
        assert slot_logits[2] == float("-inf")

    def test_per_head_log_probs(self):
        """evaluate_actions must return per-head log probs."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 5, 50)  # [batch, seq, state_dim]
        actions = {
            "slot": torch.zeros(2, 5, dtype=torch.long),
            "blueprint": torch.zeros(2, 5, dtype=torch.long),
            "blend": torch.zeros(2, 5, dtype=torch.long),
            "op": torch.zeros(2, 5, dtype=torch.long),
        }

        log_probs, values, entropy, hidden = net.evaluate_actions(state, actions)

        # Per-head log probs
        assert "slot" in log_probs
        assert "blueprint" in log_probs
        assert "blend" in log_probs
        assert "op" in log_probs

        # All should be [batch, seq]
        assert log_probs["slot"].shape == (2, 5)
        assert log_probs["blueprint"].shape == (2, 5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_tamiyo_network.py -v`
Expected: FAIL with ImportError

**Step 3: Create FactoredRecurrentActorCritic implementation**

Create `src/esper/simic/tamiyo_network.py`:

```python
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

        # Apply masks (set invalid actions to -inf)
        if slot_mask is not None:
            slot_logits = slot_logits.masked_fill(~slot_mask, float("-inf"))
        if blueprint_mask is not None:
            blueprint_logits = blueprint_logits.masked_fill(~blueprint_mask, float("-inf"))
        if blend_mask is not None:
            blend_logits = blend_logits.masked_fill(~blend_mask, float("-inf"))
        if op_mask is not None:
            op_logits = op_logits.masked_fill(~op_mask, float("-inf"))

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/simic/test_tamiyo_network.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/esper/simic/tamiyo_network.py tests/simic/test_tamiyo_network.py
git commit -m "$(cat <<'EOF'
feat(simic): add FactoredRecurrentActorCritic with LSTM

Restores temporal reasoning lost when switching to factored actions.
Seeds take 10-20 epochs to learn; LSTM enables distinguishing
"seed germinated 5 epochs ago" from "seed germinated 18 epochs ago".

Architecture:
- Feature extraction -> LSTM -> factored heads
- All heads share temporal context
- Per-head log probs for credit attribution
- Orthogonal init with forget gate bias = 1

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Per-Head Advantages with Causal Masking

**Problem:** Computing one joint advantage for all heads assigns credit/blame to heads that had no causal effect. For example, when op=WAIT, slot/blueprint/blend choices don't matter.

**Solution:** Compute per-head advantages with causal masking based on op type:
- op_head: always gets advantage (always causally relevant)
- slot_head: relevant for GERMINATE, FOSSILIZE, CULL (not WAIT)
- blueprint_head: only relevant for GERMINATE
- blend_head: only relevant for GERMINATE

**Files:**
- Create: `src/esper/simic/advantages.py`
- Create: `tests/simic/test_advantages.py`

**Step 1: Write failing test**

Create `tests/simic/test_advantages.py`:

```python
"""Tests for per-head advantage computation with causal masking."""

import pytest
import torch

from esper.simic.advantages import compute_per_head_advantages
from esper.leyline.factored_actions import LifecycleOp


class TestPerHeadAdvantages:
    """Tests for causal advantage masking."""

    def test_wait_masks_non_op_heads(self):
        """When op=WAIT, slot/blueprint/blend should get zero advantage."""
        op_actions = torch.tensor([LifecycleOp.WAIT, LifecycleOp.WAIT])
        base_advantages = torch.tensor([1.0, 2.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # op always gets advantage
        assert torch.allclose(per_head["op"], base_advantages)

        # Others should be zero when WAIT
        assert torch.allclose(per_head["slot"], torch.zeros(2))
        assert torch.allclose(per_head["blueprint"], torch.zeros(2))
        assert torch.allclose(per_head["blend"], torch.zeros(2))

    def test_germinate_all_heads_active(self):
        """When op=GERMINATE, all heads should get advantage."""
        op_actions = torch.tensor([LifecycleOp.GERMINATE, LifecycleOp.GERMINATE])
        base_advantages = torch.tensor([1.0, 2.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # All heads active for GERMINATE
        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["blueprint"], base_advantages)
        assert torch.allclose(per_head["blend"], base_advantages)

    def test_fossilize_slot_active_others_masked(self):
        """When op=FOSSILIZE, slot and op active, blueprint/blend masked."""
        op_actions = torch.tensor([LifecycleOp.FOSSILIZE])
        base_advantages = torch.tensor([5.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["blueprint"], torch.zeros(1))
        assert torch.allclose(per_head["blend"], torch.zeros(1))

    def test_cull_slot_active_others_masked(self):
        """When op=CULL, slot and op active, blueprint/blend masked."""
        op_actions = torch.tensor([LifecycleOp.CULL])
        base_advantages = torch.tensor([3.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        assert torch.allclose(per_head["op"], base_advantages)
        assert torch.allclose(per_head["slot"], base_advantages)
        assert torch.allclose(per_head["blueprint"], torch.zeros(1))
        assert torch.allclose(per_head["blend"], torch.zeros(1))

    def test_mixed_ops_correct_masking(self):
        """Mixed op types should apply correct masking per timestep."""
        op_actions = torch.tensor([
            LifecycleOp.WAIT,
            LifecycleOp.GERMINATE,
            LifecycleOp.FOSSILIZE,
            LifecycleOp.CULL,
        ])
        base_advantages = torch.tensor([1.0, 2.0, 3.0, 4.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # op always active
        assert torch.allclose(per_head["op"], base_advantages)

        # slot: active for GERMINATE, FOSSILIZE, CULL (indices 1, 2, 3)
        expected_slot = torch.tensor([0.0, 2.0, 3.0, 4.0])
        assert torch.allclose(per_head["slot"], expected_slot)

        # blueprint/blend: only active for GERMINATE (index 1)
        expected_blueprint = torch.tensor([0.0, 2.0, 0.0, 0.0])
        assert torch.allclose(per_head["blueprint"], expected_blueprint)
        assert torch.allclose(per_head["blend"], expected_blueprint)

    def test_only_wait_episode_sparse_gradients(self):
        """Episode with only WAIT actions should only give op_head gradients.

        This tests an edge case where blueprint/blend heads receive zero
        gradient signal for the entire episode.
        """
        op_actions = torch.tensor([
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
            LifecycleOp.WAIT,
        ])
        base_advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        per_head = compute_per_head_advantages(base_advantages, op_actions)

        # Only op should receive gradients
        assert torch.allclose(per_head["op"], base_advantages)

        # All other heads should be zero (no causal effect)
        assert torch.allclose(per_head["slot"], torch.zeros(5))
        assert torch.allclose(per_head["blueprint"], torch.zeros(5))
        assert torch.allclose(per_head["blend"], torch.zeros(5))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_advantages.py -v`
Expected: FAIL with ImportError

**Step 3: Create advantages module**

Create `src/esper/simic/advantages.py`:

```python
"""Per-head advantage computation with causal masking.

Causal structure for Tamiyo's factored action space:

    DECISION TREE AT EACH EPOCH:

    op_head decides: [WAIT, GERMINATE, FOSSILIZE, CULL]
        |
        +-- WAIT: No other heads matter
        |
        +-- GERMINATE:
        |   +-- slot_head: WHERE to place seed
        |   +-- blueprint_head: WHAT architecture
        |   +-- blend_head: HOW to blend
        |
        +-- FOSSILIZE:
        |   +-- slot_head: WHICH seed to fossilize (target_slot)
        |
        +-- CULL:
            +-- slot_head: WHICH seed to remove (target_slot)

When computing advantages, we mask out heads that had no causal effect
on the outcome. This reduces gradient noise significantly.
"""

from __future__ import annotations

import torch

from esper.leyline.factored_actions import LifecycleOp


def compute_per_head_advantages(
    base_advantages: torch.Tensor,
    op_actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute advantages with causal masking per head.

    Args:
        base_advantages: GAE advantages [batch] or [batch, seq]
        op_actions: Operation actions [batch] or [batch, seq] (LifecycleOp values)

    Returns:
        Dict with per-head advantages, causally masked.
    """
    device = base_advantages.device

    # Create causal masks based on op type
    is_wait = op_actions == LifecycleOp.WAIT
    is_germinate = op_actions == LifecycleOp.GERMINATE
    is_fossilize = op_actions == LifecycleOp.FOSSILIZE
    is_cull = op_actions == LifecycleOp.CULL

    # op head: always gets advantage (always causally relevant)
    op_advantages = base_advantages.clone()

    # slot head: relevant for GERMINATE, FOSSILIZE, CULL (not WAIT)
    slot_mask = ~is_wait
    slot_advantages = base_advantages * slot_mask.float()

    # blueprint head: only relevant for GERMINATE
    blueprint_mask = is_germinate
    blueprint_advantages = base_advantages * blueprint_mask.float()

    # blend head: only relevant for GERMINATE
    blend_mask = is_germinate
    blend_advantages = base_advantages * blend_mask.float()

    return {
        "op": op_advantages,
        "slot": slot_advantages,
        "blueprint": blueprint_advantages,
        "blend": blend_advantages,
    }


def compute_per_head_policy_loss(
    per_head_advantages: dict[str, torch.Tensor],
    per_head_ratios: dict[str, torch.Tensor],
    clip_epsilon: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Compute clipped PPO loss per head.

    Args:
        per_head_advantages: Dict of masked advantages per head
        per_head_ratios: Dict of probability ratios per head
        clip_epsilon: PPO clipping parameter

    Returns:
        Dict of policy losses per head (to be summed)
    """
    losses = {}

    for key in ["op", "slot", "blueprint", "blend"]:
        ratio = per_head_ratios[key]
        advantages = per_head_advantages[key]

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        losses[key] = -torch.min(surr1, surr2)

    return losses


__all__ = ["compute_per_head_advantages", "compute_per_head_policy_loss"]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/simic/test_advantages.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/esper/simic/advantages.py tests/simic/test_advantages.py
git commit -m "$(cat <<'EOF'
feat(simic): add per-head advantages with causal masking

When op=WAIT, slot/blueprint/blend choices have no causal effect on
the outcome. Assigning advantage to them adds pure noise to gradients.

Causal masking rules:
- op: always gets advantage
- slot: active for GERMINATE, FOSSILIZE, CULL
- blueprint/blend: only active for GERMINATE

This significantly reduces gradient noise in multi-head policy.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: PPOAgent Integration

**Problem:** PPOAgent has `factored` and `recurrent` as mutually exclusive modes. We need unified `tamiyo` mode that combines both.

**Files:**
- Modify: `src/esper/simic/ppo.py`
- Modify: `tests/simic/test_ppo_agent.py`

**Step 1: Write failing test for tamiyo mode**

Add to `tests/simic/test_ppo_agent.py` (or create if not exists):

```python
def test_ppo_agent_tamiyo_mode():
    """PPOAgent in tamiyo mode should use FactoredRecurrentActorCritic."""
    from esper.simic.ppo import PPOAgent

    agent = PPOAgent(
        state_dim=50,
        tamiyo=True,  # New unified mode
        num_envs=4,
        max_steps_per_env=25,
        device="cpu",
    )

    # Should have TamiyoRolloutBuffer
    assert hasattr(agent, "tamiyo_buffer")
    assert agent.tamiyo_buffer is not None

    # Should have FactoredRecurrentActorCritic
    from esper.simic.tamiyo_network import FactoredRecurrentActorCritic
    assert isinstance(agent.network, FactoredRecurrentActorCritic)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_ppo_agent.py::test_ppo_agent_tamiyo_mode -v`
Expected: FAIL

**Step 3: Add tamiyo mode to PPOAgent.__init__**

In `src/esper/simic/ppo.py`, add after line 169 (after `factored: bool = False`):

```python
        # Tamiyo mode (unified factored + recurrent)
        tamiyo: bool = False,  # Use FactoredRecurrentActorCritic + TamiyoRolloutBuffer
        num_envs: int = 4,  # For TamiyoRolloutBuffer
        max_steps_per_env: int = 25,  # For TamiyoRolloutBuffer (matches max_epochs)
        recurrent_n_epochs: int = 1,  # PPO epochs for LSTM (keep low for stability)
```

Add to imports at top of file:

```python
from esper.simic.tamiyo_buffer import TamiyoRolloutBuffer
from esper.simic.tamiyo_network import FactoredRecurrentActorCritic
from esper.simic.advantages import compute_per_head_advantages
```

Replace the network initialization block (lines 197-211) with:

```python
        self.tamiyo = tamiyo
        self.num_envs = num_envs
        self.max_steps_per_env = max_steps_per_env
        self.recurrent_n_epochs = recurrent_n_epochs  # CRITICAL: used in update_tamiyo()

        # Initialize buffer attributes to None (guard against AttributeError)
        self.tamiyo_buffer = None
        self.factored_buffer = None
        self.recurrent_buffer = None

        if tamiyo:
            # Unified factored + recurrent mode
            self.network = FactoredRecurrentActorCritic(
                state_dim=state_dim,
                lstm_hidden_dim=lstm_hidden_dim,
            ).to(device)
            self.tamiyo_buffer = TamiyoRolloutBuffer(
                num_envs=num_envs,
                max_steps_per_env=max_steps_per_env,
                state_dim=state_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                device=torch.device(device),
            )
        elif factored:
            self.network = FactoredActorCritic(state_dim=state_dim).to(device)
            self.factored_buffer = FactoredRolloutBuffer()
        elif recurrent:
            self.network = RecurrentActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_dim=lstm_hidden_dim,
            ).to(device)
            self.recurrent_buffer = RecurrentRolloutBuffer(
                chunk_length=chunk_length,
                lstm_hidden_dim=lstm_hidden_dim,
            )
        else:
            self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
```

**Step 3.5: Add tamiyo branch to weight_decay optimizer logic**

In `src/esper/simic/ppo.py`, find the weight_decay parameter group logic (around lines 233-258) and add tamiyo branch:

```python
        # In the if weight_decay > 0 block, add after factored branch:
        elif self.tamiyo:
            actor_params = (
                list(self.network.slot_head.parameters()) +
                list(self.network.blueprint_head.parameters()) +
                list(self.network.blend_head.parameters()) +
                list(self.network.op_head.parameters())
            )
            critic_params = list(self.network.value_head.parameters())
            shared_params = (
                list(self.network.feature_net.parameters()) +
                list(self.network.lstm.parameters()) +
                list(self.network.lstm_ln.parameters())  # Include LayerNorm
            )
```

**Step 4: Add update_tamiyo method**

Add to PPOAgent class (after update_factored method):

```python
    def update_tamiyo(
        self,
        clear_buffer: bool = True,
    ) -> dict:
        """PPO update for Tamiyo (factored + recurrent).

        Uses per-head advantages with causal masking and LSTM hidden states.

        Args:
            clear_buffer: Whether to clear buffer after update

        Returns:
            Dict of training metrics
        """
        if len(self.tamiyo_buffer) == 0:
            return {}

        # Compute GAE per-environment (fixes P0 bug)
        self.tamiyo_buffer.compute_advantages_and_returns(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )
        self.tamiyo_buffer.normalize_advantages()

        # Get batched data
        data = self.tamiyo_buffer.get_batched_sequences(device=self.device)
        valid_mask = data["valid_mask"]

        # Compute explained variance before updates
        valid_values = data["values"][valid_mask]
        valid_returns = data["returns"][valid_mask]
        var_returns = valid_returns.var()
        if var_returns > 1e-8:
            explained_variance = 1.0 - (valid_returns - valid_values).var() / var_returns
            explained_variance = explained_variance.item()
        else:
            explained_variance = 0.0

        metrics = defaultdict(list)
        metrics["explained_variance"] = [explained_variance]
        early_stopped = False

        for epoch_i in range(self.recurrent_n_epochs):
            if early_stopped:
                break

            # Forward pass through network
            actions = {
                "slot": data["slot_actions"],
                "blueprint": data["blueprint_actions"],
                "blend": data["blend_actions"],
                "op": data["op_actions"],
            }

            log_probs, values, entropy, _ = self.network.evaluate_actions(
                data["states"],
                actions,
                slot_mask=data["slot_masks"],
                blueprint_mask=data["blueprint_masks"],
                blend_mask=data["blend_masks"],
                op_mask=data["op_masks"],
                hidden=(data["initial_hidden_h"], data["initial_hidden_c"]),
            )

            # Extract valid timesteps
            for key in log_probs:
                log_probs[key] = log_probs[key][valid_mask]
            values = values[valid_mask]
            for key in entropy:
                entropy[key] = entropy[key][valid_mask]

            valid_advantages = data["advantages"][valid_mask]
            valid_returns = data["returns"][valid_mask]

            # Compute per-head advantages with causal masking
            valid_op_actions = data["op_actions"][valid_mask]
            per_head_advantages = compute_per_head_advantages(
                valid_advantages, valid_op_actions
            )

            # Compute per-head ratios
            old_log_probs = {
                "slot": data["slot_log_probs"][valid_mask],
                "blueprint": data["blueprint_log_probs"][valid_mask],
                "blend": data["blend_log_probs"][valid_mask],
                "op": data["op_log_probs"][valid_mask],
            }

            per_head_ratios = {}
            for key in ["slot", "blueprint", "blend", "op"]:
                per_head_ratios[key] = torch.exp(log_probs[key] - old_log_probs[key])

            # Compute policy loss per head and sum
            policy_loss = 0.0
            for key in ["slot", "blueprint", "blend", "op"]:
                ratio = per_head_ratios[key]
                adv = per_head_advantages[key]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                head_loss = -torch.min(surr1, surr2).mean()
                policy_loss = policy_loss + head_loss

            # Value loss
            valid_old_values = data["values"][valid_mask]
            if self.clip_value:
                values_clipped = valid_old_values + torch.clamp(
                    values - valid_old_values, -self.clip_ratio, self.clip_ratio
                )
                value_loss_unclipped = (values - valid_returns) ** 2
                value_loss_clipped = (values_clipped - valid_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = F.mse_loss(values, valid_returns)

            # Entropy loss (sum across heads, each normalized)
            entropy_loss = 0.0
            for key, ent in entropy.items():
                entropy_loss = entropy_loss - ent.mean()

            entropy_coef = self.get_entropy_coef()

            loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Track metrics
            joint_ratio = per_head_ratios["op"]  # Use op ratio as representative
            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["ratio_mean"].append(joint_ratio.mean().item())
            metrics["ratio_max"].append(joint_ratio.max().item())

        self.train_steps += 1

        if clear_buffer:
            self.tamiyo_buffer.reset()

        # Aggregate
        result = {}
        for k, v in metrics.items():
            result[k] = sum(v) / len(v) if v else 0.0

        return result
```

**Step 5: Run tests**

Run: `pytest tests/simic/test_ppo_agent.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_agent.py
git commit -m "$(cat <<'EOF'
feat(simic): add tamiyo mode to PPOAgent

New unified mode combining:
- FactoredRecurrentActorCritic (LSTM + factored heads)
- TamiyoRolloutBuffer (per-env storage + per-head log probs)
- Per-head advantages with causal masking

Usage: PPOAgent(state_dim=50, tamiyo=True, num_envs=4)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: vectorized.py Integration

**Problem:** `vectorized.py` currently uses `store_factored_transition()` in an interleaved loop. Need to use new `tamiyo_buffer.add()` with per-env storage.

**Files:**
- Modify: `src/esper/simic/vectorized.py`

**Key locations in vectorized.py:**
- Line 662-665: Episode initialization (existing recurrent handling)
- Line 1201-1213: Transition storage (currently `store_factored_transition()`)
- Line 1241-1247: PPO update (currently `update_factored()`)

---

**Step 1: Update episode initialization (lines 662-665)**

Find the `if recurrent:` block and add tamiyo support:

```python
# OLD (lines 662-665):
        if recurrent:
            for env_idx in range(envs_this_batch):
                agent.recurrent_buffer.start_episode(env_id=env_idx)
                env_states[env_idx].lstm_hidden = None  # Fresh hidden for new episode

# NEW:
        if agent.tamiyo:
            for env_idx in range(envs_this_batch):
                agent.tamiyo_buffer.start_episode(env_id=env_idx)
                env_states[env_idx].lstm_hidden = None  # Fresh hidden for new episode
        elif recurrent:
            for env_idx in range(envs_this_batch):
                agent.recurrent_buffer.start_episode(env_id=env_idx)
                env_states[env_idx].lstm_hidden = None
```

---

**Step 2: Update transition storage (lines 1201-1213)**

Replace the `store_factored_transition()` call with tamiyo support:

```python
# OLD (lines 1201-1213):
                # Store factored transition with per-head masks
                env_masks = {key: masks_batch[key][env_idx] for key in masks_batch}
                agent.store_factored_transition(
                    state=states_batch_normalized[env_idx],
                    action=action_dict,
                    log_prob=log_prob,
                    value=value,
                    reward=normalized_reward,
                    done=done,
                    action_masks=env_masks,
                    truncated=truncated,
                    bootstrap_value=bootstrap_value,
                )

# NEW:
                # Store transition with per-head masks
                env_masks = {key: masks_batch[key][env_idx] for key in masks_batch}

                if agent.tamiyo:
                    # Tamiyo mode: per-env storage with LSTM states
                    # Note: log_probs were captured earlier when get_action was called
                    agent.tamiyo_buffer.add(
                        env_id=env_idx,
                        state=states_batch_normalized[env_idx],
                        slot_action=action_dict["slot"],
                        blueprint_action=action_dict["blueprint"],
                        blend_action=action_dict["blend"],
                        op_action=action_dict["op"],
                        slot_log_prob=head_log_probs["slot"][env_idx].item(),
                        blueprint_log_prob=head_log_probs["blueprint"][env_idx].item(),
                        blend_log_prob=head_log_probs["blend"][env_idx].item(),
                        op_log_prob=head_log_probs["op"][env_idx].item(),
                        value=value,
                        reward=normalized_reward,
                        done=done,
                        slot_mask=env_masks["slot"],
                        blueprint_mask=env_masks["blueprint"],
                        blend_mask=env_masks["blend"],
                        op_mask=env_masks["op"],
                        hidden_h=env_states[env_idx].lstm_hidden[0] if env_states[env_idx].lstm_hidden else agent.network.get_initial_hidden(1, agent.device)[0],
                        hidden_c=env_states[env_idx].lstm_hidden[1] if env_states[env_idx].lstm_hidden else agent.network.get_initial_hidden(1, agent.device)[1],
                        truncated=truncated,
                        bootstrap_value=bootstrap_value,
                    )

                    # Episode boundary: end episode on done
                    if done:
                        agent.tamiyo_buffer.end_episode(env_id=env_idx)
                else:
                    # Factored mode (legacy - to be deleted in Task 8)
                    agent.store_factored_transition(
                        state=states_batch_normalized[env_idx],
                        action=action_dict,
                        log_prob=log_prob,
                        value=value,
                        reward=normalized_reward,
                        done=done,
                        action_masks=env_masks,
                        truncated=truncated,
                        bootstrap_value=bootstrap_value,
                    )
```

---

**Step 3: Capture per-head log probs during action selection**

Find where `get_action` is called for factored mode (around lines 1100-1130) and capture per-head log probs for tamiyo mode:

```python
# Add after the existing action selection logic:
                if agent.tamiyo:
                    # Tamiyo mode: get per-head log probs and update hidden state
                    actions, head_log_probs, values_batch, new_hidden = agent.network.get_action(
                        states_batch_normalized,
                        hidden=env_states[env_idx].lstm_hidden,
                        slot_mask=masks_batch["slot"],
                        blueprint_mask=masks_batch["blueprint"],
                        blend_mask=masks_batch["blend"],
                        op_mask=masks_batch["op"],
                    )
                    env_states[env_idx].lstm_hidden = new_hidden
```

---

**Step 4: Update PPO update call (lines 1241-1247)**

Replace the update call:

```python
# OLD (lines 1241-1247):
        if batch_rollback_occurred:
            agent.factored_buffer.clear()
            print("[PPO] Buffer cleared due to Governor rollback - skipping update")
        else:
            # Factored mode: use update_factored with factored buffer
            update_metrics = agent.update_factored(last_value=0.0)
            metrics = update_metrics

# NEW:
        if batch_rollback_occurred:
            if agent.tamiyo:
                agent.tamiyo_buffer.reset()
            else:
                agent.factored_buffer.clear()
            print("[PPO] Buffer cleared due to Governor rollback - skipping update")
        else:
            if agent.tamiyo:
                update_metrics = agent.update_tamiyo(clear_buffer=True)
            else:
                update_metrics = agent.update_factored(last_value=0.0)
            metrics = update_metrics
```

---

**Step 5: Run integration test**

```bash
pytest tests/simic/ -v
```

**Step 6: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "$(cat <<'EOF'
feat(simic): integrate tamiyo mode into vectorized_train

Key changes:
- Episode boundary tracking with start_episode/end_episode (line 662)
- Per-head log prob capture from network.get_action (line 1100)
- Per-env transition storage with LSTM hidden states (line 1201)
- update_tamiyo() call for PPO updates (line 1241)
- LSTM hidden reset on episode boundaries

This completes the P0 GAE interleaving fix.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: TrainingConfig Updates

**Problem:** Config needs new hyperparameters recommended by DRL expert for long-horizon credit assignment.

**Changes:**
- gamma: 0.99 → 0.995 (better long-horizon credit)
- gae_lambda: 0.95 → 0.97
- Add tamiyo mode flag
- Add entropy schedule for exploration decay

**Files:**
- Modify: `src/esper/simic/config.py`
- Modify: `tests/simic/test_config.py`

**Step 1: Write failing test**

Add to `tests/simic/test_config.py`:

```python
def test_tamiyo_preset():
    """for_tamiyo preset should have long-horizon hyperparameters."""
    config = TrainingConfig.for_tamiyo()

    # Long-horizon credit assignment
    assert config.gamma == 0.995
    assert config.gae_lambda == 0.97

    # Tamiyo mode enabled
    assert config.tamiyo is True
    assert config.factored is False
    assert config.recurrent is False

    # Entropy schedule
    assert config.entropy_coef_start == 0.05
    assert config.entropy_coef_end == 0.005
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/simic/test_config.py::test_tamiyo_preset -v`
Expected: FAIL

**Step 3: Add tamiyo field and preset**

In `src/esper/simic/config.py`, add fields after `factored`:

```python
    tamiyo: bool = False  # Unified factored + recurrent mode
    num_envs: int = 4  # Parallel environments for TamiyoRolloutBuffer
    recurrent_n_epochs: int = 1  # Single PPO epoch for LSTM stability (DRL expert)
```

Add preset method:

```python
    @staticmethod
    def for_tamiyo() -> "TrainingConfig":
        """Configuration for Tamiyo seed lifecycle controller.

        Optimized for:
        - 10-20 epoch seed learning cycles
        - 25 epoch episodes
        - Long-horizon credit assignment

        Hyperparameters (DRL expert recommendations):
        - gamma=0.995: γ^25 ≈ 0.88 (preserves end-of-episode signal)
        - gae_lambda=0.97: Less bias for long delays
        - Entropy schedule: 0.05 → 0.005 over 10k steps (matches training duration)
        """
        return TrainingConfig(
            tamiyo=True,
            factored=False,
            recurrent=False,
            gamma=0.995,
            gae_lambda=0.97,
            entropy_coef=0.01,  # Base (used when schedule disabled)
            entropy_coef_start=0.05,
            entropy_coef_end=0.005,
            # entropy_anneal_steps: 4 envs × 100 episodes × 25 epochs = 10000
            entropy_anneal_steps=10000,
            n_epochs=10,  # Feedforward default (recurrent uses recurrent_n_epochs)
            recurrent_n_epochs=1,  # Single epoch for LSTM stability
            lstm_hidden_dim=128,
            chunk_length=25,  # Full episode
            max_epochs=25,
            num_envs=4,
        )
```

**Step 4: Run tests**

Run: `pytest tests/simic/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/esper/simic/config.py tests/simic/test_config.py
git commit -m "$(cat <<'EOF'
feat(simic): add for_tamiyo() config preset

Hyperparameters optimized for Tamiyo seed lifecycle control:
- gamma=0.995 (γ^25 ≈ 0.88 for long-horizon credit)
- gae_lambda=0.97 (less bias for 10-20 epoch delays)
- Entropy schedule: 0.05 → 0.005 (high exploration early)
- Single PPO epoch for LSTM stability

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Delete FactoredRolloutBuffer and RecurrentRolloutBuffer

**Problem:** FactoredRolloutBuffer and RecurrentRolloutBuffer are superseded by TamiyoRolloutBuffer. Per No Legacy Code Policy, delete completely - no deprecation warnings.

**Note:** `RolloutBuffer` (the basic buffer) is KEPT - still used for non-factored, non-recurrent mode at `ppo.py:265`.

> **IMPORTANT: Tasks 7 and 8 are tightly coupled.** After completing Task 7, `ppo.py` will have broken imports referencing deleted classes. **Task 8 MUST be completed immediately after Task 7** - do not commit Task 7 in isolation or the codebase will be broken.

**Files:**
- Modify: `src/esper/simic/buffers.py`
- Delete: `tests/simic/test_buffers_factored.py`

**Step 1: Write test to verify TamiyoRolloutBuffer is the only factored buffer**

```python
def test_no_legacy_factored_buffer():
    """FactoredRolloutBuffer should not exist - replaced by TamiyoRolloutBuffer."""
    from esper.simic import buffers
    assert not hasattr(buffers, 'FactoredRolloutBuffer')
    assert not hasattr(buffers, 'RecurrentRolloutBuffer')
```

**Step 2: Delete FactoredRolloutBuffer from buffers.py**

In `src/esper/simic/buffers.py`, delete:
- `class FactoredRolloutBuffer:` (lines 508-676) - entire class
- `FactoredTransition` named tuple (lines 484-506) - if only used by FactoredRolloutBuffer
- `"FactoredRolloutBuffer"` from `__all__` export

**Step 3: Delete RecurrentRolloutBuffer from buffers.py**

In `src/esper/simic/buffers.py`, delete:
- `class RecurrentRolloutBuffer:` (lines 178-481) - entire class
- `RecurrentTransition` named tuple (lines 156-176) - if only used by RecurrentRolloutBuffer
- `"RecurrentRolloutBuffer"` from `__all__` export

**Step 4: Delete test_buffers_factored.py**

```bash
rm tests/simic/test_buffers_factored.py
```

**Step 5: Update RecurrentRolloutBuffer tests in test_simic_buffers.py**

Delete all `TestRecurrentRolloutBuffer` tests from `tests/test_simic_buffers.py` (lines 107-320).

**Step 6: DO NOT commit yet - proceed immediately to Task 8**

> Tests will fail at this point due to broken imports in `ppo.py`. This is expected. Complete Task 8 before running tests or committing.

---

## Task 8: Remove Buffer References from ppo.py (COMPLETE WITH TASK 7)

**Problem:** After deleting FactoredRolloutBuffer and RecurrentRolloutBuffer, ppo.py imports and conditionals still reference them.

**Files:**
- Modify: `src/esper/simic/ppo.py`

**Step 1: Update imports**

Replace line 19:
```python
# OLD:
from esper.simic.buffers import RolloutBuffer, RecurrentRolloutBuffer, FactoredRolloutBuffer

# NEW:
from esper.simic.buffers import RolloutBuffer
from esper.simic.tamiyo_buffer import TamiyoRolloutBuffer
```

**Step 2: Remove factored/recurrent buffer initialization**

In `PPOAgent.__init__`, after the tamiyo initialization block (added in Task 4), remove:

```python
# DELETE these elif branches:
        elif factored:
            self.network = FactoredActorCritic(state_dim=state_dim).to(device)
            self.factored_buffer = FactoredRolloutBuffer()
        elif recurrent:
            self.network = RecurrentActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_dim=lstm_hidden_dim,
            ).to(device)
            self.recurrent_buffer = RecurrentRolloutBuffer(
                chunk_length=chunk_length,
                lstm_hidden_dim=lstm_hidden_dim,
            )
```

**Step 3: Remove factored/recurrent methods**

Delete these methods from PPOAgent:
- `store_factored_transition()` - replaced by tamiyo_buffer.add()
- `update_factored()` - replaced by update_tamiyo()
- `update_recurrent()` - replaced by update_tamiyo()

**Step 3.5: Audit tests for deleted method calls**

Search for any tests calling the deleted methods:
```bash
grep -rn "store_factored_transition\|update_factored\|update_recurrent" tests/
```

If any tests are found, either delete them (if they only test the deleted methods) or update them to use the new tamiyo API.

**Step 4: Update __init__.py exports**

In `src/esper/simic/__init__.py`, remove:
- `RecurrentRolloutBuffer` from imports (line 30)
- `"RecurrentRolloutBuffer"` from `__all__` (line 124)

**Step 5: Run tests**

```bash
pytest tests/simic/ -v
```

**Step 6: Commit (Tasks 7 + 8 combined)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(simic): delete FactoredRolloutBuffer and RecurrentRolloutBuffer

Both buffer classes are superseded by TamiyoRolloutBuffer which provides:
- Per-environment storage (fixes P0 GAE interleaving bug)
- Factored actions + LSTM states (unified architecture)

Changes:
- Delete FactoredRolloutBuffer, RecurrentRolloutBuffer from buffers.py
- Delete FactoredTransition, RecurrentTransition named tuples
- Remove imports and references from ppo.py
- Delete store_factored_transition(), update_factored(), update_recurrent()
- Delete test_buffers_factored.py, remove RecurrentRolloutBuffer tests
- Update __init__.py exports

RolloutBuffer (basic) is retained for non-factored, non-recurrent mode.
Only tamiyo mode and basic RolloutBuffer mode remain.

Per No Legacy Code Policy: delete completely, no deprecation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/simic/ -v`
Expected: All PASS

**Step 2: Run type checker**

Run: `mypy src/esper/simic/tamiyo_buffer.py src/esper/simic/tamiyo_network.py src/esper/simic/advantages.py --ignore-missing-imports`
Expected: No errors

**Step 3: Verify no regressions**

Run: `pytest tests/ -x --tb=short`
Expected: All PASS

---

## Summary

| Task | What | Type |
|------|------|------|
| 1 | TamiyoRolloutBuffer (per-env storage, LSTM states, factored actions) | P0 Fix |
| 2 | FactoredRecurrentActorCritic (LSTM + factored heads) | Architecture |
| 3 | Per-head advantages with causal masking | Credit Assignment |
| 4 | PPOAgent integration (tamiyo mode) | Integration |
| 5 | vectorized.py integration | Integration |
| 6 | TrainingConfig.for_tamiyo() preset | Config |
| 7 | Delete FactoredRolloutBuffer + RecurrentRolloutBuffer | Cleanup |
| 8 | Remove ppo.py references to deleted buffers | Cleanup |

**Note:** `RolloutBuffer` (basic) is KEPT for non-factored, non-recurrent mode.

**Key fixes:**
- P0 GAE interleaving bug → per-env storage
- Missing LSTM → FactoredRecurrentActorCritic
- Noisy gradients → per-head causal masking
- Suboptimal hyperparameters → gamma=0.995, entropy schedule
- Legacy code accumulation → complete deletion per No Legacy Code Policy
