# Simic PPO/IQL Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the large ppo.py (~1600 LOC) and iql.py (~1300 LOC) files into smaller, focused modules following the existing flat structure pattern in simic/.

**Architecture:** Extract shared components (buffers, normalization, networks) into reusable modules. Keep algorithm-specific agent classes slim. Move training loops and CLI to dedicated modules. Delete duplicate code (feature extraction already exists in features.py).

**Tech Stack:** Python 3.11+, PyTorch, existing esper infrastructure

---

## Task 1: Create buffers.py

**Files:**
- Create: `src/esper/simic/buffers.py`
- Test: `tests/test_simic_buffers.py`

**Step 1: Write the failing test**

Create `tests/test_simic_buffers.py`:

```python
"""Tests for simic buffer data structures."""

import pytest
import torch

from esper.simic.buffers import (
    RolloutStep,
    RolloutBuffer,
    Transition,
    ReplayBuffer,
)


class TestRolloutStep:
    """Tests for RolloutStep NamedTuple."""

    def test_creation(self):
        """Test RolloutStep can be created with expected fields."""
        step = RolloutStep(
            state=torch.zeros(27),
            action=0,
            log_prob=-0.5,
            value=1.0,
            reward=0.1,
            done=False,
        )
        assert step.action == 0
        assert step.done is False


class TestRolloutBuffer:
    """Tests for RolloutBuffer (PPO trajectory storage)."""

    def test_add_and_len(self):
        """Test adding steps and checking length."""
        buffer = RolloutBuffer()
        assert len(buffer) == 0

        buffer.add(
            state=torch.zeros(27),
            action=0,
            log_prob=-0.5,
            value=1.0,
            reward=0.1,
            done=False,
        )
        assert len(buffer) == 1

    def test_clear(self):
        """Test clearing the buffer."""
        buffer = RolloutBuffer()
        buffer.add(torch.zeros(27), 0, -0.5, 1.0, 0.1, False)
        buffer.add(torch.zeros(27), 1, -0.3, 0.9, 0.2, False)
        assert len(buffer) == 2

        buffer.clear()
        assert len(buffer) == 0

    def test_compute_returns_and_advantages(self):
        """Test GAE computation produces correct shapes."""
        buffer = RolloutBuffer()
        for i in range(5):
            buffer.add(
                state=torch.zeros(27),
                action=i % 4,
                log_prob=-0.5,
                value=1.0 - i * 0.1,
                reward=0.1,
                done=(i == 4),
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value=0.0, gamma=0.99, gae_lambda=0.95
        )

        assert returns.shape == (5,)
        assert advantages.shape == (5,)

    def test_get_batches(self):
        """Test minibatch generation."""
        buffer = RolloutBuffer()
        for i in range(10):
            buffer.add(torch.randn(27), i % 4, -0.5, 1.0, 0.1, False)

        batches = buffer.get_batches(batch_size=4, device="cpu")

        assert len(batches) >= 2  # 10 steps / 4 batch_size = 2-3 batches
        batch, batch_idx = batches[0]
        assert "states" in batch
        assert "actions" in batch
        assert "old_log_probs" in batch


class TestTransition:
    """Tests for Transition dataclass (IQL)."""

    def test_creation(self):
        """Test Transition can be created."""
        t = Transition(
            state=[0.0] * 27,
            action=1,
            reward=0.5,
            next_state=[0.0] * 27,
            done=False,
        )
        assert t.action == 1
        assert t.reward == 0.5


class TestReplayBuffer:
    """Tests for ReplayBuffer (IQL offline data)."""

    def test_creation_and_properties(self):
        """Test buffer creation from transitions."""
        transitions = [
            Transition([0.0] * 27, 0, 0.1, [0.0] * 27, False),
            Transition([1.0] * 27, 1, 0.2, [1.0] * 27, False),
            Transition([2.0] * 27, 2, 0.3, [2.0] * 27, True),
        ]

        buffer = ReplayBuffer(transitions, device="cpu")

        assert buffer.size == 3
        assert buffer.state_dim == 27

    def test_sample(self):
        """Test sampling from buffer."""
        transitions = [
            Transition([float(i)] * 27, i % 4, 0.1, [float(i)] * 27, False)
            for i in range(100)
        ]

        buffer = ReplayBuffer(transitions, device="cpu")
        states, actions, rewards, next_states, dones = buffer.sample(16)

        assert states.shape == (16, 27)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 27)
        assert dones.shape == (16,)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_buffers.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'esper.simic.buffers'"

**Step 3: Write minimal implementation**

Create `src/esper/simic/buffers.py`:

```python
"""Buffer data structures for RL training.

This module contains trajectory and replay buffers used by PPO and IQL:
- RolloutBuffer: On-policy trajectory storage for PPO with GAE computation
- ReplayBuffer: Off-policy experience storage for IQL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch


# =============================================================================
# PPO Buffers
# =============================================================================

class RolloutStep(NamedTuple):
    """Single step in a PPO rollout."""
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class RolloutBuffer:
    """Buffer for storing PPO rollout data.

    Stores trajectory steps and computes returns/advantages using GAE.
    """
    steps: list[RolloutStep] = field(default_factory=list)

    def add(self, state: torch.Tensor, action: int, log_prob: float,
            value: float, reward: float, done: bool) -> None:
        """Add a step to the buffer."""
        self.steps.append(RolloutStep(state, action, log_prob, value, reward, done))

    def clear(self) -> None:
        """Clear all steps from the buffer."""
        self.steps = []

    def __len__(self) -> int:
        return len(self.steps)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and GAE advantages.

        Args:
            last_value: Value estimate for state after last step (0 if done)
            gamma: Discount factor
            gae_lambda: GAE lambda for bias-variance tradeoff

        Returns:
            Tuple of (returns, advantages)
        """
        n_steps = len(self.steps)
        returns = torch.zeros(n_steps)
        advantages = torch.zeros(n_steps)

        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n_steps)):
            step = self.steps[t]

            if step.done:
                next_value = 0.0
                last_gae = 0.0

            delta = step.reward + gamma * next_value - step.value
            advantages[t] = last_gae = delta + gamma * gae_lambda * last_gae
            returns[t] = advantages[t] + step.value
            next_value = step.value

        return returns, advantages

    def get_batches(self, batch_size: int, device: str) -> list[tuple[dict, torch.Tensor]]:
        """Get shuffled minibatches for PPO update.

        Returns list of (batch_dict, batch_indices) tuples.
        """
        n_steps = len(self.steps)
        indices = torch.randperm(n_steps)

        batches = []
        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            batch_idx = indices[start:end]

            batch = {
                'states': torch.stack([self.steps[i].state for i in batch_idx]).to(device),
                'actions': torch.tensor([self.steps[i].action for i in batch_idx],
                                        dtype=torch.long, device=device),
                'old_log_probs': torch.tensor([self.steps[i].log_prob for i in batch_idx],
                                               dtype=torch.float32, device=device),
            }
            batches.append((batch, batch_idx))

        return batches


# =============================================================================
# IQL Buffers
# =============================================================================

@dataclass
class Transition:
    """A single (s, a, r, s', done) transition for offline RL."""
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Replay buffer for offline RL (IQL/CQL).

    Pre-converts all data to tensors for efficient sampling.
    """

    def __init__(self, transitions: list[Transition], device: str = "cuda:0"):
        self.device = device
        self.size = len(transitions)

        self.states = torch.tensor(
            [t.state for t in transitions], dtype=torch.float32, device=device
        )
        self.actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=device
        )
        self.rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=device
        )
        self.next_states = torch.tensor(
            [t.next_state for t in transitions], dtype=torch.float32, device=device
        )
        self.dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=device
        )

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    @property
    def state_dim(self) -> int:
        """Dimension of state vectors."""
        return self.states.shape[1]


__all__ = [
    "RolloutStep",
    "RolloutBuffer",
    "Transition",
    "ReplayBuffer",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_buffers.py -v`
Expected: PASS (all tests green)

**Step 5: Commit**

```bash
git add src/esper/simic/buffers.py tests/test_simic_buffers.py
git commit -m "feat(simic): extract buffer classes to buffers.py"
```

---

## Task 2: Create normalization.py

**Files:**
- Create: `src/esper/simic/normalization.py`
- Test: `tests/test_simic_normalization.py`

**Step 1: Write the failing test**

Create `tests/test_simic_normalization.py`:

```python
"""Tests for observation normalization."""

import pytest
import torch

from esper.simic.normalization import RunningMeanStd


class TestRunningMeanStd:
    """Tests for RunningMeanStd normalizer."""

    def test_initial_state(self):
        """Test initial mean is zero and var is one."""
        rms = RunningMeanStd(shape=(4,))
        assert torch.allclose(rms.mean, torch.zeros(4))
        assert torch.allclose(rms.var, torch.ones(4))

    def test_update_changes_stats(self):
        """Test that update modifies running statistics."""
        rms = RunningMeanStd(shape=(2,))

        # Update with batch of known values
        batch = torch.tensor([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]])
        rms.update(batch)

        # Mean should move toward batch mean (4.0, 8.0)
        assert rms.mean[0] > 0
        assert rms.mean[1] > 0

    def test_normalize_output_range(self):
        """Test that normalize clips to expected range."""
        rms = RunningMeanStd(shape=(3,))

        # Update with some data
        rms.update(torch.randn(100, 3))

        # Normalize should clip to [-10, 10] by default
        extreme = torch.tensor([[1000.0, -1000.0, 0.0]])
        normalized = rms.normalize(extreme, clip=10.0)

        assert normalized.max() <= 10.0
        assert normalized.min() >= -10.0

    def test_to_device(self):
        """Test moving stats to device."""
        rms = RunningMeanStd(shape=(5,))
        rms = rms.to("cpu")  # Should work even if already on CPU

        assert rms.mean.device.type == "cpu"
        assert rms.var.device.type == "cpu"

    def test_welford_stability(self):
        """Test numerical stability with Welford's algorithm."""
        rms = RunningMeanStd(shape=(1,))

        # Multiple small updates shouldn't cause numerical issues
        for _ in range(100):
            rms.update(torch.randn(10, 1) * 0.01 + 100.0)

        # Mean should be close to 100
        assert 99.0 < rms.mean[0] < 101.0
        # Var should be small (0.01^2 scale)
        assert rms.var[0] < 1.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_normalization.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'esper.simic.normalization'"

**Step 3: Write minimal implementation**

Create `src/esper/simic/normalization.py`:

```python
"""Observation normalization for RL training.

Provides running mean/std normalization using Welford's numerically stable
online algorithm. Used by vectorized PPO for observation preprocessing.
"""

from __future__ import annotations

import torch


class RunningMeanStd:
    """Running mean and std for observation normalization.

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = epsilon  # Prevent div by zero
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        """Update running stats with new batch of observations."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor,
                             batch_count: int) -> None:
        """Update using batch moments (Welford's algorithm)."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        """Normalize observation using running stats."""
        return torch.clamp(
            (x - self.mean.to(x.device)) / torch.sqrt(self.var.to(x.device) + self.epsilon),
            -clip, clip
        )

    def to(self, device: str) -> "RunningMeanStd":
        """Move stats to device."""
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self


__all__ = ["RunningMeanStd"]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_normalization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/normalization.py tests/test_simic_normalization.py
git commit -m "feat(simic): extract RunningMeanStd to normalization.py"
```

---

## Task 3: Add RL networks to networks.py

**Files:**
- Modify: `src/esper/simic/networks.py`
- Test: `tests/test_simic_networks.py`

**Step 1: Write the failing test**

Create `tests/test_simic_networks.py`:

```python
"""Tests for RL network architectures."""

import pytest
import torch

from esper.simic.networks import ActorCritic, QNetwork, VNetwork


class TestActorCritic:
    """Tests for PPO ActorCritic network."""

    def test_forward_shapes(self):
        """Test forward pass returns correct shapes."""
        net = ActorCritic(state_dim=27, action_dim=7, hidden_dim=64)
        state = torch.randn(4, 27)  # batch of 4

        dist, value = net(state)

        assert value.shape == (4,)
        assert dist.probs.shape == (4, 7)

    def test_get_action(self):
        """Test single action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)

        action, log_prob, value = net.get_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 7
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        net = ActorCritic(state_dim=27, action_dim=7)
        state = torch.randn(1, 27)

        # Deterministic should give same action each time
        actions = [net.get_action(state, deterministic=True)[0] for _ in range(5)]
        assert len(set(actions)) == 1

    def test_get_action_batch(self):
        """Test batched action sampling."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(8, 27)

        actions, log_probs, values = net.get_action_batch(states)

        assert actions.shape == (8,)
        assert log_probs.shape == (8,)
        assert values.shape == (8,)

    def test_evaluate_actions(self):
        """Test action evaluation for PPO update."""
        net = ActorCritic(state_dim=27, action_dim=7)
        states = torch.randn(16, 27)
        actions = torch.randint(0, 7, (16,))

        log_probs, values, entropy = net.evaluate_actions(states, actions)

        assert log_probs.shape == (16,)
        assert values.shape == (16,)
        assert entropy.shape == (16,)


class TestQNetwork:
    """Tests for IQL Q-network."""

    def test_forward_shape(self):
        """Test Q-network outputs Q-values for all actions."""
        net = QNetwork(state_dim=27, action_dim=7)
        state = torch.randn(4, 27)

        q_values = net(state)

        assert q_values.shape == (4, 7)


class TestVNetwork:
    """Tests for IQL V-network."""

    def test_forward_shape(self):
        """Test V-network outputs scalar value."""
        net = VNetwork(state_dim=27)
        state = torch.randn(4, 27)

        values = net(state)

        assert values.shape == (4, 1)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_networks.py -v`
Expected: FAIL with "cannot import name 'ActorCritic' from 'esper.simic.networks'"

**Step 3: Add networks to existing file**

Add to `src/esper/simic/networks.py` (append before `__all__`):

```python
# =============================================================================
# RL Network Architectures
# =============================================================================

import math
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Uses shared feature extraction with separate actor and critic heads.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.shared(state)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        value = self.critic(features).squeeze(-1)
        return dist, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> tuple[int, float, float]:
        """Sample action from policy."""
        with torch.no_grad():
            dist, value = self.forward(state)
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()

    def get_action_batch(self, states: torch.Tensor, deterministic: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for a batch of states."""
        with torch.no_grad():
            dist, values = self.forward(states)
            if deterministic:
                actions = dist.probs.argmax(dim=-1)
            else:
                actions = dist.sample()
            log_probs = dist.log_prob(actions)
            return actions, log_probs, values

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy


class QNetwork(nn.Module):
    """Q-network for IQL: Q(s, a) for all actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions: shape (batch, action_dim)."""
        return self.net(state)


class VNetwork(nn.Module):
    """V-network for IQL: V(s) state value."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns state value: shape (batch, 1)."""
        return self.net(state)
```

Update `__all__` in networks.py:

```python
__all__ = [
    "PolicyNetwork",
    "print_confusion_matrix",
    "ActorCritic",
    "QNetwork",
    "VNetwork",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_networks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/networks.py tests/test_simic_networks.py
git commit -m "feat(simic): add ActorCritic, QNetwork, VNetwork to networks.py"
```

---

## Task 4: Add compute_seed_potential to rewards.py

**Files:**
- Modify: `src/esper/simic/rewards.py`
- Test: Update `tests/test_simic.py` or create `tests/test_simic_rewards.py`

**Step 1: Write the failing test**

Add to tests (create `tests/test_simic_rewards.py` if needed):

```python
"""Tests for reward shaping functions."""

import pytest

from esper.simic.rewards import compute_seed_potential


class TestComputeSeedPotential:
    """Tests for potential-based reward shaping."""

    def test_no_seed_returns_zero(self):
        """Test that no active seed has zero potential."""
        obs = {'has_active_seed': 0, 'seed_stage': 0, 'seed_epochs_in_stage': 0}
        assert compute_seed_potential(obs) == 0.0

    def test_germinated_has_low_potential(self):
        """Test GERMINATED stage has low potential."""
        obs = {'has_active_seed': 1, 'seed_stage': 1, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 5.0

    def test_training_has_higher_potential(self):
        """Test TRAINING stage has higher potential than GERMINATED."""
        germ = {'has_active_seed': 1, 'seed_stage': 1, 'seed_epochs_in_stage': 0}
        train = {'has_active_seed': 1, 'seed_stage': 2, 'seed_epochs_in_stage': 0}

        assert compute_seed_potential(train) > compute_seed_potential(germ)

    def test_blending_has_highest_potential(self):
        """Test BLENDING stage has highest potential."""
        obs = {'has_active_seed': 1, 'seed_stage': 3, 'seed_epochs_in_stage': 0}
        potential = compute_seed_potential(obs)
        assert potential == 25.0

    def test_progress_bonus_capped(self):
        """Test that progress bonus is capped."""
        obs = {'has_active_seed': 1, 'seed_stage': 2, 'seed_epochs_in_stage': 100}
        potential = compute_seed_potential(obs)
        # Base 15.0 + max 3.0 progress = 18.0
        assert potential == 18.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_rewards.py::TestComputeSeedPotential -v`
Expected: FAIL with "cannot import name 'compute_seed_potential'"

**Step 3: Add function to rewards.py**

Add to `src/esper/simic/rewards.py`:

```python
def compute_seed_potential(obs: dict) -> float:
    """Compute potential value Phi(s) based on seed state.

    The potential captures the expected future value of having an active seed
    in various stages. This helps bridge the temporal gap where GERMINATE
    has negative immediate reward but high future value.

    Potential-based reward shaping: r' = r + gamma*Phi(s') - Phi(s)
    This preserves optimal policy (PBRS guarantee) while improving learning.

    Args:
        obs: Observation dictionary with has_active_seed, seed_stage, seed_epochs_in_stage

    Returns:
        Potential value for the current state
    """
    has_active = obs.get('has_active_seed', 0)
    seed_stage = obs.get('seed_stage', 0)
    epochs_in_stage = obs.get('seed_epochs_in_stage', 0)

    if not has_active or seed_stage == 0:
        return 0.0

    # Stage-based potential values
    stage_potentials = {
        1: 5.0,   # GERMINATED - just started
        2: 15.0,  # TRAINING - actively learning
        3: 25.0,  # BLENDING - about to integrate
        4: 10.0,  # FOSSILIZED - value mostly realized
    }

    base_potential = stage_potentials.get(seed_stage, 0.0)
    progress_bonus = min(epochs_in_stage * 0.5, 3.0)

    return base_potential + progress_bonus
```

Update `__all__` in rewards.py to include `compute_seed_potential`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python -m pytest tests/test_simic_rewards.py::TestComputeSeedPotential -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/test_simic_rewards.py
git commit -m "feat(simic): add compute_seed_potential to rewards.py"
```

---

## Task 5: Refactor ppo.py to slim agent-only

**Files:**
- Modify: `src/esper/simic/ppo.py`
- Verify: Existing tests still pass

**Step 1: Backup and verify current tests pass**

Run: `PYTHONPATH=src python -m pytest tests/test_simic.py -v`
Expected: PASS (baseline)

**Step 2: Refactor ppo.py**

Replace `src/esper/simic/ppo.py` with slim version:

```python
"""Simic PPO Module - PPO Agent for Seed Lifecycle Control

This module contains the PPOAgent class for online policy gradient training.
For training functions, see simic.training.
For vectorized environments, see simic.vectorized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.simic.buffers import RolloutBuffer
from esper.simic.networks import ActorCritic
from esper.simic.features import safe, telemetry_to_features


# =============================================================================
# Feature Extraction (PPO-specific wrapper)
# =============================================================================

def signals_to_features(signals, model, tracker=None, use_telemetry: bool = True) -> list[float]:
    """Convert training signals to feature vector.

    This is a PPO-specific wrapper that builds features from TrainingSignals
    and model state. For the core feature extraction, see simic.features.

    Args:
        signals: TrainingSignals from tamiyo module
        model: MorphogeneticModel
        tracker: Optional DiagnosticTracker for V2 features
        use_telemetry: Whether to include telemetry features

    Returns:
        Feature vector (27 dims base, +27 if telemetry)
    """
    # Loss history (5 values)
    loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
    while len(loss_hist) < 5:
        loss_hist.insert(0, 0.0)

    # Accuracy history (5 values)
    acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
    while len(acc_hist) < 5:
        acc_hist.insert(0, 0.0)

    # Seed state from signals
    has_active_seed = 1.0 if signals.active_seeds else 0.0
    if signals.active_seeds:
        seed_state = signals.active_seeds[0]
        seed_stage = seed_state.stage.value if seed_state else 0
        seed_epochs = seed_state.epochs_in_stage if seed_state else 0
        seed_alpha = model.seed_slot.alpha if model.seed_slot else 0.0
        if seed_state and seed_state.metrics:
            seed_improvement = seed_state.metrics.current_val_accuracy - seed_state.metrics.accuracy_at_stage_start
        else:
            seed_improvement = 0.0
    else:
        seed_stage = 0
        seed_epochs = 0
        seed_alpha = 0.0
        seed_improvement = 0.0

    available_slots = signals.available_slots

    # Base features (27 dims)
    features = [
        float(signals.epoch),
        float(signals.global_step),
        safe(signals.train_loss, 10.0),
        safe(signals.val_loss, 10.0),
        safe(signals.loss_delta, 0.0),
        signals.train_accuracy,
        signals.val_accuracy,
        safe(signals.accuracy_delta, 0.0),
        float(signals.plateau_epochs),
        signals.best_val_accuracy,
        min(signals.loss_history) if signals.loss_history else 10.0,
        *[safe(v, 10.0) for v in loss_hist],
        *acc_hist,
        has_active_seed,
        float(seed_stage),
        float(seed_epochs),
        seed_alpha,
        seed_improvement,
        float(available_slots),
    ]

    # Telemetry features (27 dims)
    if use_telemetry and tracker:
        telem = tracker.compute_telemetry()
        features.extend(telemetry_to_features(telem))
    elif use_telemetry:
        features.extend([0.0] * 27)

    return features


# =============================================================================
# PPO Agent
# =============================================================================

class PPOAgent:
    """PPO agent for training Tamiyo seed lifecycle controller."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda:0",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.train_steps = 0

    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> tuple[int, float, float]:
        """Get action for current state."""
        return self.network.get_action(state, deterministic)

    def store_transition(self, state: torch.Tensor, action: int, log_prob: float,
                         value: float, reward: float, done: bool) -> None:
        """Store transition in buffer."""
        self.buffer.add(state, action, log_prob, value, reward, done)

    def update(self, last_value: float = 0.0) -> dict:
        """Perform PPO update."""
        if len(self.buffer) == 0:
            return {}

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        metrics = defaultdict(list)

        for _ in range(self.n_epochs):
            batches = self.buffer.get_batches(self.batch_size, self.device)

            for batch, batch_idx in batches:
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                batch_returns = returns[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)

                log_probs, values, entropy = self.network.evaluate_actions(states, actions)

                # PPO-Clip loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(-entropy_loss.item())
                metrics['approx_kl'].append(((ratio - 1) - (ratio.log())).mean().item())
                metrics['clip_fraction'].append(
                    ((ratio - 1).abs() > self.clip_ratio).float().mean().item()
                )

        self.train_steps += 1
        self.buffer.clear()

        return {k: sum(v) / len(v) for k, v in metrics.items()}

    def save(self, path: str | Path, metadata: dict = None) -> None:
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'config': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_ratio': self.clip_ratio,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef,
            }
        }
        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda:0") -> "PPOAgent":
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        state_dim = checkpoint['network_state_dict']['shared.0.weight'].shape[1]
        action_dim = checkpoint['network_state_dict']['actor.2.weight'].shape[0]

        agent = cls(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            **checkpoint.get('config', {})
        )

        agent.network.load_state_dict(checkpoint['network_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.train_steps = checkpoint.get('train_steps', 0)

        return agent


__all__ = [
    "PPOAgent",
    "signals_to_features",
]
```

**Step 3: Run tests to verify nothing broke**

Run: `PYTHONPATH=src python -m pytest tests/test_simic.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/simic/ppo.py
git commit -m "refactor(simic): slim ppo.py to agent-only (~200 LOC)"
```

---

## Task 6: Refactor iql.py to slim agent-only

**Files:**
- Modify: `src/esper/simic/iql.py`

**Step 1: Verify current tests pass**

Run: `PYTHONPATH=src python -m pytest tests/test_simic.py -v`
Expected: PASS

**Step 2: Refactor iql.py**

Replace `src/esper/simic/iql.py` with slim version:

```python
"""Simic IQL Module - Implicit Q-Learning for Offline RL

This module contains the IQL class for offline reinforcement learning.
For training functions, see simic.training.
For comparison modes, see simic.comparison.

References:
    - "Offline Reinforcement Learning with Implicit Q-Learning" (Kostrikov et al., 2021)
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from esper.simic.buffers import ReplayBuffer
from esper.simic.networks import QNetwork, VNetwork


# =============================================================================
# IQL Loss Functions
# =============================================================================

def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """Asymmetric loss that penalizes overestimation.

    When tau > 0.5, positive errors (overestimation) are weighted more heavily.
    This makes the V-network conservative/pessimistic.

    Args:
        diff: Q - V differences
        tau: Expectile parameter (0.5 = symmetric, >0.5 = penalize overestimation)

    Returns:
        Weighted squared loss
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return weight * (diff ** 2)


# =============================================================================
# IQL Agent
# =============================================================================

class IQL:
    """Implicit Q-Learning for offline RL.

    With optional CQL regularization to penalize OOD actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.7,
        beta: float = 3.0,
        lr: float = 3e-4,
        cql_alpha: float = 0.0,
        device: str = "cuda:0",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.action_dim = action_dim
        self.cql_alpha = cql_alpha

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.v_network = VNetwork(state_dim, hidden_dim).to(device)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_network.parameters(), lr=lr)

        self.target_update_rate = 0.005

    def update_v(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update V-network using expectile regression."""
        with torch.no_grad():
            q_values = self.q_target(states)
            q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        v_pred = self.v_network(states).squeeze(1)
        diff = q_a - v_pred
        v_loss = expectile_loss(diff, self.tau).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return v_loss.item()

    def update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[float, float]:
        """Update Q-network using V as target."""
        with torch.no_grad():
            v_next = self.v_network(next_states).squeeze(1)
            td_target = rewards + self.gamma * v_next * (1 - dones)

        q_values = self.q_network(states)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.mse_loss(q_a, td_target)

        cql_loss = 0.0
        if self.cql_alpha > 0:
            logsumexp_q = torch.logsumexp(q_values, dim=1)
            cql_loss = (logsumexp_q - q_a).mean()

        q_loss = td_loss + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return td_loss.item(), cql_loss if isinstance(cql_loss, float) else cql_loss.item()

    def update_target(self) -> None:
        """Soft update of target Q-network."""
        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.target_update_rate * param.data
                + (1 - self.target_update_rate) * target_param.data
            )

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        """Single training step."""
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        v_loss = self.update_v(states, actions)
        td_loss, cql_loss = self.update_q(states, actions, rewards, next_states, dones)
        self.update_target()

        return {"v_loss": v_loss, "q_loss": td_loss, "cql_loss": cql_loss}

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """Get action from learned Q-values."""
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
            if deterministic:
                action = q_values.argmax(dim=1)
            else:
                probs = F.softmax(q_values * self.beta, dim=1)
                action = torch.multinomial(probs, 1).squeeze(1)
        self.q_network.train()
        return action.item() if state.shape[0] == 1 else action

    def get_policy_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from Q-values."""
        with torch.no_grad():
            q_values = self.q_network(states)
            probs = F.softmax(q_values * self.beta, dim=1)
        return probs


__all__ = [
    "IQL",
    "expectile_loss",
]
```

**Step 3: Run tests**

Run: `PYTHONPATH=src python -m pytest tests/test_simic.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/simic/iql.py
git commit -m "refactor(simic): slim iql.py to agent-only (~150 LOC)"
```

---

## Task 7: Create training.py

**Files:**
- Create: `src/esper/simic/training.py`

**Step 1: Create training.py with PPO and IQL training functions**

Create `src/esper/simic/training.py`:

```python
"""Training loops for PPO and IQL.

This module contains the main training functions extracted from ppo.py and iql.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from esper.leyline import SimicAction
from esper.simic.buffers import Transition, ReplayBuffer
from esper.simic.features import obs_to_base_features, telemetry_to_features
from esper.simic.rewards import compute_shaped_reward, SeedInfo


# =============================================================================
# IQL Data Loading
# =============================================================================

def extract_transitions(
    pack: dict,
    use_telemetry: bool = True,
    reward_scale: float = 0.1,
    use_reward_shaping: bool = False,
    gamma: float = 0.99,
) -> list[Transition]:
    """Extract (s, a, r, s', done) transitions from episodes.

    Args:
        pack: Data pack dictionary
        use_telemetry: Whether to include V2 telemetry features
        reward_scale: Scale factor for rewards
        use_reward_shaping: Whether to add potential-based reward shaping
        gamma: Discount factor for reward shaping

    Returns:
        List of Transition objects
    """
    from esper.simic.rewards import compute_seed_potential

    transitions = []

    for ep in pack['episodes']:
        decisions = ep['decisions']
        telem_hist = ep.get('telemetry_history', [])

        for i, decision in enumerate(decisions):
            state = obs_to_base_features(decision['observation'])
            if use_telemetry and telem_hist:
                epoch = decision['observation']['epoch']
                if epoch <= len(telem_hist):
                    state.extend(telemetry_to_features(telem_hist[epoch - 1]))
                else:
                    state.extend([0.0] * 27)
            elif use_telemetry:
                state.extend([0.0] * 27)

            action = SimicAction[decision['action']['action']].value
            raw_reward = decision['outcome'].get('reward', 0) * reward_scale

            is_last = (i == len(decisions) - 1)
            if is_last:
                next_state = state
                done = True
                next_obs = None
            else:
                next_decision = decisions[i + 1]
                next_obs = next_decision['observation']
                next_state = obs_to_base_features(next_obs)
                if use_telemetry and telem_hist:
                    next_epoch = next_obs['epoch']
                    if next_epoch <= len(telem_hist):
                        next_state.extend(telemetry_to_features(telem_hist[next_epoch - 1]))
                    else:
                        next_state.extend([0.0] * 27)
                elif use_telemetry:
                    next_state.extend([0.0] * 27)
                done = False

            if use_reward_shaping:
                current_obs = decision['observation']
                phi_s = compute_seed_potential(current_obs)
                phi_s_prime = 0.0 if done else compute_seed_potential(next_obs)
                shaping_bonus = gamma * phi_s_prime - phi_s
                reward = raw_reward + shaping_bonus * reward_scale
            else:
                reward = raw_reward

            transitions.append(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            ))

    return transitions


# =============================================================================
# IQL Evaluation
# =============================================================================

def evaluate_iql_policy(
    iql,
    buffer: ReplayBuffer,
    n_samples: int = 1000,
) -> dict:
    """Evaluate IQL policy against behavior policy."""
    import torch

    idx = torch.randint(0, buffer.size, (n_samples,), device=buffer.device)
    states = buffer.states[idx]
    behavior_actions = buffer.actions[idx]

    with torch.no_grad():
        q_values = iql.q_network(states)
        iql_actions = q_values.argmax(dim=1)

    agreement = (iql_actions == behavior_actions).float().mean().item()
    behavior_dist = torch.bincount(behavior_actions, minlength=len(SimicAction)).float() / n_samples
    iql_dist = torch.bincount(iql_actions, minlength=len(SimicAction)).float() / n_samples

    q_behavior = q_values.gather(1, behavior_actions.unsqueeze(1)).squeeze(1).mean().item()
    q_iql = q_values.gather(1, iql_actions.unsqueeze(1)).squeeze(1).mean().item()

    return {
        "agreement": agreement,
        "q_behavior": q_behavior,
        "q_iql": q_iql,
        "q_improvement": q_iql - q_behavior,
        "behavior_dist": behavior_dist.tolist(),
        "iql_dist": iql_dist.tolist(),
    }


# =============================================================================
# IQL Training
# =============================================================================

def train_iql(
    pack_path: str,
    epochs: int = 100,
    steps_per_epoch: int = 1000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.7,
    beta: float = 3.0,
    lr: float = 3e-4,
    cql_alpha: float = 0.0,
    use_telemetry: bool = True,
    use_reward_shaping: bool = False,
    device: str = "cuda:0",
    save_path: Optional[str] = None,
):
    """Train IQL on offline data."""
    import copy
    from esper.simic.iql import IQL

    print("=" * 60)
    print("Tamiyo Phase 3: Implicit Q-Learning")
    print("=" * 60)

    print(f"Loading {pack_path}...")
    with open(pack_path) as f:
        pack = json.load(f)
    print(f"Episodes: {pack['metadata']['num_episodes']}")

    print("Extracting transitions...")
    transitions = extract_transitions(
        pack,
        use_telemetry=use_telemetry,
        use_reward_shaping=use_reward_shaping,
        gamma=gamma,
    )
    print(f"Transitions: {len(transitions)}")

    buffer = ReplayBuffer(transitions, device=device)
    print(f"State dim: {buffer.state_dim}")

    rewards = buffer.rewards
    print(f"Reward range: [{rewards.min():.2f}, {rewards.max():.2f}], mean: {rewards.mean():.2f}")

    iql = IQL(
        state_dim=buffer.state_dim,
        action_dim=len(SimicAction),
        gamma=gamma,
        tau=tau,
        beta=beta,
        lr=lr,
        cql_alpha=cql_alpha,
        device=device,
    )

    algo_name = "IQL+CQL" if cql_alpha > 0 else "IQL"
    print(f"\nTraining {algo_name} for {epochs} epochs")

    best_q_improvement = -float('inf')
    best_state_dict = None

    for epoch in range(epochs):
        epoch_v_loss = 0.0
        epoch_q_loss = 0.0
        epoch_cql_loss = 0.0

        for _ in range(steps_per_epoch):
            losses = iql.train_step(buffer, batch_size)
            epoch_v_loss += losses["v_loss"]
            epoch_q_loss += losses["q_loss"]
            epoch_cql_loss += losses["cql_loss"]

        epoch_v_loss /= steps_per_epoch
        epoch_q_loss /= steps_per_epoch
        epoch_cql_loss /= steps_per_epoch

        if epoch % 10 == 0 or epoch == epochs - 1:
            metrics = evaluate_iql_policy(iql, buffer)
            print(f"Epoch {epoch:3d}: v_loss={epoch_v_loss:.4f}, q_loss={epoch_q_loss:.4f} | "
                  f"agreement={metrics['agreement']*100:.1f}%, Q_improve={metrics['q_improvement']:.3f}")

            if metrics['q_improvement'] > best_q_improvement:
                best_q_improvement = metrics['q_improvement']
                best_state_dict = {
                    'q_network': copy.deepcopy(iql.q_network.state_dict()),
                    'v_network': copy.deepcopy(iql.v_network.state_dict()),
                }

    if best_state_dict:
        iql.q_network.load_state_dict(best_state_dict['q_network'])
        iql.v_network.load_state_dict(best_state_dict['v_network'])

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = evaluate_iql_policy(iql, buffer, n_samples=min(5000, buffer.size))
        torch.save({
            'q_network': iql.q_network.state_dict(),
            'v_network': iql.v_network.state_dict(),
            'state_dim': buffer.state_dim,
            'action_dim': len(SimicAction),
            'gamma': gamma,
            'tau': tau,
            'beta': beta,
            'metrics': metrics,
        }, save_path)
        print(f"Model saved to {save_path}")

    return iql


# =============================================================================
# PPO Episode Runner
# =============================================================================

def run_ppo_episode(
    agent,
    trainloader,
    testloader,
    max_epochs: int = 25,
    base_seed: int = 42,
    device: str = "cuda:0",
    use_telemetry: bool = True,
    collect_rollout: bool = True,
    deterministic: bool = False,
) -> tuple[float, dict[str, int], list[float]]:
    """Run a single training episode with the PPO agent."""
    from esper.leyline import SeedStage
    from esper.tolaria import create_model
    from esper.tamiyo import SignalTracker
    from esper.simic.ppo import signals_to_features

    torch.manual_seed(base_seed)
    random.seed(base_seed)

    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    host_optimizer = torch.optim.SGD(
        model.get_host_parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    seed_optimizer = None
    signal_tracker = SignalTracker()

    seeds_created = 0
    action_counts = {a.name: 0 for a in SimicAction}
    episode_rewards = []

    for epoch in range(1, max_epochs + 1):
        seed_state = model.seed_state

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if seed_state is None:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.GERMINATED:
            seed_state.transition(SeedStage.TRAINING)
            seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.TRAINING:
            if seed_optimizer is None:
                seed_optimizer = torch.optim.SGD(model.get_seed_parameters(), lr=0.01, momentum=0.9)
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.BLENDING:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                if seed_optimizer:
                    seed_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                if seed_optimizer:
                    seed_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        elif seed_state.stage == SeedStage.FOSSILIZED:
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                host_optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                host_optimizer.step()
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss /= len(testloader)
        val_acc = 100.0 * correct / total

        # Update signal tracker
        active_seeds = [seed_state] if seed_state else []
        available_slots = 0 if model.has_active_seed else 1
        signals = signal_tracker.update(
            epoch=epoch,
            global_step=epoch * len(trainloader),
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            active_seeds=active_seeds,
            available_slots=available_slots,
        )

        acc_delta = signals.accuracy_delta

        # Get features and action
        features = signals_to_features(signals, model, tracker=None, use_telemetry=False)
        state = torch.tensor([features], dtype=torch.float32, device=device)

        action_idx, log_prob, value = agent.get_action(state, deterministic=deterministic)
        action = SimicAction(action_idx)
        action_counts[action.name] += 1

        reward = compute_shaped_reward(
            action=action.value,
            acc_delta=acc_delta,
            val_acc=val_acc,
            seed_info=SeedInfo.from_seed_state(seed_state),
            epoch=epoch,
            max_epochs=max_epochs,
        )

        # Execute action
        if SimicAction.is_germinate(action):
            if not model.has_active_seed:
                blueprint_id = SimicAction.get_blueprint_id(action)
                seed_id = f"seed_{seeds_created}"
                model.germinate_seed(blueprint_id, seed_id)
                seeds_created += 1
                seed_optimizer = None

        elif action == SimicAction.ADVANCE:
            if model.has_active_seed:
                if model.seed_state.stage == SeedStage.TRAINING:
                    model.seed_state.transition(SeedStage.BLENDING)
                    model.seed_slot.start_blending(total_steps=5, temperature=1.0)
                elif model.seed_state.stage == SeedStage.BLENDING:
                    model.seed_state.transition(SeedStage.FOSSILIZED)
                    model.seed_slot.set_alpha(1.0)

        elif action == SimicAction.CULL:
            if model.has_active_seed:
                model.cull_seed()
                seed_optimizer = None

        done = (epoch == max_epochs)

        if collect_rollout:
            agent.store_transition(
                state.squeeze(0).cpu(),
                action_idx,
                log_prob,
                value,
                reward,
                done
            )

        episode_rewards.append(reward)

    return val_acc, action_counts, episode_rewards


# =============================================================================
# PPO Training Loop
# =============================================================================

def train_ppo(
    n_episodes: int = 100,
    max_epochs: int = 25,
    update_every: int = 5,
    device: str = "cuda:0",
    use_telemetry: bool = True,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    save_path: str = None,
):
    """Train PPO agent."""
    from esper.simic.ppo import PPOAgent
    from esper.utils import load_cifar10

    print("=" * 60)
    print("PPO Training for Tamiyo")
    print("=" * 60)
    print(f"Episodes: {n_episodes}, Max epochs: {max_epochs}")
    print(f"Device: {device}")

    trainloader, testloader = load_cifar10(batch_size=128)
    state_dim = 27

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=len(SimicAction),
        lr=lr,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        gamma=gamma,
        device=device,
    )

    history = []
    best_avg_acc = 0.0
    best_state = None
    recent_accuracies = []
    recent_rewards = []

    for ep in range(1, n_episodes + 1):
        base_seed = 42 + ep * 1000

        final_acc, action_counts, rewards = run_ppo_episode(
            agent=agent,
            trainloader=trainloader,
            testloader=testloader,
            max_epochs=max_epochs,
            base_seed=base_seed,
            device=device,
            use_telemetry=use_telemetry,
            collect_rollout=True,
            deterministic=False,
        )

        total_reward = sum(rewards)
        recent_accuracies.append(final_acc)
        recent_rewards.append(total_reward)

        if len(recent_accuracies) > 10:
            recent_accuracies.pop(0)
            recent_rewards.pop(0)

        if ep % update_every == 0 or ep == n_episodes:
            metrics = agent.update(last_value=0.0)

            avg_acc = sum(recent_accuracies) / len(recent_accuracies)
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            print(f"Episode {ep:3d}/{n_episodes}: acc={final_acc:.1f}% (avg={avg_acc:.1f}%), "
                  f"reward={total_reward:.1f}")

            history.append({
                'episode': ep,
                'accuracy': final_acc,
                'avg_accuracy': avg_acc,
                'total_reward': total_reward,
                'action_counts': dict(action_counts),
                **metrics,
            })

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_state = {k: v.clone() for k, v in agent.network.state_dict().items()}

    if best_state:
        agent.network.load_state_dict(best_state)
        print(f"\nLoaded best weights (avg_acc={best_avg_acc:.1f}%)")

    if save_path:
        agent.save(save_path, metadata={
            'n_episodes': n_episodes,
            'max_epochs': max_epochs,
            'best_avg_accuracy': best_avg_acc,
        })
        print(f"Model saved to {save_path}")

    return agent, history


__all__ = [
    "extract_transitions",
    "evaluate_iql_policy",
    "train_iql",
    "run_ppo_episode",
    "train_ppo",
]
```

**Step 2: Commit**

```bash
git add src/esper/simic/training.py
git commit -m "feat(simic): add training.py with train_ppo and train_iql"
```

---

## Task 8: Create vectorized.py

**Files:**
- Create: `src/esper/simic/vectorized.py`

**Step 1: Create vectorized.py**

Extract `ParallelEnvState` and `train_ppo_vectorized` from the original ppo.py into `src/esper/simic/vectorized.py`. This is a large function (~450 LOC) - copy it directly from the original ppo.py preserving all the CUDA stream logic.

**Step 2: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(simic): add vectorized.py with multi-GPU PPO training"
```

---

## Task 9: Create comparison.py

**Files:**
- Create: `src/esper/simic/comparison.py`

**Step 1: Create comparison.py**

Extract `load_iql_model`, `snapshot_to_features`, `live_comparison`, and `head_to_head_comparison` from the original iql.py into `src/esper/simic/comparison.py`.

**Step 2: Commit**

```bash
git add src/esper/simic/comparison.py
git commit -m "feat(simic): add comparison.py with IQL evaluation modes"
```

---

## Task 10: Update __init__.py exports

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Update exports**

Update `src/esper/simic/__init__.py` to include new modules:

```python
"""Simic - RL Training Infrastructure for Tamiyo

This package contains the reinforcement learning infrastructure for training
the Tamiyo seed lifecycle controller:

- buffers: Trajectory and replay buffers
- normalization: Observation normalization
- networks: Policy network architectures
- rewards: Reward computation
- features: Feature extraction (hot path)
- episodes: Episode data structures
- ppo: PPO agent
- iql: IQL agent
- training: Training loops
- vectorized: Multi-GPU training
- comparison: Policy comparison utilities
"""

# Core data structures
from esper.simic.episodes import (
    TrainingSnapshot,
    ActionTaken,
    StepOutcome,
    DecisionPoint,
    Episode,
    EpisodeCollector,
    DatasetManager,
    snapshot_from_signals,
    action_from_decision,
)

# Buffers
from esper.simic.buffers import (
    RolloutStep,
    RolloutBuffer,
    Transition,
    ReplayBuffer,
)

# Normalization
from esper.simic.normalization import RunningMeanStd

# Rewards
from esper.simic.rewards import (
    RewardConfig,
    SeedInfo,
    compute_shaped_reward,
    compute_potential,
    compute_pbrs_bonus,
    compute_seed_potential,
    get_intervention_cost,
    INTERVENTION_COSTS,
    STAGE_TRAINING,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
)

# Features (hot path)
from esper.simic.features import (
    safe,
    obs_to_base_features,
    telemetry_to_features,
)

# Networks
from esper.simic.networks import (
    PolicyNetwork,
    print_confusion_matrix,
    ActorCritic,
    QNetwork,
    VNetwork,
)

# NOTE: Heavy modules imported on demand:
#   from esper.simic.ppo import PPOAgent
#   from esper.simic.iql import IQL
#   from esper.simic.training import train_ppo, train_iql
#   from esper.simic.vectorized import train_ppo_vectorized
#   from esper.simic.comparison import live_comparison, head_to_head_comparison

__all__ = [
    # Episodes
    "TrainingSnapshot",
    "ActionTaken",
    "StepOutcome",
    "DecisionPoint",
    "Episode",
    "EpisodeCollector",
    "DatasetManager",
    "snapshot_from_signals",
    "action_from_decision",

    # Buffers
    "RolloutStep",
    "RolloutBuffer",
    "Transition",
    "ReplayBuffer",

    # Normalization
    "RunningMeanStd",

    # Rewards
    "RewardConfig",
    "SeedInfo",
    "compute_shaped_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_seed_potential",
    "get_intervention_cost",
    "INTERVENTION_COSTS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",

    # Features
    "safe",
    "obs_to_base_features",
    "telemetry_to_features",

    # Networks
    "PolicyNetwork",
    "print_confusion_matrix",
    "ActorCritic",
    "QNetwork",
    "VNetwork",
]
```

**Step 2: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "refactor(simic): update __init__.py exports for new modules"
```

---

## Task 11: Create scripts/train.py CLI

**Files:**
- Create: `src/esper/scripts/train.py`

**Step 1: Create CLI**

```python
#!/usr/bin/env python3
"""Training CLI for Simic RL algorithms.

Usage:
    # Train PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --episodes 100 --device cuda:0

    # Train IQL
    PYTHONPATH=src python -m esper.scripts.train iql --pack data/pack.json --epochs 100

    # Vectorized PPO
    PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 4 --devices cuda:0 cuda:1
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Train Simic RL agents")
    subparsers = parser.add_subparsers(dest="algorithm", required=True)

    # PPO subcommand
    ppo_parser = subparsers.add_parser("ppo", help="Train PPO agent")
    ppo_parser.add_argument("--episodes", type=int, default=100)
    ppo_parser.add_argument("--max-epochs", type=int, default=25)
    ppo_parser.add_argument("--update-every", type=int, default=5)
    ppo_parser.add_argument("--lr", type=float, default=3e-4)
    ppo_parser.add_argument("--clip-ratio", type=float, default=0.2)
    ppo_parser.add_argument("--entropy-coef", type=float, default=0.01)
    ppo_parser.add_argument("--gamma", type=float, default=0.99)
    ppo_parser.add_argument("--save", help="Path to save model")
    ppo_parser.add_argument("--device", default="cuda:0")
    ppo_parser.add_argument("--vectorized", action="store_true")
    ppo_parser.add_argument("--n-envs", type=int, default=4)
    ppo_parser.add_argument("--devices", nargs="+")

    # IQL subcommand
    iql_parser = subparsers.add_parser("iql", help="Train IQL agent")
    iql_parser.add_argument("--pack", required=True, help="Path to data pack")
    iql_parser.add_argument("--epochs", type=int, default=100)
    iql_parser.add_argument("--steps-per-epoch", type=int, default=1000)
    iql_parser.add_argument("--batch-size", type=int, default=256)
    iql_parser.add_argument("--gamma", type=float, default=0.99)
    iql_parser.add_argument("--tau", type=float, default=0.7)
    iql_parser.add_argument("--beta", type=float, default=3.0)
    iql_parser.add_argument("--lr", type=float, default=3e-4)
    iql_parser.add_argument("--cql-alpha", type=float, default=0.0)
    iql_parser.add_argument("--reward-shaping", action="store_true")
    iql_parser.add_argument("--no-telemetry", action="store_true")
    iql_parser.add_argument("--save", help="Path to save model")
    iql_parser.add_argument("--device", default="cpu")

    # Comparison subcommand
    cmp_parser = subparsers.add_parser("compare", help="Compare IQL vs heuristic")
    cmp_parser.add_argument("--model", required=True, help="Path to IQL model")
    cmp_parser.add_argument("--mode", choices=["live", "head-to-head"], default="head-to-head")
    cmp_parser.add_argument("--episodes", type=int, default=5)
    cmp_parser.add_argument("--max-epochs", type=int, default=25)
    cmp_parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.algorithm == "ppo":
        if args.vectorized:
            from esper.simic.vectorized import train_ppo_vectorized
            train_ppo_vectorized(
                n_episodes=args.episodes,
                n_envs=args.n_envs,
                max_epochs=args.max_epochs,
                device=args.device,
                devices=args.devices,
                lr=args.lr,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                gamma=args.gamma,
                save_path=args.save,
            )
        else:
            from esper.simic.training import train_ppo
            train_ppo(
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                update_every=args.update_every,
                device=args.device,
                lr=args.lr,
                clip_ratio=args.clip_ratio,
                entropy_coef=args.entropy_coef,
                gamma=args.gamma,
                save_path=args.save,
            )

    elif args.algorithm == "iql":
        from esper.simic.training import train_iql
        train_iql(
            pack_path=args.pack,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            beta=args.beta,
            lr=args.lr,
            cql_alpha=args.cql_alpha,
            use_telemetry=not args.no_telemetry,
            use_reward_shaping=args.reward_shaping,
            device=args.device,
            save_path=args.save,
        )

    elif args.algorithm == "compare":
        from esper.simic.comparison import live_comparison, head_to_head_comparison
        if args.mode == "live":
            live_comparison(
                model_path=args.model,
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                device=args.device,
            )
        else:
            head_to_head_comparison(
                model_path=args.model,
                n_episodes=args.episodes,
                max_epochs=args.max_epochs,
                device=args.device,
            )


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "feat(scripts): add train.py CLI for PPO/IQL training"
```

---

## Task 12: Run all tests and verify

**Step 1: Run full test suite**

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

Expected: All tests PASS

**Step 2: Verify imports work**

```bash
PYTHONPATH=src python -c "
from esper.simic import RolloutBuffer, ReplayBuffer, RunningMeanStd
from esper.simic import ActorCritic, QNetwork, VNetwork
from esper.simic.ppo import PPOAgent
from esper.simic.iql import IQL
from esper.simic.training import train_ppo, train_iql
print('All imports successful!')
"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "refactor(simic): complete PPO/IQL modularization

- Extract buffers.py (RolloutBuffer, ReplayBuffer)
- Extract normalization.py (RunningMeanStd)
- Add RL networks to networks.py (ActorCritic, QNetwork, VNetwork)
- Slim ppo.py to agent-only (~200 LOC)
- Slim iql.py to agent-only (~150 LOC)
- Add training.py (train_ppo, train_iql)
- Add vectorized.py (multi-GPU PPO)
- Add comparison.py (IQL evaluation)
- Add scripts/train.py CLI

Total: ppo.py 1600→200 LOC, iql.py 1300→150 LOC
"
```

---

## Summary

| Before | After | Change |
|--------|-------|--------|
| ppo.py: 1591 LOC | ppo.py: ~200 LOC | -87% |
| iql.py: 1326 LOC | iql.py: ~150 LOC | -89% |
| networks.py: 342 LOC | networks.py: ~450 LOC | +108 LOC (new networks) |
| - | buffers.py: ~200 LOC | new |
| - | normalization.py: ~50 LOC | new |
| - | training.py: ~400 LOC | new |
| - | vectorized.py: ~450 LOC | new |
| - | comparison.py: ~350 LOC | new |
| - | scripts/train.py: ~100 LOC | new |

**Total Simic LOC**: ~4600 → ~4700 (slight increase due to module boilerplate, but much better organized)

**Key wins:**
- No file over 500 LOC
- Clear separation of concerns
- Reusable components (buffers, networks, normalization)
- CLI moved out of library modules
- Follows existing simic patterns
