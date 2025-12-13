# Factored Actions Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate from flat action space to factored action space, remove dead ADVANCE operation, delete flat mode entirely, and wire up FactoredActorCritic to the training pipeline.

**Architecture:** The factored action space separates decisions into independent heads (slot, blueprint, blend, op) enabling more efficient learning. The flat action space combines germinate variants into single actions, which doesn't scale well. FactoredActorCritic already exists and is tested - we just need to wire it up and delete the flat implementation.

**Tech Stack:** PyTorch 2.9, Python 3.13, custom PPO implementation

---

## Phase 1: Remove Dead ADVANCE Operation

The ADVANCE operation was designed for explicit stage transitions, but the correct approach is mechanical transitions: seeds auto-advance when they reach the natural end of a stage. The only explicit actions are GERMINATE, FOSSILIZE, and CULL.

### Task 1.1: Remove ADVANCE from LifecycleOp

**Files:**

- Modify: `src/esper/leyline/factored_actions.py:59-66`
- Test: `tests/leyline/test_factored_actions.py`

**Step 1: Update LifecycleOp enum**

Remove ADVANCE and renumber:

```python
class LifecycleOp(IntEnum):
    """Lifecycle operation."""
    WAIT = 0
    GERMINATE = 1
    CULL = 2       # was 3
    FOSSILIZE = 3  # was 4
```

**Step 2: Remove is_advance property from FactoredAction**

Delete lines 80-82:

```python
    @property
    def is_advance(self) -> bool:
        return self.op == LifecycleOp.ADVANCE
```

**Step 3: Run tests to see what breaks**

```bash
pytest tests/leyline/test_factored_actions.py -v
```

Expected: FAIL - tests reference ADVANCE

**Step 4: Update test_factored_actions.py**

Remove/update ADVANCE-related assertions:

- Line 36: Remove `assert LifecycleOp.ADVANCE.value == 2`
- Line 90: Remove ADVANCE test case
- Update any assertions checking NUM_OPS (now 4, was 5)

**Step 5: Verify tests pass**

```bash
pytest tests/leyline/test_factored_actions.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/leyline/factored_actions.py tests/leyline/test_factored_actions.py
git commit -m "$(cat <<'EOF'
refactor(leyline): remove ADVANCE from LifecycleOp

Stage transitions are mechanical - seeds auto-advance when reaching
the natural end of a stage. Only explicit actions are:
- GERMINATE: create new seed
- FOSSILIZE: promote from PROBATIONARY
- CULL: remove seed

NUM_OPS changes from 5 to 4.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Remove ADVANCE from action_masks.py

**Files:**

- Modify: `src/esper/simic/action_masks.py`
- Test: `tests/simic/test_action_masks.py`

**Step 1: Remove _ADVANCEABLE_STAGES constant**

Delete lines 34-38:

```python
_ADVANCEABLE_STAGES = frozenset({
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
})
```

**Step 2: Remove ADVANCE from docstring**

Update module docstring (lines 1-12) to remove ADVANCE reference:

```python
"""Action Masking for Multi-Slot Control.

Only masks PHYSICALLY IMPOSSIBLE actions:
- GERMINATE: blocked if slot occupied OR at seed limit
- FOSSILIZE: blocked if not PROBATIONARY
- CULL: blocked if no seed OR seed_age < MIN_CULL_AGE
- WAIT: always valid

Does NOT mask timing heuristics (epoch, plateau, stabilization).
Tamiyo learns optimal timing from counterfactual reward signals.
"""
```

**Step 3: Remove ADVANCE masking logic from compute_action_masks()**

In `compute_action_masks()`, remove lines 139-141:

```python
        # ADVANCE: only from GERMINATED, TRAINING, BLENDING
        if stage in _ADVANCEABLE_STAGES:
            op_mask[LifecycleOp.ADVANCE] = True
```

Also update comment on line 134 from "ADVANCE/FOSSILIZE/CULL" to "FOSSILIZE/CULL".

**Step 4: Remove ADVANCE masking logic from compute_batch_masks()**

In `compute_batch_masks()`, remove lines 208-209:

```python
            if stage in _ADVANCEABLE_STAGES:
                op_masks[i, LifecycleOp.ADVANCE] = True
```

Also update comment on line 203 from "ADVANCE/FOSSILIZE/CULL" to "FOSSILIZE/CULL".

**Step 5: Run tests to see what breaks**

```bash
pytest tests/simic/test_action_masks.py -v
```

Expected: FAIL - many tests reference ADVANCE

**Step 6: Update test_action_masks.py**

Remove all ADVANCE-related assertions:

- Remove `LifecycleOp.ADVANCE` from imports if present
- Remove all `assert masks["op"][LifecycleOp.ADVANCE] == ...` assertions
- Delete `test_mask_advanceable_stages()` test entirely
- Update `test_batch_masks_...` tests to remove ADVANCE checks

**Step 7: Verify tests pass**

```bash
pytest tests/simic/test_action_masks.py -v
```

Expected: PASS

**Step 8: Commit**

```bash
git add src/esper/simic/action_masks.py tests/simic/test_action_masks.py
git commit -m "$(cat <<'EOF'
refactor(simic): remove ADVANCE masking from action_masks

Aligns with leyline change - ADVANCE operation doesn't exist.
Stage transitions are mechanical, not policy decisions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.3: Update integration tests

**Files:**

- Modify: `tests/integration/test_multislot_pipeline.py`

**Step 1: Run integration tests**

```bash
pytest tests/integration/test_multislot_pipeline.py -v
```

Expected: FAIL - references to ADVANCE

**Step 2: Remove ADVANCE assertions**

Update all tests that check ADVANCE validity:

- Line 160: Remove ADVANCE assertion
- Line 183: Remove ADVANCE check
- Line 489: Remove ADVANCE assertion
- Line 494: Remove ADVANCE assertion

**Step 3: Update NUM_OPS expectations**

Any test checking `num_ops=5` should be changed to `num_ops=4`.

**Step 4: Verify tests pass**

```bash
pytest tests/integration/test_multislot_pipeline.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/integration/test_multislot_pipeline.py
git commit -m "$(cat <<'EOF'
test(integration): remove ADVANCE from multislot pipeline tests

Aligns with removal of ADVANCE operation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Add FactoredRolloutBuffer

### Task 2.1: Create FactoredRolloutBuffer

**Files:**

- Modify: `src/esper/simic/buffers.py`
- Create: `tests/simic/test_buffers_factored.py`

**Step 1: Write failing test**

Create `tests/simic/test_buffers_factored.py`:

```python
"""Tests for FactoredRolloutBuffer."""

import pytest
import torch

from esper.simic.buffers import FactoredRolloutBuffer


def test_factored_buffer_add_and_len():
    """FactoredRolloutBuffer should store transitions."""
    buffer = FactoredRolloutBuffer()

    state = torch.randn(50)
    action = {"slot": 0, "blueprint": 1, "blend": 2, "op": 1}
    log_prob = -1.5
    value = 0.5
    reward = 1.0
    done = False
    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.tensor([True, True, False, False], dtype=torch.bool),
    }

    buffer.add(state, action, log_prob, value, reward, done, masks)

    assert len(buffer) == 1


def test_factored_buffer_get_batches():
    """get_batches should return factored action tensors."""
    buffer = FactoredRolloutBuffer()

    # Add 10 transitions
    for i in range(10):
        buffer.add(
            state=torch.randn(50),
            action={"slot": i % 3, "blueprint": i % 5, "blend": i % 3, "op": i % 4},
            log_prob=-1.0,
            value=0.5,
            reward=1.0,
            done=(i == 9),
            action_masks={
                "slot": torch.ones(3, dtype=torch.bool),
                "blueprint": torch.ones(5, dtype=torch.bool),
                "blend": torch.ones(3, dtype=torch.bool),
                "op": torch.ones(4, dtype=torch.bool),
            },
        )

    batches = list(buffer.get_batches(batch_size=5, device="cpu"))
    assert len(batches) == 2

    batch, indices = batches[0]
    assert "actions" in batch
    assert isinstance(batch["actions"], dict)
    assert set(batch["actions"].keys()) == {"slot", "blueprint", "blend", "op"}
    assert batch["actions"]["slot"].shape == (5,)


def test_factored_buffer_gae_resets_on_done():
    """GAE computation must reset advantage chain at episode boundaries."""
    buffer = FactoredRolloutBuffer()

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }

    # Episode 1: 3 steps
    for i in range(3):
        buffer.add(
            state=torch.randn(50),
            action={"slot": 0, "blueprint": 0, "blend": 0, "op": 0},
            log_prob=-1.0,
            value=1.0,
            reward=1.0,
            done=(i == 2),
            action_masks=masks,
        )

    # Episode 2: 2 steps
    for i in range(2):
        buffer.add(
            state=torch.randn(50),
            action={"slot": 0, "blueprint": 0, "blend": 0, "op": 0},
            log_prob=-1.0,
            value=1.0,
            reward=10.0,  # Different reward to distinguish
            done=(i == 1),
            action_masks=masks,
        )

    returns, advantages = buffer.compute_returns_and_advantages(
        last_value=0.0, gamma=0.99, gae_lambda=0.95, device="cpu"
    )

    # Episode 2's high rewards shouldn't bleed into Episode 1's advantages
    # Episode 1 ends at index 2, Episode 2 is indices 3-4
    # Advantage at index 2 (end of ep1) should be computed from ep1 rewards only
    assert returns.shape == (5,)
    assert advantages.shape == (5,)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/simic/test_buffers_factored.py -v
```

Expected: FAIL - FactoredRolloutBuffer doesn't exist

**Step 3: Implement FactoredRolloutBuffer**

Add to `src/esper/simic/buffers.py`:

```python
from typing import NamedTuple


class FactoredTransition(NamedTuple):
    """Single transition with factored action.

    Uses NamedTuple for immutability and _replace() support.
    """

    state: torch.Tensor
    slot_action: int
    blueprint_action: int
    blend_action: int
    op_action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    slot_mask: torch.Tensor
    blueprint_mask: torch.Tensor
    blend_mask: torch.Tensor
    op_mask: torch.Tensor
    truncated: bool = False
    bootstrap_value: float = 0.0


class FactoredRolloutBuffer:
    """Rollout buffer for factored action space.

    Stores transitions with dict-based actions and returns batched
    factored tensors for PPO updates.
    """

    def __init__(self):
        self.steps: list[FactoredTransition] = []

    def add(
        self,
        state: torch.Tensor,
        action: dict[str, int],
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_masks: dict[str, torch.Tensor],
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Add transition to buffer.

        Args:
            state: Observation tensor (detached automatically)
            action: Dict with keys {slot, blueprint, blend, op}
            log_prob: Log probability of joint action
            value: Value estimate
            reward: Reward received
            done: Whether episode ended
            action_masks: Dict of boolean mask tensors per head
            truncated: Whether episode ended due to time limit
            bootstrap_value: Value to bootstrap from if truncated
        """
        self.steps.append(FactoredTransition(
            state=state.detach(),  # Prevent gradient graph retention
            slot_action=action["slot"],
            blueprint_action=action["blueprint"],
            blend_action=action["blend"],
            op_action=action["op"],
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            slot_mask=action_masks["slot"].detach(),
            blueprint_mask=action_masks["blueprint"].detach(),
            blend_mask=action_masks["blend"].detach(),
            op_mask=action_masks["op"].detach(),
            truncated=truncated,
            bootstrap_value=bootstrap_value,
        ))

    def __len__(self) -> int:
        return len(self.steps)

    def clear(self) -> None:
        self.steps.clear()

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE returns and advantages.

        CRITICAL: Resets advantage chain at episode boundaries (done=True).
        """
        n = len(self.steps)

        # Create tensors on device
        rewards = torch.tensor([t.reward for t in self.steps], device=device)
        values = torch.tensor([t.value for t in self.steps], device=device)
        dones = torch.tensor([t.done for t in self.steps], device=device)

        returns = torch.zeros(n, device=device)
        advantages = torch.zeros(n, device=device)

        # GAE computation (reversed)
        next_value = last_value
        next_advantage = 0.0

        for i in reversed(range(n)):
            step = self.steps[i]

            # CRITICAL: Reset at episode boundaries
            if step.done:
                if step.truncated:
                    next_value = step.bootstrap_value
                else:
                    next_value = 0.0
                next_advantage = 0.0  # Reset GAE chain

            delta = rewards[i] + gamma * next_value - values[i]
            advantages[i] = delta + gamma * gae_lambda * next_advantage
            returns[i] = advantages[i] + values[i]

            next_value = values[i].item()
            next_advantage = advantages[i].item()

        return returns, advantages

    def get_batches(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ):
        """Yield batches of transitions.

        Pre-stacks all data once for efficiency, then slices for batches.
        """
        n = len(self.steps)
        indices = torch.randperm(n)

        # Pre-stack ALL data once (not per-batch) for efficiency
        all_states = torch.stack([t.state for t in self.steps]).to(device)
        all_slot_actions = torch.tensor(
            [t.slot_action for t in self.steps], dtype=torch.long, device=device
        )
        all_blueprint_actions = torch.tensor(
            [t.blueprint_action for t in self.steps], dtype=torch.long, device=device
        )
        all_blend_actions = torch.tensor(
            [t.blend_action for t in self.steps], dtype=torch.long, device=device
        )
        all_op_actions = torch.tensor(
            [t.op_action for t in self.steps], dtype=torch.long, device=device
        )
        all_log_probs = torch.tensor(
            [t.log_prob for t in self.steps], dtype=torch.float32, device=device
        )
        all_values = torch.tensor(
            [t.value for t in self.steps], dtype=torch.float32, device=device
        )

        # Pre-stack masks
        all_slot_masks = torch.stack([t.slot_mask for t in self.steps]).to(device)
        all_blueprint_masks = torch.stack([t.blueprint_mask for t in self.steps]).to(device)
        all_blend_masks = torch.stack([t.blend_mask for t in self.steps]).to(device)
        all_op_masks = torch.stack([t.op_mask for t in self.steps]).to(device)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]

            yield {
                "states": all_states[batch_idx],
                "actions": {
                    "slot": all_slot_actions[batch_idx],
                    "blueprint": all_blueprint_actions[batch_idx],
                    "blend": all_blend_actions[batch_idx],
                    "op": all_op_actions[batch_idx],
                },
                "old_log_probs": all_log_probs[batch_idx],
                "values": all_values[batch_idx],
                "action_masks": {
                    "slot": all_slot_masks[batch_idx],
                    "blueprint": all_blueprint_masks[batch_idx],
                    "blend": all_blend_masks[batch_idx],
                    "op": all_op_masks[batch_idx],
                },
            }, batch_idx
```

**Step 4: Add to __all__ in buffers.py**

```python
__all__ = [
    "RolloutBuffer",
    "RecurrentRolloutBuffer",
    "FactoredRolloutBuffer",
    "FactoredTransition",
]
```

**Step 5: Run tests**

```bash
pytest tests/simic/test_buffers_factored.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/buffers.py tests/simic/test_buffers_factored.py
git commit -m "$(cat <<'EOF'
feat(simic): add FactoredRolloutBuffer for factored actions

- Uses NamedTuple for FactoredTransition (immutable, _replace() support)
- Pre-stacks all data once for batch efficiency
- GAE correctly resets advantage chain at episode boundaries
- Stores detached tensors to prevent gradient graph retention

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Wire Up FactoredActorCritic to PPOAgent

### Task 3.1: Add factored mode to PPOAgent

**Files:**

- Modify: `src/esper/simic/ppo.py`
- Create: `tests/simic/test_ppo_factored.py`

**Step 1: Write failing test**

Create `tests/simic/test_ppo_factored.py`:

```python
"""Tests for PPOAgent with factored action space."""

import pytest
import torch

from esper.simic.ppo import PPOAgent


def test_ppo_agent_factored_init():
    """PPOAgent with factored=True should use FactoredActorCritic."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
    )

    from esper.simic.factored_network import FactoredActorCritic
    assert isinstance(agent._base_network, FactoredActorCritic)


def test_ppo_agent_factored_get_action():
    """Factored agent should return dict of actions."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
    )

    state = torch.randn(1, 50)
    masks = {
        "slot": torch.ones(1, 3, dtype=torch.bool),
        "blueprint": torch.ones(1, 5, dtype=torch.bool),
        "blend": torch.ones(1, 3, dtype=torch.bool),
        "op": torch.tensor([[True, True, False, False]], dtype=torch.bool),
    }

    action, log_prob, value, _ = agent.get_action(state, masks)

    assert isinstance(action, dict)
    assert set(action.keys()) == {"slot", "blueprint", "blend", "op"}
    assert isinstance(action["slot"], int)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/simic/test_ppo_factored.py::test_ppo_agent_factored_init -v
```

Expected: FAIL - `factored` parameter doesn't exist

**Step 3: Add factored parameter to PPOAgent.__init__**

In `src/esper/simic/ppo.py`:

Add import at top (after existing imports):

```python
from esper.simic.factored_network import FactoredActorCritic
from esper.simic.buffers import FactoredRolloutBuffer
```

Add to `__init__` parameters (around line 167, after `compile_network`):

```python
        # Factored action space (mutually exclusive with recurrent for now)
        factored: bool = False,
```

Add after parameter validation:

```python
        self.factored = factored

        if factored and recurrent:
            raise ValueError("factored=True and recurrent=True are mutually exclusive (for now)")
```

Update network initialization (replace lines 193-204):

```python
        if factored:
            from esper.leyline.factored_actions import NUM_SLOTS, NUM_BLUEPRINTS, NUM_BLENDS, NUM_OPS
            self.network = FactoredActorCritic(
                state_dim=state_dim,
                num_slots=NUM_SLOTS,
                num_blueprints=NUM_BLUEPRINTS,
                num_blends=NUM_BLENDS,
                num_ops=NUM_OPS,
                hidden_dim=hidden_dim,
            ).to(device)
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

        # torch.compile for all network types
        if compile_network:
            self.network = torch.compile(self.network, mode="default")
```

**Step 4: Update get_action for factored mode**

Modify `get_action` method signature and body:

```python
    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor | dict[str, torch.Tensor],
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[int | dict[str, int], float, float, tuple | None]:
        """Get action from policy.

        Args:
            state: Observation tensor [1, state_dim] or [state_dim]
            action_mask: For flat: tensor [action_dim], for factored: dict of tensors
            hidden: LSTM hidden state for recurrent, None otherwise
            deterministic: If True, return argmax instead of sampling

        Returns:
            action: int (flat) or dict[str, int] (factored)
            log_prob: Log probability of action
            value: State value estimate
            hidden: Updated hidden state (recurrent) or None
        """
        if self.factored:
            actions, log_prob, value = self.network.get_action_batch(
                state, action_mask, deterministic
            )
            # Convert tensor actions to Python ints
            action_dict = {k: int(v.item()) for k, v in actions.items()}
            return action_dict, float(log_prob.item()), float(value.item()), None
        elif self.recurrent:
            return self.network.get_action(state, action_mask, hidden, deterministic)
        else:
            action, log_prob, value, _ = self.network.get_action(state, action_mask, deterministic)
            return action, log_prob, value, None
```

**Step 5: Run tests**

```bash
pytest tests/simic/test_ppo_factored.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_factored.py
git commit -m "$(cat <<'EOF'
feat(simic): add factored=True mode to PPOAgent

- Uses FactoredActorCritic with multi-head policy
- Creates FactoredRolloutBuffer for dict-based actions
- get_action returns dict[str, int] for factored mode
- Mutually exclusive with recurrent=True (for now)
- torch.compile applied to all network types

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.2: Add store_factored_transition and update_factored

**Files:**

- Modify: `src/esper/simic/ppo.py`
- Modify: `tests/simic/test_ppo_factored.py`

**Step 1: Add test for update_factored**

Add to `tests/simic/test_ppo_factored.py`:

```python
def test_ppo_agent_factored_update():
    """Factored agent should perform PPO update with all diagnostics."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
        n_epochs=2,
    )

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.tensor([True, True, False, False], dtype=torch.bool),
    }
    batched_masks = {k: v.unsqueeze(0) for k, v in masks.items()}

    # Add transitions
    for i in range(10):
        state = torch.randn(1, 50)
        action, log_prob, value, _ = agent.get_action(state, batched_masks)
        agent.store_factored_transition(
            state=state.squeeze(0),
            action=action,
            log_prob=log_prob,
            value=value,
            reward=1.0,
            done=(i == 9),
            action_masks=masks,
        )

    # Update
    metrics = agent.update_factored()

    # Core metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics

    # DRL Expert required: explained variance
    assert "explained_variance" in metrics

    # DRL Expert required: ratio tracking
    assert "ratio_mean" in metrics
    assert "ratio_max" in metrics

    # DRL Expert required: approx KL
    assert "approx_kl" in metrics


def test_ppo_agent_factored_target_kl_early_stop():
    """Factored update should support target_kl early stopping."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
        n_epochs=100,  # Many epochs to trigger early stop
        target_kl=0.001,  # Very low to trigger early stop
    )

    masks = {
        "slot": torch.ones(3, dtype=torch.bool),
        "blueprint": torch.ones(5, dtype=torch.bool),
        "blend": torch.ones(3, dtype=torch.bool),
        "op": torch.ones(4, dtype=torch.bool),
    }
    batched_masks = {k: v.unsqueeze(0) for k, v in masks.items()}

    for i in range(20):
        state = torch.randn(1, 50)
        action, log_prob, value, _ = agent.get_action(state, batched_masks)
        agent.store_factored_transition(
            state=state.squeeze(0),
            action=action,
            log_prob=log_prob,
            value=value,
            reward=float(i),  # Varying rewards
            done=(i == 19),
            action_masks=masks,
        )

    metrics = agent.update_factored()

    # Should have stopped early (not all 100 epochs)
    if "early_stop_epoch" in metrics:
        assert metrics["early_stop_epoch"] < 100
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/simic/test_ppo_factored.py::test_ppo_agent_factored_update -v
```

Expected: FAIL - methods don't exist

**Step 3: Add store_factored_transition method**

Add to PPOAgent class:

```python
    def store_factored_transition(
        self,
        state: torch.Tensor,
        action: dict[str, int],
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        action_masks: dict[str, torch.Tensor],
        truncated: bool = False,
        bootstrap_value: float = 0.0,
    ) -> None:
        """Store transition for factored action space."""
        if not self.factored:
            raise RuntimeError("store_factored_transition requires factored=True")
        self.factored_buffer.add(
            state, action, log_prob, value, reward, done, action_masks,
            truncated, bootstrap_value
        )
```

**Step 4: Add update_factored method**

Add to PPOAgent class (full implementation with all DRL Expert requirements):

```python
    def update_factored(self, last_value: float = 0.0) -> dict:
        """Perform PPO update for factored action space.

        Includes all features from update():
        - Target KL early stopping
        - Value clipping
        - Adaptive entropy coefficient
        - Explained variance
        - Ratio/KL tracking for diagnostics
        """
        if not self.factored:
            raise RuntimeError("update_factored requires factored=True")

        if len(self.factored_buffer) == 0:
            return {}

        # Compute returns and advantages
        returns, advantages = self.factored_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda, device=self.device
        )

        # Compute explained variance BEFORE updates
        values_tensor = torch.tensor(
            [t.value for t in self.factored_buffer.steps],
            device=self.device,
        )
        var_returns = returns.var()
        if var_returns > 1e-8:
            explained_variance = 1.0 - (returns - values_tensor).var() / var_returns
            explained_variance = float(explained_variance.item())
        else:
            explained_variance = 0.0

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = defaultdict(list)
        metrics['explained_variance'] = [explained_variance]
        early_stopped = False

        for epoch_i in range(self.n_epochs):
            if early_stopped:
                break

            epoch_kl_sum = 0.0
            epoch_kl_count = 0

            for batch, batch_idx in self.factored_buffer.get_batches(self.batch_size, self.device):
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(
                    batch["states"],
                    batch["actions"],
                    batch["action_masks"],
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch["old_log_probs"])

                # Track ratio stats
                ratio_stats = torch.stack([ratio.mean(), ratio.std(), ratio.max(), ratio.min()])
                r_mean, r_std, r_max, r_min = ratio_stats.tolist()
                metrics['ratio_mean'].append(r_mean)
                metrics['ratio_std'].append(r_std)
                metrics['ratio_max'].append(r_max)
                metrics['ratio_min'].append(r_min)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping
                old_values = batch["values"]
                if self.clip_value:
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -self.clip_ratio, self.clip_ratio
                    )
                    value_loss_unclipped = (values - batch_returns) ** 2
                    value_loss_clipped = (values_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus with adaptive coefficient
                entropy_loss = -entropy.mean()
                representative_mask = batch["action_masks"]["op"][0]
                entropy_coef = self.get_entropy_coef(representative_mask)

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                batch_kl = (batch["old_log_probs"] - log_probs).mean()
                clip_frac = ((ratio - 1).abs() > self.clip_ratio).float().mean()

                batch_metrics = torch.stack([policy_loss, value_loss, -entropy_loss, batch_kl, clip_frac])
                pl, vl, ent, kl, cf = batch_metrics.tolist()

                metrics['policy_loss'].append(pl)
                metrics['value_loss'].append(vl)
                metrics['entropy'].append(ent)
                metrics['approx_kl'].append(kl)
                metrics['clip_fraction'].append(cf)

                epoch_kl_sum += kl
                epoch_kl_count += 1

            # Target KL early stopping
            if self.target_kl is not None and epoch_kl_count > 0:
                epoch_kl_avg = epoch_kl_sum / epoch_kl_count
                if epoch_kl_avg > 1.5 * self.target_kl:
                    early_stopped = True
                    metrics['early_stop_epoch'] = [epoch_i + 1]

        self.train_steps += 1
        self.factored_buffer.clear()

        # Aggregate metrics
        result = {}
        for k, v in metrics.items():
            result[k] = sum(v) / len(v) if v else 0.0

        if early_stopped:
            result['early_stopped'] = 1.0

        return result
```

**Step 5: Run tests**

```bash
pytest tests/simic/test_ppo_factored.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/simic/test_ppo_factored.py
git commit -m "$(cat <<'EOF'
feat(simic): add store_factored_transition and update_factored

Full PPO update implementation for factored actions:
- Target KL early stopping
- Value clipping (matching feedforward PPO)
- Adaptive entropy coefficient via get_entropy_coef()
- Explained variance computation
- Ratio/KL/clip_fraction tracking for diagnostics

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Update Training Pipeline

### Task 4.1: Add factored mode to train_ppo_vectorized

**Files:**

- Modify: `src/esper/simic/vectorized.py`

**Step 1: Add factored parameter to function signature**

Find `def train_ppo_vectorized(` and add parameter:

```python
    factored: bool = False,
```

**Step 2: Update imports**

Add near top of file:

```python
from esper.simic.action_masks import compute_action_masks, build_slot_states
from esper.leyline.factored_actions import FactoredAction, LifecycleOp
```

**Step 3: Update PPOAgent creation**

Find where PPOAgent is created and add factored parameter:

```python
agent = PPOAgent(
    state_dim=...,
    factored=factored,
    ...
)
```

**Step 4: Update mask computation**

Replace flat mask computation with factored when `factored=True`:

```python
if factored:
    slot_states = build_slot_states(model, [target_slot])
    masks = compute_action_masks(
        slot_states=slot_states,
        total_seeds=model.count_active_seeds() if model else 0,
        max_seeds=effective_max_seeds,
        device=device,
    )
    # Batch masks for network (add batch dim)
    batched_masks = {k: v.unsqueeze(0) for k, v in masks.items()}
    all_masks.append(masks)  # Store unbatched for buffer
else:
    mask = compute_flat_action_mask(...)
    all_masks.append(mask)
```

**Step 5: Update action execution**

For factored mode, map dict action to execution:

```python
if factored:
    factored_action = FactoredAction.from_indices(
        slot_idx=action["slot"],
        blueprint_idx=action["blueprint"],
        blend_idx=action["blend"],
        op_idx=action["op"],
    )

    action_success = False
    if factored_action.is_germinate:
        if not model.has_active_seed:
            env_state.acc_at_germination = env_state.val_acc
            blueprint_id = factored_action.blueprint_id
            seed_id = f"env{env_idx}_seed_{env_state.seeds_created}"
            model.germinate_seed(blueprint_id, seed_id, slot=target_slot)
            env_state.seeds_created += 1
            env_state.seed_optimizer = None
            action_success = True
    elif factored_action.is_fossilize:
        action_success = _advance_active_seed(model, slots)
    elif factored_action.is_cull:
        if model.has_active_seed:
            model.cull_seed(slot=target_slot)
            env_state.seed_optimizer = None
            action_success = True
    # WAIT does nothing, action_success stays False
else:
    # Existing flat action execution
    ...
```

**Step 6: Update transition storage**

```python
if factored:
    agent.store_factored_transition(
        state=states_batch_normalized[env_idx],
        action=action,
        log_prob=log_prob,
        value=value,
        reward=reward,
        done=done,
        action_masks=masks,  # Unbatched
        truncated=truncated,
        bootstrap_value=bootstrap_value,
    )
else:
    agent.store_transition(...)
```

**Step 7: Update PPO update call**

```python
if factored:
    update_metrics = agent.update_factored(last_value=last_values[env_idx])
else:
    update_metrics = agent.update(last_value=last_values[env_idx])
```

**Step 8: Run existing tests**

```bash
pytest tests/integration/test_multislot_pipeline.py -v
```

Expected: PASS (flat mode still works)

**Step 9: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "$(cat <<'EOF'
feat(simic): add factored mode to train_ppo_vectorized

When factored=True:
- Uses compute_action_masks() for dict-based masks
- Returns dict actions from FactoredActorCritic
- Maps actions via FactoredAction for execution
- Uses store_factored_transition and update_factored

Flat mode remains for backwards compatibility during transition.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4.2: Add factored integration test

**Files:**

- Create: `tests/integration/test_factored_training.py`

**Step 1: Create integration test**

```python
"""Integration tests for factored action space training."""

import pytest
import torch

from esper.simic.ppo import PPOAgent
from esper.simic.action_masks import compute_action_masks, MaskSeedInfo
from esper.leyline import SeedStage
from esper.leyline.factored_actions import FactoredAction, LifecycleOp


def test_factored_training_loop():
    """End-to-end factored training loop."""
    agent = PPOAgent(
        state_dim=50,
        factored=True,
        device="cpu",
        compile_network=False,
        n_epochs=2,
    )

    # Simulate one episode
    for step in range(5):
        state = torch.randn(1, 50)

        # Compute masks (empty slots = can germinate)
        slot_states = {"early": None, "mid": None, "late": None}
        masks = compute_action_masks(slot_states, total_seeds=0, max_seeds=3)

        # Batch masks for network
        batched_masks = {k: v.unsqueeze(0) for k, v in masks.items()}

        action, log_prob, value, _ = agent.get_action(state, batched_masks)

        # Verify action is factored
        assert isinstance(action, dict)
        assert set(action.keys()) == {"slot", "blueprint", "blend", "op"}

        # Can construct FactoredAction
        fa = FactoredAction.from_indices(
            action["slot"], action["blueprint"], action["blend"], action["op"]
        )
        assert fa.op in list(LifecycleOp)

        # Store transition
        agent.store_factored_transition(
            state=state.squeeze(0),
            action=action,
            log_prob=log_prob,
            value=value,
            reward=1.0,
            done=(step == 4),
            action_masks=masks,
        )

    # Update
    metrics = agent.update_factored()

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "explained_variance" in metrics
    assert metrics["policy_loss"] < 10.0  # Sanity check


def test_factored_masking_germinate():
    """GERMINATE masked correctly in factored space."""
    # Empty slots = can germinate
    slot_states = {"early": None}
    masks = compute_action_masks(slot_states, total_seeds=0, max_seeds=3)

    assert masks["op"][LifecycleOp.WAIT].item() == True
    assert masks["op"][LifecycleOp.GERMINATE].item() == True
    assert masks["op"][LifecycleOp.CULL].item() == False
    assert masks["op"][LifecycleOp.FOSSILIZE].item() == False


def test_factored_masking_fossilize():
    """FOSSILIZE masked correctly in factored space."""
    # Seed in PROBATIONARY = can fossilize
    slot_states = {
        "early": MaskSeedInfo(stage=SeedStage.PROBATIONARY.value, seed_age_epochs=5)
    }
    masks = compute_action_masks(slot_states, total_seeds=1, max_seeds=3)

    assert masks["op"][LifecycleOp.FOSSILIZE].item() == True
    assert masks["op"][LifecycleOp.CULL].item() == True  # age >= MIN_CULL_AGE


def test_factored_masking_seed_limit():
    """GERMINATE blocked at seed limit."""
    slot_states = {"early": None}
    masks = compute_action_masks(slot_states, total_seeds=3, max_seeds=3)

    assert masks["op"][LifecycleOp.GERMINATE].item() == False
    assert masks["op"][LifecycleOp.WAIT].item() == True
```

**Step 2: Run tests**

```bash
pytest tests/integration/test_factored_training.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_factored_training.py
git commit -m "$(cat <<'EOF'
test(integration): add factored action space training tests

Verifies:
- End-to-end factored training loop
- FactoredAction construction from agent output
- Action masking for GERMINATE/FOSSILIZE/CULL
- Seed limit enforcement

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Delete Flat Mode

Per CLAUDE.md "No Legacy Code Policy", we delete flat mode entirely.

### Task 5.1: Delete compute_flat_action_mask

**Files:**

- Modify: `src/esper/simic/action_masks.py`
- Modify: `tests/simic/test_action_masks.py`

**Step 1: Delete compute_flat_action_mask function**

Remove the entire `compute_flat_action_mask()` function (lines 225-273).

**Step 2: Remove from __all__**

Update `__all__`:

```python
__all__ = [
    "MaskSeedInfo",
    "build_slot_states",
    "compute_action_masks",
    "compute_batch_masks",
    "MIN_CULL_AGE",
]
```

**Step 3: Delete flat mask tests**

Remove any tests that use `compute_flat_action_mask` from `test_action_masks.py`.

**Step 4: Run tests**

```bash
pytest tests/simic/test_action_masks.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/action_masks.py tests/simic/test_action_masks.py
git commit -m "$(cat <<'EOF'
refactor(simic): delete compute_flat_action_mask

Per CLAUDE.md No Legacy Code Policy - flat mode removed.
All training now uses factored action space.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.2: Delete flat Action enum

**Files:**

- Modify: `src/esper/leyline/actions.py`

**Step 1: Delete Action class**

Remove the entire `class Action(IntEnum)` (lines 54-98).

**Step 2: Update __all__**

```python
__all__ = [
    "build_action_enum",
    "get_blueprint_from_action",
    "is_germinate_action",
]
```

**Step 3: Find and fix broken imports**

```bash
grep -r "from esper.leyline.actions import Action" src/
grep -r "from esper.leyline import.*Action" src/
```

Update any files that import `Action` to use factored actions instead.

**Step 4: Run tests**

```bash
pytest tests/ -v -k "not test_multislot" --tb=short
```

Fix any failures.

**Step 5: Commit**

```bash
git add src/esper/leyline/actions.py
git commit -m "$(cat <<'EOF'
refactor(leyline): delete flat Action enum

Per CLAUDE.md No Legacy Code Policy.
Use FactoredAction and LifecycleOp instead.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.3: Update vectorized.py to remove flat fallback

**Files:**

- Modify: `src/esper/simic/vectorized.py`

**Step 1: Remove flat mode branches**

Remove all `if factored: ... else: ...` branches, keeping only factored code.

**Step 2: Remove factored parameter**

Since factored is now the only mode, remove the parameter and always use factored logic.

**Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Fix any failures.

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "$(cat <<'EOF'
refactor(simic): remove flat mode from vectorized training

Factored action space is now the only mode.
Per CLAUDE.md No Legacy Code Policy.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.4: Update training.py (non-vectorized)

**Files:**

- Modify: `src/esper/simic/training.py`

**Step 1: Add factored support or remove flat references**

Check `training.py` for any flat action usage:

```bash
grep -n "is_germinate_action\|get_blueprint_from_action\|Action\." src/esper/simic/training.py
```

Update to use `FactoredAction` and `LifecycleOp`.

**Step 2: Run tests**

```bash
pytest tests/simic/test_training.py -v
```

**Step 3: Commit**

```bash
git add src/esper/simic/training.py
git commit -m "$(cat <<'EOF'
refactor(simic): update training.py to factored actions

Removes flat action references per No Legacy Code Policy.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.5: Clean up remaining flat references

**Step 1: Search for remaining references**

```bash
grep -r "compute_flat_action_mask\|Action\.\|flat.*mask" src/ tests/
grep -r "action_dim.*=.*7\|action_dim.*=.*8" src/ tests/
```

**Step 2: Fix any remaining references**

Update to factored equivalents.

**Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: ALL PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor: clean up remaining flat action references

Final cleanup for factored migration.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: Final Validation

### Task 6.1: Run full test suite

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS

### Task 6.2: Run type checking

```bash
python -m mypy src/esper --ignore-missing-imports
```

Expected: No new errors

### Task 6.3: Verify no legacy code remains

```bash
# Should return nothing
grep -r "backwards.compat\|legacy\|deprecated" src/
grep -r "ADVANCE" src/ tests/
grep -r "compute_flat_action_mask" src/ tests/
```

---

## Summary

This plan:

1. **Removes ADVANCE** - dead operation that shouldn't exist
2. **Adds FactoredRolloutBuffer** - with correct GAE reset at episode boundaries
3. **Wires up FactoredActorCritic** - with all PPO features (target KL, value clipping, etc.)
4. **Updates training pipeline** - vectorized.py with factored support
5. **Deletes flat mode entirely** - per CLAUDE.md No Legacy Code Policy

Key fixes from expert review:

- NamedTuple for FactoredTransition (not dataclass)
- GAE resets `next_advantage = 0.0` on done
- Pre-stack tensors once for batch efficiency
- Target KL early stopping
- Value clipping
- Adaptive entropy coefficient
- torch.compile for FactoredActorCritic
- Detach tensors in buffer to prevent gradient retention
