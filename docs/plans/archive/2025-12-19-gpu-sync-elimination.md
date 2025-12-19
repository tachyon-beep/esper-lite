# GPU Synchronization Elimination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate 825+ GPU-CPU synchronizations per episode by batching tensor operations and deferring `.item()` calls.

**Architecture:** Replace per-element `.item()` calls with batched tensor transfers. Actions and log_probs are transferred to CPU once per epoch, then indexed. Telemetry stats computed as tensor ops before extraction. Bootstrap values computed in parallel for all truncated envs.

**Tech Stack:** PyTorch tensor operations, existing `TamiyoRolloutBuffer`, vectorized training loop.

---

## Problem Summary

| Location | Issue | Syncs/Epoch |
|----------|-------|-------------|
| `vectorized.py:1757-1762` | `.item()` per action per env | 16 |
| `vectorized.py:2142-2145` | `.item()` per log_prob per env | 16 |
| `vectorized.py:1996-1999` | `.item()` per mask stat per env | 16 (telemetry) |
| `vectorized.py:2121` | `.item()` for bootstrap value | 4 (at truncation) |

**Total:** ~33+ syncs/epoch × 25 epochs = **825+ syncs/episode**

---

## Task 1: Batch Action Tensor Transfer

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1756-1762`

**Step 1: Write the failing test**

Create `tests/simic/training/test_batch_action_extraction.py`:

```python
"""Test batched action extraction eliminates per-element .item() calls."""
import torch
from unittest.mock import patch, MagicMock

def test_batch_action_extraction_single_cpu_call():
    """Verify actions are extracted via single .cpu() not per-element .item()."""
    # Create mock action tensors on "GPU" (actually CPU but we track calls)
    actions_dict = {
        "slot": torch.tensor([0, 1, 2, 3]),
        "blueprint": torch.tensor([5, 6, 7, 8]),
        "blend": torch.tensor([0, 1, 0, 1]),
        "op": torch.tensor([1, 2, 0, 3]),
    }

    # Track .item() calls - should NOT be called
    item_call_count = 0
    original_item = torch.Tensor.item
    def counting_item(self):
        nonlocal item_call_count
        item_call_count += 1
        return original_item(self)

    # Batch extraction pattern (what we want)
    with patch.object(torch.Tensor, 'item', counting_item):
        # CORRECT: Single .cpu() per head, then index with Python
        actions_cpu = {key: actions_dict[key].cpu().numpy() for key in actions_dict}
        actions = [
            {key: int(actions_cpu[key][i]) for key in actions_cpu}
            for i in range(4)
        ]

    # Verify no .item() calls were made
    assert item_call_count == 0, f"Expected 0 .item() calls, got {item_call_count}"

    # Verify correct values extracted
    assert actions[0] == {"slot": 0, "blueprint": 5, "blend": 0, "op": 1}
    assert actions[3] == {"slot": 3, "blueprint": 8, "blend": 1, "op": 3}


def test_batch_action_extraction_values_match_item():
    """Verify batched extraction produces same values as .item() approach."""
    actions_dict = {
        "slot": torch.tensor([0, 1, 2, 3]),
        "blueprint": torch.tensor([5, 6, 7, 8]),
        "blend": torch.tensor([0, 1, 0, 1]),
        "op": torch.tensor([1, 2, 0, 3]),
    }

    # Old approach (what we're replacing)
    old_actions = [
        {key: actions_dict[key][i].item() for key in actions_dict}
        for i in range(4)
    ]

    # New approach
    actions_cpu = {key: actions_dict[key].cpu().numpy() for key in actions_dict}
    new_actions = [
        {key: int(actions_cpu[key][i]) for key in actions_cpu}
        for i in range(4)
    ]

    assert old_actions == new_actions
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_batch_action_extraction.py -v
```

Expected: PASS (this is a unit test for the pattern, not the integration)

**Step 3: Modify vectorized.py to use batched extraction**

Replace lines 1756-1762:

```python
# OLD (33+ syncs):
# actions = [
#     {key: actions_dict[key][i].item() for key in HEAD_NAMES}
#     for i in range(len(env_states))
# ]

# NEW (1 sync per head = 4 syncs total):
# Transfer all actions to CPU in single call per head, then index
actions_cpu = {key: actions_dict[key].cpu().numpy() for key in HEAD_NAMES}
actions = [
    {key: int(actions_cpu[key][i]) for key in HEAD_NAMES}
    for i in range(len(env_states))
]
```

**Step 4: Run integration test to verify no regression**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_vectorized_integration.py -v -k "action" --tb=short
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py tests/simic/training/test_batch_action_extraction.py
git commit -m "perf(simic): batch action tensor transfer to eliminate .item() syncs

Replace per-element .item() calls with single .cpu().numpy() per head.
Reduces GPU syncs from 16/epoch to 4/epoch for action extraction.
"
```

---

## Task 2: Batch Value Tensor Transfer

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1762`

**Step 1: Verify current pattern**

Current code:
```python
values = values_tensor.tolist()
```

This is actually already efficient - `.tolist()` is a single transfer. No change needed.

**Step 2: Document verification**

Add comment to clarify this is intentional:

```python
# Single CPU transfer - .tolist() is efficient (no per-element sync)
values = values_tensor.tolist()
```

**Step 3: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "docs(simic): clarify values_tensor.tolist() is efficient"
```

---

## Task 3: Batch Log Prob Storage in Buffer

**Files:**
- Modify: `src/esper/simic/agent/rollout_buffer.py:206-266`
- Modify: `src/esper/simic/training/vectorized.py:2135-2157`
- Create: `tests/simic/agent/test_buffer_batch_logprobs.py`

**Step 1: Write the failing test**

```python
"""Test buffer can accept batched log_prob tensors."""
import torch
from esper.simic.agent.rollout_buffer import TamiyoRolloutBuffer
from esper.leyline.slot_config import SlotConfig


def test_buffer_add_accepts_tensor_log_probs():
    """Buffer.add() should accept tensor log_probs, not just floats."""
    buffer = TamiyoRolloutBuffer(
        num_envs=4,
        max_steps_per_env=25,
        state_dim=64,
        device=torch.device("cpu"),
    )

    state = torch.randn(64)
    slot_mask = torch.ones(5, dtype=torch.bool)
    blueprint_mask = torch.ones(13, dtype=torch.bool)
    blend_mask = torch.ones(4, dtype=torch.bool)
    op_mask = torch.ones(4, dtype=torch.bool)
    hidden_h = torch.randn(1, 1, 128)
    hidden_c = torch.randn(1, 1, 128)

    # Should accept tensors (0-dim) for log_probs
    buffer.add(
        env_id=0,
        state=state,
        slot_action=0,
        blueprint_action=1,
        blend_action=0,
        op_action=1,
        slot_log_prob=torch.tensor(-0.5),      # tensor, not float
        blueprint_log_prob=torch.tensor(-1.0),
        blend_log_prob=torch.tensor(-0.3),
        op_log_prob=torch.tensor(-0.7),
        value=0.5,
        reward=1.0,
        done=False,
        slot_mask=slot_mask,
        blueprint_mask=blueprint_mask,
        blend_mask=blend_mask,
        op_mask=op_mask,
        hidden_h=hidden_h,
        hidden_c=hidden_c,
    )

    # Verify stored correctly
    assert buffer.slot_log_probs[0, 0].item() == -0.5
    assert buffer.blueprint_log_probs[0, 0].item() == -1.0


def test_buffer_add_still_accepts_float_log_probs():
    """Buffer.add() should still accept float log_probs for backwards compat."""
    buffer = TamiyoRolloutBuffer(
        num_envs=4,
        max_steps_per_env=25,
        state_dim=64,
        device=torch.device("cpu"),
    )

    state = torch.randn(64)
    slot_mask = torch.ones(5, dtype=torch.bool)
    blueprint_mask = torch.ones(13, dtype=torch.bool)
    blend_mask = torch.ones(4, dtype=torch.bool)
    op_mask = torch.ones(4, dtype=torch.bool)
    hidden_h = torch.randn(1, 1, 128)
    hidden_c = torch.randn(1, 1, 128)

    # Should still accept floats
    buffer.add(
        env_id=0,
        state=state,
        slot_action=0,
        blueprint_action=1,
        blend_action=0,
        op_action=1,
        slot_log_prob=-0.5,      # float
        blueprint_log_prob=-1.0,
        blend_log_prob=-0.3,
        op_log_prob=-0.7,
        value=0.5,
        reward=1.0,
        done=False,
        slot_mask=slot_mask,
        blueprint_mask=blueprint_mask,
        blend_mask=blend_mask,
        op_mask=op_mask,
        hidden_h=hidden_h,
        hidden_c=hidden_c,
    )

    assert buffer.slot_log_probs[0, 0].item() == -0.5
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=src uv run pytest tests/simic/agent/test_buffer_batch_logprobs.py -v
```

Expected: May pass if torch auto-converts, but we need to verify tensor handling.

**Step 3: Update buffer type hints to accept Union[float, Tensor]**

Modify `rollout_buffer.py` lines 206-229:

```python
def add(
    self,
    env_id: int,
    state: torch.Tensor,
    slot_action: int,
    blueprint_action: int,
    blend_action: int,
    op_action: int,
    slot_log_prob: float | torch.Tensor,      # Accept both
    blueprint_log_prob: float | torch.Tensor,
    blend_log_prob: float | torch.Tensor,
    op_log_prob: float | torch.Tensor,
    value: float | torch.Tensor,              # Accept both
    reward: float,
    done: bool,
    slot_mask: torch.Tensor,
    blueprint_mask: torch.Tensor,
    blend_mask: torch.Tensor,
    op_mask: torch.Tensor,
    hidden_h: torch.Tensor,
    hidden_c: torch.Tensor,
    truncated: bool = False,
    bootstrap_value: float | torch.Tensor = 0.0,
) -> None:
```

And update the assignment lines 248-252 to handle both:

```python
# Handle tensor or float log_probs (tensor assignment works for both)
self.slot_log_probs[env_id, step_idx] = slot_log_prob
self.blueprint_log_probs[env_id, step_idx] = blueprint_log_prob
self.blend_log_probs[env_id, step_idx] = blend_log_prob
self.op_log_probs[env_id, step_idx] = op_log_prob
self.values[env_id, step_idx] = value
```

**Step 4: Update vectorized.py to pass tensors directly**

Replace lines 2142-2145:

```python
# OLD (16 syncs):
# slot_log_prob=head_log_probs["slot"][env_idx].item(),
# ...

# NEW (0 syncs - tensor assignment):
slot_log_prob=head_log_probs["slot"][env_idx],
blueprint_log_prob=head_log_probs["blueprint"][env_idx],
blend_log_prob=head_log_probs["blend"][env_idx],
op_log_prob=head_log_probs["op"][env_idx],
```

**Step 5: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/agent/test_buffer_batch_logprobs.py tests/simic/agent/test_rollout_buffer.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/agent/rollout_buffer.py src/esper/simic/training/vectorized.py tests/simic/agent/test_buffer_batch_logprobs.py
git commit -m "perf(simic): pass log_prob tensors directly to buffer

Buffer.add() now accepts torch.Tensor for log_probs, avoiding 16 .item()
GPU syncs per epoch. Tensor assignment handles both float and tensor.
"
```

---

## Task 4: Batch Telemetry Mask Stats

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1995-2004`
- Create: `tests/simic/training/test_batch_mask_stats.py`

**Step 1: Write the failing test**

```python
"""Test batched mask stat computation."""
import torch

def test_batch_mask_stats_no_item_in_loop():
    """Mask stats should be computed as batch ops, not per-env .item()."""
    HEAD_NAMES = ("slot", "blueprint", "blend", "op")
    num_envs = 4

    # Simulate masks_batch from training
    masks_batch = {
        "slot": torch.tensor([
            [True, True, False, False, False],
            [True, False, False, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]),
        "blueprint": torch.ones(4, 13, dtype=torch.bool),
        "blend": torch.ones(4, 4, dtype=torch.bool),
        "op": torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
            [True, False, False, False],
            [True, True, True, True],
        ]),
    }

    # BATCHED approach: compute all at once, then transfer
    # "masked" = not all True = some actions disabled
    masked_batch = {
        key: ~masks_batch[key].all(dim=-1)  # [num_envs] bool
        for key in HEAD_NAMES
    }
    # Single CPU transfer
    masked_cpu = {key: masked_batch[key].cpu().numpy() for key in HEAD_NAMES}

    # Now extract per-env (no GPU sync)
    for env_idx in range(num_envs):
        masked_flags = {key: bool(masked_cpu[key][env_idx]) for key in HEAD_NAMES}

        if env_idx == 0:
            assert masked_flags["slot"] is True  # Not all True
            assert masked_flags["blueprint"] is False  # All True
            assert masked_flags["op"] is True  # Not all True
        elif env_idx == 2:
            assert masked_flags["slot"] is False  # All True
            assert masked_flags["op"] is True  # Not all True


def test_batch_mask_stats_values_match_item_approach():
    """Batched approach must produce same values as .item() approach."""
    HEAD_NAMES = ("slot", "blueprint", "blend", "op")

    masks_batch = {
        "slot": torch.tensor([
            [True, True, False, False, False],
            [True, False, False, False, False],
        ]),
        "blueprint": torch.ones(2, 13, dtype=torch.bool),
        "blend": torch.ones(2, 4, dtype=torch.bool),
        "op": torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
        ]),
    }

    # Old approach (per-env .item())
    old_results = []
    for env_idx in range(2):
        masked_flags = {
            key: not bool(masks_batch[key][env_idx].all().item())
            for key in HEAD_NAMES
        }
        old_results.append(masked_flags)

    # New approach (batched)
    masked_batch = {key: ~masks_batch[key].all(dim=-1) for key in HEAD_NAMES}
    masked_cpu = {key: masked_batch[key].cpu().numpy() for key in HEAD_NAMES}
    new_results = []
    for env_idx in range(2):
        masked_flags = {key: bool(masked_cpu[key][env_idx]) for key in HEAD_NAMES}
        new_results.append(masked_flags)

    assert old_results == new_results
```

**Step 2: Run test**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_batch_mask_stats.py -v
```

Expected: PASS

**Step 3: Update vectorized.py to batch mask stats**

Before the per-env loop (around line 1766), add:

```python
# Batch compute mask stats for telemetry (eliminates 16 .item() syncs)
# "masked" means not all actions are valid for this head
if hub and use_telemetry:
    masked_batch = {
        key: ~masks_batch[key].all(dim=-1)  # [num_envs] bool tensor
        for key in HEAD_NAMES
    }
    masked_cpu = {key: masked_batch[key].cpu().numpy() for key in HEAD_NAMES}
else:
    masked_cpu = None
```

Then replace lines 1995-2000:

```python
# OLD (16 syncs):
# masked_flags = {
#     "slot": not bool(masks_batch["slot"][env_idx].all().item()),
#     ...
# }

# NEW (0 syncs - already on CPU):
masked_flags = {key: bool(masked_cpu[key][env_idx]) for key in HEAD_NAMES}
```

**Step 4: Run integration tests**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/ -v -k "telemetry or mask" --tb=short
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/vectorized.py tests/simic/training/test_batch_mask_stats.py
git commit -m "perf(simic): batch telemetry mask stat computation

Compute mask stats as batch tensor ops before per-env loop, then transfer
to CPU once. Eliminates 16 .item() GPU syncs per epoch when telemetry enabled.
"
```

---

## Task 5: Batch Bootstrap Value Computation

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:2046-2121`
- Create: `tests/simic/training/test_batch_bootstrap.py`

**Step 1: Analyze current pattern**

Current code computes bootstrap values per-env in the loop:
- For each truncated env (all of them at epoch 25):
  - Build post-action features
  - Create tensor
  - Forward pass through critic
  - `.item()` to extract value

This is 4 forward passes + 4 syncs instead of 1 forward pass + 1 sync.

**Step 2: Write the failing test**

```python
"""Test batched bootstrap value computation."""
import torch
from unittest.mock import MagicMock, patch


def test_batch_bootstrap_single_forward_pass():
    """Bootstrap values should use single batched forward pass."""
    num_envs = 4

    # Mock network that tracks forward calls
    mock_network = MagicMock()
    forward_call_count = 0

    def mock_get_action(state, hidden, **kwargs):
        nonlocal forward_call_count
        forward_call_count += 1
        batch_size = state.shape[0]
        actions = {key: torch.zeros(batch_size, dtype=torch.long) for key in ["slot", "blueprint", "blend", "op"]}
        log_probs = {key: torch.zeros(batch_size) for key in ["slot", "blueprint", "blend", "op"]}
        values = torch.randn(batch_size)  # Random values for each env
        return actions, log_probs, values, hidden

    mock_network.get_action = mock_get_action

    # Simulate batched bootstrap computation
    states = torch.randn(num_envs, 64)  # All envs' post-action states
    hidden = (torch.randn(1, num_envs, 128), torch.randn(1, num_envs, 128))

    # Single forward pass for all envs
    with torch.inference_mode():
        _, _, bootstrap_values, _ = mock_network.get_action(
            states, hidden=hidden,
            slot_mask=torch.ones(num_envs, 5, dtype=torch.bool),
            blueprint_mask=torch.ones(num_envs, 13, dtype=torch.bool),
            blend_mask=torch.ones(num_envs, 4, dtype=torch.bool),
            op_mask=torch.ones(num_envs, 4, dtype=torch.bool),
        )

    assert forward_call_count == 1, f"Expected 1 forward pass, got {forward_call_count}"
    assert bootstrap_values.shape == (num_envs,)


def test_batch_bootstrap_values_correct():
    """Batched bootstrap should produce same values as per-env approach."""
    # This test verifies mathematical equivalence
    torch.manual_seed(42)

    # Simulate simple linear critic for reproducibility
    critic_weight = torch.randn(1, 64)
    critic_bias = torch.randn(1)

    def simple_critic(state):
        """state: [batch, 64] -> [batch]"""
        return (state @ critic_weight.T + critic_bias).squeeze(-1)

    states = torch.randn(4, 64)

    # Per-env approach
    per_env_values = []
    for i in range(4):
        val = simple_critic(states[i:i+1])
        per_env_values.append(val.item())

    # Batched approach
    batched_values = simple_critic(states).tolist()

    for i in range(4):
        assert abs(per_env_values[i] - batched_values[i]) < 1e-5
```

**Step 3: Run test**

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_batch_bootstrap.py -v
```

Expected: PASS

**Step 4: Refactor bootstrap computation to batch**

This is a more complex refactor. The challenge is that we need to:
1. Collect post-action features for ALL envs first
2. Then do a single batched forward pass
3. Then distribute values back to per-env storage

Current structure (simplified):
```python
for env_idx, env_state in enumerate(env_states):
    # ... action execution ...
    if truncated:
        # Build features for this env
        post_action_features = ...
        post_action_state = torch.tensor([post_action_features], ...)
        # Forward pass for this env
        _, _, bootstrap_tensor, _ = agent.network.get_action(post_action_state, ...)
        bootstrap_value = bootstrap_tensor[0].item()
    # Store in buffer
    agent.buffer.add(..., bootstrap_value=bootstrap_value)
```

New structure:
```python
# PHASE 1: Execute actions and collect post-action states
bootstrap_features = []
bootstrap_hiddens_h = []
bootstrap_hiddens_c = []
bootstrap_masks = []

for env_idx, env_state in enumerate(env_states):
    # ... action execution (unchanged) ...
    if truncated:
        # Collect but don't compute yet
        post_action_features = ...
        bootstrap_features.append(post_action_features)
        bootstrap_hiddens_h.append(env_state.lstm_hidden[0])
        bootstrap_hiddens_c.append(env_state.lstm_hidden[1])
        bootstrap_masks.append(compute_action_masks(...))

# PHASE 2: Batched bootstrap forward pass
if bootstrap_features:
    bootstrap_states = torch.tensor(bootstrap_features, device=device)
    bootstrap_states_norm = obs_normalizer.normalize(bootstrap_states)
    bootstrap_hidden = (
        torch.cat(bootstrap_hiddens_h, dim=1),  # [layers, batch, hidden]
        torch.cat(bootstrap_hiddens_c, dim=1),
    )
    bootstrap_masks_batch = {
        key: torch.stack([m[key] for m in bootstrap_masks])
        for key in HEAD_NAMES
    }

    with torch.inference_mode():
        _, _, bootstrap_values_tensor, _ = agent.network.get_action(
            bootstrap_states_norm,
            hidden=bootstrap_hidden,
            **{f"{k}_mask": bootstrap_masks_batch[k] for k in HEAD_NAMES},
        )
    bootstrap_values_list = bootstrap_values_tensor.tolist()
else:
    bootstrap_values_list = []

# PHASE 3: Store transitions (now with pre-computed bootstrap values)
bootstrap_idx = 0
for env_idx, env_state in enumerate(env_states):
    if truncated:
        bootstrap_value = bootstrap_values_list[bootstrap_idx]
        bootstrap_idx += 1
    else:
        bootstrap_value = 0.0
    agent.buffer.add(..., bootstrap_value=bootstrap_value)
```

This is a significant refactor that changes the loop structure. Given complexity, create helper function:

Add to vectorized.py (new function before `train_ppo_vectorized`):

```python
def _compute_batched_bootstrap_values(
    agent,
    env_states: list,
    post_action_data: list[dict],  # Pre-collected features, hidden, masks
    obs_normalizer,
    device: str,
) -> list[float]:
    """Compute bootstrap values for all truncated envs in single forward pass.

    Args:
        agent: PPO agent with network
        env_states: List of environment states
        post_action_data: List of dicts with keys:
            - features: list[float] - post-action observation features
            - hidden: tuple[Tensor, Tensor] - LSTM hidden state
            - masks: dict[str, Tensor] - action masks for each head
        obs_normalizer: Observation normalizer
        device: Device string

    Returns:
        List of bootstrap values (one per entry in post_action_data)
    """
    if not post_action_data:
        return []

    # Stack all features
    features_batch = torch.tensor(
        [d["features"] for d in post_action_data],
        dtype=torch.float32,
        device=device,
    )
    features_normalized = obs_normalizer.normalize(features_batch)

    # Stack hidden states: each is [layers, 1, hidden_dim], need [layers, batch, hidden_dim]
    hidden_h = torch.cat([d["hidden"][0] for d in post_action_data], dim=1)
    hidden_c = torch.cat([d["hidden"][1] for d in post_action_data], dim=1)

    # Stack masks
    masks_batch = {
        key: torch.stack([d["masks"][key] for d in post_action_data])
        for key in ("slot", "blueprint", "blend", "op")
    }

    # Single forward pass
    with torch.inference_mode():
        _, _, values_tensor, _ = agent.network.get_action(
            features_normalized,
            hidden=(hidden_h, hidden_c),
            slot_mask=masks_batch["slot"],
            blueprint_mask=masks_batch["blueprint"],
            blend_mask=masks_batch["blend"],
            op_mask=masks_batch["op"],
            deterministic=True,
        )

    return values_tensor.tolist()
```

**Step 5: Refactor the main loop**

This requires restructuring the loop. Given the complexity and risk, this should be done carefully with full test coverage.

The refactor splits the current single loop into:
1. Action execution loop (collect bootstrap data)
2. Batched bootstrap computation
3. Buffer storage loop

Due to complexity, provide detailed line-by-line changes in implementation.

**Step 6: Run full test suite**

```bash
PYTHONPATH=src uv run pytest tests/simic/ -v --tb=short
```

Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/simic/training/vectorized.py tests/simic/training/test_batch_bootstrap.py
git commit -m "perf(simic): batch bootstrap value computation

Collect post-action data for all truncated envs, then compute bootstrap
values in single batched forward pass. Reduces from N forward passes +
N GPU syncs to 1 forward pass + 1 sync at episode end.

Adds _compute_batched_bootstrap_values() helper for clarity.
"
```

---

## Task 6: Pass Value as Tensor to Buffer

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:2146`

**Step 1: Current code uses float**

```python
value=value,  # This is already a Python float from values[env_idx]
```

The `values_tensor.tolist()` already efficiently converts, so this is fine. However, we should verify the buffer accepts tensors for consistency.

**Step 2: Verify buffer handles both**

Already done in Task 3 - buffer accepts `float | torch.Tensor` for value.

**Step 3: No changes needed**

The current `.tolist()` is efficient. Document this.

---

## Task 7: Verification and Profiling

**Files:**
- Create: `scripts/profile_gpu_sync.py`

**Step 1: Create profiling script**

```python
#!/usr/bin/env python3
"""Profile GPU synchronization in vectorized training.

Run with: PYTHONPATH=src uv run python scripts/profile_gpu_sync.py
"""
import torch
import time
from contextlib import contextmanager


@contextmanager
def sync_timer(name: str):
    """Time a block including GPU sync."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")


def profile_action_extraction():
    """Compare .item() vs .cpu().numpy() for action extraction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    num_epochs = 25
    num_heads = 4

    actions_dict = {
        f"head_{i}": torch.randint(0, 10, (num_envs,), device=device)
        for i in range(num_heads)
    }

    # Method 1: Per-element .item()
    with sync_timer("Per-element .item()"):
        for _ in range(num_epochs):
            for env_idx in range(num_envs):
                _ = {key: actions_dict[key][env_idx].item() for key in actions_dict}

    # Method 2: Batched .cpu().numpy()
    with sync_timer("Batched .cpu().numpy()"):
        for _ in range(num_epochs):
            actions_cpu = {key: actions_dict[key].cpu().numpy() for key in actions_dict}
            for env_idx in range(num_envs):
                _ = {key: int(actions_cpu[key][env_idx]) for key in actions_cpu}


def profile_log_prob_storage():
    """Compare .item() vs tensor assignment for log_probs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs = 4
    num_epochs = 25
    num_heads = 4

    log_probs = {
        f"head_{i}": torch.randn(num_envs, device=device)
        for i in range(num_heads)
    }

    # Target buffer (pre-allocated)
    buffer = torch.zeros(num_envs, num_epochs, num_heads, device=device)

    # Method 1: Per-element .item()
    with sync_timer("Log probs .item()"):
        for epoch in range(num_epochs):
            for env_idx in range(num_envs):
                for head_idx, key in enumerate(log_probs):
                    buffer[env_idx, epoch, head_idx] = log_probs[key][env_idx].item()

    # Method 2: Tensor assignment
    buffer2 = torch.zeros(num_envs, num_epochs, num_heads, device=device)
    with sync_timer("Log probs tensor assign"):
        for epoch in range(num_epochs):
            for env_idx in range(num_envs):
                for head_idx, key in enumerate(log_probs):
                    buffer2[env_idx, epoch, head_idx] = log_probs[key][env_idx]


if __name__ == "__main__":
    print("=== GPU Sync Profiling ===\n")

    print("Action Extraction:")
    profile_action_extraction()

    print("\nLog Prob Storage:")
    profile_log_prob_storage()

    print("\nDone!")
```

**Step 2: Run profiling before changes**

```bash
PYTHONPATH=src uv run python scripts/profile_gpu_sync.py
```

Document baseline timings.

**Step 3: Run profiling after changes**

After implementing all tasks, re-run to verify improvement.

**Step 4: Commit**

```bash
git add scripts/profile_gpu_sync.py
git commit -m "test(simic): add GPU sync profiling script"
```

---

## Summary

| Task | Syncs Eliminated | Complexity |
|------|------------------|------------|
| 1: Batch action transfer | 12/epoch | Low |
| 2: Value transfer | 0 (already efficient) | None |
| 3: Batch log_prob storage | 16/epoch | Medium |
| 4: Batch mask stats | 16/epoch (telemetry) | Low |
| 5: Batch bootstrap | 4/episode | High |
| 6: Value tensor | 0 (already efficient) | None |
| 7: Profiling | N/A | Low |

**Total eliminated:** ~28 syncs/epoch × 25 epochs = **700+ syncs/episode**

**Remaining syncs per episode:**
- 4 syncs for batched action transfer (one per head)
- 4 syncs for batched mask stats
- 1 sync for batched bootstrap values
- ~10 syncs for metrics during PPO update

**Expected improvement:** GPU utilization should increase from ~0% to measurable levels as the hot path no longer blocks on synchronization.
