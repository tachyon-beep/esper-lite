# Track 4: Simic Training

**Priority:** High (blocks training validation)
**Estimated Effort:** 1-2 days
**Dependencies:** Track 1 (L3, L7), Track 2 (K3)

## Overview

Simic owns the training loop execution. This track wires internal ops into the vectorized trainer and adds intervention costs to prevent thrashing.

---

## S1: Execute Internal Ops in Vectorized Trainer

**File:** `src/esper/simic/training/vectorized.py`

### ActionResult Definition (per Python specialist review)

Define the `ActionResult` dataclass in Leyline (or Simic types module):

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True, slots=True)
class ActionResult:
    """Result of executing a lifecycle op.

    Attributes:
        success: Whether the op executed successfully (False if no-op due to boundary)
        op: The lifecycle op that was executed
        slot_id: The slot this op targeted
        env_idx: The environment index
    """
    success: bool
    op: LifecycleOp = LifecycleOp.NOOP
    slot_id: Optional[str] = None
    env_idx: Optional[int] = None
```

### Specification

Extend the action execution dispatch to handle internal ops:

```python
def execute_action(
    self,
    env_idx: int,
    slot_id: str,
    action: FactoredAction,
) -> ActionResult:
    """Execute a factored action in the environment.

    Args:
        env_idx: Environment index
        slot_id: Target slot ID
        action: The factored action to execute

    Returns:
        ActionResult with success flag and any side effects
    """
    op = action.lifecycle_op

    # Existing ops
    if op == LifecycleOp.NOOP:
        return ActionResult(success=True)
    elif op == LifecycleOp.GERMINATE:
        return self._execute_germinate(env_idx, slot_id, action)
    elif op == LifecycleOp.PRUNE:
        return self._execute_prune(env_idx, slot_id)
    elif op == LifecycleOp.FOSSILIZE:
        return self._execute_fossilize(env_idx, slot_id)

    # Internal ops (Phase 0)
    elif op == LifecycleOp.GROW_INTERNAL:
        return self._execute_grow_internal(env_idx, slot_id)
    elif op == LifecycleOp.SHRINK_INTERNAL:
        return self._execute_shrink_internal(env_idx, slot_id)

    else:
        raise ValueError(f"Unknown lifecycle op: {op}")


def _execute_grow_internal(
    self,
    env_idx: int,
    slot_id: str,
) -> ActionResult:
    """Execute GROW_INTERNAL op.

    Increases internal level by 1, up to max_level.
    """
    slot = self._get_slot(env_idx, slot_id)
    success = slot.grow_internal()

    return ActionResult(
        success=success,
        op=LifecycleOp.GROW_INTERNAL,
        slot_id=slot_id,
        env_idx=env_idx,
    )


def _execute_shrink_internal(
    self,
    env_idx: int,
    slot_id: str,
) -> ActionResult:
    """Execute SHRINK_INTERNAL op.

    Decreases internal level by 1, down to 0.
    """
    slot = self._get_slot(env_idx, slot_id)
    success = slot.shrink_internal()

    return ActionResult(
        success=success,
        op=LifecycleOp.SHRINK_INTERNAL,
        slot_id=slot_id,
        env_idx=env_idx,
    )
```

### Vectorized Batch Execution

For batched execution across environments:

```python
def execute_actions_batch(
    self,
    actions: list[tuple[int, str, FactoredAction]],
) -> list[ActionResult]:
    """Execute a batch of actions across environments.

    Args:
        actions: List of (env_idx, slot_id, action) tuples

    Returns:
        List of ActionResults in same order
    """
    results = []
    for env_idx, slot_id, action in actions:
        result = self.execute_action(env_idx, slot_id, action)
        results.append(result)
    return results
```

### Acceptance Criteria
- [ ] `GROW_INTERNAL` dispatches to `slot.grow_internal()`
- [ ] `SHRINK_INTERNAL` dispatches to `slot.shrink_internal()`
- [ ] ActionResult includes op type and success flag
- [ ] Batch execution handles internal ops correctly
- [ ] Telemetry events emitted by slot (not trainer)

---

## S2: Add Intervention Costs for Internal Ops

**File:** `src/esper/simic/rewards/rewards.py`

### Config-Based Costs (per DRL and PyTorch specialist reviews)

Add intervention costs to `ContributionRewardConfig` for tunability:

```python
@dataclass
class ContributionRewardConfig:
    """Configuration for contribution-based rewards."""

    # ... existing fields ...

    # Intervention costs (Phase 0)
    # Set to -0.005 to match SET_ALPHA_TARGET cost hierarchy (per DRL specialist review)
    cost_grow_internal: float = 0.005
    cost_shrink_internal: float = 0.005
```

### Specification

Add small intervention costs to prevent thrashing:

```python
# Intervention cost constants - derived from config
def get_intervention_costs(config: ContributionRewardConfig) -> dict[LifecycleOp, float]:
    """Build intervention cost dict from config (per DRL specialist review)."""
    return {
        LifecycleOp.NOOP: 0.0,
        LifecycleOp.GERMINATE: 0.01,
        LifecycleOp.PRUNE: 0.005,
        LifecycleOp.FOSSILIZE: 0.002,
        # Internal ops (Phase 0) - match SET_ALPHA_TARGET cost hierarchy
        LifecycleOp.GROW_INTERNAL: config.cost_grow_internal,
        LifecycleOp.SHRINK_INTERNAL: config.cost_shrink_internal,
    }


def compute_intervention_cost(op: LifecycleOp) -> float:
    """Get the intervention cost for a lifecycle op.

    These costs discourage unnecessary interventions while being
    small enough not to prevent beneficial ones.

    Returns:
        Negative reward (cost) for the intervention
    """
    return -INTERVENTION_COSTS.get(op, 0.0)


def compute_reward(
    # ... existing params ...
    action_result: ActionResult,
) -> float:
    """Compute reward for a timestep.

    Includes:
    - Primary reward (ROI, loss improvement, etc.)
    - Intervention cost (small penalty for actions)
    """
    primary_reward = compute_primary_reward(...)

    # Intervention cost (negative)
    intervention_cost = compute_intervention_cost(action_result.op)

    return primary_reward + intervention_cost
```

### Cost Calibration Rationale (per DRL specialist review)

| Op | Cost | Rationale |
|----|------|-----------|
| `NOOP` | 0.0 | No cost for waiting |
| `GERMINATE` | 0.01 | Highest cost - creates new module |
| `PRUNE` | 0.005 | Moderate - irreversible removal |
| `FOSSILIZE` | 0.002 | Low - just locks in place |
| `SET_ALPHA_TARGET` | 0.005 | Reference point for tuning costs |
| `GROW_INTERNAL` | 0.005 | Match SET_ALPHA_TARGET (same reversibility) |
| `SHRINK_INTERNAL` | 0.005 | Match SET_ALPHA_TARGET (same reversibility) |

**Key principle:** Internal ops should match the existing cost hierarchy. SET_ALPHA_TARGET is the appropriate reference point since both are "adjustment" operations (reversible, local effect). Using -0.001 was too cheap relative to the existing cost structure and would encourage over-intervention.

### Anti-Thrash Detection

The telemetry system (Track 5) monitors for thrash patterns:

```python
# In reward computation, optionally penalize recent thrash
def compute_thrash_penalty(
    recent_ops: list[LifecycleOp],
    window_size: int = 10,
) -> float:
    """Compute penalty for oscillating internal level.

    Detects GROW->SHRINK->GROW->SHRINK patterns.
    """
    if len(recent_ops) < 4:
        return 0.0

    # Count direction changes in recent window
    recent = recent_ops[-window_size:]
    internal_ops = [
        op for op in recent
        if op in {LifecycleOp.GROW_INTERNAL, LifecycleOp.SHRINK_INTERNAL}
    ]

    # Need at least 2 ops to compute reversal rate (per Python specialist review)
    # This also guards against ZeroDivisionError when len(internal_ops) == 1
    if len(internal_ops) < 4:
        return 0.0

    # Count reversals
    reversals = sum(
        1 for i in range(1, len(internal_ops))
        if internal_ops[i] != internal_ops[i-1]
    )

    # Penalize high reversal rate (division safe: len >= 4 implies len - 1 >= 3 > 0)
    reversal_rate = reversals / (len(internal_ops) - 1)
    if reversal_rate > 0.7:  # More than 70% reversals = thrash
        return -0.005 * reversal_rate

    return 0.0
```

### Acceptance Criteria
- [ ] `GROW_INTERNAL` has intervention cost 0.005 (default, per DRL specialist review)
- [ ] `SHRINK_INTERNAL` has intervention cost 0.005 (default, per DRL specialist review)
- [ ] **Costs added to `ContributionRewardConfig`** (tunable via config, per DRL/PyTorch specialist reviews)
- [ ] Costs included in reward computation via `get_intervention_costs(config)`
- [ ] Thrash penalty available but optional (off by default in Phase 0)
- [ ] Thrash penalty division-safe (per Python specialist review)

---

## S3: Update Rollout Buffer State Dim Assertions

**File:** `src/esper/simic/agent/rollout_buffer.py`

### Specification

Update buffer allocation to use derived dimensions:

```python
from esper.leyline import (
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_SLOT_FEATURE_SIZE,
    NUM_SLOTS,
    NUM_OPS,
)


class RolloutBuffer:
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        base_feature_size: int = OBS_V3_BASE_FEATURE_SIZE,
        slot_feature_size: int = OBS_V3_SLOT_FEATURE_SIZE,
        num_slots: int = NUM_SLOTS,
        num_ops: int = NUM_OPS,
        device: torch.device = torch.device("cpu"),
    ):
        # Validate dims match Leyline
        assert base_feature_size == OBS_V3_BASE_FEATURE_SIZE, (
            f"base_feature_size mismatch: got {base_feature_size}, "
            f"expected {OBS_V3_BASE_FEATURE_SIZE} from Leyline"
        )
        assert slot_feature_size == OBS_V3_SLOT_FEATURE_SIZE, (
            f"slot_feature_size mismatch: got {slot_feature_size}, "
            f"expected {OBS_V3_SLOT_FEATURE_SIZE} from Leyline"
        )
        assert num_ops == NUM_OPS, (
            f"num_ops mismatch: got {num_ops}, expected {NUM_OPS} from Leyline"
        )

        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device

        # Allocate buffers using derived dims
        self.base_features = torch.zeros(
            buffer_size, num_envs, base_feature_size,
            device=device, dtype=torch.float32,
        )
        self.slot_features = torch.zeros(
            buffer_size, num_envs, num_slots, slot_feature_size,
            device=device, dtype=torch.float32,
        )
        self.action_masks = torch.zeros(
            buffer_size, num_envs, num_slots, num_ops,
            device=device, dtype=torch.float32,
        )
        # ... other buffers ...
```

### Shape Validation on Add

```python
def add(
    self,
    base_features: torch.Tensor,
    slot_features: torch.Tensor,
    action_masks: torch.Tensor,
    # ... other tensors ...
) -> None:
    """Add a timestep to the buffer.

    Validates shapes and device before adding.
    """
    # Device validation (per Python specialist review)
    if base_features.device != self.device:
        raise ValueError(
            f"base_features device {base_features.device} != buffer device {self.device}"
        )
    if slot_features.device != self.device:
        raise ValueError(
            f"slot_features device {slot_features.device} != buffer device {self.device}"
        )
    if action_masks.device != self.device:
        raise ValueError(
            f"action_masks device {action_masks.device} != buffer device {self.device}"
        )

    # Shape validation
    assert base_features.shape == (self.num_envs, OBS_V3_BASE_FEATURE_SIZE), (
        f"base_features shape {base_features.shape} != "
        f"expected ({self.num_envs}, {OBS_V3_BASE_FEATURE_SIZE})"
    )
    assert slot_features.shape == (
        self.num_envs, NUM_SLOTS, OBS_V3_SLOT_FEATURE_SIZE
    ), (
        f"slot_features shape {slot_features.shape} != "
        f"expected ({self.num_envs}, {NUM_SLOTS}, {OBS_V3_SLOT_FEATURE_SIZE})"
    )
    assert action_masks.shape == (self.num_envs, NUM_SLOTS, NUM_OPS), (
        f"action_masks shape {action_masks.shape} != "
        f"expected ({self.num_envs}, {NUM_SLOTS}, {NUM_OPS})"
    )

    # Store in buffer
    self.base_features[self.pos] = base_features
    self.slot_features[self.pos] = slot_features
    self.action_masks[self.pos] = action_masks
    # ...

    self.pos = (self.pos + 1) % self.buffer_size
```

### NUM_OPS Dependency Note (per PyTorch specialist review)

This track depends on **L3 (Add `GROW_INTERNAL`, `SHRINK_INTERNAL` to `LifecycleOp`)** being completed first.
After L3, `NUM_OPS` will increase from 4 to 6. The `action_masks` buffer dimension must match.

Verify with assertion in buffer initialization:
```python
# Sanity check: NUM_OPS includes internal ops (per PyTorch specialist review)
assert LifecycleOp.GROW_INTERNAL.value < NUM_OPS, (
    f"LifecycleOp.GROW_INTERNAL ({LifecycleOp.GROW_INTERNAL.value}) >= NUM_OPS ({NUM_OPS})"
)
assert LifecycleOp.SHRINK_INTERNAL.value < NUM_OPS, (
    f"LifecycleOp.SHRINK_INTERNAL ({LifecycleOp.SHRINK_INTERNAL.value}) >= NUM_OPS ({NUM_OPS})"
)
```

### Acceptance Criteria
- [ ] Buffer uses derived dims from Leyline
- [ ] Init assertions verify dims match Leyline
- [ ] **NUM_OPS validated to include GROW_INTERNAL/SHRINK_INTERNAL** (per PyTorch specialist review)
- [ ] `add()` validates tensor shapes
- [ ] **`add()` validates tensor devices** (per Python specialist review)
- [ ] No hardcoded dimension constants
- [ ] Tests verify shape validation catches mismatches
- [ ] Tests verify device validation catches mismatches

---

## Testing Requirements

### Unit Tests (`tests/simic/`)

**test_vectorized.py:**
```python
def test_execute_grow_internal():
    """Verify GROW_INTERNAL dispatches correctly."""
    trainer = create_test_trainer()
    slot = trainer._get_slot(0, "r0c0")
    initial_level = slot.internal_level

    action = FactoredAction(lifecycle_op=LifecycleOp.GROW_INTERNAL)
    result = trainer.execute_action(0, "r0c0", action)

    assert result.success
    assert result.op == LifecycleOp.GROW_INTERNAL
    assert slot.internal_level == initial_level + 1

def test_execute_shrink_internal():
    """Verify SHRINK_INTERNAL dispatches correctly."""
    trainer = create_test_trainer()
    slot = trainer._get_slot(0, "r0c0")
    slot.set_internal_level(2)  # Start at level 2

    action = FactoredAction(lifecycle_op=LifecycleOp.SHRINK_INTERNAL)
    result = trainer.execute_action(0, "r0c0", action)

    assert result.success
    assert result.op == LifecycleOp.SHRINK_INTERNAL
    assert slot.internal_level == 1

def test_grow_at_max_returns_false():
    """Verify GROW_INTERNAL at max level returns success=False."""
    trainer = create_test_trainer()
    slot = trainer._get_slot(0, "r0c0")
    slot.set_internal_level(slot.internal_max_level)

    action = FactoredAction(lifecycle_op=LifecycleOp.GROW_INTERNAL)
    result = trainer.execute_action(0, "r0c0", action)

    assert not result.success
```

**test_rewards.py:**
```python
def test_internal_op_intervention_costs():
    """Verify internal ops have correct intervention costs (per DRL specialist review)."""
    config = ContributionRewardConfig()  # Use defaults
    costs = get_intervention_costs(config)
    assert costs[LifecycleOp.GROW_INTERNAL] == 0.005
    assert costs[LifecycleOp.SHRINK_INTERNAL] == 0.005

def test_intervention_costs_applied_to_reward():
    """Verify intervention costs are included in reward."""
    result = ActionResult(
        success=True,
        op=LifecycleOp.GROW_INTERNAL,
    )
    reward = compute_reward(
        # ... other args ...
        action_result=result,
    )
    # Reward should include -0.001 cost component
    assert reward < compute_reward(..., action_result=ActionResult(op=LifecycleOp.NOOP))

def test_thrash_penalty_detects_oscillation():
    """Verify thrash penalty triggers on GROW/SHRINK oscillation."""
    ops = [
        LifecycleOp.GROW_INTERNAL,
        LifecycleOp.SHRINK_INTERNAL,
        LifecycleOp.GROW_INTERNAL,
        LifecycleOp.SHRINK_INTERNAL,
        LifecycleOp.GROW_INTERNAL,
        LifecycleOp.SHRINK_INTERNAL,
    ]
    penalty = compute_thrash_penalty(ops)
    assert penalty < 0  # Should be negative (penalty)
```

**test_rollout_buffer.py:**
```python
def test_buffer_uses_leyline_dims():
    """Verify buffer allocates with Leyline dims."""
    buffer = RolloutBuffer(buffer_size=100, num_envs=4)

    assert buffer.base_features.shape == (100, 4, OBS_V3_BASE_FEATURE_SIZE)
    assert buffer.slot_features.shape == (100, 4, NUM_SLOTS, OBS_V3_SLOT_FEATURE_SIZE)
    assert buffer.action_masks.shape == (100, 4, NUM_SLOTS, NUM_OPS)

def test_add_validates_shapes():
    """Verify add() rejects wrong shapes."""
    buffer = RolloutBuffer(buffer_size=100, num_envs=4)

    wrong_base = torch.randn(4, OBS_V3_BASE_FEATURE_SIZE + 1)
    with pytest.raises(AssertionError):
        buffer.add(
            base_features=wrong_base,
            # ... other tensors with correct shapes ...
        )

def test_dim_mismatch_at_init():
    """Verify init rejects mismatched dims."""
    with pytest.raises(AssertionError):
        RolloutBuffer(
            buffer_size=100,
            num_envs=4,
            base_feature_size=OBS_V3_BASE_FEATURE_SIZE + 1,  # Wrong!
        )

def test_device_mismatch_rejected():
    """Verify add() rejects tensors on wrong device (per Python specialist review)."""
    buffer = RolloutBuffer(buffer_size=100, num_envs=4, device=torch.device("cpu"))

    # Create tensor on different device (if CUDA available)
    if torch.cuda.is_available():
        wrong_device_tensor = torch.randn(4, OBS_V3_BASE_FEATURE_SIZE, device="cuda")
        with pytest.raises(ValueError, match="device"):
            buffer.add(
                base_features=wrong_device_tensor,
                # ... other tensors on correct device ...
            )
```
