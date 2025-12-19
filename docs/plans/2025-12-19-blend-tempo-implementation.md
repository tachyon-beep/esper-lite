# Implementation Plan: Blend Tempo Lever

**Date:** 2025-12-19 (Revised after 2nd Go/No-Go review)
**Status:** Ready for Implementation - Final Approval Pending
**Phase:** 2.5 (Action Space Extension)
**Priority:** High
**Estimated Effort:** 3 days

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Executive Summary

Add "blend tempo" as a 5th action head in Tamiyo's factored action space, allowing the RL agent to choose how quickly a seed is integrated into the host network.

### Design Philosophy

> "We don't prescribe what the agent should learn - we give it levers and telemetry, then observe what emerges."

### Breaking Changes

- **Observation dimension increases** - Existing checkpoints will be incompatible
- **Action space expands** - Must train from scratch (not resume existing policy)

---

## Task 1: Update HEAD_NAMES Constant (CRITICAL BLOCKER)

**File:** `src/esper/leyline/__init__.py`

This is the **most critical change**. The PPO update loop iterates over `HEAD_NAMES` to compute per-head losses, entropy, and gradients. If tempo is not in this list, the tempo head will exist but **never receive gradients**.

```python
# Line ~134 - MUST UPDATE THIS
# Current:
HEAD_NAMES: tuple[str, ...] = ("slot", "blueprint", "blend", "op")

# New:
HEAD_NAMES: tuple[str, ...] = ("slot", "blueprint", "blend", "tempo", "op")
```

Also add to exports in `__all__`.

**Verification:**
```python
def test_head_names_includes_tempo():
    from esper.leyline import HEAD_NAMES
    assert "tempo" in HEAD_NAMES
    assert HEAD_NAMES.index("tempo") == 3  # Before "op"
```

---

## Task 2: Define TempoAction Enum

**File:** `src/esper/leyline/factored_actions.py`

Add the new enum following the pattern of `BlendAction`:

```python
from enum import IntEnum

class TempoAction(IntEnum):
    """Blending tempo selection.

    Controls how many epochs the alpha ramp takes during BLENDING stage.
    Selected at GERMINATE time, stored in SeedState, used by start_blending().

    Design rationale:
    - FAST: Rapid integration, quick signal, higher risk of instability
    - STANDARD: Current default (5 epochs), balanced approach
    - SLOW: Gradual integration, better stability assessment, longer investment
    """
    FAST = 0      # 3 epochs
    STANDARD = 1  # 5 epochs (current default)
    SLOW = 2      # 8 epochs


# Mapping from enum to actual epoch count
TEMPO_TO_EPOCHS: dict[TempoAction, int] = {
    TempoAction.FAST: 3,
    TempoAction.STANDARD: 5,
    TempoAction.SLOW: 8,
}

# Module-level constant for action space sizing (follows NUM_BLUEPRINTS pattern)
NUM_TEMPO: int = len(TempoAction)
```

Add to `__all__`:
```python
__all__ = [
    # ... existing exports ...
    "TempoAction",
    "TEMPO_TO_EPOCHS",
    "NUM_TEMPO",
]
```

**Verification:**
```python
def test_tempo_action_values():
    assert len(TempoAction) == 3
    assert NUM_TEMPO == 3
    assert TEMPO_TO_EPOCHS[TempoAction.STANDARD] == 5
    assert TEMPO_TO_EPOCHS[TempoAction.FAST] == 3
    assert TEMPO_TO_EPOCHS[TempoAction.SLOW] == 8
```

---

## Task 3: Extend FactoredAction Dataclass

**File:** `src/esper/leyline/factored_actions.py`

Extend `FactoredAction` to include tempo as a **typed enum** (not int).

**Note:** The `to_indices()` method does not exist in the current codebase and must be **ADDED** (not modified). The existing class only has `from_indices()`.

```python
@dataclass(frozen=True, slots=True)
class FactoredAction:
    """Factored action representation for multi-slot morphogenetic control.

    5 action heads:
    - slot_idx: Which slot to target (0-2)
    - blueprint: Which blueprint to germinate
    - blend: Which blending algorithm
    - tempo: How fast to blend (NEW)
    - op: Lifecycle operation
    """
    slot_idx: int
    blueprint: BlueprintAction
    blend: BlendAction
    tempo: TempoAction  # NEW - typed enum, not int
    op: LifecycleOp

    @classmethod
    def from_indices(
        cls,
        slot_idx: int,
        blueprint_idx: int,
        blend_idx: int,
        tempo_idx: int,  # NEW parameter
        op_idx: int,
    ) -> "FactoredAction":
        """Create from integer indices (used by network output)."""
        return cls(
            slot_idx=slot_idx,
            blueprint=BlueprintAction(blueprint_idx),
            blend=BlendAction(blend_idx),
            tempo=TempoAction(tempo_idx),  # NEW
            op=LifecycleOp(op_idx),
        )

    def to_indices(self) -> tuple[int, int, int, int, int]:
        """Convert to integer indices for network input."""
        return (
            self.slot_idx,
            self.blueprint.value,
            self.blend.value,
            self.tempo.value,  # NEW
            self.op.value,
        )
```

**Verification:**
```python
from hypothesis import given, strategies as st

@given(
    slot=st.integers(0, 2),
    blueprint=st.integers(0, NUM_BLUEPRINTS - 1),
    blend=st.integers(0, 2),
    tempo=st.integers(0, 2),
    op=st.integers(0, 3),
)
def test_factored_action_roundtrip(slot, blueprint, blend, tempo, op):
    """Factored action survives index conversion."""
    action = FactoredAction.from_indices(slot, blueprint, blend, tempo, op)
    indices = action.to_indices()
    assert indices == (slot, blueprint, blend, tempo, op)
```

---

## Task 4: Update Action Masking

**File:** `src/esper/tamiyo/policy/action_masks.py`

Add tempo mask computation and include in batch masks:

```python
from esper.leyline.factored_actions import NUM_TEMPO, TempoAction

def compute_tempo_mask(
    slot_states: list[SeedStateReport | None],
    selected_op: LifecycleOp | None = None,
) -> torch.Tensor:
    """Compute valid tempo options.

    All tempo values are valid when germinating. The tempo choice
    only takes effect when op == GERMINATE; for other ops the
    sampled tempo is ignored but we still allow all values to
    avoid masking complexity.

    Returns:
        Shape (NUM_TEMPO,) boolean mask, True = valid
    """
    return torch.ones(NUM_TEMPO, dtype=torch.bool)


def compute_action_masks(
    slot_states: list[SeedStateReport | None],
    ...
) -> dict[str, torch.Tensor]:
    """Compute all action masks for current state."""
    return {
        "slot": compute_slot_mask(slot_states),
        "blueprint": compute_blueprint_mask(slot_states, topology),
        "blend": compute_blend_mask(slot_states),
        "tempo": compute_tempo_mask(slot_states),  # NEW
        "op": compute_op_mask(slot_states),
    }


def compute_batch_masks(
    batch_slot_states: list[list[SeedStateReport | None]],
    ...
) -> dict[str, torch.Tensor]:
    """Compute masks for a batch of states."""
    masks = [compute_action_masks(states, ...) for states in batch_slot_states]
    return {
        "slot": torch.stack([m["slot"] for m in masks]),
        "blueprint": torch.stack([m["blueprint"] for m in masks]),
        "blend": torch.stack([m["blend"] for m in masks]),
        "tempo": torch.stack([m["tempo"] for m in masks]),  # NEW
        "op": torch.stack([m["op"] for m in masks]),
    }
```

---

## Task 5: Store Tempo in SeedState

**File:** `src/esper/kasmina/slot.py`

Add tempo storage to `SeedState` dataclass:

```python
@dataclass(kw_only=True, slots=True)
class SeedState:
    """Complete state of a seed through its lifecycle."""

    seed_id: str
    blueprint_id: str
    slot_id: str = ""

    stage: SeedStage = SeedStage.DORMANT
    previous_stage: SeedStage = SeedStage.UNKNOWN
    # ... existing fields ...

    # NEW: Tempo selection from germination
    blend_tempo_epochs: int = 5  # Default to STANDARD (5 epochs)

    # ... rest of existing fields ...

    def to_dict(self) -> dict:
        """Convert to primitive dict for serialization."""
        return {
            # ... existing fields ...
            "blend_tempo_epochs": self.blend_tempo_epochs,  # NEW
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeedState":
        """Reconstruct from primitive dict."""
        state = cls(
            seed_id=data["seed_id"],
            blueprint_id=data["blueprint_id"],
            # ... existing fields ...
        )
        # ... existing restoration ...
        state.blend_tempo_epochs = data.get("blend_tempo_epochs", 5)  # NEW
        return state
```

---

## Task 6: Pass Tempo to Germination

**File:** `src/esper/kasmina/slot.py`

Modify `SeedSlot.germinate()` to accept and store tempo:

```python
def germinate(
    self,
    blueprint_id: str,
    seed_id: str | None = None,
    host_module: nn.Module | None = None,
    blend_algorithm_id: str = "sigmoid",
    blend_tempo_epochs: int = 5,  # NEW parameter
) -> SeedState:
    """Germinate a new seed in this slot.

    Args:
        blueprint_id: Blueprint to instantiate (e.g., "norm", "attention")
        seed_id: Optional unique identifier for the seed
        host_module: Host network for gradient isolation (optional)
        blend_algorithm_id: Blending algorithm ("linear", "sigmoid", "gated")
        blend_tempo_epochs: Number of epochs for blending (3, 5, or 8)
    """
    # ... existing validation and seed creation ...

    # Store blend settings for use in start_blending()
    self._blend_algorithm_id = blend_algorithm_id
    self._blend_tempo_epochs = blend_tempo_epochs  # NEW

    # ... rest of existing germinate logic ...

    # After state creation, store tempo in state
    self.state.blend_tempo_epochs = blend_tempo_epochs  # NEW

    # ... telemetry emission, return ...
```

---

## Task 7: Use Tempo in start_blending()

**File:** `src/esper/kasmina/slot.py`

Modify `_on_enter_stage()` to use stored tempo:

```python
def _on_enter_stage(self, new_stage: SeedStage, old_stage: SeedStage) -> None:
    """Handle stage entry logic uniformly."""

    if new_stage == SeedStage.TRAINING and old_stage == SeedStage.GERMINATED:
        self.isolate_gradients = True

    elif new_stage == SeedStage.BLENDING and old_stage == SeedStage.TRAINING:
        # Topology-aware gradient isolation
        topology = self.task_config.topology if self.task_config else "cnn"
        self.isolate_gradients = (topology == "cnn")

        # Snapshot accuracy at blending start
        if self.state:
            self.state.metrics.accuracy_at_blending_start = self.state.metrics.current_val_accuracy
            self.state.metrics._blending_started = True

        # Use stored tempo instead of fixed default
        # Priority: stored tempo > TaskConfig > DEFAULT_BLENDING_TOTAL_STEPS
        total_steps = getattr(self, '_blend_tempo_epochs', None)
        if total_steps is None:
            total_steps = DEFAULT_BLENDING_TOTAL_STEPS
            if self.task_config is not None:
                configured_steps = self.task_config.blending_steps
                if isinstance(configured_steps, int) and configured_steps > 0:
                    total_steps = configured_steps

        self.start_blending(total_steps=total_steps)

    # ... rest of existing stage handling ...
```

---

## Task 8: Update Network Architecture

**File:** `src/esper/simic/agent/network.py`

This is the most complex change. Update `FactoredRecurrentActorCritic`:

### 8.1 Update _ForwardOutput TypedDict

```python
class _ForwardOutput(TypedDict):
    """Type hints for forward() output."""
    slot_logits: torch.Tensor
    blueprint_logits: torch.Tensor
    blend_logits: torch.Tensor
    tempo_logits: torch.Tensor  # NEW
    op_logits: torch.Tensor
    value: torch.Tensor
    hidden: tuple[torch.Tensor, torch.Tensor]
```

### 8.2 Add tempo_head to __init__

```python
class FactoredRecurrentActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_slots: int = 3,
        num_blueprints: int = NUM_BLUEPRINTS,
        num_blend: int = 3,
        num_tempo: int = NUM_TEMPO,  # NEW parameter
        num_ops: int = 4,
    ):
        super().__init__()
        # ... existing initialization ...

        # Action heads
        self.slot_head = nn.Linear(hidden_dim, num_slots)
        self.blueprint_head = nn.Linear(hidden_dim, num_blueprints)
        self.blend_head = nn.Linear(hidden_dim, num_blend)
        self.tempo_head = nn.Linear(hidden_dim, num_tempo)  # NEW
        self.op_head = nn.Linear(hidden_dim, num_ops)

        # ... rest of init ...
```

### 8.3 Update forward() method

```python
def forward(
    self,
    obs: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    masks: dict[str, torch.Tensor] | None = None,
) -> _ForwardOutput:
    """Forward pass returning all head logits."""
    # ... existing feature extraction and LSTM ...

    # Compute logits for each head
    slot_logits = self.slot_head(lstm_out)
    blueprint_logits = self.blueprint_head(lstm_out)
    blend_logits = self.blend_head(lstm_out)
    tempo_logits = self.tempo_head(lstm_out)  # NEW
    op_logits = self.op_head(lstm_out)

    # Apply masks if provided
    if masks is not None:
        slot_logits = self._apply_mask(slot_logits, masks.get("slot"))
        blueprint_logits = self._apply_mask(blueprint_logits, masks.get("blueprint"))
        blend_logits = self._apply_mask(blend_logits, masks.get("blend"))
        tempo_logits = self._apply_mask(tempo_logits, masks.get("tempo"))  # NEW
        op_logits = self._apply_mask(op_logits, masks.get("op"))

    return {
        "slot_logits": slot_logits,
        "blueprint_logits": blueprint_logits,
        "blend_logits": blend_logits,
        "tempo_logits": tempo_logits,  # NEW
        "op_logits": op_logits,
        "value": value,
        "hidden": new_hidden,
    }
```

### 8.4 Update get_action() method

```python
def get_action(
    self,
    obs: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    masks: dict[str, torch.Tensor] | None = None,
    deterministic: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, tuple]:
    """Sample actions from policy."""
    outputs = self.forward(obs, hidden, masks)

    # Sample from each head
    slot_dist = Categorical(logits=outputs["slot_logits"])
    blueprint_dist = Categorical(logits=outputs["blueprint_logits"])
    blend_dist = Categorical(logits=outputs["blend_logits"])
    tempo_dist = Categorical(logits=outputs["tempo_logits"])  # NEW
    op_dist = Categorical(logits=outputs["op_logits"])

    if deterministic:
        slot = slot_dist.probs.argmax(dim=-1)
        blueprint = blueprint_dist.probs.argmax(dim=-1)
        blend = blend_dist.probs.argmax(dim=-1)
        tempo = tempo_dist.probs.argmax(dim=-1)  # NEW
        op = op_dist.probs.argmax(dim=-1)
    else:
        slot = slot_dist.sample()
        blueprint = blueprint_dist.sample()
        blend = blend_dist.sample()
        tempo = tempo_dist.sample()  # NEW
        op = op_dist.sample()

    actions = {
        "slot": slot,
        "blueprint": blueprint,
        "blend": blend,
        "tempo": tempo,  # NEW
        "op": op,
    }

    log_probs = {
        "slot": slot_dist.log_prob(slot),
        "blueprint": blueprint_dist.log_prob(blueprint),
        "blend": blend_dist.log_prob(blend),
        "tempo": tempo_dist.log_prob(tempo),  # NEW
        "op": op_dist.log_prob(op),
    }

    return actions, log_probs, outputs["value"], outputs["hidden"]
```

### 8.5 Update evaluate_actions() method

```python
def evaluate_actions(
    self,
    obs: torch.Tensor,
    actions: dict[str, torch.Tensor],
    hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    masks: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Evaluate log probs and entropy for given actions."""
    outputs = self.forward(obs, hidden, masks)

    slot_dist = Categorical(logits=outputs["slot_logits"])
    blueprint_dist = Categorical(logits=outputs["blueprint_logits"])
    blend_dist = Categorical(logits=outputs["blend_logits"])
    tempo_dist = Categorical(logits=outputs["tempo_logits"])  # NEW
    op_dist = Categorical(logits=outputs["op_logits"])

    log_probs = {
        "slot": slot_dist.log_prob(actions["slot"]),
        "blueprint": blueprint_dist.log_prob(actions["blueprint"]),
        "blend": blend_dist.log_prob(actions["blend"]),
        "tempo": tempo_dist.log_prob(actions["tempo"]),  # NEW
        "op": op_dist.log_prob(actions["op"]),
    }

    entropies = {
        "slot": slot_dist.entropy(),
        "blueprint": blueprint_dist.entropy(),
        "blend": blend_dist.entropy(),
        "tempo": tempo_dist.entropy(),  # NEW
        "op": op_dist.entropy(),
    }

    return log_probs, entropies, outputs["value"]
```

---

## Task 9: Update TamiyoRolloutBuffer

**File:** `src/esper/simic/agent/rollout_buffer.py`

Add tempo storage to the rollout buffer:

```python
@dataclass
class TamiyoRolloutBuffer:
    """Stores rollout data for PPO updates."""

    # ... existing fields ...

    # Actions per head
    slot_actions: torch.Tensor
    blueprint_actions: torch.Tensor
    blend_actions: torch.Tensor
    tempo_actions: torch.Tensor  # NEW
    op_actions: torch.Tensor

    # Log probs per head
    slot_log_probs: torch.Tensor
    blueprint_log_probs: torch.Tensor
    blend_log_probs: torch.Tensor
    tempo_log_probs: torch.Tensor  # NEW
    op_log_probs: torch.Tensor

    @classmethod
    def create(cls, num_steps: int, num_envs: int, state_dim: int, device: torch.device):
        """Create empty buffer with pre-allocated tensors."""
        return cls(
            # ... existing allocations ...
            tempo_actions=torch.zeros(num_steps, num_envs, dtype=torch.long, device=device),
            tempo_log_probs=torch.zeros(num_steps, num_envs, device=device),
        )

    def add_step(
        self,
        step: int,
        obs: torch.Tensor,
        actions: dict[str, torch.Tensor],
        log_probs: dict[str, torch.Tensor],
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ):
        """Add a single step to the buffer."""
        # ... existing storage ...
        self.tempo_actions[step] = actions["tempo"]  # NEW
        self.tempo_log_probs[step] = log_probs["tempo"]  # NEW

    def get_batched_sequences(self, ...):
        """Get batched sequences for training."""
        return {
            # ... existing fields ...
            "tempo_actions": self.tempo_actions[indices],  # NEW
            "tempo_log_probs": self.tempo_log_probs[indices],  # NEW
        }
```

---

## Task 10: Update PPO Causal Head Masking and Entropy Defaults

**Files:**
- `src/esper/simic/agent/ppo.py`
- `src/esper/simic/agent/advantages.py`

Add tempo to the causal head masking in the `update()` method. Tempo is only relevant when GERMINATE is selected (same as blueprint and blend).

### 10.1 Update head_masks in PPO.update()

```python
def update(self, buffer: TamiyoRolloutBuffer) -> dict[str, float]:
    """PPO update with per-head advantage masking."""

    # ... existing setup ...

    # Causal head masks: which heads are relevant for each timestep
    # - op: always relevant (agent always chooses an operation)
    # - slot: relevant when not WAIT
    # - blueprint: only relevant when GERMINATE
    # - blend: only relevant when GERMINATE
    # - tempo: only relevant when GERMINATE (NEW)
    is_wait = (batch["op_actions"] == LifecycleOp.WAIT.value)
    is_germinate = (batch["op_actions"] == LifecycleOp.GERMINATE.value)

    head_masks = {
        "op": torch.ones_like(is_wait, dtype=torch.float),
        "slot": (~is_wait).float(),
        "blueprint": is_germinate.float(),
        "blend": is_germinate.float(),
        "tempo": is_germinate.float(),  # NEW - same as blueprint/blend
    }

    # ... rest of update using head_masks for advantage weighting ...
```

### 10.2 Update compute_per_head_advantages() in advantages.py

**File:** `src/esper/simic/agent/advantages.py`

The `compute_per_head_advantages()` function must also return tempo advantages with the germinate-only mask:

```python
def compute_per_head_advantages(
    advantages: torch.Tensor,
    op_actions: torch.Tensor,
    ...
) -> dict[str, torch.Tensor]:
    """Compute per-head advantages with causal masking."""

    is_germinate = (op_actions == LifecycleOp.GERMINATE.value)
    is_wait = (op_actions == LifecycleOp.WAIT.value)

    return {
        "op": advantages,  # Always gets full advantages
        "slot": torch.where(~is_wait, advantages, torch.zeros_like(advantages)),
        "blueprint": torch.where(is_germinate, advantages, torch.zeros_like(advantages)),
        "blend": torch.where(is_germinate, advantages, torch.zeros_like(advantages)),
        "tempo": torch.where(is_germinate, advantages, torch.zeros_like(advantages)),  # NEW
    }
```

### 10.3 Update entropy_coef_per_head default in PPOAgent.__init__

**File:** `src/esper/simic/agent/ppo.py` (line ~264-269)

The default entropy coefficients must include tempo:

```python
# Current:
self.entropy_coef_per_head = entropy_coef_per_head or {
    "slot": 1.0,
    "blueprint": 1.0,
    "blend": 1.0,
    "op": 1.0,
}

# New:
self.entropy_coef_per_head = entropy_coef_per_head or {
    "slot": 1.0,
    "blueprint": 1.0,
    "blend": 1.0,
    "tempo": 1.0,  # NEW
    "op": 1.0,
}
```

---

## Task 11: Update Vectorized Environment and Host Layer

**Files:**
- `src/esper/simic/training/vectorized.py`
- `src/esper/kasmina/host.py`

Pass tempo through the action execution chain. **Note:** The vectorized environment calls `Host.germinate_seed()`, not `SeedSlot.germinate()` directly. Both layers must be updated.

### 11.1 Add imports to vectorized.py

```python
from esper.leyline.factored_actions import (
    FactoredAction,
    BlueprintAction,
    BlendAction,
    TempoAction,  # NEW
    LifecycleOp,
    TEMPO_TO_EPOCHS,  # NEW
)
```

### 11.2 Extract tempo from action indices

**Important:** The plan previously referenced `_unpack_action()` and `_execute_germinate()` methods that do not exist. The actual code is **inline** within the training loop at approximately **line 2082** of `vectorized.py`.

Find the action unpacking near line ~1905 where `FactoredAction.from_indices()` is called, and add the tempo parameter:

```python
# Current (4 parameters):
action = FactoredAction.from_indices(slot_idx, blueprint_idx, blend_idx, op_idx)

# New (5 parameters):
action = FactoredAction.from_indices(slot_idx, blueprint_idx, blend_idx, tempo_idx, op_idx)
```

### 11.3 Update Host.germinate_seed() (MISSING FROM ORIGINAL PLAN)

**File:** `src/esper/kasmina/host.py`

The `Host` class wraps `SeedSlot.germinate()`. It must accept and forward `blend_tempo_epochs`:

```python
def germinate_seed(
    self,
    blueprint_id: str,
    seed_id: str | None = None,
    slot: SeedSlot | int | None = None,
    blend_algorithm_id: str = "sigmoid",
    blend_tempo_epochs: int = 5,  # NEW parameter
) -> SeedState:
    """Germinate a seed in the specified slot.

    Args:
        ...
        blend_tempo_epochs: Number of epochs for blending (3, 5, or 8)
    """
    target_slot = self._resolve_slot(slot)

    return target_slot.germinate(
        blueprint_id=blueprint_id,
        seed_id=seed_id,
        host_module=self.model,
        blend_algorithm_id=blend_algorithm_id,
        blend_tempo_epochs=blend_tempo_epochs,  # NEW - forward to SeedSlot
    )
```

### 11.4 Pass tempo in germinate execution

Find the inline germinate execution near **line 2085-2090** in vectorized.py and update:

```python
# Current:
model.germinate_seed(
    blueprint_id,
    seed_id,
    slot=target_slot,
    blend_algorithm_id=blend_algorithm_id,
)

# New:
tempo_epochs = TEMPO_TO_EPOCHS[TempoAction(tempo_idx)]
model.germinate_seed(
    blueprint_id,
    seed_id,
    slot=target_slot,
    blend_algorithm_id=blend_algorithm_id,
    blend_tempo_epochs=tempo_epochs,  # NEW
)
```

---

## Task 12: Add Telemetry for Tempo

**File:** `src/esper/leyline/telemetry.py`

Add tempo-related telemetry fields to `SeedTelemetry`:

```python
@dataclass
class SeedTelemetry:
    """Telemetry data for a single seed."""

    # ... existing fields ...

    # Tempo telemetry (NEW)
    blend_tempo_epochs: int = 5
    blending_velocity: float = 0.0  # d(alpha) / d(epoch) - how fast blend progresses
```

**File:** `src/esper/kasmina/slot.py`

Compute blending velocity during BLENDING stage:

```python
def sync_telemetry(self, ...):
    """Sync telemetry from metrics + gradient signals."""
    # ... existing sync logic ...

    # NEW: Compute blending velocity during BLENDING
    if self.state and self.state.stage == SeedStage.BLENDING:
        self.telemetry.blend_tempo_epochs = self.state.blend_tempo_epochs
        epochs_in_blend = self.state.metrics.epochs_in_current_stage
        if epochs_in_blend > 0:
            self.telemetry.blending_velocity = self.state.alpha / epochs_in_blend
        else:
            self.telemetry.blending_velocity = 0.0
```

---

## Task 13: Add Tempo to Observation Features

**File:** `src/esper/tamiyo/policy/features.py`

Add tempo to per-slot observation features.

**Note:** The codebase uses `SLOT_FEATURE_SIZE` (not `SLOT_FEATURE_DIM`). Match existing conventions.

### 13.1 Update feature dimension constant

```python
# Current (line ~70):
SLOT_FEATURE_SIZE = 17

# New (increment by 1 for tempo):
SLOT_FEATURE_SIZE = 18  # +1 for tempo
```

### 13.2 Add tempo to feature extraction

```python
def _extract_slot_features(self, slot_state: SeedStateReport | None) -> torch.Tensor:
    """Extract features for a single slot."""
    if slot_state is None:
        # Empty slot: zeros
        return torch.zeros(SLOT_FEATURE_SIZE, dtype=torch.float32)

    features = []

    # Stage (normalized to 0-1)
    features.append(slot_state.stage.value / 10.0)

    # Alpha
    features.append(slot_state.metrics.current_alpha)

    # Improvement (normalized)
    features.append(slot_state.metrics.improvement_since_stage_start_normalized)

    # Blueprint one-hot
    blueprint_onehot = torch.zeros(NUM_BLUEPRINTS)
    # ... existing blueprint encoding ...
    features.extend(blueprint_onehot.tolist())

    # NEW: Tempo (normalized to 0-1, max is ~12 epochs)
    tempo_normalized = getattr(slot_state, 'blend_tempo_epochs', 5) / 12.0
    features.append(tempo_normalized)

    return torch.tensor(features, dtype=torch.float32)
```

### 13.3 Update get_feature_size()

```python
def get_feature_size(num_slots: int = 3) -> int:
    """Get total observation feature size."""
    base_features = 23  # ... existing base features
    slot_features = num_slots * SLOT_FEATURE_DIM
    return base_features + slot_features
```

**IMPORTANT:** This changes the observation dimension. Document that existing checkpoints will be incompatible.

---

## Task 14: Property Tests

**File:** `tests/leyline/test_factored_actions.py`

```python
import pytest
from hypothesis import given, strategies as st

from esper.leyline.factored_actions import (
    FactoredAction,
    BlueprintAction,
    BlendAction,
    TempoAction,
    LifecycleOp,
    NUM_BLUEPRINTS,
    NUM_TEMPO,
)


def test_tempo_action_enum():
    """TempoAction has expected values."""
    assert len(TempoAction) == 3
    assert TempoAction.FAST.value == 0
    assert TempoAction.STANDARD.value == 1
    assert TempoAction.SLOW.value == 2


def test_num_tempo_constant():
    """NUM_TEMPO matches enum length."""
    assert NUM_TEMPO == len(TempoAction)


@given(
    slot=st.integers(0, 2),
    blueprint=st.integers(0, NUM_BLUEPRINTS - 1),
    blend=st.integers(0, 2),
    tempo=st.integers(0, NUM_TEMPO - 1),
    op=st.integers(0, 3),
)
def test_factored_action_roundtrip(slot, blueprint, blend, tempo, op):
    """FactoredAction survives index conversion."""
    action = FactoredAction.from_indices(slot, blueprint, blend, tempo, op)
    indices = action.to_indices()
    assert indices == (slot, blueprint, blend, tempo, op)

    # Verify types
    assert isinstance(action.blueprint, BlueprintAction)
    assert isinstance(action.blend, BlendAction)
    assert isinstance(action.tempo, TempoAction)
    assert isinstance(action.op, LifecycleOp)
```

---

## Task 15: Integration Test

**File:** `tests/integration/test_tempo_lever.py`

```python
import pytest
import torch

from esper.kasmina.slot import SeedSlot, SeedStage
from esper.leyline.factored_actions import TempoAction, TEMPO_TO_EPOCHS


@pytest.fixture
def slot():
    """Create a test slot."""
    return SeedSlot(
        slot_id="test_slot",
        channels=64,
        device="cpu",
    )


def test_tempo_affects_blending_duration(slot):
    """Verify FAST tempo leads to shorter blending than SLOW."""
    # Germinate with FAST tempo (3 epochs)
    slot.germinate(
        blueprint_id="norm",
        seed_id="fast_seed",
        blend_tempo_epochs=TEMPO_TO_EPOCHS[TempoAction.FAST],
    )
    assert slot.state.blend_tempo_epochs == 3

    # Advance to TRAINING
    slot.step_epoch()
    assert slot.state.stage == SeedStage.TRAINING

    # Record accuracy to pass G2 gate
    for _ in range(3):
        slot.state.metrics.record_accuracy(50.0 + _)
        slot.state.metrics.seed_gradient_norm_ratio = 0.5
        slot.step_epoch()

    # Should transition to BLENDING
    assert slot.state.stage == SeedStage.BLENDING
    assert slot.state.blending_steps_total == 3  # FAST tempo


def test_tempo_stored_in_state(slot):
    """Verify tempo is persisted in SeedState."""
    slot.germinate(
        blueprint_id="norm",
        blend_tempo_epochs=8,  # SLOW
    )

    # Check state storage
    assert slot.state.blend_tempo_epochs == 8

    # Check serialization roundtrip
    state_dict = slot.state.to_dict()
    assert state_dict["blend_tempo_epochs"] == 8


def test_head_names_includes_tempo():
    """Verify HEAD_NAMES was updated (critical for PPO training)."""
    from esper.leyline import HEAD_NAMES
    assert "tempo" in HEAD_NAMES, "CRITICAL: tempo missing from HEAD_NAMES, PPO will not train tempo head!"
```

---

## Verification Checklist

After implementation, verify:

### Critical (Task 1 + 8 - must be atomic)
- [ ] `HEAD_NAMES` includes `"tempo"` (CRITICAL - test this first!)
- [ ] Network has `tempo_head` linear layer (must exist if HEAD_NAMES includes tempo)

### Enum and Constants (Tasks 2-3)
- [ ] `TempoAction` enum has 3 values (FAST, STANDARD, SLOW)
- [ ] `TEMPO_TO_EPOCHS` maps enum to epoch counts correctly
- [ ] `NUM_TEMPO` equals 3
- [ ] `FactoredAction` includes `tempo: TempoAction` field
- [ ] `from_indices()` accepts 5 parameters (includes tempo)
- [ ] `to_indices()` method ADDED (did not exist before) and returns 5-tuple

### Action Masking (Task 4)
- [ ] `compute_action_masks()` returns tempo mask
- [ ] `compute_batch_masks()` stacks tempo masks

### Seed State (Tasks 5-7)
- [ ] `SeedState.blend_tempo_epochs` field exists
- [ ] `SeedSlot.germinate()` accepts `blend_tempo_epochs` parameter
- [ ] `_on_enter_stage()` uses stored tempo for blending

### Network (Task 8)
- [ ] `_ForwardOutput` TypedDict includes `tempo_logits`
- [ ] `forward()` computes and returns tempo_logits
- [ ] `get_action()` samples tempo and returns in actions dict
- [ ] `evaluate_actions()` computes tempo log_prob and entropy

### Rollout Buffer (Task 9)
- [ ] `TamiyoRolloutBuffer` stores `tempo_actions` and `tempo_log_probs`
- [ ] `add_step()` accepts tempo action and log_prob
- [ ] `get_batched_sequences()` returns tempo fields

### PPO Training (Task 10)
- [ ] PPO `head_masks` includes tempo (masked same as blueprint/blend)
- [ ] `compute_per_head_advantages()` returns tempo advantages
- [ ] `entropy_coef_per_head` default includes `"tempo": 1.0`

### Vectorized Env + Host (Task 11)
- [ ] `Host.germinate_seed()` accepts `blend_tempo_epochs` parameter
- [ ] Host forwards `blend_tempo_epochs` to SeedSlot.germinate()
- [ ] Vectorized env extracts tempo from action indices
- [ ] Vectorized env passes tempo to Host.germinate_seed()

### Telemetry and Features (Tasks 12-13)
- [ ] `SeedTelemetry` includes `blend_tempo_epochs` and `blending_velocity`
- [ ] `SLOT_FEATURE_SIZE` incremented (17 → 18)
- [ ] Observation features include tempo (normalized)

### Tests (Tasks 14-15)
- [ ] Property tests pass (test_factored_actions.py)
- [ ] Integration tests pass (test_tempo_lever.py)
- [ ] torch.compile still works (no graph breaks)

---

## Training Recommendations

### Train From Scratch

**Do NOT attempt to resume from existing checkpoints.** Reasons:
1. Observation dimension changed (LSTM learned different correlations)
2. New tempo head has random weights (distributional mismatch)
3. 500-2000 episodes is tractable (few hours of training)

### Hyperparameters

Keep existing hyperparameters. The tempo head has 3 options, so:
- Max entropy: log(3) ≈ 1.1 nats
- Existing entropy coefficient (0.05) provides adequate exploration
- Existing entropy floor (0.02) prevents premature convergence

### Monitoring

After training, verify:
1. **Tempo entropy decreases** - Should drop from ~1.1 to lower value over 500 episodes
2. **All tempos used** - Check tempo selection is not collapsed to single value
3. **No accuracy regression** - Terminal accuracy should match or exceed baseline
4. **Blueprint-tempo correlations** - Look for patterns (e.g., depthwise → STANDARD, conv_heavy → SLOW)

---

## Rollback Plan

If tempo lever causes training instability:

### Quick Fix (Disable Without Removing Code)
```python
# In vectorized.py, force STANDARD tempo
tempo_epochs = 5  # Override agent choice with STANDARD
```

### Full Rollback
```bash
git revert <commit-hash>
```

No database migrations or external dependencies affected.

---

## Checkpoint Incompatibility Notice

**Breaking Change:** This implementation changes the observation dimension.

Existing checkpoints saved before this change will fail to load with:
```
RuntimeError: size mismatch for feature_net.0.weight
```

**Migration path:** None. Train new agents from scratch.

---

---

## Revision History

| Revision | Date | Changes |
|----------|------|---------|
| 1.0 | 2025-12-19 | Initial plan (13 tasks) |
| 2.0 | 2025-12-19 | First Go/No-Go review: HEAD_NAMES blocker, file paths corrected, rollout buffer + causal masking added (15 tasks) |
| 2.1 | 2025-12-19 | Second Go/No-Go review: Task 3 clarified (to_indices ADDED), Task 10 expanded (advantages.py + entropy defaults), Task 11 fixed (Host layer added, inline code noted), Task 13 fixed (SLOT_FEATURE_SIZE constant name) |

*Implementation plan revised after 2nd Go/No-Go review by PyTorch Engineering and DRL specialists. All caveats from both experts addressed.*
