# Track 2: Kasmina Mechanics

**Priority:** High (blocks policy and training integration)
**Estimated Effort:** 2-3 days
**Dependencies:** Track 1 (Leyline contracts)

## Overview

Kasmina owns the seed lifecycle mechanics. This track implements the `conv_ladder` blueprint and internal ops execution in `SeedSlot`.

---

## K1: Implement `conv_ladder` Blueprint

**File:** `src/esper/kasmina/blueprints/cnn.py`

### Specification

```python
@BlueprintRegistry.register(
    "conv_ladder",
    "cnn",
    param_estimate=4000,  # At max level; varies with level
    description="Microstructured CNN ladder with internal level control"
)
def create_conv_ladder_seed(dim: int, **kwargs: Any) -> nn.Module:
    """Create a conv ladder seed.

    The ladder has L=4 micro-blocks. `internal_level` controls how many
    are active (have requires_grad=True and contribute to output).

    Level 0 = identity (present but no contribution).
    Level 1-4 = 1-4 active blocks.
    """
    return ConvLadderSeed(dim, max_level=4)


class ConvLadderSeed(nn.Module):
    """Microstructured CNN seed with internal level control.

    Implements identity-by-mask: all blocks always exist, but only
    active blocks (level > 0) contribute to output. This keeps the
    graph shape constant for torch.compile.
    """

    def __init__(self, channels: int, max_level: int = 4):
        super().__init__()
        self.channels = channels
        self.max_level = max_level
        self._level = 1  # Start at level 1 (one active block)

        # All blocks exist; level controls which are active
        self.blocks = nn.ModuleList([
            SeedConvBlock(channels, channels) for _ in range(max_level)
        ])

        # Final projection (always active)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.proj.weight)  # Zero-init for residual safety

    @property
    def internal_kind(self) -> int:
        """Return SeedInternalKind.CONV_LADDER.value (per Python specialist review)."""
        from esper.leyline import SeedInternalKind
        return SeedInternalKind.CONV_LADDER.value  # Return int, not enum

    @property
    def internal_level(self) -> int:
        return self._level

    @property
    def internal_max_level(self) -> int:
        return self.max_level

    def set_internal_level(self, level: int) -> None:
        """Set internal level and update requires_grad accordingly.

        Args:
            level: New level in [0, max_level]
        """
        self._level = max(0, min(level, self.max_level))
        self._update_requires_grad()

    def _update_requires_grad(self) -> None:
        """Update requires_grad based on current level."""
        for i, block in enumerate(self.blocks):
            active = i < self._level
            for p in block.parameters():
                p.requires_grad_(active)

    def active_param_count(self) -> int:
        """Return count of trainable parameters at current level."""
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with identity-by-mask.

        At level 0, returns identity (x).
        At level > 0, applies first `level` blocks.
        """
        if self._level == 0:
            return x  # Identity

        residual = x
        out = x
        for i in range(self._level):
            out = self.blocks[i](out)

        return residual + self.proj(out)
```

### Design Decisions
- **Identity-by-mask:** At level 0, return input unchanged (no computation beyond the branch check)
- **Zero-init projection:** Ensures safe residual connection at initialization
- **requires_grad gating:** Only active blocks train; inactive blocks frozen

### torch.compile Behavior (per PyTorch specialist review)

**Known behavior:** Level changes via `set_internal_level()` trigger torch.compile recompilation because:
- TorchDynamo specializes on `self._level` (Python int loop bound)
- Each unique `_level` value generates a new specialized graph

**Acceptable for Phase 0** because:
- Level changes are infrequent (driven by policy decisions, not per-batch)
- Recompilation cost is amortized over many forward passes at each level
- Alternative (tensor-based level) adds complexity for minimal benefit

**Phase 2 optimization opportunity:** Register `_level` as a buffer for dynamic shape support.

### Acceptance Criteria
- [ ] Blueprint registered with `BlueprintRegistry`
- [ ] `internal_kind`, `internal_level`, `internal_max_level` properties work
- [ ] `set_internal_level()` correctly updates requires_grad
- [ ] `active_param_count()` returns correct value
- [ ] Forward pass correct at all levels (0 through max_level)
- [ ] Shape-preserving: output shape == input shape

---

## K2: Add Internal State to `SeedSlot`

**File:** `src/esper/kasmina/slot.py`

### Specification

Extend `SeedState` to track internal level:

```python
@dataclass
class SeedState:
    # ... existing fields ...

    # Internal microstructure state (Phase 0)
    internal_kind: SeedInternalKind = SeedInternalKind.NONE
    internal_level: int = 0
    internal_max_level: int = 1

    def __post_init__(self) -> None:
        # Validate internal level invariants
        assert 0 <= self.internal_level <= self.internal_max_level
        assert self.internal_max_level >= 1
```

### Protocol Definition (per Python specialist review)

Define a Protocol to replace `hasattr` checks (avoids defensive programming anti-pattern):

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class MicrostructuredSeed(Protocol):
    """Protocol for seeds with internal microstructure.

    Use isinstance(seed, MicrostructuredSeed) instead of hasattr checks.
    This provides type safety and makes the contract explicit.
    """
    @property
    def internal_kind(self) -> int: ...
    @property
    def internal_level(self) -> int: ...
    @property
    def internal_max_level(self) -> int: ...
    def set_internal_level(self, level: int) -> None: ...
    def active_param_count(self) -> int: ...
```

Add helper methods to `SeedSlot`:

```python
class SeedSlot:
    def _init_internal_state_from_seed(self) -> None:
        """Initialize internal state from seed module if it has microstructure."""
        if self._seed is None:
            self._state.internal_kind = SeedInternalKind.NONE
            self._state.internal_level = 0
            self._state.internal_max_level = 1
            return

        # Use Protocol-based check instead of hasattr (per Python specialist review)
        if isinstance(self._seed, MicrostructuredSeed):
            self._state.internal_kind = SeedInternalKind(self._seed.internal_kind)
            self._state.internal_level = self._seed.internal_level
            self._state.internal_max_level = self._seed.internal_max_level
        else:
            self._state.internal_kind = SeedInternalKind.NONE
            self._state.internal_level = 0
            self._state.internal_max_level = 1

    @property
    def has_internal_structure(self) -> bool:
        """True if current seed has internal microstructure."""
        return self._state.internal_kind != SeedInternalKind.NONE
```

### Acceptance Criteria
- [ ] `SeedState` has internal fields with correct defaults
- [ ] Invariants enforced in `__post_init__`
- [ ] `_init_internal_state_from_seed()` correctly detects microstructure
- [ ] `has_internal_structure` property works

---

## K3: Implement Internal Ops Execution in `SeedSlot`

**File:** `src/esper/kasmina/slot.py`

### Specification

Add methods for executing internal ops:

```python
class SeedSlot:
    def grow_internal(self) -> bool:
        """Increase internal level by 1.

        Returns:
            True if level changed, False if already at max.
        """
        # Explicit null check (per Python specialist review)
        if not self.has_internal_structure or self._seed is None:
            return False
        if self._state.internal_level >= self._state.internal_max_level:
            return False

        from_level = self._state.internal_level
        self._state.internal_level += 1
        self._seed.set_internal_level(self._state.internal_level)

        self._emit_internal_level_changed(from_level, self._state.internal_level)
        return True

    def shrink_internal(self) -> bool:
        """Decrease internal level by 1.

        Returns:
            True if level changed, False if already at 0.
        """
        # Explicit null check (per Python specialist review)
        if not self.has_internal_structure or self._seed is None:
            return False
        if self._state.internal_level <= 0:
            return False

        from_level = self._state.internal_level
        self._state.internal_level -= 1
        self._seed.set_internal_level(self._state.internal_level)

        self._emit_internal_level_changed(from_level, self._state.internal_level)
        return True

    def _emit_internal_level_changed(self, from_level: int, to_level: int) -> None:
        """Emit telemetry for internal level change."""
        from esper.leyline import (
            TelemetryEventType,
            SeedInternalLevelChangedPayload,
        )

        # Use Protocol-based check instead of hasattr (per Python specialist review)
        active_params = (
            self._seed.active_param_count()
            if isinstance(self._seed, MicrostructuredSeed)
            else sum(p.numel() for p in self._seed.parameters() if p.requires_grad)
        )

        payload = SeedInternalLevelChangedPayload(
            slot_id=self.slot_id,
            env_id=0,  # Placeholder - injected by emit_with_env_context (no -1 sentinel)
            blueprint_id=self._state.blueprint_id,
            internal_kind=self._state.internal_kind.value,
            from_level=from_level,
            to_level=to_level,
            max_level=self._state.internal_max_level,
            active_params=active_params,
        )

        self._emit_telemetry(TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED, payload)
```

### Acceptance Criteria
- [ ] `grow_internal()` increases level and emits telemetry
- [ ] `shrink_internal()` decreases level and emits telemetry
- [ ] Both return False at boundaries (no-op)
- [ ] Both return False if no internal structure
- [ ] Seed module's level is kept in sync with state

---

## K4: Add DDP Sync for Internal Ops

**File:** `src/esper/kasmina/slot.py`

### Specification

Mirror existing `_sync_gate_decision()` pattern (per PyTorch specialist review, use `broadcast_object_list`):

```python
from typing import Any

class SeedSlot:
    def _sync_internal_op_decision(
        self,
        should_execute: bool,
        op_type: int,  # 0 = grow, 1 = shrink
    ) -> tuple[bool, int]:
        """Broadcast internal op decision to all ranks for DDP symmetry.

        Args:
            should_execute: Whether to execute the op
            op_type: Which op (0=grow, 1=shrink)

        Returns:
            (should_execute, op_type) synchronized across ranks

        Uses broadcast_object_list (per PyTorch specialist review) to:
        - Match existing codebase patterns
        - Avoid device placement assumptions
        - Work with any serializable data
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return should_execute, op_type

        rank = torch.distributed.get_rank()

        if rank == 0:
            sync_data: dict[str, Any] | None = {"execute": should_execute, "op_type": op_type}
        else:
            sync_data = None

        object_list: list[dict[str, Any] | None] = [sync_data]
        torch.distributed.broadcast_object_list(object_list, src=0)
        synced = object_list[0]

        assert synced is not None
        return bool(synced["execute"]), int(synced["op_type"])
```

### Usage in Execution

```python
def execute_internal_op(self, op: LifecycleOp) -> bool:
    """Execute an internal op with DDP synchronization."""
    if op == LifecycleOp.GROW_INTERNAL:
        should_execute = self._state.internal_level < self._state.internal_max_level
        op_type = 0
    elif op == LifecycleOp.SHRINK_INTERNAL:
        should_execute = self._state.internal_level > 0
        op_type = 1
    else:
        return False

    # Sync decision across ranks
    should_execute, op_type = self._sync_internal_op_decision(should_execute, op_type)

    if not should_execute:
        return False

    if op_type == 0:
        return self.grow_internal()
    else:
        return self.shrink_internal()
```

### DDP Bucket Note (per PyTorch specialist review)

When using `ConvLadderSeed` with DDP, ensure `find_unused_parameters=True` is set:
- Toggling `requires_grad` on blocks does NOT automatically rebuild DDP buckets
- `find_unused_parameters=True` handles this gracefully by skipping zero-gradient parameters

### Acceptance Criteria
- [ ] DDP sync broadcasts decision from rank 0
- [ ] All ranks execute identical internal ops
- [ ] Works correctly in non-distributed mode
- [ ] **Ensure `find_unused_parameters=True` when using DDP** (per PyTorch specialist review)
- [ ] Unit test verifies symmetry

---

## K5: Wire `SeedState.to_report()` for Internal Fields

**File:** `src/esper/kasmina/slot.py`

### Specification

Update `to_report()` to include internal fields:

```python
def to_report(self) -> SeedStateReport:
    """Convert state to Leyline report format."""
    return SeedStateReport(
        # ... existing fields ...

        # Internal microstructure
        internal_kind=self._state.internal_kind,
        internal_level=self._state.internal_level,
        internal_max_level=self._state.internal_max_level,
        internal_active_params=self._get_active_params(),
    )

def _get_active_params(self) -> int:
    """Get current active parameter count."""
    if self._seed is None:
        return 0
    # Use Protocol-based check instead of hasattr (per Python specialist review)
    if isinstance(self._seed, MicrostructuredSeed):
        return self._seed.active_param_count()
    return sum(p.numel() for p in self._seed.parameters() if p.requires_grad)
```

### Acceptance Criteria
- [ ] `to_report()` includes all internal fields
- [ ] Values match current state
- [ ] Works for both microstructured and non-microstructured seeds

---

## Testing Requirements

### Unit Tests (`tests/kasmina/`)

**test_blueprints.py (X1):**
```python
def test_conv_ladder_level_0_is_identity():
    seed = create_conv_ladder_seed(64)
    seed.set_internal_level(0)
    x = torch.randn(1, 64, 8, 8)
    assert torch.allclose(seed(x), x)

def test_conv_ladder_level_max_has_all_blocks():
    seed = create_conv_ladder_seed(64)
    seed.set_internal_level(4)
    # All blocks should have requires_grad=True
    for block in seed.blocks:
        for p in block.parameters():
            assert p.requires_grad

def test_conv_ladder_active_params_scales_with_level():
    seed = create_conv_ladder_seed(64)
    params_by_level = []
    for level in range(5):
        seed.set_internal_level(level)
        params_by_level.append(seed.active_param_count())
    # Params should be monotonically increasing
    assert params_by_level == sorted(params_by_level)
```

**test_seed_slot.py (X2):**
```python
def test_grow_internal_increases_level():
    slot = create_slot_with_ladder_seed()
    initial = slot.internal_level
    assert slot.grow_internal()
    assert slot.internal_level == initial + 1

def test_shrink_internal_at_zero_returns_false():
    slot = create_slot_with_ladder_seed()
    slot.set_internal_level(0)
    assert not slot.shrink_internal()
    assert slot.internal_level == 0

def test_internal_op_emits_telemetry():
    slot = create_slot_with_ladder_seed()
    with capture_telemetry() as events:
        slot.grow_internal()
    assert any(e.type == TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED for e in events)
```

### Property Tests (`tests/kasmina/properties/`) (X3)

```python
@given(level=st.integers(min_value=0, max_value=4))
def test_internal_level_invariant(level):
    slot = create_slot_with_ladder_seed()
    slot.set_internal_level(level)
    assert 0 <= slot.internal_level <= slot.internal_max_level

@given(ops=st.lists(st.sampled_from(['grow', 'shrink']), max_size=20))
def test_internal_ops_maintain_invariants(ops):
    slot = create_slot_with_ladder_seed()
    for op in ops:
        if op == 'grow':
            slot.grow_internal()
        else:
            slot.shrink_internal()
        assert 0 <= slot.internal_level <= slot.internal_max_level
```

### DDP Test (X4)

```python
@pytest.mark.distributed
def test_ddp_internal_op_symmetry():
    """Verify all ranks execute identical internal ops."""
    # Setup distributed environment
    # Execute internal ops on rank 0
    # Verify all ranks have same internal_level
```
