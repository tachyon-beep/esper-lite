# Track 6: Testing

**Priority:** High (validates all other tracks)
**Estimated Effort:** 1-2 days
**Dependencies:** All other tracks (testing follows implementation)

## Overview

This track defines comprehensive tests for Phase 0. Tests are organized by type: unit tests, property tests, DDP tests, and integration tests.

---

## Test Infrastructure Requirements (per specialist reviews)

### conftest.py Additions (per Python and PyTorch specialist reviews)

Add to `tests/conftest.py`:

```python
import pytest
import torch
import socket
from datetime import timedelta


@pytest.fixture(autouse=True)
def cuda_memory_cleanup():
    """Clean up CUDA memory after each test (per PyTorch specialist review)."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_free_port() -> int:
    """Find a free port for distributed testing (per PyTorch specialist review)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(rank: int, world_size: int, port: int):
    """Set up distributed environment with timeout (per Python specialist review)."""
    import os
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),  # Fail fast, don't hang
    )
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment (per PyTorch specialist review)."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
```

### Hypothesis Profile (per Python specialist review)

Property tests must specify max_examples explicitly:

```python
from hypothesis import settings

@settings(max_examples=100, deadline=None, derandomize=True)
class TestInternalLevelInvariants:
    ...
```

### Test Helper Location (per Python specialist review)

All test helpers referenced below should be defined in `tests/kasmina/conftest.py`:
- `create_test_slot()`
- `create_test_slot_with_ladder()`
- `create_conv_ladder_seed()`
- etc.

---

## X1: Unit Tests for `conv_ladder` Blueprint

**File:** `tests/kasmina/test_blueprints.py`

### Specification

```python
import pytest
import torch
from esper.kasmina.blueprints.cnn import create_conv_ladder_seed, ConvLadderSeed
from esper.leyline import SeedInternalKind


class TestConvLadderSeed:
    """Unit tests for conv_ladder blueprint."""

    @pytest.fixture
    def seed(self) -> ConvLadderSeed:
        """Create a test seed with 64 channels."""
        return create_conv_ladder_seed(64)

    def test_level_0_is_identity(self, seed: ConvLadderSeed):
        """At level 0, output should equal input."""
        seed.set_internal_level(0)
        x = torch.randn(1, 64, 8, 8)
        with torch.no_grad():  # Per PyTorch specialist: inference tests don't need gradients
            y = seed(x)
        assert torch.allclose(y, x, rtol=1e-5, atol=1e-8), "Level 0 should be identity"

    def test_level_max_uses_all_blocks(self, seed: ConvLadderSeed):
        """At max level, all blocks should have requires_grad=True."""
        seed.set_internal_level(seed.max_level)

        for i, block in enumerate(seed.blocks):
            for p in block.parameters():
                assert p.requires_grad, f"Block {i} should have requires_grad=True at max level"

    def test_level_0_freezes_all_blocks(self, seed: ConvLadderSeed):
        """At level 0, all blocks should have requires_grad=False."""
        seed.set_internal_level(0)

        for i, block in enumerate(seed.blocks):
            for p in block.parameters():
                assert not p.requires_grad, f"Block {i} should have requires_grad=False at level 0"

    def test_active_params_scales_with_level(self, seed: ConvLadderSeed):
        """Active params should increase monotonically with level."""
        params_by_level = []
        for level in range(seed.max_level + 1):
            seed.set_internal_level(level)
            params_by_level.append(seed.active_param_count())

        # Should be strictly increasing (or equal for level 0 to 1 if proj is always active)
        for i in range(1, len(params_by_level)):
            assert params_by_level[i] >= params_by_level[i-1], (
                f"Params should increase: level {i-1}={params_by_level[i-1]}, "
                f"level {i}={params_by_level[i]}"
            )

    def test_shape_preserved(self, seed: ConvLadderSeed):
        """Output shape should match input shape at all levels."""
        x = torch.randn(2, 64, 16, 16)

        with torch.no_grad():  # Per PyTorch specialist: inference tests don't need gradients
            for level in range(seed.max_level + 1):
                seed.set_internal_level(level)
                y = seed(x)
                assert y.shape == x.shape, f"Shape mismatch at level {level}"

    def test_internal_kind_property(self, seed: ConvLadderSeed):
        """internal_kind should return CONV_LADDER."""
        assert seed.internal_kind == SeedInternalKind.CONV_LADDER

    def test_internal_level_property(self, seed: ConvLadderSeed):
        """internal_level property should reflect current level."""
        for level in range(seed.max_level + 1):
            seed.set_internal_level(level)
            assert seed.internal_level == level

    def test_internal_max_level_property(self, seed: ConvLadderSeed):
        """internal_max_level should return max_level."""
        assert seed.internal_max_level == seed.max_level

    def test_set_internal_level_clamps(self, seed: ConvLadderSeed):
        """set_internal_level should clamp to valid range."""
        seed.set_internal_level(-5)
        assert seed.internal_level == 0

        seed.set_internal_level(100)
        assert seed.internal_level == seed.max_level

    def test_forward_differs_by_level(self, seed: ConvLadderSeed):
        """Forward pass should produce different outputs at different levels (except level 0)."""
        x = torch.randn(1, 64, 8, 8)

        seed.set_internal_level(1)
        y1 = seed(x)

        seed.set_internal_level(2)
        y2 = seed(x)

        # At different levels, outputs should differ
        assert not torch.allclose(y1, y2), "Different levels should produce different outputs"

    def test_gradient_flows_through_active_blocks(self, seed: ConvLadderSeed):
        """Gradients should flow through active blocks only."""
        seed.set_internal_level(2)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        y = seed(x)
        loss = y.sum()
        loss.backward()

        # Active blocks should have gradients
        for i in range(2):
            for p in seed.blocks[i].parameters():
                assert p.grad is not None, f"Active block {i} should have gradients"

        # Inactive blocks should NOT have gradients (per PyTorch specialist: use abs().max())
        for i in range(2, seed.max_level):
            for p in seed.blocks[i].parameters():
                assert p.grad is None or p.grad.abs().max() < 1e-10, (
                    f"Inactive block {i} should not have gradients"
                )

    def test_gradient_flows_after_level_change(self, seed: ConvLadderSeed):
        """Gradients should flow correctly after level transitions (per DRL specialist).

        This catches bugs where requires_grad state isn't properly updated
        after grow_internal() or shrink_internal().
        """
        seed.set_internal_level(2)

        # Do a forward-backward at level 2
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        y = seed(x)
        y.sum().backward()

        # Zero grads and grow to level 3
        seed.zero_grad()
        seed.set_internal_level(3)

        # Forward-backward at level 3
        y = seed(x)
        y.sum().backward()

        # Newly active block should have gradients
        for name, p in seed.blocks[2].named_parameters():
            assert p.grad is not None, f"Block 2 param {name} should have gradient at level 3"
            assert p.grad.abs().sum() > 0, f"Block 2 param {name} gradient should be non-zero"


class TestConvLadderRegistration:
    """Tests for blueprint registration."""

    def test_blueprint_registered(self):
        """conv_ladder should be registered with BlueprintRegistry."""
        from esper.kasmina.blueprints import BlueprintRegistry

        assert "conv_ladder" in BlueprintRegistry.list_blueprints()

    def test_create_via_registry(self):
        """Should be creatable via registry."""
        from esper.kasmina.blueprints import BlueprintRegistry

        seed = BlueprintRegistry.create("conv_ladder", dim=64)
        assert isinstance(seed, ConvLadderSeed)
```

### Acceptance Criteria
- [ ] All 13+ test cases pass (including gradient after level change)
- [ ] Level 0 is verified as identity
- [ ] Shape preservation verified at all levels
- [ ] Gradient flow verified for active blocks
- [ ] **Gradient flow after level change verified** (per DRL specialist review)
- [ ] Registry integration verified
- [ ] Inference tests use `torch.no_grad()` (per PyTorch specialist review)
- [ ] Tensor comparisons use explicit tolerances (per PyTorch specialist review)

---

## X2: Unit Tests for Internal Ops Execution

**File:** `tests/kasmina/test_seed_slot.py`

### Specification

```python
import pytest
from unittest.mock import MagicMock, patch
from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import SeedInternalKind, LifecycleOp, TelemetryEventType


class TestInternalOpsExecution:
    """Tests for grow_internal() and shrink_internal() execution."""

    @pytest.fixture
    def slot_with_ladder(self) -> SeedSlot:
        """Create a slot with a conv_ladder seed at level 2."""
        slot = create_test_slot(blueprint_id="conv_ladder")
        slot._state.internal_kind = SeedInternalKind.CONV_LADDER
        slot._state.internal_level = 2
        slot._state.internal_max_level = 4
        return slot

    def test_grow_internal_increases_level(self, slot_with_ladder: SeedSlot):
        """grow_internal() should increase level by 1."""
        initial = slot_with_ladder._state.internal_level
        result = slot_with_ladder.grow_internal()

        assert result is True
        assert slot_with_ladder._state.internal_level == initial + 1

    def test_grow_internal_at_max_returns_false(self, slot_with_ladder: SeedSlot):
        """grow_internal() at max level should return False."""
        slot_with_ladder._state.internal_level = slot_with_ladder._state.internal_max_level

        result = slot_with_ladder.grow_internal()

        assert result is False
        assert slot_with_ladder._state.internal_level == slot_with_ladder._state.internal_max_level

    def test_shrink_internal_decreases_level(self, slot_with_ladder: SeedSlot):
        """shrink_internal() should decrease level by 1."""
        initial = slot_with_ladder._state.internal_level
        result = slot_with_ladder.shrink_internal()

        assert result is True
        assert slot_with_ladder._state.internal_level == initial - 1

    def test_shrink_internal_at_zero_returns_false(self, slot_with_ladder: SeedSlot):
        """shrink_internal() at level 0 should return False."""
        slot_with_ladder._state.internal_level = 0

        result = slot_with_ladder.shrink_internal()

        assert result is False
        assert slot_with_ladder._state.internal_level == 0

    def test_grow_internal_syncs_to_seed(self, slot_with_ladder: SeedSlot):
        """grow_internal() should sync level to seed module."""
        slot_with_ladder.grow_internal()

        assert slot_with_ladder._seed.internal_level == slot_with_ladder._state.internal_level

    def test_shrink_internal_syncs_to_seed(self, slot_with_ladder: SeedSlot):
        """shrink_internal() should sync level to seed module."""
        slot_with_ladder.shrink_internal()

        assert slot_with_ladder._seed.internal_level == slot_with_ladder._state.internal_level

    def test_grow_internal_emits_telemetry(self, slot_with_ladder: SeedSlot):
        """grow_internal() should emit telemetry event."""
        with patch.object(slot_with_ladder, '_emit_telemetry') as mock_emit:
            slot_with_ladder.grow_internal()

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED

    def test_shrink_internal_emits_telemetry(self, slot_with_ladder: SeedSlot):
        """shrink_internal() should emit telemetry event."""
        with patch.object(slot_with_ladder, '_emit_telemetry') as mock_emit:
            slot_with_ladder.shrink_internal()

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED

    def test_ops_on_non_microstructured_seed_return_false(self):
        """Internal ops on non-microstructured seeds should return False."""
        slot = create_test_slot(blueprint_id="simple_conv")  # No microstructure
        slot._state.internal_kind = SeedInternalKind.NONE

        assert slot.grow_internal() is False
        assert slot.shrink_internal() is False


class TestInternalStateInitialization:
    """Tests for _init_internal_state_from_seed()."""

    def test_init_from_microstructured_seed(self):
        """Should initialize state from seed with microstructure."""
        slot = SeedSlot(slot_id="r0c0")
        seed = create_conv_ladder_seed(64)
        seed.set_internal_level(2)

        slot._seed = seed
        slot._init_internal_state_from_seed()

        assert slot._state.internal_kind == SeedInternalKind.CONV_LADDER
        assert slot._state.internal_level == 2
        assert slot._state.internal_max_level == 4

    def test_init_from_non_microstructured_seed(self):
        """Should set NONE for seeds without microstructure."""
        slot = SeedSlot(slot_id="r0c0")
        seed = create_simple_conv_seed(64)  # No internal_kind property

        slot._seed = seed
        slot._init_internal_state_from_seed()

        assert slot._state.internal_kind == SeedInternalKind.NONE
        assert slot._state.internal_level == 0
        assert slot._state.internal_max_level == 1

    def test_init_with_no_seed(self):
        """Should set defaults when no seed is present."""
        slot = SeedSlot(slot_id="r0c0")
        slot._seed = None

        slot._init_internal_state_from_seed()

        assert slot._state.internal_kind == SeedInternalKind.NONE
        assert slot._state.internal_level == 0
        assert slot._state.internal_max_level == 1


class TestHasInternalStructure:
    """Tests for has_internal_structure property."""

    def test_true_for_conv_ladder(self):
        """Should return True for CONV_LADDER kind."""
        slot = create_test_slot(blueprint_id="conv_ladder")
        slot._state.internal_kind = SeedInternalKind.CONV_LADDER

        assert slot.has_internal_structure is True

    def test_false_for_none(self):
        """Should return False for NONE kind."""
        slot = create_test_slot(blueprint_id="simple_conv")
        slot._state.internal_kind = SeedInternalKind.NONE

        assert slot.has_internal_structure is False
```

### Acceptance Criteria
- [ ] All 15+ test cases pass
- [ ] Level changes verified
- [ ] Boundary conditions verified
- [ ] Telemetry emission verified
- [ ] Seed sync verified
- [ ] Non-microstructured seeds handled correctly

---

## X3: Property Tests for Internal Level Invariants

**File:** `tests/kasmina/properties/test_internal_level.py`

### Specification

```python
import pytest
from hypothesis import given, strategies as st, assume
from esper.kasmina.slot import SeedSlot, SeedState
from esper.leyline import SeedInternalKind


class TestInternalLevelInvariants:
    """Property-based tests for internal level invariants."""

    @given(level=st.integers(min_value=-10, max_value=20))
    def test_level_always_in_valid_range(self, level: int):
        """internal_level should always be in [0, max_level]."""
        slot = create_test_slot_with_ladder()

        slot.set_internal_level(level)

        assert 0 <= slot._state.internal_level <= slot._state.internal_max_level

    @given(ops=st.lists(
        st.sampled_from(['grow', 'shrink']),
        min_size=0,
        max_size=50
    ))
    def test_ops_maintain_level_invariant(self, ops: list[str]):
        """Any sequence of grow/shrink ops should maintain level invariant."""
        slot = create_test_slot_with_ladder()

        for op in ops:
            if op == 'grow':
                slot.grow_internal()
            else:
                slot.shrink_internal()

            # Invariant must hold after every op
            assert 0 <= slot._state.internal_level <= slot._state.internal_max_level

    @given(ops=st.lists(
        st.sampled_from(['grow', 'shrink']),
        min_size=0,
        max_size=50
    ))
    def test_state_and_seed_always_in_sync(self, ops: list[str]):
        """State internal_level should always match seed internal_level."""
        slot = create_test_slot_with_ladder()

        for op in ops:
            if op == 'grow':
                slot.grow_internal()
            else:
                slot.shrink_internal()

            # Sync invariant
            assert slot._state.internal_level == slot._seed.internal_level

    @given(
        initial_level=st.integers(min_value=0, max_value=4),
        n_grows=st.integers(min_value=0, max_value=10),
        n_shrinks=st.integers(min_value=0, max_value=10),
    )
    def test_net_level_change_bounded(
        self,
        initial_level: int,
        n_grows: int,
        n_shrinks: int,
    ):
        """Net level change should be bounded by max_level and 0."""
        slot = create_test_slot_with_ladder()
        slot.set_internal_level(initial_level)

        for _ in range(n_grows):
            slot.grow_internal()
        for _ in range(n_shrinks):
            slot.shrink_internal()

        expected_min = 0
        expected_max = slot._state.internal_max_level

        assert expected_min <= slot._state.internal_level <= expected_max


class TestSeedStateInvariants:
    """Property tests for SeedState dataclass invariants."""

    @given(
        internal_level=st.integers(min_value=-100, max_value=100),
        internal_max_level=st.integers(min_value=-100, max_value=100),
    )
    def test_post_init_enforces_invariants(
        self,
        internal_level: int,
        internal_max_level: int,
    ):
        """SeedState __post_init__ should enforce invariants or raise.

        Fixed per Python specialist review: removed early returns that
        prevented testing the success case assertions.
        """
        is_valid_max = internal_max_level >= 1
        is_valid_level = 0 <= internal_level <= internal_max_level if is_valid_max else False

        if not is_valid_max or not is_valid_level:
            with pytest.raises(AssertionError):
                SeedState(
                    internal_level=internal_level,
                    internal_max_level=internal_max_level,
                )
        else:
            # Valid inputs should succeed
            state = SeedState(
                internal_level=internal_level,
                internal_max_level=internal_max_level,
            )
            assert 0 <= state.internal_level <= state.internal_max_level
            assert state.internal_max_level >= 1


class TestRewardInvariants:
    """Property tests for reward function invariants (per DRL specialist).

    These tests ensure the reward signal has the mathematical properties
    required for stable RL training.
    """

    @given(
        internal_level=st.integers(min_value=0, max_value=4),
        op=st.sampled_from([LifecycleOp.NOOP, LifecycleOp.GROW_INTERNAL, LifecycleOp.SHRINK_INTERNAL]),
    )
    def test_reward_always_bounded(self, internal_level: int, op: LifecycleOp):
        """Reward should always be in a bounded range.

        Unbounded rewards cause numerical instability in PPO's advantage calculation.
        """
        reward_computer = create_test_reward_computer()
        slot_state = create_test_slot_state(internal_level=internal_level)
        action_result = create_action_result(op=op, success=True)

        reward = reward_computer.compute(slot_state, action_result)

        # Reward should be bounded (adjust bounds based on actual design)
        assert -10.0 <= reward <= 10.0, f"Reward {reward} out of bounds for op {op}"

    @given(
        internal_level=st.integers(min_value=0, max_value=4),
        op=st.sampled_from([LifecycleOp.GROW_INTERNAL, LifecycleOp.SHRINK_INTERNAL]),
    )
    def test_intervention_cost_always_applies(self, internal_level: int, op: LifecycleOp):
        """Internal ops should always include intervention cost.

        This ensures the policy has consistent cost signal for interventions.
        """
        reward_computer = create_test_reward_computer()
        slot_state = create_test_slot_state(internal_level=internal_level)

        # Compute reward for internal op
        internal_result = create_action_result(op=op, success=True)
        internal_reward = reward_computer.compute(slot_state, internal_result)

        # Compute reward for NOOP (no intervention cost)
        noop_result = create_action_result(op=LifecycleOp.NOOP, success=True)
        noop_reward = reward_computer.compute(slot_state, noop_result)

        # Internal op should have lower reward due to intervention cost
        # (unless contribution benefit outweighs cost)
        expected_cost = reward_computer.config.cost_grow_internal
        assert internal_reward <= noop_reward + expected_cost, (
            f"Internal op reward {internal_reward} should be <= NOOP {noop_reward} + cost {expected_cost}"
        )

    @given(
        internal_level=st.integers(min_value=0, max_value=4),
        seed=st.integers(min_value=0, max_value=1000),
    )
    def test_reward_is_deterministic(self, internal_level: int, seed: int):
        """Same state/action should produce identical reward.

        Non-deterministic rewards break PPO's variance estimates.
        """
        torch.manual_seed(seed)

        reward_computer = create_test_reward_computer()
        slot_state = create_test_slot_state(internal_level=internal_level)
        action_result = create_action_result(op=LifecycleOp.GROW_INTERNAL, success=True)

        reward1 = reward_computer.compute(slot_state, action_result)

        torch.manual_seed(seed)  # Reset seed
        reward2 = reward_computer.compute(slot_state, action_result)

        assert reward1 == reward2, f"Reward not deterministic: {reward1} != {reward2}"

    @given(
        level_sequence=st.lists(
            st.integers(min_value=0, max_value=4),
            min_size=2,
            max_size=10,
        ),
    )
    def test_thrash_penalty_scales_with_reversals(self, level_sequence: list[int]):
        """Thrash penalty should increase with level reversals.

        This tests the penalty mechanism that discourages grow-shrink-grow patterns.
        """
        assume(len(set(level_sequence)) > 1)  # Ensure some variation

        reward_computer = create_test_reward_computer()

        # Count direction changes
        reversals = 0
        for i in range(1, len(level_sequence)):
            if i >= 2:
                prev_delta = level_sequence[i-1] - level_sequence[i-2]
                curr_delta = level_sequence[i] - level_sequence[i-1]
                if prev_delta * curr_delta < 0:  # Sign change = reversal
                    reversals += 1

        # Compute thrash penalty
        penalty = reward_computer.compute_thrash_penalty(level_sequence)

        # Penalty should be proportional to reversals (or zero if no reversals)
        if reversals == 0:
            assert penalty == 0.0
        else:
            assert penalty > 0.0, f"Should have penalty for {reversals} reversals"
```

### Acceptance Criteria
- [ ] All property tests pass with 100+ examples
- [ ] Level range invariant verified
- [ ] State/seed sync invariant verified
- [ ] SeedState __post_init__ invariants verified
- [ ] No flaky failures under randomized inputs
- [ ] **Reward bounded invariant verified** (per DRL specialist review)
- [ ] **Intervention cost invariant verified** (per DRL specialist review)
- [ ] **Reward determinism verified** (per DRL specialist review)
- [ ] **Thrash penalty scales correctly** (per DRL specialist review)

---

## X4: DDP Symmetry Test for Internal Ops

**File:** `tests/kasmina/test_ddp_internal_ops.py`

### Specification

```python
import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
from tests.conftest import find_free_port, setup_distributed, cleanup_distributed


@pytest.mark.distributed
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDDPInternalOpSymmetry:
    """Tests verifying DDP symmetry for internal ops.

    Uses dynamic port allocation (per PyTorch specialist review) to avoid
    CI conflicts with hardcoded ports.
    """

    def test_sync_internal_op_decision(self):
        """Verify _sync_internal_op_decision broadcasts from rank 0."""
        port = find_free_port()  # Per PyTorch specialist: dynamic port allocation

        def run_test(rank: int, world_size: int, port: int):
            setup_distributed(rank, world_size, port)

            try:
                slot = create_test_slot_with_ladder()
                slot.device = torch.device(f"cuda:{rank}")

                # Rank 0 decides to grow, others decide shrink
                if rank == 0:
                    should_execute = True
                    op_type = 0  # grow
                else:
                    should_execute = False
                    op_type = 1  # shrink

                # Sync should broadcast rank 0's decision
                synced_execute, synced_op = slot._sync_internal_op_decision(
                    should_execute, op_type
                )

                # All ranks should have rank 0's decision
                assert synced_execute is True
                assert synced_op == 0
            finally:
                cleanup_distributed()

        spawn(run_test, args=(2, port), nprocs=2, join=True)

    def test_execute_internal_op_ddp_symmetry(self):
        """Verify all ranks execute identical internal ops."""
        port = find_free_port()

        def run_test(rank: int, world_size: int, port: int):
            setup_distributed(rank, world_size, port)

            try:
                # Create identical slots on all ranks
                slot = create_test_slot_with_ladder()
                slot._state.internal_level = 2
                slot.device = torch.device(f"cuda:{rank}")

                # Execute GROW_INTERNAL
                result = slot.execute_internal_op(LifecycleOp.GROW_INTERNAL)

                # Gather results
                results = [None] * world_size
                dist.all_gather_object(results, {
                    "success": result,
                    "level": slot._state.internal_level,
                })

                # All ranks should have same level
                levels = [r["level"] for r in results]
                assert all(l == levels[0] for l in levels), f"Level mismatch: {levels}"
            finally:
                cleanup_distributed()

        spawn(run_test, args=(2, port), nprocs=2, join=True)

    def test_ddp_gradient_aggregation_after_level_change(self):
        """Verify DDP gradients aggregate correctly after level change (per DRL specialist).

        This is critical: when internal_level changes, requires_grad toggles on
        different parameters. DDP must handle this correctly.
        """
        port = find_free_port()

        def run_test(rank: int, world_size: int, port: int):
            setup_distributed(rank, world_size, port)

            try:
                # Create seed wrapped in DDP
                seed = create_conv_ladder_seed(64).to(f"cuda:{rank}")
                seed = torch.nn.parallel.DistributedDataParallel(
                    seed,
                    device_ids=[rank],
                    find_unused_parameters=True,  # Per PyTorch specialist: required for level changes
                )

                # Change level
                seed.module.set_internal_level(2)

                # Forward-backward
                x = torch.randn(4, 64, 8, 8, device=f"cuda:{rank}")
                y = seed(x)
                y.sum().backward()

                # Verify gradients on active blocks
                for i in range(2):
                    for name, p in seed.module.blocks[i].named_parameters():
                        # Gather gradients across ranks
                        grad_norms = [None] * world_size
                        grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
                        dist.all_gather_object(grad_norms, grad_norm)

                        # All ranks should have similar gradient norms (within tolerance)
                        assert all(
                            abs(g - grad_norms[0]) < 1e-5 for g in grad_norms
                        ), f"Gradient mismatch for {name}: {grad_norms}"
            finally:
                cleanup_distributed()

        spawn(run_test, args=(2, port), nprocs=2, join=True)

    def test_non_distributed_fallback(self):
        """Verify sync works in non-distributed mode."""
        slot = create_test_slot_with_ladder()

        # Should pass through unchanged when not distributed
        should_execute, op_type = slot._sync_internal_op_decision(True, 0)

        assert should_execute is True
        assert op_type == 0
```

### Acceptance Criteria
- [ ] Sync broadcasts from rank 0
- [ ] All ranks execute identical ops
- [ ] All ranks end up with same internal_level
- [ ] **DDP gradient aggregation works after level change** (per DRL specialist review)
- [ ] Non-distributed fallback works
- [ ] Tests use dynamic port allocation (per PyTorch specialist review)
- [ ] Tests run on multi-GPU system

---

## X5: Integration Test: Internal Ops in Training Loop

**File:** `tests/integration/test_internal_ops_training.py`

### Specification

```python
import pytest
import torch
from esper.simic.training.vectorized import VectorizedTrainer
from esper.tamiyo.policy import create_test_policy
from esper.leyline import LifecycleOp


@pytest.mark.integration
class TestInternalOpsInTraining:
    """Integration tests for internal ops in the training loop."""

    @pytest.fixture
    def trainer_with_ladder_seeds(self) -> VectorizedTrainer:
        """Create a trainer with conv_ladder seeds."""
        config = create_test_config(
            num_envs=4,
            slots=["r0c0", "r0c1"],
            blueprint_id="conv_ladder",
        )
        trainer = VectorizedTrainer(config)
        trainer.reset()
        return trainer

    def test_grow_internal_in_training_step(
        self,
        trainer_with_ladder_seeds: VectorizedTrainer,
    ):
        """Verify GROW_INTERNAL works in training step."""
        trainer = trainer_with_ladder_seeds

        # Get initial level
        slot = trainer._get_slot(0, "r0c0")
        initial_level = slot._state.internal_level

        # Create action with GROW_INTERNAL
        action = create_action(
            slot_id="r0c0",
            lifecycle_op=LifecycleOp.GROW_INTERNAL,
        )

        # Execute through trainer
        result = trainer.execute_action(0, "r0c0", action)

        assert result.success
        assert slot._state.internal_level == initial_level + 1

    def test_internal_ops_affect_param_count(
        self,
        trainer_with_ladder_seeds: VectorizedTrainer,
    ):
        """Verify internal ops change active param count."""
        trainer = trainer_with_ladder_seeds
        slot = trainer._get_slot(0, "r0c0")

        # Record initial params
        initial_params = slot.to_report().internal_active_params

        # Grow
        slot.grow_internal()
        grown_params = slot.to_report().internal_active_params

        assert grown_params > initial_params

        # Shrink back
        slot.shrink_internal()
        shrunk_params = slot.to_report().internal_active_params

        assert shrunk_params == initial_params

    def test_action_mask_updated_after_internal_op(
        self,
        trainer_with_ladder_seeds: VectorizedTrainer,
    ):
        """Verify action mask reflects internal level after op."""
        trainer = trainer_with_ladder_seeds
        slot = trainer._get_slot(0, "r0c0")

        # Go to max level
        while slot._state.internal_level < slot._state.internal_max_level:
            slot.grow_internal()

        # Get observation
        obs = trainer.get_observation(0)
        mask = obs["action_masks"]

        # GROW should be masked (at max)
        grow_idx = LifecycleOp.GROW_INTERNAL.value
        assert mask[0, grow_idx] == 0.0  # slot 0, grow op

        # SHRINK should be available
        shrink_idx = LifecycleOp.SHRINK_INTERNAL.value
        assert mask[0, shrink_idx] == 1.0

    def test_telemetry_emitted_during_training(
        self,
        trainer_with_ladder_seeds: VectorizedTrainer,
    ):
        """Verify telemetry events are emitted during training."""
        trainer = trainer_with_ladder_seeds

        # Capture telemetry
        events = []
        trainer.on_telemetry = lambda e: events.append(e)

        # Execute internal op
        slot = trainer._get_slot(0, "r0c0")
        slot.grow_internal()

        # Should have emitted event
        level_changed_events = [
            e for e in events
            if e.type == TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED
        ]
        assert len(level_changed_events) == 1

    def test_reward_includes_intervention_cost(
        self,
        trainer_with_ladder_seeds: VectorizedTrainer,
    ):
        """Verify reward includes intervention cost for internal ops."""
        trainer = trainer_with_ladder_seeds

        # Execute NOOP and get reward
        noop_result = trainer.execute_action(
            0, "r0c0",
            create_action(lifecycle_op=LifecycleOp.NOOP),
        )
        noop_reward = trainer.compute_reward(0, noop_result)

        # Execute GROW and get reward
        grow_result = trainer.execute_action(
            0, "r0c0",
            create_action(lifecycle_op=LifecycleOp.GROW_INTERNAL),
        )
        grow_reward = trainer.compute_reward(0, grow_result)

        # GROW should have intervention cost
        # (assuming same primary reward conditions)
        assert grow_reward < noop_reward  # Due to intervention cost


@pytest.mark.integration
@pytest.mark.slow
class TestInternalOpsLearning:
    """Integration tests verifying policy can learn internal ops (per DRL specialist).

    These tests verify the RL loop actually learns the intended behaviors,
    not just that mechanics work correctly.
    """

    @pytest.fixture
    def trainer_for_learning(self) -> VectorizedTrainer:
        """Create trainer configured for learning tests."""
        config = create_test_config(
            num_envs=8,
            slots=["r0c0"],
            blueprint_id="conv_ladder",
            # Short episodes for faster convergence
            max_episode_steps=50,
            # Use simplified reward for clearer signal
            reward_mode="simplified",
        )
        return VectorizedTrainer(config)

    def test_policy_learns_to_grow_when_underfitting(
        self,
        trainer_for_learning: VectorizedTrainer,
    ):
        """Verify policy learns GROW_INTERNAL when seed is underfitting (per DRL specialist).

        Setup: Seed starts at level 0 (identity), task requires actual computation.
        Expected: Policy should learn to grow internal level.
        """
        trainer = trainer_for_learning
        policy = create_test_policy()

        # Record initial grow probability
        initial_obs = trainer.reset()
        with torch.no_grad():
            initial_output = policy(initial_obs)
        grow_idx = LifecycleOp.GROW_INTERNAL.value
        initial_grow_prob = initial_output.probs[0, 0, grow_idx].item()

        # Train for N episodes
        for _ in range(100):  # Enough to see learning signal
            train_one_episode(trainer, policy)

        # Check grow probability increased
        final_obs = trainer.reset()
        with torch.no_grad():
            final_output = policy(final_obs)
        final_grow_prob = final_output.probs[0, 0, grow_idx].item()

        # Policy should have learned to grow more often
        assert final_grow_prob > initial_grow_prob + 0.1, (
            f"Policy should learn to grow when underfitting: "
            f"initial={initial_grow_prob:.3f}, final={final_grow_prob:.3f}"
        )

    def test_intervention_cost_discourages_thrashing(
        self,
        trainer_for_learning: VectorizedTrainer,
    ):
        """Verify intervention cost discourages grow/shrink thrashing (per DRL specialist).

        Setup: Train with and without intervention costs.
        Expected: With costs, policy should stabilize at fewer level changes.
        """
        trainer = trainer_for_learning

        # Train with intervention costs
        config_with_cost = trainer.config
        config_with_cost.contribution_reward.cost_grow_internal = 0.005
        config_with_cost.contribution_reward.cost_shrink_internal = 0.005

        policy_with_cost = create_test_policy()
        ops_with_cost = []
        for _ in range(50):
            episode_ops = train_one_episode(trainer, policy_with_cost)
            ops_with_cost.extend(episode_ops)

        # Count level changes
        level_changes_with_cost = sum(
            1 for op in ops_with_cost
            if op in [LifecycleOp.GROW_INTERNAL, LifecycleOp.SHRINK_INTERNAL]
        )

        # Train without costs (for comparison)
        config_no_cost = trainer.config
        config_no_cost.contribution_reward.cost_grow_internal = 0.0
        config_no_cost.contribution_reward.cost_shrink_internal = 0.0

        policy_no_cost = create_test_policy()
        ops_no_cost = []
        for _ in range(50):
            episode_ops = train_one_episode(trainer, policy_no_cost)
            ops_no_cost.extend(episode_ops)

        level_changes_no_cost = sum(
            1 for op in ops_no_cost
            if op in [LifecycleOp.GROW_INTERNAL, LifecycleOp.SHRINK_INTERNAL]
        )

        # Intervention costs should reduce thrashing
        assert level_changes_with_cost < level_changes_no_cost * 0.8, (
            f"Intervention cost should reduce thrashing: "
            f"with_cost={level_changes_with_cost}, no_cost={level_changes_no_cost}"
        )

    def test_reward_reflects_internal_op_outcome(
        self,
        trainer_for_learning: VectorizedTrainer,
    ):
        """Verify reward correctly reflects internal op outcomes (per DRL specialist).

        Setup: Compare rewards for beneficial vs non-beneficial grows.
        Expected: Growing when beneficial should yield higher return.
        """
        trainer = trainer_for_learning

        # Scenario 1: Grow when underfitting (beneficial)
        trainer.reset()
        slot = trainer._get_slot(0, "r0c0")
        slot._state.internal_level = 0  # Start at identity (underfitting)

        # Execute grow
        grow_result = trainer.execute_action(
            0, "r0c0",
            create_action(lifecycle_op=LifecycleOp.GROW_INTERNAL),
        )

        # Get reward signal
        # (Beneficial grow should have positive contribution component)
        reward_beneficial = trainer.compute_reward(0, grow_result)

        # Scenario 2: Grow when already at good level (wasteful)
        trainer.reset()
        slot = trainer._get_slot(0, "r0c0")
        slot._state.internal_level = 3  # Already high capacity

        grow_result_wasteful = trainer.execute_action(
            0, "r0c0",
            create_action(lifecycle_op=LifecycleOp.GROW_INTERNAL),
        )

        reward_wasteful = trainer.compute_reward(0, grow_result_wasteful)

        # Beneficial grow should have better reward (less negative due to contribution)
        # Note: Both have intervention cost, but beneficial should have offsetting gain
        assert reward_beneficial > reward_wasteful - 0.001, (
            f"Beneficial grow should have better reward: "
            f"beneficial={reward_beneficial:.4f}, wasteful={reward_wasteful:.4f}"
        )
```

### Acceptance Criteria
- [ ] Internal ops execute in training loop
- [ ] Param count changes with internal ops
- [ ] Action masks update correctly
- [ ] Telemetry emitted during training
- [ ] Reward includes intervention cost
- [ ] **Policy learns to grow when underfitting** (per DRL specialist review)
- [ ] **Intervention cost discourages thrashing** (per DRL specialist review)
- [ ] **Reward reflects internal op outcome** (per DRL specialist review)

---

## X6: Obs Shape Assertion Tests

**File:** `tests/tamiyo/policy/test_obs_shapes.py`

### Specification

```python
import pytest
import torch
from esper.leyline import (
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_SLOT_FEATURE_SIZE,
    NUM_SLOTS,
    NUM_OPS,
    BASE_FEATURE_FIELDS,
    SLOT_FEATURE_FIELDS,
)
from esper.tamiyo.policy.features import extract_base_features, extract_slot_features
from esper.tamiyo.networks.factored_lstm import FactoredLSTMPolicy


class TestObsShapeConsistency:
    """Tests verifying obs shape consistency across the system."""

    def test_base_feature_size_matches_field_list(self):
        """OBS_V3_BASE_FEATURE_SIZE should be derived from field list."""
        # BASE_FEATURE_SIZE = len(BASE_FEATURE_FIELDS) + NUM_OPS (for one-hot)
        expected = len(BASE_FEATURE_FIELDS) + NUM_OPS
        assert OBS_V3_BASE_FEATURE_SIZE == expected, (
            f"BASE_FEATURE_SIZE mismatch: {OBS_V3_BASE_FEATURE_SIZE} vs "
            f"len(fields)={len(BASE_FEATURE_FIELDS)} + NUM_OPS={NUM_OPS} = {expected}"
        )

    def test_slot_feature_size_matches_field_list(self):
        """OBS_V3_SLOT_FEATURE_SIZE should be derived from field list."""
        expected = len(SLOT_FEATURE_FIELDS)
        assert OBS_V3_SLOT_FEATURE_SIZE == expected, (
            f"SLOT_FEATURE_SIZE mismatch: {OBS_V3_SLOT_FEATURE_SIZE} vs "
            f"len(fields)={expected}"
        )

    def test_internal_level_norm_in_slot_fields(self):
        """internal_level_norm should be in SLOT_FEATURE_FIELDS."""
        assert "internal_level_norm" in SLOT_FEATURE_FIELDS, (
            "internal_level_norm missing from SLOT_FEATURE_FIELDS"
        )

    def test_extracted_base_features_match_size(self):
        """Extracted base features should match OBS_V3_BASE_FEATURE_SIZE."""
        obs = create_test_observation()
        features = extract_base_features(obs)

        assert features.shape[-1] == OBS_V3_BASE_FEATURE_SIZE, (
            f"Base features shape {features.shape[-1]} != {OBS_V3_BASE_FEATURE_SIZE}"
        )

    def test_extracted_slot_features_match_size(self):
        """Extracted slot features should match OBS_V3_SLOT_FEATURE_SIZE."""
        report = create_test_seed_state_report()
        features = extract_slot_features(report)

        assert features.shape[-1] == OBS_V3_SLOT_FEATURE_SIZE, (
            f"Slot features shape {features.shape[-1]} != {OBS_V3_SLOT_FEATURE_SIZE}"
        )

    def test_policy_accepts_correct_shapes(self):
        """Policy should accept tensors with correct shapes."""
        policy = FactoredLSTMPolicy()

        obs = {
            "base_features": torch.randn(4, OBS_V3_BASE_FEATURE_SIZE),
            "slot_features": torch.randn(4, NUM_SLOTS, OBS_V3_SLOT_FEATURE_SIZE),
            "action_masks": torch.ones(4, NUM_SLOTS, NUM_OPS),
        }

        # Should not raise
        output = policy(obs)
        assert output is not None

    def test_policy_rejects_wrong_base_shape(self):
        """Policy should reject wrong base feature shape."""
        policy = FactoredLSTMPolicy()

        obs = {
            "base_features": torch.randn(4, OBS_V3_BASE_FEATURE_SIZE + 1),  # Wrong!
            "slot_features": torch.randn(4, NUM_SLOTS, OBS_V3_SLOT_FEATURE_SIZE),
            "action_masks": torch.ones(4, NUM_SLOTS, NUM_OPS),
        }

        with pytest.raises(AssertionError):
            policy(obs)

    def test_policy_rejects_wrong_slot_shape(self):
        """Policy should reject wrong slot feature shape."""
        policy = FactoredLSTMPolicy()

        obs = {
            "base_features": torch.randn(4, OBS_V3_BASE_FEATURE_SIZE),
            "slot_features": torch.randn(4, NUM_SLOTS, OBS_V3_SLOT_FEATURE_SIZE + 1),  # Wrong!
            "action_masks": torch.ones(4, NUM_SLOTS, NUM_OPS),
        }

        with pytest.raises(AssertionError):
            policy(obs)

    def test_num_ops_matches_lifecycle_op_enum(self):
        """NUM_OPS should match length of LifecycleOp enum."""
        from esper.leyline import LifecycleOp

        assert NUM_OPS == len(LifecycleOp), (
            f"NUM_OPS={NUM_OPS} != len(LifecycleOp)={len(LifecycleOp)}"
        )


class TestFeatureFieldPositions:
    """Tests verifying feature field positions match extraction."""

    def test_slot_feature_positions_match(self):
        """Feature positions in tensor should match SLOT_FEATURE_FIELDS order."""
        report = create_test_seed_state_report(
            internal_kind=SeedInternalKind.CONV_LADDER,
            internal_level=2,
            internal_max_level=4,
        )

        features = extract_slot_features(report)

        # Find internal_level_norm position
        idx = SLOT_FEATURE_FIELDS.index("internal_level_norm")

        # Value should be 2/4 = 0.5
        assert features[idx] == pytest.approx(0.5)
```

### Acceptance Criteria
- [ ] Dimension constants match field lists
- [ ] `internal_level_norm` in slot fields
- [ ] Extracted features match expected sizes
- [ ] Policy accepts correct shapes
- [ ] Policy rejects incorrect shapes
- [ ] Field positions match extraction order
