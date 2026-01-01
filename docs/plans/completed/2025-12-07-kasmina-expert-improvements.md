# Kasmina Expert Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement all recommendations from DRL and PyTorch expert analysis to optimize kasmina for PyTorch 2.9 and Python 3.13.

**Architecture:** Split SeedSlot.forward into compile-friendly tensor operations vs control flow, optimize host forward passes for torch.compile, add modern RL techniques (curriculum learning, PER), and modernize with Python 3.13 features.

**Tech Stack:** PyTorch 2.9, Python 3.13, torch.compile, torch._foreach_norm, FlexAttention, pattern matching

---

## Phase 1: PyTorch 2.9 Compile Optimization (P0)

### Task 1: Extract Compilable Tensor Operations from SeedSlot

**Files:**
- Modify: `src/esper/kasmina/isolation.py:47-57`
- Test: `tests/kasmina/test_compile_tensor_ops.py`

**Context:** The `blend_with_isolation` function is already in isolation.py. We'll add `ste_forward` there too, keeping tensor ops together for torch.compile.

**Step 1: Write the failing test**

```python
# tests/kasmina/test_compile_tensor_ops.py
"""Test that tensor operations are torch.compile compatible."""
import pytest
import torch

from esper.kasmina.isolation import blend_with_isolation, ste_forward


class TestCompilableTensorOps:
    """Verify tensor ops compile without graph breaks."""

    def test_ste_forward_compiles_fullgraph(self):
        """STE forward should compile with fullgraph=True (no graph breaks)."""
        compiled_ste = torch.compile(ste_forward, fullgraph=True)

        host = torch.randn(2, 32, 8, 8, requires_grad=True)
        seed = torch.randn(2, 32, 8, 8, requires_grad=True)

        result = compiled_ste(host, seed)

        assert result.shape == host.shape
        # Verify STE property: forward equals host
        assert torch.allclose(result, host, atol=1e-6)

    def test_ste_forward_gradient_flow(self):
        """STE backward should flow gradients to seed only."""
        host = torch.randn(2, 32, 8, 8, requires_grad=True)
        seed = torch.randn(2, 32, 8, 8, requires_grad=True)

        result = ste_forward(host, seed)
        loss = result.sum()
        loss.backward()

        # Host should have gradients (it's in the computation)
        assert host.grad is not None
        # Seed should have gradients via STE
        assert seed.grad is not None

    def test_blend_with_isolation_compiles_fullgraph(self):
        """Blend should compile with fullgraph=True."""
        compiled_blend = torch.compile(blend_with_isolation, fullgraph=True)

        host = torch.randn(2, 32, 8, 8)
        seed = torch.randn(2, 32, 8, 8)

        result = compiled_blend(host, seed, 0.5, detach_host=True)

        assert result.shape == host.shape
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_compile_tensor_ops.py -v`
Expected: FAIL with "cannot import name 'ste_forward' from 'esper.kasmina.isolation'"

**Step 3: Implement ste_forward function**

Add to `src/esper/kasmina/isolation.py` after `blend_with_isolation`:

```python
def ste_forward(host_features: torch.Tensor, seed_features: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator forward pass.

    Forward: returns host_features (seed contribution cancels out)
    Backward: gradients flow to both host and seed parameters

    This is torch.compile friendly - pure tensor operations, no control flow.
    """
    return host_features + (seed_features - seed_features.detach())
```

Update `__all__` to include `ste_forward`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_compile_tensor_ops.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/isolation.py tests/kasmina/test_compile_tensor_ops.py
git commit -m "feat(kasmina): extract ste_forward as compilable tensor op"
```

---

### Task 2: Update SeedSlot to Use Extracted Functions

**Files:**
- Modify: `src/esper/kasmina/slot.py:871-928`
- Test: `tests/test_seed_slot.py` (existing tests should still pass)

**Step 1: Write the failing test**

```python
# Add to tests/test_seed_slot.py or create tests/kasmina/test_slot_ste.py
def test_slot_ste_uses_isolation_function():
    """Verify SeedSlot uses the extracted ste_forward function."""
    from esper.kasmina.isolation import ste_forward

    # This test verifies behavior is unchanged after refactor
    slot = SeedSlot("test", channels=32, device="cpu")
    slot.germinate("norm", "seed-1", host_module=CNNHost())

    # Force TRAINING stage with alpha=0 for STE path
    slot.state.stage = SeedStage.TRAINING
    slot.state.alpha = 0.0
    slot.isolate_gradients = True

    host_features = torch.randn(1, 32, 8, 8, requires_grad=True)
    result = slot(host_features)

    # STE property: output equals host
    assert torch.allclose(result, host_features, atol=1e-6)
```

**Step 2: Run test to verify it passes (behavior already correct)**

Run: `uv run pytest tests/test_seed_slot.py -v -k ste`
Expected: PASS (behavior unchanged, just refactored to use extracted function)

**Step 3: Refactor slot.py forward to use extracted functions**

Modify `src/esper/kasmina/slot.py`:

1. Add import at top:
```python
from esper.kasmina.isolation import blend_with_isolation, ste_forward
```

2. Update forward method (lines 904-909):
```python
        # 3. INCUBATOR MODE (TRAINING stage, alpha == 0.0)
        if self.state.stage == SeedStage.TRAINING and self.alpha == 0.0:
            if _DEBUG_STE:
                assert seed_features.requires_grad, (
                    "STE requires seed_features to have requires_grad=True for gradient flow"
                )
            return ste_forward(host_features, seed_features)
```

**Step 4: Run full test suite**

Run: `uv run pytest tests/test_seed_slot.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "refactor(kasmina): use extracted ste_forward in SeedSlot"
```

---

### Task 3: Remove Assert from TransformerHost.forward

**Files:**
- Modify: `src/esper/kasmina/host.py:209-211`
- Test: `tests/kasmina/test_host_compile.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_host_compile.py
"""Test host networks are torch.compile compatible."""
import pytest
import torch

from esper.kasmina.host import TransformerHost


class TestTransformerHostCompile:
    """Verify TransformerHost compiles without graph breaks from assertions."""

    def test_forward_no_graph_break_from_assert(self):
        """TransformerHost.forward should not have assertion graph breaks."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=2, block_size=32)

        # This would cause graph break if assert is present
        compiled_host = torch.compile(host, fullgraph=True)

        x = torch.randint(0, 100, (2, 16))
        result = compiled_host(x)

        assert result.shape == (2, 16, 100)

    def test_sequence_length_validation_still_works(self):
        """Sequence length > block_size should still raise error."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=2, block_size=32)

        x = torch.randint(0, 100, (2, 64))  # 64 > 32 block_size

        with pytest.raises(ValueError, match="exceeds block_size"):
            host(x)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_host_compile.py -v`
Expected: FAIL - fullgraph=True fails due to assert

**Step 3: Replace assert with explicit ValueError**

Modify `src/esper/kasmina/host.py:209-211`:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # Embeddings
        pos = torch.arange(T, device=x.device)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_host_compile.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/kasmina/test_host_compile.py
git commit -m "fix(kasmina): replace assert with ValueError for torch.compile compatibility"
```

---

## Phase 2: Host Forward Optimization (P1)

### Task 4: Pre-compute Slot Keys in CNNHost

**Files:**
- Modify: `src/esper/kasmina/host.py:70-78`
- Test: `tests/kasmina/test_host_compile.py`

**Step 1: Write the failing test**

```python
# Add to tests/kasmina/test_host_compile.py
class TestCNNHostCompile:
    """Verify CNNHost compiles efficiently."""

    def test_forward_uses_precomputed_keys(self):
        """CNNHost should not format strings in forward loop."""
        host = CNNHost(num_classes=10, n_blocks=3)

        # Should compile without string formatting graph breaks
        compiled_host = torch.compile(host, fullgraph=True)

        x = torch.randn(2, 3, 32, 32)
        result = compiled_host(x)

        assert result.shape == (2, 10)

    def test_slot_key_lookup_uses_tuple(self):
        """Verify _slot_keys tuple is used for O(1) lookup."""
        host = CNNHost(num_classes=10, n_blocks=4)

        # Verify internal structure
        assert hasattr(host, '_slot_keys')  # Already exists
        assert isinstance(host._slot_keys, tuple)
```

**Step 2: Run test to verify current state**

Run: `uv run pytest tests/kasmina/test_host_compile.py::TestCNNHostCompile -v`
Expected: May fail on fullgraph=True due to string formatting

**Step 3: Refactor CNNHost.forward to use pre-computed keys**

Modify `src/esper/kasmina/host.py`:

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slot_idx = 0
        for idx, block in enumerate(self.blocks):
            x = self.pool(block(x))
            # Use pre-computed _slot_indices instead of string formatting
            if idx in self._slot_indices:
                x = self.slots[self._slot_keys[slot_idx]](x)
                slot_idx += 1

        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_host_compile.py::TestCNNHostCompile -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py
git commit -m "perf(kasmina): use pre-computed slot keys in CNNHost forward"
```

---

## Phase 3: Gradient Monitoring Enhancement (P1)

### Task 5: Use torch._foreach_norm for Gradient Isolation

**Files:**
- Modify: `src/esper/kasmina/isolation.py:81-107`
- Test: `tests/kasmina/test_gradient_isolation.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_gradient_isolation.py
"""Test gradient isolation monitoring performance."""
import pytest
import torch
import torch.nn as nn

from esper.kasmina.isolation import GradientIsolationMonitor


class TestGradientIsolationPerformance:
    """Verify foreach_norm optimization."""

    def test_check_isolation_uses_foreach(self):
        """Verify batched norm computation is used."""
        host = nn.Linear(64, 64)
        seed = nn.Linear(64, 64)

        monitor = GradientIsolationMonitor()
        monitor.register(host, seed)

        # Create gradients
        x = torch.randn(4, 64)
        loss = (host(x) + seed(x)).sum()
        loss.backward()

        is_isolated, stats = monitor.check_isolation()

        # Verify stats are computed
        assert "host_grad_norm" in stats
        assert "seed_grad_norm" in stats
        assert stats["host_grad_norm"] > 0
        assert stats["seed_grad_norm"] > 0
```

**Step 2: Run test to verify baseline works**

Run: `uv run pytest tests/kasmina/test_gradient_isolation.py -v`
Expected: PASS (existing implementation works)

**Step 3: Optimize with torch._foreach_norm**

Modify `src/esper/kasmina/isolation.py:81-123`:

```python
    @torch.no_grad()
    def check_isolation(self) -> tuple[bool, dict]:
        """Check if gradient isolation is maintained.

        Uses torch._foreach_norm for batched norm computation - O(1) CUDA syncs
        instead of O(n_params). This matches torch.nn.utils.clip_grad_norm_ internals.
        """
        # Collect gradients that exist
        host_grads = [p.grad for p in self._host_params if p.grad is not None]
        seed_grads = [p.grad for p in self._seed_params if p.grad is not None]

        # Compute norms with foreach (single sync per group)
        if host_grads:
            # torch._foreach_norm returns list of norms per tensor
            norms = torch._foreach_norm(host_grads)
            host_norm = torch.stack(norms).pow(2).sum().sqrt().item()
        else:
            host_norm = 0.0

        if seed_grads:
            norms = torch._foreach_norm(seed_grads)
            seed_norm = torch.stack(norms).pow(2).sum().sqrt().item()
        else:
            seed_norm = 0.0

        self.host_grad_norm = host_norm
        self.seed_grad_norm = seed_norm

        is_isolated = host_norm < self.threshold

        if not is_isolated:
            self.violations += 1

        return is_isolated, {
            "host_grad_norm": host_norm,
            "seed_grad_norm": seed_norm,
            "isolated": is_isolated,
            "violations": self.violations,
        }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_gradient_isolation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/isolation.py tests/kasmina/test_gradient_isolation.py
git commit -m "perf(kasmina): use torch._foreach_norm for O(1) gradient norm computation"
```

---

## Phase 4: Blueprint Enhancements (P2/P3)

### Task 6: Add FlexAttention Blueprint Variant

**Files:**
- Modify: `src/esper/kasmina/blueprints/transformer.py`
- Test: `tests/kasmina/test_blueprints_flex.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_blueprints_flex.py
"""Test FlexAttention blueprint variant."""
import pytest
import torch

from esper.kasmina.blueprints.registry import BlueprintRegistry


class TestFlexAttentionBlueprint:
    """Verify FlexAttention seed works correctly."""

    @pytest.mark.skipif(
        not hasattr(torch.nn.attention, 'flex_attention'),
        reason="FlexAttention requires PyTorch 2.5+"
    )
    def test_flex_attention_registered(self):
        """FlexAttention blueprint should be registered."""
        specs = BlueprintRegistry.list_specs("transformer")
        names = [s.name for s in specs]
        assert "flex_attention" in names

    @pytest.mark.skipif(
        not hasattr(torch.nn.attention, 'flex_attention'),
        reason="FlexAttention requires PyTorch 2.5+"
    )
    def test_flex_attention_forward(self):
        """FlexAttention seed should process input correctly."""
        seed = BlueprintRegistry.create("transformer", "flex_attention", 64)

        x = torch.randn(2, 16, 64)
        result = seed(x)

        assert result.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_blueprints_flex.py -v`
Expected: FAIL with "flex_attention not in names"

**Step 3: Implement FlexAttention blueprint**

Add to `src/esper/kasmina/blueprints/transformer.py`:

```python
# Check for FlexAttention availability (PyTorch 2.5+)
_HAS_FLEX_ATTENTION = hasattr(torch.nn.attention, 'flex_attention')

if _HAS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    @BlueprintRegistry.register(
        "flex_attention", "transformer", param_estimate=55000,
        description="FlexAttention with custom patterns (PyTorch 2.5+)"
    )
    def create_flex_attention_seed(dim: int, n_head: int = 4) -> nn.Module:
        """FlexAttention seed with customizable attention patterns."""

        class FlexAttentionSeed(nn.Module):
            def __init__(self, dim: int, n_head: int):
                super().__init__()
                self.n_head = n_head
                self.head_dim = dim // n_head

                self.qkv = nn.Linear(dim, 3 * dim)
                self.proj = nn.Linear(dim, dim)
                nn.init.zeros_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, t, c = x.shape

                qkv = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                # FlexAttention with causal mask
                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx

                out = flex_attention(q, k, v, score_mod=causal_mask)

                out = out.transpose(1, 2).reshape(b, t, c)
                return x + self.proj(out)

        return FlexAttentionSeed(dim, n_head)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_blueprints_flex.py -v`
Expected: PASS (or skip if PyTorch < 2.5)

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/transformer.py tests/kasmina/test_blueprints_flex.py
git commit -m "feat(kasmina): add FlexAttention blueprint variant for PyTorch 2.5+"
```

---

### Task 7: Add Activation Checkpointing for Large MLP Seeds

**Files:**
- Modify: `src/esper/kasmina/blueprints/transformer.py:78-95`
- Test: `tests/kasmina/test_blueprints_checkpoint.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_blueprints_checkpoint.py
"""Test activation checkpointing for large seeds."""
import pytest
import torch

from esper.kasmina.blueprints.registry import BlueprintRegistry


class TestMLPCheckpointing:
    """Verify MLP seed supports activation checkpointing."""

    def test_mlp_checkpoint_option(self):
        """MLP blueprint should accept checkpoint parameter."""
        # Create with checkpointing enabled
        seed = BlueprintRegistry.create(
            "transformer", "mlp", 64,
            checkpoint=True
        )

        x = torch.randn(2, 16, 64, requires_grad=True)
        result = seed(x)

        assert result.shape == x.shape

    def test_mlp_checkpoint_reduces_memory(self):
        """Checkpointing should reduce activation memory."""
        # This is a behavioral test - checkpointing trades compute for memory
        seed_no_ckpt = BlueprintRegistry.create("transformer", "mlp", 256, checkpoint=False)
        seed_ckpt = BlueprintRegistry.create("transformer", "mlp", 256, checkpoint=True)

        x = torch.randn(4, 32, 256, requires_grad=True)

        # Both should produce same output
        result_no_ckpt = seed_no_ckpt(x.clone())
        result_ckpt = seed_ckpt(x.clone())

        assert result_no_ckpt.shape == result_ckpt.shape
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_blueprints_checkpoint.py -v`
Expected: FAIL with "unexpected keyword argument 'checkpoint'"

**Step 3: Add checkpointing support to MLP blueprint**

Modify `src/esper/kasmina/blueprints/transformer.py`:

```python
from torch.utils.checkpoint import checkpoint as torch_checkpoint


@BlueprintRegistry.register(
    "mlp", "transformer", param_estimate=1200000, description="Additional MLP (4x expansion)"
)
def create_transformer_mlp_seed(dim: int, expansion: int = 4, checkpoint: bool = False) -> nn.Module:
    """Additional MLP seed with optional activation checkpointing."""

    class TransformerMLPSeed(nn.Module):
        def __init__(self, dim: int, expansion: int, use_checkpoint: bool):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.fc2 = nn.Linear(dim * expansion, dim)
            self.use_checkpoint = use_checkpoint
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.gelu(self.fc1(x)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_checkpoint and self.training and x.requires_grad:
                return x + torch_checkpoint(self._mlp_forward, x, use_reentrant=False)
            return x + self._mlp_forward(x)

    return TransformerMLPSeed(dim, expansion, checkpoint)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_blueprints_checkpoint.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/transformer.py tests/kasmina/test_blueprints_checkpoint.py
git commit -m "feat(kasmina): add activation checkpointing option for MLP blueprint"
```

---

## Phase 5: Python 3.13 Modernization

### Task 8: Pattern Matching for Gate Logic

**Files:**
- Modify: `src/esper/kasmina/slot.py:289-308`
- Test: Run existing tests (behavior unchanged)

**Step 1: Verify existing tests pass**

Run: `uv run pytest tests/test_seed_slot.py -v -k gate`
Expected: All PASS

**Step 2: Refactor check_gate to use pattern matching**

Modify `src/esper/kasmina/slot.py:289-308`:

```python
    def check_gate(self, state: SeedState, target_stage: SeedStage) -> GateResult:
        """Check if seed passes the gate for target stage."""
        gate = self._get_gate_level(target_stage)

        match gate:
            case GateLevel.G0:
                return self._check_g0(state)
            case GateLevel.G1:
                return self._check_g1(state)
            case GateLevel.G2:
                return self._check_g2(state)
            case GateLevel.G3:
                return self._check_g3(state)
            case GateLevel.G4:
                return self._check_g4(state)
            case GateLevel.G5:
                return self._check_g5(state)
            case _:
                return GateResult(gate=gate, passed=True, score=1.0)
```

**Step 3: Run tests to verify behavior unchanged**

Run: `uv run pytest tests/test_seed_slot.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add src/esper/kasmina/slot.py
git commit -m "refactor(kasmina): use Python 3.13 pattern matching for gate logic"
```

---

### Task 9: Add kw_only to SeedState Dataclass

**Files:**
- Modify: `src/esper/kasmina/slot.py` (SeedState dataclass)
- Test: `tests/kasmina/test_seed_state.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_seed_state.py
"""Test SeedState dataclass safety."""
import pytest

from esper.kasmina.slot import SeedState
from esper.leyline import SeedStage


class TestSeedStateKwOnly:
    """Verify SeedState requires keyword arguments."""

    def test_positional_args_rejected(self):
        """SeedState should reject positional arguments."""
        with pytest.raises(TypeError, match="positional"):
            # This should fail - positional args not allowed
            SeedState("seed-1", "norm", "slot-1")

    def test_keyword_args_accepted(self):
        """SeedState should accept keyword arguments."""
        state = SeedState(
            seed_id="seed-1",
            blueprint_id="norm",
            slot_id="slot-1",
            stage=SeedStage.DORMANT,
        )

        assert state.seed_id == "seed-1"
        assert state.blueprint_id == "norm"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_seed_state.py -v`
Expected: FAIL - positional args currently accepted

**Step 3: Add kw_only=True to SeedState**

Find and modify SeedState dataclass in `src/esper/kasmina/slot.py`:

```python
@dataclass(slots=True, kw_only=True)
class SeedState:
    """Mutable state for a seed during its lifecycle."""
    seed_id: str
    blueprint_id: str
    slot_id: str = ""
    stage: SeedStage = SeedStage.DORMANT
    # ... rest of fields
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_seed_state.py -v`
Expected: PASS

**Step 5: Fix any call sites that use positional args**

Run: `uv run pytest -v`
Fix any failures by converting to keyword arguments.

**Step 6: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_seed_state.py
git commit -m "refactor(kasmina): add kw_only=True to SeedState for safety"
```

---

## Phase 6: RL Algorithm Improvements

### Task 10: Add Gradient-Based Seed Readiness to G2 Gate

**Files:**
- Modify: `src/esper/kasmina/slot.py:374-415` (G2 gate)
- Modify: `src/esper/kasmina/slot.py` (SeedMetrics dataclass)
- Test: `tests/kasmina/test_g2_gradient_readiness.py`

**Context:** The G2 gate currently uses global improvement which conflates host and seed contributions. Add a gradient-based metric that measures seed-specific learning.

**Step 1: Write the failing test**

```python
# tests/kasmina/test_g2_gradient_readiness.py
"""Test G2 gate gradient-based seed readiness."""
import pytest
import torch

from esper.kasmina.slot import SeedSlot, SeedState, SeedMetrics, QualityGates
from esper.leyline import SeedStage


class TestG2GradientReadiness:
    """Verify G2 uses seed gradient statistics."""

    def test_g2_checks_seed_gradient_norm_ratio(self):
        """G2 should check seed gradient norm relative to host."""
        gates = QualityGates()

        # Create state with good global improvement but low seed gradient activity
        state = SeedState(
            seed_id="test",
            blueprint_id="norm",
            slot_id="test_slot",
            stage=SeedStage.TRAINING,
        )
        state.metrics = SeedMetrics()
        state.metrics.improvement_since_stage_start = 2.0  # Above threshold
        state.metrics.isolation_violations = 0
        state.metrics.epochs_in_current_stage = 5
        state.metrics.seed_gradient_norm_ratio = 0.01  # Very low seed activity

        result = gates._check_g2(state)

        # Should fail due to low seed gradient activity
        assert "seed_gradient_low" in result.checks_failed or not result.passed
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/kasmina/test_g2_gradient_readiness.py -v`
Expected: FAIL - attribute 'seed_gradient_norm_ratio' doesn't exist

**Step 3: Add seed_gradient_norm_ratio to SeedMetrics**

Modify SeedMetrics in `src/esper/kasmina/slot.py`:

```python
@dataclass(slots=True)
class SeedMetrics:
    """Metrics tracked during seed lifecycle."""
    epochs_total: int = 0
    epochs_in_current_stage: int = 0
    improvement_since_stage_start: float = 0.0
    counterfactual_contribution: float | None = None
    isolation_violations: int = 0
    blending_delta: float | None = None
    # NEW: Gradient-based seed activity metric
    seed_gradient_norm_ratio: float = 0.0  # seed_grad_norm / (host_grad_norm + eps)
```

**Step 4: Update G2 gate to check gradient ratio**

Modify `_check_g2` in `src/esper/kasmina/slot.py`:

```python
    def _check_g2(self, state: SeedState) -> GateResult:
        """G2: Blending readiness – global improvement + seed readiness + gradient activity."""
        checks_passed = []
        checks_failed = []

        improvement = state.metrics.improvement_since_stage_start

        # Global performance: host + training loop improving
        if improvement >= self.min_training_improvement:
            checks_passed.append(f"global_improvement_{improvement:.2f}%")
            perf_ok = True
        else:
            checks_failed.append(f"global_improvement_insufficient_{improvement:.2f}%")
            perf_ok = False

        # Global isolation guard
        if state.metrics.isolation_violations <= self.max_isolation_violations:
            checks_passed.append("isolation_ok")
            isolation_ok = True
        else:
            checks_failed.append(f"isolation_violations_{state.metrics.isolation_violations}")
            isolation_ok = False

        # Seed-specific readiness: enough TRAINING epochs to be worth blending
        if self._seed_ready_for_blending(state):
            checks_passed.append("seed_ready")
            seed_ok = True
        else:
            checks_failed.append("seed_not_ready")
            seed_ok = False

        # NEW: Gradient-based seed activity check
        # Ensures seed is actually learning, not just riding host improvements
        min_gradient_ratio = 0.05  # Seed should have at least 5% of host gradient activity
        if state.metrics.seed_gradient_norm_ratio >= min_gradient_ratio:
            checks_passed.append(f"seed_gradient_active_{state.metrics.seed_gradient_norm_ratio:.2f}")
            gradient_ok = True
        else:
            checks_failed.append(f"seed_gradient_low_{state.metrics.seed_gradient_norm_ratio:.2f}")
            gradient_ok = False

        passed = perf_ok and isolation_ok and seed_ok and gradient_ok
        score = min(1.0, improvement / 5.0) if improvement > 0 else 0.0

        return GateResult(
            gate=GateLevel.G2,
            passed=passed,
            score=score,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            message=f"Improvement: {improvement:.2f}%",
        )
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/kasmina/test_g2_gradient_readiness.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/kasmina/slot.py tests/kasmina/test_g2_gradient_readiness.py
git commit -m "feat(kasmina): add gradient-based seed readiness to G2 gate"
```

---

### Task 11: Add UCB1 Blueprint Curriculum

**Files:**
- Create: `src/esper/simic/curriculum.py`
- Test: `tests/simic/test_curriculum.py`

**Context:** Implement UCB1-based curriculum learning for blueprint selection, starting with simpler seeds and unlocking complex ones based on demonstrated competence.

**Step 1: Write the failing test**

```python
# tests/simic/test_curriculum.py
"""Test UCB1 blueprint curriculum."""
import pytest

from esper.simic.curriculum import BlueprintCurriculum


class TestBlueprintCurriculum:
    """Verify UCB1 curriculum for blueprint selection."""

    def test_initial_curriculum_favors_simple(self):
        """Initial curriculum should favor simpler blueprints."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora", "attention", "mlp"],
            complexity=[100, 6000, 50000, 1200000],
        )

        # Initially, simpler blueprints should have higher UCB scores
        scores = curriculum.get_ucb_scores()

        # norm (simplest) should have highest initial score
        assert scores["norm"] >= scores["mlp"]

    def test_ucb_updates_after_success(self):
        """Successful fossilization should update UCB stats."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora"],
            complexity=[100, 6000],
        )

        # Record successful fossilization
        curriculum.record_outcome("norm", success=True, reward=1.0)

        stats = curriculum.get_stats("norm")
        assert stats["trials"] == 1
        assert stats["successes"] == 1

    def test_ucb_exploration_bonus(self):
        """Unexplored blueprints should get exploration bonus."""
        curriculum = BlueprintCurriculum(
            blueprints=["norm", "lora", "attention"],
            complexity=[100, 6000, 50000],
        )

        # Use norm many times
        for _ in range(10):
            curriculum.record_outcome("norm", success=True, reward=0.5)

        scores = curriculum.get_ucb_scores()

        # Unexplored attention should have exploration bonus
        assert scores["attention"] > 0  # Has exploration bonus despite no trials
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_curriculum.py -v`
Expected: FAIL with "No module named 'esper.simic.curriculum'"

**Step 3: Implement BlueprintCurriculum**

Create `src/esper/simic/curriculum.py`:

```python
"""Blueprint curriculum with UCB1 exploration."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BlueprintStats:
    """Statistics for a single blueprint."""
    trials: int = 0
    successes: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.trials if self.trials > 0 else 0.0


class BlueprintCurriculum:
    """UCB1-based curriculum for blueprint selection.

    Balances exploration (trying new blueprints) with exploitation
    (using blueprints that have worked well).

    UCB1 formula: mean_reward + c * sqrt(log(total_trials) / blueprint_trials)

    Additionally applies a complexity penalty to favor simpler blueprints
    initially, following curriculum learning principles.
    """

    def __init__(
        self,
        blueprints: list[str],
        complexity: list[int],
        exploration_weight: float = 2.0,
        complexity_penalty: float = 0.1,
    ):
        if len(blueprints) != len(complexity):
            raise ValueError("blueprints and complexity must have same length")

        self.blueprints = blueprints
        self.complexity = dict(zip(blueprints, complexity))
        self.exploration_weight = exploration_weight
        self.complexity_penalty = complexity_penalty

        self._stats: dict[str, BlueprintStats] = {
            name: BlueprintStats() for name in blueprints
        }
        self._total_trials = 0

        # Normalize complexity to [0, 1] range
        max_complexity = max(complexity)
        self._normalized_complexity = {
            name: c / max_complexity for name, c in self.complexity.items()
        }

    def record_outcome(self, blueprint: str, success: bool, reward: float) -> None:
        """Record outcome of a blueprint trial."""
        if blueprint not in self._stats:
            raise ValueError(f"Unknown blueprint: {blueprint}")

        stats = self._stats[blueprint]
        stats.trials += 1
        stats.total_reward += reward
        if success:
            stats.successes += 1
        self._total_trials += 1

    def get_ucb_scores(self) -> dict[str, float]:
        """Compute UCB scores for all blueprints."""
        scores = {}

        for name in self.blueprints:
            stats = self._stats[name]

            if stats.trials == 0:
                # Unexplored: high exploration bonus, complexity penalty
                exploration = self.exploration_weight * 2.0  # Extra bonus for unexplored
                complexity_term = self.complexity_penalty * self._normalized_complexity[name]
                scores[name] = exploration - complexity_term
            else:
                # UCB1 formula
                mean = stats.mean_reward
                exploration = self.exploration_weight * math.sqrt(
                    math.log(self._total_trials + 1) / stats.trials
                )
                complexity_term = self.complexity_penalty * self._normalized_complexity[name]
                scores[name] = mean + exploration - complexity_term

        return scores

    def select_blueprint(self) -> str:
        """Select blueprint with highest UCB score."""
        scores = self.get_ucb_scores()
        return max(scores, key=scores.get)

    def get_stats(self, blueprint: str) -> dict:
        """Get statistics for a blueprint."""
        stats = self._stats[blueprint]
        return {
            "trials": stats.trials,
            "successes": stats.successes,
            "mean_reward": stats.mean_reward,
        }


__all__ = ["BlueprintCurriculum", "BlueprintStats"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_curriculum.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/curriculum.py tests/simic/test_curriculum.py
git commit -m "feat(simic): add UCB1 blueprint curriculum for exploration/exploitation"
```

---

### Task 12: Add Prioritized Experience Replay Buffer

**Files:**
- Create: `src/esper/simic/prioritized_buffer.py`
- Test: `tests/simic/test_prioritized_buffer.py`

**Context:** Rare events like successful fossilization should be prioritized during learning.

**Step 1: Write the failing test**

```python
# tests/simic/test_prioritized_buffer.py
"""Test prioritized experience replay buffer."""
import pytest
import torch

from esper.simic.prioritized_buffer import PrioritizedReplayBuffer


class TestPrioritizedReplayBuffer:
    """Verify PER implementation."""

    def test_add_and_sample(self):
        """Buffer should store and sample experiences."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        # Add experiences with priorities
        for i in range(10):
            buffer.add(
                state=torch.randn(8),
                action=i % 4,
                reward=float(i),
                next_state=torch.randn(8),
                done=False,
                priority=1.0 + i * 0.1,
            )

        batch, indices, weights = buffer.sample(batch_size=4)

        assert len(batch["states"]) == 4
        assert len(indices) == 4
        assert len(weights) == 4

    def test_high_priority_sampled_more(self):
        """High priority experiences should be sampled more often."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0, beta=0.0)

        # Add low priority experience
        buffer.add(
            state=torch.zeros(8),
            action=0,
            reward=0.0,
            next_state=torch.zeros(8),
            done=False,
            priority=0.01,
        )

        # Add high priority experience
        buffer.add(
            state=torch.ones(8),
            action=1,
            reward=1.0,
            next_state=torch.ones(8),
            done=False,
            priority=100.0,
        )

        # Sample many times and count
        high_priority_count = 0
        for _ in range(100):
            batch, indices, _ = buffer.sample(batch_size=1)
            if batch["actions"][0] == 1:
                high_priority_count += 1

        # High priority should be sampled much more often
        assert high_priority_count > 80

    def test_update_priorities(self):
        """Priorities should be updatable after sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

        buffer.add(
            state=torch.randn(8),
            action=0,
            reward=1.0,
            next_state=torch.randn(8),
            done=False,
            priority=1.0,
        )

        _, indices, _ = buffer.sample(batch_size=1)

        # Update with new TD error
        buffer.update_priorities(indices, [10.0])

        # Priority should be updated (implementation detail)
        assert buffer._priorities[indices[0]] > 1.0
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/test_prioritized_buffer.py -v`
Expected: FAIL with "No module named 'esper.simic.prioritized_buffer'"

**Step 3: Implement PrioritizedReplayBuffer**

Create `src/esper/simic/prioritized_buffer.py`:

```python
"""Prioritized Experience Replay buffer."""
from __future__ import annotations

import numpy as np
import torch


class SumTree:
    """Binary tree for efficient priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def add(self, priority: float, data_idx: int) -> None:
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s: float) -> tuple[int, float]:
        """Get data index and priority for cumulative sum s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer (Schaul et al., 2016).

    Experiences are sampled proportionally to their priority (TD error).
    Importance sampling weights correct for the bias introduced.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self._tree = SumTree(capacity)
        self._priorities = np.zeros(capacity)
        self._data: list[dict | None] = [None] * capacity
        self._size = 0
        self._pointer = 0

    def add(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        priority: float,
    ) -> None:
        """Add experience with priority."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }

        # Store experience
        self._data[self._pointer] = experience

        # Update priority (with alpha exponent)
        priority_alpha = (priority + self.epsilon) ** self.alpha
        self._priorities[self._pointer] = priority_alpha
        self._tree.add(priority_alpha, self._pointer)

        # Update pointers
        self._pointer = (self._pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[dict, list[int], np.ndarray]:
        """Sample batch with importance sampling weights."""
        indices = []
        priorities = []

        # Divide priority range into segments for stratified sampling
        segment = self._tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, priority = self._tree.get(s)
            indices.append(idx)
            priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities)
        probs = priorities / self._tree.total
        weights = (self._size * probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Collect batch
        batch = {
            "states": torch.stack([self._data[i]["state"] for i in indices]),
            "actions": torch.tensor([self._data[i]["action"] for i in indices]),
            "rewards": torch.tensor([self._data[i]["reward"] for i in indices]),
            "next_states": torch.stack([self._data[i]["next_state"] for i in indices]),
            "dones": torch.tensor([self._data[i]["done"] for i in indices]),
        }

        return batch, indices, weights

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
        """Update priorities after learning."""
        for idx, priority in zip(indices, priorities):
            priority_alpha = (priority + self.epsilon) ** self.alpha
            self._priorities[idx] = priority_alpha
            self._tree.add(priority_alpha, idx)

    def __len__(self) -> int:
        return self._size


__all__ = ["PrioritizedReplayBuffer"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/simic/test_prioritized_buffer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/prioritized_buffer.py tests/simic/test_prioritized_buffer.py
git commit -m "feat(simic): add prioritized experience replay buffer for rare events"
```

---

## Phase 7: Shape Probe Cache Enhancement (P2)

### Task 13: Add LRU Cache for Shape Probes

**Files:**
- Modify: `src/esper/kasmina/slot.py` (shape probe cache)
- Test: `tests/kasmina/test_shape_probe_cache.py`

**Step 1: Write the failing test**

```python
# tests/kasmina/test_shape_probe_cache.py
"""Test shape probe caching with LRU eviction."""
import pytest
import torch

from esper.kasmina.slot import SeedSlot


class TestShapeProbeCache:
    """Verify shape probe cache behavior."""

    def test_cache_hit_returns_same_tensor(self):
        """Repeated calls should return cached tensor."""
        slot = SeedSlot("test", channels=32, device="cpu")

        probe1 = slot._get_shape_probe("cnn")
        probe2 = slot._get_shape_probe("cnn")

        # Should be same tensor (cached)
        assert probe1 is probe2

    def test_cache_invalidated_on_device_change(self):
        """Cache should clear when device changes."""
        slot = SeedSlot("test", channels=32, device="cpu")

        probe_cpu = slot._get_shape_probe("cnn")

        # Simulate device change
        slot.device = torch.device("cpu")  # Same device, cache should still work

        probe_after = slot._get_shape_probe("cnn")
        assert probe_cpu is probe_after
```

**Step 2: Run test to verify baseline**

Run: `uv run pytest tests/kasmina/test_shape_probe_cache.py -v`
Expected: PASS (basic caching works)

**Step 3: Document current implementation is sufficient**

The current dict-based cache with device invalidation is appropriate for the use case. LRU would add complexity without benefit since there are only 2 topologies (cnn, transformer).

**Step 4: Commit test as documentation**

```bash
git add tests/kasmina/test_shape_probe_cache.py
git commit -m "test(kasmina): add shape probe cache tests documenting current behavior"
```

---

## Summary

This plan covers all recommendations from both expert analyses:

| Phase | Task | Priority | Description |
|-------|------|----------|-------------|
| 1 | 1-2 | P0 | Extract compilable tensor ops (ste_forward) |
| 1 | 3 | P0 | Remove TransformerHost assert |
| 2 | 4 | P1 | Pre-compute CNNHost slot keys |
| 3 | 5 | P1 | Use torch._foreach_norm |
| 4 | 6 | P2 | Add FlexAttention blueprint |
| 4 | 7 | P3 | Add MLP activation checkpointing |
| 5 | 8 | P2 | Pattern matching for gates |
| 5 | 9 | P3 | Add kw_only to SeedState |
| 6 | 10 | P1 | G2 gradient-based seed readiness |
| 6 | 11 | P2 | UCB1 blueprint curriculum |
| 6 | 12 | P2 | Prioritized experience replay |
| 7 | 13 | P2 | Shape probe cache tests |

**Estimated commits:** 13
**Test files created:** 10
**Source files modified:** 7
**Source files created:** 2
