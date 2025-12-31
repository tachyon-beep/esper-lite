# Remove Legacy Compatibility Code + TransformerHost Multi-Slot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all backwards compatibility code from MorphogeneticModel and add proper multi-slot support to TransformerHost.

**Architecture:** MorphogeneticModel will have a single code path using the `slots` parameter explicitly. TransformerHost will gain `segment_channels`, `forward_to_segment()`, and `forward_from_segment()` methods mirroring CNNHost. All call sites will be updated to use explicit slot names instead of legacy `seed_slot`/`seed_state` properties.

**Tech Stack:** Python, PyTorch, pytest

---

## Summary of Changes

### What Gets Removed (CLAUDE.md Compliance)
1. `_legacy_single_slot` flag and all branching on it
2. `seed_slot` property (legacy single-slot accessor)
3. `seed_state` property (legacy first-active-seed accessor)
4. Default slot inference in `germinate_seed()` and `cull_seed()`
5. Fallback slot_id mapping logic ("mid" -> "block2_post")

### What Gets Added
1. TransformerHost: `segment_channels` attribute
2. TransformerHost: `forward_to_segment()` method
3. TransformerHost: `forward_from_segment()` method
4. Updated call sites with explicit slot parameters

### API Changes
| Old API | New API |
|---------|---------|
| `model.seed_slot` | `model.seed_slots["mid"]` |
| `model.seed_state` | Iterate `seed_slots.values()` to find active |
| `MorphogeneticModel(host, device)` | `MorphogeneticModel(host, device, slots=["mid"])` |
| `model.germinate_seed(bp, id)` | `model.germinate_seed(bp, id, slot="mid")` |
| `model.cull_seed()` | `model.cull_seed(slot="mid")` |

---

## Task 1: Add segment_channels to TransformerHost

**Files:**
- Modify: `src/esper/kasmina/host.py:279-352` (TransformerHost class)
- Test: `tests/test_host.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_host.py
"""Host network tests."""

import pytest
import torch

from esper.kasmina.host import CNNHost, TransformerHost


class TestTransformerHostSegments:
    """Test TransformerHost segment_channels and segment methods."""

    def test_segment_channels_exists(self):
        """TransformerHost must expose segment_channels attribute."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        assert hasattr(host, "segment_channels")
        assert isinstance(host.segment_channels, dict)
        assert set(host.segment_channels.keys()) == {"early", "mid", "late"}

    def test_segment_channels_values(self):
        """All segments should map to n_embd dimension."""
        n_embd = 128
        host = TransformerHost(vocab_size=100, n_embd=n_embd, n_head=4, n_layer=6, block_size=32)
        for segment, dim in host.segment_channels.items():
            assert dim == n_embd, f"Segment {segment} should have dim {n_embd}, got {dim}"

    def test_forward_to_segment_returns_embeddings(self):
        """forward_to_segment should return hidden states at segment boundary."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        x = torch.randint(0, 100, (2, 16))  # batch=2, seq=16

        h = host.forward_to_segment("mid", x)
        assert h.shape == (2, 16, 64)  # (batch, seq, n_embd)

    def test_forward_from_segment_returns_logits(self):
        """forward_from_segment should return logits from hidden states."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        h = torch.randn(2, 16, 64)  # (batch, seq, n_embd)

        logits = host.forward_from_segment("mid", h)
        assert logits.shape == (2, 16, 100)  # (batch, seq, vocab_size)

    def test_segment_round_trip_matches_forward(self):
        """forward_to_segment + forward_from_segment should match full forward."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        host.eval()  # Disable dropout for deterministic comparison
        x = torch.randint(0, 100, (2, 16))

        # Full forward
        with torch.no_grad():
            full_out = host(x)

        # Segment round-trip through "mid"
        with torch.no_grad():
            h = host.forward_to_segment("mid", x)
            segment_out = host.forward_from_segment("mid", h)

        # Should be identical (deterministic with eval mode)
        torch.testing.assert_close(full_out, segment_out, rtol=1e-5, atol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_host.py -v`
Expected: FAIL with "AttributeError: 'TransformerHost' object has no attribute 'segment_channels'"

**Step 3: Implement segment_channels and segment methods in TransformerHost**

In `src/esper/kasmina/host.py`, modify `TransformerHost.__init__` to add after `self.head.weight = self.tok_emb.weight`:

```python
        # Segment channel counts for multi-slot support
        # For transformers, all segments have the same embedding dimension
        # Segments map to layer ranges: early (0-1), mid (2-3), late (4-5)
        self.segment_channels = {
            "early": n_embd,
            "mid": n_embd,
            "late": n_embd,
        }

        # Layer range boundaries for segments (layer index where segment ENDS)
        # For n_layer=6: early=0-1, mid=2-3, late=4-5
        third = n_layer // 3
        self._segment_boundaries = {
            "early": third,           # Layers 0 to third-1
            "mid": 2 * third,         # Layers third to 2*third-1
            "late": n_layer,          # Layers 2*third to n_layer-1
        }
```

Add new methods after `unregister_slot`:

```python
    def forward_to_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
        """Forward through network up to and including a segment.

        Args:
            segment: Target segment name ("early", "mid", or "late")
            x: Input token indices (B, T)

        Returns:
            Hidden states at the specified segment boundary (B, T, n_embd)
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")

        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        # Embeddings
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        # Forward through layers up to segment boundary
        end_layer = self._segment_boundaries[segment]
        for i in range(end_layer):
            h = self.layers[i](h)
            h = self.slots[self._slot_keys[i]](h)

        return h

    def forward_from_segment(self, segment: str, h: torch.Tensor) -> torch.Tensor:
        """Forward from a segment boundary to output logits.

        Args:
            segment: Starting segment name ("early", "mid", or "late")
            h: Hidden states at segment boundary (B, T, n_embd)

        Returns:
            Output logits (B, T, vocab_size)
        """
        if segment not in self.segment_channels:
            raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")

        # Forward through remaining layers
        start_layer = self._segment_boundaries[segment]
        for i in range(start_layer, self.n_layer):
            h = self.layers[i](h)
            h = self.slots[self._slot_keys[i]](h)

        # Output
        h = self.ln_f(h)
        return self.head(h)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_host.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/test_host.py
git commit -m "feat(kasmina): add segment_channels and segment methods to TransformerHost"
```

---

## Task 2: Remove _legacy_single_slot from MorphogeneticModel

**Files:**
- Modify: `src/esper/kasmina/host.py:359-593` (MorphogeneticModel class)
- Test: `tests/test_morphogenetic_model.py` (new file)

**Step 1: Write the failing test**

```python
# tests/test_morphogenetic_model.py
"""MorphogeneticModel tests for multi-slot architecture."""

import pytest
import torch

from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel


class TestMorphogeneticModelMultiSlot:
    """Test MorphogeneticModel without legacy single-slot mode."""

    def test_requires_slots_parameter(self):
        """MorphogeneticModel must require explicit slots parameter."""
        host = CNNHost(num_classes=10)

        # Without slots parameter should raise
        with pytest.raises(TypeError):
            MorphogeneticModel(host, device="cpu")

    def test_creates_specified_slots(self):
        """MorphogeneticModel should create only the requested slots."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["mid", "late"])

        assert "mid" in model.seed_slots
        assert "late" in model.seed_slots
        assert "early" not in model.seed_slots
        assert len(model.seed_slots) == 2

    def test_no_seed_slot_property(self):
        """MorphogeneticModel should NOT have seed_slot property."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["mid"])

        # Accessing seed_slot should raise AttributeError
        with pytest.raises(AttributeError):
            _ = model.seed_slot

    def test_no_seed_state_property(self):
        """MorphogeneticModel should NOT have seed_state property."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["mid"])

        # Accessing seed_state should raise AttributeError
        with pytest.raises(AttributeError):
            _ = model.seed_state

    def test_germinate_requires_slot(self):
        """germinate_seed must require explicit slot parameter."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["mid"])

        # Without slot should raise TypeError
        with pytest.raises(TypeError):
            model.germinate_seed("residual", "seed_1")

    def test_cull_requires_slot(self):
        """cull_seed must require explicit slot parameter."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["mid"])
        model.germinate_seed("residual", "seed_1", slot="mid")

        # Without slot should raise TypeError
        with pytest.raises(TypeError):
            model.cull_seed()

    def test_forward_with_cnn_host(self):
        """Forward pass should work with CNNHost multi-slot."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["mid"])
        x = torch.randn(2, 3, 32, 32)

        out = model(x)
        assert out.shape == (2, 10)

    def test_forward_with_transformer_host(self):
        """Forward pass should work with TransformerHost multi-slot."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        model = MorphogeneticModel(host, device="cpu", slots=["mid"])
        x = torch.randint(0, 100, (2, 16))

        out = model(x)
        assert out.shape == (2, 16, 100)

    def test_no_legacy_single_slot_attribute(self):
        """MorphogeneticModel should not have _legacy_single_slot attribute."""
        host = CNNHost(num_classes=10)
        model = MorphogeneticModel(host, device="cpu", slots=["mid"])

        assert not hasattr(model, "_legacy_single_slot")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_morphogenetic_model.py -v`
Expected: FAIL (multiple tests will fail due to existing legacy code)

**Step 3: Remove legacy code from MorphogeneticModel**

Replace the entire `MorphogeneticModel` class in `src/esper/kasmina/host.py` with:

```python
class MorphogeneticModel(nn.Module):
    """Model with Kasmina seed slots registered into host injection points.

    Multi-slot architecture for managing multiple concurrent seeds at different
    network segments (early/mid/late).
    """

    def __init__(
        self,
        host: nn.Module,
        device: str = "cpu",
        *,  # Force keyword-only for slots
        slots: list[str],
        task_config=None,
        fast_mode: bool = False,
    ):
        super().__init__()
        self.host = host
        self._device = device
        self.task_config = task_config

        # Host must expose segment_channels for multi-slot support
        segment_channels = host.segment_channels

        # Create seed slots as ModuleDict for proper submodule registration
        slots_dict = {}
        for slot_name in slots:
            if slot_name not in segment_channels:
                raise ValueError(
                    f"Unknown slot: {slot_name}. Available: {list(segment_channels.keys())}"
                )
            slots_dict[slot_name] = SeedSlot(
                slot_id=slot_name,
                channels=segment_channels[slot_name],
                device=device,
                task_config=task_config,
                fast_mode=fast_mode,
            )
        self.seed_slots = nn.ModuleDict(slots_dict)

        # Track slot order for forward pass
        self._slot_order = ["early", "mid", "late"]
        self._active_slots = [s for s in self._slot_order if s in self.seed_slots]

        # Move host to device
        self.host = self.host.to(device)

    def to(self, *args, **kwargs):
        """Override to() to update device tracking after transfer."""
        result = super().to(*args, **kwargs)

        # Query actual device from parameters (canonical source of truth)
        try:
            actual_device = next(self.parameters()).device
        except StopIteration:
            return result

        # Update tracking for all slots
        for slot in self.seed_slots.values():
            slot.device = actual_device
        self._device = str(actual_device)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through host with all active slots.

        Processes sequentially through network segments, applying slot
        transformations at each segment boundary.
        """
        # Detect host type by checking for CNN-specific attributes
        if hasattr(self.host, "blocks"):
            return self._forward_cnn(x)
        else:
            return self._forward_transformer(x)

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for CNN hosts."""
        segment_to_block = {"early": 0, "mid": 1, "late": 2}

        # Convert to channels_last for Tensor Core optimization
        if self.host._memory_format == torch.channels_last:
            x = x.to(memory_format=torch.channels_last)

        current_block = 0
        for slot_id in self._active_slots:
            target_block = segment_to_block[slot_id]

            # Forward through blocks up to this segment
            for idx in range(current_block, target_block + 1):
                x = self.host.blocks[idx](x)
                if idx < self.host._pool_layers:
                    x = self.host.pool(x)

            # Apply slot transformation
            x = self.seed_slots[slot_id].forward(x)
            current_block = target_block + 1

        # Forward through remaining blocks
        for idx in range(current_block, self.host.n_blocks):
            x = self.host.blocks[idx](x)
            if idx < self.host._pool_layers:
                x = self.host.pool(x)

        # Final classification
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.host.classifier(x)

    def _forward_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Transformer hosts."""
        # Use segment methods for clean segmented forward
        current_segment_idx = 0
        segment_order = ["early", "mid", "late"]

        # Initial embedding
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.host.drop(self.host.tok_emb(x) + self.host.pos_emb(pos))

        # Process through segments
        for i, segment in enumerate(segment_order):
            if segment in self._active_slots:
                # Forward through layers for this segment
                start = self.host._segment_boundaries.get(segment_order[i-1], 0) if i > 0 else 0
                end = self.host._segment_boundaries[segment]

                for layer_idx in range(start, end):
                    h = self.host.layers[layer_idx](h)
                    h = self.host.slots[self.host._slot_keys[layer_idx]](h)

                # Apply slot transformation
                h = self.seed_slots[segment].forward(h)
                current_segment_idx = i + 1

        # Forward through remaining layers
        if current_segment_idx < len(segment_order):
            last_processed = segment_order[current_segment_idx - 1] if current_segment_idx > 0 else None
            start = self.host._segment_boundaries.get(last_processed, 0) if last_processed else 0

            for layer_idx in range(start, self.host.n_layer):
                h = self.host.layers[layer_idx](h)
                h = self.host.slots[self.host._slot_keys[layer_idx]](h)

        # Output
        h = self.host.ln_f(h)
        return self.host.head(h)

    def germinate_seed(
        self,
        blueprint_id: str,
        seed_id: str,
        *,  # Force keyword-only
        slot: str,
    ) -> None:
        """Germinate a new seed in a specific slot."""
        if slot not in self.seed_slots:
            raise ValueError(f"Unknown slot: {slot}. Available: {list(self.seed_slots.keys())}")

        self.seed_slots[slot].germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=self.host,
        )

    def cull_seed(self, *, slot: str) -> None:
        """Cull the seed in a specific slot."""
        if slot not in self.seed_slots:
            raise ValueError(f"Unknown slot: {slot}. Available: {list(self.seed_slots.keys())}")
        self.seed_slots[slot].cull()

    def get_seed_parameters(self, slot: str | None = None):
        """Get seed parameters from specific slot or all slots."""
        if slot:
            return self.seed_slots[slot].get_parameters()
        for s in self.seed_slots.values():
            yield from s.get_parameters()

    def get_host_parameters(self):
        """Return host backbone parameters only (exclude seed slots)."""
        for name, param in self.host.named_parameters():
            if "slots" in name:
                continue
            yield param

    @property
    def has_active_seed(self) -> bool:
        """Check if any slot has an active seed."""
        return any(s.is_active for s in self.seed_slots.values())

    def has_active_seed_in_slot(self, slot: str) -> bool:
        """Check if specific slot has active seed."""
        return self.seed_slots[slot].is_active

    def get_slot_states(self) -> dict:
        """Get state of all slots."""
        return {
            slot_id: slot.state
            for slot_id, slot in self.seed_slots.items()
        }

    @property
    def active_seed_params(self) -> int:
        """Total trainable params across all active seeds."""
        return sum(s.active_seed_params for s in self.seed_slots.values())
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_morphogenetic_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py tests/test_morphogenetic_model.py
git commit -m "refactor(kasmina): remove legacy single-slot code from MorphogeneticModel

BREAKING CHANGE: MorphogeneticModel now requires explicit slots parameter.
- Removed _legacy_single_slot flag and all branching
- Removed seed_slot property (use seed_slots dict)
- Removed seed_state property (iterate seed_slots.values())
- germinate_seed() and cull_seed() now require explicit slot parameter"
```

---

## Task 3: Update task factories to use explicit slots

**Files:**
- Modify: `src/esper/runtime/tasks.py:94-176`
- Test: `tests/test_task_spec.py` (existing)

**Step 1: Run existing test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_task_spec.py -v`
Expected: FAIL with TypeError about missing slots parameter

**Step 2: Update task factories**

In `src/esper/runtime/tasks.py`, update all `_make_model` functions:

For `_cifar10_spec` (line 94-98):
```python
    def _make_model(device: str) -> MorphogeneticModel:
        host = CNNHost(num_classes=10, base_channels=8)
        return MorphogeneticModel(host, device=device, slots=["mid"], task_config=cifar_config)
```

For `_cifar10_deep_spec` (line 132-139):
```python
    def _make_model(device: str) -> MorphogeneticModel:
        host = CNNHost(num_classes=10, base_channels=8, n_blocks=5, pool_layers=5)
        return MorphogeneticModel(host, device=device, slots=["mid"], task_config=cifar_config)
```

For `_tinystories_spec` (line 167-176):
```python
    def _make_model(device: str) -> MorphogeneticModel:
        host = TransformerHost(
            vocab_size=vocab_size,
            n_embd=384,
            n_head=6,
            n_layer=6,
            block_size=block_size,
            dropout=0.1,
        )
        return MorphogeneticModel(host, device=device, slots=["mid"], task_config=ts_config)
```

**Step 3: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_task_spec.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/esper/runtime/tasks.py
git commit -m "fix(runtime): update task factories to use explicit slots parameter"
```

---

## Task 4: Update tolaria/trainer.py call sites

**Files:**
- Modify: `src/esper/tolaria/trainer.py:151, 360-361`
- Test: `tests/test_tolaria_trainer.py` (existing)

**Step 1: Analyze current usage**

Lines using `model.seed_slot`:
- Line 151: `seed_slot = model.seed_slot` in `train_epoch_incubator_mode`
- Line 360-361: `seed_slot = model.seed_slot` in `validate_with_attribution`

Both functions assume single-slot mode. For multi-slot, we need to:
1. Accept a slot parameter
2. Default to "mid" for backwards compatibility with callers

**Step 2: Update trainer functions**

In `src/esper/tolaria/trainer.py`:

Update `train_epoch_incubator_mode` signature (line 117) and body:
```python
def train_epoch_incubator_mode(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str = "classification",
    gradient_telemetry_stride: int = 10,
    slot: str = "mid",
) -> None:
    """Train one epoch with seed in isolation (seed output doesn't affect forward pass).
    ...
    Args:
        ...
        slot: Which slot to train (default "mid").
    """
    model.train()
    seed_slot = model.seed_slots[slot]

    # ... rest unchanged
```

Update `validate_with_attribution` signature (line 318) and body:
```python
def validate_with_attribution(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str = "classification",
    slot: str = "mid",
) -> AttributionResult:
    """Counterfactual validation for true seed contribution measurement.
    ...
    Args:
        ...
        slot: Which slot to validate (default "mid").
    """
    was_training = model.training
    model.eval()

    try:
        real_loss, real_accuracy = _run_validation_pass(
            model, testloader, criterion, device, task_type
        )

        seed_slot = model.seed_slots[slot]
        with seed_slot.force_alpha(0.0):
            baseline_loss, baseline_accuracy = _run_validation_pass(
                model, testloader, criterion, device, task_type
            )
        # ... rest unchanged
```

**Step 3: Run tests**

Run: `PYTHONPATH=src pytest tests/test_tolaria_trainer.py -v`
Expected: Some tests may fail - need to update test mocks

**Step 4: Update test mocks**

In `tests/test_tolaria_trainer.py`, update `MockModel` class to use `seed_slots` dict:

```python
class MockModel(nn.Module):
    def __init__(self, with_seed: bool = False):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.seed_slots = {"mid": MockSeedSlot()}
        # ... rest of initialization
```

**Step 5: Run tests again**

Run: `PYTHONPATH=src pytest tests/test_tolaria_trainer.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/tolaria/trainer.py tests/test_tolaria_trainer.py
git commit -m "refactor(tolaria): update trainer to use seed_slots dict with explicit slot parameter"
```

---

## Task 5: Update tolaria/governor.py call sites

**Files:**
- Modify: `src/esper/tolaria/governor.py:182-184`
- Test: Run existing governor tests

**Step 1: Update governor rollback code**

In `src/esper/tolaria/governor.py`, line 182-184:

The current code uses `hasattr` to detect MorphogeneticModel:
```python
if hasattr(self.model, 'seed_slot'):  # hasattr AUTHORIZED...
    self.model.seed_slot.cull("governor_rollback")
```

Change to use `seed_slots` and cull all slots:
```python
# hasattr AUTHORIZED by John on 2025-12-01 16:30:00 UTC
# Justification: Feature detection - MorphogeneticModel has seed_slots, base models may not
if hasattr(self.model, 'seed_slots'):
    for slot in self.model.seed_slots.values():
        slot.cull("governor_rollback")
```

**Step 2: Run tests**

Run: `PYTHONPATH=src pytest tests/ -k governor -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/tolaria/governor.py
git commit -m "refactor(tolaria): update governor to cull all seed_slots on rollback"
```

---

## Task 6: Update scripts/evaluate.py call sites

**Files:**
- Modify: `src/esper/scripts/evaluate.py` (multiple lines)
- Test: Manual run of evaluate script

**Step 1: Identify all seed_slot/seed_state usages**

Lines to update:
- Line 170: `seed_state = model.seed_state` - need helper function
- Line 319: `model.seed_slot.step_epoch()` - iterate all slots
- Line 320: `seed_state = model.seed_state` - use helper
- Line 387: `model.seed_state.stage` - use helper
- Line 390: `model.seed_slot.advance_stage(...)` - specify slot
- Line 393: `model.seed_slot.set_alpha(1.0)` - specify slot

**Step 2: Add helper function and update code**

Add a helper function at the top of the file (after imports):
```python
def get_active_seed_state(model):
    """Get the first active seed state from any slot, or None."""
    for slot in model.seed_slots.values():
        if slot.is_active:
            return slot.state
    return None
```

Update all usages:
- Replace `model.seed_state` with `get_active_seed_state(model)`
- Replace `model.seed_slot` with `model.seed_slots["mid"]` (evaluate.py only uses single slot)

**Step 3: Run evaluate script**

Run: `PYTHONPATH=src python -m esper.scripts.evaluate --help`
Expected: Help text displayed without errors

**Step 4: Commit**

```bash
git add src/esper/scripts/evaluate.py
git commit -m "refactor(scripts): update evaluate.py to use seed_slots dict"
```

---

## Task 7: Update test files

**Files:**
- Modify: `tests/test_lifecycle_fix.py`
- Modify: `tests/test_seed_slot.py`
- Modify: `tests/test_simic_vectorized.py`

**Step 1: Update test_lifecycle_fix.py**

Many lines use `model.seed_state` and `model.seed_slot`. Update all to use:
- `model.seed_slots["mid"]` instead of `model.seed_slot`
- Helper function or direct iteration instead of `model.seed_state`

**Step 2: Update test_seed_slot.py**

Lines 319, 326, 329 use `model.seed_slot`. Update to `model.seed_slots["mid"]`.

**Step 3: Update test_simic_vectorized.py**

The test mocks need to use `seed_slots` dict instead of `seed_state`.

**Step 4: Run full test suite**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS (or identify remaining failures)

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: update all tests to use seed_slots dict instead of legacy accessors"
```

---

## Task 8: Run full test suite and fix remaining issues

**Files:**
- Any files with remaining failures

**Step 1: Run full test suite**

Run: `PYTHONPATH=src pytest tests/ -v 2>&1 | head -100`

**Step 2: Fix any remaining failures**

Address each failure individually.

**Step 3: Run tests again**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: All tests PASS

**Step 4: Final commit**

```bash
git add -A
git commit -m "fix: resolve remaining test failures after legacy code removal"
```

---

## Task 9: Integration test - verify tinystories works

**Files:**
- Test: `tests/test_task_spec.py::test_tinystories_spec_builds_model_and_loaders`

**Step 1: Run the specific failing test**

Run: `PYTHONPATH=src pytest tests/test_task_spec.py::test_tinystories_spec_builds_model_and_loaders -v`
Expected: PASS

**Step 2: Run a quick forward pass**

```python
# Quick smoke test
from esper.runtime import get_task_spec
import torch

spec = get_task_spec("tinystories")
model = spec.create_model(device="cpu")
x = torch.randint(0, 100, (2, 32))
out = model(x)
print(f"Output shape: {out.shape}")  # Should be (2, 32, 50257)
```

**Step 3: Commit if needed**

```bash
git add -A
git commit -m "fix: ensure tinystories task spec works with TransformerHost multi-slot"
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] `PYTHONPATH=src pytest tests/ -v` - All tests pass
- [ ] No `_legacy_single_slot` in codebase: `grep -r "_legacy_single_slot" src/`
- [ ] No `model.seed_slot` direct access: `grep -r "model\.seed_slot[^s]" src/`
- [ ] No `model.seed_state` direct access: `grep -r "model\.seed_state" src/`
- [ ] TransformerHost has segment_channels: `grep "segment_channels" src/esper/kasmina/host.py`
- [ ] tinystories test passes: `pytest tests/test_task_spec.py::test_tinystories_spec_builds_model_and_loaders`

---

## Rollback Plan

If issues are found after deployment:

1. The changes are isolated to:
   - `src/esper/kasmina/host.py` (MorphogeneticModel + TransformerHost)
   - `src/esper/runtime/tasks.py` (task factories)
   - `src/esper/tolaria/trainer.py` (training functions)
   - `src/esper/tolaria/governor.py` (rollback logic)
   - `src/esper/scripts/evaluate.py` (evaluation script)
   - Test files

2. Revert commits in reverse order if needed

3. All changes are additive to TransformerHost and breaking to MorphogeneticModel API

