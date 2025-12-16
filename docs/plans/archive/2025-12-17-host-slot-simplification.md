# Host Slot Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the internal slot system from CNNHost/TransformerHost, making MorphogeneticModel the single interface for seed management.

**Architecture:** Hosts become pure backbone networks with segment routing (`forward_to_segment`, `forward_from_segment`). All slot management moves to MorphogeneticModel which already handles SeedSlots. This eliminates the dual slot system, internal key mapping, and 6 unused per-layer slots in TransformerHost.

**Tech Stack:** PyTorch nn.Module, nn.ModuleDict

---

## Specialist Review Feedback

> **Reviewed by:** PyTorch Specialist (2025-12-17)
> **Verdict:** ✅ Architecturally sound, will improve torch.compile behavior

### Issues to Address During Implementation

| Priority | Issue | Recommendation |
|----------|-------|----------------|
| 🔴 **HIGH** | Segment routing rebuilds dict per call via `injection_specs()` | Cache `_segment_to_block` with `@functools.cached_property` |
| 🟡 **MEDIUM** | Redundant `channels_last` conversion in chained segment calls | Only convert when `from_segment is None` |
| 🟡 **MEDIUM** | No torch.compile verification | Add integration test with `fullgraph=True` |
| 🟢 **LOW** | Old checkpoints will warn about unexpected `slots.*` keys | Document in commit message |
| 🟢 **LOW** | Class docstrings reference old slot behavior | Update to reflect pure backbone role |

### torch.compile Impact (Positive)

The simplified `forward()` methods eliminate guard-heavy patterns:

```python
# BEFORE: Guard overhead from ModuleDict lookup
if idx in self._slot_indices:  # tuple membership check = guard
    x = self.slots[self._slot_keys[slot_idx]](x)  # string dict lookup = guard

# AFTER: Clean loop, no guards
for idx, block in enumerate(self.blocks):
    x = block(x)
```

### Gradient/Parameter Tracking: No Issues

- Removed `nn.ModuleDict` only contained zero-parameter `nn.Identity()` modules
- Gradient flow unchanged (handled by SeedSlot, not host)
- No silent failure risks identified

---

## Background

Currently there are two parallel slot systems:
1. `host.slots` (nn.ModuleDict) - internal, uses keys like `block2_post`, `layer_1_post_block`
2. `MorphogeneticModel.seed_slots` (nn.ModuleDict) - external, uses canonical IDs like `r0c1`

MorphogeneticModel never uses `host.register_slot()` - it routes through `forward_to_segment()` and applies its own SeedSlots. The host slot system is effectively dead code in production.

**Files to modify:**
- `src/esper/kasmina/host.py` - Remove slots from CNNHost and TransformerHost
- `src/esper/kasmina/protocol.py` - Remove register_slot/unregister_slot from HostProtocol
- `tests/kasmina/test_host_protocol.py` - Remove slot registration tests
- `tests/kasmina/test_host_edge_cases.py` - Remove slot error tests
- `tests/kasmina/test_host.py` - Remove direct slot usage tests
- `tests/integration/test_transformer_integration.py` - Refactor to use MorphogeneticModel

---

## Task 1: Update HostProtocol

**Files:**
- Modify: `src/esper/kasmina/protocol.py:17-36`

**Step 1: Remove slot methods from protocol**

Replace the HostProtocol class:

```python
class HostProtocol(Protocol):
    """Contract for graftable host networks.

    Hosts are pure backbone networks that provide:
    - injection_points: Available segment boundaries for seed attachment
    - segment_channels: Channel dimensions at each boundary
    - forward_to_segment/forward_from_segment: Segment routing for MorphogeneticModel
    - forward: Standard backbone forward pass (no slot application)

    Slot management is handled by MorphogeneticModel, not hosts directly.
    """

    @property
    def injection_points(self) -> dict[str, int]:
        """Map of slot_id -> channel/embedding dimension."""
        ...

    @property
    def segment_channels(self) -> dict[str, int]:
        """Map of canonical slot_id -> channel dimension."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through backbone (no slot application)."""
        ...

    def forward_to_segment(self, segment: str, x: Tensor, from_segment: str | None = None) -> Tensor:
        """Forward from input or segment to target segment boundary."""
        ...

    def forward_from_segment(self, segment: str, x: Tensor) -> Tensor:
        """Forward from segment boundary to output."""
        ...
```

**Step 2: Run protocol tests to see what fails**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_host_protocol.py -v`
Expected: Several failures for removed methods

**Step 3: Commit protocol change**

```bash
git add src/esper/kasmina/protocol.py
git commit -m "refactor(kasmina): remove register_slot from HostProtocol

Hosts are now pure backbones. Slot management moves to MorphogeneticModel."
```

---

## Task 2: Simplify CNNHost

**Files:**
- Modify: `src/esper/kasmina/host.py` (CNNHost class, lines ~27-265)

**Step 1: Remove slot infrastructure from __init__**

In CNNHost.__init__, remove these lines:

```python
# DELETE these lines (around line 74-85):
# Slots after each block except the first (aligns with previous block2_post default)
self._slot_indices = tuple(range(1, n_blocks))
# Keep legacy-friendly naming (block2_post) while allowing multiple slots
self._slot_keys = tuple(f"block{idx + 1}_post" for idx in self._slot_indices)
self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

# Build canonical ID -> internal key mapping for register_slot()
from esper.leyline.slot_id import format_slot_id
self._canonical_to_slot_key = {
    format_slot_id(0, idx): key
    for idx, key in zip(self._slot_indices, self._slot_keys)
}
```

**Step 2: Remove register_slot and unregister_slot methods**

Delete the entire `register_slot` method (lines ~128-141) and `unregister_slot` method (lines ~144-155).

**Step 3: Remove @override decorators**

Remove `@override` from `injection_points` property since it no longer overrides a protocol method with slots.

**Step 4: Simplify forward() to remove slot application**

Replace the forward method:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through CNN backbone (no slot application)."""
    # Convert to channels_last ONCE before processing for Tensor Core optimization
    if self._memory_format == torch.channels_last:
        x = x.to(memory_format=torch.channels_last)

    for idx, block in enumerate(self.blocks):
        x = block(x)
        # Only pool on first pool_layers blocks (avoids 0x0 spatial on deep nets)
        if idx < self._pool_layers:
            x = self.pool(x)

    # flatten() handles memory format conversion automatically (returns contiguous)
    x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return self.classifier(x)
```

**Step 5: Add cached segment-to-block mapping** *(addresses HIGH priority specialist feedback)*

Add a cached property to avoid rebuilding the mapping on every `forward_to_segment` call:

```python
@functools.cached_property
def _segment_to_block(self) -> dict[str, int]:
    """Map segment ID to block index (cached)."""
    return {spec.slot_id: spec.layer_range[0] for spec in self.injection_specs()}
```

**Step 6: Simplify forward_to_segment** *(uses cached mapping, optimizes channels_last)*

Replace `forward_to_segment` with optimized version:

```python
def forward_to_segment(
    self,
    segment: str,
    x: torch.Tensor,
    from_segment: str | None = None
) -> torch.Tensor:
    """Forward from one segment boundary to another.

    Args:
        segment: Target segment (e.g., "r0c0", "r0c1", "r0c2")
        x: Raw input if from_segment is None, else features at from_segment boundary
        from_segment: Starting point (None = network input)

    Returns:
        Features at segment boundary
    """
    if segment not in self.segment_channels:
        raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")
    if from_segment is not None and from_segment not in self.segment_channels:
        raise ValueError(f"Unknown from_segment: {from_segment}. Available: {list(self.segment_channels.keys())}")

    # Only convert at entry point (MEDIUM priority fix: avoid redundant conversion)
    if from_segment is None and self._memory_format == torch.channels_last:
        x = x.to(memory_format=torch.channels_last)

    # Use cached mapping (HIGH priority fix: avoid per-call dict rebuilding)
    target_block = self._segment_to_block[segment]
    start_block = 0 if from_segment is None else self._segment_to_block[from_segment] + 1

    # Forward through blocks in range [start_block, target_block]
    for idx in range(start_block, target_block + 1):
        x = self.blocks[idx](x)
        if idx < self._pool_layers:
            x = self.pool(x)

    return x
```

**Step 7: Simplify forward_from_segment** *(uses cached mapping, no redundant conversion)*

```python
def forward_from_segment(self, segment: str, x: torch.Tensor) -> torch.Tensor:
    """Forward from a segment to output.

    Args:
        segment: Starting segment ID ("r0c0", "r0c1", or "r0c2")
        x: Feature map at segment boundary (already in correct memory format)

    Returns:
        Classification logits
    """
    if segment not in self.segment_channels:
        raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")

    # No memory format conversion needed - tensor already converted by forward_to_segment
    # Use cached mapping (HIGH priority fix)
    start_block = self._segment_to_block[segment]

    # Forward through remaining blocks
    for idx in range(start_block + 1, self.n_blocks):
        x = self.blocks[idx](x)
        if idx < self._pool_layers:
            x = self.pool(x)

    # Global average pooling and classification
    x = F.adaptive_avg_pool2d(x, 1).flatten(1)
    return self.classifier(x)
```

**Step 8: Run CNN tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_host.py -k CNN -v`
Expected: Some failures for tests using slots directly

**Step 9: Commit CNNHost changes**

```bash
git add src/esper/kasmina/host.py
git commit -m "refactor(kasmina): remove slot system from CNNHost

CNNHost is now a pure backbone. forward() no longer applies slots.
Slot management handled by MorphogeneticModel."
```

---

## Task 3: Simplify TransformerHost

**Files:**
- Modify: `src/esper/kasmina/host.py` (TransformerHost class, lines ~340-550)

**Step 1: Remove slot infrastructure from __init__**

In TransformerHost.__init__, remove:

```python
# DELETE these lines (around line 378-392):
# Injection points with compile-friendly ModuleDict
self._slot_keys = tuple(f"layer_{i}_post_block" for i in range(n_layer))
self.slots = nn.ModuleDict({k: nn.Identity() for k in self._slot_keys})

# And in the segment boundaries loop, remove:
self._canonical_to_slot_key[slot_id] = self._slot_keys[end_layer - 1]
```

Keep `_segment_boundaries` as it's used for routing.

**Step 2: Add cached segment-to-layer mapping** *(addresses HIGH priority specialist feedback)*

TransformerHost already has `_segment_boundaries` which serves a similar purpose, but ensure it's cached:

```python
# _segment_boundaries is already set in __init__, so no additional caching needed
# Just use self._segment_boundaries[segment] directly in routing methods
```

**Step 3: Remove register_slot and unregister_slot methods**

Delete the entire `register_slot` method (lines ~435-448) and `unregister_slot` method (lines ~451-462).

**Step 4: Simplify forward() to remove per-layer slot application**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through transformer backbone (no slot application)."""
    B, T = x.shape
    if T > self.block_size:
        raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

    # Embeddings
    pos = torch.arange(T, device=x.device)
    h = self.drop(self.tok_emb(x) + self.pos_emb(pos))

    # Transformer layers (no slot application)
    for layer in self.layers:
        h = layer(h)

    # Output
    h = self.ln_f(h)
    return self.head(h)
```

**Step 5: Simplify forward_to_segment**

```python
def forward_to_segment(
    self,
    segment: str,
    x: torch.Tensor,
    from_segment: str | None = None
) -> torch.Tensor:
    """Forward from one segment boundary to another."""
    if segment not in self.segment_channels:
        raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")
    if from_segment is not None and from_segment not in self.segment_channels:
        raise ValueError(f"Unknown from_segment: {from_segment}. Available: {list(self.segment_channels.keys())}")

    # If starting from network input, apply embeddings
    if from_segment is None:
        B, T = x.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        start_layer = 0
    else:
        h = x
        start_layer = self._segment_boundaries[from_segment]

    # Forward through layers in range [start_layer, end_layer)
    end_layer = self._segment_boundaries[segment]
    for i in range(start_layer, end_layer):
        h = self.layers[i](h)

    return h
```

**Step 6: Simplify forward_from_segment**

```python
def forward_from_segment(self, segment: str, h: torch.Tensor) -> torch.Tensor:
    """Forward from a segment boundary to output logits."""
    if segment not in self.segment_channels:
        raise ValueError(f"Unknown segment: {segment}. Available: {list(self.segment_channels.keys())}")

    # Forward through remaining layers
    start_layer = self._segment_boundaries[segment]
    for i in range(start_layer, self.n_layer):
        h = self.layers[i](h)

    # Output
    h = self.ln_f(h)
    return self.head(h)
```

**Step 7: Run Transformer tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_host.py -k Transformer -v`
Expected: Some failures for tests using slots directly

**Step 8: Commit TransformerHost changes**

```bash
git add src/esper/kasmina/host.py
git commit -m "refactor(kasmina): remove slot system from TransformerHost

TransformerHost is now a pure backbone with 0 slot modules (was 6).
forward() no longer applies per-layer slots."
```

---

## Task 4: Update test_host_protocol.py

**Files:**
- Modify: `tests/kasmina/test_host_protocol.py`

**Step 1: Remove slot-related tests**

Delete these test functions entirely:
- `test_host_cnn_register_slot` (line ~44)
- `test_host_cnn_register_invalid_slot_raises` (line ~56)
- `test_host_cnn_unregister_slot` (line ~67)
- `test_host_cnn_forward_with_slot` (line ~95)
- `test_transformer_host_register_unregister` (line ~156)

**Step 2: Update test_host_protocol_has_required_methods**

```python
def test_host_protocol_has_required_methods():
    """HostProtocol should define required interface methods."""
    assert hasattr(HostProtocol, "injection_points")
    assert hasattr(HostProtocol, "segment_channels")
    assert hasattr(HostProtocol, "forward")
    assert hasattr(HostProtocol, "forward_to_segment")
    assert hasattr(HostProtocol, "forward_from_segment")
    # register_slot/unregister_slot removed - slots managed by MorphogeneticModel
```

**Step 3: Update test_host_cnn_forward_with_identity**

Rename to `test_host_cnn_forward` and simplify:

```python
def test_host_cnn_forward():
    """CNNHost forward should work without slots."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    out = host(x)

    assert out.shape == (2, 10)
```

**Step 4: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_host_protocol.py -v`
Expected: All remaining tests pass

**Step 5: Commit**

```bash
git add tests/kasmina/test_host_protocol.py
git commit -m "test(kasmina): remove slot registration tests from protocol tests

Slot management moved to MorphogeneticModel."
```

---

## Task 5: Update test_host_edge_cases.py

**Files:**
- Modify: `tests/kasmina/test_host_edge_cases.py`

**Step 1: Remove TestHostRegistrationErrors class**

Delete the entire `TestHostRegistrationErrors` class (lines ~300-328) which tests:
- `test_cnn_register_unknown_slot_raises`
- `test_cnn_unregister_unknown_slot_raises`
- `test_transformer_register_unknown_slot_raises`
- `test_transformer_unregister_unknown_slot_raises`

**Step 2: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_host_edge_cases.py -v`
Expected: All remaining tests pass

**Step 3: Commit**

```bash
git add tests/kasmina/test_host_edge_cases.py
git commit -m "test(kasmina): remove slot registration error tests

Hosts no longer have register_slot/unregister_slot methods."
```

---

## Task 6: Update test_host.py

**Files:**
- Modify: `tests/kasmina/test_host.py`

**Step 1: Remove test_segment_channels_match_injection_points**

This test references `host.injection_points["block2_post"]` which uses internal keys. Delete it (line ~76-82).

**Step 2: Update test_host_slots_applied test**

Find and delete `test_host_slots_applied` or similar tests that use `host.register_slot()` directly. Around line 200:

```python
# DELETE this test:
def test_host_slots_applied():
    host = CNNHost()
    host.register_slot("block2_post", AddConstant(0.1))
    ...
```

**Step 3: Run tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/test_host.py -v`
Expected: All remaining tests pass

**Step 4: Commit**

```bash
git add tests/kasmina/test_host.py
git commit -m "test(kasmina): remove direct slot usage tests

Direct slot usage removed. Use MorphogeneticModel for slot management."
```

---

## Task 7: Update test_transformer_integration.py

**Files:**
- Modify: `tests/integration/test_transformer_integration.py`

**Step 1: Refactor tests to use MorphogeneticModel**

The tests at lines 26, 36, 52, 71 use `host.register_slot()` directly. Refactor to use MorphogeneticModel:

```python
def test_transformer_with_seed_lifecycle():
    """Test transformer host with seed through MorphogeneticModel."""
    host = TransformerHost(n_layer=6, num_segments=3)
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    # Germinate seed at first segment
    model.germinate_seed("attention", "test_seed", slot="r0c0")

    # Forward should work
    x = torch.randint(0, 1000, (2, 32))
    out = model(x)
    assert out.shape == (2, 32, 50257)
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/integration/test_transformer_integration.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/integration/test_transformer_integration.py
git commit -m "test(integration): use MorphogeneticModel for transformer slot tests

Direct host.register_slot() removed. All slot management through MorphogeneticModel."
```

---

## Task 8: Clean up injection_points property

**Files:**
- Modify: `src/esper/kasmina/host.py`

**Step 1: Remove injection_points property from CNNHost**

The `injection_points` property (line ~120-122) returns internal keys. Since we now use `segment_channels` (canonical IDs), we can either:
- Remove it entirely
- Make it an alias to `segment_channels`

Replace with alias:

```python
@property
def injection_points(self) -> dict[str, int]:
    """Map of slot_id -> channel dimension. Alias for segment_channels."""
    return self.segment_channels
```

**Step 2: Same for TransformerHost**

Replace TransformerHost's `injection_points` (line ~428-430):

```python
@property
def injection_points(self) -> dict[str, int]:
    """Map of slot_id -> embedding dimension. Alias for segment_channels."""
    return self.segment_channels
```

**Step 3: Update class docstrings** *(LOW priority specialist feedback)*

Update CNNHost class docstring to reflect pure backbone role:

```python
class CNNHost(nn.Module):
    """CNN backbone with segment routing for external slot attachment.

    Provides segment boundaries for MorphogeneticModel to attach SeedSlots.
    The host itself performs no slot application - it routes activations
    between segment boundaries.

    ...existing args docstring...
    """
```

Similarly update TransformerHost:

```python
class TransformerHost(nn.Module):
    """Transformer backbone with segment routing for external slot attachment.

    Provides segment boundaries for MorphogeneticModel to attach SeedSlots.
    The host itself performs no slot application - it routes hidden states
    between segment boundaries.

    ...existing args docstring...
    """
```

**Step 4: Run all kasmina tests**

Run: `PYTHONPATH=src uv run pytest tests/kasmina/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/esper/kasmina/host.py
git commit -m "refactor(kasmina): make injection_points alias segment_channels

- Both properties now return canonical IDs
- Updated class docstrings to reflect pure backbone role
- Eliminated internal key exposure"
```

---

## Task 9: Final Verification

**Step 1: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/ -v -m "" --tb=short 2>&1 | tail -50`
Expected: No new failures related to slot changes

**Step 2: Verify MorphogeneticModel still works**

```python
# Quick smoke test
from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel
import torch

# CNN
host = CNNHost()
model = MorphogeneticModel(host, device="cpu", slots=["r0c1", "r0c2"])
model.germinate_seed("norm", "seed1", slot="r0c1")
out = model(torch.randn(2, 3, 32, 32))
assert out.shape == (2, 10)

# Transformer
host2 = TransformerHost()
model2 = MorphogeneticModel(host2, device="cpu", slots=["r0c0", "r0c1"])
out2 = model2(torch.randint(0, 1000, (2, 32)))
assert out2.shape == (2, 32, 50257)

print("All smoke tests pass!")
```

**Step 3: Add torch.compile verification tests** *(MEDIUM priority specialist feedback)*

Create `tests/kasmina/test_host_compile.py`:

```python
"""Verify hosts compile without graph breaks after slot simplification."""
import torch
import pytest
from esper.kasmina.host import CNNHost, TransformerHost


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
def test_cnnhost_compiles_fullgraph():
    """CNNHost.forward() should compile without graph breaks."""
    host = CNNHost().cuda()
    compiled = torch.compile(host, fullgraph=True)
    x = torch.randn(2, 3, 32, 32, device="cuda")

    # Should not raise Dynamo error
    out = compiled(x)
    assert out.shape == (2, 10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
def test_transformerhost_compiles_fullgraph():
    """TransformerHost.forward() should compile without graph breaks."""
    host = TransformerHost().cuda()
    compiled = torch.compile(host, fullgraph=True)
    x = torch.randint(0, 1000, (2, 32), device="cuda")

    out = compiled(x)
    assert out.shape == (2, 32, 50257)


def test_cnnhost_segment_routing_no_graph_breaks():
    """Segment routing should not cause graph breaks (CPU test)."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # These should work without errors even on CPU
    features = host.forward_to_segment("r0c1", x)
    assert features.shape[0] == 2

    out = host.forward_from_segment("r0c1", features)
    assert out.shape == (2, 10)
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "refactor(kasmina): complete host slot simplification

Summary:
- Removed host.slots ModuleDict from CNNHost and TransformerHost
- Removed register_slot/unregister_slot methods
- Hosts are now pure backbones with segment routing
- MorphogeneticModel is the single interface for slot management
- TransformerHost reduced from 6 slots to 0 (was applying Identity after every layer)
- Eliminated internal key system (block2_post, layer_1_post_block)
- All tests updated to use MorphogeneticModel for slot operations
- Added torch.compile verification tests (fullgraph=True)
- Optimized segment routing with cached _segment_to_block mapping

BREAKING CHANGE: state_dicts from previous versions will log warnings about
unexpected 'slots.*' keys. These can be safely ignored - the host slot
system is now managed by MorphogeneticModel."
```

---

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| CNNHost.slots | ModuleDict with 2 entries | Removed |
| TransformerHost.slots | ModuleDict with 6 entries | Removed |
| register_slot() | Method on both hosts | Removed |
| unregister_slot() | Method on both hosts | Removed |
| HostProtocol | Includes slot methods | Pure backbone contract |
| Internal keys | block2_post, layer_1_post_block | Eliminated |
| _canonical_to_slot_key | Mapping dict | Eliminated |
| forward() | Applied slots | Pure backbone forward |
| MorphogeneticModel | Used segment routing | Unchanged (already correct) |
| _segment_to_block | Per-call dict rebuild | `@cached_property` *(specialist fix)* |
| channels_last conversion | In every segment method | Entry point only *(specialist fix)* |
| torch.compile tests | None | `fullgraph=True` verification *(specialist fix)* |

**Lines removed:** ~100 production, ~150 test
**Lines added:** ~50 (torch.compile tests, cached property)
**Complexity reduction:** Single slot system (MorphogeneticModel) instead of two
**Performance improvement:** Segment routing no longer rebuilds dict per call
