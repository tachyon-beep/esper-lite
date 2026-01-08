# Blueprint Compiler & Curriculum Seeds Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the triple-sync blueprint maintenance burden by compiling the BlueprintRegistry into a manifest, then add LayerScale helper and 4 new curriculum blueprints.

**Architecture:** The `BlueprintRegistry` becomes the single source of truth. A `BlueprintCompiler` generates `BlueprintManifest` objects that Tamiyo consumes. The `BlueprintAction` enum is deleted and replaced with manifest lookups. New blueprints only require registration in one place.

**Tech Stack:** Python dataclasses, PyTorch nn.Module, existing BlueprintRegistry pattern

---

## Phase 1: Blueprint Compiler Infrastructure

### Task 1.1: Add action_index to BlueprintSpec

**Files:**
- Modify: `src/esper/kasmina/blueprints/registry.py:35-44`
- Test: `tests/kasmina/test_blueprint_registry.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprint_registry.py`:

```python
def test_blueprint_spec_has_action_index():
    """BlueprintSpec can have explicit action_index for checkpoint stability."""
    from esper.kasmina.blueprints.registry import BlueprintSpec

    spec = BlueprintSpec(
        name="test",
        topology="cnn",
        factory=lambda dim: None,  # type: ignore
        param_estimate=100,
        action_index=5,
    )
    assert spec.action_index == 5

    # Default should be None
    spec_no_idx = BlueprintSpec(
        name="test2",
        topology="cnn",
        factory=lambda dim: None,  # type: ignore
        param_estimate=100,
    )
    assert spec_no_idx.action_index is None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprint_registry.py::test_blueprint_spec_has_action_index -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'action_index'`

**Step 3: Write minimal implementation**

In `src/esper/kasmina/blueprints/registry.py`, modify `BlueprintSpec`:

```python
@dataclass(frozen=True, slots=True)
class BlueprintSpec:
    """Specification for a registered blueprint."""

    name: str
    topology: str
    factory: BlueprintFactory
    param_estimate: int
    description: str = ""
    action_index: int | None = None  # Explicit action ordering for checkpoint stability

    def actual_param_count(self, dim: int) -> int:
        """Compute actual param count for given dimension."""
        module = self.factory(dim)
        return sum(p.numel() for p in module.parameters())
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprint_registry.py::test_blueprint_spec_has_action_index -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/registry.py tests/kasmina/test_blueprint_registry.py
git commit -m "feat(kasmina): add action_index field to BlueprintSpec

Enables explicit action ordering for checkpoint stability when
blueprints are compiled into manifests.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.2: Update BlueprintRegistry.register to accept action_index

**Files:**
- Modify: `src/esper/kasmina/blueprints/registry.py:56-81`
- Test: `tests/kasmina/test_blueprint_registry.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprint_registry.py`:

```python
def test_registry_register_with_action_index():
    """Decorator can specify action_index."""
    from esper.kasmina.blueprints import BlueprintRegistry
    import torch.nn as nn

    @BlueprintRegistry.register(
        "test_indexed", "cnn",
        param_estimate=100,
        action_index=99
    )
    def create_test(dim: int) -> nn.Module:
        return nn.Linear(dim, dim)

    try:
        spec = BlueprintRegistry.get("cnn", "test_indexed")
        assert spec.action_index == 99
    finally:
        BlueprintRegistry.unregister("cnn", "test_indexed")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprint_registry.py::test_registry_register_with_action_index -v
```

Expected: FAIL with `TypeError: register() got an unexpected keyword argument 'action_index'`

**Step 3: Write minimal implementation**

In `src/esper/kasmina/blueprints/registry.py`, modify the `register` method:

```python
@classmethod
def register(
    cls,
    name: str,
    topology: str,
    param_estimate: int,
    description: str = "",
    action_index: int | None = None,
) -> Callable[[BlueprintFactory], BlueprintFactory]:
    """Decorator to register a blueprint factory.

    Args:
        name: Blueprint identifier (e.g., "conv_light", "dilated")
        topology: Target topology ("cnn" or "transformer")
        param_estimate: Approximate parameter count for canonical dim
        description: Human-readable description
        action_index: Explicit action space index for checkpoint stability.
            If None, compiler assigns index based on registration order.

    Note: Action enum caches (in tamiyo.action_enums) are automatically
    invalidated by version-keying on registry state. No callback needed.
    """

    def decorator(factory: BlueprintFactory) -> BlueprintFactory:
        key = f"{topology}:{name}"
        cls._blueprints[key] = BlueprintSpec(
            name=name,
            topology=topology,
            factory=factory,
            param_estimate=param_estimate,
            description=description,
            action_index=action_index,
        )
        return factory

    return decorator
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprint_registry.py::test_registry_register_with_action_index -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/registry.py tests/kasmina/test_blueprint_registry.py
git commit -m "feat(kasmina): register() accepts action_index parameter

Blueprints can now specify explicit action indices for checkpoint
compatibility during manifest compilation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.3: Create BlueprintManifest dataclass

**Files:**
- Create: `src/esper/kasmina/blueprints/manifest.py`
- Test: `tests/kasmina/test_blueprint_manifest.py`

**Step 1: Write the failing test**

Create `tests/kasmina/test_blueprint_manifest.py`:

```python
"""Tests for BlueprintManifest - compiled blueprint action space."""

import pytest


def test_manifest_basic_properties():
    """BlueprintManifest exposes name/index lookups."""
    from esper.kasmina.blueprints.manifest import BlueprintManifest

    manifest = BlueprintManifest(
        topology="cnn",
        names=("noop", "norm", "conv_light"),
        param_estimates=(0, 130, 37000),
        descriptions=("Identity", "GroupNorm", "Light conv"),
    )

    assert manifest.topology == "cnn"
    assert manifest.num_blueprints == 3
    assert manifest.names == ("noop", "norm", "conv_light")
    assert manifest.index_of("norm") == 1
    assert manifest.name_of(2) == "conv_light"


def test_manifest_index_of_unknown_raises():
    """index_of raises KeyError for unknown blueprint."""
    from esper.kasmina.blueprints.manifest import BlueprintManifest

    manifest = BlueprintManifest(
        topology="cnn",
        names=("noop", "norm"),
        param_estimates=(0, 130),
        descriptions=("", ""),
    )

    with pytest.raises(KeyError, match="unknown_blueprint"):
        manifest.index_of("unknown_blueprint")


def test_manifest_is_immutable():
    """BlueprintManifest is frozen dataclass."""
    from esper.kasmina.blueprints.manifest import BlueprintManifest

    manifest = BlueprintManifest(
        topology="cnn",
        names=("noop",),
        param_estimates=(0,),
        descriptions=("",),
    )

    with pytest.raises(AttributeError):
        manifest.topology = "transformer"  # type: ignore
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'esper.kasmina.blueprints.manifest'`

**Step 3: Write minimal implementation**

Create `src/esper/kasmina/blueprints/manifest.py`:

```python
"""BlueprintManifest - Compiled blueprint action space for Tamiyo.

This module provides the compiled view of blueprints that Tamiyo consumes.
The manifest is generated from BlueprintRegistry by BlueprintCompiler,
eliminating the need for manual enum synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True, slots=True)
class BlueprintManifest:
    """Compiled blueprint action space for a topology.

    This is the interface between Kasmina (blueprint definitions) and
    Tamiyo (action space). All blueprint information Tamiyo needs is
    exposed through this manifest.

    Attributes:
        topology: The topology this manifest is for ("cnn" or "transformer")
        names: Blueprint names in action index order
        param_estimates: Parameter estimates in action index order
        descriptions: Human-readable descriptions in action index order
    """

    topology: str
    names: tuple[str, ...]
    param_estimates: tuple[int, ...]
    descriptions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate manifest consistency."""
        if not (len(self.names) == len(self.param_estimates) == len(self.descriptions)):
            raise ValueError(
                f"Manifest length mismatch: names={len(self.names)}, "
                f"param_estimates={len(self.param_estimates)}, "
                f"descriptions={len(self.descriptions)}"
            )

    @property
    def num_blueprints(self) -> int:
        """Number of blueprints in this manifest."""
        return len(self.names)

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        """Mapping from blueprint name to action index."""
        return {name: i for i, name in enumerate(self.names)}

    def index_of(self, name: str) -> int:
        """Get action index for a blueprint name.

        Args:
            name: Blueprint name (e.g., "conv_light", "dilated")

        Returns:
            Action index for use in policy networks.

        Raises:
            KeyError: If blueprint name is not in this manifest.
        """
        try:
            return self.name_to_index[name]
        except KeyError:
            raise KeyError(
                f"Unknown blueprint '{name}' for topology '{self.topology}'. "
                f"Available: {list(self.names)}"
            )

    def name_of(self, index: int) -> str:
        """Get blueprint name for an action index.

        Args:
            index: Action index from policy network.

        Returns:
            Blueprint name.

        Raises:
            IndexError: If index is out of range.
        """
        return self.names[index]


__all__ = ["BlueprintManifest"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/manifest.py tests/kasmina/test_blueprint_manifest.py
git commit -m "feat(kasmina): add BlueprintManifest dataclass

Compiled blueprint action space that Tamiyo consumes. Provides
name/index lookups without hardcoded enum dependencies.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.4: Create BlueprintCompiler

**Files:**
- Modify: `src/esper/kasmina/blueprints/manifest.py`
- Test: `tests/kasmina/test_blueprint_manifest.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprint_manifest.py`:

```python
def test_compiler_generates_manifest_from_registry():
    """BlueprintCompiler.compile() generates manifest from registry."""
    from esper.kasmina.blueprints.manifest import BlueprintCompiler

    manifest = BlueprintCompiler.compile("cnn")

    assert manifest.topology == "cnn"
    assert manifest.num_blueprints > 0
    assert "noop" in manifest.names
    assert "conv_light" in manifest.names
    # Verify param estimates are populated
    noop_idx = manifest.index_of("noop")
    assert manifest.param_estimates[noop_idx] == 0


def test_compiler_caches_manifests():
    """BlueprintCompiler caches compiled manifests."""
    from esper.kasmina.blueprints.manifest import BlueprintCompiler

    BlueprintCompiler.invalidate()  # Clear cache
    m1 = BlueprintCompiler.compile("cnn")
    m2 = BlueprintCompiler.compile("cnn")

    assert m1 is m2  # Same object (cached)


def test_compiler_invalidate_clears_cache():
    """BlueprintCompiler.invalidate() clears the cache."""
    from esper.kasmina.blueprints.manifest import BlueprintCompiler

    m1 = BlueprintCompiler.compile("cnn")
    BlueprintCompiler.invalidate("cnn")
    m2 = BlueprintCompiler.compile("cnn")

    assert m1 is not m2  # New object after invalidation


def test_compiler_sorts_by_action_index():
    """Blueprints with explicit action_index are sorted correctly."""
    from esper.kasmina.blueprints import BlueprintRegistry
    from esper.kasmina.blueprints.manifest import BlueprintCompiler
    import torch.nn as nn

    # Register test blueprints with explicit indices
    @BlueprintRegistry.register("test_z", "cnn", param_estimate=100, action_index=1000)
    def create_z(dim: int) -> nn.Module:
        return nn.Identity()

    @BlueprintRegistry.register("test_a", "cnn", param_estimate=100, action_index=999)
    def create_a(dim: int) -> nn.Module:
        return nn.Identity()

    try:
        BlueprintCompiler.invalidate("cnn")
        manifest = BlueprintCompiler.compile("cnn")

        # test_a (999) should come before test_z (1000) regardless of name
        idx_a = manifest.index_of("test_a")
        idx_z = manifest.index_of("test_z")
        assert idx_a < idx_z, f"test_a ({idx_a}) should be before test_z ({idx_z})"
    finally:
        BlueprintRegistry.unregister("cnn", "test_z")
        BlueprintRegistry.unregister("cnn", "test_a")
        BlueprintCompiler.invalidate("cnn")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py::test_compiler_generates_manifest_from_registry -v
```

Expected: FAIL with `ImportError: cannot import name 'BlueprintCompiler'`

**Step 3: Write minimal implementation**

Add to `src/esper/kasmina/blueprints/manifest.py`:

```python
from typing import ClassVar


class BlueprintCompiler:
    """Compiles BlueprintRegistry into topology-specific manifests.

    The compiler is the bridge between Kasmina's blueprint definitions
    and Tamiyo's action space. It reads the registry and produces
    immutable manifests that define the action indices.

    Caching: Manifests are cached by topology. Call invalidate() when
    the registry changes (e.g., in tests that register/unregister blueprints).
    """

    _cache: ClassVar[dict[str, BlueprintManifest]] = {}

    @classmethod
    def compile(cls, topology: str) -> BlueprintManifest:
        """Compile a manifest for the given topology.

        Args:
            topology: Target topology ("cnn" or "transformer")

        Returns:
            Compiled BlueprintManifest with action indices assigned.

        Blueprint ordering:
            1. Blueprints with explicit action_index (sorted ascending)
            2. Blueprints without action_index (sorted by name for stability)
        """
        if topology in cls._cache:
            return cls._cache[topology]

        from esper.kasmina.blueprints import BlueprintRegistry

        specs = BlueprintRegistry.list_for_topology(topology)

        # Sort: explicit action_index first (ascending), then by name
        def sort_key(spec):
            if spec.action_index is not None:
                return (0, spec.action_index, spec.name)
            return (1, 0, spec.name)

        sorted_specs = sorted(specs, key=sort_key)

        manifest = BlueprintManifest(
            topology=topology,
            names=tuple(s.name for s in sorted_specs),
            param_estimates=tuple(s.param_estimate for s in sorted_specs),
            descriptions=tuple(s.description for s in sorted_specs),
        )

        cls._cache[topology] = manifest
        return manifest

    @classmethod
    def invalidate(cls, topology: str | None = None) -> None:
        """Clear cached manifests.

        Args:
            topology: Specific topology to invalidate, or None for all.
        """
        if topology is not None:
            cls._cache.pop(topology, None)
        else:
            cls._cache.clear()


__all__ = ["BlueprintManifest", "BlueprintCompiler"]
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/manifest.py tests/kasmina/test_blueprint_manifest.py
git commit -m "feat(kasmina): add BlueprintCompiler for manifest generation

Compiles BlueprintRegistry into topology-specific manifests with
deterministic action index ordering. Eliminates need for manual
enum synchronization.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.5: Assign action_index to existing CNN blueprints

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py`
- Test: `tests/kasmina/test_blueprint_manifest.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprint_manifest.py`:

```python
def test_cnn_manifest_matches_legacy_blueprint_action_order():
    """CNN manifest indices match legacy BlueprintAction enum values.

    This ensures checkpoint compatibility - existing checkpoints store
    action indices that must map to the same blueprints.

    Legacy BlueprintAction values:
        NOOP=0, CONV_LIGHT=1, ATTENTION=2, NORM=3, DEPTHWISE=4,
        BOTTLENECK=5, CONV_SMALL=6, CONV_HEAVY=7
    """
    from esper.kasmina.blueprints.manifest import BlueprintCompiler

    BlueprintCompiler.invalidate("cnn")
    manifest = BlueprintCompiler.compile("cnn")

    # These must match BlueprintAction enum values for checkpoint compat
    expected_order = {
        "noop": 0,
        "conv_light": 1,
        "attention": 2,
        "norm": 3,
        "depthwise": 4,
        "bottleneck": 5,
        "conv_small": 6,
        "conv_heavy": 7,
    }

    for name, expected_idx in expected_order.items():
        actual_idx = manifest.index_of(name)
        assert actual_idx == expected_idx, (
            f"Blueprint '{name}' has index {actual_idx}, expected {expected_idx} "
            f"for checkpoint compatibility"
        )
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py::test_cnn_manifest_matches_legacy_blueprint_action_order -v
```

Expected: FAIL (indices don't match because no action_index set)

**Step 3: Write minimal implementation**

Modify `src/esper/kasmina/blueprints/cnn.py` to add `action_index` to each blueprint:

```python
@BlueprintRegistry.register(
    "noop", "cnn",
    param_estimate=0,
    action_index=0,
    description="Identity seed - placeholder before bursting"
)
def create_noop_seed(dim: int, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

```python
@BlueprintRegistry.register(
    "norm", "cnn",
    param_estimate=130,
    action_index=3,
    description="GroupNorm enhancement"
)
def create_norm_seed(dim: int, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

```python
@BlueprintRegistry.register(
    "attention", "cnn",
    param_estimate=2000,
    action_index=2,
    description="SE-style channel attention"
)
def create_attention_seed(dim: int, reduction: int = 4, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

```python
@BlueprintRegistry.register(
    "depthwise", "cnn",
    param_estimate=4800,
    action_index=4,
    description="Depthwise-separable conv"
)
def create_depthwise_seed(dim: int, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

```python
@BlueprintRegistry.register(
    "bottleneck", "cnn",
    param_estimate=4500,
    action_index=5,
    description="Bottleneck conv (1x1→3x3→1x1) - same tier as conv_small"
)
def create_bottleneck_seed(dim: int, reduction: int = 4, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

```python
@BlueprintRegistry.register(
    "conv_small", "cnn",
    param_estimate=4200,
    action_index=6,
    description="Small 1x1 conv - same tier as bottleneck"
)
def create_conv_small_seed(dim: int, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

```python
@BlueprintRegistry.register(
    "conv_light", "cnn",
    param_estimate=37000,
    action_index=1,
    description="Light conv block"
)
def create_conv_light_seed(dim: int, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

```python
@BlueprintRegistry.register(
    "conv_heavy", "cnn",
    param_estimate=74000,
    action_index=7,
    description="Heavy conv block"
)
def create_conv_heavy_seed(dim: int, **kwargs: Any) -> nn.Module:
    # ... existing implementation unchanged
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py::test_cnn_manifest_matches_legacy_blueprint_action_order -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprint_manifest.py
git commit -m "feat(kasmina): assign action_index to CNN blueprints

Explicit indices match legacy BlueprintAction enum values for
checkpoint compatibility. New blueprints will use indices >= 8.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.6: Assign action_index to existing transformer blueprints

**Files:**
- Modify: `src/esper/kasmina/blueprints/transformer.py`
- Test: `tests/kasmina/test_blueprint_manifest.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprint_manifest.py`:

```python
def test_transformer_manifest_matches_legacy_blueprint_action_order():
    """Transformer manifest indices match legacy BlueprintAction enum values.

    Legacy BlueprintAction values for transformer:
        NOOP=0, NORM=3 (shared), ATTENTION=2 (shared),
        LORA=8, LORA_LARGE=9, MLP_SMALL=10, MLP=11, FLEX_ATTENTION=12
    """
    from esper.kasmina.blueprints.manifest import BlueprintCompiler

    BlueprintCompiler.invalidate("transformer")
    manifest = BlueprintCompiler.compile("transformer")

    # These must match BlueprintAction enum values
    expected_order = {
        "noop": 0,
        "attention": 2,
        "norm": 3,
        "lora": 8,
        "lora_large": 9,
        "mlp_small": 10,
        "mlp": 11,
        "flex_attention": 12,
    }

    for name, expected_idx in expected_order.items():
        if name not in manifest.name_to_index:
            # flex_attention may not be available on all torch builds
            continue
        actual_idx = manifest.index_of(name)
        assert actual_idx == expected_idx, (
            f"Blueprint '{name}' has index {actual_idx}, expected {expected_idx}"
        )
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py::test_transformer_manifest_matches_legacy_blueprint_action_order -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

First, read the transformer blueprints file to see current structure, then add action_index to each. The transformer blueprints file is at `src/esper/kasmina/blueprints/transformer.py`. Add `action_index` matching the BlueprintAction enum:

- `noop`: action_index=0 (shared with CNN)
- `norm`: action_index=3 (shared with CNN)
- `attention`: action_index=2 (shared with CNN)
- `lora`: action_index=8
- `lora_large`: action_index=9
- `mlp_small`: action_index=10
- `mlp`: action_index=11
- `flex_attention`: action_index=12

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py::test_transformer_manifest_matches_legacy_blueprint_action_order -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/transformer.py tests/kasmina/test_blueprint_manifest.py
git commit -m "feat(kasmina): assign action_index to transformer blueprints

Explicit indices match legacy BlueprintAction enum values.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 1.7: Export manifest from blueprints package

**Files:**
- Modify: `src/esper/kasmina/blueprints/__init__.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprint_manifest.py`:

```python
def test_manifest_importable_from_blueprints_package():
    """BlueprintManifest and BlueprintCompiler are public exports."""
    from esper.kasmina.blueprints import BlueprintCompiler, BlueprintManifest

    assert BlueprintManifest is not None
    assert BlueprintCompiler is not None
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py::test_manifest_importable_from_blueprints_package -v
```

Expected: FAIL with `ImportError: cannot import name 'BlueprintManifest'`

**Step 3: Write minimal implementation**

Add to `src/esper/kasmina/blueprints/__init__.py`:

```python
from .manifest import BlueprintCompiler, BlueprintManifest

__all__ = [
    # ... existing exports ...
    "BlueprintCompiler",
    "BlueprintManifest",
]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprint_manifest.py::test_manifest_importable_from_blueprints_package -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/__init__.py tests/kasmina/test_blueprint_manifest.py
git commit -m "feat(kasmina): export BlueprintManifest and BlueprintCompiler

Public API for blueprint compilation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Leyline Migration (Replace BlueprintAction enum)

### Task 2.1: Add manifest accessor functions to leyline

**Files:**
- Modify: `src/esper/leyline/factored_actions.py`
- Create: `tests/leyline/test_blueprint_manifest_accessors.py`

**Step 1: Write the failing test**

Create `tests/leyline/test_blueprint_manifest_accessors.py`:

```python
"""Tests for leyline blueprint manifest accessors."""


def test_get_blueprint_manifest_returns_manifest():
    """get_blueprint_manifest() returns compiled manifest."""
    from esper.leyline import get_blueprint_manifest

    manifest = get_blueprint_manifest("cnn")
    assert manifest.topology == "cnn"
    assert manifest.num_blueprints > 0


def test_get_num_blueprints_from_manifest():
    """get_num_blueprints() uses manifest."""
    from esper.leyline import get_num_blueprints

    num = get_num_blueprints("cnn")
    assert num >= 8  # At least the original 8 CNN blueprints


def test_get_blueprint_name():
    """get_blueprint_name() looks up from manifest."""
    from esper.leyline import get_blueprint_name

    # Index 0 should be "noop" for both topologies
    assert get_blueprint_name("cnn", 0) == "noop"
    assert get_blueprint_name("transformer", 0) == "noop"


def test_get_blueprint_index():
    """get_blueprint_index() looks up from manifest."""
    from esper.leyline import get_blueprint_index

    assert get_blueprint_index("cnn", "noop") == 0
    assert get_blueprint_index("cnn", "conv_light") == 1


def test_is_valid_blueprint_for_topology():
    """is_valid_blueprint_for_topology() validates against manifest."""
    from esper.leyline import is_valid_blueprint_for_topology

    # CNN blueprints
    assert is_valid_blueprint_for_topology("cnn", "conv_light")
    assert is_valid_blueprint_for_topology("cnn", "noop")

    # Transformer blueprints
    assert is_valid_blueprint_for_topology("transformer", "lora")

    # Cross-topology should fail
    assert not is_valid_blueprint_for_topology("cnn", "lora")
    assert not is_valid_blueprint_for_topology("transformer", "conv_light")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/leyline/test_blueprint_manifest_accessors.py -v
```

Expected: FAIL with `ImportError: cannot import name 'get_blueprint_manifest'`

**Step 3: Write minimal implementation**

Add to `src/esper/leyline/factored_actions.py` (at the end, before `__all__`):

```python
# =============================================================================
# Manifest-Based Blueprint Accessors
# =============================================================================
# These functions provide the manifest-based interface for blueprint lookups.
# They replace direct BlueprintAction enum access for forward compatibility.

def get_blueprint_manifest(topology: str) -> "BlueprintManifest":
    """Get compiled blueprint manifest for a topology.

    This is the primary interface for accessing blueprint information.
    The manifest is compiled from BlueprintRegistry and cached.

    Args:
        topology: Target topology ("cnn" or "transformer")

    Returns:
        BlueprintManifest with names, indices, and metadata.
    """
    from esper.kasmina.blueprints import BlueprintCompiler
    return BlueprintCompiler.compile(topology)


def get_num_blueprints(topology: str) -> int:
    """Get number of blueprints for a topology.

    Args:
        topology: Target topology ("cnn" or "transformer")

    Returns:
        Number of blueprints in the manifest.
    """
    return get_blueprint_manifest(topology).num_blueprints


def get_blueprint_name(topology: str, index: int) -> str:
    """Get blueprint name for an action index.

    Args:
        topology: Target topology ("cnn" or "transformer")
        index: Action index from policy network

    Returns:
        Blueprint name (e.g., "conv_light", "lora")
    """
    return get_blueprint_manifest(topology).name_of(index)


def get_blueprint_index(topology: str, name: str) -> int:
    """Get action index for a blueprint name.

    Args:
        topology: Target topology ("cnn" or "transformer")
        name: Blueprint name (e.g., "conv_light", "lora")

    Returns:
        Action index for use in policy networks.
    """
    return get_blueprint_manifest(topology).index_of(name)


def is_valid_blueprint_for_topology(topology: str, name: str) -> bool:
    """Check if a blueprint is valid for a topology.

    Args:
        topology: Target topology ("cnn" or "transformer")
        name: Blueprint name to check

    Returns:
        True if the blueprint is registered for this topology.
    """
    manifest = get_blueprint_manifest(topology)
    return name in manifest.name_to_index
```

Also add the type import at the top of the file:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esper.kasmina.blueprints import BlueprintManifest
```

And update `__all__` to include the new functions:

```python
__all__ = [
    # ... existing exports ...
    # Manifest-based accessors
    "get_blueprint_manifest",
    "get_num_blueprints",
    "get_blueprint_name",
    "get_blueprint_index",
    "is_valid_blueprint_for_topology",
]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/leyline/test_blueprint_manifest_accessors.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/factored_actions.py tests/leyline/test_blueprint_manifest_accessors.py
git commit -m "feat(leyline): add manifest-based blueprint accessors

Functions for blueprint lookup using compiled manifests instead of
hardcoded BlueprintAction enum. Enables single-source-of-truth
blueprint registration.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.2: Update leyline __init__ to export new accessors

**Files:**
- Modify: `src/esper/leyline/__init__.py`

**Step 1: Write the failing test**

The test from Task 2.1 already imports from `esper.leyline`. If exports aren't in `__init__.py`, it will fail.

```bash
uv run pytest tests/leyline/test_blueprint_manifest_accessors.py -v
```

**Step 2: Run test to verify current state**

If it's failing due to import, proceed to Step 3.

**Step 3: Write minimal implementation**

Add to `src/esper/leyline/__init__.py`:

```python
from .factored_actions import (
    # ... existing imports ...
    get_blueprint_manifest,
    get_num_blueprints,
    get_blueprint_name,
    get_blueprint_index,
    is_valid_blueprint_for_topology,
)
```

And update `__all__`.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/leyline/test_blueprint_manifest_accessors.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "feat(leyline): export blueprint manifest accessors

Public API for manifest-based blueprint lookups.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.3: Migrate action_masks.py to use manifest

**Files:**
- Modify: `src/esper/tamiyo/policy/action_masks.py:28-34,216-227`
- Test: `tests/tamiyo/policy/test_action_masks.py`

**Step 1: Write the failing test**

Add to `tests/tamiyo/policy/test_action_masks.py` (or create if needed):

```python
def test_blueprint_mask_uses_manifest():
    """Blueprint mask is generated from manifest, not hardcoded set."""
    from esper.tamiyo.policy.action_masks import compute_action_masks
    from esper.leyline import get_blueprint_manifest
    from esper.leyline.slot_config import SlotConfig

    slot_config = SlotConfig.default()
    enabled_slots = list(slot_config.slot_ids)
    slot_states = {slot_id: None for slot_id in enabled_slots}

    masks = compute_action_masks(
        slot_states=slot_states,
        enabled_slots=enabled_slots,
        total_seeds=0,
        max_seeds=3,
        slot_config=slot_config,
        topology="cnn",
    )

    manifest = get_blueprint_manifest("cnn")
    blueprint_mask = masks["blueprint"]

    # All blueprints in manifest should be valid except NOOP
    noop_idx = manifest.index_of("noop")
    for i, name in enumerate(manifest.names):
        if name == "noop":
            assert not blueprint_mask[i], "NOOP should be masked out"
        else:
            assert blueprint_mask[i], f"Blueprint {name} should be valid"
```

**Step 2: Run test to verify it passes (should already work)**

```bash
uv run pytest tests/tamiyo/policy/test_action_masks.py::test_blueprint_mask_uses_manifest -v
```

**Step 3: Refactor implementation to use manifest**

Modify `src/esper/tamiyo/policy/action_masks.py`:

Replace the imports:
```python
# OLD
from esper.leyline import (
    BlueprintAction,
    CNN_BLUEPRINTS,
    ...
)

# NEW
from esper.leyline import (
    get_blueprint_manifest,
    ...
)
```

Replace the blueprint mask generation (around line 216-227):
```python
# OLD
blueprint_mask = torch.zeros(NUM_BLUEPRINTS, dtype=torch.bool, device=device)
valid_blueprints = TRANSFORMER_BLUEPRINTS if topology == "transformer" else CNN_BLUEPRINTS
for bp in valid_blueprints:
    blueprint_mask[bp] = True
blueprint_mask[BlueprintAction.NOOP] = False

# NEW
manifest = get_blueprint_manifest(topology)
blueprint_mask = torch.ones(manifest.num_blueprints, dtype=torch.bool, device=device)
# NOOP is a placeholder - always disable for germination
noop_idx = manifest.index_of("noop")
blueprint_mask[noop_idx] = False
```

Also update `NUM_BLUEPRINTS` usage to `manifest.num_blueprints` where needed.

**Step 4: Run tests to verify refactor doesn't break anything**

```bash
uv run pytest tests/tamiyo/policy/test_action_masks.py -v
uv run pytest tests/leyline/test_action_schema_drift.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tamiyo/policy/action_masks.py
git commit -m "refactor(tamiyo): migrate action_masks to use blueprint manifest

Replaces CNN_BLUEPRINTS/TRANSFORMER_BLUEPRINTS frozensets with
manifest-based validation. Blueprint mask now derived from
compiled manifest.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.4: Migrate action_enums.py to use manifest

**Files:**
- Modify: `src/esper/tamiyo/action_enums.py`
- Test: `tests/kasmina/test_blueprint_registry.py::test_action_enum_reflects_registry_changes`

**Step 1: Understand current behavior**

The test `test_action_enum_reflects_registry_changes` asserts that `build_action_enum` does NOT change when registry changes (because it uses static `BlueprintAction`).

With the manifest approach, this behavior changes - the enum SHOULD reflect registry changes. We need to update the test expectation.

**Step 2: Update the test**

Modify `tests/kasmina/test_blueprint_registry.py::test_action_enum_reflects_registry_changes`:

```python
def test_action_enum_reflects_registry_changes():
    """Action enum now reflects registry changes via manifest.

    With the blueprint compiler, dynamically registered blueprints
    ARE included in the action enum (manifest is recompiled).
    """
    from esper.kasmina.blueprints import BlueprintRegistry, BlueprintCompiler
    from esper.tamiyo.action_enums import build_action_enum, clear_action_enum_cache

    # Build initial enum
    BlueprintCompiler.invalidate("cnn")
    clear_action_enum_cache()
    enum_before = build_action_enum("cnn")
    initial_members = set(enum_before.__members__.keys())

    # Register a test blueprint with high action_index
    @BlueprintRegistry.register(
        "test_registry_change", "cnn",
        param_estimate=99999,
        action_index=100,  # High index to not conflict
    )
    def create_test(dim: int) -> nn.Module:
        return nn.Linear(dim, dim)

    try:
        # Rebuild enum - NOW includes the new blueprint
        BlueprintCompiler.invalidate("cnn")
        clear_action_enum_cache()
        enum_after = build_action_enum("cnn")
        new_members = set(enum_after.__members__.keys())

        assert "GERMINATE_TEST_REGISTRY_CHANGE" in new_members

    finally:
        # Clean up
        BlueprintRegistry.unregister("cnn", "test_registry_change")
        BlueprintCompiler.invalidate("cnn")

    # Final rebuild should NOT have the test blueprint
    clear_action_enum_cache()
    enum_final = build_action_enum("cnn")
    final_members = set(enum_final.__members__.keys())
    assert "GERMINATE_TEST_REGISTRY_CHANGE" not in final_members
    assert final_members == initial_members, "Should return to original state"
```

**Step 3: Refactor action_enums.py to use manifest**

Modify `src/esper/tamiyo/action_enums.py`:

```python
"""Tamiyo Action Enums - Dynamic action enum construction for heuristic policies.

This module builds topology-specific action enums from the blueprint manifest.
The enums are used by HeuristicTamiyo for baseline comparison.
"""

from enum import IntEnum

from esper.leyline import get_blueprint_manifest


# Cache for built enums, keyed by topology
_action_enum_cache: dict[str, type[IntEnum]] = {}


def build_action_enum(topology: str) -> type[IntEnum]:
    """Build action enum from blueprint manifest for a topology.

    Action layout:
        0: WAIT
        1-N: GERMINATE_<BLUEPRINT> (in manifest order, NOOP excluded)
        N+1: FOSSILIZE
        N+2: PRUNE
        N+3: ADVANCE

    Args:
        topology: The topology type ("cnn" or "transformer")

    Returns:
        IntEnum class with action members for this topology.
    """
    if topology in _action_enum_cache:
        return _action_enum_cache[topology]

    manifest = get_blueprint_manifest(topology)

    # Filter out "noop" and keep manifest order
    germination_blueprints = [
        (name, idx) for idx, name in enumerate(manifest.names)
        if name != "noop"
    ]

    members = {"WAIT": 0}
    for i, (name, _) in enumerate(germination_blueprints, start=1):
        # Use uppercase name (e.g., conv_light -> GERMINATE_CONV_LIGHT)
        members[f"GERMINATE_{name.upper()}"] = i

    members["FOSSILIZE"] = len(germination_blueprints) + 1
    members["PRUNE"] = len(germination_blueprints) + 2
    members["ADVANCE"] = len(germination_blueprints) + 3

    member_list = list(members.items())
    action_enum = IntEnum(f"{topology.title()}Action", member_list)  # type: ignore[misc]
    _action_enum_cache[topology] = action_enum
    return action_enum


def clear_action_enum_cache() -> None:
    """Clear the action enum cache (for testing)."""
    _action_enum_cache.clear()


__all__ = ["build_action_enum", "clear_action_enum_cache"]
```

**Step 4: Run tests**

```bash
uv run pytest tests/kasmina/test_blueprint_registry.py::test_action_enum_reflects_registry_changes -v
uv run pytest tests/tamiyo/ -v -k "action"
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/tamiyo/action_enums.py tests/kasmina/test_blueprint_registry.py
git commit -m "refactor(tamiyo): migrate action_enums to use manifest

Action enums now built from compiled blueprint manifest instead of
static BlueprintAction enum. Enables dynamic blueprint registration.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.5: Migrate simic/training/helpers.py

**Files:**
- Modify: `src/esper/simic/training/helpers.py:361-375`

**Step 1: Identify usage**

The file uses `BlueprintAction[blueprint_name_upper]` to convert action names back to indices. This needs to use manifest lookup instead.

**Step 2: Refactor**

Replace:
```python
from esper.leyline import AlphaTargetAction, BlueprintAction
# ...
blueprint = BlueprintAction[blueprint_name_upper]
```

With:
```python
from esper.leyline import AlphaTargetAction, get_blueprint_index
# ...
# Need topology context - add parameter or infer from action name
blueprint_idx = get_blueprint_index(topology, blueprint_name_lower)
```

This requires understanding the calling context to determine topology. Review the function signature and callers.

**Step 3: Run tests**

```bash
uv run pytest tests/simic/training/ -v
```

**Step 4: Commit**

```bash
git add src/esper/simic/training/helpers.py
git commit -m "refactor(simic): migrate helpers.py to use manifest

Blueprint lookups use get_blueprint_index() instead of BlueprintAction enum.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2.6: Update remaining consumers and run full test suite

**Files:**
- Potentially: `src/esper/leyline/reports.py`
- Potentially: `src/esper/karn/sanctum/widgets/tamiyo/action_heads_panel.py`
- Potentially: `src/esper/tamiyo/networks/factored_lstm.py`

**Step 1: Search for remaining BlueprintAction usage**

```bash
uv run grep -r "BlueprintAction" src/esper/ --include="*.py"
```

For each file found, migrate to use manifest functions.

**Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Fix any failures.

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: complete BlueprintAction migration to manifest

All blueprint lookups now use compiled manifest. BlueprintAction
enum retained for backwards compatibility but marked for deprecation.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: LayerScale Helper & Fix Existing Blueprints

### Task 3.1: Add LayerScale module

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprints.py`:

```python
def test_layer_scale_multiplies_by_gamma():
    """LayerScale applies per-channel learned scaling."""
    import torch
    from esper.kasmina.blueprints.cnn import LayerScale

    ls = LayerScale(channels=64, init_value=0.1)
    x = torch.ones(2, 64, 8, 8)
    y = ls(x)

    # Output should be input * 0.1 (init_value)
    assert torch.allclose(y, x * 0.1)

    # Gamma should be learnable
    assert ls.gamma.requires_grad


def test_layer_scale_preserves_shape():
    """LayerScale preserves input shape."""
    import torch
    from esper.kasmina.blueprints.cnn import LayerScale

    ls = LayerScale(channels=32)
    x = torch.randn(4, 32, 16, 16)
    y = ls(x)

    assert y.shape == x.shape
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_layer_scale_multiplies_by_gamma -v
```

Expected: FAIL with `ImportError: cannot import name 'LayerScale'`

**Step 3: Write minimal implementation**

Add to `src/esper/kasmina/blueprints/cnn.py` (after imports, before first blueprint):

```python
class LayerScale(nn.Module):
    """Per-channel residual scaling (ConvNeXt/CaiT style).

    Multiplies input by a learned per-channel factor initialized to a small
    value (default 1e-3). This allows residual branches to start near-identity
    while maintaining gradient flow, avoiding the dead-branch trap of
    zero-init under ReLU.

    Args:
        channels: Number of channels to scale
        init_value: Initial value for scale factors (default 1e-3)

    Reference:
        - ConvNeXt: https://arxiv.org/abs/2201.03545
        - CaiT: https://arxiv.org/abs/2103.17239
    """

    def __init__(self, channels: int, init_value: float = 1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((channels,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape gamma for broadcasting: [C] -> [1, C, 1, 1]
        return x * self.gamma.view(1, -1, 1, 1)
```

Also add to `__all__`:
```python
__all__ = ["ConvBlock", "SeedConvBlock", "get_num_groups", "LayerScale"]
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_layer_scale_multiplies_by_gamma tests/kasmina/test_blueprints.py::test_layer_scale_preserves_shape -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprints.py
git commit -m "feat(kasmina): add LayerScale module for safe residual scaling

Per-channel learned scaling (ConvNeXt/CaiT pattern) that avoids
the dead-branch trap of zero-init under ReLU. Initial value 1e-3
provides near-identity start with maintained gradients.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3.2: Fix bottleneck blueprint dead-branch risk

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py:203-239`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprints.py`:

```python
def test_bottleneck_has_gradient_at_init():
    """Bottleneck should have non-zero gradients at initialization.

    The old zero-init pattern under ReLU caused dead branches.
    LayerScale fix ensures gradients flow.
    """
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "bottleneck", dim=64)
    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    y = seed(x)
    loss = y.sum()
    loss.backward()

    # All parameters should have gradients
    for name, param in seed.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} has zero gradient"
```

**Step 2: Run test to check current state**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_bottleneck_has_gradient_at_init -v
```

This might pass or fail depending on current init. If it fails (zero gradients), proceed.

**Step 3: Refactor implementation**

Modify `BottleneckSeed` class in `src/esper/kasmina/blueprints/cnn.py`:

```python
@BlueprintRegistry.register(
    "bottleneck", "cnn",
    param_estimate=4500,
    action_index=5,
    description="Bottleneck conv (1x1→3x3→1x1) - same tier as conv_small"
)
def create_bottleneck_seed(dim: int, reduction: int = 4, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Bottleneck convolution seed using 1x1 → 3x3 → 1x1 structure.

    Uses LayerScale instead of zero-init to avoid dead-branch trap.
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for cnn/bottleneck: {sorted(kwargs)}")

    class BottleneckSeed(nn.Module):
        def __init__(self, channels: int, reduction: int, layer_scale_init: float):
            super().__init__()
            bottleneck_dim = max(8, channels // reduction)
            self.down = nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=False)
            self.conv = nn.Conv2d(
                bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1, bias=False
            )
            self.up = nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.relu(self.down(x))
            y = F.relu(self.conv(y))
            y = F.relu(self.gn(self.up(y)))
            return x + self.ls(y)

    return BottleneckSeed(dim, reduction, layer_scale)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_bottleneck_has_gradient_at_init -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprints.py
git commit -m "fix(kasmina): bottleneck uses LayerScale to avoid dead branch

Replaces zero-init pattern that caused dead branches when output
goes through ReLU(0) -> zero gradient. LayerScale provides
near-identity start with maintained gradient flow.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3.3: Fix conv_small blueprint dead-branch risk

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py:242-271`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprints.py`:

```python
def test_conv_small_has_gradient_at_init():
    """ConvSmall should have non-zero gradients at initialization."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "conv_small", dim=64)
    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    y = seed(x)
    loss = y.sum()
    loss.backward()

    for name, param in seed.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum() > 0, f"{name} has zero gradient"
```

**Step 2: Run test to check current state**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_conv_small_has_gradient_at_init -v
```

**Step 3: Refactor implementation**

Modify `ConvSmallSeed` class:

```python
@BlueprintRegistry.register(
    "conv_small", "cnn",
    param_estimate=4200,
    action_index=6,
    description="Small 1x1 conv - same tier as bottleneck"
)
def create_conv_small_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Small 1x1 convolution seed - lightweight channel mixing.

    Uses LayerScale instead of zero-init to avoid dead-branch trap.
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for cnn/conv_small: {sorted(kwargs)}")

    class ConvSmallSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.relu(self.gn(self.conv(x)))
            return x + self.ls(y)

    return ConvSmallSeed(dim, layer_scale)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_conv_small_has_gradient_at_init -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprints.py
git commit -m "fix(kasmina): conv_small uses LayerScale to avoid dead branch

Same fix as bottleneck - LayerScale provides gradient-safe
near-identity initialization.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: New Curriculum Blueprints (Gemini's 4)

### Task 4.1: Add dilated blueprint

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprints.py`:

```python
def test_dilated_seed_preserves_shape():
    """Dilated seed should preserve input shape."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "dilated", dim=64)
    x = torch.randn(2, 64, 32, 32)
    y = seed(x)

    assert y.shape == x.shape


def test_dilated_seed_is_near_identity_at_init():
    """Dilated seed should be near-identity at initialization."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "dilated", dim=64)
    x = torch.randn(2, 64, 32, 32)
    y = seed(x)

    # Should be close to identity (LayerScale init is 1e-3)
    diff = (y - x).abs().max()
    assert diff < 0.1, f"Dilated seed too far from identity at init: {diff}"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_dilated_seed_preserves_shape -v
```

Expected: FAIL with `ValueError: Unknown blueprint 'dilated'`

**Step 3: Write minimal implementation**

Add to `src/esper/kasmina/blueprints/cnn.py`:

```python
@BlueprintRegistry.register(
    "dilated", "cnn",
    param_estimate=37000,
    action_index=8,  # First new blueprint after conv_heavy (7)
    description="Dilated 3x3 conv (receptive field expansion)"
)
def create_dilated_seed(dim: int, dilation: int = 2, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Dilated convolution seed for receptive field expansion.

    Lesson: Receptive field can be expanded without extra parameters.
    Trade-off: Can cause "gridding" artifacts at higher dilation rates.

    Args:
        dim: Channel dimension
        dilation: Dilation rate (default 2)
        layer_scale: LayerScale initialization value
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for cnn/dilated: {sorted(kwargs)}")

    class DilatedSeed(nn.Module):
        def __init__(self, channels: int, dilation: int, layer_scale_init: float):
            super().__init__()
            self.conv = nn.Conv2d(
                channels, channels, kernel_size=3,
                padding=dilation, dilation=dilation, bias=False
            )
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.relu(self.gn(self.conv(x)))
            return x + self.ls(y)

    return DilatedSeed(dim, dilation, layer_scale)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_dilated_seed_preserves_shape tests/kasmina/test_blueprints.py::test_dilated_seed_is_near_identity_at_init -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprints.py
git commit -m "feat(kasmina): add dilated blueprint for receptive field expansion

Curriculum seed teaching receptive field concepts. Uses dilation=2
by default. LayerScale for safe initialization.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4.2: Add asymmetric blueprint

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprints.py`:

```python
def test_asymmetric_seed_preserves_shape():
    """Asymmetric seed should preserve input shape."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "asymmetric", dim=64)
    x = torch.randn(2, 64, 32, 32)
    y = seed(x)

    assert y.shape == x.shape


def test_asymmetric_seed_has_factored_convs():
    """Asymmetric seed uses factored 3x1 then 1x3 convolutions."""
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "asymmetric", dim=64)

    # Check it has the expected structure
    assert hasattr(seed, 'conv_v'), "Missing vertical conv"
    assert hasattr(seed, 'conv_h'), "Missing horizontal conv"
    assert seed.conv_v.kernel_size == (3, 1)
    assert seed.conv_h.kernel_size == (1, 3)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_asymmetric_seed_preserves_shape -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/esper/kasmina/blueprints/cnn.py`:

```python
@BlueprintRegistry.register(
    "asymmetric", "cnn",
    param_estimate=25000,
    action_index=9,
    description="Factorised 3x1 then 1x3 conv"
)
def create_asymmetric_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Asymmetric factorized convolution seed.

    Lesson: Same receptive field as 3x3 but fewer parameters.
    Trade-off: Deeper path, two kernel launches, sometimes slower on GPU.

    Args:
        dim: Channel dimension
        layer_scale: LayerScale initialization value
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for cnn/asymmetric: {sorted(kwargs)}")

    class AsymmetricSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            self.conv_v = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), bias=False)
            self.conv_h = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.conv_h(F.relu(self.conv_v(x)))
            y = F.relu(self.gn(y))
            return x + self.ls(y)

    return AsymmetricSeed(dim, layer_scale)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprints.py -k "asymmetric" -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprints.py
git commit -m "feat(kasmina): add asymmetric blueprint for factored convs

Curriculum seed teaching parameter efficiency via factorization.
3x1 then 1x3 = same receptive field as 3x3, fewer params.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4.3: Add coord blueprint

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprints.py`:

```python
def test_coord_seed_preserves_shape():
    """Coord seed should preserve input shape."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "coord", dim=64)
    x = torch.randn(2, 64, 32, 32)
    y = seed(x)

    assert y.shape == x.shape


def test_coord_seed_injects_coordinates():
    """Coord seed should inject spatial coordinates."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "coord", dim=64)

    # The embed layer should take channels+2 (for x,y coords)
    assert seed.embed.in_channels == 66  # 64 + 2
    assert seed.embed.out_channels == 64
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_coord_seed_preserves_shape -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/esper/kasmina/blueprints/cnn.py`:

```python
@BlueprintRegistry.register(
    "coord", "cnn",
    param_estimate=4500,
    action_index=10,
    description="Coordinate injection (CoordConv-lite)"
)
def create_coord_seed(dim: int, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """Coordinate injection seed (CoordConv-lite).

    Lesson: Break translation invariance; "where" matters.
    Trade-off: Can overfit or leak positional shortcuts depending on task.

    Args:
        dim: Channel dimension
        layer_scale: LayerScale initialization value
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for cnn/coord: {sorted(kwargs)}")

    class CoordSeed(nn.Module):
        def __init__(self, channels: int, layer_scale_init: float):
            super().__init__()
            self.embed = nn.Conv2d(channels + 2, channels, kernel_size=1, bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            # Generate normalized coordinate grids
            yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
            xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
            yy, xx = torch.meshgrid(yy, xx, indexing="ij")
            yy = yy.expand(b, 1, h, w)
            xx = xx.expand(b, 1, h, w)

            inp = torch.cat([x, yy, xx], dim=1)
            y = F.relu(self.gn(self.embed(inp)))
            return x + self.ls(y)

    return CoordSeed(dim, layer_scale)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprints.py -k "coord" -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprints.py
git commit -m "feat(kasmina): add coord blueprint for position encoding

Curriculum seed teaching position-aware features (CoordConv-lite).
Injects normalized x,y coordinates before 1x1 conv.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4.4: Add gated blueprint

**Files:**
- Modify: `src/esper/kasmina/blueprints/cnn.py`
- Test: `tests/kasmina/test_blueprints.py`

**Step 1: Write the failing test**

Add to `tests/kasmina/test_blueprints.py`:

```python
def test_gated_seed_preserves_shape():
    """Gated seed should preserve input shape."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "gated", dim=64)
    x = torch.randn(2, 64, 32, 32)
    y = seed(x)

    assert y.shape == x.shape


def test_gated_seed_uses_glu():
    """Gated seed should use GLU-style gating."""
    import torch
    from esper.kasmina.blueprints import BlueprintRegistry

    seed = BlueprintRegistry.create("cnn", "gated", dim=64)

    # Projection should output 2x channels (value + gate)
    assert seed.proj.out_channels == 128  # 64 * 2
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_gated_seed_preserves_shape -v
```

Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/esper/kasmina/blueprints/cnn.py`:

```python
@BlueprintRegistry.register(
    "gated", "cnn",
    param_estimate=8400,
    action_index=11,
    description="GLU-style gated 1x1 mixer"
)
def create_gated_seed(dim: int, gate_bias: float = 2.0, layer_scale: float = 1e-3, **kwargs: Any) -> nn.Module:
    """GLU-style gated mixer seed.

    Lesson: Explicit suppression/selection via learned gates.
    Trade-off: Can saturate (sigmoid) if initialization too aggressive.

    Args:
        dim: Channel dimension
        gate_bias: Initial bias for gate (2.0 = sigmoid(2) ≈ 0.88, mostly open)
        layer_scale: LayerScale initialization value
    """
    if kwargs:
        raise ValueError(f"Unexpected kwargs for cnn/gated: {sorted(kwargs)}")

    class GatedSeed(nn.Module):
        def __init__(self, channels: int, gate_bias: float, layer_scale_init: float):
            super().__init__()
            self.proj = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=True)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)
            self.ls = LayerScale(channels, init_value=layer_scale_init)

            # Bias the gate half towards "open"
            with torch.no_grad():
                self.proj.bias[:channels].zero_()
                self.proj.bias[channels:].fill_(gate_bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v, g = self.proj(x).chunk(2, dim=1)
            y = v * torch.sigmoid(g)
            y = self.gn(y)
            return x + self.ls(y)

    return GatedSeed(dim, gate_bias, layer_scale)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/kasmina/test_blueprints.py -k "gated" -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/kasmina/blueprints/cnn.py tests/kasmina/test_blueprints.py
git commit -m "feat(kasmina): add gated blueprint for GLU-style mixing

Curriculum seed teaching gating mechanisms. Gate initialized
towards 'open' (bias=2.0) to avoid dead start.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4.5: Run full test suite and integration tests

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v --tb=short
```

**Step 2: Run integration tests specifically**

```bash
uv run pytest tests/integration/ -v --tb=short
```

**Step 3: Run blueprint-specific tests**

```bash
uv run pytest tests/kasmina/test_blueprints.py tests/kasmina/test_blueprint_registry.py tests/kasmina/test_blueprint_manifest.py -v
```

**Step 4: Verify param estimates**

```bash
uv run pytest tests/kasmina/test_blueprints.py::test_param_estimate_accuracy -v
```

Fix any failing tests.

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: verify all new blueprints pass integration tests

All 4 curriculum blueprints (dilated, asymmetric, coord, gated)
are registered with correct action indices and pass shape/gradient tests.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

This plan delivers:

1. **Blueprint Compiler Infrastructure** (Tasks 1.1-1.7)
   - `BlueprintSpec.action_index` for checkpoint stability
   - `BlueprintManifest` dataclass for compiled action space
   - `BlueprintCompiler` for manifest generation
   - Existing blueprints assigned explicit indices

2. **Leyline Migration** (Tasks 2.1-2.6)
   - Manifest-based accessor functions
   - Migration of `action_masks.py`, `action_enums.py`, `helpers.py`
   - Full test suite verification

3. **LayerScale & Fixes** (Tasks 3.1-3.3)
   - `LayerScale` module for safe residual scaling
   - `bottleneck` and `conv_small` fixed for dead-branch risk

4. **Curriculum Blueprints** (Tasks 4.1-4.5)
   - `dilated` - receptive field expansion
   - `asymmetric` - factorized convolutions
   - `coord` - position encoding (CoordConv-lite)
   - `gated` - GLU-style mixing

**Adding future blueprints now requires only:**
1. Register with `@BlueprintRegistry.register(..., action_index=N)`
2. Implement the `nn.Module`
3. Done - no leyline changes needed!
