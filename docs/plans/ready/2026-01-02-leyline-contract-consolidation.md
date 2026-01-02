# Leyline Contract Consolidation Plan

**Date:** 2026-01-02
**Status:** Ready for Implementation
**Priority:** High (architectural debt)
**Estimated Effort:** 2-3 focused sessions

## Executive Summary

Cross-subsystem imports currently bypass leyline, creating tight coupling between domains. This plan consolidates shared protocols and contracts into leyline, establishing it as the single source of truth for all cross-domain interfaces.

### Architecture Correction

**Current understanding (corrected):**
- **Nissa** = Telemetry Hub (event router, broadcasts to backends)
- **Karn** = Telemetry Consumer (storage, analysis, visualization via Sanctum/Overwatch)
- **Leyline** = Contracts module (all cross-subsystem types should flow through here)

---

## Problem Statement

Analysis identified **61 cross-subsystem imports** that bypass leyline:

| Subsystem | Violations | Severity |
|-----------|------------|----------|
| simic | 28 | CRITICAL - imports from 6 subsystems |
| tamiyo | 3 | Medium |
| tolaria | 4 | Medium |
| karn | 3 | Low (telemetry consumer) |
| nissa | 1 | Low |
| kasmina | 2 | Low (TYPE_CHECKING only) |

The **simic** subsystem is the most problematic—it's a "god module" that imports from nearly every other domain.

---

## Guiding Principles

### What Belongs in Leyline

1. **Protocols** — Abstract interfaces that multiple subsystems implement or consume
2. **Shared dataclasses** — Data contracts that flow between domains
3. **Constants** — Training behavior constants (already mostly there)
4. **Utility functions** — Pure functions used across domains (e.g., `safe()`)

### What Stays in Subsystems

1. **Concrete implementations** — Classes with actual logic (networks, hosts, trackers)
2. **Factory functions** — `create_model()`, `get_hub()`, etc.
3. **Domain-specific helpers** — `build_slot_states()`, `compute_action_masks()`
4. **Local protocols** — Protocols for internal decoupling (e.g., karn/contracts.py)

### The simic/contracts.py Pattern

This existing pattern shows the right approach:
- **Lightweight protocols** that capture just the interface needed
- **TYPE_CHECKING guards** to avoid runtime import overhead
- **Structural typing** — implementations don't inherit, they just match the shape

---

## Phase 1: Foundation Protocols (Critical Path)

These are the highest-value moves—fundamental interfaces used across multiple domains.

### 1.1 Move HostProtocol to Leyline

**Current location:** `kasmina/protocol.py`
**Used by:** kasmina, simic, tolaria
**Rationale:** Foundational to "Train Anything" principle (ROADMAP #5)

```python
# leyline/host_protocol.py
@runtime_checkable
class HostProtocol(Protocol):
    """Contract for graftable host networks.

    Hosts are pure backbone networks that provide segment routing.
    Slot management is handled by MorphogeneticModel, not hosts directly.
    """
    def injection_specs(self) -> list["InjectionSpec"]: ...
    @property
    def injection_points(self) -> dict[str, int]: ...
    @property
    def segment_channels(self) -> dict[str, int]: ...
    @property
    def topology(self) -> str: ...  # 'cnn' or 'transformer'
    def forward(self, x: Tensor) -> Tensor: ...
    def forward_to_segment(self, segment: str, x: Tensor, from_segment: str | None = None) -> Tensor: ...
    def forward_from_segment(self, segment: str, x: Tensor) -> Tensor: ...
```

**Migration steps:**
1. Create `leyline/host_protocol.py`
2. Move HostProtocol definition
3. Update `kasmina/protocol.py` to re-export from leyline
4. Update `leyline/__init__.py` to export HostProtocol

### 1.2 Move PolicyBundle to Leyline

**Current location:** `tamiyo/policy/protocol.py`
**Used by:** tamiyo, simic
**Rationale:** Core swap point for policy implementations (LSTM, MLP, Heuristic)

```python
# leyline/policy_protocol.py
class PolicyBundle(Protocol):
    """Protocol for swappable Tamiyo policy implementations.

    Implementations: FactoredRecurrentActorCritic (LSTM), HeuristicTamiyo, future SAC
    """
    @property
    def network(self) -> nn.Module: ...
    @property
    def device(self) -> torch.device: ...
    def get_action(
        self,
        obs: Tensor,
        hidden_state: Any,
        deterministic: bool = False,
    ) -> tuple[TamiyoDecision, Tensor, Any]: ...
    def evaluate_actions(
        self,
        obs: Tensor,
        actions: dict[str, Tensor],
        hidden_states: Any,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]: ...
    # ... additional methods
```

**Migration steps:**
1. Create `leyline/policy_protocol.py`
2. Move PolicyBundle and related protocols
3. Consolidate `TamiyoPolicy` (from heuristic.py) into PolicyBundle
4. Update tamiyo to re-export from leyline

### 1.3 Move Seed Contracts to Leyline

**Current location:** `simic/contracts.py`
**Used by:** simic, tamiyo
**Rationale:** These are simic's "view" of kasmina, but they're stable enough to be universal

```python
# leyline/seed_protocols.py
class SeedStateProtocol(Protocol):
    """Protocol for seed lifecycle state objects."""
    seed_id: str
    metrics: Any  # SeedMetrics
    blueprint_id: str
    @property
    def stage(self) -> SeedStage: ...
    @property
    def alpha(self) -> float: ...
    @property
    def epochs_in_stage(self) -> int: ...
    def can_transition_to(self, new_stage: SeedStage) -> bool: ...
    def sync_telemetry(self, ...) -> None: ...

@runtime_checkable
class SeedSlotProtocol(Protocol):
    """Protocol for seed slots in host models."""
    # Telemetry configuration
    fast_mode: bool
    telemetry_lifecycle_only: bool
    on_telemetry: Any
    isolate_gradients: bool
    # State access
    @property
    def state(self) -> SeedStateProtocol | None: ...
    @property
    def seed(self) -> nn.Module | None: ...
    @property
    def alpha(self) -> float: ...
    # Lifecycle operations
    def advance_stage(self, target_stage: SeedStage | None = None) -> GateResult: ...
    def step_epoch(self) -> None: ...
    def set_alpha(self, value: float) -> None: ...
    @contextmanager
    def force_alpha(self, value: float) -> Iterator[None]: ...

class SlottedHostProtocol(Protocol):
    """Protocol for host models with seed slots."""
    @property
    def seed_slots(self) -> Any: ...  # ModuleDict
    @property
    def active_seed_params(self) -> int: ...
    @property
    def has_active_seed(self) -> bool: ...
    def germinate_seed(self, blueprint_id: str, seed_id: str, *, slot: str, ...) -> None: ...
    def prune_seed(self, *, slot: str) -> None: ...
    def get_host_parameters(self) -> Iterator[nn.Parameter]: ...
    def get_seed_parameters(self, slot: str | None = None) -> Iterator[nn.Parameter]: ...
```

**Migration steps:**
1. Create `leyline/seed_protocols.py`
2. Move all three protocols from `simic/contracts.py`
3. Update `simic/contracts.py` to re-export from leyline (backwards compat)
4. Deprecate direct imports from simic/contracts.py

---

## Phase 2: Shared Data Contracts

### 2.1 Move TaskConfig to Leyline

**Current location:** `tamiyo/policy/features.py` (HOT PATH!)
**Used by:** runtime, tolaria, simic, tamiyo
**Problem:** Currently in hot path module which should only import leyline

```python
# leyline/task_config.py
@dataclass
class TaskConfig:
    """Task-specific feature dimensions for observation encoding."""
    base_feature_size: int
    slot_feature_size: int
    num_slots: int
    num_blueprints: int
    # ... other fields
```

**Migration steps:**
1. Create `leyline/task_config.py`
2. Move TaskConfig definition
3. Update `tamiyo/policy/features.py` to import from leyline
4. This fixes the hot path import discipline violation

### 2.2 Move TaskSpec to Leyline

**Current location:** `runtime/tasks.py`
**Used by:** tolaria, simic
**Rationale:** Cross-domain task configuration contract

```python
# leyline/task_spec.py
@dataclass
class TaskSpec:
    """Task specification binding all training components."""
    name: str
    host_factory: Callable[[], HostProtocol]
    dataset_factory: Callable[[], Dataset]
    task_config: TaskConfig
    loss_config: LossRewardConfig | None = None
    # ... other fields
```

**Migration steps:**
1. Create `leyline/task_spec.py`
2. Move TaskSpec dataclass only (not VALID_TASKS registry)
3. Keep get_task_spec() and VALID_TASKS in runtime

### 2.3 Move EpisodeOutcome to Leyline

**Current location:** `karn/store.py`
**Used by:** simic/training
**Problem:** Karn is a consumer, not a contract source. Simic shouldn't import from consumers.

```python
# leyline/episode_outcome.py
@dataclass
class EpisodeOutcome:
    """Result of a training episode."""
    episode_id: int
    final_accuracy: float
    total_reward: float
    steps: int
    outcome_type: str  # 'completed', 'timeout', 'early_stop'
    # ... other fields
```

**Migration steps:**
1. Create `leyline/episode_outcome.py`
2. Move EpisodeOutcome definition
3. Update karn/store.py to import from leyline

---

## Phase 3: Protocol Abstractions

### 3.1 Create OutputBackendProtocol in Leyline

**Current location:** Implicit in `nissa/output.py`
**Used by:** nissa (defines), karn (implements)

```python
# leyline/output_protocol.py
@runtime_checkable
class OutputBackendProtocol(Protocol):
    """Protocol for telemetry output backends.

    Backends receive events from NissaHub and route to destinations.
    Implementations: ConsoleOutput, FileOutput, SanctumBackend, OverwatchBackend
    """
    def emit(self, event: TelemetryEvent) -> None: ...
    def start(self) -> None: ...
    def close(self) -> None: ...
```

**Migration steps:**
1. Create `leyline/output_protocol.py`
2. Update nissa/output.py to import protocol from leyline
3. Update karn backends to import protocol from leyline
4. Breaks the nissa ↔ karn coupling

### 3.2 Create GovernorProtocol in Leyline

**Current location:** Only `TolariaGovernor` concrete class exists
**Used by:** simic/training
**Rationale:** Governor behavior is training-critical but simic imports concrete class

```python
# leyline/governor_protocol.py
class GovernorProtocol(Protocol):
    """Protocol for catastrophic failure detection and recovery."""
    def check(self, loss: float, epoch: int) -> GovernorReport | None: ...
    def reset(self) -> None: ...
    def record_checkpoint(self, state_dict: dict) -> None: ...
    def get_rollback_state(self) -> dict | None: ...
```

**Migration steps:**
1. Create `leyline/governor_protocol.py`
2. Have TolariaGovernor implement the protocol
3. Update simic to type hint with GovernorProtocol

---

## Phase 4: Utility Functions

### 4.1 Move safe() to Leyline

**Current location:** `tamiyo/policy/features.py`
**Used by:** Feature extraction hot path
**Rationale:** Pure numerical utility with no domain logic

```python
# leyline/utils.py
def safe(v: float | int | None, default: float = 0.0, max_val: float = 100.0) -> float:
    """Safely convert value to float, handling None/inf/nan."""
    if v is None:
        return default
    try:
        v_float = float(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"safe() expected numeric, got {type(v)!r}") from exc
    if not math.isfinite(v_float):
        return default
    return max(-max_val, min(v_float, max_val))
```

**Migration steps:**
1. Create `leyline/utils.py`
2. Move safe() function
3. Update tamiyo/policy/features.py to import from leyline

---

## Phase 5: Cleanup and Verification

### 5.1 Update leyline/__init__.py Exports

After all migrations, leyline should export:

```python
# leyline/__init__.py additions
from esper.leyline.host_protocol import HostProtocol
from esper.leyline.policy_protocol import PolicyBundle
from esper.leyline.seed_protocols import (
    SeedStateProtocol,
    SeedSlotProtocol,
    SlottedHostProtocol,
)
from esper.leyline.task_config import TaskConfig
from esper.leyline.task_spec import TaskSpec
from esper.leyline.episode_outcome import EpisodeOutcome
from esper.leyline.output_protocol import OutputBackendProtocol
from esper.leyline.governor_protocol import GovernorProtocol
from esper.leyline.utils import safe
```

### 5.2 Backwards Compatibility (Temporary)

For each moved type, the original location should re-export:

```python
# simic/contracts.py (example)
"""DEPRECATED: Import from esper.leyline instead."""
from esper.leyline.seed_protocols import (
    SeedStateProtocol,
    SeedSlotProtocol,
    SlottedHostProtocol,
)
# Re-export for backwards compatibility
__all__ = ["SeedStateProtocol", "SeedSlotProtocol", "SlottedHostProtocol"]
```

**Note:** Per CLAUDE.md "No Legacy Code Policy", these re-exports should be removed in a follow-up PR once all import sites are updated.

### 5.3 Import Validation Test

Add a test to enforce leyline-only cross-subsystem imports:

```python
# tests/test_import_discipline.py
def test_no_cross_subsystem_imports_bypass_leyline():
    """Verify all cross-subsystem imports go through leyline."""
    subsystems = ['kasmina', 'simic', 'tamiyo', 'tolaria', 'nissa', 'karn']

    for subsystem in subsystems:
        subsystem_path = Path(f'src/esper/{subsystem}')
        for py_file in subsystem_path.rglob('*.py'):
            content = py_file.read_text()
            for other in subsystems:
                if other == subsystem:
                    continue
                # Check for direct imports (not through leyline)
                pattern = rf'from esper\.{other}\b(?!.*leyline)'
                matches = re.findall(pattern, content)
                assert not matches, f"{py_file} imports from {other} without leyline"
```

---

## Proposed File Structure

After migration, leyline will have these new files:

```
leyline/
├── __init__.py              # Updated exports
├── host_protocol.py         # HostProtocol (from kasmina)
├── policy_protocol.py       # PolicyBundle (from tamiyo)
├── seed_protocols.py        # Seed*Protocol (from simic)
├── task_config.py           # TaskConfig (from tamiyo)
├── task_spec.py             # TaskSpec (from runtime)
├── episode_outcome.py       # EpisodeOutcome (from karn)
├── output_protocol.py       # OutputBackendProtocol (new)
├── governor_protocol.py     # GovernorProtocol (new)
├── utils.py                 # safe() and other utilities
└── ... (existing files)
```

---

## What NOT to Move

These types should **stay in their subsystems**:

| Type | Location | Reason |
|------|----------|--------|
| `FactoredRecurrentActorCritic` | tamiyo | Concrete network implementation |
| `MorphogeneticModel` | kasmina | Concrete host implementation |
| `SeedSlot`, `SeedState` | kasmina | Rich implementation classes |
| `CNNHost`, `TransformerHost` | kasmina | Concrete implementations |
| `BlendCatalog` | kasmina | Internal registry |
| `SignalTracker` | tamiyo | Business logic |
| `HeuristicTamiyo` | tamiyo | Specific policy impl |
| `TolariaGovernor` | tolaria | Concrete governor |
| `HealthMonitor` | karn | Monitoring implementation |
| `get_hub()`, `emit()` | nissa | Factory functions |
| `create_model()` | tolaria | Factory function |
| `VALID_TASKS`, `get_task_spec()` | runtime | Registry functions |
| karn/contracts.py types | karn | Local decoupling only |

---

## Dependency Graph (Target State)

```
                    ┌─────────────────────────────────────┐
                    │            LEYLINE                  │
                    │  Protocols, Contracts, Constants    │
                    │  ─────────────────────────────────  │
                    │  HostProtocol, PolicyBundle,        │
                    │  SeedSlotProtocol, TaskConfig,      │
                    │  OutputBackendProtocol, ...         │
                    └─────────────────────────────────────┘
                                    ▲
                    ┌───────────────┼───────────────┐
                    │               │               │
              ┌─────┴─────┐   ┌─────┴─────┐   ┌─────┴─────┐
              │  KASMINA  │   │   SIMIC   │   │  TAMIYO   │
              │implements │   │  uses     │   │implements │
              │HostProto  │   │ protocols │   │PolicyProto│
              └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    │         ┌─────┴─────┐         │
                    └────────►│  TOLARIA  │◄────────┘
                              │implements │
                              │GovernProto│
                              └───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
              ┌───────────┐                   ┌───────────┐
              │   NISSA   │──────────────────►│   KARN    │
              │ hub(emit) │                   │implements │
              │           │                   │OutputProto│
              └───────────┘                   └───────────┘
```

All arrows now point TO leyline (imports) or between subsystems via leyline protocols.

---

## Success Criteria

1. **No direct cross-subsystem imports** — All imports go through leyline
2. **Hot path clean** — `tamiyo/policy/features.py` imports ONLY from leyline
3. **Protocols in leyline** — All shared interfaces defined in leyline
4. **Tests pass** — Import isolation test enforces discipline
5. **Circular dependencies eliminated** — No import cycles between subsystems

---

## Implementation Order

| Phase | Files | Blocking? | Notes |
|-------|-------|-----------|-------|
| 1.1 | HostProtocol | Yes | Foundation |
| 1.2 | PolicyBundle | Yes | Depends on 1.1 |
| 1.3 | Seed protocols | Yes | Depends on 1.1 |
| 2.1 | TaskConfig | Yes | Fixes hot path |
| 2.2 | TaskSpec | No | Can parallel |
| 2.3 | EpisodeOutcome | No | Can parallel |
| 3.1 | OutputBackendProtocol | No | Can parallel |
| 3.2 | GovernorProtocol | No | Can parallel |
| 4.1 | safe() | No | Can parallel |
| 5.x | Cleanup | Yes | After all moves |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing imports | Re-export from original locations temporarily |
| Missing protocol methods | Review all usage sites before moving |
| Circular import during migration | Move in dependency order (leyline first) |
| Test failures | Run full test suite after each phase |
| Performance regression | Profile hot path before/after |

---

## References

- Analysis from agent explorations (2026-01-02)
- `simic/contracts.py` — existing pattern template
- CLAUDE.md — Leyline ownership rules
- ROADMAP.md — "Train Anything" principle (#5)
