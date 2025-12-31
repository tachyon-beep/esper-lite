# Seed Telemetry Design

**Date:** 2025-11-29
**Status:** Draft (Reviewed)
**Author:** Claude + John
**Reviewers:** DRL Expert Agent, Code Architecture Agent

## Overview

Design a per-seed telemetry system that enables seeds to report their "local picture" to decision-makers (Tamiyo). This supports the immediate need (fair evaluation in comparison.py) while laying groundwork for multi-seed and hierarchical Tamiyo architectures.

## Goals

1. **Fair evaluation** - IQL models trained with telemetry should be evaluated with real telemetry, not zero-padded features
2. **Richer diagnostics** - Comparison output shows gradient health, per-class accuracy to debug policy differences
3. **Future-ready** - Architecture supports multi-seed (50+ seeds) and hierarchical Tamiyo (strategic/tactical layers)

## Non-Goals (for now)

- Implementing Strategic/Tactical Tamiyo hierarchy
- Multi-seed slot management
- Full DiagnosticTracker integration (sharpness estimation, etc.)

## Architecture

### Future Vision: Hierarchical Tamiyo

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategic Tamiyo                          │
│  - Sees aggregate telemetry from each Tactical Tamiyo       │
│  - Allocates "seed budgets" to tacticals                    │
│  - Makes model-wide decisions                               │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Tactical A    │  │ Tactical B    │  │ Tactical C    │
│ (early layers)│  │ (attention)   │  │ (head)        │
└───────────────┘  └───────────────┘  └───────────────┘
   │  │  │           │  │  │           │  │  │
   ▼  ▼  ▼           ▼  ▼  ▼           ▼  ▼  ▼
  Seeds...          Seeds...          Seeds...
```

Each seed reports its `SeedTelemetry`. Tacticals see per-seed data for their domain. Strategic sees aggregates.

### Current Implementation: Single Seed

```
┌─────────────────────────────────────────────────────────────┐
│                       Tamiyo                                 │
│  - Consumes SeedTelemetry from active seed                  │
│  - Makes germinate/advance/cull decisions                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  SeedState    │
                    │  .telemetry   │◄── SeedTelemetry
                    └───────────────┘
```

### Kasmina Future: Seed Plane + Telemetry Plane

As Kasmina evolves, seeds will be organized into:
- **Seed Plane** - actual seed parameters/computation
- **Telemetry Plane** - each seed's SeedTelemetry reporting

The `SeedTelemetry` contract in leyline is the interface between these planes and Tamiyo.

## Data Model

### SeedTelemetry (leyline/telemetry.py)

```python
from datetime import datetime, timezone

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

@dataclass(slots=True)
class SeedTelemetry:
    """Per-seed telemetry snapshot - the seed's 'local picture'.

    Contract between seed implementations (Kasmina/Simic) and
    decision-makers (Tamiyo). Designed for:
    - Single seed (current): one instance
    - Multi-seed (future): collection managed by registry
    - Hierarchical (stretch): tactical aggregates for strategic

    Note: Uses slots=True for memory efficiency in multi-seed scenarios.
    """
    seed_id: str
    blueprint_id: str = ""   # e.g., "conv_enhance", "attention"
    layer_id: str = ""       # e.g., "block2.conv1" - for spatial grouping

    # Health signals (lightweight, always collected)
    gradient_norm: float = 0.0
    gradient_health: float = 1.0      # 0-1, higher is healthier
    has_vanishing: bool = False
    has_exploding: bool = False

    # Progress signals
    accuracy: float = 0.0             # percentage (0-100)
    accuracy_delta: float = 0.0       # positive = improving
    epochs_in_stage: int = 0

    # Stage context
    stage: int = 1                    # SeedStage enum value (1-7)
    alpha: float = 0.0                # blending weight (0-1)

    # Temporal context (added per architecture review)
    epoch: int = 0                    # current epoch in training
    max_epochs: int = 25              # total epochs (for normalization)

    # Timestamp for staleness detection
    captured_at: datetime = field(default_factory=_utc_now)

    def to_features(self) -> list[float]:
        """Convert to 10-dim feature vector for RL policies.

        All features normalized to approximately [0, 1] range.
        """
        return [
            min(self.gradient_norm, 10.0) / 10.0,
            self.gradient_health,
            float(self.has_vanishing),
            float(self.has_exploding),
            min(self.epochs_in_stage, 50) / 50.0,
            self.accuracy / 100.0,
            max(-1.0, min(1.0, self.accuracy_delta / 10.0)),
            (self.stage - 1) / 6.0,  # stages 1-7 -> [0, 1]
            self.alpha,
            self.epoch / max(self.max_epochs, 1),  # temporal position
        ]

    @classmethod
    def feature_dim(cls) -> int:
        """Return current feature vector dimension."""
        return 10

    @property
    def is_stale(self, threshold_seconds: float = 60.0) -> bool:
        """Check if telemetry is stale (older than threshold)."""
        age = (datetime.now(timezone.utc) - self.captured_at).total_seconds()
        return age > threshold_seconds
```

### Design Decisions

1. **10 features per seed** - Lightweight, normalized to [0, 1] range
2. **`layer_id` and `blueprint_id`** - Enable future grouping strategies (spatial, blueprint-based, dynamic)
3. **Grouping is pluggable** - TBD how seeds map to tactical domains
4. **Seed owns its telemetry** - `SeedState.telemetry` field, updated via callbacks
5. **`slots=True`** - Memory efficient for multi-seed scenarios (50+ seeds)
6. **Temporal position** - `epoch/max_epochs` replaces reserved slot (per DRL review)
7. **Stage normalization** - `(stage-1)/6.0` gives proper [0,1] range (per DRL review)
8. **Timestamp** - Enables staleness detection for multi-seed (per architecture review)

### Future: SeedTelemetryRegistry

When multi-seed is implemented:

```python
class SeedTelemetryRegistry:
    """Collection of all seed telemetry. Supports flexible queries."""

    def register(self, seed_id: str, blueprint_id: str, layer_id: str) -> SeedTelemetry
    def update(self, seed_id: str, **kwargs) -> None
    def get(self, seed_id: str) -> SeedTelemetry | None
    def remove(self, seed_id: str) -> None

    # Flexible grouping for tactical/strategic
    def by_blueprint(self, blueprint_id: str) -> list[SeedTelemetry]
    def by_layer(self, layer_id: str) -> list[SeedTelemetry]
    def by_stage(self, stage: SeedStage) -> list[SeedTelemetry]
    def all(self) -> list[SeedTelemetry]

    # Aggregation helpers
    def aggregate(self, seeds: list[SeedTelemetry]) -> dict[str, float]
```

## Integration Points

### 1. Contract Layer (leyline)

Add to `leyline/telemetry.py`:
- `SeedTelemetry` dataclass
- Export in `__init__.py`

### 2. Seed Layer (kasmina)

Modify `kasmina/slot.py`:
- Add `telemetry: SeedTelemetry` field to `SeedState`
- Update telemetry in `SeedMetrics.record_accuracy()` or via separate callback
- Initialize telemetry on seed creation with `seed_id`, `blueprint_id`, `layer_id`

### 3. Training Layer (simic)

Modify training loops to:
- Capture gradient norms for active seed (filter by `model.get_seed_parameters()`)
- Update `seed_state.telemetry` after each epoch
- Lightweight: gradient health + accuracy signals only (no sharpness)

### 4. Comparison Layer (simic/comparison.py)

Modify `snapshot_to_features()`:
- When `use_telemetry=True` and seed is active, use `seed_state.telemetry.to_features()`
- When no seed or telemetry unavailable, fall back to zeros (with warning)

### 5. Feature Layer (simic/features.py)

The existing `telemetry_to_features()` function expects a dict from `DiagnosticTracker`. We have options:

**Option A:** Keep separate functions
- `telemetry_to_features(telem: dict)` - for DiagnosticTracker output (27 dims)
- `SeedTelemetry.to_features()` - for seed local picture (10 dims)

**Option B:** Unify to SeedTelemetry
- Deprecate `telemetry_to_features()`
- Always use `SeedTelemetry.to_features()` for RL policies

**Recommendation:** Option A for now. The 27-dim telemetry captures full-model signals; the 10-dim SeedTelemetry captures per-seed signals. Different purposes.

### 6. Feature Dimension Strategy (per architecture review)

**Clarification on dimensions:**
- **27 dims**: Base features (from `obs_to_base_features`)
- **27 dims**: Full-model telemetry (from `telemetry_to_features`) - gradient health, per-class accuracy, sharpness
- **10 dims**: Per-seed telemetry (from `SeedTelemetry.to_features`) - seed-local signals

**State dimension options:**
- 27 (base only) - no telemetry
- 37 (base + seed) - seed telemetry only
- 54 (base + full) - current implementation
- 64 (base + full + seed) - complete telemetry

**For Phase 1:** Use 37-dim (base + seed). This is the minimal useful telemetry for comparison.py. The 27-dim full-model telemetry is redundant when we have per-seed telemetry.

**Migration:** Models trained with 54-dim will need retraining or adapter layers.

## Telemetry Collection (Lightweight)

For comparison.py, we collect:

### Gradient Health
- Use DiagnosticTracker with seed-parameter filtering
- After each backward pass, update `seed_state.telemetry.gradient_*` fields
- Compute health score from vanishing/exploding detection

### Accuracy Signals
- Already computed at validation time
- Update `seed_state.telemetry.accuracy` and `accuracy_delta`

### Stage Context
- Read from `seed_state.stage` and `seed_slot.alpha`
- Updated automatically on stage transitions

### What We Skip (for now)
- Sharpness estimation (expensive)
- Per-class accuracy breakdown (can add later)
- Loss landscape analysis

### Update Mechanism (per architecture review)

**Source of truth:** `SeedMetrics` remains the source of truth for accuracy/epoch data.
`SeedTelemetry` is derived/synchronized from `SeedMetrics` plus gradient signals.

```python
# In SeedState or via explicit update method
def sync_telemetry(self, gradient_norm: float, gradient_health: float,
                   has_vanishing: bool, has_exploding: bool) -> None:
    """Sync telemetry from metrics + gradient signals."""
    self.telemetry.accuracy = self.metrics.current_val_accuracy
    self.telemetry.accuracy_delta = self.metrics.improvement_since_stage_start
    self.telemetry.epochs_in_stage = self.metrics.epochs_in_current_stage
    self.telemetry.stage = self.stage.value
    self.telemetry.alpha = self.metrics.current_alpha

    # Gradient signals from training loop
    self.telemetry.gradient_norm = gradient_norm
    self.telemetry.gradient_health = gradient_health
    self.telemetry.has_vanishing = has_vanishing
    self.telemetry.has_exploding = has_exploding

    # Update timestamp
    self.telemetry.captured_at = datetime.now(timezone.utc)
```

### Update Frequency

**Per-epoch** (per DRL review):
1. Gradient health is noisy per-batch - per-epoch smooths this
2. Policy decisions are epoch-grained (Tamiyo decides once per epoch)
3. Reduces hot-path overhead
4. Aligns with base features (`accuracy_history_5`, etc.)

### Fast Mode Interaction (per architecture review + Gemini)

When `SeedSlot.fast_mode=True` (PPO rollouts), **expensive** telemetry collection (gradient hooks) is skipped, but `sync_telemetry` is still called at epoch boundaries with cached/default gradient values:

```python
def sync_telemetry(self, ...):
    # Always sync accuracy/stage from metrics (cheap)
    # Only skip expensive gradient collection in fast mode
    self.telemetry.accuracy = self.metrics.current_val_accuracy
    self.telemetry.stage = self.stage.value
    # ... etc ...

    if not self.fast_mode:
        # Only collect fresh gradient stats when not in fast mode
        self.telemetry.gradient_norm = gradient_norm
        self.telemetry.gradient_health = gradient_health
        # ...
```

**Rationale (per Gemini review):** Stale telemetry is acceptable; missing telemetry is not. PPO can work with slightly laggy gradient indicators, but needs current accuracy/stage info.

## Migration Path

### Phase 1: Contract + Basic Integration (this PR)
1. Add `SeedTelemetry` to leyline
2. Add `telemetry` field to `SeedState`
3. Update comparison.py to use real telemetry
4. Lightweight gradient collection in comparison training loops

### Phase 2: Training Integration (future)
1. Add telemetry updates to `training.py` and `vectorized.py`
2. Tamiyo consumes `SeedTelemetry` for decisions

### Phase 3: Multi-Seed (future)
1. Implement `SeedTelemetryRegistry`
2. Multiple seed slots in model
3. Tactical Tamiyo per domain

### Phase 4: Hierarchical (stretch)
1. Strategic Tamiyo layer
2. Budget allocation between tacticals
3. Aggregation queries on registry

## Testing Strategy

### Unit Tests
- `SeedTelemetry.to_features()` returns correct dimension and normalized values
- `SeedTelemetry` serialization/deserialization

### Integration Tests
- comparison.py produces non-zero telemetry features when seed active
- Telemetry updates correctly during training loop

### Compatibility Tests
- Models saved with telemetry can be loaded and evaluated
- Zero-padding fallback works when telemetry unavailable

## Open Questions

1. ~~**Feature dimension:** Is 10 dims per seed the right balance? May need tuning.~~ **Resolved:** 10 dims confirmed, with temporal position replacing reserved slot.
2. **Gradient filtering:** Exact mechanism to filter DiagnosticTracker to seed params? (Implementation detail for Phase 1)
3. ~~**Update frequency:** Per-batch or per-epoch telemetry updates?~~ **Resolved:** Per-epoch (per DRL review).

## Review Findings

### DRL Expert Review

**Key recommendations incorporated:**
- Fixed stage normalization: `(stage-1)/6.0` for proper [0,1] range
- Added temporal position: `epoch/max_epochs` replaces reserved slot
- Per-epoch update frequency confirmed
- Action masking recommended (future work)
- Telemetry-enhanced PBRS potential function (future work)

**Multi-seed architecture guidance:**
- Attention + hierarchical pooling for variable seed counts
- Masking essential (zeros have semantic meaning)
- Consider blueprint/layer embeddings for seed identity

**Anti-pattern flagged:**
- Zero-padding telemetry creates distribution shift
- Use mean values or always collect telemetry

### Code Architecture Review

**Key recommendations incorporated:**
- Added `slots=True` for memory efficiency
- Added `captured_at` timestamp for staleness detection
- Added `sync_telemetry()` update mechanism
- Clarified SeedMetrics as source of truth
- Added fast mode interaction

**Integration clarifications:**
- Must add to `leyline/__init__.py` exports
- 37-dim (base + seed) for Phase 1 vs 54-dim (base + full)
- Migration path for existing models

### Gemini Review

**Key insight:** "By giving each seed its own SeedTelemetry struct, you are effectively creating a distributed sensor network within your model. This enables Credit Assignment - Tamiyo can now punish a specific lazy seed without punishing the entire garden."

**Validation:**
- 10-dim features: Correct (27-dim full telemetry is overkill for single seed)
- Temporal context: Crucial (agent must know "what time it is")
- Fast mode: Necessary but dangerous (mitigated by epoch-boundary sync)
- `slots=True`: Excellent choice for 50+ seed scenarios

**Future optimizations noted:**
- `gradient_norm` baseline normalization per layer type (Conv vs LayerNorm)
- `layer_id` string -> integer mapping at init time
- `torch._foreach_norm` for gradient calculation speed

## Appendix: Existing Telemetry Contracts

For reference, leyline already has:
- `TelemetryEvent` - event-based telemetry (things that happened)
- `TelemetryEventType` - lifecycle, training, health, command events
- `PerformanceBudgets` - timing and memory constraints

`SeedTelemetry` is a **snapshot contract** (point-in-time state) complementing these **event contracts**.
