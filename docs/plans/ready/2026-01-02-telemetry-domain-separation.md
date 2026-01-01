# Telemetry Domain Separation Plan

**Status:** Ready for Implementation
**Created:** 2026-01-02
**Author:** Claude Code
**Priority:** High (blocks telemetry interpretability)

## Executive Summary

This plan addresses two categories of issues discovered in the telemetry system:

1. **Data Bugs**: `inner_epoch` always reports 150 (max), `grad_norm` always reports 1.0 (post-clip)
2. **Architectural Confusion**: Event types don't indicate which domain (Tamiyo/Tolaria/Kasmina/Simic) they belong to, making telemetry data hard to interpret

The solution is to:
- Rename event types to include domain prefixes
- Fix the payload bugs
- Update all consumers (MCP views, Sanctum widgets, Overwatch)
- Maintain backward compatibility during migration

## Problem Statement

### Bug 1: `inner_epoch` Always 150

**Location:** `src/esper/simic/training/vectorized.py:3541`

**Root Cause:** PPO updates happen AFTER the epoch loop completes. The `epoch` variable passed to telemetry is the loop variable's final value (`max_epochs`), not the PPO update iteration.

```python
# Current (broken):
for epoch in range(1, max_epochs + 1):
    # ... epoch loop runs 1-150 ...
    pass
# epoch == 150 here

# PPO update happens here, OUTSIDE the loop
batch_emitter.on_ppo_update(epoch=epoch, ...)  # Always 150!
```

**Impact:** Cannot track which PPO update within a batch produced the metrics.

### Bug 2: `grad_norm` Always 1.0

**Location:** `src/esper/simic/agent/ppo.py:792` and `vectorized.py:3475`

**Root Cause:** Gradient norm is measured AFTER `clip_grad_norm_()` clips gradients to `max_grad_norm=1.0`.

```python
# In ppo.py:792 - clips gradients, DISCARDS the pre-clip norm
nn.utils.clip_grad_norm_(self.policy.network.parameters(), self.max_grad_norm)

# In vectorized.py:3475 - measures AFTER clipping
ppo_grad_norm = compute_grad_norm_surrogate(agent.policy.network)  # Always ~1.0!
```

**Impact:** Cannot detect gradient explosion (pre-clip norm >> 1.0) vs healthy gradients.

### Architectural Issue: Domain Confusion

Current event types don't indicate their source domain:

| Event | Emitted By | Semantic Domain | Confusion |
|-------|-----------|-----------------|-----------|
| `PPO_UPDATE_COMPLETED` | Simic | **Tamiyo** (policy) | Sounds like Simic |
| `EPOCH_COMPLETED` | Simic | **Tolaria** (host training) | Generic name |
| `BATCH_EPOCH_COMPLETED` | Simic | **Tolaria** | Generic name |
| `GOVERNOR_ROLLBACK` | Tolaria | Tolaria | Partially correct |

When analyzing telemetry, users must know implementation details to understand which subsystem generated each event.

## Design Goals

1. **Self-Documenting Events**: Event type names should indicate their domain
2. **Accurate Metrics**: Fix `inner_epoch` and `grad_norm` bugs
3. **Backward Compatibility**: Support both old and new event types during migration
4. **Minimal UX Disruption**: Existing dashboards continue working

## Detailed Changes

### Phase 1: Leyline Contract Changes

#### 1.1 New Event Types

Add to `TelemetryEventType` enum in `src/esper/leyline/telemetry.py`:

```python
class TelemetryEventType(Enum):
    # === Tamiyo Domain (Brain/Policy) ===
    TAMIYO_POLICY_UPDATE = auto()      # PPO gradient update completed
    TAMIYO_INITIATED = auto()          # (existing) Host stabilized

    # === Tolaria Domain (Metabolism/Training) ===
    TOLARIA_EPOCH_COMPLETED = auto()   # Per-env host training epoch
    TOLARIA_BATCH_COMPLETED = auto()   # Batch of envs finished
    TOLARIA_ROLLBACK = auto()          # Emergency rollback

    # === Kasmina Domain (Stem Cells/Seeds) ===
    # SEED_* events already correctly named - no changes

    # === Simic Domain (Evolution/Fitness) ===
    # PLATEAU_DETECTED, DEGRADATION_DETECTED, etc. - keep as-is
    # These are fitness signals, correctly in Simic domain

    # === Deprecated (emit both old and new during migration) ===
    PPO_UPDATE_COMPLETED = auto()      # -> TAMIYO_POLICY_UPDATE
    EPOCH_COMPLETED = auto()           # -> TOLARIA_EPOCH_COMPLETED
    BATCH_EPOCH_COMPLETED = auto()     # -> TOLARIA_BATCH_COMPLETED
    GOVERNOR_ROLLBACK = auto()         # -> TOLARIA_ROLLBACK

    # === Dead Code (delete) ===
    # ISOLATION_VIOLATION - never emitted, remove
```

#### 1.2 Payload Changes

**Rename:** `PPOUpdatePayload` → `TamiyoPolicyUpdatePayload`

**New fields:**
```python
@dataclass(slots=True, frozen=True)
class TamiyoPolicyUpdatePayload:
    # ... existing fields ...

    # NEW: PPO update iteration within the batch (replaces misleading inner_epoch)
    ppo_update_idx: int = 1  # 1, 2, 3... for ppo_updates_per_batch

    # NEW: Pre-clip gradient norm (actual gradient magnitude)
    pre_clip_grad_norm: float = 0.0

    # EXISTING: Post-clip gradient norm (kept for reference)
    grad_norm: float = 0.0

    # DEPRECATED: inner_epoch (remove after migration)
    # Was always max_epochs, semantically meaningless for PPO updates
```

**Rename:** `EpochCompletedPayload` → `TolariaEpochPayload`
**Rename:** `BatchEpochCompletedPayload` → `TolariaBatchPayload`
**Rename:** `GovernorRollbackPayload` → `TolariaRollbackPayload`

### Phase 2: Emission Changes

#### 2.1 Fix grad_norm Bug

**File:** `src/esper/simic/agent/ppo.py`

```python
# Line 792 - capture pre-clip norm
pre_clip_norm = nn.utils.clip_grad_norm_(
    self.policy.network.parameters(),
    self.max_grad_norm
)
metrics["pre_clip_grad_norm"].append(float(pre_clip_norm))
```

#### 2.2 Fix inner_epoch Bug

**File:** `src/esper/simic/training/vectorized.py`

```python
# In _run_ppo_updates, track update index
for update_idx in range(updates_to_run):
    # ... existing code ...
    metrics["ppo_update_idx"] = update_idx + 1  # 1-indexed

# Pass to telemetry
batch_emitter.on_ppo_update(
    ppo_update_idx=metrics.get("ppo_update_idx", 1),  # NEW
    # Remove: epoch=epoch (was always max_epochs)
    ...
)
```

#### 2.3 Emit New Event Types

**File:** `src/esper/simic/telemetry/emitters.py`

During migration, emit BOTH old and new event types:

```python
def emit_ppo_update_event(...):
    # Emit new event type
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.TAMIYO_POLICY_UPDATE,
        data=TamiyoPolicyUpdatePayload(...),
    ))

    # Also emit old event type (backward compat)
    hub.emit(TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(...),  # Old payload format
    ))
```

### Phase 3: Consumption Changes

#### 3.1 MCP View Updates

**File:** `src/esper/karn/mcp/views.py`

Create new views alongside old ones:

```sql
-- New view (primary)
CREATE OR REPLACE VIEW tamiyo_updates AS
SELECT ...
FROM raw_events
WHERE event_type IN ('TAMIYO_POLICY_UPDATE', 'PPO_UPDATE_COMPLETED')

-- Old view (deprecated, points to new)
CREATE OR REPLACE VIEW ppo_updates AS
SELECT * FROM tamiyo_updates
```

| Old View | New View | Notes |
|----------|----------|-------|
| `epochs` | `tolaria_epochs` | Accept both event types |
| `ppo_updates` | `tamiyo_updates` | Accept both event types |
| `batch_epochs` | `tolaria_batches` | Accept both event types |

#### 3.2 Sanctum Aggregator Updates

**File:** `src/esper/karn/sanctum/aggregator.py`

Update event routing to handle both:

```python
def _handle_event(self, event: TelemetryEvent) -> None:
    event_type = event.event_type.name

    # Handle both old and new event types
    if event_type in ("TAMIYO_POLICY_UPDATE", "PPO_UPDATE_COMPLETED"):
        self._handle_tamiyo_policy_update(event)
    elif event_type in ("TOLARIA_EPOCH_COMPLETED", "EPOCH_COMPLETED"):
        self._handle_tolaria_epoch(event)
    # ... etc
```

#### 3.3 Event Log Widget Updates

**File:** `src/esper/karn/sanctum/widgets/event_log.py`

Update color mappings:

```python
EVENT_COLORS = {
    # New names (primary)
    "TAMIYO_POLICY_UPDATE": "bright_magenta",
    "TOLARIA_EPOCH_COMPLETED": "bright_blue",
    "TOLARIA_BATCH_COMPLETED": "bright_blue",
    "TOLARIA_ROLLBACK": "bright_red",

    # Old names (deprecated, same colors)
    "PPO_UPDATE_COMPLETED": "bright_magenta",
    "EPOCH_COMPLETED": "bright_blue",
    "BATCH_EPOCH_COMPLETED": "bright_blue",
    "GOVERNOR_ROLLBACK": "bright_red",
}
```

#### 3.4 Dashboard HTML Updates

**File:** `src/esper/karn/dashboard.html`

Update JavaScript event handlers:

```javascript
case 'TAMIYO_POLICY_UPDATE':
case 'PPO_UPDATE_COMPLETED':  // Backward compat
    handleTamiyoUpdate(event);
    break;
```

### Phase 4: Cleanup (Post-Migration)

After all consumers are updated and tested:

1. Stop emitting deprecated event types
2. Remove deprecated event types from enum
3. Remove backward-compat view aliases
4. Remove old payload classes
5. Update documentation

## File Impact Summary

### Must Change (Core)

| File | Changes |
|------|---------|
| `src/esper/leyline/telemetry.py` | New event types, payload renames, new fields |
| `src/esper/simic/agent/ppo.py` | Capture pre_clip_grad_norm |
| `src/esper/simic/training/vectorized.py` | Pass ppo_update_idx, remove epoch |
| `src/esper/simic/telemetry/emitters.py` | Emit new event types, update payloads |
| `src/esper/tolaria/governor.py` | Emit TOLARIA_ROLLBACK |

### Must Change (Consumption)

| File | Changes |
|------|---------|
| `src/esper/karn/mcp/views.py` | New view names, accept both event types |
| `src/esper/karn/mcp/server.py` | Update view descriptions |
| `src/esper/karn/mcp/reports.py` | Update queries to use new views |
| `src/esper/karn/collector.py` | Handle both event types |
| `src/esper/karn/store.py` | Handle both event types |
| `src/esper/karn/sanctum/aggregator.py` | Handle both event types |
| `src/esper/karn/sanctum/widgets/event_log.py` | Color mappings for new types |
| `src/esper/karn/dashboard.html` | JS event handlers |

### Must Change (Tests)

| Pattern | Estimated Count |
|---------|-----------------|
| `tests/**/test_*.py` referencing old event types | ~20 files |
| Mock data generators | ~5 files |

## Migration Timeline

### Week 1: Contract & Emission
- [ ] Add new event types to Leyline
- [ ] Add new payload classes
- [ ] Fix grad_norm bug in ppo.py
- [ ] Fix inner_epoch bug in vectorized.py
- [ ] Update emitters to emit both old and new

### Week 2: Consumption
- [ ] Update MCP views (accept both)
- [ ] Update Sanctum aggregator (accept both)
- [ ] Update Sanctum widgets
- [ ] Update dashboard.html

### Week 3: Testing & Validation
- [ ] Update all tests
- [ ] Run full test suite
- [ ] Manual testing with Sanctum TUI
- [ ] Manual testing with Overwatch web

### Week 4: Cleanup
- [ ] Remove deprecated event types
- [ ] Remove backward-compat code
- [ ] Update documentation
- [ ] Final validation

## Rollback Plan

If issues arise:

1. **Emission rollback**: Revert emitter changes to only emit old event types
2. **Consumption rollback**: Views/aggregators already handle both, no change needed
3. **Full rollback**: Revert all Leyline changes, return to previous state

The dual-emit strategy means consumers can be rolled back independently of emitters.

## Success Criteria

1. **Bug fixes verified:**
   - `ppo_update_idx` shows 1, 2, 3... (not always 150)
   - `pre_clip_grad_norm` shows actual gradient magnitude (varies with training)
   - `grad_norm` shows post-clip value (typically ~1.0 when clipping active)

2. **Domain clarity:**
   - `TAMIYO_*` events clearly indicate policy/brain operations
   - `TOLARIA_*` events clearly indicate host training/metabolism
   - `KASMINA_*` (SEED_*) events clearly indicate seed lifecycle

3. **No regression:**
   - All existing tests pass
   - Sanctum TUI displays correctly
   - Overwatch web dashboard displays correctly
   - MCP queries return expected data

## Open Questions

1. **View naming:** Should we use `tamiyo_updates` or `tamiyo_policy_updates`?
2. **Backward compat duration:** How long to emit both old and new? (Proposed: 2 weeks)
3. **ANALYTICS_SNAPSHOT:** This multi-kind event spans domains - should it be split?

## Appendix A: Complete Event Domain Mapping

| Event Type | Domain | Emitter | Payload |
|-----------|--------|---------|---------|
| `TAMIYO_POLICY_UPDATE` | Tamiyo | Simic (emitters.py) | TamiyoPolicyUpdatePayload |
| `TAMIYO_INITIATED` | Tamiyo | Tamiyo (tracker.py) | TamiyoInitiatedPayload |
| `TOLARIA_EPOCH_COMPLETED` | Tolaria | Simic (vectorized.py) | TolariaEpochPayload |
| `TOLARIA_BATCH_COMPLETED` | Tolaria | Simic (emitters.py) | TolariaBatchPayload |
| `TOLARIA_ROLLBACK` | Tolaria | Tolaria (governor.py) | TolariaRollbackPayload |
| `SEED_GERMINATED` | Kasmina | Kasmina (slot.py) | SeedGerminatedPayload |
| `SEED_STAGE_CHANGED` | Kasmina | Kasmina (slot.py) | SeedStageChangedPayload |
| `SEED_GATE_EVALUATED` | Kasmina | Kasmina (slot.py) | SeedGateEvaluatedPayload |
| `SEED_FOSSILIZED` | Kasmina | Kasmina (slot.py) | SeedFossilizedPayload |
| `SEED_PRUNED` | Kasmina | Kasmina (slot.py) | SeedPrunedPayload |
| `PLATEAU_DETECTED` | Simic | Simic (vectorized.py) | TrendDetectedPayload |
| `DEGRADATION_DETECTED` | Simic | Simic (vectorized.py) | TrendDetectedPayload |
| `IMPROVEMENT_DETECTED` | Simic | Simic (vectorized.py) | TrendDetectedPayload |
| `TRAINING_STARTED` | Simic | Simic (vectorized.py) | TrainingStartedPayload |
| `CHECKPOINT_LOADED` | Simic | Simic (vectorized.py) | (none) |
| `EPISODE_OUTCOME` | Simic | Simic (vectorized.py) | EpisodeOutcomePayload |
| `COUNTERFACTUAL_MATRIX_COMPUTED` | Simic | Simic (vectorized.py) | CounterfactualMatrixPayload |
| `ANALYTICS_SNAPSHOT` | Mixed | Simic (emitters.py) | AnalyticsSnapshotPayload |
| `GRADIENT_ANOMALY` | Simic | Simic (vectorized.py) | AnomalyDetectedPayload |
| `RATIO_EXPLOSION_DETECTED` | Simic | Simic (vectorized.py) | AnomalyDetectedPayload |
| `RATIO_COLLAPSE_DETECTED` | Simic | Simic (vectorized.py) | AnomalyDetectedPayload |
| `VALUE_COLLAPSE_DETECTED` | Simic | Simic (vectorized.py) | AnomalyDetectedPayload |
| `GRADIENT_PATHOLOGY_DETECTED` | Simic | Simic (vectorized.py) | AnomalyDetectedPayload |
| `NUMERICAL_INSTABILITY_DETECTED` | Simic | Simic (vectorized.py) | AnomalyDetectedPayload |
| `PERFORMANCE_DEGRADATION` | Simic | Simic (emitters.py) | PerformanceDegradationPayload |
| `MEMORY_WARNING` | Simic | (unused) | - |
| `REWARD_HACKING_SUSPECTED` | Simic | (unused) | - |

## Appendix B: MCP View Dependency Graph

```
raw_events (base)
    │
    ├── runs ← TRAINING_STARTED
    │
    ├── tolaria_epochs ← TOLARIA_EPOCH_COMPLETED, EPOCH_COMPLETED
    │       │
    │       └── (used by: Sanctum env_overview, health monitors)
    │
    ├── tamiyo_updates ← TAMIYO_POLICY_UPDATE, PPO_UPDATE_COMPLETED
    │       │
    │       └── (used by: Sanctum tamiyo_brain, health monitors)
    │
    ├── tolaria_batches ← TOLARIA_BATCH_COMPLETED, BATCH_EPOCH_COMPLETED
    │       │
    │       └── (used by: Sanctum progress, run_header)
    │
    ├── seed_lifecycle ← SEED_* events
    │       │
    │       └── (used by: Sanctum event_log, env_detail)
    │
    ├── decisions ← ANALYTICS_SNAPSHOT (kind=last_action)
    │       │
    │       └── (used by: Sanctum tamiyo_brain)
    │
    ├── rewards ← ANALYTICS_SNAPSHOT (kind=last_action) + RewardComponents
    │       │
    │       └── (used by: Sanctum reward panels)
    │
    ├── anomalies ← All anomaly events
    │       │
    │       └── (used by: Sanctum anomaly_strip, health monitors)
    │
    └── episode_outcomes ← EPISODE_OUTCOME
            │
            └── (used by: Sanctum scoreboard, Pareto analysis)
```

## Appendix C: DRL Expert Recommendations (Tamiyo Domain)

The following fields are recommended by the DRL specialist for comprehensive policy telemetry.

### C.1 Critical Bug Fixes (TIER 1)

| Field | Type | Why It Matters | Audience |
|-------|------|----------------|----------|
| `pre_clip_grad_norm` | `float` | **Bug Fix**: Detect gradient explosion before clipping | All |
| `ppo_update_idx` | `int` | **Bug Fix**: Track which PPO update (1,2,3...) vs always 150 | All |

### C.2 Trust Region Diagnostics (HIGH PRIORITY)

| Field | Type | Why It Matters | Audience |
|-------|------|----------------|----------|
| `approx_kl_max` | `float` | Maximum KL per timestep. High max with low mean = localized drift | WandB, SQL |
| `kl_reverse` | `float` | KL(new \|\| old) - asymmetry signals mode collapse direction | WandB, SQL |
| `trust_region_violations` | `int` | Count of timesteps where ratio exceeded clip bounds | TUI, WandB, SQL |

### C.3 Value Function Quality (HIGH PRIORITY)

| Field | Type | Why It Matters | Audience |
|-------|------|----------------|----------|
| `value_prediction_error_mean` | `float` | Mean of (V(s) - returns). Persistent bias = systematic error | WandB, SQL |
| `value_prediction_error_std` | `float` | Variance in prediction errors | WandB, SQL |
| `td_error_mean` | `float` | Mean temporal difference error from GAE | WandB |
| `return_mean` | `float` | Mean of discounted returns | TUI, WandB, SQL |
| `return_std` | `float` | Return variance - collapsed or unstable rewards | WandB, SQL |
| `value_target_correlation` | `float` | Correlation between V(s) and returns. <0.5 = value not learning | WandB |

### C.4 Per-Head Policy Dynamics (MEDIUM PRIORITY)

| Field | Type | Why It Matters | Audience |
|-------|------|----------------|----------|
| `head_{name}_kl` | `float` (x8) | Per-head KL divergence | WandB, SQL |
| `head_{name}_clip_fraction` | `float` (x8) | Per-head clipping rates | WandB, SQL |
| `head_{name}_active_ratio` | `float` (x8) | Fraction of timesteps causally active | TUI, WandB |
| `causal_mask_coverage` | `dict[str, float]` | Per-head batch relevance fraction | SQL |

### C.5 Policy Quality Assessment (MEDIUM PRIORITY)

| Field | Type | Why It Matters | Audience |
|-------|------|----------------|----------|
| `policy_determinism` | `float` | 1 - (entropy / max_entropy). >0.95 = near-deterministic | TUI, WandB |
| `action_diversity` | `int` | Unique action combinations in batch | WandB |
| `dominant_action_ratio` | `float` | Fraction taking most common action. >0.8 = collapse | TUI, WandB |
| `op_distribution` | `dict[str, float]` | Per-op probabilities. WAIT >70% = risk-avoidance | TUI, WandB |

### C.6 Numerical Stability (HIGH for debugging)

| Field | Type | Why It Matters | Audience |
|-------|------|----------------|----------|
| `log_prob_underflow_count` | `int` | Timesteps with log_prob < -50. NaN precursor | TUI, WandB |
| `numerical_stability_score` | `float` | Composite 0-1 score (1=healthy) | TUI |
| `grad_clip_fraction` | `float` | Fraction of updates where gradients were clipped | WandB |

### C.7 Sample Efficiency (MEDIUM PRIORITY)

| Field | Type | Why It Matters | Audience |
|-------|------|----------------|----------|
| `valid_sample_ratio` | `float` | valid_samples / total_samples after masking | WandB |
| `policy_update_magnitude` | `float` | L2 norm of weight change | WandB |
| `gradient_utilization` | `float` | Ratio of gradient update to max_grad_norm | WandB |

## Appendix D: PyTorch Expert Recommendations (Tolaria/Simic Infrastructure)

The following fields are recommended by the PyTorch specialist for infrastructure telemetry.

### D.1 torch.compile & Inductor Metrics (PyTorch 2.x)

| Field | Type | Domain | Why It Matters | Overhead |
|-------|------|--------|----------------|----------|
| `compile_graph_breaks` | `int` | Tolaria | Graph breaks = suboptimal compilation | Medium |
| `compile_recompilations` | `int` | Tolaria | Dynamic shape issues or guard failures | Medium |
| `compile_cache_hits` | `int` | Tolaria | Low ratio = excessive overhead | Medium |
| `dynamo_guard_failures` | `int` | Tolaria | Guard failures triggering recompile | Medium |

### D.2 GPU Utilization & Memory (CRITICAL)

| Field | Type | Domain | Why It Matters | Overhead |
|-------|------|--------|----------------|----------|
| `gpu_utilization_pct` | `float` | Tolaria | SM utilization. Low = CPU-bound | Medium (pynvml) |
| `cuda_memory_active_gb` | `float` | Tolaria | Actually used memory (more accurate) | Low |
| `cuda_memory_inactive_gb` | `float` | Tolaria | Cached but not active = fragmentation | Low |
| `cuda_oom_events` | `int` | Tolaria | Any >0 indicates memory pressure | Low |
| `memory_allocated_delta_gb` | `float` | Tolaria | Memory change since last batch (leak detection) | Low |

### D.3 Forward/Backward Pass Timing

| Field | Type | Domain | Why It Matters | Overhead |
|-------|------|--------|----------------|----------|
| `forward_time_ms` | `float` | Tolaria | Host forward pass time | Low |
| `backward_time_ms` | `float` | Tolaria | Should be ~2x forward | Low |
| `optimizer_step_time_ms` | `float` | Tolaria | High = large parameter count | Low |
| `data_transfer_time_ms` | `float` | Tolaria | High = DataLoader bottleneck | Low |
| `host_epoch_time_ms` | `float` | Tolaria | Total host training epoch time | Low |

### D.4 Gradient Flow Health (Per-Layer)

| Field | Type | Domain | Why It Matters | Overhead |
|-------|------|--------|----------------|----------|
| `grad_norm_per_layer` | `dict[str, float]` | Simic | Per-layer norms (pre-clip) | Medium |
| `grad_zero_fraction` | `float` | Simic | High = dead neurons | Low |
| `grad_norm_ratio` | `float` | Simic | max/min layer norm. High = imbalance | Low |
| `grad_nan_layers` | `list[str]` | Simic | Names of NaN gradient layers | Low |

### D.5 AMP & Loss Scaling

| Field | Type | Domain | Why It Matters | Overhead |
|-------|------|--------|----------------|----------|
| `loss_scale_updates` | `int` | Simic | Number of scale adjustments this batch | Low |
| `amp_overflow_count` | `int` | Simic | Overflows detected. >0 = instability | Low |
| `grad_scaler_state` | `str` | Simic | "growing" / "stable" / "decaying" | Low |

### D.6 Host Training Metrics (Tolaria-Specific)

| Field | Type | Domain | Why It Matters | Overhead |
|-------|------|--------|----------------|----------|
| `host_train_loss` | `float` | Tolaria | Host model training loss (CE) | Low |
| `host_grad_norm_pre_clip` | `float` | Tolaria | Host gradient norm BEFORE clipping | Low |
| `host_lr` | `float` | Tolaria | Current learning rate (for schedulers) | Low |

### D.7 New Payload: InfrastructureSnapshot

For periodic (every N batches) infrastructure health checks:

```python
@dataclass(slots=True, frozen=True)
class InfrastructureSnapshotPayload:
    """Periodic infrastructure health snapshot. Collected every N batches."""

    # CUDA Memory (detailed)
    cuda_memory_allocated_gb: float
    cuda_memory_reserved_gb: float
    cuda_memory_active_gb: float
    cuda_memory_inactive_gb: float
    cuda_memory_peak_gb: float
    cuda_memory_fragmentation: float

    # GPU Utilization (requires pynvml)
    gpu_utilization_pct: float | None = None
    gpu_memory_bandwidth_pct: float | None = None

    # torch.compile stats
    compile_graph_breaks: int = 0
    compile_recompilations: int = 0
    compile_cache_hits: int = 0
    compile_cache_misses: int = 0

    # Timing breakdown (CUDA events)
    avg_forward_time_ms: float = 0.0
    avg_backward_time_ms: float = 0.0
    avg_optimizer_time_ms: float = 0.0
    avg_data_transfer_time_ms: float = 0.0

    # Collection context
    batch_window: int = 10  # Number of batches this snapshot covers
```

## Appendix E: Field Priority Matrix

### Implementation Phases

| Phase | Category | Fields | Effort |
|-------|----------|--------|--------|
| **1** | Bug Fixes | `pre_clip_grad_norm`, `ppo_update_idx` | 1 day |
| **2** | Trust Region | `approx_kl_max`, `trust_region_violations` | 2 days |
| **3** | Value Function | `value_prediction_error_*`, `return_*` | 2 days |
| **4** | Infrastructure | `forward/backward_time_ms`, memory deltas | 2 days |
| **5** | Per-Head | `head_{name}_kl`, `head_{name}_clip_fraction` | 3 days |
| **6** | torch.compile | `compile_graph_breaks`, cache stats | 3 days |
| **7** | GPU Utilization | `gpu_utilization_pct` (pynvml integration) | 2 days |

### Collection Overhead Summary

| Overhead | Description | When to Collect |
|----------|-------------|-----------------|
| **Low** | Already computed or single tensor.item() | Every step |
| **Medium** | Requires iteration or extra GPU sync | Every N batches (N=5-10) |
| **High** | Full graph traversal or external tool call | Every N episodes or on-demand |

## Appendix F: WandB Integration Recommendations

### Recommended WandB Logging Groups

```python
wandb.log({
    # Tamiyo (Policy) - log every PPO update
    "tamiyo/policy_loss": ...,
    "tamiyo/value_loss": ...,
    "tamiyo/entropy": ...,
    "tamiyo/kl_divergence": ...,
    "tamiyo/pre_clip_grad_norm": ...,  # NEW
    "tamiyo/ppo_update_idx": ...,       # NEW

    # Tamiyo per-head (sparse logging every N updates)
    "tamiyo/head_slot_entropy": ...,
    "tamiyo/head_slot_kl": ...,         # NEW
    "tamiyo/head_op_clip_fraction": ..., # NEW

    # Tolaria (Host Training) - log every epoch
    "tolaria/train_loss": ...,
    "tolaria/val_accuracy": ...,
    "tolaria/forward_time_ms": ...,      # NEW
    "tolaria/backward_time_ms": ...,     # NEW

    # Simic (Fitness) - log every batch
    "simic/episode_reward": ...,
    "simic/stability_score": ...,

    # Infrastructure - log every N batches
    "infra/gpu_utilization_pct": ...,    # NEW
    "infra/cuda_memory_active_gb": ...,  # NEW
    "infra/compile_graph_breaks": ...,   # NEW
})
```

### WandB Custom Charts

1. **Gradient Health Dashboard**: `pre_clip_grad_norm` vs `grad_norm` over time
2. **Trust Region Monitor**: `kl_divergence`, `approx_kl_max`, `clip_fraction`
3. **Value Function Quality**: `explained_variance`, `value_target_correlation`
4. **Per-Head Entropy**: All 8 head entropies as stacked area chart
5. **Infrastructure Timeline**: Memory, GPU util, timing breakdowns

## Appendix G: Sanctum UX Telemetry Categories

The Sanctum TUI consumes telemetry across **9 distinct information categories**. These categories must be preserved regardless of event naming changes.

### G.1 Information Categories

| Category | Description | Update Frequency | Widgets Consuming |
|----------|-------------|------------------|-------------------|
| **Run Configuration** | Task, hyperparams, devices, slot layout | Once (start) | RunHeader, EnvOverview |
| **Host Training State** | Per-env accuracy, loss, epoch progress | Per-epoch | EnvOverview, EnvDetail |
| **Policy Optimization** | Loss, entropy, gradients, clip fraction | Per-PPO-update | TamiyoBrain, HealthStatus |
| **Per-Head Dynamics** | Head entropies, grad norms, ratios | Per-PPO-update | HeadsGrid, AttentionHeatmap |
| **Seed Lifecycle** | Stage transitions, alpha, contribution | Per-event | EnvOverview, EnvDetail, SlotsPanel |
| **Decision Context** | Action choice, confidence, alternatives | Per-step | DecisionsColumn, ActionContext |
| **Reward Breakdown** | Components: base, PBRS, penalties, credit | Per-step | RewardHealth, EnvDetail |
| **Attribution Analysis** | Counterfactuals, Shapley values | Per-episode | CounterfactualPanel, ShapleyPanel |
| **System Health** | GPU/CPU/RAM, throughput, alarms | Continuous | RunHeader, SystemStatus, AnomalyStrip |

### G.2 Domain Mapping of Categories

| Category | Primary Domain | Secondary Domain |
|----------|----------------|------------------|
| Run Configuration | Simic | - |
| Host Training State | **Tolaria** | Kasmina (seed data within) |
| Policy Optimization | **Tamiyo** | - |
| Per-Head Dynamics | **Tamiyo** | - |
| Seed Lifecycle | **Kasmina** | - |
| Decision Context | **Tamiyo** | - |
| Reward Breakdown | **Simic** | Tolaria (host contribution) |
| Attribution Analysis | **Simic** | Kasmina (per-seed values) |
| System Health | Infrastructure | - |

### G.3 Critical Sanctum Fields by Priority

**TIER 1 - Core Display (Breaks UI if missing):**
- `task_name`, `current_episode`, `current_epoch`, `max_epochs`
- `val_accuracy`, `val_loss` (per-env)
- `policy_loss`, `value_loss`, `entropy`, `kl_divergence`
- `seed.stage`, `seed.blueprint_id`, `seed.alpha`

**TIER 2 - Health Monitoring (Degrades insight if missing):**
- `grad_norm`, `clip_fraction`, `explained_variance`
- `head_*_entropy`, `head_*_grad_norm` (8 heads each)
- `dead_layers`, `exploding_layers`, `entropy_collapsed`
- `advantage_mean`, `advantage_std`, `ratio_max`

**TIER 3 - Deep Diagnostics (Nice-to-have):**
- `log_prob_min`, `log_prob_max` (NaN prediction)
- `head_*_ratio_max` (per-head PPO ratio extremes)
- `q_germinate`, `q_advance`, etc. (Q-values)
- `counterfactual_matrix`, `shapley_values`

### G.4 Identified Gaps

| Gap | Current State | Recommendation |
|-----|---------------|----------------|
| `pre_clip_grad_norm` | Not captured (always 1.0) | **BUG FIX** - Phase 1 |
| `ppo_update_idx` | Incorrect (always 150) | **BUG FIX** - Phase 1 |
| `training_thread_alive` | Hardcoded true | Add heartbeat event |
| `total_events_received` | Computed locally | Add to snapshot schema |
| Per-layer gradient health | Optional dict | Ensure consistent emission |
| Seed diversity metrics | Not captured | Future enhancement |

### G.5 Backward Compatibility for Sanctum

The Sanctum aggregator (`aggregator.py`) must handle both old and new event types during migration:

```python
# Example: Handle both PPO_UPDATE_COMPLETED and TAMIYO_POLICY_UPDATE
if event_type.name in ("TAMIYO_POLICY_UPDATE", "PPO_UPDATE_COMPLETED"):
    self._handle_tamiyo_policy_update(event)
```

**Key requirement:** All 9 information categories must continue flowing to widgets regardless of event name changes. The aggregator acts as the translation layer.
