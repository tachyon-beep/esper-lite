# SME Report: esper.leyline

**Package:** Data Contracts Layer
**Location:** `src/esper/leyline/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert (Merged)

---

## 1. Executive Summary

The leyline package is a well-structured data contracts layer with a notable two-tier signal architecture (FastTrainingSignals for hot-path inference, TrainingSignals for rich context). It defines the shared language for seed lifecycle stages, actions, observations, and telemetry events. Overall complexity is LOW-MEDIUM with appropriate coupling patterns for a contracts layer.

---

## 2. Key Features & Responsibilities

| Feature | Description |
|---------|-------------|
| **Seed Lifecycle** | 10-stage state machine (SeedStage enum) with explicit transitions |
| **Action Space** | Dynamic action enum building per topology |
| **Signals** | Two-tier representation (fast + rich) for different performance needs |
| **Commands** | Immutable AdaptationCommand for controller→subsystem communication |
| **Reports** | SeedMetrics, SeedStateReport, FieldReport for lifecycle tracking |
| **Telemetry** | Structured event types with performance budgets |

---

## 3. Notable Innovations

### Two-Tier Signal Architecture
- **FastTrainingSignals**: NamedTuple (immutable, GC-free) for hot-path inference
- **TrainingSignals**: Dataclass with rich context for decision-making
- **Benefit**: Eliminates garbage collection pressure during PPO rollouts

### TensorSchema IntEnum
- Maps feature names to vector indices (0-26)
- Enables symbolic indexing: `obs[TensorSchema.VAL_ACCURACY]`
- Avoids string dictionary lookups in hot path

### State Machine with Helpers
- `VALID_TRANSITIONS` dict defines allowed stage changes
- `is_terminal_stage()`, `is_active_stage()`, `is_failure_stage()` predicates
- Clean API for lifecycle validation

---

## 4. Complexity Analysis

| Aspect | Rating | Notes |
|--------|--------|-------|
| Overall | LOW-MEDIUM | Pure data contracts, minimal logic |
| Coupling | APPROPRIATE | Most-imported package (expected) |
| State Machine | MEDIUM | 10 stages, 12+ transitions |

---

## 5. DRL Specialist Assessment

### Signal Quality for RL

| Aspect | Rating | Notes |
|--------|--------|-------|
| Feature Coverage | GOOD | 27 features cover essential decision factors |
| Normalization | NEEDS WORK | Unbounded features (epoch, global_step) may hurt policy |
| History | GOOD | 5-epoch windows for loss/accuracy trends |
| Seed State | GOOD | Stage, alpha, improvement captured |

### Action Space Design

| Aspect | Rating | Notes |
|--------|--------|-------|
| Completeness | GOOD | WAIT, GERMINATE_*, FOSSILIZE, CULL |
| Topology-Aware | EXCELLENT | Dynamic enum per architecture |
| Ordering | GOOD | Actions sorted by parameter count |

### Issues

1. **Unbounded Features**: `epoch` and `global_step` should be normalized to [0,1]
2. **Missing Features**: No gradient health in base features (available via telemetry extension)

---

## 6. PyTorch Specialist Assessment

### Tensor-Friendly Design

| Aspect | Rating | Notes |
|--------|--------|-------|
| Vectorization | GOOD | `to_vector()` methods on signals |
| Device Awareness | MISSING | No `.to(device)` methods on signals |
| Memory Efficiency | GOOD | Slots optimization, NamedTuple for hot path |

### GPU-Native Considerations

| Aspect | Rating | Notes |
|--------|--------|-------|
| Allocation | GOOD | Fixed-size history tuples avoid GC |
| Conversion | NEEDS WORK | History padding in `to_fast()` is O(n) |

### Recommendations

1. Add `to_tensor(device)` method for direct GPU tensor creation
2. Pre-allocate history padding to avoid repeated tuple creation

---

## 7. Risks & Technical Debt

| Risk | Severity | Description |
|------|----------|-------------|
| SimicAction Alias | MEDIUM | `SimicAction = Action` violates no-legacy-code policy |
| Schema Versioning | LOW | No version field for backwards compatibility |
| Unbounded Features | MEDIUM | May cause policy instability |

---

## 8. Opportunities for Improvement

### High Value
1. **Normalize unbounded features** in feature extraction layer
2. **Remove SimicAction alias** - clean up deprecated export
3. **Add schema versioning** for future-proofing

### Medium Value
4. **Add `to_tensor()` methods** for GPU-native conversion
5. **Optimize history padding** with pre-allocated buffers

### Low Value
6. **Add batch conversion utilities** for vectorized training

---

## 9. Critical Issues

### SimicAction Alias (MEDIUM)
```python
# leyline/actions.py:6
SimicAction = Action  # deprecated alias
```
**Issue:** Violates codebase no-legacy-code policy
**Fix:** Remove alias, update all call sites

### Unbounded Features (MEDIUM)
```python
# leyline/signals.py - TensorSchema
EPOCH = 0           # Unbounded: 0-75+
GLOBAL_STEP = 1     # Unbounded: 0-10000+
```
**Issue:** Policy networks may struggle with unbounded inputs
**Fix:** Normalize in feature extraction: `epoch / max_epochs`

---

## 10. Recommendations Summary

| Priority | Recommendation | Effort |
|----------|----------------|--------|
| P0 | Remove SimicAction alias | 30 min |
| P0 | Normalize unbounded features | 1 hour |
| P1 | Add to_tensor() methods | 2 hours |
| P2 | Add schema versioning | 1 hour |
| P3 | Optimize history padding | 30 min |

---

**Quality Score:** 8/10 - Production-ready with minor improvements recommended
**Confidence:** HIGH
