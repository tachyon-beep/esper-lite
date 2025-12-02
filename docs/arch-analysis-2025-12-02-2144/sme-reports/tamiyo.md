# SME Report: esper.tamiyo

**Package:** Strategic Decision-Making Layer
**Location:** `src/esper/tamiyo/`
**Analysis Date:** 2025-12-02
**Specialists:** DRL Expert + PyTorch Expert (Merged)

---

## 1. Executive Summary

The tamiyo package is well-architected for eventual RL policy integration, with clean Protocol abstractions (`TamiyoPolicy`) and a 27-feature observation space that maps directly to `TensorSchema`. The heuristic baseline (`HeuristicTamiyo`) provides reasonable domain-appropriate thresholds but offers significant opportunities for learned policy improvement.

---

## 2. Key Features & Responsibilities

| Feature | Description |
|---------|-------------|
| **SignalTracker** | Aggregates training metrics into observation signals |
| **TamiyoPolicy** | Protocol for policy implementations |
| **HeuristicTamiyo** | Rule-based baseline policy |
| **TamiyoDecision** | Decision output with action, reason, confidence |
| **Blueprint Penalties** | Decay-based rotation to avoid failing blueprints |

---

## 3. Notable Innovations

### Blueprint Penalty System
```python
# Thompson Sampling-like exploration
self._blueprint_penalties: dict[str, float] = {}
penalty = self._blueprint_penalties.get(bp_name, 0.0)
if penalty > self.config.blueprint_penalty_threshold:
    continue  # Skip penalized blueprint
```
**Benefit:** Avoids repeatedly selecting blueprints that fail

### Anti-Thrashing Embargo
```python
if self._last_cull_epoch is not None:
    embargo_remaining = (self._last_cull_epoch + embargo_epochs) - epoch
    if embargo_remaining > 0:
        return TamiyoDecision(action=Action.WAIT, ...)
```
**Benefit:** Prevents GERMINATE-CULL oscillation

### Two-Tier Signal Flow
```
Raw Metrics → SignalTracker.update() → TrainingSignals → Policy.decide() → TamiyoDecision
```

---

## 4. Complexity Analysis

| Aspect | Rating | Notes |
|--------|--------|-------|
| Overall | LOW-MEDIUM | Clean abstractions |
| Decision Logic | MEDIUM | Cascading conditionals for stages |
| Signal Aggregation | LOW | Simple delta/plateau tracking |

---

## 5. DRL Specialist Assessment

### Policy Design Quality

| Aspect | Rating | Notes |
|--------|--------|-------|
| Protocol Abstraction | EXCELLENT | Clean `TamiyoPolicy` interface |
| Action Space | GOOD | 7 discrete actions |
| Observation Space | GOOD | 27 features from TensorSchema |
| Baseline Quality | GOOD | Domain-appropriate heuristics |

### Signal-to-Action Mapping

| Feature | Decision Impact |
|---------|-----------------|
| plateau_epochs | Triggers germination |
| seed_improvement | Determines fossilization |
| seed_epochs_in_stage | Age-based culling |
| accuracy_delta | Progress monitoring |

### RL Improvement Potential

| Capability | Heuristic | Learned | Gap |
|------------|-----------|---------|-----|
| Adaptive Thresholds | Fixed | Dynamic | HIGH |
| Blueprint Selection | Penalty-based | Value-aware | HIGH |
| Fossilization Timing | Rule-based | Reward-optimized | MEDIUM |

---

## 6. PyTorch Specialist Assessment

### Integration with Neural Policy

| Aspect | Rating | Notes |
|--------|--------|-------|
| Feature Extraction | GOOD | 27-dim maps to policy input |
| Action Encoding | GOOD | IntEnum for discrete actions |
| Batching | N/A | Single-step decisions |

### Efficiency Considerations

| Aspect | Rating | Notes |
|--------|--------|-------|
| Hot Path Impact | LOW | Decision-making is infrequent |
| Memory | LOW | Minimal state in tracker |

---

## 7. Risks & Technical Debt

| Risk | Severity | Description |
|------|----------|-------------|
| Single-Seed Assumption | MEDIUM | SignalTracker.update() takes single seed |
| Penalty Decay Timing | MEDIUM | Decays per-call, not per-epoch |
| Plateau Sensitivity | LOW | 0.5% threshold may be noisy |

---

## 8. Opportunities for Improvement

### High Value
1. **Multi-seed support** in SignalTracker
2. **Fix penalty decay timing** - decay once per episode
3. **Add telemetry** to decision events

### Medium Value
4. **LearnedTamiyo wrapper** for ActorCritic integration
5. **Confidence calibration** for heuristic decisions

### Low Value
6. **Decision caching** for repeated observations

---

## 9. Critical Issues

### Single-Seed Assumption (MEDIUM)
```python
# tracker.py:85
if seeds:
    seed = seeds[0]  # Only uses first seed
```
**Issue:** Will break with multi-slot configurations
**Fix:** Aggregate across all seeds or add slot_id parameter

### Penalty Decay Timing (MEDIUM)
```python
# heuristic.py:156
self._blueprint_penalties[name] *= (1 - self.config.blueprint_penalty_decay)
```
**Issue:** Decays every `_get_next_blueprint()` call
**Fix:** Decay once at episode start in `reset()`

---

## 10. Recommendations Summary

| Priority | Recommendation | Effort |
|----------|----------------|--------|
| P0 | Fix multi-seed signal extraction | 2 hours |
| P0 | Fix penalty decay timing | 30 min |
| P1 | Add decision telemetry | 1 hour |
| P2 | Create LearnedTamiyo wrapper | 4 hours |
| P3 | Calibrate plateau sensitivity | 1 hour |

---

**Quality Score:** 7.5/10 - Good baseline with room for RL enhancement
**Confidence:** HIGH
