# SME Report: esper.tamiyo Package

**Package Path:** `/home/john/esper-lite/src/esper/tamiyo/`
**Report Date:** 2025-12-02
**Analyst:** DRL/PyTorch Subject Matter Expert

---

## 1. Executive Summary

The `esper.tamiyo` package implements the strategic decision-making layer for Esper's seed lifecycle management, providing both signal aggregation (`SignalTracker`) and a heuristic policy baseline (`HeuristicTamiyo`) that manages GERMINATE/FOSSILIZE/CULL decisions. The architecture is well-designed for eventual replacement by a learned RL policy, with clean Protocol-based abstractions and a thoughtfully constructed observation space that maps directly to `TensorSchema` (27 features). The heuristic policy demonstrates sound domain knowledge with anti-thrashing mechanisms and adaptive blueprint selection via penalty tracking.

---

## 2. Key Features & Responsibilities

### SignalTracker (tracker.py)
- **Running statistics computation**: Maintains sliding windows of loss/accuracy history (configurable `history_window`, default 10)
- **Plateau detection**: Tracks consecutive epochs without sufficient improvement (`plateau_threshold=0.5%`)
- **Delta computation**: Computes loss and accuracy deltas for trend analysis
- **Leyline integration**: Produces `TrainingSignals` and nested `TrainingMetrics` objects for downstream consumers

### TamiyoDecision (decisions.py)
- **Action encapsulation**: Wraps topology-specific action enums with target seed ID, reason, and confidence
- **Command conversion**: `to_command()` produces Leyline's `AdaptationCommand` with risk level assessment
- **Blueprint extraction**: `blueprint_id` property for germinate actions

### HeuristicTamiyo (heuristic.py)
- **Plateau-triggered germination**: Initiates seeds after configurable plateau epochs (default 3)
- **Stage-aware seed management**: Handles TRAINING/BLENDING/SHADOWING/PROBATIONARY stages with per-stage failure detection
- **Fossilization decision**: Commits seeds with positive total improvement at PROBATIONARY stage
- **Anti-thrashing embargo**: Cooldown period after culls to prevent oscillation
- **Blueprint penalty system**: Tracks culled blueprints and decays penalties over time

---

## 3. Notable Innovations

### Signal Aggregation Design
The `SignalTracker` produces a rich observation structure that includes:
- Nested `TrainingMetrics` for core training signals
- Seed-specific context (`seed_stage`, `seed_epochs_in_stage`, `seed_alpha`, `seed_improvement`)
- Slot availability for multi-seed scenarios
- Bounded history windows (last 5 values exposed to policy)

This structure maps cleanly to `TensorSchema` (27 features) via `TrainingSignals.to_fast()`, enabling zero-copy conversion to PPO data planes.

### Blueprint Penalty System
```python
# From heuristic.py lines 231-256
def _get_next_blueprint(self) -> str:
    """Get next blueprint, avoiding heavily penalized ones."""
    # Decay penalties
    for bp in list(self._blueprint_penalties.keys()):
        self._blueprint_penalties[bp] *= self.config.blueprint_penalty_decay
        if self._blueprint_penalties[bp] < 0.1:
            del self._blueprint_penalties[bp]

    # Find blueprint below penalty threshold
    for _ in range(len(blueprints)):
        blueprint_id = blueprints[self._blueprint_index % len(blueprints)]
        self._blueprint_index += 1
        penalty = self._blueprint_penalties.get(blueprint_id, 0.0)
        if penalty < self.config.blueprint_penalty_threshold:
            return blueprint_id

    # All penalized - pick lowest
    return min(blueprints, key=lambda bp: self._blueprint_penalties.get(bp, 0.0))
```

This implements a simple form of **Thompson Sampling-like exploration** without explicit Bayesian machinery:
- Failed blueprints receive additive penalties (`blueprint_penalty_on_cull=2.0`)
- Penalties decay exponentially (`blueprint_penalty_decay=0.5`)
- Selection avoids heavily penalized options but falls back to least-penalized when all are high

### Anti-Thrashing Embargo
The `embargo_epochs_after_cull` parameter (default 5) prevents the common RL failure mode of GERMINATE-CULL oscillation:
```python
epochs_since_cull = signals.metrics.epoch - self._last_cull_epoch
if epochs_since_cull < self.config.embargo_epochs_after_cull:
    return TamiyoDecision(action=Action.WAIT, reason=f"Embargo ...")
```

---

## 4. Complexity Analysis

### Overall Complexity Rating: **LOW-MEDIUM**

| Component | Complexity | Rationale |
|-----------|------------|-----------|
| SignalTracker | LOW | Stateless computation with bounded deques |
| TamiyoDecision | LOW | Data class with straightforward conversion logic |
| HeuristicTamiyo | MEDIUM | Stage-based state machine with multiple decision paths |

### Decision Logic Complexity

The heuristic policy follows a clear hierarchical decision structure:

```
decide()
  |
  +-- No live seeds? --> _decide_germination()
  |                          |-- Embargo active? --> WAIT
  |                          |-- Too early? --> WAIT
  |                          |-- Plateau detected? --> GERMINATE_<blueprint>
  |                          +-- Otherwise --> WAIT
  |
  +-- Has live seeds? --> _decide_seed_management()
                              |-- Per stage:
                              |   GERMINATED --> WAIT (auto-advance)
                              |   TRAINING --> cull check, else WAIT
                              |   BLENDING --> cull check, else WAIT
                              |   SHADOWING --> cull check, else WAIT
                              |   PROBATIONARY --> FOSSILIZE or CULL
                              +-- Default --> WAIT
```

**Total decision paths:** ~12 distinct outcomes
**Cyclomatic complexity:** Moderate (~8-10 for main decide methods)

---

## 5. DRL Specialist Assessment

### Policy Design Quality: **GOOD**

**Strengths:**
1. **Clean Protocol abstraction**: `TamiyoPolicy` Protocol enables drop-in replacement with learned policies
2. **Action space is well-bounded**: Dynamic action enum via `build_action_enum()` scales with blueprints (typically 7 actions for CNN topology)
3. **Observation space is RL-ready**: 27-dimensional `TensorSchema` with normalized features
4. **Reward shaping compatibility**: The decision structure aligns with the reward shaping in `simic.rewards`

**Weaknesses:**
1. **No explicit value function baseline**: Heuristic lacks value estimates, making reward attribution difficult
2. **Deterministic policy**: No exploration mechanism beyond blueprint rotation
3. **Missing uncertainty quantification**: Confidence scores are heuristic, not principled

### Signal-to-Action Mapping: **EXCELLENT**

The observation space captures the essential features for seed lifecycle decisions:

| Feature Group | Relevance to Decision |
|---------------|----------------------|
| `plateau_epochs` | Primary germination trigger |
| `seed_stage`, `seed_epochs_in_stage` | Stage-aware management |
| `seed_improvement` | Cull/fossilize threshold |
| `accuracy_delta`, `loss_delta` | Trend detection |
| `available_slots` | Multi-seed capacity |

**Key insight**: The heuristic essentially implements a hand-coded policy that maps `(plateau_epochs, seed_stage, improvement)` to actions. A learned policy should be able to discover more nuanced mappings.

### Heuristic Baseline Quality: **GOOD**

The heuristic demonstrates domain expertise:

1. **Plateau detection** is a reasonable proxy for "host is stuck, try adaptation"
2. **Epoch-gated culling** (`cull_after_epochs_without_improvement=5`) prevents premature abandonment
3. **Percentage-based cull threshold** (`cull_if_accuracy_drops_by=2.0%`) is scale-invariant
4. **Zero improvement threshold for fossilization** (`min_improvement_to_fossilize=0.0`) is conservative

**Potential improvements:**
- Consider gradient health signals for cull decisions
- Add confidence-weighted fossilization (higher improvement = higher confidence)

### Comparison to Learned Policy Potential: **HIGH POTENTIAL**

| Aspect | Heuristic | Learned Policy Opportunity |
|--------|-----------|---------------------------|
| Germination timing | Fixed plateau threshold | Adaptive threshold based on training dynamics |
| Blueprint selection | Round-robin with penalties | Contextual bandit per training state |
| Cull decision | Fixed thresholds | Per-stage learned thresholds |
| Fossilization | Binary improvement check | Value-aware commitment (opportunity cost) |

**Expected RL gains:**
- PPO/SAC should learn to germinate earlier when accuracy is naturally plateauing
- Contextual blueprint selection can exploit task-specific correlations
- Value function can estimate "is it worth continuing this seed?" better than fixed thresholds

---

## 6. PyTorch Specialist Assessment

### Integration with Neural Policy: **READY**

The existing `ActorCritic` network in `simic/networks.py` is designed for this observation space:
```python
# From networks.py line 348
def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
    # state_dim=27 matches TensorSchema
    # action_dim=7 matches CNN topology actions
```

**Integration points:**
1. `TrainingSignals.to_fast()` produces `FastTrainingSignals` (NamedTuple, zero GC pressure)
2. `FastTrainingSignals.to_vector()` produces `list[float]` for tensor conversion
3. `TamiyoPolicy` Protocol matches `HeuristicTamiyo.decide()` signature

### Feature Extraction Efficiency: **GOOD**

**Strengths:**
- `TrainingMetrics` uses `__slots__` for reduced memory footprint
- `FastTrainingSignals` is a NamedTuple (stack-allocated, immutable)
- History windows are bounded (`deque(maxlen=history_window)`)

**Concerns:**
- `to_fast()` pads history lists inline (creates temporary lists)
- No batch processing path for parallel environments

**Recommended optimization:**
```python
# Pre-allocate padding buffer
_ZERO_HIST_5 = (0.0, 0.0, 0.0, 0.0, 0.0)

def to_fast(self, ...) -> FastTrainingSignals:
    loss_hist = self.loss_history[-5:]
    if len(loss_hist) < 5:
        loss_hist = _ZERO_HIST_5[:5 - len(loss_hist)] + tuple(loss_hist)
    # ...
```

---

## 7. Risks & Technical Debt

### Current Risks

1. **Single-seed assumption**: `SignalTracker.update()` extracts seed info from `active_seeds[0]` only
   ```python
   # tracker.py line 95
   if active_seeds and active_seeds[0] is not None:
       first_seed = active_seeds[0]
   ```
   This will break for multi-slot configurations.

2. **Epoch-global signals**: The tracker conflates host training progress with seed-specific improvement, which can mislead the policy when seeds are in early stages.

3. **No rollback telemetry**: Unlike Kasmina's slot, Tamiyo doesn't emit telemetry for decision events, making debugging harder.

### Technical Debt

1. **Missing unit tests for SignalTracker**: No dedicated test file for tracker logic
2. **Hardcoded topology in PolicyNetwork**: `build_action_enum("cnn")` in networks.py line 65
3. **Decision history unbounded**: `_decisions_made` list grows without bound during long training runs

---

## 8. Opportunities for Improvement

### Short-term (Low effort, high impact)

1. **Add telemetry to HeuristicTamiyo**: Emit decision events for observability
2. **Bound decision history**: Use `deque(maxlen=1000)` for `_decisions_made`
3. **Add confidence calibration**: Use plateau severity for germination confidence

### Medium-term (Enables RL transition)

1. **Implement `LearnedTamiyo`**: Wrapper that uses ActorCritic for action selection
2. **Add entropy tracking**: Monitor policy entropy decay during RL training
3. **Batch signal conversion**: Support vectorized `to_fast()` for parallel rollouts

### Long-term (Architecture evolution)

1. **Hierarchical policy**: Separate high-level strategy (when to germinate) from low-level selection (which blueprint)
2. **Multi-seed signal aggregation**: Extend tracker for per-seed signal windows
3. **Model-based planning**: Use world model to simulate seed outcomes before commitment

---

## 9. Critical Issues

### Issue 1: Plateau threshold sensitivity

**Location:** `heuristic.py` line 121
```python
if signals.metrics.plateau_epochs >= self.config.plateau_epochs_to_germinate:
```

**Problem:** The `plateau_epochs` counter in `SignalTracker` uses `plateau_threshold=0.5%` improvement. For tasks with noisy accuracy (e.g., small validation sets), this can trigger premature germination due to natural variance.

**Recommendation:** Consider using a smoothed improvement metric:
```python
# Exponential moving average of accuracy_delta
smoothed_delta = 0.9 * self._prev_smoothed + 0.1 * accuracy_delta
```

### Issue 2: Blueprint penalty decay timing

**Location:** `heuristic.py` lines 236-239
```python
for bp in list(self._blueprint_penalties.keys()):
    self._blueprint_penalties[bp] *= self.config.blueprint_penalty_decay
    if self._blueprint_penalties[bp] < 0.1:
        del self._blueprint_penalties[bp]
```

**Problem:** Penalties decay on every `_get_next_blueprint()` call, not on epoch boundary. Multiple germination attempts in rapid succession will prematurely decay penalties.

**Recommendation:** Track last decay epoch and only decay once per epoch:
```python
if self._last_penalty_decay_epoch < current_epoch:
    # apply decay
    self._last_penalty_decay_epoch = current_epoch
```

---

## 10. Recommendations Summary

| Priority | Recommendation | Effort | Impact |
|----------|---------------|--------|--------|
| P0 | Fix multi-seed signal extraction in SignalTracker | Low | High |
| P0 | Fix blueprint penalty decay timing | Low | Medium |
| P1 | Add telemetry to HeuristicTamiyo decisions | Low | High |
| P1 | Bound decision history with deque | Low | Low |
| P2 | Implement LearnedTamiyo wrapper for ActorCritic | Medium | High |
| P2 | Add smoothed improvement metric for plateau detection | Medium | Medium |
| P3 | Batch signal conversion for parallel rollouts | Medium | Medium |
| P3 | Hierarchical policy architecture | High | High |

---

## Appendix: Code Quality Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| Total lines (excluding blanks/comments) | ~350 | Compact |
| Number of classes | 4 | Appropriate |
| Average method length | ~15 lines | Good |
| Test coverage | Unknown (no dedicated tests found) | Needs improvement |
| Cyclomatic complexity (max) | ~10 | Acceptable |
| Dependency depth | 2 (leyline, kasmina) | Good |

---

*Report generated by DRL/PyTorch SME analysis.*
