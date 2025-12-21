---
id: "tamiyo"
title: "Tamiyo - Strategic Controller"
aliases: ["gardener", "strategic-controller", "decision-maker"]
biological_role: "Gardener"
layer: "Control"
criticality: "Tier-1"
tech_stack:
  - "Python 3.11+"
  - "dataclasses"
  - "typing.Protocol"
primary_device: "cpu"
thread_safety: "unsafe"
owners: "core-team"
compliance_tags:
  - "CPU-Only"
  - "Stateful"
  - "Telemetry-Emitting"
schema_version: "1.0"
last_updated: "2025-12-14"
last_reviewed_commit: "db3b9c1"
---

# Tamiyo Bible

# 1. Prime Directive

**Role:** Tamiyo observes training signals and makes strategic decisions about seed lifecycle (GERMINATE, FOSSILIZE, CULL, WAIT), acting as the "gardener" that determines when to introduce, retain, or remove neural components from the host model.

**Anti-Scope:** Tamiyo does NOT execute lifecycle transitions directly - it produces `TamiyoDecision` objects that are converted to `AdaptationCommand` for Kasmina to execute. It does NOT manage slot mechanics, alpha blending, or gradient flow - those are Kasmina's responsibilities.

---

# 2. Interface Contract

## 2.1 Entry Points (Public API)

### `TamiyoPolicy` (Protocol)
> Abstract interface for strategic decision-making policies.

```python
class TamiyoPolicy(Protocol):
    def decide(self, signals: TrainingSignals, active_seeds: list[SeedState]) -> TamiyoDecision: ...
    def reset(self) -> None: ...
```

- **Invariants:**
  - `signals` must be a valid `TrainingSignals` from the current epoch
  - `active_seeds` may be empty (no active seed = germination candidate)
- **Implementations:** `HeuristicTamiyo` (rule-based), future learned policies

### `HeuristicTamiyo`
> Rule-based strategic controller with plateau detection, stabilization gating, and blueprint rotation.

**Constructor:**
- `__init__(config: HeuristicPolicyConfig | None = None, topology: str = "cnn")`

**Key Methods:**
- `decide(signals: TrainingSignals, active_seeds: list[SeedState]) -> TamiyoDecision` - Make strategic decision
- `reset() -> None` - Reset all internal state for new episode

**Properties:**
- `decisions: list[TamiyoDecision]` - History of all decisions made (read-only copy)

### `SignalTracker`
> Tracks training signals over time and computes derived metrics including stabilization detection.

**Constructor:**
- `SignalTracker(plateau_threshold_pct: float = 0.5, history_window: int = 10, ...)`

**Key Methods:**
- `update(epoch, global_step, train_loss, train_accuracy, val_loss, val_accuracy, active_seeds, available_slots) -> TrainingSignals` - Update tracker and return current signals
- `reset() -> None` - Reset tracker state

**Properties:**
- `is_stabilized: bool` - Whether host training has stabilized (latch - stays True once set)

### `TamiyoDecision`
> Immutable decision structure with action, target, reason, and confidence.

```python
@dataclass
class TamiyoDecision:
    action: IntEnum           # From build_action_enum(topology)
    target_seed_id: str | None = None
    reason: str = ""
    confidence: float = 1.0
```

**Key Methods:**
- `to_command() -> AdaptationCommand` - Convert to Leyline's canonical command format

**Computed Properties:**
- `blueprint_id: str | None` - Extracted from GERMINATE_* action names

## 2.2 Configuration Schema

```python
@dataclass
class HeuristicPolicyConfig:
    # Germination triggers
    plateau_epochs_to_germinate: int = 3      # Consecutive plateau epochs before germinate
    min_epochs_before_germinate: int = 5      # Minimum training epochs before first germinate

    # Culling thresholds
    cull_after_epochs_without_improvement: int = 5  # Min epochs in stage before cull eligible
    cull_if_accuracy_drops_by: float = 2.0          # % drop that triggers cull

    # Fossilization threshold
    min_improvement_to_fossilize: float = 0.0  # Counterfactual contribution threshold

    # Anti-thrashing
    embargo_epochs_after_cull: int = 5  # Cooldown after cull before new germination

    # Blueprint selection with penalty tracking
    blueprint_rotation: list[str] = ["conv_light", "conv_heavy", "attention", "norm", "depthwise"]
    blueprint_penalty_on_cull: float = 2.0    # Penalty applied when blueprint is pruned
    blueprint_penalty_decay: float = 0.5      # Per-epoch multiplicative decay
    blueprint_penalty_threshold: float = 3.0  # Skip blueprints above this penalty
```

### Stabilization Parameters (SignalTracker)

```python
# Global defaults (can be overridden per-tracker)
STABILIZATION_THRESHOLD = 0.03  # 3% relative improvement = "explosive growth"
STABILIZATION_EPOCHS = 3        # Consecutive stable epochs required

# Task-specific tuning:
# - CIFAR-10: 3% threshold, 3 epochs (standard)
# - TinyStories/LLMs: Consider 1% threshold (smaller relative improvements)
# - Set stabilization_epochs=0 to disable gating entirely
```

## 2.3 Events (Pub/Sub via Nissa)

### Emits
| Event | Trigger | Payload |
|-------|---------|---------|
| `TAMIYO_INITIATED` | Host stabilized (first time) | `{env_id, epoch, stable_count, stabilization_epochs, val_loss}` |

### Subscribes
*None - Tamiyo is a passive observer, not an event subscriber.*

---

# 3. Tensor Contracts

## 3.1 Input Data

Tamiyo operates on **pure Python data structures**, not tensors. All inputs come from:

| Source | Type | Description |
|--------|------|-------------|
| `TrainingSignals` | Leyline dataclass | Nested metrics, seed info, history |
| `TrainingMetrics` | Leyline dataclass | epoch, losses, accuracies, deltas |
| `SeedState` | Kasmina dataclass | Seed lifecycle state (via TYPE_CHECKING) |

## 3.2 Output Data

| Output | Type | Description |
|--------|------|-------------|
| `TamiyoDecision` | Tamiyo dataclass | Action + target + reason + confidence |
| `AdaptationCommand` | Leyline dataclass | Canonical command for Kasmina (via `to_command()`) |

## 3.3 Gradient Flow

```
N/A - Tamiyo is a CPU-only control module.
No tensors, no gradients, no GPU operations.
```

---

# 4. Operational Physics

## 4.1 State Machine

### SignalTracker Stabilization Latch

```
[UNSTABLE] --(relative_improvement < threshold for N epochs)--> [STABILIZED]
[STABILIZED] --(latch: stays forever)--> [STABILIZED]

Note: Latch intentionally never resets. Call tracker.reset() for full reset.
```

### HeuristicTamiyo Decision Flow

```
[DECIDE] --(no live seeds)--> [GERMINATION_CHECK]
[DECIDE] --(has live seeds)--> [SEED_MANAGEMENT]

[GERMINATION_CHECK] --(embargo active)--> WAIT
[GERMINATION_CHECK] --(not stabilized)--> WAIT
[GERMINATION_CHECK] --(too early)--> WAIT
[GERMINATION_CHECK] --(plateau detected)--> GERMINATE_<BLUEPRINT>
[GERMINATION_CHECK] --(progressing normally)--> WAIT

[SEED_MANAGEMENT] --(stage=GERMINATED)--> WAIT (auto-advance pending)
[SEED_MANAGEMENT] --(stage=TRAINING, failing)--> CULL
[SEED_MANAGEMENT] --(stage=TRAINING, ok)--> WAIT
[SEED_MANAGEMENT] --(stage=BLENDING, failing)--> CULL
[SEED_MANAGEMENT] --(stage=BLENDING, ok)--> WAIT
[SEED_MANAGEMENT] --(stage=HOLDING, no counterfactual)--> WAIT
[SEED_MANAGEMENT] --(stage=HOLDING, contribution > threshold)--> FOSSILIZE
[SEED_MANAGEMENT] --(stage=HOLDING, contribution <= threshold)--> CULL
```

## 4.2 Data Governance

### Authoritative (Source of Truth)
- `SignalTracker._is_stabilized`: Canonical stabilization state
- `SignalTracker._best_accuracy`: Best validation accuracy seen
- `HeuristicTamiyo._decisions_made`: Decision history

### Ephemeral (Cached/Temporary)
- `SignalTracker._loss_history`: Rolling window (maxlen=history_window)
- `SignalTracker._accuracy_history`: Rolling window (maxlen=history_window)
- `HeuristicTamiyo._blueprint_penalties`: Decays each epoch

### Read-Only (Consumed)
- `TrainingSignals`: From training loop via Tolaria
- `SeedState`: From Kasmina slots (read-only observation)

## 4.3 Concurrency Model

- **Thread Safety:** `unsafe` - Single-threaded operation assumed
- **Async Pattern:** Synchronous/blocking
- **GPU Streams:** N/A (CPU-only)
- **Synchronization:** None required

## 4.4 Memory Lifecycle

- **Allocation:** Deques created in `__post_init__`, lists in `__init__`
- **Retention:** Until `reset()` called or object destroyed
- **Cleanup:** Python GC handles cleanup
- **Peak Usage:** Negligible (~KB for decision history)

---

# 5. Dependencies

## 5.1 Upstream (Modules that call Tamiyo)

| Module | Interaction | Failure Impact |
|--------|-------------|----------------|
| `tolaria` | Calls `tracker.update()` and `policy.decide()` | Training continues without strategic decisions |
| `scripts/train.py` | Creates `HeuristicTamiyo` instance | No seed lifecycle management |

## 5.2 Downstream (Modules Tamiyo depends on)

| Module | Interaction | Failure Handling |
|--------|-------------|------------------|
| `leyline` | Uses `TrainingSignals`, `SeedStage`, `CommandType`, `AdaptationCommand` | **Fatal** - Core types required |
| `leyline.actions` | Uses `build_action_enum()` for dynamic action enum | **Fatal** - Cannot create decisions |
| `nissa` | Emits `TAMIYO_INITIATED` telemetry | **Graceful** - Continues without telemetry |
| `kasmina` | TYPE_CHECKING import for `SeedState` | **Graceful** - Runtime unaffected |

## 5.3 External Dependencies

| Package | Version | Purpose | Fallback |
|---------|---------|---------|----------|
| `collections.deque` | stdlib | Rolling history windows | None (required) |
| `dataclasses` | stdlib | Data structures | None (required) |
| `typing.Protocol` | stdlib | Interface definition | None (required) |

---

# 6. Esper Integration

## 6.1 Commandment Compliance

| # | Commandment | Status | Notes |
|---|-------------|--------|-------|
| 1 | Sensors match capabilities | N/A | Tamiyo doesn't define sensors |
| 2 | Complexity pays rent | ✅ | Minimal overhead, clear purpose |
| 3 | GPU-first iteration | N/A | CPU-only control logic |
| 4 | Progressive curriculum | N/A | Not applicable |
| 5 | Train Anything protocol | ✅ | Uses generic `TrainingSignals`, no host-specific code |
| 6 | Morphogenetic plane | ✅ | Produces `AdaptationCommand` for Kasmina |
| 7 | Governor prevents catastrophe | ⚠️ | Heuristic lacks ransomware protection (see Tribal Knowledge) |
| 8 | Hierarchical scaling | N/A | Future consideration |
| 9 | Frozen Core economy | N/A | Future consideration |

## 6.2 Biological Role

**Analogy:** Tamiyo is the **Gardener** - observes the garden (training signals), decides when to plant new seeds (germinate), when to make a seedling permanent (fossilize), and when to remove failing plants (cull).

**Responsibilities in the organism:**
- Observe host training dynamics (stabilization detection)
- Decide optimal timing for seed introduction
- Evaluate seed performance and make lifecycle decisions
- Prevent thrashing via embargo periods and blueprint penalties

**Interaction with other organs:**
- Receives signals from: `Tolaria` (training metrics), `Kasmina` (seed state via SeedState)
- Sends signals to: `Kasmina` (via AdaptationCommand), `Nissa` (telemetry events)

## 6.3 CLI Integration

| Command | Flags | Effect on Module |
|---------|-------|------------------|
| `esper ppo` | `--plateau-epochs N` | Configures `plateau_epochs_to_germinate` |
| `esper ppo` | `--min-germinate-epoch N` | Configures `min_epochs_before_germinate` |
| `esper ppo` | `--embargo-epochs N` | Configures `embargo_epochs_after_cull` |

---

# 7. Cross-References

## 7.1 Related Bibles

| Bible | Relationship | Integration Point |
|-------|--------------|-------------------|
| [leyline](leyline.md) | **Implements** | Uses `TrainingSignals`, `SeedStage`, `CommandType`, `AdaptationCommand` |
| [kasmina](kasmina.md) | **Feeds** | Produces `AdaptationCommand` for slot lifecycle |
| [tolaria](tolaria.md) | **Called by** | Training loop invokes `tracker.update()` and `policy.decide()` |
| [nissa](nissa.md) | **Emits to** | Publishes `TAMIYO_INITIATED` event |
| [simic](simic.md) | **Parallel to** | Both consume `TrainingSignals`; Simic uses for RL, Tamiyo for heuristics |

## 7.2 Key Source Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/esper/tamiyo/__init__.py` | Public exports | `TamiyoDecision`, `SignalTracker`, `TamiyoPolicy`, `HeuristicTamiyo` |
| `src/esper/tamiyo/decisions.py` | Decision structure | `TamiyoDecision`, `to_command()` |
| `src/esper/tamiyo/tracker.py` | Signal tracking | `SignalTracker`, stabilization detection |
| `src/esper/tamiyo/heuristic.py` | Heuristic policy | `HeuristicTamiyo`, `HeuristicPolicyConfig`, `TamiyoPolicy` |

## 7.3 Test Coverage

| Test File | Coverage | Critical Tests |
|-----------|----------|----------------|
| `tests/tamiyo/test_heuristic_decisions.py` | High | Germination, cull, fossilize decisions |
| `tests/tamiyo/test_tracker.py` | High | TAMIYO_INITIATED telemetry emission |
| `tests/test_stabilization_tracking.py` | High | Latch behavior, threshold boundaries |
| `tests/tamiyo/properties/test_tamiyo_properties.py` | **Property** | 14 property tests (see below) |

**Property Tests (Hypothesis):**
- `test_latch_monotonicity` - Once stabilized, stays stabilized
- `test_stabilization_requires_consecutive_epochs` - Exact epoch count
- `test_best_accuracy_invariant` - best >= current always
- `test_history_window_invariant` - Bounded deque length
- `test_plateau_count_non_negative` - Never negative
- `test_decide_is_deterministic` - Same inputs → same outputs
- `test_to_command_preserves_target` - Information preservation
- `test_germinate_preserves_blueprint` - Blueprint in command
- `test_penalty_decay_monotonic` - Penalties never increase
- `test_penalty_eventually_clears` - Penalties decay to zero
- `test_embargo_blocks_germination` - Enforces cooldown
- `TestTamiyoSequences` - Stateful machine testing
- `test_first_epoch_no_crash` - Regression: first epoch safety
- `test_zero_loss_no_division_error` - Regression: division safety

---

# 8. Tribal Knowledge

## 8.1 Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Single-slot only | Cannot manage multiple concurrent seeds | Future: Multi-slot Tamiyo |
| Heuristic-only | No learning from experience | Future: RL-based Tamiyo |
| No ransomware protection | Heuristic may fossilize seeds that hurt training | Check `total_improvement` in rewards.py |

## 8.2 Performance Cliffs

| Operation | Trigger | Symptom | Mitigation |
|-----------|---------|---------|------------|
| None | N/A | N/A | CPU-only module, negligible overhead |

## 8.3 Common Pitfalls

| Pitfall | Why It Happens | Correct Approach |
|---------|----------------|------------------|
| **Premature stabilization** | 3 stable epochs can occur during temporary plateau before breakthrough | [DRL Expert] Consider EMA of improvement rates or hysteresis; current threshold tuned for CIFAR-10 |
| **Stabilization latch too sticky** | Once `_is_stabilized=True`, never resets even after fossilization changes model dynamics | [DRL Expert] Call `tracker.reset()` if model fundamentally changes; latch is intentional to prevent re-locking |
| **Forgetting to reset HeuristicTamiyo** | State variables (_blueprint_index, _penalties, etc.) persist across episodes | [Python Expert] Always call `policy.reset()` at episode start; consider extracting state to nested dataclass |
| **Plateau detection gaming** | Mediocre seed providing 0.5%/epoch blocks new germination indefinitely | [DRL Expert] 0.5% threshold may be noise; consider dynamic thresholds based on training phase |
| **Stage baseline reset** | `improvement_since_stage_start` resets at each stage transition, hiding net improvement | [DRL Expert] Always check BOTH `improvement_since_stage_start` AND `total_improvement` for lifecycle decisions |
| **Missing ransomware check** | HeuristicTamiyo only checks `counterfactual_contribution > 0`, not `total_improvement` | [DRL Expert] The G5 gate in slot.py checks `total_improvement >= 1.0%`, but heuristic should also verify |
| **Mutable default trap** | Using `[]` or `{}` as default in dataclass shares across instances | [Python Expert] Always use `field(default_factory=list)` for mutable defaults |
| **Deque maxlen timing** | `deque(maxlen=N)` must be set at construction, but N comes from another field | [Python Expert] Use `__post_init__` to recreate deque with correct maxlen (tracker.py:71-75) |

## 8.4 Historical Context / Technical Debt

| Item | Reason It Exists | Future Plan |
|------|------------------|-------------|
| Stabilization gating | Prevents credit misattribution during explosive growth | Re-enabled per DRL expert review (2025-12) |
| Blueprint penalty decay | Per-epoch decay (~10 epoch persistence) vs per-decision (~4 decisions) | Tuned for CIFAR-10; may need adjustment |
| Dynamic action enum | Actions depend on registered blueprints | Consider compile-time enum generation |

## 8.5 Debugging Tips

- **Symptom:** Seeds never germinate
  - **Likely Cause:** Host not stabilized OR embargo active OR plateau not detected
  - **Diagnostic:** Check `tracker.is_stabilized`, `signals.metrics.plateau_epochs`, `signals.metrics.epoch`
  - **Fix:** Lower `stabilization_threshold`, reduce `stabilization_epochs`, or check plateau_threshold

- **Symptom:** Seeds germinate too early (during explosive growth)
  - **Likely Cause:** Stabilization threshold too low for task
  - **Diagnostic:** Log relative improvements: `(prev_loss - curr_loss) / prev_loss`
  - **Fix:** Increase `stabilization_threshold` (e.g., 5% for volatile tasks)

- **Symptom:** Same blueprint keeps getting selected then pruned
  - **Likely Cause:** Blueprint penalty decays too fast (0.5^N per epoch)
  - **Diagnostic:** Check `_blueprint_penalties` dict values
  - **Fix:** Increase `blueprint_penalty_on_cull` or decrease `blueprint_penalty_decay`

- **Symptom:** FOSSILIZE decisions never pass
  - **Likely Cause:** Counterfactual contribution not available (alpha=0 during TRAINING)
  - **Diagnostic:** Check `seed.metrics.counterfactual_contribution` is not None
  - **Fix:** Wait for BLENDING stage when counterfactual validation runs

## 8.6 Expert Insights

### DRL Expert Analysis (Credit Assignment)

> **Stabilization gating is fundamentally sound** but has edge cases: (1) threshold sensitivity to task dynamics - 3% works for CIFAR-10 but LLMs may need 1%, (2) premature stabilization - 3 epochs is short, consider EMA or hysteresis, (3) sticky latch - correct for preventing re-locking but loses validity after fossilization.

> **Counterfactual requirement for FOSSILIZE is the gold standard** for causal attribution in RL. The code correctly blocks fossilization until counterfactual is available and uses a minimum threshold.

> **Temporal credit assignment gap:** Single-step deltas cannot distinguish seed contribution vs host improvement vs noise. The `improvement_since_stage_start` metric partially addresses this but resets at stage transitions.

> **"Ransomware seed" risk:** A seed can create structural dependencies (host adapts to it), show high counterfactual contribution (removing hurts because host depends on it), yet provide no real value. HeuristicTamiyo checks only counterfactual > 0, not total_improvement. The G5 gate has this check, but heuristic should too.

### Python Expert Analysis (Patterns)

> **Protocol usage is excellent.** Consider adding `@runtime_checkable` for isinstance() checks if needed.

> **Dataclass patterns are idiomatic.** The `__post_init__` pattern for deque maxlen is necessary and correct - can't use default_factory because maxlen depends on history_window.

> **State management coupling risk:** HeuristicTamiyo has 6 state variables that must all be reset in `reset()`. Consider extracting to nested `@dataclass` for single-point-of-truth reset.

> **TYPE_CHECKING import is correctly used** to break circular dependency with Kasmina. At runtime, `SeedState` is only needed for type hints.

---

# 9. Changelog

| Date | Change | Commit | Impact |
|------|--------|--------|--------|
| 2025-12-14 | Initial bible creation | `db3b9c1` | Documentation only |
| 2025-12-13 | Stabilization gating re-enabled | Prior | Prevents germination during explosive growth |
| 2025-12-10 | Blueprint penalty decay per-epoch | Prior | Penalties persist ~10 epochs instead of ~4 decisions |
