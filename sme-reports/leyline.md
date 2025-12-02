# SME Report: esper.leyline Package

**Package Path:** `/home/john/esper-lite/src/esper/leyline/`
**Version:** 0.2.0
**Report Date:** 2025-12-02
**Analyst:** Deep RL / PyTorch SME

---

## 1. Executive Summary

The `esper.leyline` package provides a well-structured data contracts layer for a morphogenetic neural network training system, defining seed lifecycle stages, training signals, actions, and telemetry events. The design demonstrates strong RL-awareness with a two-tier signal architecture (FastTrainingSignals vs TrainingSignals) that separates hot-path inference needs from rich contextual data. Overall, this is a solid foundation for policy learning with appropriate abstractions for both vectorized training and human-readable debugging.

---

## 2. Key Features & Responsibilities

### Core Modules

| Module | Responsibility |
|--------|---------------|
| `stages.py` | Seed lifecycle state machine (11 stages, valid transitions, helper predicates) |
| `signals.py` | Training observations - two-tier design with FastTrainingSignals (NamedTuple) and TrainingSignals (dataclass) |
| `actions.py` | Action space definitions - both static Action enum and dynamic `build_action_enum()` for topology-specific actions |
| `schemas.py` | Command contracts (AdaptationCommand), gate results, blueprint specifications |
| `reports.py` | Structured reports (SeedMetrics, SeedStateReport, FieldReport) |
| `telemetry.py` | Telemetry event types, performance budgets, SeedTelemetry snapshots |

### Key Responsibilities

1. **State Machine Definition**: Defines the seed lifecycle as a trust escalation model (DORMANT -> FOSSILIZED) with failure paths
2. **Observation Contracts**: Provides tensorizable signal representations for RL policy networks
3. **Action Space Management**: Dynamic action enum construction from registered blueprints
4. **Inter-component Contracts**: Standardizes communication between Tamiyo (controller), Kasmina (host), Simic (RL trainer), and Nissa (observability)

---

## 3. Notable Innovations

### Two-Tier Signal Architecture

```python
# Hot path: Zero GC pressure, immutable, stack-allocated
class FastTrainingSignals(NamedTuple):
    epoch: int
    global_step: int
    # ... 19 total fields

# Rich context: Full history, timestamps, GPU metrics
@dataclass
class TrainingSignals:
    metrics: TrainingMetrics
    loss_history: list[float]  # Variable length
    timestamp: datetime
```

This separation is excellent for RL training where the policy network needs fast, fixed-size inputs while debugging/logging benefits from richer data.

### TensorSchema Enum Pattern

```python
class TensorSchema(IntEnum):
    EPOCH = 0
    GLOBAL_STEP = 1
    # ... enables symbolic indexing into state vectors
```

Using IntEnum for tensor indices allows compile-time validation and symbolic access without string dictionary lookups. This is a best practice for RL observation spaces.

### Explicit State Machine with Predicates

```python
VALID_TRANSITIONS: dict[SeedStage, tuple[SeedStage, ...]] = {...}

def is_valid_transition(from_stage, to_stage) -> bool
def is_terminal_stage(stage) -> bool
def is_active_stage(stage) -> bool
def is_failure_stage(stage) -> bool
```

This makes the state machine explicit and queryable, enabling both validation and policy conditioning.

---

## 4. Complexity Analysis

### Overall Complexity Rating: **LOW-MEDIUM**

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Cognitive Load | LOW | Clear separation of concerns, well-documented enums |
| Structural Complexity | LOW | Flat module structure, minimal nesting |
| Type Complexity | MEDIUM | Mix of dataclasses, NamedTuples, Protocols, Enums |
| Runtime Complexity | LOW | Mostly static definitions, few dynamic computations |

### Coupling Assessment

**Inbound Coupling (High):** 192 files import from leyline - this is expected for a contracts layer.

**Outbound Coupling (Minimal):**
- `actions.py` imports from `esper.kasmina.blueprints.BlueprintRegistry` (deferred/lazy)
- `reports.py` imports from `signals.py` (internal)
- `schemas.py` imports from `stages.py` (internal)

**Assessment:** The coupling profile is appropriate. Leyline serves as the shared vocabulary and correctly has high inbound but minimal outbound dependencies.

---

## 5. DRL Specialist Assessment

### How Well Does This Support RL Training?

**Rating: STRONG**

The design shows clear understanding of RL requirements:

1. **Fixed-Size Observation Vector**: `FastTrainingSignals.to_vector()` produces a deterministic 27-element list
2. **Discrete Action Space**: IntEnum-based actions map cleanly to policy network outputs
3. **Episode Boundaries**: `FieldReport` captures germination-to-terminal lifecycle data for experience collection
4. **Reward Shaping Compatibility**: Signals include `accuracy_delta`, `loss_delta`, `plateau_epochs` - useful for potential-based shaping

### Signal Representations for Policy Learning

**Strengths:**
- History buffers (5-element loss/accuracy history) enable temporal pattern recognition
- Seed state features (`has_active_seed`, `seed_stage`, `seed_alpha`) provide full context
- `SeedTelemetry.to_features()` returns normalized [0,1] features ready for neural network input

**Potential Issues:**
- History padding uses `insert(0, 0.0)` which is O(n) - consider using deque or right-padding
- `best_val_loss` clamped to 10.0 in `to_fast()` - may lose information for high-loss scenarios
- No explicit feature normalization in `FastTrainingSignals.to_vector()` (epoch, global_step are unbounded)

### Action Space Design

**Current Design:**
```python
class Action(IntEnum):
    WAIT = 0
    GERMINATE_CONV = 1
    GERMINATE_ATTENTION = 2
    # ...
    FOSSILIZE = 5
    CULL = 6
```

**Assessment:**
- Action space is appropriately discrete and manageable (7 actions in legacy enum)
- Dynamic `build_action_enum()` allows topology-specific action spaces - good for generalization
- Clear semantic grouping (lifecycle actions, blueprint-specific germinate variants)

**Minor Issue:** The `SimicAction = Action` alias with comment `# deprecated alias` in `__init__.py` violates the codebase's no-legacy-code policy.

---

## 6. PyTorch Specialist Assessment

### Tensor-Friendly Data Structures?

**Rating: GOOD**

- `FastTrainingSignals.to_vector()` returns `list[float]` - trivially convertible to tensor
- `SeedTelemetry.to_features()` returns normalized `list[float]` with explicit dimension
- Fixed-size tuples for history (`tuple[float, float, float, float, float]`) ensure consistent batch dimensions

**Improvement Opportunity:**
```python
# Current: Returns list, requires torch.tensor() call
def to_vector(self) -> list[float]

# Could provide: Direct tensor output with device placement
def to_tensor(self, device: torch.device | None = None) -> torch.Tensor
```

### GPU-Native Considerations

**Current State:** The contracts layer is correctly device-agnostic. It defines data shapes but leaves tensor creation/placement to consumers.

**Assessment:** This is the right design - contracts should not have PyTorch dependencies. The Simic trainer should handle device placement.

### Memory Efficiency of Signals

| Structure | Memory Strategy | Assessment |
|-----------|----------------|------------|
| `FastTrainingSignals` | NamedTuple (immutable, no __dict__) | EXCELLENT |
| `TrainingMetrics` | `slots=True` | GOOD |
| `SeedMetrics` | `slots=True` | GOOD |
| `SeedTelemetry` | `slots=True` | GOOD |
| `TrainingSignals` | No slots (has list fields) | ACCEPTABLE |

The explicit use of `slots=True` and NamedTuple for high-frequency objects demonstrates memory-conscious design.

**Quantitative Estimate:**
- `FastTrainingSignals`: ~200 bytes (19 scalars + 2 5-tuples)
- `TrainingSignals`: ~400-600 bytes (depends on history length)
- Per-epoch overhead for signals: negligible (<1KB)

---

## 7. Risks & Technical Debt

### Identified Risks

| Risk | Severity | Description |
|------|----------|-------------|
| Deprecated Alias | LOW | `SimicAction = Action` alias violates no-legacy-code policy |
| Unbounded Features | MEDIUM | `epoch`, `global_step` in state vector are unbounded - may cause policy network issues |
| History Padding | LOW | O(n) insertion for history padding in `to_fast()` |
| Float Precision | LOW | Using float('inf') for `best_val_loss` default may cause issues in some serialization |

### Technical Debt Items

1. **Legacy Action Enum**: The static `Action` class duplicates functionality of `build_action_enum()` - should consolidate
2. **Schema Version Coupling**: `TENSOR_SCHEMA_SIZE = 27` is hardcoded - schema evolution requires coordinated updates
3. **Datetime in Signals**: `TrainingSignals.timestamp` adds serialization overhead and is not used in policy

---

## 8. Opportunities for Improvement

### Short-Term (Low Effort)

1. **Remove SimicAction alias**: Delete `SimicAction = Action` line and update any remaining call sites
2. **Add feature normalization**: Normalize epoch/global_step in `to_vector()` or document expected ranges
3. **Right-pad history**: Change history padding to append zeros (O(1)) instead of prepending

### Medium-Term (Moderate Effort)

1. **Add tensor conversion method**:
   ```python
   def to_tensor(self, device: str = "cpu") -> "torch.Tensor":
       import torch
       return torch.tensor(self.to_vector(), dtype=torch.float32, device=device)
   ```

2. **Schema versioning**: Add `TENSOR_SCHEMA_VERSION` constant and migration helpers for observation space evolution

3. **Vectorized batch conversion**: Add classmethod for batch conversion:
   ```python
   @classmethod
   def batch_to_tensor(cls, signals: list["FastTrainingSignals"]) -> "torch.Tensor":
       # Returns (batch_size, 27) tensor
   ```

### Long-Term (Architecture Level)

1. **Protocol-based signals**: Consider defining a `SignalProtocol` for extensibility
2. **Observation space registry**: Formalize observation space definitions for compatibility with Gym/Gymnasium
3. **Action masking support**: Add infrastructure for invalid action masking (e.g., can't FOSSILIZE if no active seed)

---

## 9. Critical Issues

### None Identified

The leyline package is well-designed and has no critical issues that would block RL training or cause correctness problems.

### Minor Issue Worth Noting

**File:** `/home/john/esper-lite/src/esper/leyline/actions.py` (lines 87-88)

```python
SimicAction = Action
```

And in `__init__.py` (line 76):
```python
"SimicAction",  # deprecated alias
```

This violates the codebase's strict no-legacy-code policy. The alias should be deleted and call sites updated.

---

## 10. Recommendations Summary

| Priority | Recommendation | Effort | Impact |
|----------|---------------|--------|--------|
| HIGH | Remove `SimicAction` alias (policy violation) | Low | Code quality |
| HIGH | Normalize unbounded features in state vector | Low | Training stability |
| MEDIUM | Add `to_tensor()` method for GPU-native workflows | Low | Developer ergonomics |
| MEDIUM | Add schema versioning mechanism | Medium | Maintainability |
| LOW | Optimize history padding (O(1) vs O(n)) | Low | Performance |
| LOW | Add batch tensor conversion | Medium | Training throughput |

### Final Assessment

The `esper.leyline` package is a well-designed contracts layer that demonstrates strong understanding of both RL training requirements and PyTorch performance considerations. The two-tier signal architecture is particularly noteworthy. The main improvement opportunities are around feature normalization, tensor ergonomics, and removing the deprecated alias.

**Quality Score: 8/10** - Production-ready with minor improvements recommended.
