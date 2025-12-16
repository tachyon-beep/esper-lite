# Simic Utility Files Audit Report

**Date:** 2025-12-16
**Auditor:** PyTorch Engineering Specialist
**Files Reviewed:**
- `/home/john/esper-lite/src/esper/simic/slots.py`
- `/home/john/esper-lite/src/esper/simic/reward_telemetry.py`
- `/home/john/esper-lite/src/esper/simic/debug_telemetry.py`
- `/home/john/esper-lite/src/esper/simic/config.py`
- `/home/john/esper-lite/src/esper/simic/telemetry_config.py`
- `/home/john/esper-lite/src/esper/simic/__init__.py`

---

## Executive Summary

The utility files in the Simic package are generally well-designed and follow good PyTorch practices. The most significant finding is a **name collision risk** between `simic.telemetry_config.TelemetryConfig` and `nissa.config.TelemetryConfig`. The debug telemetry module has some GPU synchronization patterns that are appropriate for debug-only code but should remain guarded. No critical PyTorch issues were found.

**Overall Assessment:** Clean utility modules with minor integration risks.

---

## File-by-File Analysis

### 1. slots.py

**Purpose:** Canonical slot ordering for deterministic feature/mask construction.

**Code Quality:** Excellent - minimal, focused, well-documented.

```python
CANONICAL_SLOTS: tuple[str, ...] = ("early", "mid", "late")

def ordered_slots(enabled_slots: Iterable[str]) -> tuple[str, ...]:
    enabled_set = set(enabled_slots)
    return tuple(slot for slot in CANONICAL_SLOTS if slot in enabled_set)
```

#### Findings

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| S-01 | Info | Code Quality | Pure Python utility, no PyTorch involvement. Well-tested. |

**Integration Usage:**
- `training.py`: Uses `ordered_slots` for slot ordering
- `ppo.py`: Uses `CANONICAL_SLOTS` and `ordered_slots`
- `vectorized.py`: Uses `ordered_slots`
- `action_masks.py`: Uses `ordered_slots`

**Verdict:** No issues. This module is a model of focused utility design.

---

### 2. reward_telemetry.py

**Purpose:** Dataclass for capturing reward component breakdown for debugging.

**Code Quality:** Excellent - clean dataclass with explicit `to_dict()` for performance.

```python
@dataclass(slots=True)
class RewardComponentsTelemetry:
    # 20+ fields capturing all reward components

    def to_dict(self) -> dict:
        # Explicit dict construction instead of asdict() for 3-5x performance
        return { ... }
```

#### Findings

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| RT-01 | Info | Performance | Explicit `to_dict()` vs `asdict()` is correct - avoids recursive copy overhead in hot path. |
| RT-02 | Low | Maintainability | `to_dict()` must be kept manually in sync with dataclass fields. |
| RT-03 | Info | Type Safety | Uses `slots=True` for memory efficiency and attribute typo detection. |

**RT-02 Detail:**
If a new field is added to `RewardComponentsTelemetry` but not to `to_dict()`, it will be silently omitted from telemetry output. The docstring mentions this is a deliberate tradeoff for performance.

**Recommendation:** Add a test that verifies `to_dict()` returns all dataclass fields:
```python
def test_to_dict_includes_all_fields():
    from dataclasses import fields
    telemetry = RewardComponentsTelemetry()
    dict_keys = set(telemetry.to_dict().keys())
    field_names = {f.name for f in fields(RewardComponentsTelemetry)}
    assert dict_keys == field_names
```

**Verdict:** Minor maintainability concern, otherwise excellent.

---

### 3. debug_telemetry.py

**Purpose:** Expensive debug-level telemetry for gradient analysis and numerical stability.

**Code Quality:** Good - appropriately batched GPU operations with documented performance costs.

#### PyTorch-Specific Findings

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| DT-01 | Low | GPU Sync | `check_numerical_stability` calls `.any()` per parameter (acceptable for debug code). |
| DT-02 | Info | torch.compile | Debug telemetry functions are not compiled (correct - dynamic control flow). |
| DT-03 | Low | Device Safety | Assumes all parameters on same device for `torch.stack()`. |
| DT-04 | Info | Memory | Uses `.detach()` correctly to avoid holding gradient computation graph. |

**DT-01 Detail: Per-Parameter GPU Sync**
```python
for name, param in model.named_parameters():
    if torch.isnan(param.data).any():  # GPU sync per param
        nan_weights.append(name)
```

This triggers one GPU synchronization per parameter. For a model with 100+ parameters, this could be 100+ syncs. However, the function is documented as debug-only and called only on anomaly detection.

**Recommendation:** Consider batching NaN/Inf detection:
```python
# More efficient for many parameters:
all_params = torch.cat([p.data.flatten() for p in model.parameters()])
has_nan = torch.isnan(all_params).any()
has_inf = torch.isinf(all_params).any()
```

However, this loses per-parameter identification. Current implementation is acceptable for debug use.

**DT-03 Detail: Device Safety**
```python
all_stats = torch.stack(stat_tensors).tolist()  # Assumes same device
```

If model parameters span devices (e.g., model parallelism), `torch.stack()` will fail. This is unlikely for current use but should be documented.

**DT-04 Detail: Correct Detach Usage**
```python
grad = param.grad.detach()
```
Properly detaches to avoid holding computation graph references.

#### RatioExplosionDiagnostic Analysis

```python
@classmethod
def from_batch(cls, ratio, old_log_probs, new_log_probs, actions, ...):
    bad_mask = (ratio > max_threshold) | (ratio < min_threshold)
    bad_indices = bad_mask.nonzero(as_tuple=True)[0].tolist()
```

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| DT-05 | Info | Performance | `.tolist()` triggers GPU sync - appropriate for diagnostic output. |
| DT-06 | Info | Unused Params | `states` and `action_masks` parameters are reserved but unused. |

**DT-06 Detail:** The `states` and `action_masks` parameters are documented as "reserved for future" but this could be considered dead code under the No Legacy Code Policy. However, they are explicitly documented placeholders for planned functionality, not backwards compatibility.

**Verdict:** Appropriate for debug-only use. Document device assumptions.

---

### 4. config.py (TrainingConfig)

**Purpose:** Strict, JSON-loadable hyperparameter configuration with validation.

**Code Quality:** Excellent - rigorous validation, clear presets, proper serialization.

#### Findings

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| C-01 | Info | Design | Unknown key rejection prevents "paper surface" drift. |
| C-02 | Low | Validation | `chunk_length != max_epochs` validation may be overly restrictive. |
| C-03 | Info | Integration | `to_train_kwargs()` tested against `train_ppo_vectorized` signature. |
| C-04 | Info | Type Coercion | Handles string-to-enum conversion in `__post_init__`. |

**C-02 Detail: Chunk Length Restriction**
```python
if self.chunk_length != self.max_epochs:
    raise ValueError(
        "chunk_length must match max_epochs for the current training loop"
    )
```

This validation is appropriate for the current LSTM-based implementation where truncated BPTT must align with episode boundaries. If the training loop is modified to support variable chunk lengths, this validation must be updated.

**C-04 Detail: Safe Enum Conversion**
```python
def __post_init__(self):
    if isinstance(self.reward_family, str):
        self.reward_family = RewardFamily(self.reward_family)
```

This is appropriate for JSON deserialization where enums arrive as strings.

#### Preset Analysis

| Preset | Entropy | Plateau Threshold | Notes |
|--------|---------|-------------------|-------|
| Default | 0.1 (leyline) | 0.5 | Baseline |
| CIFAR-10 | 0.1 | 0.4 | Lower plateau detection |
| CIFAR-10 Deep | 0.08 | 0.35 | Less exploration for deeper models |
| TinyStories | 0.05 | 0.3 | Conservative for language modeling |

Presets are well-reasoned for their target domains.

**Verdict:** Excellent configuration design with proper validation.

---

### 5. telemetry_config.py (TelemetryConfig - Simic)

**Purpose:** Telemetry verbosity control with automatic escalation on anomaly detection.

**Code Quality:** Good - clear state machine for escalation.

#### Findings

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| TC-01 | **Medium** | **Integration Risk** | Name collision with `nissa.config.TelemetryConfig`. |
| TC-02 | Low | State Management | `_escalation_epochs_remaining` is mutable state in dataclass. |
| TC-03 | Info | Design | `should_collect()` API is clean and testable. |

**TC-01 Detail: Name Collision Risk**

Two classes named `TelemetryConfig` exist:
1. `esper.simic.telemetry_config.TelemetryConfig` - IntEnum-based verbosity levels
2. `esper.nissa.config.TelemetryConfig` - Pydantic model with rich gradient/loss config

**Current usage:**
```python
# train.py
from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel

# nissa/tracker.py
from esper.nissa.config import TelemetryConfig
```

Both are re-exported from their package `__init__.py`:
```python
# simic/__init__.py
from esper.simic.telemetry_config import TelemetryLevel, TelemetryConfig

# nissa/__init__.py
from esper.nissa.config import TelemetryConfig
```

**Risk:** If a developer imports both packages or uses bare `TelemetryConfig`, they may get the wrong class. The Simic version is a simple dataclass; the Nissa version is a Pydantic model with profile support.

**Recommendation:** Rename one of the classes to disambiguate:
- `simic.telemetry_config.TelemetryConfig` -> `SimicTelemetryConfig` or `TelemetryLevelConfig`
- `nissa.config.TelemetryConfig` -> `NissaTelemetryConfig` or `DiagnosticTelemetryConfig`

**TC-02 Detail: Mutable State**
```python
@dataclass
class TelemetryConfig:
    _escalation_epochs_remaining: int = field(default=0, repr=False)

    def escalate_temporarily(self, epochs: int | None = None) -> None:
        self._escalation_epochs_remaining = epochs
```

Using mutable state in a dataclass is appropriate here - the config is intentionally stateful to track escalation. The underscore prefix and `repr=False` correctly signal internal state.

**Verdict:** Name collision should be addressed before it causes confusion.

---

### 6. __init__.py

**Purpose:** Package exports with deferred imports for heavy modules.

**Code Quality:** Excellent - clear organization with lazy import comments.

#### Findings

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| I-01 | Info | Import Strategy | Heavy modules (ppo, vectorized, training) are imported on-demand. |
| I-02 | Info | Export Completeness | All public APIs are in `__all__`. |
| I-03 | Low | Documentation | Module docstring mentions 3 heavy modules but doesn't list all exports. |

**I-01 Detail: Deferred Imports**
```python
# NOTE: Heavy modules imported on demand:
#   from esper.simic.ppo import PPOAgent
#   from esper.simic.vectorized import train_ppo_vectorized
#   from esper.simic.training import train_heuristic
```

This is correct - these modules import PyTorch and should not be loaded until needed. CLI tools that only need `TrainingConfig` won't pay the import cost of the full training infrastructure.

**Export Organization:**
The exports are well-organized by category:
- Normalization
- Rewards (7 exports)
- Features (2 exports)
- Action Masks (2 exports)
- Telemetry (10 exports)

**Verdict:** Well-structured package initialization.

---

## Cross-Cutting Concerns

### torch.compile Compatibility

| Module | Compile Compatible | Notes |
|--------|-------------------|-------|
| slots.py | N/A | Pure Python, no torch |
| reward_telemetry.py | N/A | Pure Python dataclass |
| debug_telemetry.py | No (by design) | Dynamic control flow, debug-only |
| config.py | N/A | Pure Python configuration |
| telemetry_config.py | N/A | Pure Python configuration |

Debug telemetry is correctly excluded from compilation - it uses dynamic control flow and is only called on anomalies.

### Device Placement

| Module | Device Handling | Assessment |
|--------|-----------------|------------|
| debug_telemetry.py | Assumes single device | Appropriate for current use |

### Gradient Flow

Only `debug_telemetry.py` interacts with gradients, and it correctly uses `.detach()` to avoid graph retention.

### Memory Management

| Module | Memory Considerations |
|--------|----------------------|
| reward_telemetry.py | Uses `slots=True` for ~40% memory reduction |
| debug_telemetry.py | Uses `slots=True`, properly detaches tensors |

---

## Summary of Issues

| ID | Severity | File | Description | Recommendation |
|----|----------|------|-------------|----------------|
| TC-01 | **Medium** | telemetry_config.py | Name collision with nissa.TelemetryConfig | Rename one class |
| RT-02 | Low | reward_telemetry.py | to_dict() sync with fields | Add sync test |
| DT-01 | Low | debug_telemetry.py | Per-param GPU sync | Acceptable for debug |
| DT-03 | Low | debug_telemetry.py | Assumes single device | Document assumption |
| C-02 | Low | config.py | chunk_length restriction | Document reason |
| DT-06 | Info | debug_telemetry.py | Unused reserved params | Acceptable as documented |

---

## Recommendations

### Priority 1: Address Name Collision (TC-01)

Rename `esper.simic.telemetry_config.TelemetryConfig` to `SimicTelemetryConfig` or similar to prevent confusion with `esper.nissa.config.TelemetryConfig`. Update all import sites.

### Priority 2: Add Field Sync Test (RT-02)

Add a test to verify `RewardComponentsTelemetry.to_dict()` includes all dataclass fields:

```python
def test_to_dict_includes_all_fields():
    from dataclasses import fields
    telemetry = RewardComponentsTelemetry()
    dict_keys = set(telemetry.to_dict().keys())
    field_names = {f.name for f in fields(RewardComponentsTelemetry)}
    assert dict_keys == field_names
```

### Priority 3: Document Assumptions

Add docstring notes to `debug_telemetry.py` functions clarifying:
- Single-device assumption for `torch.stack()` operations
- Expected performance cost (explicit GPU sync counts)

---

## Test Coverage Assessment

| Module | Test File | Coverage |
|--------|-----------|----------|
| slots.py | test_slots.py | Good - canonical order, unknown slot handling |
| reward_telemetry.py | test_reward_telemetry.py | Good - integration with rewards |
| debug_telemetry.py | test_debug_telemetry.py | Good - healthy model, NaN detection |
| config.py | test_config.py | Good - kwargs sync, defaults |
| telemetry_config.py | test_telemetry_config.py | Good - levels, escalation |

All utility modules have dedicated test files with reasonable coverage.

---

## Conclusion

The Simic utility files are well-designed with appropriate PyTorch practices. The main actionable finding is the `TelemetryConfig` name collision which should be resolved to prevent future confusion. Debug telemetry appropriately trades performance for diagnostic detail and is correctly guarded behind anomaly detection.
