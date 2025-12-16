# Simic Audit: anomaly_detector.py

**File:** `/home/john/esper-lite/src/esper/simic/anomaly_detector.py`
**Date:** 2025-12-16
**Auditor:** Claude (PyTorch Engineering Specialist)

---

## Executive Summary

The `anomaly_detector.py` module is a **pure Python utility class** with **no PyTorch dependencies**. It provides threshold-based anomaly detection for PPO training metrics (ratio explosion/collapse, value function collapse, numerical instability). The implementation is clean, well-tested, and correctly integrated into the vectorized training loop.

**Overall Assessment:** LOW RISK - This is a well-designed utility module with minimal complexity and no PyTorch-specific concerns.

---

## Module Overview

### Purpose
Detects training anomalies that should trigger:
1. Telemetry escalation to DEBUG level
2. Dense trace capture in Karn
3. Per-layer gradient diagnostics (when debug enabled)

### Architecture
```
anomaly_detector.py (simic)     -->  vectorized.py (simic)
        |                                   |
        v                                   v
  AnomalyReport                    _emit_anomaly_diagnostics()
                                           |
                                           v
                                   Karn triggers.py
                                   (separate AnomalyDetector)
```

**Key Insight:** There are **two different `AnomalyDetector` classes**:
1. `esper.simic.anomaly_detector.AnomalyDetector` - Phase-dependent PPO metric thresholds
2. `esper.karn.triggers.AnomalyDetector` - EMA-based epoch snapshot analysis

These serve complementary purposes but the naming collision warrants attention.

---

## Findings

### 1. No PyTorch Dependencies

| Category | Status | Notes |
|----------|--------|-------|
| torch.compile compatibility | N/A | Pure Python, no tensors |
| Device placement | N/A | No tensor operations |
| Gradient flow | N/A | No autograd involvement |
| Memory management | N/A | Lightweight dataclasses only |

**Severity:** N/A - Not applicable to this module.

---

### 2. Code Quality Analysis

#### 2.1 Clean Design (POSITIVE)

The module demonstrates excellent separation of concerns:

```python
@dataclass
class AnomalyReport:
    """Report of detected anomalies."""
    has_anomaly: bool = False
    anomaly_types: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)
```

- Uses `@dataclass(slots=True)` for memory efficiency
- Immutable threshold configuration via dataclass fields
- Clear method signatures with explicit parameters

**Severity:** N/A - This is a positive finding.

---

#### 2.2 Phase-Dependent Thresholds (POSITIVE)

The value collapse detection uses training-phase-aware thresholds:

```python
# Phase boundaries as fractions of total training
warmup_fraction: float = 0.10   # 0-10% of training
early_fraction: float = 0.25    # 10-25% of training
mid_fraction: float = 0.75      # 25-75% of training

# EV thresholds for each phase
ev_threshold_warmup: float = -0.5   # Allow anti-correlated (random init)
ev_threshold_early: float = -0.2    # Expect some learning
ev_threshold_mid: float = 0.0       # Expect positive correlation
ev_threshold_late: float = 0.1      # Expect useful predictions
```

This addresses a common PPO pitfall where early low explained variance is falsely flagged as problematic.

**Severity:** N/A - This is a positive finding.

---

#### 2.3 Strict Episode Context Requirement (POSITIVE)

```python
def check_value_function(
    self,
    explained_variance: float,
    current_episode: int = 0,
    total_episodes: int = 0,
) -> AnomalyReport:
    # ...
    if current_episode <= 0 or total_episodes <= 0:
        raise ValueError("current_episode and total_episodes are required (> 0)")
```

Per the codebase's No Legacy Code Policy, there are no fallback defaults or compatibility shims. Callers must provide episode context explicitly.

**Severity:** N/A - This aligns with project conventions.

---

### 3. Integration Analysis

#### 3.1 Vectorized Training Integration (CORRECT)

The detector is properly instantiated and used in `vectorized.py`:

```python
# Line 1050
anomaly_detector = AnomalyDetector()

# Lines 2251-2259
anomaly_report = anomaly_detector.check_all(
    ratio_max=metrics.get("ratio_max", 1.0),
    ratio_min=metrics.get("ratio_min", 1.0),
    explained_variance=metrics.get("explained_variance", 0.0),
    has_nan=has_nan,
    has_inf=has_inf,
    current_episode=batch_epoch_id,
    total_episodes=total_episodes,
)
```

**Observation:** The detector is instantiated once per training run and reused across batches. This is correct since the class is stateless.

**Severity:** N/A - Integration is correct.

---

#### 3.2 Naming Collision with Karn (LOW)

Two classes named `AnomalyDetector` exist:

| Module | Purpose | Stateful |
|--------|---------|----------|
| `esper.simic.anomaly_detector` | PPO metric thresholds | No |
| `esper.karn.triggers` | EMA-based epoch analysis | Yes (RollingStats) |

**Risk:** Import confusion. The Karn collector imports:
```python
from esper.karn.triggers import AnomalyDetector, PolicyAnomalyDetector
```

While simic's `__init__.py` exports:
```python
from esper.simic.anomaly_detector import AnomalyDetector, AnomalyReport
```

**Current Mitigation:** Full path imports prevent collision:
```python
# vectorized.py
from esper.simic.anomaly_detector import AnomalyDetector, AnomalyReport
```

**Severity:** LOW - Naming collision exists but is currently handled via qualified imports.

**Recommendation:** Consider renaming to `PPOAnomalyDetector` for clarity, though this is not urgent.

---

#### 3.3 Telemetry Escalation Flow (CORRECT)

```python
def _handle_telemetry_escalation(
    anomaly_report: AnomalyReport | None,
    telemetry_config: TelemetryConfig | None,
) -> None:
    if telemetry_config is None:
        return
    if anomaly_report is None or not anomaly_report.has_anomaly:
        return
    telemetry_config.escalate_temporarily()
```

The escalation correctly:
1. Guards against None configs
2. Only escalates when anomalies exist
3. Uses temporary escalation (auto-decays)

**Severity:** N/A - Integration is correct.

---

### 4. Test Coverage Analysis

The test file `/home/john/esper-lite/tests/simic/test_anomaly_detector.py` covers:

| Test Case | Coverage |
|-----------|----------|
| Ratio explosion detection | COVERED |
| Ratio collapse detection | COVERED |
| Healthy ratios (no anomaly) | COVERED |
| Value collapse in late training | COVERED |
| Low EV in warmup (expected) | COVERED |
| Threshold progression across phases | COVERED |
| Phase scaling with total_episodes | COVERED |
| Episode context requirement | COVERED |
| Numerical instability detection | COVERED |
| Combined check_all() | COVERED |

**Missing Test Cases:**

1. **Edge case: current_episode == total_episodes** (100% progress)
2. **Edge case: current_episode > total_episodes** (overflow handling)
3. **Multiple anomalies in single check_all()** (e.g., ratio + value + numerical)

**Severity:** LOW - Core functionality is well-tested, edge cases are minor.

---

### 5. Potential Improvements

#### 5.1 Type Annotations Enhancement (COSMETIC)

```python
# Current
def check_ratios(
    self,
    ratio_max: float,
    ratio_min: float,
) -> AnomalyReport:

# Suggested: Add Final for threshold constants in dataclass
from typing import Final

@dataclass
class AnomalyDetector:
    max_ratio_threshold: Final[float] = 5.0  # Immutable after init
```

**Severity:** COSMETIC - Would improve IDE support but not required.

---

#### 5.2 Consider Enum for Anomaly Types (COSMETIC)

```python
# Current: String literals
report.add_anomaly("ratio_explosion", detail)

# Alternative: Enum for type safety
class AnomalyType(str, Enum):
    RATIO_EXPLOSION = "ratio_explosion"
    RATIO_COLLAPSE = "ratio_collapse"
    VALUE_COLLAPSE = "value_collapse"
    NUMERICAL_INSTABILITY = "numerical_instability"
```

**Severity:** COSMETIC - Strings work fine, enum would add marginal type safety.

---

## Risk Assessment Summary

| Category | Risk Level | Notes |
|----------|------------|-------|
| torch.compile | N/A | No PyTorch code |
| Device placement | N/A | No tensors |
| Gradient flow | N/A | No autograd |
| Memory leaks | NONE | Stateless dataclass |
| Integration | NONE | Correctly integrated |
| Naming collision | LOW | Two `AnomalyDetector` classes |
| Test coverage | LOW | Edge cases missing |
| Code quality | EXCELLENT | Clean, well-documented |

---

## Conclusion

The `anomaly_detector.py` module is a **low-risk, high-quality utility** that requires no immediate action. The only notable concern is the naming collision with `esper.karn.triggers.AnomalyDetector`, which is currently mitigated by qualified imports but could cause confusion during maintenance.

**Recommended Actions:**
1. (Optional) Rename to `PPOAnomalyDetector` for disambiguation
2. (Optional) Add edge case tests for 100% and >100% progress
3. No urgent changes required

---

## Appendix: Related Files

| File | Relationship |
|------|--------------|
| `/home/john/esper-lite/src/esper/simic/vectorized.py` | Primary consumer |
| `/home/john/esper-lite/src/esper/karn/triggers.py` | Similar class (naming collision) |
| `/home/john/esper-lite/src/esper/karn/collector.py` | Imports Karn's AnomalyDetector |
| `/home/john/esper-lite/tests/simic/test_anomaly_detector.py` | Test suite |
