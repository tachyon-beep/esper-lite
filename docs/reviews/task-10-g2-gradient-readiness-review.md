# Task 10: G2 Gradient-Based Seed Readiness - Code Review

**Reviewer:** Claude Code Review Agent
**Date:** 2025-12-07
**Commit Range:** 03a02e6 → a30880d
**Plan Reference:** `/home/john/esper-lite/docs/plans/2025-12-07-kasmina-expert-improvements.md` (Task 10, lines 788-927)

---

## Executive Summary

**Assessment:** ⚠️ **PARTIAL IMPLEMENTATION - REQUIRES INTEGRATION**

The implementation correctly adds the `seed_gradient_norm_ratio` field to `SeedMetrics` and integrates gradient-based checking into the G2 gate. The tests are comprehensive and pass. However, **the metric is never populated from actual gradient data during training**, making it a non-functional gate in production. The field defaults to 0.0, which will cause all G2 checks to fail with "seed_gradient_low".

**Quality Score:** 6.5/10

**Status:** Implementation is structurally correct but functionally incomplete. Requires Task 10.1 to wire gradient data from `GradientIsolationMonitor` to `SeedMetrics`.

---

## 1. Plan Alignment Analysis

### What Was Planned

From `/home/john/esper-lite/docs/plans/2025-12-07-kasmina-expert-improvements.md` (Task 10):

1. Add `seed_gradient_norm_ratio` field to `SeedMetrics` dataclass
2. Update G2 gate (`_check_g2`) to check gradient ratio (minimum 5%)
3. Add tests in `tests/kasmina/test_g2_gradient_readiness.py`
4. Context: "The G2 gate currently uses global improvement which conflates host and seed contributions. Add a gradient-based metric that measures seed-specific learning."

### What Was Implemented

**Files Modified:**
- `/home/john/esper-lite/src/esper/kasmina/slot.py` (17 line changes)
  - Added `seed_gradient_norm_ratio: float = 0.0` to `SeedMetrics` (line 85)
  - Updated `_check_g2` to check gradient ratio with 5% minimum threshold (lines 407-415)
  - Updated docstring to mention gradient activity (line 377)

- `/home/john/esper-lite/tests/kasmina/test_g2_gradient_readiness.py` (84 new lines)
  - 3 comprehensive test cases covering pass/fail scenarios

### Deviation Analysis

✅ **Planned items completed:**
- Field added to SeedMetrics
- G2 gate updated with gradient check
- Tests added and passing

❌ **Critical missing item:**
- **No integration between `GradientIsolationMonitor` and `SeedMetrics`**
- The plan states the metric should be "seed_grad_norm / (host_grad_norm + eps)"
- `GradientIsolationMonitor.check_isolation()` already computes these values (lines 116-117 in isolation.py)
- **But nothing transfers these values to `state.metrics.seed_gradient_norm_ratio`**

This is a **problematic departure** - the implementation is incomplete and will not work in production.

---

## 2. Code Quality Assessment

### File: `/home/john/esper-lite/src/esper/kasmina/slot.py`

#### Strengths

**1. Field Definition (Line 85)**
```python
# Gradient-based seed activity metric
seed_gradient_norm_ratio: float = 0.0  # seed_grad_norm / (host_grad_norm + eps)
```
- Clear comment explaining the formula
- Appropriate default value
- Follows dataclass conventions

**2. G2 Gate Logic (Lines 407-417)**
```python
# NEW: Gradient-based seed activity check
# Ensures seed is actually learning, not just riding host improvements
min_gradient_ratio = 0.05  # Seed should have at least 5% of host gradient activity
if state.metrics.seed_gradient_norm_ratio >= min_gradient_ratio:
    checks_passed.append(f"seed_gradient_active_{state.metrics.seed_gradient_norm_ratio:.2f}")
    gradient_ok = True
else:
    checks_failed.append(f"seed_gradient_low_{state.metrics.seed_gradient_norm_ratio:.2f}")
    gradient_ok = False
```
- Excellent explanatory comments about *why* this check exists
- Magic number (0.05) has inline explanation
- Consistent with existing gate check patterns
- Good formatting of check messages for debugging

**3. Gate Conjunction Logic (Line 417)**
```python
passed = perf_ok and isolation_ok and seed_ok and gradient_ok
```
- Correctly adds new condition to existing gate logic
- All four conditions must pass (proper AND conjunction)

#### Issues

**CRITICAL: Field Never Populated**

**Location:** Entire implementation
**Severity:** Critical (blocks production use)

**Problem:** `seed_gradient_norm_ratio` is never set from actual gradient data.

**Evidence:**
```bash
$ grep -r "seed_gradient_norm_ratio.*=" src/
src/esper/kasmina/slot.py:85:    seed_gradient_norm_ratio: float = 0.0  # Default
```

Only the default initialization exists. The field should be updated during training when `GradientIsolationMonitor.check_isolation()` is called.

**Expected Integration Point:**

`GradientIsolationMonitor.check_isolation()` already computes the required values:
```python
# src/esper/kasmina/isolation.py:116-117
self.host_grad_norm = host_norm
self.seed_grad_norm = seed_norm
```

The ratio should be computed and stored:
```python
# MISSING CODE - should exist somewhere in training loop
ratio = seed_norm / (host_norm + 1e-8)  # eps for numerical stability
state.metrics.seed_gradient_norm_ratio = ratio
```

**Impact:**
- G2 gate will **always fail** the gradient check (0.0 < 0.05)
- Seeds can never progress to BLENDING stage
- Entire feature is non-functional

**Recommendation:**
Add Task 10.1 to wire gradient statistics from monitor to metrics. Likely integration points:
1. After `isolation_monitor.check_isolation()` call in training loop
2. In `SeedSlot` if it manages the monitor
3. In the training orchestration layer (simic/tolaria)

---

**IMPORTANT: Hardcoded Threshold**

**Location:** Line 409
**Severity:** Important (should fix)

```python
min_gradient_ratio = 0.05  # Seed should have at least 5% of host gradient activity
```

**Problem:** Magic number hardcoded in method, not configurable

**Existing Pattern:** Other thresholds are in `QualityGates.__init__`:
```python
# src/esper/kasmina/slot.py:325-330
self.min_training_improvement = 1.0
self.max_isolation_violations = 3
self.min_training_epochs = 3
```

**Recommendation:**
```python
# In QualityGates.__init__
self.min_gradient_ratio = 0.05  # Seed should have ≥5% of host gradient activity

# In _check_g2
if state.metrics.seed_gradient_norm_ratio >= self.min_gradient_ratio:
```

This maintains consistency with existing gate configuration patterns.

---

**SUGGESTION: Numerical Stability**

**Location:** Line 85 comment
**Severity:** Suggestion (nice to have)

The comment mentions "seed_grad_norm / (host_grad_norm + eps)" but doesn't specify eps value.

**Recommendation:**
Document expected epsilon value in comment or provide a helper method:
```python
@property
def seed_gradient_activity_ratio(self) -> float:
    """Compute seed gradient norm as ratio of host gradient norm.

    Uses eps=1e-8 for numerical stability when host gradients are near zero.
    """
    return self.seed_gradient_norm / (self.host_gradient_norm + 1e-8)
```

But this requires storing both raw norms, not just the ratio.

---

### File: `/home/john/esper-lite/tests/kasmina/test_g2_gradient_readiness.py`

#### Strengths

**1. Test Coverage**
- ✅ Low gradient ratio fails (test_g2_checks_seed_gradient_norm_ratio)
- ✅ Sufficient gradient ratio passes (test_g2_passes_with_sufficient_gradient_ratio)
- ✅ Low gradient fails despite global improvement (test_g2_fails_with_low_gradient_despite_global_improvement)

**2. Test Quality**

Each test follows proper structure:
```python
def test_g2_passes_with_sufficient_gradient_ratio(self):
    """G2 should pass when seed gradient ratio is sufficient."""
    gates = QualityGates()

    # Arrange - Create state with good metrics
    state = SeedState(...)
    state.metrics.seed_gradient_norm_ratio = 0.10  # Good seed activity

    # Act
    result = gates._check_g2(state)

    # Assert
    assert result.passed
    assert any("seed_gradient_active" in check for check in result.checks_passed)
```

**3. Test Documentation**
- Clear docstrings explaining what each test verifies
- Inline comments explain test data values
- Tests are self-documenting

**4. Test Independence**
- Each test creates its own `QualityGates` instance
- No shared state between tests
- Tests can run in any order

#### Issues

**IMPORTANT: Tests Use Manual Fixture Data**

**Location:** All tests
**Severity:** Important (should improve)

**Problem:** Tests manually set `seed_gradient_norm_ratio` instead of demonstrating integration:

```python
state.metrics.seed_gradient_norm_ratio = 0.01  # Manual fixture
```

**Missing Integration Test:**
```python
def test_g2_gradient_ratio_populated_from_training(self):
    """G2 should use gradient ratio computed during training."""
    # This test would fail because integration doesn't exist
    slot = SeedSlot("test", channels=32, device="cpu")
    slot.germinate("norm", "seed-1", host_module=CNNHost())

    # Simulate training step
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    output = slot(x)
    loss = output.sum()
    loss.backward()

    # Check isolation (this should populate the metric)
    is_isolated, stats = slot.isolation_monitor.check_isolation()

    # EXPECTED: ratio should now be populated
    assert slot.state.metrics.seed_gradient_norm_ratio > 0
```

This test would currently fail because the wiring doesn't exist.

**Recommendation:**
Once Task 10.1 completes the integration, add end-to-end test verifying the full flow.

---

**SUGGESTION: Test Boundary Conditions**

**Location:** Test suite
**Severity:** Suggestion (nice to have)

Missing edge case tests:
- Exactly at threshold (0.05)
- Division by zero protection (host_norm = 0)
- Negative values (should be impossible but worth testing)
- Very large ratios (seed_norm >> host_norm, possible during early training)

**Recommendation:**
```python
def test_g2_gradient_ratio_at_exact_threshold(self):
    """G2 should pass when gradient ratio exactly equals threshold."""
    state.metrics.seed_gradient_norm_ratio = 0.05  # Exactly at threshold
    result = gates._check_g2(state)
    assert result.passed  # >= means this should pass

def test_g2_handles_zero_host_gradient(self):
    """G2 should handle edge case of zero host gradients."""
    # If host_grad_norm is 0, ratio computation needs eps
    state.metrics.seed_gradient_norm_ratio = float('inf')  # What should happen?
    # Define expected behavior
```

---

## 3. Architecture and Design

### SOLID Principles

✅ **Single Responsibility:**
- `SeedMetrics` responsible for storing metrics (includes gradient ratio)
- `QualityGates._check_g2` responsible for gate logic
- `GradientIsolationMonitor` responsible for computing gradients (already exists)

⚠️ **Open/Closed:**
- Adding new metric and gate check required modifying existing code (expected)
- But the separation is good - `QualityGates` remains extensible

✅ **Liskov Substitution:** Not applicable (no inheritance)

✅ **Interface Segregation:** Not applicable (dataclass and functions)

⚠️ **Dependency Inversion:**
- **Issue:** Tight coupling between gate logic and metric storage
- `QualityGates` directly accesses `state.metrics.seed_gradient_norm_ratio`
- But this is acceptable for the current architecture

### Integration Architecture

**Problem: Missing Data Flow**

```
┌─────────────────────────────────────────────────────────────┐
│ Current State (BROKEN)                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  GradientIsolationMonitor                                   │
│  ├─ check_isolation() ──> computes host_norm, seed_norm   │
│  └─ stores in self.host_grad_norm, self.seed_grad_norm    │
│                                                             │
│          [NO CONNECTION]  ← ← ← ← ← ←                       │
│                                                             │
│  SeedMetrics                                                │
│  └─ seed_gradient_norm_ratio = 0.0  ← NEVER UPDATED       │
│                                                             │
│  QualityGates._check_g2()                                   │
│  └─ checks seed_gradient_norm_ratio ← ALWAYS 0.0          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Required Architecture (NEEDS IMPLEMENTATION)                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Loop (simic/tolaria)                              │
│  └─ after backward():                                       │
│      ├─ is_isolated, stats = monitor.check_isolation()     │
│      └─ state.metrics.seed_gradient_norm_ratio =           │
│          stats['seed_grad_norm'] / (stats['host_...'] + ε) │
│                                                             │
│          ↓                                                  │
│                                                             │
│  SeedMetrics                                                │
│  └─ seed_gradient_norm_ratio ← POPULATED FROM TRAINING     │
│                                                             │
│          ↓                                                  │
│                                                             │
│  QualityGates._check_g2()                                   │
│  └─ checks seed_gradient_norm_ratio ← HAS REAL VALUE      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Recommendation:**
The architecture requires a bridge between gradient monitoring and metric storage. Options:

1. **Option A: Training loop updates metrics (RECOMMENDED)**
   - After calling `monitor.check_isolation()`, training code computes ratio
   - Pro: Keeps concerns separated
   - Con: Requires finding/modifying training orchestration code

2. **Option B: SeedSlot auto-updates on forward**
   - `SeedSlot` owns `isolation_monitor`, could update metrics
   - Pro: Encapsulated within SeedSlot
   - Con: Mixes lifecycle concerns with gradient tracking

3. **Option C: GradientIsolationMonitor directly updates SeedMetrics**
   - Pass `state.metrics` to monitor, it updates the field
   - Pro: Direct data flow
   - Con: Tight coupling, monitor now knows about SeedMetrics

**Recommended:** Option A - find training loop and add metric update after isolation check.

---

## 4. Documentation and Standards

### Code Comments

✅ **Good:**
- Line 85: Clear explanation of formula
- Lines 407-408: Explains *why* check is needed ("not just riding host improvements")
- Line 409: Documents threshold rationale

✅ **Docstring Updated:**
- Line 377: Updated `_check_g2` docstring to mention "gradient activity"

### Missing Documentation

⚠️ **No usage documentation:**
- How should gradient ratio be computed?
- Where should it be updated?
- What epsilon value for division stability?

**Recommendation:**
Add to `SeedMetrics` docstring:
```python
@dataclass(slots=True)
class SeedMetrics:
    """Metrics tracked during seed lifecycle.

    ...

    seed_gradient_norm_ratio: Ratio of seed gradient norm to host gradient norm.
        Computed as: seed_grad_norm / (host_grad_norm + 1e-8)
        Updated after each backward pass via GradientIsolationMonitor.
        Used by G2 gate to verify independent seed learning.
    """
```

---

## 5. Testing Quality

### Test Execution

```bash
$ uv run pytest tests/kasmina/test_g2_gradient_readiness.py -v
============================= test session starts ==============================
tests/kasmina/test_g2_gradient_readiness.py::TestG2GradientReadiness::test_g2_checks_seed_gradient_norm_ratio PASSED [ 33%]
tests/kasmina/test_g2_gradient_readiness.py::TestG2GradientReadiness::test_g2_passes_with_sufficient_gradient_ratio PASSED [ 66%]
tests/kasmina/test_g2_gradient_readiness.py::TestG2GradientReadiness::test_g2_fails_with_low_gradient_despite_global_improvement PASSED [100%]

============================== 3 passed in 0.68s
```

✅ All tests pass

### Integration Test Status

```bash
$ uv run pytest tests/test_seed_slot.py -v
# All 17 tests pass - no regressions
```

✅ No regressions in existing tests

### Test Coverage Analysis

**Unit Test Coverage:** Good (3 tests covering pass/fail/edge cases)

**Integration Test Coverage:** **MISSING**
- No test verifying gradient data flows from monitor → metrics
- No test verifying G2 behavior with real training
- Tests use manual fixture data only

**Missing Tests:**
1. Integration test: gradient ratio populated during training
2. Edge case: ratio at exact threshold (0.05)
3. Edge case: host gradients near zero
4. End-to-end: seed blocked from BLENDING with low gradient activity

---

## 6. Issues Summary

### Critical Issues

| # | Issue | Severity | Location | Impact |
|---|-------|----------|----------|--------|
| 1 | `seed_gradient_norm_ratio` never populated | **CRITICAL** | Entire implementation | Feature non-functional in production |

### Important Issues

| # | Issue | Severity | Location | Impact |
|---|-------|----------|----------|--------|
| 2 | Hardcoded threshold (0.05) | Important | `slot.py:409` | Not configurable like other gates |
| 3 | No integration tests | Important | Test suite | Can't verify end-to-end behavior |

### Suggestions

| # | Issue | Severity | Location | Impact |
|---|-------|----------|----------|--------|
| 4 | Missing boundary tests | Suggestion | Test suite | Edge cases not verified |
| 5 | Epsilon not documented | Suggestion | `slot.py:85` | Formula incomplete |
| 6 | No usage documentation | Suggestion | Docstrings | Integration unclear |

---

## 7. Recommendations

### Immediate Actions Required (Before Production)

**1. Implement Task 10.1: Wire Gradient Data to Metrics**

Create new task to complete integration:

```python
# Pseudo-code for required implementation
# Location: Training loop (needs investigation)

def training_step(...):
    # ... existing training code ...
    loss.backward()

    # NEW: Update gradient ratio metric
    if seed_slot.isolation_monitor is not None:
        is_isolated, stats = seed_slot.isolation_monitor.check_isolation()

        # Compute ratio with numerical stability
        host_norm = stats['host_grad_norm']
        seed_norm = stats['seed_grad_norm']
        ratio = seed_norm / (host_norm + 1e-8)

        # Update metric
        seed_slot.state.metrics.seed_gradient_norm_ratio = ratio

        # Optionally store raw norms for debugging
        # seed_slot.state.metrics.host_gradient_norm = host_norm
        # seed_slot.state.metrics.seed_gradient_norm = seed_norm

    optimizer.step()
```

**Investigation needed:**
- Find where training loops call backward()
- Locate where `isolation_monitor.check_isolation()` is currently called
- Verify this is the correct integration point

**2. Make Threshold Configurable**

Move hardcoded value to `QualityGates.__init__`:

```python
# src/esper/kasmina/slot.py

class QualityGates:
    def __init__(
        self,
        min_training_improvement: float = 1.0,
        max_isolation_violations: int = 3,
        min_training_epochs: int = 3,
        min_gradient_ratio: float = 0.05,  # NEW
    ):
        self.min_training_improvement = min_training_improvement
        self.max_isolation_violations = max_isolation_violations
        self.min_training_epochs = min_training_epochs
        self.min_gradient_ratio = min_gradient_ratio  # NEW

    def _check_g2(self, state: SeedState) -> GateResult:
        # ...
        if state.metrics.seed_gradient_norm_ratio >= self.min_gradient_ratio:  # Use self
```

**3. Add Integration Tests**

Once wiring is complete:

```python
# tests/kasmina/test_g2_gradient_integration.py

def test_g2_gradient_ratio_populated_during_training():
    """Verify gradient ratio is computed and stored during training."""
    # Setup slot with seed
    # Run training step with backward
    # Verify ratio was populated
    # Verify G2 gate uses real value

def test_g2_blocks_seed_with_low_gradient_activity():
    """End-to-end: seed with low gradient activity blocked at G2."""
    # Create scenario where host learns but seed doesn't
    # Verify G2 fails with gradient_low message
```

---

### Future Improvements (Non-Blocking)

**1. Store Raw Gradient Norms**

Consider adding to `SeedMetrics`:
```python
host_gradient_norm: float = 0.0
seed_gradient_norm: float = 0.0
# Computed property:
@property
def seed_gradient_norm_ratio(self) -> float:
    return self.seed_gradient_norm / (self.host_gradient_norm + 1e-8)
```

Benefits:
- Better debugging (can see both raw values)
- Ratio always uses consistent epsilon
- Can log/plot individual norms

**2. Add Telemetry**

Log gradient ratios for analysis:
```python
# After computing ratio
logger.info(
    f"Gradient activity - Seed: {seed_norm:.4f}, "
    f"Host: {host_norm:.4f}, Ratio: {ratio:.2%}"
)
```

**3. Adaptive Threshold**

Consider making threshold adaptive based on training phase:
- Early training: Lower threshold (seeds still initializing)
- Mid training: Standard threshold (0.05)
- Late training: Higher threshold (expect strong signal)

---

## 8. Final Assessment

### Strengths

1. **Clean Implementation:** Code follows existing patterns perfectly
2. **Good Tests:** Test cases cover the right scenarios
3. **Clear Intent:** Comments explain why check is needed
4. **No Regressions:** All existing tests still pass
5. **Correct Logic:** Gate conjunction and check logic are correct

### Critical Gaps

1. **Missing Integration:** Metric never populated, feature non-functional
2. **Incomplete Testing:** No integration tests verify data flow
3. **Missing Documentation:** Integration point not documented

### Verdict

**Implementation Quality:** 9/10 (what's there is excellent)
**Functional Completeness:** 3/10 (doesn't work in production)
**Overall Score:** 6.5/10

**Status:** ⚠️ **PARTIAL IMPLEMENTATION**

The code that exists is high quality and well-tested at the unit level. However, the missing integration means this feature will not work in production. Every seed will fail G2 because `seed_gradient_norm_ratio` remains 0.0.

**Required Action:** Complete Task 10.1 to wire gradient data from `GradientIsolationMonitor` to `SeedMetrics` before considering this task complete.

---

## 9. Comparison with Plan

### Plan Expectations

Task 10 plan included 6 steps:
1. ✅ Write failing test
2. ✅ Run test to verify it fails
3. ✅ Add field to SeedMetrics
4. ✅ Update G2 gate logic
5. ✅ Run test to verify it passes
6. ✅ Commit

### What Was Missing from Plan

**The plan did not include:**
- Integration between `GradientIsolationMonitor` and `SeedMetrics`
- Investigation of where gradient data should flow
- Documentation of epsilon value for division

**Plan Oversight:**
The plan assumed the gradient ratio would be populated by existing infrastructure, but this connection doesn't exist. The plan should have included a step for wiring the data flow.

### Suggested Plan Amendment

Add between Step 4 and Step 5:

```markdown
**Step 4.5: Wire gradient data to metrics**

Find where `GradientIsolationMonitor.check_isolation()` is called during training.
After the call, compute and store the ratio:

```python
is_isolated, stats = isolation_monitor.check_isolation()
ratio = stats['seed_grad_norm'] / (stats['host_grad_norm'] + 1e-8)
state.metrics.seed_gradient_norm_ratio = ratio
```

Verify with a training run that the metric is populated.
```

---

## 10. Files Modified

### Source Files

- `/home/john/esper-lite/src/esper/kasmina/slot.py` (+17 lines, -2 lines)
  - Added `seed_gradient_norm_ratio` field
  - Updated `_check_g2` gate logic
  - Updated docstring

### Test Files

- `/home/john/esper-lite/tests/kasmina/test_g2_gradient_readiness.py` (new file, 84 lines)
  - 3 test methods in `TestG2GradientReadiness` class

### Total Changes

- **Files changed:** 2
- **Lines added:** 101
- **Lines removed:** 2
- **Net change:** +99 lines

---

## Appendix: Code Snippets

### Implementation: Field Addition

```python
# src/esper/kasmina/slot.py:85
# Gradient-based seed activity metric
seed_gradient_norm_ratio: float = 0.0  # seed_grad_norm / (host_grad_norm + eps)
```

### Implementation: G2 Gate Check

```python
# src/esper/kasmina/slot.py:407-417
# NEW: Gradient-based seed activity check
# Ensures seed is actually learning, not just riding host improvements
min_gradient_ratio = 0.05  # Seed should have at least 5% of host gradient activity
if state.metrics.seed_gradient_norm_ratio >= min_gradient_ratio:
    checks_passed.append(f"seed_gradient_active_{state.metrics.seed_gradient_norm_ratio:.2f}")
    gradient_ok = True
else:
    checks_failed.append(f"seed_gradient_low_{state.metrics.seed_gradient_norm_ratio:.2f}")
    gradient_ok = False

passed = perf_ok and isolation_ok and seed_ok and gradient_ok
```

### Available Gradient Data (Not Connected)

```python
# src/esper/kasmina/isolation.py:116-117
# These values exist but aren't transferred to SeedMetrics
self.host_grad_norm = host_norm
self.seed_grad_norm = seed_norm
```

---

**Review completed:** 2025-12-07
**Next action:** Implement Task 10.1 to complete gradient data wiring
