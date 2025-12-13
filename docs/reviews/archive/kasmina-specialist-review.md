  ✅ ALREADY FIXED (2)

  | Item                                    | Location          |
  |-----------------------------------------|-------------------|
  | seed_gradient_norm_ratio never computed | slot.py:903-930   |
  | PROBATIONARY lifecycle deadlock         | slot.py:1144-1182 |

  ---
  🔴 HIGH (3)

  | #   | Action                                                                                        | Location      | Blocker?                          |
  |-----|-----------------------------------------------------------------------------------------------|---------------|-----------------------------------|
  | 1   | Add immediate cull for negative counterfactual (don't wait for timeout)                       | slot.py:1154  | No                                |
  | 2   | Verify Tolaria calls update_gradient_metrics() after backward pass                            | Training loop | Yes - fix is useless without this |
  | 3   | Parameter normalization mismatch: SeedSlot uses raw ratio, vectorized uses √params-normalized | slot.py:925   | No                                |

  ---
  🟡 MEDIUM (4)

  | #   | Action                                                                      | Location               |
  |-----|-----------------------------------------------------------------------------|------------------------|
  | 4   | Clear FlexAttention cache on device transfer (memory leak)                  | transformer.py:141     |
  | 5   | Simplify torch._foreach_norm → torch.linalg.vector_norm(torch.stack(norms)) | isolation.py:105-112   |
  | 6   | FIFO eviction not true LRU - use OrderedDict.move_to_end()                  | transformer.py:149-153 |
  | 7   | Add test coverage for step_epoch() lifecycle transitions                    | tests/kasmina/         |

  ---
  🟢 LOW (6)

  | #   | Action                                                                  | Location          |
  |-----|-------------------------------------------------------------------------|-------------------|
  | 8   | Add @override decorators for protocol implementations                   | host.py           |
  | 9   | Extract magic numbers (0.05, 1e-8, 0.9/0.1) to named constants          | slot.py           |
  | 10  | NormSeed unbounded scale parameter could cause gradient explosion       | blueprints/cnn.py |
  | 11  | SHADOWING stage has no distinct behavior (redundant with late BLENDING) | slot.py           |
  | 12  | SeedSlot.forward() compile disable may be over-conservative             | slot.py:903-960   |
  | 13  | Consider parameter-normalized gradient ratio alignment with vectorized  | slot.py           |

# Kasmina Subsystem Multi-Agent Deep Dive Analysis

**Date:** 2025-12-09
**Files Analyzed:** `src/esper/kasmina/` (~2,400 LOC across 9 files)
**Agents:** DRL Expert, PyTorch Expert, Code Reviewer

---

## Executive Summary

**Overall Grade: A-** (production-quality research code)

### Critical Bugs Found and Fixed

| # | Bug | Fix Applied | File |
|---|-----|-------------|------|
| 1 | `seed_gradient_norm_ratio` never computed - G2 gate always fails | Added `update_gradient_metrics()` method | `slot.py:903-930` |
| 2 | PROBATIONARY stage has no exit path - lifecycle deadlock | Added PROBATIONARY handling in `step_epoch()` | `slot.py:1144-1182` |

### Remaining Issues After Fixes

| Priority | Issue | Location |
|----------|-------|----------|
| HIGH | Negative counterfactual should cull immediately, not wait for timeout | `slot.py:1154` |
| MEDIUM | Tolaria must call `update_gradient_metrics()` - verify integration | Training loop |
| MEDIUM | Parameter normalization mismatch between SeedSlot and vectorized training | `slot.py:925` |
| LOW | FlexAttention cache not cleared on device transfer | `transformer.py:141` |
| LOW | `torch._foreach_norm` combined norm could be simplified | `isolation.py:105-112` |

---

## Fix 1: `update_gradient_metrics()` (lines 903-930)

### Code Added

```python
def update_gradient_metrics(self, host_grad_norm: float, seed_grad_norm: float) -> None:
    if not self.state:
        return
    if self.isolate_gradients and host_grad_norm > 1e-6:
        self.state.metrics.isolation_violations += 1
    eps = 1e-8
    ratio = seed_grad_norm / (host_grad_norm + eps)
    self.state.metrics.seed_gradient_norm_ratio = ratio
    current_avg = self.state.metrics.gradient_norm_avg
    self.state.metrics.gradient_norm_avg = 0.9 * current_avg + 0.1 * seed_grad_norm
```

### Expert Assessments

**PyTorch Expert:**

- Numerical stability: PASS (eps=1e-8 appropriate for float32)
- torch.compile: PASS (pure Python, no decorator needed)
- Performance: PASS (<50ns overhead)
- Thread safety: Single-threaded assumption OK

**DRL Expert:**

- Metric is adequate but NOT parameter-normalized
- Vectorized training uses `(seed/√params) / (host/√params)`
- SeedSlot uses raw `seed / host`
- 0.05 threshold may be too strict for small seeds

### Integration Requirement

```python
# In Tolaria training loop, after loss.backward():
if slot.isolation_monitor:
    _, stats = slot.isolation_monitor.check_isolation()
    slot.update_gradient_metrics(
        host_grad_norm=stats["host_grad_norm"],
        seed_grad_norm=stats["seed_grad_norm"]
    )
```

---

## Fix 2: PROBATIONARY Handling (lines 1144-1182)

### Code Added

```python
if stage == SeedStage.PROBATIONARY:
    max_probation_epochs = 5
    if self.task_config:
        max_probation_epochs = max(3, int(self.task_config.max_epochs * 0.1))

    # Path 1: FOSSILIZE if counterfactual proves worth
    if self.state.metrics.counterfactual_contribution is not None:
        gate_result = self.gates.check_gate(self.state, SeedStage.FOSSILIZED)
        if gate_result.passed:
            # ... transition to FOSSILIZED
            return

    # Path 2: CULL on timeout
    if self.state.metrics.epochs_in_current_stage >= max_probation_epochs:
        self.cull(reason="probation_timeout")
        return
```

### Expert Assessments

**PyTorch Expert:**

- Logic correctness: PASS with GAP
- **GAP IDENTIFIED:** If counterfactual <= 0, seed waits for timeout instead of immediate cull
- Suggested fix:

```python
elif self.state.metrics.counterfactual_contribution <= 0:
    self.cull(reason="negative_counterfactual")
    return
```

**DRL Expert:**

- Lifecycle now complete - no more zombie processes
- State transitions valid per Leyline contract
- Timeout mechanism (5 epochs or 10% of max) is reasonable
- Counterfactual-only G5 gate is correct design (no fallback)

---

## Original Analysis Findings (Pre-Fix)

### DRL Expert Top Findings

1. **seed_gradient_norm_ratio Never Computed** - G2 gate checked value that was never set
2. **G5 Counterfactual Deadlock** - Seeds stuck in PROBATIONARY forever
3. **SHADOWING Stage Has No Distinct Behavior** - Redundant with late BLENDING

### PyTorch Expert Top Findings

1. **torch._foreach_norm Combined Norm Suboptimal** - `torch.stack(norms).pow(2).sum().sqrt()` should be `torch.linalg.vector_norm(torch.stack(norms))`
2. **FlexAttention Cache Issues** - Not cleared on device transfer, FIFO not true LRU
3. **SeedSlot.forward() Compile Disable May Be Over-Conservative** - Could enable partial compilation

### Code Reviewer Findings

1. **No Bugs Found** - Code passes all quality checks
2. **CLAUDE.md Compliant** - No hasattr, no legacy code
3. **Missing `@override` Decorator** - Protocol implementations should use Python 3.12+ decorator
4. **Test Coverage Gaps** - `step_epoch()`, `cull()`, G3/G4 gates not directly tested

---

## Test Results After Fixes

```
52 tests passed, 0 failed (8.82s)
```

---

## Recommended Next Actions

### Immediate (Critical)

1. Add immediate cull for negative counterfactual in PROBATIONARY
2. Verify Tolaria calls `update_gradient_metrics()` after backward pass

### Near-Term (Important)

3. Clear FlexAttention cache on device transfer
4. Simplify foreach_norm computation in isolation.py
5. Add test coverage for `step_epoch()` lifecycle transitions

### Future (Nice-to-have)

6. Add `@override` decorators for protocol implementations
7. Consider parameter-normalized gradient ratio alignment
8. Extract magic numbers (0.05, 1e-8, 0.9/0.1) to named constants

---

## Files Modified

| File | Lines Added | Changes |
|------|-------------|---------|
| `src/esper/kasmina/slot.py` | ~70 | Added `update_gradient_metrics()`, PROBATIONARY handling |

## Test Files

All 52 tests in `tests/kasmina/` pass after fixes.

=============

Kasmina Subsystem Multi-Agent Deep Dive Analysis

  Executive Summary

  Three specialist agents (DRL Expert, PyTorch Expert, Code Reviewer) conducted a comprehensive review of the Kasmina subsystem (~2,400 LOC across 9 files). Overall assessment: A- (production-quality research code) with well-designed architecture and excellent documentation.

  Critical Findings (Priority Order)

  | #   | Finding                                                                      | Severity | Expert  | File                   |

  |-----|------------------------------------------------------------------------------|----------|---------|------------------------|

  | 1   | seed_gradient_norm_ratio never computed - G2 gate checks uninitialized value | 🐛 Bug   | DRL     | slot.py:101,430        |

  | 2   | G5 counterfactual requirement may cause lifecycle deadlock                   | ⚠️ Risk  | DRL     | slot.py:485-504        |

  | 3   | FlexAttention block mask cache not cleared on device transfer                | ⚠️ Risk  | PyTorch | transformer.py:141     |

  | 4   | torch._foreach_norm combined norm computation suboptimal                     | 📉 Perf  | PyTorch | isolation.py:105-112   |

  | 5   | FIFO eviction in FlexAttention cache is not true LRU                         | 🔧 BP    | All     | transformer.py:149-153 |

  ---

  Detailed Findings by Agent

  Agent 1: Deep Reinforcement Learning Expert

  Top 3 Findings

  1. seed_gradient_norm_ratio Never Computed 🐛

- Location: slot.py:101, 430-435

- Impact: G2 gate checks seed_gradient_norm_ratio >= 0.05 but value defaults to 0.0 and is never set

- Result: All seeds fail G2 unless external code sets this field

- Fix: Add method to update metric or remove from G2 check

  2. G5 Counterfactual Deadlock Risk ⚠️

- Location: slot.py:485-523

- Impact: Seeds in PROBATIONARY cannot advance to FOSSILIZED without counterfactual_contribution being set externally

- Result: Lifecycle trap if vectorized training doesn't populate this field

- Fix: Add timeout/fallback for PROBATIONARY seeds

  3. SHADOWING Stage Has No Distinct Behavior 🔧

- Location: slot.py:949-953

- Impact: SHADOWING behaves identically to late BLENDING (alpha=1.0, same detach logic)

- Result: Redundant lifecycle stage

- Fix: Define unique behavior or merge into PROBATIONARY

  DRL-Specific Recommendations

- STE implementation correct but documentation misleading

- Alpha schedule sigmoid is conservative (spends time at fractional alphas)

- Counterfactual attribution design is excellent (correct causal approach)

- NormSeed unbounded scale parameter could cause gradient explosion

  ---

  Agent 2: PyTorch Expert (PyTorch 2.9 / Python 3.13)

  Top 3 Findings

  1. torch._foreach_norm Combined Norm Inefficient 📉

- Location: isolation.py:105-112

- Current: torch.stack(norms).pow(2).sum().sqrt()

- Better: torch.linalg.vector_norm(torch.stack(norms))

- Impact: Extra intermediate tensor allocation and operations

  2. FlexAttention Block Mask Cache Issues ⚠️

- Location: transformer.py:141-167

- Issues:

  - Cache not cleared on .to() device transfer (memory leak)

  - FIFO eviction is not true LRU

- Fix: Override _apply() to clear cache; use OrderedDict.move_to_end() for LRU

  3. SeedSlot.forward() Compile Disable May Be Over-Conservative ✨

- Location: slot.py:903-960

- Current: Entire method disabled from torch.compile

- Opportunity: Refactor to enable partial compilation of BLENDING path

- Tool: Use torch._dynamo.error_on_graph_break() to identify specific breaks

  PyTorch 2.9 Opportunities

- Add @override decorators for protocol implementations

- Consider FP16 support on X86 CPU (now stable)

- Test with error_on_graph_break() context manager

- RoPE support could leverage fused Inductor kernels

  Positive Findings

- torch.lerp usage in blending is optimal (fused kernel)

- ste_forward() is canonical STE implementation

- Host networks (CNNHost, TransformerHost) are fully compile-friendly

- SDPA usage in CausalSelfAttention is optimal for PyTorch 2.9

  ---

  Agent 3: Code Reviewer

  Summary: No Bugs Found ✅

  The code passes all quality checks with only best practice recommendations:

  1. Private API Usage 🔧

- torch._foreach_norm is private but matches clip_grad_norm_ internals

- Acceptable risk with version documentation

  2. Missing @override Decorator 🔧

- Protocol implementations should use Python 3.12+ @override

- Affects: CNNHost, TransformerHost, MorphogeneticModel

  3. CLAUDE.md Compliance ✅

- No hasattr() without authorization

- No legacy/compatibility code

- No archive code references

  Test Coverage Gaps

  | Component                  | Status                             |

  |----------------------------|------------------------------------|

  | SeedSlot.step_epoch()      | Not tested (complex state machine) |

  | SeedSlot.cull()            | Only indirect coverage             |

  | QualityGates._check_g3/g4  | Not tested                         |

  | MorphogeneticModel.to()    | Not tested                         |

  | SeedState.sync_telemetry() | Not tested                         |

  Strengths Identified

- Excellent module-level documentation (torch.compile strategy in slot.py)

- Modern type hints (X | None, dict[str, int])

- Well-structured dataclasses (slots=True, kw_only=True)

- Comprehensive error messages with actionable information

  ---

  Consolidated Recommendations

  Priority 1: Bug Fixes (Immediate)

# 1. Fix seed_gradient_norm_ratio computation in SeedSlot

  def update_gradient_metrics(self, host_grad_norm: float, seed_grad_norm: float):

      eps = 1e-8

      self.state.metrics.seed_gradient_norm_ratio = seed_grad_norm / (host_grad_norm + eps)

# 2. Add PROBATIONARY timeout in step_epoch()

  if stage == SeedStage.PROBATIONARY:

      if self.state.metrics.epochs_in_current_stage > MAX_PROBATION_EPOCHS:

          self.cull("probation_timeout")

  Priority 2: Risk Mitigation (Near-term)

# 3. Clear FlexAttention cache on device transfer

  class FlexAttentionSeed(nn.Module):

      def _apply(self, fn):

          self._block_mask_cache.clear()

          return super()._apply(fn)

# 4. Fix foreach_norm computation

  if host_grads:

      norms = torch._foreach_norm(host_grads)

      host_norm = torch.linalg.vector_norm(torch.stack(norms)).item()

  Priority 3: Modernization (Future)

# 5. Add @override decorators (Python 3.12+)

  from typing import override

  class CNNHost(nn.Module):

      @property

      @override

      def injection_points(self) -> dict[str, int]:

          ...

  ---

  Dependency Graph

  ┌─────────────────────────────────────────────────────────────────┐

  │                    Implementation Order                          │

  ├─────────────────────────────────────────────────────────────────┤

  │  1. seed_gradient_norm_ratio fix                                │

  │     └── Unblocks: G2 gate correctness                           │

  │                                                                 │

  │  2. FlexAttention cache fixes                                   │

  │     └── Independent: Can be done in parallel                    │

  │                                                                 │

  │  3. foreach_norm optimization                                   │

  │     └── Independent: Performance only                           │

  │                                                                 │

  │  4. PROBATIONARY timeout                                        │

  │     └── Depends on: 1 (G2 must work first)                      │

  │                                                                 │

  │  5. @override decorators                                        │

  │     └── Independent: Python modernization                       │

  │                                                                 │

  │  6. Test coverage expansion                                     │

  │     └── Depends on: 1, 4 (test the fixes)                       │

  └─────────────────────────────────────────────────────────────────┘

  ---

  Appendix: File-by-File Summary

  | File                      | Lines | Assessment | Action Items                                        |

  |---------------------------|-------|------------|-----------------------------------------------------|

  | slot.py                   | 1160  | Good       | Fix seed_gradient_norm_ratio, add PROBATION timeout |

  | host.py                   | 380   | Excellent  | Add @override decorators                            |

  | isolation.py              | 150   | Good       | Optimize foreach_norm                               |

  | blueprints/transformer.py | 187   | Good       | Fix FlexAttention cache                             |

  | blueprints/cnn.py         | 172   | Excellent  | None                                                |

  | blueprints/registry.py    | 126   | Excellent  | None                                                |

  | protocol.py               | 40    | Excellent  | None                                                |

● ---

  Analysis Complete

  The multi-agent deep dive of the Kasmina subsystem is complete. Three specialist agents analyzed ~2,400 LOC across 9 files.

  Key Takeaways

  Overall Grade: A- (production-quality research code)

  1 Bug Found:

- seed_gradient_norm_ratio never computed → G2 gate checks uninitialized value (slot.py:101,430)

  2 Risk Items:

- G5 counterfactual requirement may cause lifecycle deadlock

- FlexAttention cache not cleared on device transfer

  2 Performance Opportunities:

- torch._foreach_norm combined norm could be simplified

- SeedSlot.forward() compilation could be partially enabled

  Positive Highlights:

- Excellent documentation and type hints

- Correct STE and SDPA implementations

- CLAUDE.md compliant (no hasattr, no legacy code)

- Host networks are fully torch.compile-friendly

  Test Coverage Gaps:

- step_epoch() state machine not tested

- cull() only indirect coverage

- G3/G4 gates not tested
