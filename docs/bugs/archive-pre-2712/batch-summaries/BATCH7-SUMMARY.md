# Batch 7 Summary: Simic Telemetry - Training Sensors

**Domain:** `/home/john/esper-lite/src/esper/simic/telemetry/`
**Files Reviewed:** 9 files (anomaly_detector.py, debug_telemetry.py, emitters.py, gradient_collector.py, gradient_ema.py, __init__.py, lstm_health.py, profiler.py, telemetry_config.py)
**Date:** 2024-12-27

---

## Executive Summary

The Simic telemetry subsystem demonstrates **strong PyTorch engineering** with well-designed async patterns for GPU efficiency. However, **a systemic pattern emerges: many telemetry components are fully implemented but never wired into the training loop**. This violates CLAUDE.md's explicit guidance: *"If you are being asked to deliver a telemetry component, do not defer or put it off."*

### Key Themes

1. **Dead Telemetry Components** (3 P1 tickets): GradientEMATracker, check_performance_degradation(), check_gradient_drift() - all implemented but never called
2. **LSTM Health Unwired** (1 P3): Policy uses LSTM but compute_lstm_health() is never invoked
3. **Good GPU Patterns**: Single-sync points, torch._foreach_norm usage, async collection with deferred materialization
4. **Test Coverage Gaps**: Stateful components (EMA tracker, LSTM health) have zero tests

---

## Ticket Summary

### By Severity After Cross-Review

| Severity | Count | Action |
|----------|-------|--------|
| P1 | 4 | Must fix |
| P2 | 7 | Should fix |
| P3 | 6 | Nice to fix |
| P4 | 2 | Documentation |
| WON'T FIX | 3 | Closed |

### Tickets to Fix

| Ticket | Severity | Consensus | Issue |
|--------|----------|-----------|-------|
| **B7-CR-01** | P1 | ✅ UNANIMOUS | check_value_function() defaults immediately raise - remove defaults |
| **B7-DRL-01** | P1 | ✅ UNANIMOUS | GradientEMATracker never used - wire up per CLAUDE.md |
| **B7-DRL-02** | P1 | ✅ UNANIMOUS | check_performance_degradation() explicitly unwired - wire up |
| **B7-PT-01** | P1 | ✅ ENDORSE | compute_grad_norm_surrogate() overflow - use torch._foreach_norm |
| **B7-CR-02** | P2 | REFINE→P2 | Type annotation injection - use dataclasses.replace() |
| **B7-PT-02** | P2 | REFINE | O(n) GPU syncs in debug - acceptable but document trade-off |
| **B7-PT-03** | P2 | ✅ ENDORSE | Large tensor allocation - implement per-param accumulation |
| **B7-CR-03** | P2 | ✅ UNANIMOUS | Unused future params - remove per YAGNI |
| **B7-CR-04** | P2 | DEFER | EMA bias correction - bundle with B7-DRL-01 workstream |
| **B7-PT-04** | P2→P3 | REFINE | GradientEMATracker tests - wire first, then test |
| **B7-PT-05** | P2→P3 | REFINE | LSTMHealthMetrics tests - wire first, then test |
| **B7-PT-06** | P2→P3 | OBJECT | training_profiler - intentionally opt-in, add docs |
| **B7-DRL-03** | P3 | DEFER | check_gradient_drift() - bundle with B7-DRL-01 |
| **B7-DRL-04** | P3 | ✅ WIRE UP | compute_lstm_health() - policy uses LSTM, wire it |
| **B7-CR-06** | P3 | ✅ UNANIMOUS | Exception swallows errors - narrow + exc_info |
| **B7-CR-07** | P3→P4 | REFINE | Private API fallback - lower priority |
| **B7-CR-08** | P3 | ✅ UNANIMOUS | AnomalyReport.details overwrites - use list[str] |
| **B7-CR-09** | P3 | ✅ ENDORSE | Unknown categories silent - warn + return False |
| **B7-PT-07** | P4 | ✅ ENDORSE | torch.compile decorator - add @torch.compiler.disable |
| **B7-PT-08** | P4 | DOCS ONLY | Non-standard metric - document formula rationale |

### Tickets Closed (Won't Fix)

| Ticket | Reason |
|--------|--------|
| **B7-CR-05** | emit_last_action return value is used by tests - not legacy |
| **B7-CR-10** | normalized_ratio edge case - code is actually correct |

---

## Recommended Workstreams

### 1. Gradient Drift Detection (Bundle: B7-DRL-01 + B7-CR-04 + B7-DRL-03)

Wire up complete gradient drift detection:
1. Call `grad_ema_tracker.update()` after each PPO update
2. Add bias correction to EMA (B7-CR-04)
3. Wire `check_gradient_drift()` into `check_all()` (B7-DRL-03)
4. Add tests for GradientEMATracker (B7-PT-04)

**DRL Impact:** Catches slow training degradation before catastrophic failure.

### 2. Performance Degradation Detection (B7-DRL-02)

Wire `check_performance_degradation()` at epoch end. Function is fully implemented with proper warmup thresholds.

**DRL Impact:** Detects catastrophic forgetting and reward hacking.

### 3. LSTM Health Monitoring (B7-DRL-04 + B7-PT-05)

Wire `compute_lstm_health()` into rollout loop. Policy uses LSTM (confirmed: `policy_type="lstm"`).

**DRL Impact:** Catches hidden state explosion/vanishing before NaN losses.

### 4. API Cleanup (B7-CR-01, B7-CR-03, B7-CR-06, B7-CR-08)

Quick fixes with unanimous consensus:
- Remove misleading defaults from check_value_function()
- Remove unused future parameters
- Narrow exception handling + add exc_info
- Change AnomalyReport.details to list[str]

---

## Severity Distribution

```
P1: ████ 4 tickets (18%)
P2: ███████ 7 tickets (32%)
P3: ██████ 6 tickets (27%)
P4: ██ 2 tickets (9%)
WF: ███ 3 tickets (14%)
```

---

## Files Changed Summary

| File | Tickets | Key Issues |
|------|---------|------------|
| anomaly_detector.py | 3 | Misleading defaults, check_gradient_drift orphaned, details overwrite |
| emitters.py | 4 | Type mutation, perf degradation unwired, exception swallowing, grad norm overflow |
| gradient_collector.py | 3 | Large tensor alloc, private API, non-standard metric |
| gradient_ema.py | 3 | Never used, no bias correction, no tests |
| lstm_health.py | 2 | Never called, no tests |
| debug_telemetry.py | 3 | O(n) syncs, unused params, torch.compile |
| profiler.py | 1 | Not integrated (intentional - opt-in) |
| telemetry_config.py | 1 | Silent failure for unknown categories |

---

## Cross-Review Statistics

| Verdict | Count |
|---------|-------|
| UNANIMOUS ENDORSE | 8 |
| ENDORSE (2/3) | 4 |
| REFINE | 7 |
| OBJECT | 3 |

**High Agreement Rate:** 55% unanimous, 82% consensus (at least 2/3 agreement).
