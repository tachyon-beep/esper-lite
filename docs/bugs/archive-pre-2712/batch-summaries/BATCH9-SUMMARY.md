# Batch 9 Summary: Tamiyo Core + Networks

**Domain:** `/home/john/esper-lite/src/esper/tamiyo/`
**Files Reviewed:** 6 files (decisions.py, heuristic.py, __init__.py, tracker.py, networks/factored_lstm.py, networks/__init__.py)
**Date:** 2025-12-27

---

## Executive Summary

The Tamiyo domain is the **brain of Esper** - implementing both a rule-based heuristic policy baseline and an LSTM-based factored actor-critic network for strategic seed lifecycle decisions. The code demonstrates **high quality architecture** with proper LayerNorm placement, forget gate bias initialization, and careful hidden state memory management documentation.

### Key Themes

1. **Policy Compliance Issue**: One P1 finding - a "Legacy" comment that violates the No Legacy Code Policy
2. **LSTM Memory Safety**: Hidden state detachment is well-documented but caller responsibility - helper method recommended
3. **Code Duplication Risk**: Style mask logic between get_action/evaluate_actions could diverge (P2 upgrade)
4. **Minor Cleanups**: Several P4 documentation and style issues

---

## Ticket Summary

### By Severity After Cross-Review

| Severity | Count | Action |
|----------|-------|--------|
| P1 | 1 | Must fix |
| P2 | 3 | Should fix |
| P3 | 4 | Nice to fix |
| P4 | 9 | Documentation/Style |
| WON'T FIX | 4 | Closed |

### Tickets to Fix

| Ticket | Severity | Consensus | Issue |
|--------|----------|-----------|-------|
| **B9-CR-01** | P1 | ✅ UNANIMOUS | "Legacy" comment violates No Legacy Code Policy |
| **B9-CR-02** | P2 | ✅ UNANIMOUS | best_val_loss name misleading (window-best not global) |
| **B9-PT-01** | P2 | ✅ UNANIMOUS | Style mask clone allocates on every get_action() |
| **B9-PT-03** | P2 | ✅ ENDORSE (2/3) | Style mask logic duplicated - divergence risk |
| **B9-CR-03** | P3 | ✅ UNANIMOUS | Multi-slot priority semantics undocumented |
| **B9-CR-04** | P3 | ✅ ENDORSE (2/3) | Duplicate TrainingMetrics construction in update/peek |
| **B9-DRL-05** | P3 | ✅ UNANIMOUS | Hidden state detachment is caller responsibility |
| **B9-PT-06** | P3 | ✅ ENDORSE (2/3) | Two config paths for network head sizes |
| **B9-DRL-06** | P4 | ✅ UNANIMOUS | No option to skip hidden state return (minor overhead) |
| **B9-CR-05** | P4 | ✅ UNANIMOUS | float('inf') vs math.inf style inconsistency |
| **B9-DRL-08** | P4 | ✅ UNANIMOUS | TamiyoDecision.confidence field is unused |
| **B9-DRL-09** | P4 | ✅ ENDORSE (2/3) | Hardcoded 5.0 divisor (blocked by B9-DRL-08) |
| **B9-DRL-10** | P4 | ✅ ENDORSE (2/3) | env_id conditional silently skips telemetry |
| **B9-PT-04** | P4 | ✅ UNANIMOUS | Redundant type annotation comments |
| **B9-PT-05** | P4 | ✅ UNANIMOUS | Inconsistent docstring depth |
| **B9-PT-02** | P3 | ✅ UNANIMOUS | No runtime guard for inference_mode backprop |

### Tickets Closed (Won't Fix)

| Ticket | Reason |
|--------|--------|
| **B9-DRL-01** | REFINE→P2: Blueprint index IS incremented in loop; fallback behavior is design choice |
| **B9-DRL-02** | OBJECT (2/3): Misreads conjunction logic; loss_delta >= 0 prevents regression |
| **B9-DRL-03** | OBJECT (2/3): Defensive programming prohibited by CLAUDE.md |
| **B9-DRL-04** | REFINE→P3: Intentional safety design; needs documentation only |
| **B9-DRL-07** | SPLIT (1/1/1): Action helpers are specific to heuristic dynamic enum |

---

## Recommended Workstreams

### 1. Policy Compliance Fix (Bundle: B9-CR-01)

Single-line fix - update the misleading comment:

```python
# Before:
# Legacy heuristic (kept for backwards compatibility)

# After:
# Heuristic policy (baseline for comparison)
```

**Impact:** Policy compliance, removes audit risk.

### 2. Style Mask Safety (Bundle: B9-PT-01 + B9-PT-03)

Extract style mask logic to a shared helper to prevent divergence:

```python
def _apply_style_irrelevance_mask(
    self,
    style_mask: torch.Tensor,
    ops: torch.Tensor,
) -> torch.Tensor:
    """Force style to SIGMOID_ADD when op doesn't use style."""
    style_irrelevant = (ops != LifecycleOp.GERMINATE) & (
        ops != LifecycleOp.SET_ALPHA_TARGET
    )
    result = style_mask.clone()
    result[style_irrelevant] = False
    result[style_irrelevant, int(GerminationStyle.SIGMOID_ADD)] = True
    return result
```

**Impact:** Prevents PPO ratio computation errors from behavior divergence.

### 3. API Clarity (Bundle: B9-CR-02 + B9-PT-06)

1. Rename `best_val_loss` to `window_best_val_loss` or document semantics in leyline
2. Document config precedence (explicit params override slot_config) or use factory methods

**Impact:** Clearer API contracts, fewer consumer mistakes.

### 4. Hidden State Helper (Bundle: B9-DRL-05)

Add convenience method for LSTM hidden state detachment:

```python
@staticmethod
def detach_hidden(hidden: HiddenState) -> HiddenState:
    """Detach hidden state gradient graphs at episode boundaries."""
    h, c = hidden
    return (h.detach(), c.detach())
```

**Impact:** Safer API, reduces memory leak risk from missed detachment.

---

## Severity Distribution

```
P1: █ 1 ticket (5%)
P2: ███ 3 tickets (14%)
P3: ████ 4 tickets (19%)
P4: █████████ 9 tickets (43%)
WF: ████ 4 tickets (19%)
```

---

## Files Changed Summary

| File | Tickets | Key Issues |
|------|---------|------------|
| __init__.py | 1 | "Legacy" comment policy violation |
| tracker.py | 4 | best_val_loss naming, duplicate construction, telemetry skip |
| factored_lstm.py | 7 | Style mask clone/duplication, hidden state API, config paths |
| heuristic.py | 3 | Multi-slot priority, blueprint fallback (closed) |
| decisions.py | 2 | confidence unused, action helpers (closed) |

---

## Cross-Review Statistics

| Verdict | Count |
|---------|-------|
| UNANIMOUS ENDORSE | 10 |
| ENDORSE (2/3) | 6 |
| REFINE | 3 |
| OBJECT (Close) | 2 |

**High Agreement Rate:** 76% clear endorsements at first verdict.

---

## Notable Insights

### PyTorch Engineering Strengths
- Excellent LSTM initialization (forget gate bias = 1.0 per Gers et al.)
- Proper LayerNorm placement (pre-LSTM and post-LSTM)
- FP16-safe masking with MASKED_LOGIT_VALUE = -1e4
- Clear inference_mode vs training mode separation
- Well-documented hidden state memory management requirements

### DRL Implementation Quality
- Factored action space with 8 independent heads
- MaskedCategorical for safe action sampling
- Ransomware detection in heuristic (seeds with high counterfactual but negative improvement)
- Blueprint penalty system with decay prevents thrashing
- Correct separation of get_action (inference) vs evaluate_actions (training)

### Areas for Improvement
- Style mask logic duplication creates divergence risk (P2)
- Hidden state detachment could have helper method for safety
- TrainingMetrics construction duplicated between update/peek
- Some network parameters have dual configuration paths

---

## Key False Positives Caught

1. **B9-DRL-02 (Stabilization edge case)**: The conjunction logic `loss_delta >= 0 AND relative_improvement < threshold AND val_loss < prev*1.5` was misread. The `loss_delta >= 0` check already prevents any regression from counting.

2. **B9-DRL-03 (Missing None validation)**: Adding defensive `if improvement is None` handling would violate CLAUDE.md's prohibition on defensive programming. The upstream contract ensures HOLDING seeds have metrics.

3. **B9-DRL-01 (Blueprint fallback)**: The loop increments `_blueprint_index` N times before reaching fallback, so rotation continues correctly on next call.

---

## Conclusion

Batch 9 (Tamiyo Core + Networks) demonstrates **high code quality** with solid RL architecture. The primary actionable finding is the P1 policy violation (trivial fix). The P2 style mask duplication is important for training correctness. Most findings are P4 documentation improvements.

The domain is **production-ready** with the noted minor improvements.
