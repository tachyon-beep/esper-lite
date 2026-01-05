# Defensive Programming Cleanup - Wave 2

**Created:** 2026-01-02
**Status:** Ready for Implementation
**Reference:** Follow-up to `2026-01-02-defensive-programming-cleanup.md`

## Executive Summary

After the initial cleanup (30 fixes, all tests passing), a second deep-dive audit using 6 Explore agents identified 13 additional potential issues. These were reviewed by DRL and PyTorch specialist agents, resulting in:

- **8 CONFIRMED** issues requiring fixes
- **4 REJECTED** as legitimate patterns (documented below for reference)

## Specialist Sign-Off

### DRL Expert Verdict

| Issue | Verdict | Rationale |
|-------|---------|-----------|
| NEW-01 | **CONFIRMED HIGH** | Shapley attribution returning 0.0 for missing slots corrupts reward signal |
| NEW-02 | **CONFIRMED CRITICAL** | Truthiness bug in null coalition baseline systematically biases Shapley values |
| NEW-03 | **CONFIRMED MEDIUM** | "unknown" seed_id corrupts analytics; lifecycle events must have valid IDs |
| NEW-04 | **CONFIRMED MEDIUM** | Same pattern in wandb - pollutes experiment tracking |
| NEW-08 | **REJECTED** | Penalty of 0.0 is semantically correct default (no penalty = 0) |

### PyTorch Expert Verdict

| Issue | Verdict | Rationale |
|-------|---------|-----------|
| NEW-05 | **CONFIRMED HIGH** | Legacy checkpoint shim violates no-legacy policy; should fail-fast |
| NEW-06 | **REJECTED** | Idiomatic Python for Optional with factory default |
| NEW-07 | **REJECTED** | Same as NEW-06 |
| NEW-09 | **REJECTED** | Keys guaranteed present by contract |
| NEW-10 | **CONFIRMED MEDIUM** | Device truthiness bug - torch.device can't be falsy but pattern is wrong |
| NEW-11 | **CONFIRMED MEDIUM** | Optional field truthiness - 0.0 train_loss would be silently treated as None |

## Confirmed Issues by Priority

### Phase 0: CRITICAL (Systematic Training Bias)

#### NEW-02: Shapley Baseline Truthiness Bug
**File:** `src/esper/simic/training/vectorized.py:2291`
**Impact:** Systematically biases Shapley values when null coalition accuracy is 0.0

```python
# CURRENT (buggy):
all_off_acc = all_disabled_accs.get(i) or min(baseline_accs[i].values())

# FIX:
all_off_acc = all_disabled_accs.get(i)
if all_off_acc is None:
    all_off_acc = min(baseline_accs[i].values())
```

**Why Critical:** The null coalition (all seeds disabled) is the reference point for computing marginal contributions. If baseline accuracy is legitimately 0.0 (model predicts nothing correctly), this bug silently substitutes a different value, corrupting all downstream Shapley calculations.

---

### Phase 1: HIGH (Reward Attribution / Policy Violations)

#### NEW-01: Shapley Missing Slot Default
**File:** `src/esper/simic/attribution/counterfactual.py:127`
**Impact:** Returns 0.0 for missing slots instead of failing fast

```python
# CURRENT (buggy):
return self._marginal_contributions.get(slot_id, 0.0)

# FIX:
return self._marginal_contributions[slot_id]
```

**Why High:** A slot_id not in marginal_contributions is a bug - either the computation failed or the caller passed an invalid ID. Returning 0.0 silently corrupts reward attribution.

---

#### NEW-05: Legacy Checkpoint Shim
**File:** `src/esper/kasmina/alpha_controller.py:179`
**Impact:** Violates no-legacy-code policy

```python
# CURRENT (violates policy):
alpha_steepness=float(data.get("alpha_steepness", 12.0)),  # Default for old checkpoints

# FIX:
alpha_steepness=float(data["alpha_steepness"]),
```

**Why High:** CLAUDE.md explicitly prohibits backwards compatibility shims. Old checkpoints without `alpha_steepness` should fail to load, forcing migration.

---

### Phase 2: MEDIUM (Analytics / Telemetry)

#### NEW-03: Analytics seed_id Fallback
**File:** `src/esper/nissa/analytics.py:181,199,231,233`
**Impact:** "unknown" seed pollutes lifecycle analytics

```python
# CURRENT (buggy):
seed_id = event.seed_id or "unknown"

# FIX:
if event.seed_id is None:
    raise ValueError(f"seed_id required for {event.event_type} event")
seed_id = event.seed_id
```

**Why Medium:** Lifecycle events (germination, fossilization, pruning) must have valid seed identifiers. "unknown" corrupts analytics but doesn't affect training.

---

#### NEW-04: WandB slot_id Fallback
**File:** `src/esper/nissa/wandb_backend.py:347,373,398,436`
**Impact:** "unknown" slot pollutes experiment tracking

```python
# CURRENT (buggy):
slot_id = event.slot_id or "unknown"

# FIX:
if event.slot_id is None:
    raise ValueError(f"slot_id required for {event.event_type} event")
slot_id = event.slot_id
```

**Why Medium:** Same pattern as NEW-03. Seed events must have valid slot identifiers.

---

#### NEW-10: Device Truthiness
**File:** `src/esper/tamiyo/policy/action_masks.py:184`
**Impact:** Incorrect pattern (though torch.device can't be falsy)

```python
# CURRENT (wrong pattern):
device = device or torch.device("cpu")

# FIX:
device = device if device is not None else torch.device("cpu")
```

**Why Medium:** While torch.device objects can't be falsy in practice, this pattern is incorrect by convention and should use explicit None check.

---

#### NEW-11: Collector Optional Field Truthiness
**File:** `src/esper/karn/collector.py:370-372`
**Impact:** 0.0 values treated as None in aggregation

```python
# CURRENT (buggy):
total_train_loss += payload.train_loss or 0.0
total_train_accuracy += payload.train_accuracy or 0.0
total_host_grad_norm += payload.host_grad_norm or 0.0

# FIX:
total_train_loss += payload.train_loss if payload.train_loss is not None else 0.0
total_train_accuracy += payload.train_accuracy if payload.train_accuracy is not None else 0.0
total_host_grad_norm += payload.host_grad_norm if payload.host_grad_norm is not None else 0.0
```

**Why Medium:** If train_loss is legitimately 0.0, this bug treats it as None and substitutes 0.0 anyway (coincidentally correct result, but wrong reasoning).

---

## Rejected Issues (No Action Needed)

### NEW-06: slot.py:1014 - Idiomatic Default
```python
metrics = metrics if metrics is not None else SlotMetrics()
```
**Verdict:** This is idiomatic Python for "Optional with factory default". The pattern handles both `None` and explicit factory creation correctly.

### NEW-07: heuristic.py:107 - Same Pattern
Same as NEW-06. Legitimate Optional handling.

### NEW-08: heuristic.py:351,359,367 - Semantic Zero
```python
penalty = penalties.get(slot_id, 0.0)
```
**Verdict:** A penalty of 0.0 is the correct semantic default - "no penalty" equals zero. Missing keys mean the slot has no penalty, which is correctly represented as 0.0.

### NEW-09: slot.py:1834-1835 - Contract Guarantee
```python
window = self.accuracy_windows.get(slot_id, [])
```
**Verdict:** Keys are guaranteed present by the contract (slots register their windows on creation). The `.get()` is defensive but harmless since the code path is unreachable.

---

## Implementation Checklist

- [ ] **Phase 0:** Fix NEW-02 (vectorized.py Shapley baseline)
- [ ] **Phase 1A:** Fix NEW-01 (counterfactual.py missing slot)
- [ ] **Phase 1B:** Fix NEW-05 (alpha_controller.py legacy shim)
- [ ] **Phase 2A:** Fix NEW-03 (analytics.py seed_id)
- [ ] **Phase 2B:** Fix NEW-04 (wandb_backend.py slot_id)
- [ ] **Phase 2C:** Fix NEW-10 (action_masks.py device)
- [ ] **Phase 2D:** Fix NEW-11 (collector.py optional fields)
- [ ] Run full test suite
- [ ] Commit with reference to this plan

## Testing Strategy

1. **After Phase 0:** Run Shapley-specific tests to verify baseline handling
2. **After Phase 1:** Run full test suite - expect failures if tests relied on legacy checkpoint format
3. **After Phase 2:** Run full test suite - may need to update test fixtures that pass None
4. **Final:** `uv run pytest` - all 2,745+ tests must pass

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 0 | High - affects all Shapley calculations | Isolated change, easy rollback |
| 1A | Medium - may expose hidden bugs | KeyError will identify invalid callers |
| 1B | Medium - breaks old checkpoints | Intended behavior per policy |
| 2 | Low - analytics only | ValueError will identify bad event sources |
