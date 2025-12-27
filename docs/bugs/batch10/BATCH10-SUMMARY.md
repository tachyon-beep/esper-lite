# Batch 10 Summary: Tamiyo Policy Abstraction Layer

**Date:** 2025-12-28
**Domain:** `src/esper/tamiyo/policy/`
**Files Reviewed:** 9 files (protocol.py, types.py, registry.py, factory.py, action_masks.py, features.py, lstm_bundle.py, heuristic_bundle.py, __init__.py)

---

## Executive Summary

Batch 10 reviewed the Tamiyo Policy Abstraction Layer - a well-designed Protocol-based architecture that avoids nn.Module MRO conflicts while providing clean separation between neural (LSTM) and heuristic policies.

**Cross-review identified 2 false positives** (B10-DRL-02, B10-DRL-03) where findings were actually intentional design choices, and **2 No Legacy Code Policy violations** (B10-DRL-04, B10-DRL-05) elevated to P2.

---

## Final Issue Counts (After Cross-Review)

| Severity | Count | Tickets |
|----------|-------|---------|
| **P1** | 1 | B10-PT-02 |
| **P2** | 5 | B10-DRL-01, B10-PT-01, B10-CR-02, B10-PT-03, B10-DRL-04, B10-DRL-05 |
| **P3** | 4 | B10-CR-03, B10-DRL-06, B10-PT-05, B10-CR-01 |
| **P4** | 6 | B10-PT-04, B10-CR-04, B10-PT-06, B10-DRL-08, B10-PT-07, B10-DRL-07 |
| **Won't Fix** | 2 | B10-DRL-02, B10-DRL-03 |
| **Total** | 18 | |

---

## Cross-Review Verdicts

### P1 Issues

| Ticket | Original | Verdict | Final | Reasoning |
|--------|----------|---------|-------|-----------|
| B10-DRL-01 | P1 | 2/3 REFINE | **P2** | Valid but sync assertion is easy fix |
| B10-PT-01 | P1 | Mixed | **P2** | Docstring already documents; middle ground |
| B10-PT-02 | P1 | 2/3 ENDORSE | **P1** | Real risk: post-compile .to() silently breaks |

### P2 Issues

| Ticket | Original | Verdict | Final | Reasoning |
|--------|----------|---------|-------|-----------|
| B10-CR-01 | P2 | UNANIMOUS REFINE | **P3** | Negligible fragmentation for typical env counts |
| B10-CR-02 | P2 | UNANIMOUS ENDORSE | **P2** | Valid optimization; already has TODO |
| B10-DRL-02 | P2 | 2/3 OBJECT | **Won't Fix** | Docstring already clear |
| B10-DRL-03 | P2 | 2/3 OBJECT | **Won't Fix** | Implementation is correct, not redundant |
| B10-PT-03 | P2 | 2/3 ENDORSE | **P2** | Fragile _orig_mod pattern |

### P3 Issues

| Ticket | Original | Verdict | Final | Reasoning |
|--------|----------|---------|-------|-----------|
| B10-CR-03 | P3 | 2/3 ENDORSE | **P3** | Missing protocol properties in validation |
| B10-DRL-04 | P3 | UPGRADE | **P2** | No Legacy Code Policy violation |
| B10-PT-04 | P3 | UNANIMOUS REFINE | **P4** | Debug-only, already opt-in |
| B10-DRL-05 | P3 | UPGRADE | **P2** | No Legacy Code Policy violation |
| B10-DRL-06 | P3 | UNANIMOUS ENDORSE | **P3** | Compile failure semantics undocumented |
| B10-CR-04 | P3 | UNANIMOUS REFINE | **P4** | Feature request, not bug |
| B10-PT-05 | P3 | 2/3 ENDORSE | **P3** | CPU sync on validation (toggle exists) |
| B10-PT-06 | P3 | 2/3 REFINE | **P4** | forward() not hot path for this class |
| B10-DRL-08 | P3 | UNANIMOUS REFINE | **P4** | Well documented intentional design |

### P4 Issues

| Ticket | Original | Verdict | Final | Reasoning |
|--------|----------|---------|-------|-----------|
| B10-PT-07 | P4 | Mixed | **P4** | Likely circular import avoidance |
| B10-DRL-07 | P4 | 2/3 ENDORSE | **P4** | Ambiguous docstring |

---

## Key Findings

### Policy Violations (Elevated to P2)

1. **B10-DRL-04: Deprecated num_slots parameter**
   - `factory.py` still accepts `num_slots=4` despite being marked deprecated
   - Per No Legacy Code Policy: "DELETE THE OLD CODE COMPLETELY"
   - Fix: Remove parameter, require `slot_config`

2. **B10-DRL-05: Unused dropout parameter**
   - `lstm_bundle.py` accepts `dropout` documented as "currently unused"
   - Per No Legacy Code Policy: dead code must be removed
   - Fix: Delete parameter entirely

### Critical Issue (P1)

3. **B10-PT-02: No enforcement against .to(device) after compile**
   - Post-compile device move silently invalidates compilation
   - Fix: Override `to()` to error if `is_compiled`

### Notable Won't Fix (False Positives)

4. **B10-DRL-03: MaskedCategorical entropy "redundancy"**
   - The custom entropy computation is **required** for masked distributions
   - Using `Categorical.entropy()` would include masked actions incorrectly

5. **B10-DRL-02: get_value() inference_mode documentation**
   - Existing docstring already clearly explains the inference-mode semantics
   - Additional warnings would be noise

---

## Actionable Items by Priority

### P1 (Critical)
- [ ] **B10-PT-02**: Add `.to()` override that errors if `is_compiled`

### P2 (Important)
- [ ] **B10-DRL-01**: Add module-level sync assertion for `_BLUEPRINT_TO_INDEX`
- [ ] **B10-PT-01**: Consider adding hidden state inference-mode marker
- [ ] **B10-CR-02**: Vectorize per-slot feature extraction (tracked by TODO)
- [ ] **B10-PT-03**: Track compilation state explicitly instead of `_orig_mod`
- [ ] **B10-DRL-04**: Remove deprecated `num_slots` parameter
- [ ] **B10-DRL-05**: Remove unused `dropout` parameter

### P3 (Moderate)
- [ ] **B10-CR-03**: Add missing protocol properties to registry validation
- [ ] **B10-DRL-06**: Document compile failure semantics in protocol
- [ ] **B10-PT-05**: Document validation performance trade-off better
- [ ] **B10-CR-01**: Add TODO for batch mask pre-allocation

### P4 (Low Priority)
- [ ] **B10-PT-04**: Consider if/raise instead of assert (optional)
- [ ] **B10-CR-04**: Add unregister_policy() for dev ergonomics (optional)
- [ ] **B10-PT-06**: Move expand_mask to module level (optional)
- [ ] **B10-DRL-08**: Consider renaming HeuristicPolicyBundle (optional)
- [ ] **B10-PT-07**: Document why import is inside function (optional)
- [ ] **B10-DRL-07**: Clarify hidden_dim return value for non-recurrent

---

## Architecture Observations

### Strengths
- **Protocol-based design** cleanly avoids nn.Module MRO conflicts
- **Action masking** correctly derives from single source of truth (VALID_TRANSITIONS)
- **Normalized entropy** in MaskedCategorical provides fair exploration incentives
- **Comprehensive test coverage** (117 tests, all passing)

### Integration Risks (Low)
- Blueprint mapping sync (mitigated by adding assertion)
- Feature dimension consistency (auto-computed from slot_config)
- LSTM hidden state gradient tracking (well-documented)

---

## Cross-Review Statistics

- **Total tickets reviewed:** 19
- **False positives caught:** 2 (10.5%)
- **Severity adjustments:** 9 tickets (47%)
  - Upgrades: 2 (P3→P2 for policy violations)
  - Downgrades: 7 (various P1→P2, P2→P3, P3→P4)
- **Unanimous verdicts:** 7 (37%)

---

*Report generated 2025-12-28*
