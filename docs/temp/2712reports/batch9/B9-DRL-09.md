# Finding Ticket: Hardcoded 5.0 Divisor in Confidence Calculations

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-09` |
| **Severity** | `P4` |
| **Status** | `open` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/heuristic` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/heuristic.py` |
| **Line(s)** | `188`, `276` |
| **Function/Class** | `HeuristicTamiyo._decide_germination()`, `_decide_seed_management()` |

---

## Summary

**One-line summary:** Magic number 5.0 used as divisor for confidence calculation in multiple places.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [x] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
# Line 188 (germination)
confidence = min(1.0, signals.metrics.plateau_epochs / 5.0)

# Line 276 (fossilization)
confidence = min(1.0, improvement / 5.0)
```

The magic number `5.0` appears in two places as a divisor for confidence calculation. This represents:
- 5 plateau epochs for full germination confidence
- 5% improvement for full fossilization confidence

### Impact

- **Magic number**: No semantic name explains why 5.0
- **Tuning difficulty**: Must search for literal 5.0 to change
- **Inconsistency risk**: Two places could diverge if one is updated

Note: Since confidence field is unused (B9-DRL-08), this is very low priority.

---

## Recommended Fix

Extract to config constants:

```python
# In HeuristicPolicyConfig
@dataclass
class HeuristicPolicyConfig:
    ...
    confidence_plateau_scale: float = 5.0  # Epochs for full germination confidence
    confidence_improvement_scale: float = 5.0  # % improvement for full fossilization confidence

# Usage
confidence = min(1.0, signals.metrics.plateau_epochs / self.config.confidence_plateau_scale)
confidence = min(1.0, improvement / self.config.confidence_improvement_scale)
```

Or define named constants if not configurable:

```python
# At module level
_PLATEAU_EPOCHS_FOR_FULL_CONFIDENCE = 5.0
_IMPROVEMENT_FOR_FULL_CONFIDENCE = 5.0
```

---

## Verification

### How to Verify the Fix

- [ ] Extract constants (config or module-level)
- [ ] Update both usage sites
- [ ] Consider if confidence is even needed (see B9-DRL-08)

---

## Related Findings

- B9-DRL-08: confidence field is unused (makes this lower priority)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "H4 - Magic number in confidence calculation"
