# Finding Ticket: Blueprint Selection Fallback Doesn't Increment Index

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-01` |
| **Severity** | `P1` |
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
| **Line(s)** | `331` |
| **Function/Class** | `HeuristicTamiyo._get_next_blueprint()` |

---

## Summary

**One-line summary:** When all blueprints are penalized above threshold, fallback picks minimum-penalty blueprint but doesn't increment _blueprint_index.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [x] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
# Line 331 (approximate - in fallback path)
# When all blueprints are penalized above threshold:
return min(self._blueprint_penalties.keys(), key=lambda b: self._blueprint_penalties[b])
```

The fallback returns the minimum-penalty blueprint but does NOT increment `_blueprint_index`. This means:

1. Repeated calls in the same state will always return the same blueprint
2. The round-robin rotation mechanism is bypassed
3. In multi-env scenarios, all envs hitting this fallback get the same blueprint

### Impact

- **Determinism concerns**: Multi-env vectorized training may have correlated blueprint selections
- **Exploration reduction**: Less blueprint diversity when all are penalized
- **Potential bias**: One blueprint may dominate when penalties are high across the board

---

## Recommended Fix

Either:
1. Increment `_blueprint_index` even in fallback path, OR
2. Document this is intentional behavior (prefer least-penalized over rotation), OR
3. Use index-based selection from sorted list to maintain rotation semantics

```python
# Option 1: Increment in fallback
fallback_bp = min(penalized_blueprints, key=lambda b: self._blueprint_penalties[b])
self._blueprint_index += 1  # Maintain rotation state
return fallback_bp
```

---

## Verification

### How to Verify the Fix

- [ ] Add test case for "all blueprints penalized" scenario
- [ ] Verify blueprint diversity in multi-env runs with high penalty states
- [ ] Document intended behavior in method docstring

---

## Related Findings

- B9-DRL-02: Missing None validation for improvement (related heuristic logic)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "H1 - Potential starvation in `_get_next_blueprint`"
