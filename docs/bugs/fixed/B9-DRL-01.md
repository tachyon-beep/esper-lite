# Finding Ticket: Blueprint Selection Fallback Doesn't Increment Index

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-01` |
| **Severity** | `P1` |
| **Status** | `closed` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/heuristic` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-29 |

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

- [x] Add test case for "all blueprints penalized" scenario
- [x] Verify blueprint diversity in multi-env runs with high penalty states
- [x] Document intended behavior in method docstring

---

## Related Findings

- B9-DRL-02: Missing None validation for improvement (related heuristic logic)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "H1 - Potential starvation in `_get_next_blueprint`"

---

## Resolution

| Field | Value |
|-------|-------|
| **Fixed By** | Claude Code |
| **Fixed Date** | 2025-12-29 |
| **Resolution** | `closed - exploration collapse fixed` |

### Investigation Findings

Deep investigation confirmed the bug was real - **Silent Exploration Collapse**:

1. The fallback used `min()` which always returned the same blueprint (lowest penalty)
2. This caused all parallel environments to synchronously select identical architectures
3. Exploration diversity was lost when all blueprints were penalized

### Fix Applied

Implemented **Option 3** from the recommended fix (index-based selection from sorted list):

```python
def _get_next_blueprint(self) -> str:
    blueprints = self.config.blueprint_rotation
    n = len(blueprints)
    start_idx = self._blueprint_index  # Remember starting position

    # Find blueprint below penalty threshold
    for _ in range(n):
        blueprint_id = blueprints[self._blueprint_index % n]
        self._blueprint_index += 1
        penalty = self._blueprint_penalties.get(blueprint_id, 0.0)
        if penalty < self.config.blueprint_penalty_threshold:
            return blueprint_id

    # All penalized - rotate through penalty-sorted list for diversity
    # Tiebreak by name for deterministic ordering when penalties are equal
    sorted_by_penalty = sorted(
        blueprints,
        key=lambda bp: (self._blueprint_penalties.get(bp, 0.0), bp)
    )
    # Advance index by 1 for next call (loop advanced by n, rewind to start+1)
    self._blueprint_index = start_idx + 1
    return sorted_by_penalty[start_idx % n]
```

### Key Changes

1. **Remember `start_idx`** before the loop to track original position
2. **Sort by penalty** with alphabetical tiebreaker for determinism
3. **Reset index to `start_idx + 1`** (not `start_idx + n`) to maintain single-step rotation
4. **Select from sorted list** using `start_idx % n` for round-robin through penalty order

### Test Added

`test_all_penalized_rotates_through_sorted` in `tests/tamiyo/test_heuristic_unit.py`:
- Verifies 6 consecutive calls return blueprints in penalty-sorted order
- Confirms wraparound after 3 calls maintains same order
- First blueprint is always the lowest-penalty one

### Reviews

- **DRL Specialist**: Approved - maintains exploration-exploitation balance
- **Code Reviewer**: Approved - logic correct, added tiebreaker for equal penalties
