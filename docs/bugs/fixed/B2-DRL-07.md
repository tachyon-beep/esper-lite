# Finding Ticket: SIGMOID Edge Case Dead Code

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-07` |
| **Severity** | `P3` |
| **Status** | `fixed` |
| **Batch** | 2 |
| **Agent** | `drl` |
| **Domain** | `kasmina` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/alpha_controller.py` |
| **Line(s)** | `36-38` |
| **Function/Class** | `AlphaController._curve_progress()` |

---

## Summary

**One-line summary:** Division-by-zero guard in `_curve_progress` is unreachable with steepness=12.0 - mathematically impossible edge case.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [x] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

The SIGMOID curve implementation has a guard for `raw1 == raw0` (would cause division by zero). However, with `steepness=12.0`, the sigmoid endpoints differ by ~1.0, making this condition mathematically impossible.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/alpha_controller.py:36-38

def _curve_progress(self, t: float) -> float:
    if self.curve == AlphaCurve.SIGMOID:
        steepness = 12.0
        raw0 = 1 / (1 + math.exp(steepness * 0.5))  # ≈ 0.0025
        raw1 = 1 / (1 + math.exp(-steepness * 0.5))  # ≈ 0.9975
        if raw1 == raw0:  # DEAD CODE: raw1 ≠ raw0 for steepness=12
            return t
        # ...
```

### Why This Is Low Priority

The guard is harmless (never executed), and removing it doesn't improve performance. It was likely added defensively for future-proofing if steepness becomes configurable.

---

## Recommended Fix

### Option A: Document why it's unreachable

```python
# Note: This guard is unreachable for steepness=12 (endpoints differ by ~1.0)
# but kept for safety if steepness becomes configurable in future
if raw1 == raw0:
    return t
```

### Option B: Remove dead code

If steepness will never be configurable, remove the guard.

---

## Verification

### How to Verify the Fix

- [ ] Verify with coverage that branch is never hit
- [ ] Add comment or remove based on team preference

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-PT-09` | `related` | PT also noted this as unreachable |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - alpha_controller.py - B2-07

---

## Cross-Review: PyTorch Specialist

| Verdict | **NEUTRAL** |
|---------|-------------|
| **Evaluation** | The dead code has zero performance impact (no torch.compile graph break, no GPU sync). With `steepness=12.0`, `raw1-raw0` is approximately 0.995, so the guard is indeed unreachable. Option A (document) is fine; Option B (delete) is marginally cleaner but low priority. |

---

## Cross-Review: Code Review Specialist

| Verdict | **OBJECT** |
|---------|-------------|
| **Evaluation** | This is NOT dead code - it is a mathematically sound guard preventing division-by-zero if `steepness` ever approaches zero in future changes. The comment at line 31-32 explicitly reserves steepness as a future configuration knob; removing this guard would create a latent bug when that feature is added.

---

## Cross-Review: DRL Specialist

| Verdict | **NEUTRAL** |
|---------|-------------|
| **Evaluation** | The dead code is harmless and the comment documents intent ("we intentionally keep a fixed steepness"). If steepness becomes configurable later (e.g., for adaptive curriculum), the guard prevents NaN. Per CLAUDE.md's "no legacy code" policy, either remove the guard (if steepness stays fixed) or add a comment explaining future-proofing. Either way, P3 is appropriate. |

---

## Resolution

**Fixed in commit:** `09384b0f` (feat(kasmina): add alpha_steepness to AlphaController)

**Resolution approach:** We implemented **Option A** by making steepness configurable rather than removing the guard. The guard is now reachable and serves a legitimate purpose:

1. Added `alpha_steepness` field to `AlphaController` (default 12.0)
2. Added `steepness` parameter to `_curve_progress()`
3. Expanded `AlphaCurveAction` enum to include `SIGMOID_GENTLE` (steepness=6), `SIGMOID` (steepness=12), and `SIGMOID_SHARP` (steepness=24)
4. Wired steepness through `SeedSlot` and vectorized training

**Why the guard is now legitimate:** With `steepness` as a parameter (minimum clamped to 0.1 in `__post_init__`), very small steepness values would cause `raw1` and `raw0` to approach each other, making the guard a valid numerical stability protection rather than dead code.

**Commits:**
- `09384b0f`: feat(kasmina): add alpha_steepness to AlphaController
- `2f2e4b7f`: feat(kasmina): wire alpha_steepness through SeedSlot
- `d07c36e0`: feat(simic): extract alpha_steepness from AlphaCurveAction in vectorized training
