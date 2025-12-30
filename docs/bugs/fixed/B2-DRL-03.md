# Finding Ticket: fused_forward alpha_override Shape Assumption

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-03` |
| **Severity** | `P2` |
| **Status** | `closed` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/host.py` |
| **Line(s)** | `565-587` |
| **Function/Class** | `MorphogeneticModel.fused_forward()` |

---

## Summary

**One-line summary:** `fused_forward()` expects alpha_overrides of shape `[K*B, 1, 1, 1]` for CNN but doesn't validate shape or handle transformer topology.

**Category:**
- [x] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

The `fused_forward()` method accepts `alpha_overrides` for batch-mode alpha control. The expected shape is `[K*B, 1, 1, 1]` for CNN (4D for broadcasting over spatial dims). However:

1. No shape validation is performed
2. Transformer topology would need `[K*B, 1, 1]` (3D for broadcasting over sequence)
3. Silent shape mismatch could cause broadcasting errors or wrong results

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/host.py:565-587

def fused_forward(
    self,
    x: torch.Tensor,
    alpha_overrides: torch.Tensor | None = None,  # Expected: [K*B, 1, 1, 1] for CNN
) -> torch.Tensor:
    # ... no shape validation
    # Uses alpha_overrides directly in blend operations
```

### Impact

- Broadcasting errors could crash silently or produce garbage outputs
- Wrong alpha values applied to wrong samples
- Policy learning corrupted by misaligned alphas

---

## Recommended Fix

Add topology-aware shape validation:

```python
def fused_forward(self, x: torch.Tensor, alpha_overrides: torch.Tensor | None = None) -> torch.Tensor:
    if alpha_overrides is not None:
        expected_shape = self._get_alpha_override_shape(x.shape[0])
        if alpha_overrides.shape != expected_shape:
            raise ValueError(
                f"alpha_overrides shape {alpha_overrides.shape} doesn't match "
                f"expected {expected_shape} for {self.topology} topology"
            )
    # ...

def _get_alpha_override_shape(self, batch_size: int) -> tuple[int, ...]:
    k = len(self.seed_slots)
    if self.topology == "cnn":
        return (k * batch_size, 1, 1, 1)
    else:  # transformer
        return (k * batch_size, 1, 1)
```

---

## Verification

### How to Verify the Fix

- [x] Add test for shape validation error on mismatch
- [x] Test both CNN and Transformer topologies with fused_forward
- [x] Verify broadcasting is correct for both topologies

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-CR-02` | `related` | cached_property on mutable state in same file |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - host.py - B2-03

---

## Cross-Review: DRL Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `drl` |

**Evaluation:** Shape misalignment in batched alpha injection would cause silent policy corruption - wrong alphas applied to wrong samples destroys the MDP's action-outcome correspondence. Validation is essential for reproducible training.

---

## Cross-Review: Code Review Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `codereview` |

**Evaluation:** Shape mismatches are silent killers in tensor code. Proposed fix is solid: validate shape early, provide topology-aware expected shape in error message, fail fast. P2 priority is appropriate.

---

## Cross-Review: PyTorch Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `pytorch` |

**Evaluation:** Shape mismatches in alpha broadcasting can cause silent semantic errors (wrong samples get wrong alphas) without runtime exceptions when shapes happen to broadcast.
The proposed validation is essential - use `assert` for compile-time elimination or explicit `if` check with `raise ValueError` for production diagnostics.

---

## Resolution

**Status:** Fixed

**Fix:** Implemented topology-aware shape validation in `fused_forward()` and fixed caller site in `vectorized.py`.

**Changes:**
1. `src/esper/kasmina/host.py`:
   - Added `_get_expected_alpha_shape(batch_size)` helper returning `(B, 1, 1, 1)` for CNN, `(B, 1, 1)` for transformer
   - Added validation loop in `fused_forward()` with fail-fast assertions and clear error messages

2. `src/esper/simic/training/vectorized.py`:
   - Made `alpha_shape` topology-aware based on `task_spec.topology`
   - Updated comment and docstring to document both CNN and transformer shapes

3. `tests/kasmina/test_morphogenetic_model.py`:
   - Added `TestFusedForwardAlphaShapeValidation` class with 6 tests

**Verification:**
- mypy passes on all modified files
- 6/6 shape validation tests pass
- Assertions are torch.compile compatible (elided in optimized mode)

**Sign-off:** Approved by `pytorch-expert`

**Commits:** `45ddfdc7`
