# Finding Ticket: Silent Cap at 100 Permutations for Shapley

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-04` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 5 |
| **Agent** | `drl` |
| **Domain** | `simic/attribution` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/attribution/counterfactual.py` |
| **Line(s)** | `396` |
| **Function/Class** | `CounterfactualEngine._generate_shapley_configs()` |

---

## Summary

**One-line summary:** Shapley sampling caps at 100 permutations regardless of `shapley_samples` config, silently ignoring larger values.

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
n_perms = min(n_perms, 100)  # Cap for performance
```

If a user configures `shapley_samples=200`, only 100 permutations are actually used. This silent cap could surprise users expecting more samples for higher precision.

---

## Recommended Fix

Either:

1. **Document the cap:**
```python
# Maximum permutations capped at 100 for computational tractability.
# Beyond this, antithetic sampling provides diminishing returns.
n_perms = min(n_perms, 100)
```

2. **Warn when cap is applied:**
```python
if n_perms > 100:
    logger.warning(
        f"Shapley permutations capped at 100 (requested {n_perms})"
    )
    n_perms = 100
```

3. **Make it configurable in CounterfactualConfig:**
```python
max_permutations: int = 100  # New config field
```

---

## Verification

### How to Verify the Fix

- [ ] Add documentation or warning
- [ ] Consider making configurable
- [ ] No functional change needed

---

## Related Findings

None.

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Valid contract violation. The code shows `n_perms = min(100, n_samples)` at line 396, silently capping at 100 regardless of `shapley_samples` config. This violates principle of least surprise since users setting `shapley_samples=200` for higher precision will not get it. The warning approach (option 2) is preferred over silent documentation since it provides runtime feedback. Note: 100 permutations is actually reasonable for Shapley convergence (theoretical variance decreases as 1/n), but the silent override is the problem.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** Valid finding at line 396: `n_perms = min(100, n_samples)` silently caps user configuration. From a PyTorch perspective, 100 permutations is computationally reasonable (pure Python list operations, not GPU-bound), so the cap exists for API clarity not performance. The warning approach is preferred over silent capping. Additionally, consider: if the cap is truly necessary for computational tractability, it should be a documented constant in CounterfactualConfig with a validator, not a buried magic number.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Confirmed at line 396: `n_perms = min(100, n_samples)`. This silently ignores user-requested values above 100, which violates the principle of least surprise. The recommended fix (adding a warning log) is appropriate and low-risk. Making it configurable via `CounterfactualConfig.max_permutations` would be overkill for a reasonable performance guard.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "P3 - Code Quality/Maintainability" (ID 5.6)
