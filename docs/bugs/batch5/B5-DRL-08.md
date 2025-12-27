# Finding Ticket: Interaction Terms Capped at n<=3 Seeds

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B5-DRL-08` |
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
| **Line(s)** | N/A (design limitation) |
| **Function/Class** | `CounterfactualEngine.compute_interaction_terms()` |

---

## Summary

**One-line summary:** `compute_interaction_terms` returns empty for n>3 seeds, silently failing synergy/interference detection.

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

The `compute_interaction_terms` function returns an empty list for n>3 seeds. This is fine for complexity reasons (pairwise interactions grow as O(n^2)), but means synergy/interference detection fails silently when you have 4+ active seeds.

### Impact

With 4+ seeds:
- No warning that interaction terms are unavailable
- Callers may assume zero interactions when in fact they're just not computed
- Could lead to missing important seed synergies

---

## Recommended Fix

Either:

1. **Log a warning when skipping:**
```python
def compute_interaction_terms(self, matrix: CounterfactualMatrix) -> list[InteractionTerm]:
    if len(matrix.slot_ids) > 3:
        logger.debug(
            f"Skipping interaction terms for {len(matrix.slot_ids)} seeds "
            "(complexity cap at n=3)"
        )
        return []
```

2. **Document the limitation in docstring:**
```python
def compute_interaction_terms(self, matrix: CounterfactualMatrix) -> list[InteractionTerm]:
    """Compute pairwise interaction terms.

    Note:
        For computational tractability, interaction terms are only computed
        when n <= 3 seeds. For larger seed sets, returns empty list.
    """
```

---

## Verification

### How to Verify the Fix

- [ ] Add warning or documentation
- [ ] Consider raising error if caller expects interactions for n>3

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch5-drl.md`
**Section:** "RL-Specific Observations" (Interaction terms capped)

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Verified in `counterfactual.py:477-478`. The code does exactly `if n > 3: return {}` with no logging or warning. The docstring at line 469 mentions "Only valid for n <= 3 active seeds" but callers have no runtime visibility into this limitation. The recommendation to add `logger.debug()` is low-friction and would aid debugging when interaction terms are unexpectedly empty. The silent-failure pattern here could indeed cause confusion for operators wondering why synergy detection stopped working when a fourth seed was activated.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |
| **Date** | 2024-12-27 |

**Evaluation:** The n<=3 cap is a reasonable complexity bound (O(n^2) pairwise interactions from O(2^n) configs). The silent `return {}` at L478 is the issue: callers cannot distinguish "no interactions exist" from "interactions not computed due to complexity". Adding a debug log as proposed is minimal; alternatively, the function could return a sentinel object or raise when requested for n>3. From a torch.compile perspective, returning early with empty dict is fine (no graph break concerns).

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** The n<=3 cap is reasonable for full pairwise interactions (O(n^2) pairs, each requiring 2^n evaluations for exact computation). However, the silent empty-dict return is poor API design -- callers cannot distinguish "no interactions detected" from "interactions not computed due to complexity cap." The recommended debug log is minimal overhead. For n>3, consider computing only top-k strongest pairwise interactions using the ablation configs already generated (comparing f({i,j}) - f({i}) - f({j}) + f(empty) for the subset of pairs where all four configs happen to exist in the matrix).
