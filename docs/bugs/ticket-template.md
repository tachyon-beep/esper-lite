# Finding Ticket Template

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B{batch}-{agent}-{sequence}` (e.g., `B3-PT-01`) |
| **Severity** | `P0` / `P1` / `P2` / `P3` / `P4` |
| **Status** | `open` / `investigating` / `fix-in-progress` / `in-review` / `closed` / `wont-fix` |
| **Batch** | {1-10} |
| **Agent** | `drl` / `pytorch` / `codereview` |
| **Domain** | `tolaria` / `kasmina` / `simic` / `tamiyo` |
| **Assignee** | |
| **Created** | YYYY-MM-DD |
| **Updated** | YYYY-MM-DD |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/...` |
| **Line(s)** | e.g., `320-325` or `multiple (see below)` |
| **Function/Class** | e.g., `TolariaGovernor.execute_rollback()` |

**Additional locations (if multi-file):**
- `path/to/file2.py:100-110` - description
- `path/to/file3.py:50` - description

---

## Summary

**One-line summary:** [Brief description of the issue]

**Category:** (check all that apply)
- [ ] Correctness bug
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

[Detailed explanation of the issue. Include:
- What the code currently does
- What it should do instead
- Why this is a problem]

### Code Evidence

```python
# Current problematic code (with file:line reference)
# path/to/file.py:123-130

def problematic_function():
    # ... the actual code ...
    pass
```

### Why This Matters

[Explain the impact:
- What could go wrong at runtime?
- What symptoms would a user see?
- How does this affect the RL training / policy learning?
- Is data at risk? Performance? Correctness?]

### Reproduction Steps (if applicable)

1. Step one
2. Step two
3. Observe: [what happens]
4. Expected: [what should happen]

---

## System Context

[Relevant background from the Esper architecture that helps understand the issue]

**Related concepts:**
- [e.g., "This affects gradient isolation during BLENDING stage"]
- [e.g., "The policy cannot observe this metric, creating a blind spot"]

**Relevant architectural principles:**
- [ ] Signal-to-Noise Hypothesis (sensors match capabilities)
- [ ] Rent Economy (complexity pays rent)
- [ ] Inverted Control Flow (GPU-first)
- [ ] Governor prevents catastrophe
- [ ] No defensive programming policy
- [ ] No legacy code policy

---

## Recommended Fix

### Approach

[What the reviewing agent recommended]

### Suggested Code Change

```python
# Suggested fix (pseudocode or actual)
# path/to/file.py:123-130

def fixed_function():
    # ... the corrected code ...
    pass
```

### Alternative Approaches (if any)

1. **Alternative A:** [description] - Pros: ... Cons: ...
2. **Alternative B:** [description] - Pros: ... Cons: ...

---

## Verification

### How to Verify the Fix

- [ ] Unit test: [describe test to add/modify]
- [ ] Integration test: [describe]
- [ ] Manual verification: [steps]
- [ ] Property-based test: [if applicable]

### Regression Risk

[What could break if the fix is implemented incorrectly?]

### Test Files to Update

- `tests/path/to/test_file.py` - add test for X
- `tests/path/to/other_test.py` - update assertion for Y

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B{x}-{agent}-{yy}` | `duplicate` / `related` / `blocks` / `blocked-by` | [brief note] |

---

## Investigation Notes

[Space for the assignee to document their investigation]

### Questions to Answer

- [ ] Question 1 from the finding?
- [ ] Question 2?

### Findings During Investigation

[Chronological notes as investigation proceeds]

**YYYY-MM-DD:** [Note]

---

## Resolution

### Final Fix Description

[What was actually done to fix the issue]

### Commits

- `abc1234` - [commit message]
- `def5678` - [commit message]

### Tests Added/Modified

- `tests/...` - [description]

### Verified By

- [ ] Automated tests pass
- [ ] Manual verification complete
- [ ] Code review approved

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch{N}-{agent}.md`
**Section:** [heading in the report]

### Additional Context

[Any other relevant information, links to documentation, external references, etc.]
