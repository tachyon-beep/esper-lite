# Finding Ticket: MULTIPLY seed_input Default Creates Gradient Coupling

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-10` |
| **Severity** | `P3` |
| **Status** | `open` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blend_ops.py` |
| **Line(s)** | `59-94` |
| **Function/Class** | `blend_multiply()` |

---

## Summary

**One-line summary:** When `seed_input` is None, defaults to `host_features` which creates implicit gradient coupling - violates isolation semantics.

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

The `blend_multiply` function has an optional `seed_input` parameter that defaults to `host_features` when None. This creates an implicit dependency: gradients flow through `host_features` inside the tanh, violating the expected gradient isolation.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blend_ops.py:59-94

def blend_multiply(
    host_features: torch.Tensor,
    seed_modulation: torch.Tensor,
    alpha: torch.Tensor,
    seed_input: torch.Tensor | None = None,  # Defaults to host_features
) -> torch.Tensor:
    if seed_input is None:
        seed_input = host_features  # Gradient coupling!
    # ...
    multiplier = multiply_valve_multiplier(alpha, seed_input)
```

### DRL Impact

If a caller forgets to pass `seed_input`:
1. Gradients flow through host features into the seed modulation path
2. Seed learns from host gradients (breaks isolation)
3. Credit assignment becomes muddled

### Current Mitigation

`SeedSlot.forward()` consistently passes `seed_input` for MULTIPLY mode, so this doesn't manifest in practice.

---

## Recommended Fix

Make `seed_input` required with explicit opt-in for non-isolated mode:

```python
def blend_multiply(
    host_features: torch.Tensor,
    seed_modulation: torch.Tensor,
    alpha: torch.Tensor,
    seed_input: torch.Tensor,  # Now required
) -> torch.Tensor:
    """...

    Args:
        seed_input: Input to seed network. Pass host_features explicitly
                   if gradient coupling is intentional.
    """
```

---

## Verification

### How to Verify the Fix

- [ ] Update all callers to pass seed_input explicitly
- [ ] Add test for gradient isolation with explicit seed_input

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - blend_ops.py - B2-10

---

## Cross-Review: PyTorch Specialist

| Verdict | ENDORSE |
|---------|---------|

The gradient coupling concern is valid: `seed_input = host_features` creates a shared tensor reference, so gradients through `seed_features - seed_input` flow back into `host_features` via the subtraction.
However, current callers always pass explicit `seed_input`, and making the parameter required is a clean API hardening with no torch.compile implications.

---

## Cross-Review: DRL Specialist

| Verdict | NEUTRAL |
|---------|---------|

Finding is technically accurate but overstates severity. The docstring at lines 73-81 explicitly explains the gradient isolation contract and when to pass `seed_input`; the default exists for backward-compatible call sites that intentionally want coupled gradients.
Making the parameter required would break the documented "CNN isolation contract" pattern where callers *choose* isolation by passing `host_features.detach()`.

---

## Cross-Review: Code Review Specialist

| Verdict | NEUTRAL |
|---------|---------|

The default `seed_input=host_features` is intentional API ergonomics for external callers; all internal SeedSlot calls pass explicit `seed_input` (line 1997 of slot.py).
Making it required would break the public API for minimal gain since the "bug" never manifests in practice.
