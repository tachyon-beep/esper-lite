# Finding Ticket: Defensive .get() in AlphaController.from_dict() (DRL Perspective)

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-06` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/alpha_controller.py` |
| **Line(s)** | `147-155` |
| **Function/Class** | `AlphaController.from_dict()` |

---

## Summary

**One-line summary:** `from_dict()` uses `.get(key, default)` pattern which violates the "no bug-hiding patterns" policy and could silently corrupt training state.

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
- [ ] Documentation / naming
- [x] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

`AlphaController.from_dict()` uses `.get(key, default)` for checkpoint deserialization. Per CLAUDE.md, this is a prohibited defensive programming pattern that hides bugs rather than failing fast.

### DRL Impact

If a checkpoint is missing required fields:
1. Silent substitution of default values (e.g., `alpha=0.0`)
2. Training resumes with wrong alpha state
3. Policy sees inconsistent observations
4. Credit assignment corrupted

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/alpha_controller.py:147-155

return cls(
    alpha=float(data.get("alpha", 0.0)),  # Should fail if missing
    alpha_start=float(data.get("alpha_start", 0.0)),
    alpha_target=float(data.get("alpha_target", 1.0)),
    step=int(data.get("step", 0)),
    total_steps=int(data.get("total_steps", 1)),
    # ...
)
```

### Comparison

`SeedState.from_dict()` in slot.py correctly uses direct access with explicit error messages.

---

## Recommended Fix

Align with `SeedState.from_dict()` pattern:

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "AlphaController":
    """Deserialize from checkpoint dict.

    Raises:
        KeyError: If required fields are missing
    """
    return cls(
        alpha=float(data["alpha"]),  # Fails fast on missing key
        alpha_start=float(data["alpha_start"]),
        alpha_target=float(data["alpha_target"]),
        step=int(data["step"]),
        total_steps=int(data["total_steps"]),
        mode=AlphaMode(data["mode"]),
        curve=AlphaCurve(data["curve"]),
    )
```

---

## Verification

### How to Verify the Fix

- [ ] Add test for malformed checkpoint deserialization
- [ ] Verify KeyError raised on missing fields
- [ ] Test checkpoint roundtrip with all fields

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-CR-04` | `duplicate` | Primary ticket from CodeReview perspective |
| `B2-PT-09` | `related` | PT noted this as "acceptable" for compatibility (disagree) |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - alpha_controller.py - B2-06

---

## Cross-Review: PyTorch Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | Checkpoint integrity is essential for reproducible training. Silent fallback to `alpha=0.0` corrupts policy gradient scaling and breaks resumable training. The fix aligns with PyTorch's `torch.load(strict=True)` philosophy of fail-fast on schema mismatch. |

---

## Cross-Review: Code Review Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | Confirmed violation of CLAUDE.md defensive programming policy at lines 147-155; `.get(key, default)` silently substitutes defaults for corrupted checkpoints. Direct key access with clear `KeyError` messaging aligns with `SeedState.from_dict()` pattern and fail-fast principles required by project standards.

---

## Cross-Review: DRL Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | Silent checkpoint corruption is catastrophic for RL training: alpha controls the host/seed blending ratio, and wrong alpha silently corrupts the observation distribution the policy sees. Loading a checkpoint with `alpha=0.0` (default) when actual alpha was `0.7` would cause immediate policy degradation with no clear diagnostic. This violates fail-fast principles and CLAUDE.md's defensive programming prohibition. |
