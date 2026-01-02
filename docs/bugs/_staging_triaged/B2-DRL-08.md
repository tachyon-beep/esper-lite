# Finding Ticket: get_alpha() Semantic Inconsistency

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-08` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/blending.py` |
| **Line(s)** | `167-180` |
| **Function/Class** | `GatedBlend.get_alpha()`, `GatedBlend.get_alpha_for_blend()` |

---

## Summary

**One-line summary:** `get_alpha(step)` returns blending progress while `get_alpha_for_blend(x)` returns learned gate output - same name, different semantics.

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

For `GatedBlend`:
- `get_alpha(step)` returns `step / total_steps` (blending progress, scalar in [0, 1])
- `get_alpha_for_blend(x)` returns the learned gate output (per-sample tensor)

These are conceptually different quantities with the same "alpha" name in different methods. Could confuse callers.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blending.py:167-180

def get_alpha(self, step: int | None = None) -> float:
    """Returns blending progress (step / total_steps)."""
    current = step if step is not None else self._current_step
    return min(1.0, current / self._total_steps)

def get_alpha_for_blend(self, x: torch.Tensor, ...) -> torch.Tensor:
    """Returns per-sample gate output."""
    pooled = self._pool_features(x)
    return torch.sigmoid(self.gate(pooled))
```

### DRL Impact

If policy code calls the wrong method, it observes the wrong "alpha":
- `get_alpha()` → controllable progress metric
- `get_alpha_for_blend()` → learned gate decisions (not directly controllable)

---

## Recommended Fix

### Option A: Rename for clarity

```python
def get_blending_progress(self, step: int | None = None) -> float:
    """Returns blending progress as fraction of total steps."""

def get_gate_output(self, x: torch.Tensor, ...) -> torch.Tensor:
    """Returns learned per-sample gate decisions."""
```

### Option B: Document distinction clearly

Add clear docstrings explaining when each is appropriate.

---

## Verification

### How to Verify the Fix

- [ ] Audit all callers to verify correct method used
- [ ] Add clearer docstrings

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - blending.py - B2-08

---

## Cross-Review: PyTorch Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | The semantic confusion is real: `get_alpha()` returns a scalar for lifecycle tracking while `get_alpha_for_blend()` returns a per-sample tensor from the learned gate. Option A (rename to `get_blending_progress` / `get_gate_output`) is preferred as it eliminates ambiguity in nn.Module method semantics. |

---

## Cross-Review: Code Review Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | Valid API design concern: `get_alpha()` at line 167 returns schedule progress while `get_alpha_for_blend()` at line 182 returns learned gate output - conflating distinct semantics. Option A (rename to `get_blending_progress()` and `get_gate_output()`) provides immediate clarity; current naming risks incorrect method selection by callers.

---

## Cross-Review: DRL Specialist

| Verdict | **ENDORSE** |
|---------|-------------|
| **Evaluation** | The semantic overloading is confusing: `get_alpha(step)` returns time-based progress (for G3 gate lifecycle tracking), while `get_alpha_for_blend(x)` returns learned per-sample gates (actual blending weights). Callers must understand this distinction or risk mixing schedule progress with gate outputs. Rename to `get_blending_progress()` and `get_gate_output()` for clarity. |
