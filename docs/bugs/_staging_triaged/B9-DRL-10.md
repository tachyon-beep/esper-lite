# Finding Ticket: env_id Conditional Silently Skips Telemetry

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B9-DRL-10` |
| **Severity** | `P4` |
| **Status** | `open` |
| **Batch** | 9 |
| **Agent** | `drl` |
| **Domain** | `tamiyo/tracker` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tamiyo/tracker.py` |
| **Line(s)** | `145-161` |
| **Function/Class** | `SignalTracker._update_stabilization()` |

---

## Summary

**One-line summary:** TAMIYO_INITIATED telemetry is only emitted when env_id is not None, silently skipping in single-env tests.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [x] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```python
# Lines 145-161
if env_id is not None:
    emit(TamiyoInitiatedPayload(
        env_id=env_id,
        epoch=epoch,
        ...
    ))
```

The TAMIYO_INITIATED event is only emitted when `env_id is not None`. This is correct for multi-env scenarios where each env needs an identifier.

However, in single-env testing or local development where env_id might not be set:
1. Telemetry is silently skipped
2. No indication that telemetry was suppressed
3. Test coverage of telemetry emission is reduced

### Impact

- **Silent skip**: No warning when telemetry is not emitted
- **Testing gaps**: Single-env tests don't verify telemetry
- **Debugging difficulty**: Missing telemetry hard to diagnose

---

## Recommended Fix

Either:

1. **Emit with sentinel** for the "no env" case:
```python
effective_env_id = env_id if env_id is not None else -1
emit(TamiyoInitiatedPayload(
    env_id=effective_env_id,
    ...
))
```

2. **Add warning** when telemetry skipped:
```python
if env_id is None:
    _logger.debug("Skipping TAMIYO_INITIATED telemetry: env_id not set")
else:
    emit(TamiyoInitiatedPayload(...))
```

3. **Require env_id** (strict approach):
```python
def __init__(self, env_id: int):  # Required, not optional
    self._env_id = env_id
```

---

## Verification

### How to Verify the Fix

- [ ] Decide on approach (sentinel, warning, or required)
- [ ] Implement chosen approach
- [ ] Add test for single-env telemetry behavior

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch9-drl.md`
**Section:** "T4 - env_id conditional telemetry"
