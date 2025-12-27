# Finding Ticket: Untyped Telemetry Payload for GOVERNOR_ROLLBACK (DRL Perspective)

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B1-DRL-06` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 1 |
| **Agent** | `drl` |
| **Domain** | `tolaria` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/tolaria/governor.py` |
| **Line(s)** | `251-264` |
| **Function/Class** | `TolariaGovernor.execute_rollback()` |

---

## Summary

**One-line summary:** GOVERNOR_ROLLBACK telemetry uses untyped dict with `type: ignore`, creating contract ambiguity for downstream consumers.

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

The telemetry payload is an untyped dictionary, requiring `# type: ignore[arg-type]` to suppress type errors.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/tolaria/governor.py:251-264

hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.GOVERNOR_ROLLBACK,
    ...
    data={  # type: ignore[arg-type]
        "env_id": env_id,
        "device": str(device),
        ...
    },
))
```

### Why This Matters

From an RL perspective:
- Telemetry consumers (dashboards, analytics) can't rely on typed contracts
- Makes it harder to build reliable monitoring for governor rollback patterns
- Could miss important debugging information if fields are spelled wrong

---

## Recommended Fix

See `B1-CR-01` for the recommended typed payload implementation.

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B1-CR-01` | `duplicate` | Same issue from code review perspective with full fix details |

---

## Cross-Review

| Agent | Verdict | Evaluation |
|-------|---------|------------|
| **DRL** | NEUTRAL | Typed telemetry payloads are good engineering hygiene, but this is observability infrastructure, not training-loop code. RL training stability is unaffected by payload typing; the fix belongs in a code quality sweep, not an urgent RL correctness pass. |
| **PyTorch** | NEUTRAL | Telemetry typing is a code quality concern unrelated to PyTorch compilation, tensor operations, or CUDA correctness. Type safety for telemetry payloads is valuable for maintainability but has no impact on training performance or memory management. |
| **CodeReview** | ENDORSE | Untyped dict payloads with `type: ignore` undermine the project's typed contract philosophy and violate leyline conventions. Should be deduplicated with B1-CR-01 and resolved with a proper typed dataclass payload in leyline. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch1-drl.md`
**Section:** "T1-G-6"
