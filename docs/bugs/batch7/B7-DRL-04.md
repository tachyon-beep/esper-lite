# Finding Ticket: compute_lstm_health() Never Called

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B7-DRL-04` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 7 |
| **Agent** | `drl` |
| **Domain** | `simic/telemetry` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/telemetry/lstm_health.py` |
| **Line(s)** | All |
| **Function/Class** | `compute_lstm_health()` |

---

## Summary

**One-line summary:** LSTM health monitoring is implemented but never called from training loop.

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
- [x] Legacy code policy violation

---

## Detailed Description

### What's Wrong

```bash
grep -r "compute_lstm_health" src/esper/ --include="*.py" | grep -v "telemetry"
# No results
```

`compute_lstm_health()` is:
1. Fully implemented (lines 70-120)
2. Exported in `__init__.py`
3. Never called from vectorized.py or any training code

**DRL Context:** Hidden state drift is a critical LSTM failure mode:
- Pascanu et al., 2013 documents vanishing/exploding gradient issues in RNNs
- Hidden states can drift to very large or very small magnitudes
- NaN/Inf can propagate through hidden states before appearing in loss

If the policy uses LSTM, this monitoring should be integrated.
If the policy doesn't use LSTM, this module is dead code.

---

## Recommended Fix

**Option 1 - Wire it up (if LSTM is used):**
```python
# In vectorized.py after LSTM forward pass:
if hasattr(policy, 'lstm') and policy.lstm is not None:
    lstm_health = compute_lstm_health(policy.lstm.hidden_state)
    if not lstm_health.is_healthy():
        _logger.warning("LSTM hidden state unhealthy: %s", lstm_health)
```

**Option 2 - Delete (if LSTM is not used):**
Per No Legacy Code Policy, remove unused code.

---

## Verification

### How to Verify the Fix

- [ ] Determine if policy uses LSTM
- [ ] If yes, wire up monitoring
- [ ] If no, delete lstm_health.py

---

## Related Findings

- B7-PT-05: Missing tests for LSTMHealthMetrics
- B7-DRL-01: GradientEMATracker also never used

---

## Cross-Review (DRL Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `REFINE` |
| **Reviewer** | DRL Specialist |

**Evaluation:** Policy IS LSTM-based (confirmed: ppo.py uses `policy_type="lstm"`). Hidden state drift is real failure mode - wire it up, don't delete.

---

## Cross-Review (PyTorch Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | PyTorch Specialist |

**Evaluation:** Implementation is well-optimized (M14 fix: single GPU-CPU sync). Worth keeping since policy uses LSTM.

---

## Cross-Review (Code Review Specialist)

| Field | Value |
|-------|-------|
| **Verdict** | `REFINE` |
| **Reviewer** | Code Review Specialist |

**Evaluation:** Function exported but never imported elsewhere. Per CLAUDE.md telemetry guidance: wire up (don't delete) since policy uses LSTM architecture.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch7-drl.md`
**Section:** "compute_lstm_health() is never called - dead code if LSTM not used"
