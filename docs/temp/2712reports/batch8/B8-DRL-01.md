# Finding Ticket: LSTM Hidden State Not Reset in reset_episode_state()

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-01` |
| **Severity** | `P1` |
| **Status** | `open` |
| **Batch** | 8 |
| **Agent** | `drl` |
| **Domain** | `simic/training` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/parallel_env_state.py` |
| **Line(s)** | `167-197` |
| **Function/Class** | `ParallelEnvState.reset_episode_state()` |

---

## Summary

**One-line summary:** `lstm_hidden` field not reset in `reset_episode_state()`, risking credit leakage across episodes.

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
# reset_episode_state() does NOT reset lstm_hidden to None
def reset_episode_state(self) -> None:
    """Reset episode-level state for reuse."""
    # ... resets accumulators, counters ...
    # BUT lstm_hidden is NOT reset
```

For episodic RL tasks, LSTM hidden state should be reset at episode boundaries. If hidden state from a previous episode persists, it can:
1. Cause temporal credit assignment corruption
2. Allow the policy to "remember" previous episode states
3. Bias early-episode actions based on late-episode information from prior runs

### Mitigation Analysis

The DRL specialist notes that this is **mitigated by batch-level recreation**:
- Each batch creates fresh environments with fresh models (line 1700)
- `batched_lstm_hidden` is initialized to None at line 1711 and created fresh on first use
- The concern only applies if episodes span batches AND `reset_episode_state()` is called

However, the API contract is misleading - a method called `reset_episode_state()` should reset ALL episode state, including LSTM hidden.

---

## Recommended Fix

Add explicit hidden state reset:

```python
def reset_episode_state(self) -> None:
    """Reset episode-level state for reuse."""
    # ... existing resets ...

    # Reset LSTM hidden state for new episode
    self.lstm_hidden = None
```

---

## Verification

### How to Verify the Fix

- [ ] Add `self.lstm_hidden = None` to `reset_episode_state()`
- [ ] Verify LSTM policy training still converges
- [ ] Add test for hidden state reset behavior

---

## Related Findings

- C8-16 in DRL report: Hidden state reset logic in main loop

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-11 - lstm_hidden not reset in reset_episode_state()"

**Mitigation Note:** The DRL specialist notes this is mitigated by batch-level environment recreation, but the API contract should still be fixed for correctness.
