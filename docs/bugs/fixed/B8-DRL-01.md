# Finding Ticket: LSTM Hidden State Not Reset in reset_episode_state()

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-01` |
| **Severity** | `P1` |
| **Status** | `closed` |
| **Batch** | 8 |
| **Agent** | `drl` |
| **Domain** | `simic/training` |
| **Assignee** | |
| **Created** | 2025-12-27 |
| **Updated** | 2025-12-29 |

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
- [x] Dead code / unwired functionality
- [ ] API design / contract violation
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

- [x] ~~Add `self.lstm_hidden = None` to `reset_episode_state()`~~ N/A - field removed
- [x] Verify LSTM policy training still converges
- [x] ~~Add test for hidden state reset behavior~~ N/A - dead code removed

---

## Related Findings

- C8-16 in DRL report: Hidden state reset logic in main loop

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-11 - lstm_hidden not reset in reset_episode_state()"

**Mitigation Note:** The DRL specialist notes this is mitigated by batch-level environment recreation, but the API contract should still be fixed for correctness.

---

## Resolution

| Field | Value |
|-------|-------|
| **Fixed By** | Claude Code |
| **Fixed Date** | 2025-12-29 |
| **Resolution** | `closed - dead code removed` |

### Investigation Findings

Deep investigation revealed that **this was not a bug but dead code**:

1. **`ParallelEnvState.lstm_hidden`** was declared but **never used** in the codebase
2. The actual LSTM hidden state is managed via `batched_lstm_hidden` (a local variable in `vectorized.py`)
3. The batched implementation correctly resets hidden state on episode boundaries (lines 2977-2990)
4. There was **no credit leakage** in practice

### Evidence

```bash
# Searching for actual usage of the field
$ grep -r "env_state\.lstm_hidden\|state\.lstm_hidden" src/
# No matches found - field was never read or written
```

### Fix Applied

Per the project's **No Legacy Code Policy**, the dead code was removed rather than "fixed":

1. **Removed** `lstm_hidden` field from `ParallelEnvState` (lines 75-79)
2. **Removed** `tests/simic/test_recurrent_vectorized.py` (tested dead code)
3. **Updated** module docstring to remove LSTM reference

### Architectural Note

The `lstm_hidden` field represented an earlier per-environment hidden state design that was superseded by a more efficient batched approach (`batched_lstm_hidden`). The batched implementation:
- Avoids per-step slice/cat overhead
- Correctly resets individual env hidden state on `done=True`
- Follows standard LSTM-PPO practice
