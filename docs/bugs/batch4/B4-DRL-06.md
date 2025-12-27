# Finding Ticket: Missing Hidden State Contract Comment

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B4-DRL-06` |
| **Severity** | `P3` |
| **Status** | `open` |
| **Batch** | 4 |
| **Agent** | `drl` |
| **Domain** | `simic` |
| **Assignee** | |
| **Created** | 2024-12-27 |
| **Updated** | 2024-12-27 |

---

## Location

| Field | Value |
|-------|-------|
| **File(s)** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Line(s)** | `571-574` |
| **Function/Class** | `PPOAgent.update()` |

---

## Summary

**One-line summary:** Hidden state passed to `evaluate_actions` is from rollout buffer (inference-mode, detached) - this contract should be documented.

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
- [x] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

The code passes stored hidden states to `evaluate_actions`:

```python
# Lines 571-574
hidden=(
    data["initial_hidden_h"],
    data["initial_hidden_c"]
),
```

These tensors are:
1. From the rollout buffer (stored during action collection)
2. Created in inference mode (no gradients tracked)
3. Explicitly `.detach()`ed in rollout_buffer.py:342-343

The question: are these tensors suitable for gradient computation? Yes, but the reasoning is subtle:

- We don't backprop THROUGH the initial hidden state
- The network reconstructs LSTM evolution during the forward pass
- The LSTM produces new hidden states that ARE gradient-compatible

This contract should be documented.

---

## Recommended Fix

Add a clarifying comment:

```python
# NOTE: initial_hidden_h/c are detached tensors from rollout collection.
# This is CORRECT for recurrent PPO:
# 1. We use them as starting points for LSTM reconstruction
# 2. The LSTM forward pass produces new, gradient-enabled hidden states
# 3. BPTT happens within the reconstructed sequence, not through initial_hidden
# See rollout_buffer.py:342-343 for detach() and lstm_bundle.py docstring.
hidden=(
    data["initial_hidden_h"],
    data["initial_hidden_c"]
),
```

---

## Verification

### How to Verify the Fix

- [ ] Add clarifying comment
- [ ] No functional change needed
- [ ] Consider adding a test that verifies gradients flow correctly

---

## Related Findings

None.

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch4-drl.md`
**Section:** "P3 (Code Quality)" (P-9)

---

## Cross-Review: PyTorch Specialist

| Verdict | ENDORSE |
|---------|---------|

**Evaluation:** The hidden state detach semantics are correct - LSTM initial hidden states only need to seed the forward pass, and gradients flow through the reconstructed sequence. Documenting this contract prevents future developers from incorrectly "fixing" it by removing the detach().

---

## Cross-Review: DRL Specialist

| Verdict | ENDORSE |
|---------|---------|

**Evaluation:** The hidden state detachment is a subtle but critical contract in recurrent PPO. Developers unfamiliar with BPTT may incorrectly assume gradients should flow through stored hidden states, when in fact the LSTM reconstructs temporal dependencies during the forward pass. A docstring here prevents future debugging confusion.

---

## Cross-Review: Code Review Specialist

| Verdict | ENDORSE |
|---------|---------|

**Evaluation:** The subtle contract between inference-mode collection and training-mode evaluation deserves documentation; the proposed comment is accurate and prevents well-intentioned but incorrect "fixes".
