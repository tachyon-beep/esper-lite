# Finding Ticket: Missing advantage_mean/std in PPOUpdateMetrics

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B4-DRL-10` |
| **Severity** | `P4` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/agent/types.py` |
| **Line(s)** | `39-67` |
| **Function/Class** | `PPOUpdateMetrics` |

---

## Summary

**One-line summary:** `advantage_mean` and `advantage_std` are computed in ppo.py but not declared in PPOUpdateMetrics TypedDict.

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

In ppo.py (lines 532-536), advantage statistics are computed and stored:

```python
# These are computed...
advantage_mean = ...
advantage_std = ...

# And stored in metrics
epoch_metrics["advantage_mean"] = advantage_mean
epoch_metrics["advantage_std"] = advantage_std
```

But `PPOUpdateMetrics` TypedDict doesn't include these fields:

```python
class PPOUpdateMetrics(TypedDict, total=False):
    policy_loss: float
    value_loss: float
    entropy: float
    # ... advantage_mean and advantage_std are missing
```

---

## Recommended Fix

Add the missing fields to PPOUpdateMetrics:

```python
class PPOUpdateMetrics(TypedDict, total=False):
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    advantage_mean: float  # Add
    advantage_std: float   # Add
    # ...
```

---

## Verification

### How to Verify the Fix

- [ ] Add `advantage_mean` and `advantage_std` to PPOUpdateMetrics
- [ ] Run mypy to verify type consistency
- [ ] Verify tests still pass

---

## Related Findings

- B4-PT-10: HeadGradientNorms not exported (related TypedDict issue)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch4-drl.md`
**Section:** "P4 (Style/Minor)" (T-1)

---

## Cross-Review (DRL Specialist)

| Verdict | **ENDORSE** |
|---------|-------------|

**Evaluation:** Advantage mean/std are essential PPO health diagnostics - post-normalization mean should be near 0, std near 1. These fields are actively computed (ppo.py:527-536) and consumed by TUI status banners. Add to TypedDict for type safety and downstream tooling.

---

## Cross-Review (Code Review Specialist)

| Verdict | **ENDORSE** |
|---------|-------------|

**Evaluation:** Valid finding - ppo.py lines 532-536 store `advantage_mean`/`advantage_std` in metrics dict but `PPOUpdateMetrics` TypedDict lacks these fields. This breaks type contract integrity and allows mypy to miss errors. Add both fields to TypedDict.

---

## Cross-Review (PyTorch Specialist)

| Verdict | **ENDORSE** |
|---------|-------------|

**Evaluation:** Valid finding: ppo.py lines 527-536 compute `advantage_mean`/`advantage_std` via batched GPU ops (`torch.stack` + `.tolist()`) and store in metrics dict, but `PPOUpdateMetrics` TypedDict omits these fields. Add for type completeness - no runtime impact but aids mypy and IDE tooling.
