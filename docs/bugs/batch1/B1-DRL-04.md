# Finding Ticket: Hardcoded Random Guess Loss Assumes CIFAR-10

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B1-DRL-04` |
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
| **Line(s)** | `77-81` |
| **Function/Class** | `TolariaGovernor.__init__()` |

---

## Summary

**One-line summary:** Default `random_guess_loss = math.log(10)` assumes CIFAR-10 (10 classes) but Governor is used for other tasks.

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

The lobotomy detection uses `random_guess_loss` to detect when the model outputs uniform probabilities (loss = ln(num_classes)). The default value is `math.log(10)` which is correct for CIFAR-10 but wrong for other tasks.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/tolaria/governor.py:77-81

def __init__(
    self,
    ...
    random_guess_loss: float = math.log(10),  # Assumes 10 classes
    ...
):
```

### Why This Matters

- TinyStories has different vocabulary size
- Future tasks may have different class counts
- Lobotomy detection could fail silently or false-positive on non-CIFAR tasks

---

## Recommended Fix

### Option A: Require explicit specification

```python
def __init__(
    self,
    ...
    random_guess_loss: float,  # No default, must be specified
    ...
):
```

### Option B: Derive from task metadata

```python
def __init__(
    self,
    ...
    num_classes: int | None = None,
    random_guess_loss: float | None = None,
    ...
):
    if random_guess_loss is None:
        if num_classes is None:
            raise ValueError("Must specify either random_guess_loss or num_classes")
        random_guess_loss = math.log(num_classes)
```

### Option C: Document the assumption clearly

```python
random_guess_loss: float = math.log(10),  # Default for CIFAR-10 (10 classes)
```

---

## Verification

### How to Verify the Fix

- [ ] Check: Verify TinyStories training uses correct random_guess_loss
- [ ] Unit test: Test lobotomy detection with different class counts

---

## Cross-Review

| Agent | Verdict | Evaluation |
|-------|---------|------------|
| **DRL** | ENDORSE | Hardcoded log(10) for lobotomy detection is a genuine correctness risk - TinyStories with different vocabulary would trigger false positives/negatives. Lobotomy detection is critical for RL training health; incorrect thresholds could mask catastrophic policy collapse or waste compute on false alarms. |
| **PyTorch** | NEUTRAL | Valid API design concern but entirely domain-specific (RL task configuration) with no PyTorch implications. The math.log(num_classes) calculation is scalar Python math, not tensor operations; this is a hyperparameter correctness issue for the RL team. |
| **CodeReview** | ENDORSE | Hardcoded CIFAR-10 assumption is a genuine API design flaw that silently produces incorrect lobotomy detection for other tasks. Option B (derive from num_classes) is the cleanest fix; requiring explicit specification prevents silent misconfiguration. |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch1-drl.md`
**Section:** "T1-G-4"
