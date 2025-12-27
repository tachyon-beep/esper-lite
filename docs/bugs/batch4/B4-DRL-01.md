# Finding Ticket: Causal Mask Logic Duplicated Between advantages.py and ppo.py

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B4-DRL-01` |
| **Severity** | `P1` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/agent/advantages.py`, `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Line(s)** | advantages.py:60-94, ppo.py:616-629 |
| **Function/Class** | `compute_per_head_advantages()`, `PPOAgent.update()` |

---

## Summary

**One-line summary:** The causal masking logic for per-head advantage computation is duplicated in two files - if either changes, gradient signals will become inconsistent.

**Category:**
- [x] Correctness bug
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

Two locations compute the same causal masks:

1. `advantages.py:60-94` - `compute_per_head_advantages()` uses masks for advantage computation
2. `ppo.py:616-629` - `PPOAgent.update()` uses masks for entropy weighting

Both must use IDENTICAL masks or:
- Advantage attribution won't match entropy bonus attribution
- Policy gradients will be inconsistent across heads

### Code Evidence

```python
# advantages.py:60-68
is_wait = op_actions == LifecycleOp.WAIT
is_germinate = op_actions == LifecycleOp.GERMINATE
is_set_alpha = op_actions == LifecycleOp.SET_ALPHA_TARGET
is_prune = op_actions == LifecycleOp.PRUNE

slot_mask = ~is_wait
blueprint_mask = is_germinate
style_mask = is_germinate | is_set_alpha
```

```python
# ppo.py:616-629 (similar but separate implementation)
is_wait = op_actions == LifecycleOp.WAIT
is_germinate = op_actions == LifecycleOp.GERMINATE
# ... duplicated logic
```

### Why This Matters

This is a DRY violation with **correctness implications**. If the causal structure is updated (e.g., adding a new operation or changing which heads are relevant), both files must be updated in lockstep.

---

## Recommended Fix

Extract causal mask computation to a single source of truth:

```python
# In advantages.py
def compute_causal_masks(op_actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute causal relevance masks for each action head.

    Returns a dict mapping head names to boolean masks indicating
    whether that head is causally relevant for the given operations.
    """
    is_wait = op_actions == LifecycleOp.WAIT
    is_germinate = op_actions == LifecycleOp.GERMINATE
    is_set_alpha = op_actions == LifecycleOp.SET_ALPHA_TARGET
    is_prune = op_actions == LifecycleOp.PRUNE

    return {
        "op": torch.ones_like(is_wait),
        "slot": ~is_wait,
        "blueprint": is_germinate,
        "style": is_germinate | is_set_alpha,
        "tempo": is_germinate,
        "alpha_target": is_set_alpha | is_germinate,
        "alpha_speed": is_set_alpha | is_prune,
        "alpha_curve": is_set_alpha | is_prune,
    }

# Then import and use in ppo.py:
from esper.simic.agent.advantages import compute_causal_masks
```

---

## Verification

### How to Verify the Fix

- [ ] Extract `compute_causal_masks()` to advantages.py
- [ ] Import and use in both advantages.py and ppo.py
- [ ] Verify all tests still pass
- [ ] Add test ensuring mask dictionaries are identical

---

## Related Findings

- B4-CR-01: Missing SET_ALPHA_TARGET test (would catch mask inconsistencies)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch4-drl.md`
**Section:** "P1 (Correctness Bugs)" (P-1)

---

## Cross-Review (DRL Specialist)

| Verdict | Reviewer | Date |
|---------|----------|------|
| **ENDORSE** | DRL Agent | 2024-12-27 |

This duplication is a training correctness landmine: advantages.py masks determine credit assignment, while ppo.py masks determine entropy bonus weighting (H3/H6 fixes).
If one is updated and the other forgotten, the policy receives conflicting gradient signals - entropy encouraging exploration on heads that receive zero advantage, causing optimization instability.

---

## Cross-Review (Code Review Specialist)

| Verdict | Reviewer | Date |
|---------|----------|------|
| **ENDORSE** | Code Review Specialist | 2024-12-27 |

Verified verbatim duplication: 8 mask definitions in advantages.py:60-94 are replicated in ppo.py:620-629 with identical boolean logic.
The proposed `compute_causal_masks()` extraction follows project conventions and should be placed in advantages.py per CLAUDE.md leyline guidance for shared contracts.

---

## Cross-Review (PyTorch Specialist)

| Verdict | Reviewer | Date |
|---------|----------|------|
| **ENDORSE** | PyTorch Agent | 2024-12-27 |

Confirmed duplication at advantages.py:60-94 and ppo.py:616-629. Extraction to `compute_causal_masks()` is torch.compile-friendly (pure tensor comparison ops, no graph breaks).
Single source of truth prevents inconsistent gradient attribution between advantage masking and entropy weighting. The refactor has zero performance cost under TorchInductor.
