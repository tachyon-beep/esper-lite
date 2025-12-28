# Finding Ticket: GatedBlend Credit Assignment Confounding

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-01` |
| **Severity** | `P2` |
| **Status** | `closed` |
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
| **Line(s)** | `133-192` |
| **Function/Class** | `GatedBlend` |

---

## Summary

**One-line summary:** GatedBlend's learned gate introduces observation-action confounding - the gate learns alongside the policy, creating credit assignment ambiguity.

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

The gate network `self.gate` learns to modulate alpha per-sample. This means the effective blending is no longer under direct policy control. When the policy selects GATE mode, it's delegating per-sample decisions to a learned network that trains alongside the seed.

### Why This Matters for RL

**Credit assignment problem:** Did the fossilization succeed because:
1. The policy chose good timing (action deserves credit)?
2. The gate learned good sample weighting (gate deserves credit)?

The policy observes aggregate outcomes (accuracy, fossilization success) but can't distinguish its contribution from the gate's.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blending.py:133-192

class GatedBlend(BlendAlgorithm):
    """Per-sample gating for adaptive blending."""

    def __init__(self, channels: int, topology: str, ...):
        self.gate = nn.Sequential(  # Learned network
            nn.Linear(channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_alpha_for_blend(self, x: torch.Tensor, ...) -> torch.Tensor:
        pooled = self._pool_features(x)
        gate_logit = self.gate(pooled)  # Learned gating
        return torch.sigmoid(gate_logit)
```

### Current Mitigation

`SeedMetrics.current_alpha` reports `step / total_steps` for GatedBlend (line 179), not the gate's output. This is intentional - observations reflect controllable state.

---

## Recommended Fix

### Option A: Expose gate statistics as observations

Add gate summary statistics to SeedMetrics so policy can learn when gated blending is working:

```python
@dataclass
class SeedMetrics:
    current_alpha: float
    gate_mean: float | None = None  # Mean gate output this epoch
    gate_std: float | None = None   # Gate output variance
```

### Option B: Document the design trade-off

If the two-level control (policy controls amplitude, gate controls per-sample) is intentional, document clearly that GATE mode delegates sample-level decisions.

---

## Verification

### How to Verify the Fix

- [ ] Add test for gate statistics collection
- [ ] Verify Tamiyo can observe gate behavior
- [ ] Analyze if gate statistics improve policy learning

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-DRL-02` | `related` | GatedBlend topology mismatch |
| `B2-PT-05` | `related` | GatedBlend hidden_dim degenerate case |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - blending.py - B2-01

---

## Cross-Review: DRL Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `drl` |

**Evaluation:** Classic credit assignment confounding (Sutton & Barto Ch.17). Hierarchical control with learned inner-loop is valid but requires observable gate statistics; Option A exposing gate_mean/gate_std is the correct fix for policy learning.

---

## Cross-Review: Code Review Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `codereview` |

**Evaluation:** Valid architectural concern. The gate learning alongside policy creates unobservable confounding. Option A (expose gate stats) is cleaner; provides observability without changing two-level control design.

---

## Cross-Review: PyTorch Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | `pytorch` |

**Evaluation:** The gate network is correctly structured as an nn.Module submodule with proper registration - parameters will serialize and train correctly.
The credit assignment concern is valid RL design critique but has no PyTorch-level correctness implications; Option A (gate statistics) would require negligible compute overhead via `detach()`.

---

## Resolution

**Status:** Already Fixed (Option B)

**Evidence:** The design trade-off is already documented in `src/esper/kasmina/slot.py` lines 143-148:

```python
# Note on alpha semantics (DRL Expert review 2025-12-17):
# current_alpha represents "blending progress" (step/total_steps), not actual
# blend values. For GatedBlend, actual per-sample alpha is learned and input-
# dependent. The agent controls blending TIMELINE, not per-sample gates.
# This is intentional for credit assignment - observations should reflect
# controllable state, not emergent gate behavior.
```

**Why Option A (gate_mean/gate_std) is NOT needed:**
- `counterfactual_contribution` already provides causal attribution (real_acc - baseline_acc with alpha=0)
- The policy needs to know *what contribution resulted*, not *how the gate achieved it*
- Gate statistics would create observation noise without improving credit assignment

**Enhancement Applied:**
- Added "Credit Assignment Note" docstring to `GatedBlend` class cross-referencing the SeedMetrics design

**Sign-off:** Approved by `drl-expert`

**Commits:** `206841ef`
