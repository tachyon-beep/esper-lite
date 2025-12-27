# Finding Ticket: GatedBlend Topology Mismatch If Instance Reused

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-02` |
| **Severity** | `P2` |
| **Status** | `open` |
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
| **Line(s)** | `158-165` |
| **Function/Class** | `GatedBlend._pool_features()` |

---

## Summary

**One-line summary:** GatedBlend topology detection relies on construction-time `self.topology` - reusing instance across topologies produces wrong pooling.

**Category:**
- [x] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [ ] Numerical stability
- [ ] torch.compile compatibility
- [ ] Dead code / unwired functionality
- [ ] API design / contract violation
- [ ] Test coverage gap
- [ ] Documentation / naming
- [ ] Defensive programming violation
- [ ] Legacy code policy violation

---

## Detailed Description

### What's Wrong

`_pool_features()` branches on `self.topology` which is set at construction time. If the same GatedBlend instance is reused across topologies (e.g., in test fixtures or configuration errors), the pooling will be wrong.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/blending.py:158-165

def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
    if self.topology == "cnn":
        return x.mean(dim=[2, 3])  # CNN: pool spatial dims
    else:
        return x.mean(dim=1)  # Transformer: pool sequence dim
```

### Failure Scenario

1. Create `GatedBlend(channels=64, topology="cnn")`
2. Accidentally use same instance with Transformer input (shape `[B, T, C]`)
3. Pooling is wrong: `x.mean(dim=[2, 3])` on `[B, T, C]` tensor fails or produces garbage

### Likelihood

Low in production (each slot creates its own blend algorithm), but could cause silent test failures if fixtures are misconfigured.

---

## Recommended Fix

### Option A: Validate input shape

```python
def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
    if self.topology == "cnn":
        assert x.ndim == 4, f"CNN expects 4D input, got {x.ndim}D"
        return x.mean(dim=[2, 3])
    else:
        assert x.ndim == 3, f"Transformer expects 3D input, got {x.ndim}D"
        return x.mean(dim=1)
```

### Option B: Infer topology from input

```python
def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:  # [B, C, H, W]
        return x.mean(dim=[2, 3])
    elif x.ndim == 3:  # [B, T, C]
        return x.mean(dim=1)
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}")
```

---

## Verification

### How to Verify the Fix

- [ ] Add test for topology mismatch detection
- [ ] Verify assertion catches misuse in tests

---

## Related Findings

| Ticket ID | Relationship | Notes |
|-----------|--------------|-------|
| `B2-DRL-01` | `related` | GatedBlend credit assignment |
| `B2-PT-05` | `related` | GatedBlend hidden_dim |

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - blending.py - B2-02

---

## Cross-Review: DRL Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `NEUTRAL` |
| **Reviewer** | `drl` |

**Evaluation:** Valid defensive concern but low production risk. Option A (assert validation) is preferred over Option B; inferring topology from shape is fragile since `[B,C,1,1]` vs `[B,T,C]` with T=C would be ambiguous.

---

## Cross-Review: Code Review Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `codereview` |

**Evaluation:** Option A (validate input shape) is strongly preferred. Option B (infer topology) violates fail-fast principles and masks configuration errors. Assert ndim matches stored topology to surface misuse immediately.

---

## Cross-Review: PyTorch Specialist

| Field | Value |
|-------|-------|
| **Verdict** | `ENDORSE` |
| **Reviewer** | `pytorch` |

**Evaluation:** Option A (explicit assertions) is strongly preferred - it fails fast with clear diagnostics and is torch.compile compatible with zero overhead (assertions are elided in optimized mode).
Option B (infer from ndim) risks masking genuine shape errors; prefer explicit contracts over implicit inference in nn.Module forward paths.
