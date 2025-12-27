# Finding Ticket: TransformerHost pos_indices Buffer Lacks Shape Validation

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B2-DRL-11` |
| **Severity** | `P3` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/kasmina/host.py` |
| **Line(s)** | `320-321` |
| **Function/Class** | `TransformerHost.__init__()` |

---

## Summary

**One-line summary:** `pos_indices` buffer has implicit block_size dependency - model serialized with one block_size and loaded with another silently fails.

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

The `pos_indices` buffer is registered as non-persistent (not saved in state_dict) but has implicit dependency on `block_size`. If a model is serialized and loaded with a different `block_size` configuration, the reconstruction is silent and could produce wrong positional encodings.

### Code Evidence

```python
# /home/john/esper-lite/src/esper/kasmina/host.py:320-321

self.register_buffer('pos_indices', torch.arange(block_size), persistent=False)
```

### Failure Scenario

1. Create TransformerHost with block_size=512
2. Train and save checkpoint
3. Load checkpoint with block_size=1024 (config mismatch)
4. pos_indices is wrong size → indexing errors or wrong positions

---

## Recommended Fix

Add shape validation in forward:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, T = x.shape
    if T > self.pos_indices.shape[0]:
        raise ValueError(
            f"Sequence length {T} exceeds block_size {self.pos_indices.shape[0]}"
        )
    # ...
```

Or store block_size in state_dict metadata for validation on load.

---

## Verification

### How to Verify the Fix

- [ ] Add test for block_size mismatch detection
- [ ] Test checkpoint roundtrip with different block_sizes

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch2-drl.md`
**Section:** "File-by-File Analysis" - host.py - B2-11

---

## Cross-Review: PyTorch Specialist

| Verdict | ENDORSE |
|---------|---------|

Using `persistent=False` for `pos_indices` is correct (buffer is deterministically reconstructable from `block_size`), but the ticket correctly identifies missing runtime validation.
Adding a bounds check in `forward()` before `pos_indices[:T]` slicing is cheap and prevents silent shape mismatches that would cause cryptic indexing errors or wrong embeddings.

---

## Cross-Review: DRL Specialist

| Verdict | ENDORSE |
|---------|---------|

Valid correctness concern for checkpoint portability. The `persistent=False` buffer is reconstructed from constructor `block_size`, but no validation exists to catch mismatches on load.
Forward-time shape check (`T > pos_indices.shape[0]`) is the minimal fix; storing `block_size` in state_dict metadata would be more robust for early detection. P3 is appropriate given this requires an active configuration mismatch to trigger.

---

## Cross-Review: Code Review Specialist

| Verdict | ENDORSE |
|---------|---------|

Shape mismatch between serialized `block_size` and runtime config is a real footgun that silently corrupts positional encodings.
Adding a bounds check in `forward()` is low-cost, high-value defensive validation for checkpoint portability.
