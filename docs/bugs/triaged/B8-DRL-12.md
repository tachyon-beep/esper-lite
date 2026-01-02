# Finding Ticket: MD5 Seed Offset is Fragile to Group ID Changes

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-12` |
| **Severity** | `P4` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/dual_ab.py` |
| **Line(s)** | `191-192` |
| **Function/Class** | `train_dual_ab()` |

---

## Summary

**One-line summary:** Seed offset uses MD5(group_id), making seeds depend on arbitrary group naming.

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
# Line 191-192
seed_offset = int(hashlib.md5(group_id.encode()).hexdigest()[:8], 16)
group_seed = base_seed + seed_offset
```

This creates reproducibility fragility:
1. Renaming "A" to "alpha" changes all seeds for that group
2. Seeds depend on arbitrary naming convention, not logical group index
3. Hard to reproduce exact experiment without knowing original group names

### Impact

- **Reproducibility risk**: Seed changes if group IDs are renamed
- **Documentation burden**: Must document exact group IDs for reproduction
- **Confusing behavior**: Same logical experiment may have different seeds

---

## Recommended Fix

Use fixed offset table:

```python
# Predefined offsets for reproducibility
GROUP_SEED_OFFSETS = {
    "A": 0,
    "B": 1000000,
    "C": 2000000,
    # ...
}

seed_offset = GROUP_SEED_OFFSETS.get(group_id, hash(group_id) % 1000000)
group_seed = base_seed + seed_offset
```

Or use group index:

```python
for idx, (group_id, group) in enumerate(groups.items()):
    group_seed = base_seed + idx * 1000000
```

---

## Verification

### How to Verify the Fix

- [ ] Switch to fixed offset table or index-based offsets
- [ ] Document seed computation for reproducibility
- [ ] Add test for seed determinism

---

## Related Findings

- B8-DRL-03: Sequential A/B training bias (related A/B testing concern)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-05 - MD5-based seed offset is fragile to group_id changes"
