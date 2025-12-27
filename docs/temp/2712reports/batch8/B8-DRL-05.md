# Finding Ticket: GradScaler Divergence When Multiple Envs on Same GPU

---

## Ticket Metadata

| Field | Value |
|-------|-------|
| **Ticket ID** | `B8-DRL-05` |
| **Severity** | `P2` |
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
| **File(s)** | `/home/john/esper-lite/src/esper/simic/training/vectorized.py` |
| **Line(s)** | `1132-1136` |
| **Function/Class** | `create_env_state()` |

---

## Summary

**One-line summary:** Multiple envs on same GPU have independent GradScalers that can diverge, causing different effective learning rates.

**Category:**
- [ ] Correctness bug
- [ ] Race condition / concurrency
- [ ] Memory leak / resource issue
- [ ] Performance bottleneck
- [x] Numerical stability
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

When `env_device_map = ["cuda:0", "cuda:0", ...]` (multiple envs on same GPU):

```python
# Each env gets its own GradScaler
env_state.scaler = torch.amp.GradScaler('cuda')
```

These independent scalers:
1. Start at the same initial scale
2. Update independently based on each env's gradient overflow patterns
3. Can diverge over training, leading to different effective learning rates

### Impact

- **Learning rate mismatch**: Seeds in different envs on the same GPU train at different effective rates
- **A/B test confound**: If comparing reward modes with envs on same GPU, results are confounded by scaler divergence
- **Subtle bias**: Some seeds may learn faster/slower due to their env's scaler state

---

## Recommended Fix

**Option 1 - Shared scaler per GPU:**
```python
# Share GradScaler across envs on same GPU
gpu_scalers = {}
def get_scaler_for_device(device):
    device_str = str(device)
    if device_str not in gpu_scalers:
        gpu_scalers[device_str] = torch.amp.GradScaler('cuda')
    return gpu_scalers[device_str]
```

**Option 2 - Document and ensure 1 env per GPU:**
Add validation that each GPU only has one env for fair comparisons.

---

## Verification

### How to Verify the Fix

- [ ] Decide on shared vs per-env scaler policy
- [ ] If shared: implement GPU-keyed scaler registry
- [ ] If per-env: document the divergence risk
- [ ] Add monitoring for scaler divergence across envs

---

## Related Findings

- B8-PT-04: Per-batch env_states recreation (related resource allocation)

---

## Appendix

### Original Report Reference

**Report file:** `docs/temp/2712reports/batch8-drl.md`
**Section:** "C8-24 - Multiple envs on same GPU have independent GradScalers"
