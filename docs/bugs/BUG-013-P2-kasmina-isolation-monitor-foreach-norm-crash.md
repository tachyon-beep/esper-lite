# BUG-013: GradientIsolationMonitor torch.stack fails on mixed devices

- **Title:** GradientHealthMonitor crashes on torch.stack when host/seed on different devices
- **Context:** Kasmina isolation monitor (`src/esper/kasmina/isolation.py`)
- **Impact:** P2 – Requires unsupported mixed-device configuration to trigger
- **Environment:** PyTorch 2.x; only triggers with manual mixed-device setup
- **Status:** Deferred (downgraded from P0)

## Root Cause Analysis (2025-12-17)

### Original Report vs Reality

The bug report claimed `torch._foreach_norm` crashes on mixed/empty grads. **This is incorrect.**

**Actual root cause:** `torch.stack()` fails when host and seed parameters are on different devices.

```python
# Line 152-154 in isolation.py
norms = torch._foreach_norm(host_grads)  # Works! Returns norms on original devices
result['_host_norm_sq'] = torch.stack(norms).pow(2).sum()  # CRASH HERE
```

### What Actually Works

| Scenario | Result |
|----------|--------|
| Empty grads | ✅ Already handled by `if host_grads:` guard |
| Same device (normal case) | ✅ Works correctly |
| `torch._foreach_norm` | ✅ Handles mixed devices, returns norms on original devices |

### What Crashes

```python
>>> norms = torch._foreach_norm([cpu_tensor, cuda_tensor])
>>> norms
(tensor(3.39), tensor(3.19, device='cuda:0'))  # Works!
>>> torch.stack(norms)  # CRASH
RuntimeError: Expected all tensors to be on the same device
```

### When Does This Happen?

Only with **unsupported configurations**:

1. Manual mixed-device setup (host on CPU, seed on CUDA)
2. `.to(device)` after monitor registration (→ FEAT-006)

Normal usage always has host and seed on the same device.

### Relationship to Other Issues

- **BUG-005:** Unrelated (was channels_last + detach segfault)
- **FEAT-006:** Root cause - if device tracking is implemented, this bug becomes moot

## Why P2 (Not P0)

1. **Normal usage works** - Host and seed are always on the same device
2. **Mixed-device is unsupported** - No documented use case for this
3. **FEAT-006 is the real fix** - Device-aware monitor re-registration
4. **Private API is stable** - `torch._foreach_norm` stable since PyTorch 1.9

## Fix Options (Future)

### Option A: Move norms to common device before stack

```python
if host_grads:
    norms = torch._foreach_norm(host_grads)
    # Move all norms to first tensor's device
    target_device = norms[0].device
    norms = [n.to(target_device) for n in norms]
    result['_host_norm_sq'] = torch.stack(norms).pow(2).sum()
```

### Option B: Sum without stacking

```python
if host_grads:
    norms = torch._foreach_norm(host_grads)
    # Sum squared norms without stacking (avoids device issue)
    result['_host_norm_sq'] = sum(n.pow(2) for n in norms)
```

### Option C: Fix via FEAT-006 (Recommended)

Implement device-aware monitor registration. When device changes, re-register with current parameters. This makes mixed-device impossible by design.

## Links

- `src/esper/kasmina/isolation.py` (GradientHealthMonitor.compute_gradient_health_async)
- FEAT-006: Device-aware isolation telemetry (root cause fix)
- BUG-005: Unrelated (channels_last segfault, already fixed)
