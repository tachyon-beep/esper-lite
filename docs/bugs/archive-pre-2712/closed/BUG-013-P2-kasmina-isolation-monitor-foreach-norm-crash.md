# BUG-013: GradientIsolationMonitor torch.stack fails on mixed devices

- **Title:** GradientHealthMonitor crashes on torch.stack when host/seed on different devices
- **Context:** Kasmina isolation monitor (`src/esper/kasmina/isolation.py`)
- **Impact:** P2 – Requires unsupported mixed-device configuration to trigger
- **Environment:** PyTorch 2.x; only triggers with manual mixed-device setup
- **Status:** Closed (Fixed)
- **Resolution:** Fixed by adding a device unification step in `compute_gradient_health_async`. Scalar norms from `torch._foreach_norm` are now moved to a common device (the device of the first tensor) before being stacked, preventing `RuntimeError` on mixed-device models.
- **Links:** `src/esper/kasmina/isolation.py` (GradientHealthMonitor.compute_gradient_health_async)
