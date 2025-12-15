# BUG Template

- **Title:** GradientIsolationMonitor uses torch._foreach_norm without device checks, crashes on mixed/empty grads
- **Context:** Kasmina isolation monitor (`src/esper/kasmina/isolation.py`) `check_isolation_async` calls `torch._foreach_norm` over host/seed grads without guarding empty/mixed-device tensors.
- **Impact:** P0 – can segfault or throw runtime errors when some parameters have no grad, live on different devices (CPU seed + GPU host), or when torch._foreach_norm behavior changes. Blocks training when isolation telemetry is enabled (matches earlier gradient telemetry crash).
- **Environment:** PyTorch 2.9; mixed-device or CPU runs with isolation monitor active.
- **Reproduction Steps:** Run isolation monitor on CPU model with some params lacking grad; `torch._foreach_norm` may crash (similar to seed telemetry segfault).
- **Expected Behavior:** Isolation checks handle empty/mixed gradients safely, falling back to per-tensor norms or skipping when no grads.
- **Observed Behavior:** Potential native crash/exception due to foreach on heterogeneous inputs; private API risk.
- **Hypotheses:** `_foreach_norm` is a private API; no device/empty guards; similar crash seen in BUG-005 seed telemetry.
- **Fix Plan:** Add guards for empty grad lists; ensure host/seed grads are on the same device or fallback to safe norm computation; consider removing foreach usage on CPU/mixed setups.
- **Validation Plan:** Add unit covering CPU/mixed-device/empty grads; ensure no segfault and stats are sane.
- **Status:** Open
- **Links:** `src/esper/kasmina/isolation.py::check_isolation_async`, related BUG-005 (gradient telemetry segfault)
