# JANK Template

- **Title:** Kasmina relies on private torch._foreach_norm; needs upstream-safe fallback
- **Category:** maintainability / future-proofing
- **Symptoms:** `GradientIsolationMonitor` uses `torch._foreach_norm`, a private API that may change/vanish in future PyTorch. If PyTorch 2.10+ alters it, isolation telemetry will break or crash. No fallback path is implemented.
- **Impact:** Lower severity now but high future risk—compile/runtime failures when upgrading PyTorch; hard to debug in production.
- **Triggers:** Upgrading PyTorch; running on CPUs where foreach is less stable.
- **Root-Cause Hypothesis:** Foreach used for perf; fallback was noted in comments but not implemented.
- **Remediation Options:** Implement a guarded fallback to per-tensor `p.norm()` when foreach is unavailable or raises; add a feature flag to disable foreach on CPU/mixed modes; wrap in try/except with telemetry/logging.
- **Validation Plan:** Add a test that forces the fallback (monkeypatch torch._foreach_norm to raise) and ensure isolation stats still compute.
- **Status:** Closed (Risk accepted)
- **Resolution:** The project intentionally relies on `torch._foreach_norm` for performance and accepts the upgrade risk; no fallback will be added unless/until a real upstream break occurs.
- **Links:** `src/esper/kasmina/isolation.py` foreach usage/comments
