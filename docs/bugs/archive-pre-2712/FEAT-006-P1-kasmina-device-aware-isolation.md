# FEAT Template

- **Title:** Make Kasmina isolation telemetry device-aware after `.to()` moves
- **Problem Statement:** After germination, `SeedSlot.capture_gradient_telemetry` monitors a fixed set of host/seed parameters captured on the original device. When the model is moved to a new device (common for checkpoint load→to(cuda)), the isolation monitor keeps references to the old tensors, so subsequent telemetry reports zero host/seed gradients and G2 gate decisions become meaningless.
- **Goal:** Keep gradient isolation telemetry accurate across device transfers by re-registering the monitor with the live parameters.
- **Scope:** Kasmina `SeedSlot` isolation monitor lifecycle; `MorphogeneticModel.to()` and checkpoint restore paths.
- **Non-Goals:** Changes to PPO/Simic telemetry consumption or the gating thresholds themselves.
- **Requirements:**
  - Detect device moves (via `SeedSlot.to()` / `MorphogeneticModel.to()` / checkpoint restore) and refresh isolation monitor parameter bindings.
  - Ensure telemetry uses current device tensors to avoid zeroed norms after migration.
  - Maintain backward compatibility for fast_mode (telemetry disabled).
- **Stakeholders/Owners:** Kasmina lifecycle owners; Tolaria/Simic consumers of G2 gate and isolation telemetry.
- **Design Sketch:** Add a device-change hook that re-calls `isolation_monitor.register(host, seed)` whenever the slot or model changes device, and/or store weak refs to modules instead of parameter lists. Persist device metadata in extra_state to detect stale bindings on load.
- **Dependencies/Risks:** Must avoid extra CUDA syncs; ensure re-registration is deterministic under DDP and torch.compile.
- **Telemetry Needs:** Log when isolation monitor bindings are refreshed; optionally report device mismatch warnings.
- **Acceptance Criteria:** After germinating on CPU and moving model to GPU, gradient telemetry reports non-zero norms and G2 gate behaves identically to a model germinated directly on GPU.
- **Rollout/Backout:** Gate behind a flag or versioned checkpoint extra_state; back out by falling back to current behavior if issues arise.
- **Validation Plan:** Add a test that germinates on CPU, calls `.to(device)`, runs a backward pass, and asserts isolation telemetry reflects the new device grads (no zeros, no crashes).
- **Status:** Draft
- **Links:** `src/esper/kasmina/slot.py` (`capture_gradient_telemetry`, isolation_monitor handling); `src/esper/kasmina/host.py` (`MorphogeneticModel.to`)
