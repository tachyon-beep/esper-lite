# JANK Template

- **Title:** Gradient isolation telemetry counts normal host grads as violations
- **Category:** correctness-risk
- **Symptoms:** `SeedSlot.capture_gradient_telemetry()` increments `isolation_violations` whenever `isolate_gradients` is True because any non-zero host gradient trips the check. With the current training loop, host gradients are expected (host keeps training while seeds incubate), so the counter steadily climbs even when isolation is healthy.
- **Impact:** Seeds hit the `DEFAULT_MAX_ISOLATION_VIOLATIONS` threshold after ~11 captures (stride 10 ⇒ ~110 steps) and fail the G2 gate, stalling BLENDING for CNN slots. Isolation telemetry becomes unusable for diagnostics and can trigger unnecessary culls.
- **Triggers:** Run any training loop with gradient telemetry enabled (e.g., Tolaria incubator) where `isolate_gradients=True` (TRAINING and CNN BLENDING). Minimal repro:
  ```bash
  PYTHONPATH=src python - <<'PY'
  import torch
  from esper.kasmina.host import CNNHost, MorphogeneticModel
  from esper.leyline import SeedStage

  model = MorphogeneticModel(CNNHost(num_classes=10, base_channels=8), device="cpu", slots=["mid"])
  model.germinate_seed("norm", "seed-iso", slot="mid")
  slot = model.seed_slots["mid"]
  slot.state.stage = SeedStage.TRAINING
  slot.isolate_gradients = True
  # Fake normal gradients without touching seed->host path
  for p in model.host.parameters():
      if p.requires_grad:
          p.grad = torch.ones_like(p) * 0.01
  for p in slot.seed.parameters():
      if p.requires_grad:
          p.grad = torch.ones_like(p) * 0.02
  print("violations before", slot.state.metrics.isolation_violations)
  slot.capture_gradient_telemetry()
  print("violations after", slot.state.metrics.isolation_violations)
  PY
  ```
  → violations increment from 0 to 1 despite no seed-to-host gradient path.
- **Root-Cause Hypothesis:** `capture_gradient_telemetry` flags any host gradient when `isolate_gradients` is True (`host_grad_norm > 1e-6`), but host gradients are expected because the backbone keeps training. There is no baseline comparison to distinguish leakage from normal host updates.
- **Remediation Options:** 
  - A) Compare host gradients with and without detached seed input to detect deltas (store a baseline norm).
  - B) Only count violations when `isolate_gradients` is False but seed inputs are undetached (true leakage path), and treat incubator host grads as expected.
  - C) Make isolation violations opt-in/telemetry-only and remove them from the G2 gate, relying on seed gradient ratio instead.
- **Risks of Change:** Might mask real seed→host leakage if detection becomes too lax; Simic gating logic and telemetry consumers need alignment.
- **Stopgap Mitigation:** Set `gradient_telemetry_stride=0` during incubator/BLENDING for CNNs or raise `DEFAULT_MAX_ISOLATION_VIOLATIONS` until detection is fixed.
- **Validation Plan:** Add a unit test where host gradients exist but seed input is detached — violations should stay at 0; add a contrasting test with intentional leakage (seed input not detached) that increments violations. Verify G2 gate uses the corrected signal.
- **Status:** Open
- **Links:** `src/esper/kasmina/slot.py` around `capture_gradient_telemetry`
