# Kasmina — PyTorch 2.8 Mandatory Upgrades (No Tech Debt)

Scope
- Apply PyTorch 2.8–specific improvements directly (no feature flags). These changes are mandatory for Esper‑Lite and avoid tech debt by using a single, guarded code path with eager fallbacks where necessary.

Baseline
- Current Kasmina already implements: Leyline 11‑state lifecycle + G0–G5 gates, breaker + monotonic timers, alpha blending with host.detach(), isolation hooks, TTL caches, per‑seed/teacher registry, HMAC/nonce/freshness verification.

Mandatory changes

1) Pre‑warm only in Kasmina; Tezzeret owns compilation
- What: Kasmina must not call `torch.compile`. Tezzeret pre‑compiles kernels; Kasmina only pre‑warms the loaded artifact to hydrate caches.
- How:
  - Kasmina: after attach, optionally run a single forward with a representative batch to pre‑warm (no compile calls).
  - Tezzeret: see `docs/prototype-delta/tezzeret/pytorch-2.8-upgrades.md` for the mandatory `torch.compile` pipeline.
- Acceptance: No runtime compilation in Kasmina; first BLENDING/TRAINING iterations do not pay compile latency.

2) Use `torch.inference_mode()` for SHADOWING/PROBATIONARY probes
- What: Wrap any forward probes during SHADOWING/PROBATIONARY in `torch.inference_mode()` to prevent autograd overhead and reduce memory churn.
- How: Wherever probe forwards occur (Kasmina or Tolaria), gate them under `inference_mode()` when `stage in {SHADOWING, PROBATIONARY}`.
- Acceptance: Probe forwards do not populate autodiff graphs; memory drops during SHADOWING.

3) Set global matmul precision/TF32 defaults
- What: Optimise float32 matmul on A100‑class hardware.
- How (at process init):
  - `torch.set_float32_matmul_precision('high')`
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
- Acceptance: Improved throughput with numerics aligned to 2.8 defaults; no regressions in tests.

4) Make isolation monitoring lighter and targeted
- What: Keep hooks active only during TRAINING and BLENDING; switch to online projections to avoid storing full gradient tensors.
- How:
  - In `src/esper/kasmina/isolation.py`, add a “projection” path: pre‑seed random projections per parameter id; accumulate scalar `gᵀr` at hook time; compute dot product from accumulators.
  - Open sessions on attach; close on stage changes away from TRAINING/BLENDING.
- Acceptance: Isolation stats available with reduced memory and Python overhead; G1 gate continues to enforce health based on violations.

5) Advance alpha per batch during BLENDING
- What: Ensure alpha ramps smoothly within BLENDING, not only on transitions.
- How:
  - Add `advance_alpha(seed_id)` that increments `alpha_steps` and recomputes `alpha` from `AlphaSchedule`.
  - Call from Tolaria per batch while `stage == BLENDING`.
- Acceptance: Alpha progresses over configured steps; telemetry reflects `alpha` ramp; blend stays smooth.

6) Route telemetry by priority to Oona
- What: Map message priority to Oona routing so CRITICAL/WARNING paths avoid queue head‑of‑line blocking.
- How:
  - Kasmina annotates telemetry with `MessagePriority`; Weatherlight/Oona use this to select emergency vs normal streams.
- Acceptance: Gate failures/breaker events appear on emergency stream (via supervisor routing); normal events stay on normal stream.

7) TTL cleanup on epoch boundaries
- What: Trigger TTL cleanup regularly in long runs.
- How:
  - Call `self._memory.cleanup()` in `update_epoch()` in addition to command handling.
- Acceptance: Cache eviction statistics remain bounded over long runs; no slow memory creep.

Defaults and constraints
- No feature flags: these changes are unconditional with eager fallbacks where compilation fails.
- Guard `torch.compile` with try/except and breaker events to prevent regressions; always keep the eager path working.
- Keep dtype consistent across host/seed to avoid implicit casts in the blender.

Optional next (post‑upgrade)
- CUDA Graphs capture/replay for stable BLENDING/TRAINING windows with fixed shapes (guarded; fall back if shapes change).
- `torch.compile` tuning per path (`mode='reduce-overhead'` for inference‑dominant; defaults for training‑dominant) once representative workloads are stable.

References
- Blending and schedule: `src/esper/kasmina/blending.py`, `seed_manager.py::blend/_handle_post_transition`
- Isolation hooks: `src/esper/kasmina/isolation.py`, `seed_manager.py::_attach_kernel/isolation_stats`
- Breaker/timers: `src/esper/kasmina/safety.py`
- TTL caches: `src/esper/kasmina/memory.py`
