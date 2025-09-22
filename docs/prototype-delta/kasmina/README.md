# Kasmina — Prototype Delta (Execution Layer)

Executive summary: the prototype implements the Leyline 11‑state lifecycle with gate checks (G0–G5), kernel prefetch via Oona/Urza with GPU cache reuse, projection-based gradient isolation monitoring with breaker escalation, a per‑seed/teacher parameter registry, TTL memory caches with epoch GC, a circuit breaker + monotonic timers, and HMAC/nonce/freshness verification for commands. Structured telemetry reports seed stages, gate/prefetch events, health, and priority. The per‑batch α ramp during BLENDING is integrated via Tolaria’s training loop and covered by tests. Remaining work includes a production‑grade performance validation harness, explicit telemetry bypass transport (Kasmina emits; Oona/Weatherlight route), and KD loss wiring.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — suggested backlog to close gaps
- `pytorch-2.8-upgrades.md` — mandatory PyTorch 2.8 changes (no feature flags)

Integration guide: per‑batch alpha ramp (Tolaria)
- Goal: While a seed is in BLENDING, advance α each batch to ensure a smooth ramp.
- Where to call: inside Tolaria’s training loop after loss.backward()/optimizer.step(), per batch.
- How to discover active seeds: call `export_seed_states()` on the Kasmina client and look for `stage == SEED_STAGE_BLENDING`.
- How to advance: call `advance_alpha(seed_id)` for each seed in BLENDING.
- Example (pseudo‑code using current trainer skeleton):
  
  ```python
  # Inside TolariaTrainer._train_single_epoch() loop, after backward()/step()
  exporter = getattr(self._kasmina, "export_seed_states", None)
  advancer = getattr(self._kasmina, "advance_alpha", None)
  if callable(exporter) and callable(advancer):
      try:
          for seed_state in exporter():
              if seed_state.stage == leyline_pb2.SEED_STAGE_BLENDING:
                  advancer(seed_state.seed_id)
      except Exception:
          pass  # best-effort integration in prototype
  ```

Notes
- α changes do not retrace host graphs (blend uses runtime buffer/tensor on Kasmina’s side).
- You can throttle to once every N batches if you use gradient accumulation.
- Telemetry: Kasmina publishes current α in seed context metadata; expect to see the ramp in telemetry.

Integration guide: per-seed telemetry flush (Tolaria)
- Goal: Ensure Kasmina emits exactly one telemetry packet per seed per training step.
- Where to call: immediately after each optimizer step/microbatch, using the same cadence as gradient aggregation.
- How to invoke: call `finalize_step(step_index=<int>)` on the Kasmina client. The `step_index` should match Tolaria’s global step counter so downstream consumers can correlate packets with steps.
- Behaviour: `finalize_step` drains buffered seed/global events accumulated since the previous call, emits one packet per active seed, and removes telemetry-only seed contexts. Skipping the call causes telemetry to queue until the next invocation.
- Example (see `TolariaTrainer._train_single_epoch()` for reference):

```python
self._global_step += 1
finalizer = getattr(self._kasmina, "finalize_step", None)
if callable(finalizer):
    finalizer(step_index=self._global_step)
```

Notes
- Finalization is best-effort: Kasmina guards against failures so the training loop can continue.
- Telemetry packets still surface seed health summaries, but system-wide events (e.g. emergency cleanup) are emitted via a separate global packet in the same flush.
- CRITICAL seed events (e.g., isolation breaker open) bypass the per-step cadence: Kasmina emits the per-seed packet immediately and Weatherlight pushes it to Oona without waiting for the periodic drain.

Outstanding Items (for coders)

- Alternate blend modes (executor‑side)
  - Implement the approved modes (Residual, Channel/Group‑wise, Confidence‑gated) with safety rails (host.detach, α clamps, hysteresis) and telemetry.
  - Pointers: `docs/prototype-delta/kasmina/blending-upgrade.md`, `src/esper/kasmina/blending.py`, `seed_manager.blend()`.

- Performance validation harness
  - Add micro‑benchmarks for kernel load latency and isolation overhead; emit telemetry for operator visibility.
  - Pointers: `docs/design/detailed_design/02.5-kasmina-performance-validation.md`, `scripts/bench_kasmina.py`.

- Telemetry priority routing tests
  - Ensure CRITICAL/WARNING events (e.g., isolation breaker open) route to Oona emergency via Weatherlight; add tests.
  - Pointers: `src/esper/core/telemetry.py`, `src/esper/weatherlight/service_runner.py` (Kasmina telemetry forward).

- Distributed coordination (optional)
  - Add epoch barriers/quorum semantics if needed; otherwise document single‑process assumptions.
  - Pointers: `seed_manager.update_epoch()` and memory GC cadence.

- KD loss wiring (optional)
  - Plumb KD losses/activations using teacher model buffers; retain memory budget checks.
  - Pointers: `seed_manager.register_teacher_model()` and blend hooks.

Blend mechanism ownership
- Blending is a seed‑integration concern. Tamiyo selects the blending mechanism from a small approved list (policy), and Kasmina executes the requested mode safely; Kasmina does not choose the mode.
- Prototype default is convex blend with `host.detach()`. Advanced modes are described in `blending-upgrade.md` and should be activated by Tamiyo via command annotations/parameters when ready.

Design sources:
- `docs/design/detailed_design/02-kasmina-unified-design.md`
- `docs/design/detailed_design/02.1-kasmina-kernel-execution.md`
- `docs/design/detailed_design/02.2-kasmina-memory-pools.md`
- `docs/design/detailed_design/02.3-kasmina-parameter-registration.md`
- `docs/design/detailed_design/02.4-kasmina-safety-mechanisms.md`
- `docs/design/detailed_design/02.5-kasmina-performance-validation.md`

- Implementation evidence (primary):
  - `src/esper/kasmina/lifecycle.py`
  - `src/esper/kasmina/seed_manager.py`, `src/esper/kasmina/prefetch.py`
  - `src/esper/core/telemetry.py`
  - `src/esper/security/signing.py`
  - Tests: `tests/kasmina/*`, `tests/integration/test_control_loop.py`
- Benchmarks: `scripts/bench_kasmina.py`
