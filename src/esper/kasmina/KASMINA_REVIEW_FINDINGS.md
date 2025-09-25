# Kasmina Review Findings


## blending.py
- `_clamp01` clamps tensors in place, so shared alpha buffers get mutated; return a clamped copy instead to preserve caller state (src/esper/kasmina/blending.py:61-64).
- Channel mode treats a missing `alpha_vec` as a silent fallback to convex blending; per the delta docs we should reject the configuration so Tamiyo notices the error (src/esper/kasmina/blending.py:166-168).
- `compute_confidence_gate` assumes 2D logits and is fed Kasmina activations, so confidence gating works on the wrong tensor and produces meaningless gates (src/esper/kasmina/blending.py:112-184).


## gates.py
- `GateInputs.expected_stage`/`telemetry_stage` are captured but unused, so gate evaluation can’t detect Tamiyo requesting out-of-order lifecycle transitions (src/esper/kasmina/gates.py:16-166).
- G2 stability ignores the `fallback_used` flag and still returns `passed=True`; repeated fallback execution should fail the gate to honour the strict dependency policy (src/esper/kasmina/gates.py:109-124).
- G3 only checks that `host_params_registered` is truthy, not that the mesh actually covers the requested seed; a mis-registered model can slip through (src/esper/kasmina/gates.py:126-145).
- G4 treats `performance_status="fallback"` as nominal; prototype intent is to pause/demote when running on fallback kernels (src/esper/kasmina/gates.py:147-165).


## __init__.py
- Docstring still references the legacy detailed design; switch to the prototype-delta docs so readers hit the current guidance (src/esper/kasmina/__init__.py:1-4).
- Export list exposes internals wholesale; narrow `__all__` to the supported surface so we don’t promise stability for helper classes (src/esper/kasmina/__init__.py:11-15).


## seed_manager.py
- `_load_fallback` swaps in `self._fallback_blueprint_id` or an `nn.Identity` whenever Urza fetch fails; that hides real outages and breaks the strict dependency policy. Fail the graft once the runtime fetch raises (src/esper/kasmina/seed_manager.py:1754-1783).
- G2 stability surfaces `fallback_used` in attributes but still returns `passed=True`, so seeds continue after running on fallback kernels instead of being culled (src/esper/kasmina/gates.py:109-124 applied via `_ensure_gate`).
- `_apply_blend_annotations` accepts arbitrarily long JSON arrays and casts them straight to lists without limits or telemetry; a malformed annotation can explode memory. Clamp size or reject oversize vectors (src/esper/kasmina/seed_manager.py:1563-1584).
- Fetch failures silently fall back to `self._fallback_blueprint_id` or an `nn.Identity` kernel (`_load_fallback`), masking missing kernels despite the strict-dependency policy; abort the graft instead of injecting placeholders.
- Gate enforcement passes the `fallback_used` signal into G2 but the gate never fails on fallback, so seeds continue even after running on emergency kernels; treat fallback usage as a gate failure per the delta docs.
- `_apply_blend_annotations` accepts arbitrary-length `alpha_vec` JSON without telemetry or size caps, so a malformed annotation can explode memory or bypass the channel-wise safety rails called out in P8.


## isolation.py
- `_prepare_projection` uses `torch.randperm(numel)` for each parameter, allocating O(numel) CPU memory; large layers blow up the monitor. Sample via `torch.randint` or a reservoir instead (src/esper/kasmina/isolation.py:95-112).
- `verify()` compares the raw dot product to the threshold without normalising by host/seed norms; thresholds become layer-size dependent. Compute cosine similarity before comparing (src/esper/kasmina/isolation.py:47-58, 66-112).


## kernel_cache.py
- Cache methods mutate an `OrderedDict` without any locking; async prefetch and seed-manager threads can call `get`/`set` concurrently and corrupt the LRU ordering. Guard mutations or funnel access through a single thread (src/esper/kasmina/kernel_cache.py:21-57).
- Hit/miss counters never decay, so long-running systems report hit rates based on ancient data; consider exposing interval stats if telemetry relies on it (src/esper/kasmina/kernel_cache.py:34-56).


## lifecycle.py
- Docstring references the legacy detailed design; point to the prototype-delta lifecycle docs instead (src/esper/kasmina/lifecycle.py:1-4).
- `allowed_next` hardcodes the stage matrix; if Leyline introduces new stages we silently ignore them. Consider deriving allowed transitions from Leyline enums or documenting the maintenance requirement (src/esper/kasmina/lifecycle.py:31-64).


## memory.py
- `cleanup()` drops expired entries silently; emit telemetry or at least return counts so operators see cache churn (src/esper/kasmina/memory.py:83-87).
- TTL cache metrics accumulate forever; expose a way to reset or snapshot per-interval stats if telemetry depends on recent behaviour (src/esper/kasmina/memory.py:31-61).


## prefetch.py
- `request_kernel` invents a `training_run_id` when none is provided; prototype policy forbids placeholder IDs (src/esper/kasmina/prefetch.py:25-36).\n- `start()` assumes an active event loop; when Kasmina is embedded synchronously it raises `RuntimeError` (src/esper/kasmina/prefetch.py:41-49).\n- `_schedule()` falls back to `asyncio.run`, which will nest event loops if called from async contexts (src/esper/kasmina/prefetch.py:58-64).\n- Consumer loops run forever with `block_ms=500` even after shutdown; add cancellation or timeout handling (src/esper/kasmina/prefetch.py:66-82).\n

## registry.py
- Teacher registrations accumulate forever; swapping the host leaves stale parameter IDs and future seed registrations fail (src/esper/kasmina/registry.py:42-54).\n- No API to drop a teacher or completely reset the index; long-running tests/offline runs inherit stale state (src/esper/kasmina/registry.py:20-54).\n- `validate_update` returns False silently; callers get no reason and can’t distinguish teacher vs. unregistered failures (src/esper/kasmina/registry.py:46-54).\n

## safety.py
- Breaker emits events only on transitions; HALF_OPEN successes never surface so operators can’t observe recovery (src/esper/kasmina/safety.py:118-143).\n- `allow()` returns `(False, BreakerEvent)` when open, but callers don’t distinguish failure vs. cooldown transition; consider explicit states or doc the contract (src/esper/kasmina/safety.py:83-117).\n- `force_state` sets `_failure_count` to threshold on OPEN but doesn’t emit telemetry, so manual overrides are invisible (src/esper/kasmina/safety.py:160-180).\n

## security.py
- NonceLedger only purges during `register`; long idle periods keep stale nonces forever (src/esper/kasmina/security.py:24-43).\n- CommandVerifier has no telemetry hook; rejected commands are invisible unless callers emit events (src/esper/kasmina/security.py:45-78).\n- Freshness window is hard-coded; expose it in telemetry or allow dynamic tuning via settings (src/esper/kasmina/security.py:55-78).\n

## Architectural Improvements
- Surface blend-mode telemetry (alpha mean/p95, sparsity, gate stats) and enforce confidence mode on Tamiyo logits; current implementation gates on activations.
- Treat fallback kernel usage as a gate failure so seeds don’t continue on emergency kernels; strict dependency policy forbids silent fallbacks.
- Implement emergency bypass routing for CRITICAL telemetry (Weatherlight/Oona path) per delta plan.
- Add PyTorch 2.8 torch.compile fallback instrumentation or explicit breaker telemetry to mirror delta notes.
- Ship the performance validation harness (micro-benchmarks, telemetry) called out as missing in delta matrix.
- Wire knowledge distillation losses/activation plumbing; KD remains a TODO in delta docs.
- Instrument rollback readiness SLA (500 ms/12 s) so metrics expose compliance.
- Emit explicit telemetry for command verifier failures (HMAC/nonce/freshness) to satisfy security envelope expectations.
- Document or implement distributed coordination/barrier semantics as outlined in the fix plan.
- Consolidate blend-mode parsing, validation, and PyTorch 2.8 kernels into a single component so Tamiyo annotations, safety rails, and runtime execution stay in sync.
- Wrap gate evaluation in a dedicated service that enforces stage ordering, fallback handling, and telemetry, keeping `_ensure_gate` lean.
- Merge kernel caching and Oona prefetch into a `KernelLoadManager` that handles strict failure, concurrency, and telemetry.
- Extract command verification/annotation parsing into a reusable validator returning structured results (pass/reason/metadata).
- Introduce a shared Kasmina settings/config object (PyTorch 2.8 baseline) so fallback IDs, cache sizes, and gate budgets are resolved once.
