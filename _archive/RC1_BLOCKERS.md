# RC1 Blockers

**Progress update (2024-02-01):** Phase 0 safeguards, Phase 1 Step 1.1 (Kasmina blend enforcement), Phase 1 Step 1.2 (G3/G4 hardening), and Phase 2 Step 1 (Tamiyo command builder strictness) have landed. Items closed by that work are annotated below but kept in place for traceability and cross-references.

**Progress update (2024-02-02):** Phase 1 Step 1.3 (Kasmina emergency & command validation), Step 1.4 (integration validation for confidence/channel rejection), and Phase 2 Step 2.2 (Tamiyo synthetic pause removal) are complete: `_GateEvaluator` now fails fast on Tamiyo fallback annotations with telemetry, `CommandAnnotationValidator` enforces training IDs, mesh coverage, and confidence logits, Tamiyo stops emitting synthetic pauses for missing seed candidates or policy timeouts, and Tolaria+Tamiyo integration tests confirm annotation rejections propagate. Kasmina blockers #19/#20, Tamiyo #14/#15, and architectural gap #10 are annotated as resolved below.

The following items from `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/KASMINA_REVIEW_FINDINGS.md` remain unresolved after RC1. Each entry lists what is still broken and the concrete work required to close it.

## Code-Level Findings (20)

1. **blending.py::`_clamp01` still mutates shared tensors**  
   - Location: `src/esper/kasmina/blending.py:61`  
   - Fix: Stop using `Tensor.clamp_` in-place; return a clamped clone/value so callers do not see unexpected alpha-vector mutations.
2. **RESOLVED – blending.py CHANNEL mode silently falls back when `alpha_vec` missing**  
   - Location: `src/esper/kasmina/blending.py:165-168`  
   - Resolution: Phase 1 Step 1.1 now rejects missing channel vectors (2024-02-01).  
   - Follow-up: ensure downstream telemetry continues to flag unexpected config gaps during integration drills.
3. **RESOLVED – blending.py confidence gate works on activations, not logits**  
   - Location: `src/esper/kasmina/blending.py:112-184`  
   - Resolution: Phase 1 Step 1.1 plumbs Tamiyo-provided logits into the gate; confidence gating now reflects the intended margin metric (2024-02-01).
4. **RESOLVED – gates.py G3 does not verify mesh coverage**  
   - Location: `src/esper/kasmina/gates.py:126-145`  
   - Resolution: Gate now checks Tamiyo-provided mesh requirements against registered host layers and rejects missing coverage (2024-02-01).  
   - Telemetry: Emits `mesh_coverage_missing` with offending layers to aid debugging.
5. **RESOLVED – gates.py G4 treats fallback status as nominal**  
   - Location: `src/esper/kasmina/gates.py:147-165`  
   - Resolution: G4 fails on `performance_status="fallback"` and raises a critical telemetry event highlighting the degraded seed (2024-02-01).
6. **`kasmina/__init__.py` still references legacy docs**  
   - Location: `src/esper/kasmina/__init__.py:1-4`  
   - Fix: Update the module docstring to reference the prototype-delta documentation (`docs/prototype-delta/kasmina/...`).
7. **`kasmina/__all__` exports internals wholesale**  
   - Location: `src/esper/kasmina/__init__.py:11-15`  
   - Fix: Narrow the export surface to the supported API only (e.g., lifecycle + seed manager façade) and keep helpers private.
8. **lifecycle docstring still points at detailed design**  
   - Location: `src/esper/kasmina/lifecycle.py:1-4`  
   - Fix: Reference the prototype-delta lifecycle spec so readers land on the current contract.
9. **Lifecycle transition matrix hard-coded without guard**  
   - Location: `src/esper/kasmina/lifecycle.py:47-83`  
   - Fix: Derive transitions from Leyline enums or add validation/telemetry when unexpected stages appear to avoid silent drift.
10. **Isolation monitor samples entire tensors with `torch.randperm`**  
    - Location: `src/esper/kasmina/isolation.py:168-179`  
    - Fix: Replace `randperm(numel)` with a memory-safe sampler (`torch.randint` reservoir or streaming) so large layers do not blow CPU RAM.
11. **Isolation verification uses raw dot product**  
    - Location: `src/esper/kasmina/isolation.py:96-104`  
    - Fix: Normalise by the host/seed norms (cosine similarity) before comparing to the threshold so results are scale-independent.
12. **Kernel cache is not thread-safe**  
    - Location: `src/esper/kasmina/kernel_cache.py:37-57`  
    - Fix: Protect `OrderedDict` mutations with a lock or funnel access through a single thread to avoid corruption under concurrent prefetch + attach.
13. **Kernel cache hit/miss counters never reset**  
    - Location: `src/esper/kasmina/kernel_cache.py:31-66`  
    - Fix: Provide windowed stats (e.g., rolling counters or reset API) so telemetry reflects current behaviour.
14. **TTL memory cache cleanup is silent**  
    - Location: `src/esper/kasmina/memory.py:83-87`  
    - Fix: Emit telemetry or return eviction counts from `cleanup()` so operators can see cache churn.
15. **TTL cache stats accumulate forever**  
    - Location: `src/esper/kasmina/memory.py:31-61`  
    - Fix: Add a snapshot/reset mechanism (or rolling window) for hit/miss metrics to avoid stale hit-rate reporting.
16. **Prefetch coordinator assumes a running asyncio loop**  
    - Location: `src/esper/kasmina/prefetch.py:65-86`  
    - Fix: Support synchronous embeddings (start a loop or require AsyncWorker) instead of calling `get_running_loop()` unguarded.
17. **Prefetch scheduler nests event loops via `asyncio.run`**  
    - Location: `src/esper/kasmina/prefetch.py:136-142`  
    - Fix: Use `asyncio.run_coroutine_threadsafe` or the injected worker; never call `asyncio.run` from potentially running event loops.
18. **Prefetch consumers block indefinitely on shutdown**  
    - Location: `src/esper/kasmina/prefetch.py:144-162`  
    - Fix: Add cancellation/timeouts (e.g., shorter `block_ms`, sentinel messages, or graceful exit on `_running` flag) so shutdown is timely.
19. **Teacher parameter IDs accumulate across registrations**  
    - Location: `src/esper/kasmina/registry.py:42-44`  
    - Fix: Clear `_teacher_parameter_ids` before registering a new teacher (or compare diffs) to prevent stale ownership clashes.
20. **Registry `validate_update` fails silently**  
    - Location: `src/esper/kasmina/registry.py:51-57`  
    - Fix: Return structured results or raise with context so callers can distinguish teacher clashes vs. unregistered parameters and emit precise telemetry.

## Architectural Gaps (11)

1. **Missing blend-mode telemetry + Tamiyo-logit enforcement**  
   - Need: Emit alpha summary/gate sparsity metrics and ensure confidence mode runs on Tamiyo-provided logits per prototype delta.
2. **No emergency bypass for CRITICAL telemetry**  
   - Need: Wire Weatherlight/Oona escape hatch so critical events bypass normal batching and hit operators immediately.
3. **No torch.compile fallback instrumentation**  
   - Need: Add PyTorch 2.8 `torch.compile` fallback metrics or explicit breaker telemetry covering compilation failures.
4. **Performance validation harness absent**  
   - Need: Ship the Kasmina micro-bench + telemetry harness promised in the delta matrix.
5. **Knowledge-distillation plumbing still TODO**  
   - Need: Implement KD loss/activation wiring between teacher and seeds as outlined in the prototype scope.
6. **Rollback readiness SLA metrics missing**  
   - Need: Instrument the 500 ms / 12 s rollback SLA so telemetry captures compliance.
7. **Distributed coordination/barrier semantics undocumented**  
   - Need: Implement or document the coordination protocol per fix plan to avoid divergence in multi-worker runs.
8. **Blend-mode validation/parsing not consolidated**  
   - Need: Centralise annotation parsing, safety rails, and kernel dispatch for blend modes to keep Tamiyo/Kasmina in sync.
9. **Kernel load responsibilities fragmented**  
   - Need: Merge caching + Oona prefetch into a `KernelLoadManager` that enforces strict failure, concurrency limits, and telemetry.
10. **Annotation/command validation lacks reusable component**  
    - Need: Extract verifier/annotation parsing into a shared validator returning structured pass/fail metadata.
11. **No shared Kasmina configuration object**  
    - Need: Introduce a config surface for fallback IDs, cache budgets, gate thresholds, etc., so settings are resolved once and reused.

_Total outstanding blockers: 27 (16 code-level, 11 architectural)._ 

## Tamiyo Subsystem

Outstanding items from `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/TAMIYO_REVIEW_FINDINGS.md` remain unresolved as of RC1.

### Code-Level Findings (21)

1. **GNN edge-attribute handling still crashes without features**  
   - Location: `src/esper/tamiyo/gnn.py:143-167`  
   - Fix: Provide zero/identity edge tensors for relations lacking attributes or require features per relation before calling `GATConv` so missing metadata no longer raises.
2. **GNN picks device from input tensors instead of module state**  
   - Location: `src/esper/tamiyo/gnn.py:151-152`  
   - Fix: Drive edge-attribute device placement from a module parameter (e.g., `self._policy_head.weight.device`) so pre-moved modules stay coherent even when `x_dict` is empty.
3. **Node encoder schema hard-coded to a fixed relation list**  
   - Location: `src/esper/tamiyo/gnn.py:41-76`  
   - Fix: Declare/validate the node types via `TamiyoGNNConfig` (or fail fast) so new relations cannot silently bypass the encoder.
4. **Graph builder invents `bp-unknown` blueprint IDs**  
   - Location: `src/esper/tamiyo/graph_builder.py:180`  
   - Fix: Require a real blueprint ID (or raise) instead of falling back to packet/run IDs, per strict dependency policy.
5. **Feature normaliser silently drops persistence errors**  
   - Location: `src/esper/tamiyo/graph_builder.py:103-116`  
   - Fix: Surface IO failures via telemetry or exceptions so operators know the normalisation stats are stale.
6. **Edge population ignores truncation limits**  
   - Location: `src/esper/tamiyo/graph_builder.py:816-908`  
   - Fix: Validate metadata counts against `max_layers`/`max_parameters` and adjust coverage/edges accordingly to avoid dangling indices.
7. **Feature builders still inject zero defaults with presence bits**  
   - Location: `src/esper/tamiyo/graph_builder.py:308-363`  
   - Fix: Treat missing metrics as hard failures (or emit alerts) instead of masking instrumentation gaps with zeroed features.
8. **Tamiyo package docstring references legacy detailed design**  
   - Location: `src/esper/tamiyo/__init__.py:1-4`  
   - Fix: Point to the prototype-delta docs so readers land on the current contract.
9. **`tamiyo/__all__` exposes every internal helper**  
   - Location: `src/esper/tamiyo/__init__.py:13-24`  
   - Fix: Narrow exports to the supported surface (`TamiyoService`, policy entry points) and keep internals private.
10. **Field-report store docstring still cites legacy design/backlog**  
    - Location: `src/esper/tamiyo/persistence.py:1-5`  
    - Fix: Update the reference to the prototype-delta persistence plan.
11. **WAL loader aborts on first truncated entry**  
    - Location: `src/esper/tamiyo/persistence.py:114-150`  
    - Fix: Skip only the bad record and continue scanning so later field reports are preserved.
12. **Retention treats missing `issued_at` as “now”**  
    - Location: `src/esper/tamiyo/persistence.py:88-95`  
    - Fix: Require timestamps (or reject the report) so retention windows function correctly.
13. **RESOLVED – Policy command builder still fabricates IDs**  
   - Location: `src/esper/tamiyo/policy.py:899-933`  
   - Resolution: Phase 2 Step 1 removed placeholder IDs and now raises dependency violations when command context is incomplete (2024-02-01).
14. **Policy returns a synthetic pause when no seeds appear**  
    - Location: `src/esper/tamiyo/policy.py:270-325`  
    - Fix: Treat an empty candidate set as a hard error/alert rather than silently degrading to `COMMAND_PAUSE`.
15. **Runtime/compile failures still degrade to CPU or pause**  
    - Location: `src/esper/tamiyo/policy.py:343-379`  
    - Fix: Surface compilation/inference failures as explicit errors (and notify operators) instead of auto-pausing after silent CPU fallbacks.
16. **Blend-mode annotations instantiate `EsperSettings` per call**  
    - Location: `src/esper/tamiyo/policy.py:653-709`  
    - Fix: Cache a resolved settings object during policy initialisation to keep the hot path deterministic.
17. **Backwards-compatible `evaluate_epoch` API still exposed**  
    - Location: `src/esper/tamiyo/service.py:1423-1426`  
    - Fix: Remove the legacy entry point and enforce the step-based API mandated by ADR-001.
18. **Policy timeouts still rely on non-cancellable futures**  
    - Location: `src/esper/tamiyo/service.py:1746-1774`  
    - Fix: Replace `AsyncWorker` cancellation with a mechanism that actually stops the worker (separate process or cooperative shutdown) and surface the timeout as a failure.
19. **Timeouts emit synthetic pause commands**  
    - Location: `src/esper/tamiyo/service.py:1763-1777`  
    - Fix: Propagate timeout failures instead of issuing `COMMAND_PAUSE`, per strict failure policy.
20. **Blueprint metadata prewarm uses cancelling futures**  
    - Location: `src/esper/tamiyo/service.py:1879-1911`  
    - Fix: Move Urza fetches to a dedicated worker with hard cancellation semantics or fail the step when metadata cannot be fetched in time.
21. **Risk evaluators keep reloading EsperSettings**  
    - Location: `src/esper/tamiyo/service.py:508-530`  
    - Fix: Resolve `EsperSettings` once (or inject a config object) so risk/telemetry logic shares consistent, cached configuration.

### Architectural Gaps (5)

1. **No unified Tamiyo configuration surface**  
   - Need: Build the shared settings object that feeds policy, graph builder, risk gates, and service timeouts, replacing scattered `EsperSettings()` calls.
2. **Inference/metadata execution still uses ad-hoc thread pools**  
   - Need: Introduce a persistent worker layer (or async runtime) with explicit cancellation semantics instead of per-call thread submissions that degrade to pause.
3. **Risk gating, policy inference, and telemetry remain intertwined**  
   - Need: Split these responsibilities into dedicated components so the service orchestrator only coordinates, simplifying future policy/risk changes.
4. **Blueprint metadata lacks a coherent cache with Urza notifications**  
   - Need: Implement the planned metadata cache that listens for Urza change events and ensures deterministic eviction rather than ad-hoc prewarm calls.
5. **Normalizer/registry persistence not centralised**  
   - Need: Add the storage manager described in the delta to coordinate flush cadence, error reporting, and reuse across policy + graph builder.

_Total Tamiyo blockers: 25 (20 code-level, 5 architectural)._ 

**Updated combined total:** 52 open items across Kasmina and Tamiyo.

## Tolaria Subsystem

These issues from `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/TOLARIA_REVIEW_FINDINGS.md` remain unresolved after RC1.

### Code-Level Findings (25)

1. **PyTorch 2.8 remains optional via compatibility shims**  
   - Location: `src/esper/tolaria/rollback.py:31-37`  
   - Fix: Drop the conditional `_WEIGHTS_ONLY_SUPPORTED` / legacy guards and assume the 2.8 baseline so loaders no longer branch for older releases.
2. **Emergency bypass counters are unused bookkeeping**  
   - Location: `src/esper/tolaria/emergency.py:35-49`  
   - Fix: Either enforce bypass-capped dispatch (and telemetry) or remove `_bypass_cap` / `_bypass_count` entirely.
3. **Local emergency clears leave stale metadata**  
   - Location: `src/esper/tolaria/emergency.py:70-89`  
   - Fix: Reset level, reason, and timestamp when `clear()` is called so downstream readers do not consume old data.
4. **Local triggers omit monotonic timestamp by default**  
   - Location: `src/esper/tolaria/emergency.py:70-83`  
   - Fix: Populate `_timestamp_ms` using `monotonic()` whenever `monotonic_ms` is not provided to keep parity with the shared signal.
5. **Emergency level transitions are unchecked**  
   - Location: `src/esper/tolaria/emergency.py:52-86`  
   - Fix: Validate requested levels against the Leyline enum and support explicit downgrades instead of silently ignoring invalid transitions.
6. **Tolaria docstring still cites legacy spec**  
   - Location: `src/esper/tolaria/__init__.py:1-4`  
   - Fix: Point the package docstring at the prototype-delta documentation and drop references to `old/01-tolaria.md`.
7. **`__all__` exports internal client helpers**  
   - Location: `src/esper/tolaria/__init__.py:6-8`  
   - Fix: Restrict exports to `TolariaTrainer`/`TrainingLoopConfig` (or other supported APIs) so internal clients stay private.
8. **LR reader silently falls back to 0.0**  
   - Location: `src/esper/tolaria/lr_controller.py:48-66`  
   - Fix: Raise when optimizer groups lack explicit `lr` entries instead of muting the schedule.
9. **Cosine schedule freezes after `t_max`**  
   - Location: `src/esper/tolaria/lr_controller.py:23-36`  
   - Fix: Continue the cosine cycle (or document wrap semantics) instead of clamping the step to `t_max`.
10. **Warmup wrapper enforces a hard 10% floor**  
    - Location: `src/esper/tolaria/lr_controller.py:43-52`  
    - Fix: Remove the fixed 0.1 minimum or make it configurable so warmup can start near zero as the spec expects.
11. **Unknown LR policies are silently ignored**  
    - Location: `src/esper/tolaria/lr_controller.py:73-92`  
    - Fix: Raise on unrecognised policy strings rather than returning `None`.
12. **Step/epoch semantics remain ambiguous**  
    - Location: `src/esper/tolaria/lr_controller.py:23-60`  
    - Fix: Document or enforce whether `step` is global or per-epoch to avoid mis-scheduling.
13. **Optimizer rebuild loses per-param settings**  
    - Location: `src/esper/tolaria/optimizer_manager.py:33-57`  
    - Fix: Capture full parameter-group configurations when rebuilding instead of copying only `defaults`.
14. **Optimizer state-load failures are swallowed**  
    - Location: `src/esper/tolaria/optimizer_manager.py:55-60`  
    - Fix: Surface the exception (telemetry or raise) so rebuild incompatibilities do not go unnoticed.
15. **Optimizer getter leaks internal reference**  
    - Location: `src/esper/tolaria/optimizer_manager.py:30-33`  
    - Fix: Return a read-only view or document the mutability contract to protect manager invariants.
16. **Breaker telemetry never reflects post-rebuild failures**  
    - Location: `src/esper/tolaria/optimizer_manager.py:35-66`  
    - Fix: Hook rebuild outcomes into the circuit breaker even when downstream application fails, so churn is observable.
17. **Profiler retains unused `active_steps` knob**  
    - Location: `src/esper/tolaria/profiler.py:9-25`  
    - Fix: Implement bounded profiling windows or drop the parameter altogether.
18. **Profiler swallows all exceptions**  
    - Location: `src/esper/tolaria/profiler.py:25-44`  
    - Fix: Emit telemetry/logging when profiling fails instead of silently continuing.
19. **Profiler capability guard is incomplete**  
    - Location: `src/esper/tolaria/profiler.py:15-23`  
    - Fix: Ensure `torch.profiler.ProfilerActivity` exists (and GPU availability) before use to prevent attribute errors on unsupported builds.
20. **Deadline signal timestamps are still hidden**  
    - Location: `src/esper/tolaria/rollback.py:207-291`  
    - Fix: Return the `SharedDeadlineSignal` timestamp (or propagate via `RollbackResult`) so operators know when the deadline tripped.
21. **Profiler path handling breaks when unset**  
    - Location: `src/esper/tolaria/trainer.py:2689-2713`  
    - Fix: Require a concrete `tolaria_profiler_dir` or short-circuit profiling before calling `Path(trace_dir)`.
22. **Loss computation forces tensors onto config device**  
    - Location: `src/esper/tolaria/trainer.py:2913-2923`  
    - Fix: Move targets to the model/output device (or perform device inference) so CPU fallbacks no longer crash.
23. **Legacy aggregation toggles still honoured**  
    - Location: `src/esper/tolaria/trainer.py:452-479`  
    - Fix: Remove `tolaria_agg_per_layer_*` flags in favour of the prototype `tolaria_seed_layer_*` settings.
24. **Emergency broadcasts are dropped after cap**  
    - Location: `src/esper/tolaria/trainer.py:4041-4055`  
    - Fix: Preserve unsent emergency signals or escalate failures instead of clearing the queue silently.
25. **Checkpoints still rely on pickle-based `torch.save`**  
    - Location: `src/esper/tolaria/trainer.py:4148-4172`  
    - Fix: Switch to safer formats (e.g., `state_dict` bytes guarded by `weights_only` loaders or explicit serialization) to eliminate pickle execution risk.

### Architectural Gaps (5)

1. **Async mediator for Tamiyo/Kasmina still embedded in trainer**  
   - Need: Stand up the shared mediator described in the delta (unifying cancellation, telemetry, and retry) instead of ad-hoc `_submit_async` calls inside the trainer.
2. **Trainer monolith remains undivided**  
   - Need: Split the core loop, aggregation, telemetry, and safety logic into dedicated components to ease evolution and testing.
3. **State/rollback management is still fragmented**  
   - Need: Consolidate fast-cache snapshots and WAL checkpointing under a single manager that owns serialization safety.
4. **Typed Tolaria configuration object still missing**  
   - Need: Replace scattered `EsperSettings` probes with a typed config schema that encodes prototype defaults explicitly.
5. **Telemetry/event emission is still ad-hoc**  
   - Need: Introduce the central emitter for batching, shared attributes, and rate-limiting before publishing to Oona.

_Total Tolaria blockers: 30 (25 code-level, 5 architectural)._ 

**Updated combined total:** 82 open items across Kasmina, Tamiyo, and Tolaria.
