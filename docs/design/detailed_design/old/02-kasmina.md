# Kasmina Combined Design

---
File: docs/design/detailed_design/02-kasmina-unified-design.md
---
# Kasmina Unified Design (v4.0)

## Snapshot
- **Role**: Execution layer that grafts pre-approved kernels into the host network without stalling Tolaria’s training loop.
- **Scope**: Kernel execution, gradient isolation, memory governance, parameter registration, safety controls, performance validation, checkpoint-driven pruning.
- **Status**: Production (C-022 hardening + C-024 knowledge-distillation support).
- **Core Invariant**: `∇L_host ∩ ∇L_seed = ∅` – enforced per forward pass with runtime hooks and circuit breakers.

## Architecture Highlights
- **Pure Executor**: Kasmina loads GPU kernels compiled by Tezzeret and never mutates model structure on its own. Compilation, validation, and policy decisions remain external.
- **Security Envelope**: All critical messages (optimizer commands, lifecycle transitions) use Leyline contracts signed with HMAC-SHA256, nonce tracking, and 60 s freshness windows. Replay attempts or bad signatures trigger conservative mode.
- **Safety Stack**: Backward hook monitors, circuit breakers, and emergency checkpoints replace assert-driven crashes. Violations after three consecutive detections quarantine the seed and notify Tolaria.
- **State Machine**: Eleven lifecycle states (Dormant → … → Terminated) with validation gates G1–G5. Each transition verifies registration, memory budgets, and gradient isolation readiness.
- **Telemetry Pipeline**: Kasmina ships structured telemetry directly to Tamiyo, which now owns aggregation and prioritisation for Esper-Lite.
- **Distributed Coordination**: Epoch-aligned barriers keep multi-GPU replicas consistent; Byzantine detection is log-only but flags outliers for SRE follow-up.
- **Structured Pruning (C-020)**: Kasmina exposes basic importance telemetry only; external pruning integrations are deferred in Esper-Lite.
- **Knowledge Distillation (C-024)**: Optional teacher model loaded with gradient checkpointing (14 GB → 7 GB). Teacher params register as immutable, and KD losses integrate with gradient isolation checks.

## Responsibilities by Subsystem
| Document | Focus | Key Guarantees |
| --- | --- | --- |
| `02.1-kasmina-kernel-execution.md` | GPU kernel loading, alpha blending, KD hooks | <0.1 ms cached load, host activations always `.detach()`, HMAC-protected commands. |
| `02.2-kasmina-memory-pools.md` | TTL caches, tensor pools, KD memory budgeting | Prevents 24–48 h memory creep; teacher allocation capped at 7 GB with checkpointing. |
| `02.3-kasmina-parameter-registration.md` | Registration protocol + optimizer integration | Every seed param mapped to LR group; teacher params tracked but never updated. |
| `02.4-kasmina-safety-mechanisms.md` | Circuit breakers, timing guards, pause semantics | No asserts; conservative mode on repeated violations; torch.compile stability checks. |
| `02.5-kasmina-performance-validation.md` | Benchmarks and regression harness | Verifies Leyline Option B targets (<80 µs / <280 B), KD overhead limits, isolation latency. |

## Interfaces & Data Contracts
- **Leyline (`esper.leyline.contracts`)**
  - `AdaptationCommand`, `SeedLifecycleStage`, `TelemetryPacket`, `GradientIsolationAlert`, `StructuralPruning*` messages.
  - Single `uint32 version` field; native `map<string, float>` usage keeps serialization <80 µs and messages <280 B.
- **Tolaria**: Sends signed optimizer directives and lifecycle transitions; receives health telemetry and isolation alerts. Tolaria retains optimizer ownership (two-optimizer pattern).
- **Tamiyo**: Consumes aggregated telemetry and gradient risk scores each epoch.
- **Urza**: Urza supplies kernel artifacts vetted offline before release.
- **Elesh / Emrakul**: Deferred; Esper-Lite only emits basic telemetry for potential future use.

## Operational Guarantees
- **Non-Blocking Execution**: Kernel swaps, telemetry, and pruning coordination run asynchronously from the training hot path; GPU caches prevent PCIe thrash.
- **Gradient Isolation Runtime Check**: Backward hooks verify `seed_grad • host_grad = 0`; any anomaly increments a counter feeding the circuit breaker. Fix introduced in C-022: host activations blend via `.detach()`.
- **Security & Replay Protection**: Nonce table (5 min TTL) and timestamp window reject replays; invalid messages logged and escalated.
- **Telemetry Backpressure**: Emergency events bypass queues; non-critical streams drop on saturation with rate-limited warnings.
- **Rollback Readiness**: All stateful changes emit checkpoints compatible with Tolaria’s 500 ms/12 s rollback stack.

## Key Metrics & Alerts
- Isolation violation rate, KD memory usage, kernel cache hit rate, TTL cache evictions, circuit breaker state transitions, Leyline serialization latency, checkpoint pruning success/failure counts.

## Future Work
- Extend Byzantine logging into active quorum scoring.
- Expand KD support beyond single-teacher architectures.
- Explore programmable blending schedules driven by Tamiyo policies (requires new safety proofs).

## References
- `docs/design/detailed_design/02.1-kasmina-kernel-execution.md`
- `docs/design/detailed_design/02.2-kasmina-memory-pools.md`
- `docs/design/detailed_design/02.3-kasmina-parameter-registration.md`
- `docs/design/detailed_design/02.4-kasmina-safety-mechanisms.md`
- `docs/design/detailed_design/02.5-kasmina-performance-validation.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

---
File: docs/design/detailed_design/02.1-kasmina-kernel-execution.md
---
# Kasmina – Kernel Execution & GPU Management

## Scope
GPU-native execution of pre-compiled kernels, including command authentication, gradient isolation, and optional knowledge-distillation support (C-022/024). This is the hot path that must preserve sub-millisecond latency.

## Kernel Pipeline
- **Kernel manager** loads artifacts from Urza, keeps them GPU resident, and records load times for regression tracking.
- **GPU cache** uses LRU eviction and explicit `release_gpu_memory()` calls; budgeted by MB to prevent VRAM creep.
- **Knowledge distillation** (optional) spins up a checkpointed teacher model:
  - Gradient checkpointing halves memory (14 GB → ~7 GB on A100 40 GB).
  - Teacher forward pass runs under `torch.no_grad()` with CUDA sync when multi-GPU.
  - Teacher params are registered as immutable (see parameter registration doc).

```python
# Alpha blending fix (C-022)
output = alpha * seed_output + (1 - alpha) * host_activations.detach()
```
- Host activations are always detached before blending; regression tests assert the `.grad_fn` disconnect.

### GPU Kernel Cache
```python
cache = GPUKernelCache(size_mb)
if cached := cache.get(kernel_id):
    return cached
kernel = GPUKernel.from_binary(urza.fetch(kernel_id))
cache.put(kernel_id, kernel)
```
- Tracks hit rate, evictions, and current footprint for telemetry (`leyline.kernel_cache.*`).

### Knowledge Distillation Hook
```python
if config.kd_enabled:
    kernel, teacher_logits = kernel_manager.load_kernel_with_teacher(kernel_id)
    if teacher_logits is not None:
        kd_loss = kl_div(student_logits, teacher_logits, T=config.kd_temperature)
```
- KD loss budget: <2 ms per batch; teacher forward pass budget: <12 ms.

## Command Handling & Security
- `AdaptationCommand` decoded via Leyline Option B structures.
- HMAC-SHA256 with 32-byte key, 60 s skew window, nonce TTL 5 min.
- Replay attempts or signature mismatches drop the command, log an error, and increment a security breaker.

```python
if nonce in seen or skew > 60 or !compare_digest(expected, signature):
    return False
seen.add(nonce)
```

## Gradient Isolation Runtime Guards
- Every seed is wrapped in `GradientIsolatedSeed` with backward hooks.
- Hooks compute dot products between host and seed gradients; any non-zero result increments violation count.
- Circuit breaker trips after three violations: seed transitions to QUARANTINE, alerts Tolaria, and schedules an emergency checkpoint.
- Isolation monitor records per-layer timings (target <8 ms overhead) and feeds telemetry for Tamiyo.

## Execution Metrics
| Metric | Target | Notes |
| --- | --- | --- |
| Kernel load (cached) | <0.1 ms avg | Cached in GPU memory. |
| Kernel load (cold) | <0.5 ms | Includes PCIe transfer. |
| Adaptation handling | <0.2 ms | After authentication. |
| Gradient isolation overhead | <8 ms | Max <12 ms; logged when exceeded. |
| KD teacher forward | <12 ms | Gradient checkpointed teacher. |
| KD memory usage | ≤7 GB | Pre-flight check before enabling KD. |

## Telemetry & Alerts
- Kernel cache stats, authentication failures, KD memory warnings, isolation violations, and hook latency buckets are emitted through Leyline telemetry packets with priority escalation for CRITICAL issues.
- Emergency path bypasses queue (sent directly to Oona) so Tolaria can react before the next epoch.

Kasmina’s execution module therefore remains a pure, authenticated kernel runner that enforces gradient isolation continuously while providing hooks for pruning and KD without breaking the latency envelope.

---
File: docs/design/detailed_design/02.2-kasmina-memory-pools.md
---
# Kasmina – Memory Pools & Tensor Lifecycle

## Scope
Protect GPU and host memory from long-run creep while accommodating knowledge-distillation resources. These controls were introduced after C-016 outages (24–48 h) and extended in C-024 for teacher models.

## Core Structures
- `TTLMemoryCache` controls kernel, blueprint, and telemetry buffers.
  - Keys scoped by `(epoch, request_id)` with configurable TTL.
  - Hit/miss/eviction counters feed Leyline telemetry for trend analysis.
  - `cleanup_expired()` runs on a timer and during GC passes.
- `KasminaMemoryPoolManager` owns separate caches for kernels, blueprints, telemetry, and KD assets and executes epoch-based GC.
  - `gc_frequency`: default every N epochs; records `gc_counter` and emits stats.
  - Telemetry reporter pushes usage and GC effectiveness metrics upstream.

## Knowledge-Distillation Memory Handling (C-024)
- `allocate_teacher_memory()` computes required GB (7 GB checkpointed, 14 GB otherwise) and checks available headroom before enabling KD.
- If allocation fails, KD is disabled and a Leyline alert is emitted.
- `track_checkpoint_effectiveness()` logs how much memory checkpointing saved and the recomputation overhead.
- Emergency cleanup path frees teacher allocations if fragmentation or OOM risk detected.

## Telemetry & Monitoring
- Every cache exposes `get_cache_stats()` → `{'size', 'hit_rate', 'evictions', 'ttl_seconds'}` for dashboards.
- Additional KD metrics: `teacher_memory_gb`, `checkpoint_effectiveness`, `memory_saved_gb`.
- Memory pressure > threshold triggers conservative mode (Kasmina halts new seed activation until stabilized).

## Operational Practices
| Routine | Why | Notes |
| --- | --- | --- |
| TTL cleanup | Prevent stale tensors from accumulating | Runs on schedule + manual trigger during GC. |
| Epoch GC | Bound long-tail memory | Configurable frequency (default: every 5 epochs). |
| KD allocation check | Avoid teacher OOM | Requires ≥7 GB free on GPU; otherwise KD disabled. |
| Telemetry buffer purge | Avoid queue bloat | TTL ensures telemetry memory remains bounded even during outages. |

These mechanisms keep Kasmina’s memory footprint predictable across multi-day training runs while providing explicit knobs for KD deployments.

---
File: docs/design/detailed_design/02.3-kasmina-parameter-registration.md
---
# Kasmina – Parameter Registration & Enforcement

## Scope
Guarantee that every parameter Kasmina touches belongs to a registered seed group, and that teacher weights remain immutable. Violations were a major contributor to Invariant L2 incidents pre C-016.

## Protocol Overview
1. **Teacher registration (C-024)**
   - Parameters loaded from the teacher checkpoint are registered once with `requires_grad=False`.
   - Any teacher tensor found with gradients re-enabled is corrected and reported.
2. **Seed registration**
   - Seeds announce their parameters (path, tensor id) before activation.
   - Each parameter is associated with an LR group handled by Tolaria’s UnifiedLRController.
3. **Update validation**
   - Before any optimizer step Kasmina verifies:
     - Seed is registered.
     - All tensors in the update belong to that seed.
     - No teacher parameters are present.
   - Failures raise an error, increment telemetry counters, and transition the seed to QUARANTINE until Tolaria intervenes.

## Key Data Structures
- `registered_parameter_groups: Dict[str, Set[int]]` – tracks param ids per LR group.
- `seed_parameter_registry: Dict[str, List[nn.Parameter]]` – per-seed parameter lists.
- `teacher_parameter_ids: Set[int]` – immutable set checked on every update.
- All operations protected by a registration lock to keep state consistent.

## Telemetry Hooks
- Successful registrations emit `leyline.parameter.registration` events (seed id, count).
- Failures emit `leyline.parameter.violation` with reason (unregistered, cross-seed, teacher contamination).
- Teacher gradient violations increment a dedicated counter for KD health.

## Runtime Safeguards
- Registration is idempotent; re-registering updates the stored list without duplication.
- Teacher verification runs periodically to ensure eval mode remains enforced.
- Parameter migration (e.g., new kernels) must call registration before invoking Tolaria’s optimizer rebuild.

With these checks Kasmina ensures that only approved parameters participate in optimization while providing the audit trail required for debugging and compliance.

---
File: docs/design/detailed_design/02.4-kasmina-safety-mechanisms.md
---
# Kasmina – Safety Mechanisms & Production Controls

## Scope
Replace assertion-driven crashes with observable, recoverable safety nets. Covers circuit breakers, timing guards, pause handling, and C-024 additions (checkpoint recompilation & teacher overflow protection).

## Circuit Breaker Framework
- `KasminaCircuitBreaker` implements CLOSED → OPEN → HALF_OPEN transitions with monotonic timers and Leyline telemetry.
- Config: `failure_threshold`, `timeout_ms`, `success_threshold` (defaults 5 / 60 000 / 3).
- Every critical operation (gradient hooks, kernel loads, checkpoint recompiles, KD teacher fetches) executes through a breaker.
- Breaker state transitions emit telemetry events; failures also capture the exception string for post-mortems.

## Timing Guards
- `MonotonicTimer` provides millisecond clocks immune to system time jumps.
- Guards wrap epoch boundary work, kernel blending, KD paths, and checkpoint recompilation.
- Violations trigger conservative mode: Kasmina halts new seed operations, downgrades telemetry volume, and escalates to Tolaria.

## Pause & Identity Kernels
- When Tolaria pauses a seed, Kasmina swaps in an identity kernel that protects gradient isolation while retaining signal flow.
- Resume path restores the cached kernel after verifying memory budgets and registration state.

## C-024 Additions
- **Checkpoint recompilation breaker**: Detects repeated failures when recomputing KD checkpoints; after threshold, KD is disabled and Tolaria notified.
- **Teacher overflow guard**: Monitors teacher memory consumption; if usage > configured limit, KD allocations are freed and breaker opened.
- **torch.compile monitor**: Collects compile-time and runtime exceptions, opens breaker on instability, falls back to eager execution.

## Telemetry & Alerting
- Breaker stats, timing violations, pause/resume events, and KD safety alerts are pushed as Leyline telemetry packets with severity levels.
- Aggregated inside Tamiyo’s telemetry store for decision support and long-term SRE dashboards.

These controls keep Kasmina resilient in production by catching faults early, isolating misbehaving seeds, and providing actionable observability instead of hard crashes.

---
File: docs/design/detailed_design/02.5-kasmina-performance-validation.md
---
# Kasmina – Performance Validation & Benchmarks

## Scope
Continuous verification that Kasmina meets latency, memory, and serialization targets, including Leyline Option B gains and C-024 KD overhead budgets.

## Target Matrix
| Metric | Target | Notes |
| --- | --- | --- |
| Kernel load (cache hit) | <0.1 ms avg / <0.5 ms max | Cold load path monitored separately. |
| Gradient isolation overhead | <8 ms avg / <12 ms max | Per seed invocation. |
| Leyline serialization | <80 µs, <280 B, ≤4 allocations | Confirm Option B benefits (73% faster, 57% smaller, 88% fewer allocs). |
| Telemetry generation | <0.5 ms | Prioritizes critical events. |
| KD teacher forward | <12 ms | With checkpointing enabled. |
| KD memory usage | ≤8 GB | 7 GB typical with checkpointing. |
| Training throughput with KD | ≥80% of baseline | 15–20% overhead acceptable. |
| Emergency rollback trigger | <150 ms | Must stay within Tolaria SLA. |

## Validation Suite
- **Kernel loading test**: Loads 100 kernels, tracks avg/max/p95 load times, ensures cache hit rates, records metrics.
- **Gradient isolation profiler**: Runs 100 iterations with synthetic seeds, captures CUDA profiler data, extracts isolation overhead.
- **Leyline benchmark**: Serializes/deserializes reference `SystemStatePacket` and `AdaptationCommand`, asserts size/time thresholds and allocation counts.
- **KD performance check**: Measures teacher forward pass, KD loss computation latency, and GPU memory saved via checkpointing; fails if limits exceeded.
- **Regression harness**: Runs nightly with torch.compile enabled; any regression beyond tolerance opens a breaker and blocks deployment.

## Reporting
- Results aggregated into `performance_metrics` dict and shipped via Leyline telemetry (pass/fail, averages, percentiles).
- Historical trend analysis flags drift (e.g., kernel load creeping above 0.1 ms) before hitting hard limits.

This suite ensures Kasmina’s production runtime stays inside the tight latency envelope while validating the newer KD and Leyline optimizations.
