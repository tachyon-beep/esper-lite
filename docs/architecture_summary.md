# Esper‑Lite Architecture Summary

This document is a practical crib sheet for engineers working on Esper‑Lite. It summarizes subsystem responsibilities, key APIs, contracts, flows, telemetry, and operational notes. Use it as a quick entry point when changing or extending any part of the system.

See also: docs/architecture_diagrams.md for Mermaid diagrams covering context, sequences, and pipelines.

## Prototype Delta vs Detailed Design

- The prototype scope we are targeting (“green for prototype”) is intentionally narrower than the full system described under `docs/design/detailed_design/`.
- The differences are curated in `docs/prototype-delta/` per subsystem (with a rubric in `docs/prototype-delta/rubric.md`).
- When implementing or reviewing changes, optimise for the prototype acceptance criteria first; only pull in full‑design features when the delta explicitly calls for them.
- Weatherlight and several safety/observability paths follow the prototype‑delta docs rather than the canonical full design to accelerate delivery.

## High‑Level Overview

Esper‑Lite orchestrates adaptive training and runtime kernel grafting for ML models using a contract‑first approach (Leyline protobufs) and a message fabric (Oona/Redis Streams).

- Tolaria runs training loops and emits state snapshots.
- Tamiyo evaluates system state and issues adaptation commands.
- Kasmina enforces lifecycle/gates, grafts kernels, manages safety/isolation, and reports telemetry.
- Karn defines blueprint descriptors and selection.
- Tezzeret compiles blueprints into runnable artifacts.
- Urza stores artifacts/metadata and serves them at runtime, including prefetch.
- Oona provides the message fabric for commands, telemetry, policy updates, and prefetch traffic.
- Simic trains Tamiyo policies offline from field reports and emits policy updates.
- Nissa ingests telemetry/state/reports into Prometheus/Elasticsearch with alerts and SLO tracking.
- Weatherlight supervises the above into a single, async process for a runnable prototype.

Core contracts: Leyline protobufs for SystemState, TelemetryPacket, AdaptationCommand, FieldReport, Kernel* messages, etc. (see `src/esper/leyline/_generated/leyline_pb2.py`).

## Design Principles

- Contracts First: Leyline enums/messages are the source of truth; no shadow enums.
- Safety Built‑In: Circuit breakers, gating, latency budgets, readiness/rollback, and signature+nonce freshness enforcement.
- Observability by Default: All subsystems emit structured telemetry; important events are explicit.
- Durability Where It Matters: WAL and checkpoints for recovery (Tolaria, Tezzeret, Urza, Tamiyo reports).
- Extensibility: Protocols and interfaces keep subsystems loosely coupled.

## Data Contracts (Leyline)

Key messages used across the system (not exhaustive):

- `SystemStatePacket`: training snapshot from Tolaria; includes metrics map and seed states.
- `TelemetryPacket`: metrics + events + system health status/summary/indicators.
- `AdaptationCommand`: Tamiyo’s decision (SEED/OPTIMIZER/PAUSE/CIRCUIT_BREAKER/EMERGENCY), with annotations.
- `SeedState`: Kasmina lifecycle export (for Tolaria and observability).
- `FieldReport`: outcome/metrics of control actions (Tamiyo → Simic).
- `KernelPrefetchRequest` / `KernelArtifactReady` / `KernelArtifactError`: Oona prefetch flow.
- `KernelCatalogUpdate`: Tezzeret → Urza update metadata.

Telemetry helpers live in `src/esper/core/telemetry.py` and are used project‑wide to ensure consistent packet structure.

## Observability Model

- Standardized telemetry via `build_telemetry_packet` with metrics and events.
- Health status/summary/indicators encode synthetic service health.
- Nissa ingests telemetry/state/field reports and exposes Prometheus metrics and Elasticsearch documents.
- Default alert rules cover: training latency, Kasmina isolation violations, Oona queue depth, Tezzeret retries.
- SLOs are driven by `slo.*` metrics (objective/actual) and tracked as rolling burn rate.

## Subsystems

### Tolaria (Training Orchestrator)

- Files: `src/esper/tolaria/trainer.py`, `src/esper/tolaria/__init__.py`
- Core classes:
  - `TolariaTrainer`: epoch loop with CE loss by default; exports `SystemStatePacket` per epoch; invokes Tamiyo → Kasmina; checkpoints to `var/tolaria/checkpoints/` + WAL (`wal.json`).
  - `TrainingLoopConfig`: device, epochs, grad accumulation.
  - `EpochStats`: running aggregates (loss, accuracy, gradient norm, throughput, latency).
- Integration:
  - Tamiyo via `evaluate_epoch(SystemStatePacket) → AdaptationCommand`.
  - Kasmina via `apply_command(AdaptationCommand)` and opt‑in seed state export + alpha advancement during BLENDING.
- Telemetry:
  - `tolaria.training.loss|accuracy|latency_ms|seeds.active|epoch_hook.latency_ms` and events for high latency/zero samples.

### Tamiyo (Strategic Controller)

- Files: `src/esper/tamiyo/policy.py`, `src/esper/tamiyo/service.py`, `src/esper/tamiyo/persistence.py`, `src/esper/tamiyo/__init__.py`
- Policy:
  - `TamiyoPolicy`: feed‑forward stub; fuses numeric metrics + seed/blueprint embeddings; outputs action logits + param delta; builds `AdaptationCommand` with annotations.
  - `TamiyoPolicyConfig`: feature dims, embeddings, registries, etc.
- Service:
  - `TamiyoService`: wraps policy with risk gating.
    - Conservative mode or loss spikes → `COMMAND_PAUSE` with reasons.
    - Optional Urza lookup enriches/overrides commands for quarantine/high risk blueprints.
    - Publishes telemetry; tracks `FieldReport`s in an append‑only log with retention enforced by rewriting (`var/tamiyo/field_reports.log`).
    - Ingests `PolicyUpdate`s (hot swap policy) directly or from Oona.
- Persistence: `FieldReportStore` is WAL‑backed with TTL pruning.
- Telemetry:
  - `tamiyo.validation_loss|loss_delta|inference.latency_ms|conservative_mode|blueprint.risk`, events for pause triggers/blueprint quarantine.

### Kasmina (Execution Controller)

- Files: many under `src/esper/kasmina/`: `seed_manager.py`, `gates.py`, `lifecycle.py`, `blending.py`, `isolation.py`, `memory.py`, `kernel_cache.py`, `prefetch.py`, `security.py`, `safety.py`, `registry.py`.
- Responsibilities:
  - Lifecycle: `KasminaLifecycle` enforces Leyline’s `SeedLifecycleStage` transitions.
  - Gates: `KasminaGates` evaluates G0–G5 (sanity, gradient health, stability, interface, system impact, reset).
  - Command handling: `KasminaSeedManager.handle_command` routes SEED/OPTIMIZER/PAUSE/BREAKER/EMERGENCY; verifies HMAC+nonce+freshness if configured (see `CommandVerifier`).
  - Kernel grafting: fetch via `BlueprintRuntime.fetch_kernel` (UrzaRuntime implementation), enforce latency budgets and breaker; attach kernels and progress through gates; GPU LRU cache available.
  - Alpha blending: `AlphaBlender` + `AlphaSchedule` for BLENDING stage and `advance_alpha`.
  - Isolation: gradient isolation monitoring sessions; violations feed their own breaker; telemetry.
  - Memory governance: TTL caches + periodic GC + emergency cleanup with optional teacher preservation.
  - Prefetch: `KasminaPrefetchCoordinator` bridges to Oona; on `KernelArtifactReady/Error` attaches or handles failure.
  - Registry: tracks parameter ownership (seeds vs teacher) to validate updates.
- Telemetry: rich events for gates, breakers, prefetch, isolation; metrics include last fetch latency, cache stats, isolation violations, priority, etc.

### Karn (Blueprint Catalog)

- Files: `src/esper/karn/catalog.py`, `src/esper/karn/templates.py`, `src/esper/karn/__init__.py`
- `KarnCatalog`: in‑memory registry of `BlueprintDescriptor` with parameter bound validation, tier filtering/selection, and circuit breaker around selection.
- Templates: `DEFAULT_BLUEPRINTS` (SAFE/EXPERIMENTAL/HIGH_RISK; BP001–BP050) including `quarantine_only` and `approval_required` flags.
- Telemetry: emits selection/breaker events (buffered); metrics cover selection counts per tier and selection latency.

### Tezzeret (Compilation Engine)

- Files: `src/esper/tezzeret/compiler.py`, `src/esper/tezzeret/runner.py`, `src/esper/tezzeret/__init__.py`
- `TezzeretCompiler`:
  - Compiles a `BlueprintDescriptor` + parameters with `torch.compile(dynamic=True)`; prewarms; falls back to eager on errors or “conservative” strategy.
  - Produces `CompiledBlueprint` (a torch `nn.Module` wrapper) with guard spec/digest/summary and artifacts saved to disk.
  - Records WAL for in‑flight jobs; exposes latest result and a `KernelCatalogUpdate` (checksum, prewarm, guard digest).
- `TezzeretForge`:
  - Discovers jobs (or via WAL), skips if already in Urza, compiles with timeout + thread isolation, tracks failures and opens breaker/backoff, can switch to conservative mode.
  - Builds telemetry packets with compiler/forge metrics and buffered events.

### Urza (Artifact Library & Runtime)

- Files: `src/esper/urza/library.py`, `src/esper/urza/runtime.py`, `src/esper/urza/prefetch.py`, `src/esper/urza/pipeline.py`, `src/esper/urza/__init__.py`
- Library:
  - SQLite table `blueprints` plus filesystem artifacts; WAL for atomic saves and crash recovery; durable extras (guard/spec/summary, checksum, compile stats, tags).
  - LRU‑ish memory cache with TTL support; circuit breaker on slow queries toggles conservative mode; maintenance to evict expired/missing artifacts.
- Runtime:
  - `UrzaRuntime` implements `BlueprintRuntime`; verifies checksum, loads `CompiledBlueprint` safely (registered globals), returns module + fetch latency.
- Prefetch Worker:
  - Consumes `KernelPrefetchRequest` from Oona; validates existence & checksum; computes fallback guard digest; publishes `KernelArtifactReady` with prewarm p50/p95 or `KernelArtifactError`.
- Pipeline:
  - `BlueprintPipeline` composes Karn+Tezzeret+Urza; validates, compiles, saves, optionally notifies via Oona; returns metadata+artifact path.

### Oona (Messaging Fabric)

- Files: `src/esper/oona/messaging.py`, `src/esper/oona/__init__.py`
- `OonaClient`:
  - Publishes: adaptation commands, field reports, telemetry, policy updates, kernel prefetch request/ready/error/catalog update.
  - Consumes: generic stream, or dedicated kernel streams.
  - Safety: HMAC signing/verification (optional), retries with dead‑letter, circuit breakers for publish/consume, freshness/replay guards for kernel messages, TTL housekeeping per stream, backpressure drop threshold, emergency stream routing when backlog crosses thresholds.
  - Metrics snapshot and telemetry emission helper.

### Simic (Offline Trainer)

- Files: `src/esper/simic/replay.py`, `src/esper/simic/trainer.py`, `src/esper/simic/validation.py`, `src/esper/simic/registry.py`, `src/esper/simic/__init__.py`
- Replay buffer converts `FieldReport` → `SimicExperience` (numeric features, metric sequence, categorical indices), with TTL and capacity pruning; can ingest from Oona.
- PPO‑style trainer with optional LoRA and metric attention; emits `PolicyUpdate`s after `PolicyValidator` passes; telemetry for losses/rewards/iterations and validation flag.
- Embedding registries persist seed/blueprint indices for stability.

### Nissa (Observability)

- Files: `src/esper/nissa/observability.py`, `alerts.py`, `slo.py`, `server.py`, `service_runner.py`, `__init__.py`
- Ingests `TelemetryPacket`, `SystemStatePacket`, `FieldReport` from Oona, maps to Prometheus counters and Elasticsearch documents.
- Alert engine + router with default rules; SLO tracker uses `slo.*` metrics to compute burn rate; `/metrics` and `/metrics/summary` endpoints expose Prometheus and SLO/alerts summary.
- Service runner boots Oona group, drains telemetry/state/reports, and runs FastAPI+Uvicorn; uses an in‑memory Elasticsearch stub if ES is unavailable.

### Weatherlight (Supervisor)

- Files: `src/esper/weatherlight/service_runner.py`, `__init__.py`
- Composes Oona, Urza library/runtime, Urza prefetch worker, Kasmina seed manager + prefetch coordinator, and Tamiyo service.
- Spawns background workers with restart/backoff; periodic telemetry aggregation; housekeeping for Oona and Urza; forwards Kasmina telemetry to Oona.
- Requires `ESPER_LEYLINE_SECRET` for HMAC signing (validated at start).

## End‑to‑End Flows

### Training Control Loop (Live)

1. Tolaria trains for an epoch; assembles `SystemStatePacket` (includes training metrics and optional Kasmina seed states).
2. Tamiyo evaluates state (policy inference + risk gating); may enrich/gate via Urza blueprint metadata; emits `AdaptationCommand`.
3. Kasmina receives the command; enforces lifecycle gates; fetches kernel via UrzaRuntime; attaches or falls back; advances lifecycle and alpha; emits telemetry.
4. Tolaria records telemetry and persists checkpoint/WAL for rollback.
5. Optional: Tolaria/Tamiyo publish state/telemetry via Oona for Nissa.

### Blueprint Compilation Pipeline

1. KarnCatalog provides `BlueprintDescriptor` and parameter bounds.
2. TezzeretCompiler compiles the blueprint (standard or conservative/eager), prewarms, creates `CompiledBlueprint`, and a `KernelCatalogUpdate`.
3. UrzaLibrary saves artifact + extras and persists DB row; WAL ensures recovery.
4. Optional Oona notification: `KernelCatalogUpdate` for downstream awareness.

### Prefetch Flow (Async)

1. KasminaPrefetchCoordinator publishes `KernelPrefetchRequest` to Oona.
2. UrzaPrefetchWorker consumes the request; validates checksum; publishes `KernelArtifactReady` (with prewarm p50/p95) or `KernelArtifactError`.
3. Kasmina processes ready/error and attaches kernel or records failure; telemetry is emitted in both cases.

### Offline Policy Improvement (Simic)

1. Tamiyo generates `FieldReport`s and appends to its WAL; can publish via Oona.
2. Simic ingests reports from Oona into replay buffer; performs PPO‑like updates.
3. Policy is validated; if passing, a `PolicyUpdate` is emitted (can be published to Oona).
4. Tamiyo ingests policy updates and hot‑swaps the in‑memory policy.

### Observability & SLOs (Nissa)

1. Nissa drains telemetry/state/reports from Oona.
2. Prometheus counters increment; alert rules evaluate; SLO tracker updates burn rates from `slo.*` metrics.
3. Documents are indexed to Elasticsearch (or in‑mem stub).
4. `/metrics` and `/metrics/summary` provide operational endpoints.

## Safety & Security

- Signing & Freshness: HMAC signatures on Oona envelopes (`ESPER_LEYLINE_SECRET`), nonce replay prevention (Kasmina `NonceLedger`), timestamp freshness windows.
- Circuit Breakers: Oona publish/consume, Karn selection, Tezzeret forge compile retries/timeouts, Kasmina kernel fetch and isolation violations, Urza DB latency;
  often switch to conservative modes or deny risky actions.
- Lifecycle Gates: G0–G5 gate policies centralize risk checks at each stage (sanity, gradient health, stability, interface, system impact, reset).
- Latency Budgets: Kernel fetch budgets enforced; breaker opens on exceed and fallbacks are applied.
- Durability: WAL/checkpoints for Tolaria; Tezzeret job WAL; Urza WAL for saves; Tamiyo field report WAL; defensive JSON load paths.

## Storage & Persistence

- Tolaria: `var/tolaria/checkpoints/ckpt-epoch-*.pt` + `var/tolaria/wal.json`.
- Tamiyo: `var/tamiyo/field_reports.log` append‑only WAL with periodic retention rewrite.
- Urza: `var/urza/catalog.db` (SQLite) + artifact files in `var/urza/artifacts/`.

## Configuration

- `EsperSettings` (Pydantic settings, env‑driven) in `src/esper/core/config.py`:
  - Redis/Oona: `REDIS_URL`, `OONA_*` streams & TTLs.
  - Leyline schema namespace & path.
  - Urza DB/artifact dir and cache TTL.
  - Tezzeret inductor cache dir.
  - Tamiyo policy dir, conservative mode flag, and field report retention.
  - Observability: Prometheus Pushgateway, Elasticsearch URL.
  - Logging: `ESP_LOG_LEVEL`.
- Secrets: `ESPER_LEYLINE_SECRET` required by Weatherlight/Oona/Kasmina command verification.

## Telemetry Cheat Sheet (Selected)

- Tolaria: `tolaria.training.{loss,accuracy,latency_ms}`, `tolaria.seeds.active`, `tolaria.epoch_hook.latency_ms`.
- Tamiyo: `tamiyo.validation_loss`, `tamiyo.loss_delta`, `tamiyo.inference.latency_ms`, `tamiyo.conservative_mode`, `tamiyo.blueprint.risk`.
- Kasmina: seed lifecycle/gates events, `kasmina.isolation.violations`, fetch latency, GPU cache stats; message priority as indicator in packet.
  - Blend telemetry: event `blend_config` with attributes `{mode, source, alpha_vec_len}`; per-seed metrics `kasmina.seed.{alpha,alpha_steps}`.
  - Pre-warm telemetry: `kasmina.prewarm.latency_ms` (global); per-seed metadata `prewarm_ms`.
- Karn: `karn.selection.{safe,experimental,adversarial}`, `karn.selection.latency_ms`, breaker state/open count.
- Tezzeret: compiler durations, prewarm times, eager fallbacks, forge breaker state and job counts.
- Urza: `urza.library.{cache_hits,cache_misses,evictions,integrity_failures,slow_queries,breaker_state,conservative_mode}`; prefetch `{hits,misses,errors,latency_ms}`.
- Oona: queue depths per stream, breaker states, conservative mode, retry/dead‑letter counts, kernel staleness/replay drops.
- Simic: `simic.training.{loss,reward,iterations}`, `simic.value.loss`, `simic.policy.{loss,entropy}`, optional validation pass.
- Weatherlight: worker running/backoff/restarts, uptime, last error timestamp.

## Running Locally (Pointers)

- Weatherlight Supervisor
  - Ensure Redis and (optionally) Elasticsearch are accessible; set `ESPER_LEYLINE_SECRET`.
  - Entry: `python -m esper.weatherlight.service_runner` (or import and call `run_service()`).

- Nissa Observability Service
  - Entry: `python -m esper.nissa.service_runner` or `uvicorn --factory esper.nissa.server:create_default_app --port 9100`.

- Tezzeret Forge
  - Instantiate `KarnCatalog`, `UrzaLibrary`, `TezzeretCompiler`, wire into `TezzeretForge`, and call `run()`.

- Simic Trainer
  - Build `FieldReportReplayBuffer`, ingest from Oona if desired, construct `SimicTrainer`, call `run_training()`, then emit policy updates.

## Dev Scripts & Utilities

Convenience scripts live under `scripts/`. Helpful ones when developing locally:

- `start_all_infra.sh` / `stop_all_infra.sh`
  - Bring up/down Redis, Prometheus, Grafana, and Elasticsearch via compose.
  - Use before running Weatherlight/Nissa locally.

- `check_infra.sh`
  - Quick health checks for Redis (container), Prometheus `/ready`, Grafana `/api/health`, and Elasticsearch cluster health.

- `run_integration.sh`
  - Spins up Redis and runs `pytest -m integration`, then tears Redis down.

- `run_nissa_service.py`
  - Simple wrapper invoking the packaged Nissa service runner.

- `run_fault_drills.py`
  - Executes Nissa fault drills to exercise alert rules and SLO tracker.

- `run_demo.py`
  - End‑to‑end demo: initialises Urza + Tezzeret, runs Tolaria→Tamiyo→Kasmina, publishes telemetry to Oona, executes Simic offline training, and applies policy updates.

- `generate_leyline.py`
  - Regenerates Leyline protobuf bindings into `src/esper/leyline/_generated/`.

- `bench_kasmina.py`
  - Micro‑benchmark for Kasmina graft/attach path; prints cache hit rate and timings, with optional simulated Urza latency.

- `profile_tolaria.py`
  - Profiles Tolaria training loop (PyTorch profiler), optional Chrome trace output.

- `check_shared_types.py`
  - Guardrail to ensure shared enums remain centralized in Leyline; fails the run if stray Python Enum classes are detected in `src/esper`.

Documentation‑adjacent (not core to runtime):

- `split_paper.py`
  - Utility to split `docs/paper/draft_paper.md` into chapters; useful for documentation workflows, not required for building/running services.

## Dev Scripts & Utilities

Convenience scripts live under `scripts/`. Helpful ones when developing locally:

- `start_all_infra.sh` / `stop_all_infra.sh`
  - Bring up/down Redis, Prometheus, Grafana, and Elasticsearch via compose.
  - Use before running Weatherlight/Nissa locally.

- `check_infra.sh`
  - Quick health checks for Redis (container), Prometheus `/ready`, Grafana `/api/health`, and Elasticsearch cluster health.

- `run_integration.sh`
  - Spins up Redis and runs `pytest -m integration`, then tears Redis down.

- `run_nissa_service.py`
  - Simple wrapper invoking the packaged Nissa service runner.

- `run_fault_drills.py`
  - Executes Nissa fault drills to exercise alert rules and SLO tracker.

- `run_demo.py`
  - End‑to‑end demo: initialises Urza + Tezzeret, runs Tolaria→Tamiyo→Kasmina, publishes telemetry to Oona, executes Simic offline training, and applies policy updates.

- `generate_leyline.py`
  - Regenerates Leyline protobuf bindings into `src/esper/leyline/_generated/`.

- `bench_kasmina.py`
  - Micro‑benchmark for Kasmina graft/attach path; prints cache hit rate and timings, with optional simulated Urza latency.

- `profile_tolaria.py`
  - Profiles Tolaria training loop (PyTorch profiler), optional Chrome trace output.

- `check_shared_types.py`
  - Guardrail to ensure shared enums remain centralized in Leyline; fails the run if stray Python Enum classes are detected in `src/esper`.

Documentation‑adjacent (not core to runtime):

- `split_paper.py`
  - Utility to split `docs/paper/draft_paper.md` into chapters; useful for documentation workflows, not required for building/running services.

## Testing & Structure Notes

- Tests mirror `src/esper/*` layout under `tests/` (unit) and `tests/integration/` (slower flows).
- Prefer pytest fixtures for shared infra fakes (e.g., Redis Streams, Prometheus/ES stubs).
- Target ≥80% coverage; add property tests for Leyline serialization where feasible.

## Quick File Index (by subsystem)

- Tolaria: `src/esper/tolaria/trainer.py`
- Tamiyo: `src/esper/tamiyo/{policy.py,service.py,persistence.py}`
- Kasmina: `src/esper/kasmina/{seed_manager.py,gates.py,lifecycle.py,blending.py,isolation.py,memory.py,kernel_cache.py,prefetch.py,security.py,safety.py,registry.py}`
- Karn: `src/esper/karn/{catalog.py,templates.py}`
- Tezzeret: `src/esper/tezzeret/{compiler.py,runner.py}`
- Urza: `src/esper/urza/{library.py,runtime.py,prefetch.py,pipeline.py}`
- Oona: `src/esper/oona/messaging.py`
- Simic: `src/esper/simic/{replay.py,trainer.py,validation.py,registry.py}`
- Nissa: `src/esper/nissa/{observability.py,alerts.py,slo.py,server.py,service_runner.py}`
- Weatherlight: `src/esper/weatherlight/service_runner.py`

## Gotchas & Best Practices

- Always use Leyline enums/messages; avoid re‑defining states or codes.
- Treat all message publish/consume paths as potentially lossy; rely on idempotent handlers and retries.
- Keep breakers conservative—prefer safe degradation and explicit telemetry over silent failures.
- When adding telemetry, include clear descriptions and useful attributes; keep metric naming consistent.
- Update `docs/` when behavior diverges from the canonical design docs; keep `.env.example` in sync for new config.
