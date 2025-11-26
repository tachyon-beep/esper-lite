# Plane Refresh RC1 Execution Work Package

## Intent

Plane Refresh RC1 consolidates remediation for Tolaria, Tamiyo, and Kasmina so the execution stack fully matches the prototype-delta expectations:
- Enforce strict dependencies (no synthetic fallbacks or placeholder IDs).
- Align PyTorch 2.8 posture across subsystems (compile/inference mode, fallbacks).
- Harden timeout handling, cancellation, and telemetry routing end-to-end.
- Surface complete telemetry for blends, gates, command verification, and emergency signals.

### Working Notes
- See `KNOWLEDGE_DUMP.md` for a consolidated execution briefing (scope, planned schema changes, helper modules, integration touchpoints, testing/doc updates).
- `09_strict_dependency_plan.md` captures the detailed execution plan for Risk R2 (strict dependency guard).

This work package unifies the subsystem review findings into a single remediation plan while preserving the per-component master lists.

## Scope

- Tolaria (training orchestrator): PCGrad correctness, timeout handling, fallback removal, telemetry routing.
- Tamiyo (policy + risk engine): strict timeout behaviour, blend configuration validation, logarithmic gating, telemetry/coverage outputs.
- Kasmina (execution layer): gate enforcement, fallback handling, blend telemetry, kernel dependency validation, security telemetry.

## Work Package Considerations

- Reduce per-file complexity and fix lint/static analysis issues as part of the remediation; refuse to ship lingering warnings.
- Consolidate all shared enums/data classes into Leyline; remove local enums unless a compelling exception is documented.
- Eliminate backwards-compatibility scaffolding (legacy APIs, placeholder IDs, synthetic fallbacks); prototype remains fix-on-fail pre-1.0.
- **Strict Dependency Alignment**: All subsystems must fail fast when kernels, metadata, or commands are missing. No synthetic pause commands or identity kernels.
- **Timeout Cancellation**: Replace per-call ThreadPoolExecutor usage with a shared async worker that supports proper cancellation and telemetry.
- **Telemetry Priority & Routing**: Ensure CRITICAL/WARNING events land on the emergency stream; expose blend/coverage/command verifier metrics.
- **PyTorch 2.8 Baseline**: Remove pre-2.8 paths and ensure inference/compile behaviour is consistent (torch.inference_mode, fallback breakers).
- **Gate & Lifecycle Consistency**: Gates must reflect fallback usage and stage expectations so Tamiyo risk decisions, Tolaria events, and Kasmina lifecycle stay aligned.
- **Shared Configuration**: Introduce consistent settings/config objects across subsystems to eliminate scattered defaults (fallback IDs, cache sizes, timeout budgets).

## Risks

- Masked failures (pause commands, fallback kernels) will hide outages unless removed.
- Telemetry gaps (blend metrics, command verification, emergency routing) prevent operators from seeing degraded states.
- Divergent blend semantics (Tamiyo gating on logits vs Kasmina gating on activations) can invalidate policy decisions.
- Lack of performance harness and rollback SLA metrics makes it hard to prove acceptance criteria.

## Next Steps

1. Review the per-subsystem master lists for reference; work packages WP-T, WP-A, WP-K are complete.
2. Launch **WP-CS1 (Cross-System QA & Performance Harness)** to execute the RC1 performance plan and close out rollback/latency validation.
3. Ensure harness outputs update the changelog/runbook and feed final RC1 sign-off.


## Module Work Packages and Sequencing

### Tolaria
- **WP-T1: Gradient Aggregation Correctness** — Fix PCGrad handling, weighted aggregation broadcasting, and empty tensor device/dtype issues (before downstream training work).
- **WP-T2: Timeout & Async Executor Unification** — Replace per-call executors with shared cancellable workers; update emergency controller logging and telemetry.
- **WP-T3: Safety & Rollback Hardening** — Fix checkpoint snapshot accounting, enforce `torch.load(..., weights_only=True)` when available, and ensure rollback deadlines really cancel work.
- **WP-T4: Telemetry & Backcompat Cleanup** — Remove legacy flags, ensure profiler outputs unique traces, and align public exports/docs with prototype delta.

### Tamiyo
- **WP-A1: Strict Decision Path** — Enforce fail-fast behaviour (no synthetic pause on failure), shared async worker integration, strict timeout telemetry.
- **WP-A2: Policy & Blending Validation** — Require real seed/blueprint IDs, validate blend configs, ensure confidence gating uses logits.
- **WP-A3: Telemetry Completeness** — Emit coverage/annotation metrics, route emergency telemetry correctly, surface command verifier failures.
- **WP-A4: Persistence & Registry Hardening** — WAL rewrite fsync, retention validation, registry cleanup.

### Kasmina
- **WP-K1: Gate & Fallback Enforcement** — Treat fallback kernels as gate failures, enforce stage expectations, update gate telemetry.
- **WP-K2: Blending & Isolation Upgrades** — Fix confidence gating inputs, add blend telemetry, normalise isolation metrics.
- **WP-K3: Command / Security / Registry** — Expose verifier telemetry, nonce ledger cleanup, teacher deregistration API.
- **WP-K4: Prefetch & Cache Reliability** — Hard-fail missing training IDs, make prefetch tasks cancellable, add cache locking and telemetry.

## RC1_EXECUTION Milestone Sequencing
1. **Shared Foundations**: Define strict dependency guidelines, shared async worker, telemetry routing updates, Leyline enum alignment (affects all modules).
2. **Tolaria WP-T1 & WP-T2**: Fix aggregation and adopt new async worker before other modules depend on it.
3. **Tamiyo WP-A1**: Integrate strict decision path with new worker + timeouts.
4. **Kasmina WP-K1**: Enforce gate/fallback rules so Tamiyo/Tolaria risk signals align.
5. **Tolaria WP-T3 / Tamiyo WP-A2 / Kasmina WP-K2**: Safety and blending corrections in parallel once core infra is ready.
6. **Telemetry Pass**: Execute WP-T4, WP-A3, WP-K3, ensuring telemetry is consistent and routed.
7. **Persistence & Cache Hardening**: Tamiyo WP-A4 and Kasmina WP-K4.
8. **Cross-System QA (WP-CS1)**: execute performance validation plan, capture rollback SLA instrumentation, refresh lint/complexity snapshots.
