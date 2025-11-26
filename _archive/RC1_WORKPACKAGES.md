# RC1 Blocker Work Packages

This document groups the outstanding RC1 blockers (see `RC1_BLOCKERS.md`) into risk- and priority-aligned work packages. Numbering references the order of items inside each subsystem section of `RC1_BLOCKERS.md` (e.g., “Kasmina #5” denotes the fifth Kasmina code-level bullet; “Kasmina Arch #2” denotes the second Kasmina architectural improvement bullet).

Risk tiers follow prototype policy:

- **P0 – Critical risk:** safety envelopes, strict-dependency enforcement, or data-loss regressions.
- **P1 – High risk:** concurrency/cancellation defects and correctness gaps that can stall control loops or corrupt state.
- **P2 – Medium risk:** configuration hygiene, deterministic behaviour, and maintainability work required before broadening scope.
- **P3 – Lower risk:** telemetry polish, performance tooling, and documentation hygiene that can trail higher tiers but must stay visible for RC2.

## Summary Table

| Work Package | Risk Tier | Priority | Scope Highlights |
|--------------|-----------|----------|------------------|
| **WP0: Safety & Strict Failure Enforcement** | P0 | Must land before any new prototyping | Kasmina fallback gating, Tamiyo synthetic commands/timeouts, Tolaria checkpoint & emergency handling |
| **WP1: Concurrency & Async Correctness** | P1 | Schedule immediately after WP0 | Prefetch pipelines, kernel cache locking, async cancellation for Tamiyo/Tolaria |
| **WP2: Configuration & Interface Hygiene** | P2 | Plan alongside WP1 | Shared config surfaces, scheduler policies, legacy toggle removal, deterministic lifecycle rules |
| **WP3: Telemetry, Observability & Tooling** | P2/P3 | Parallel once WP0/WP1 underway | Emergency telemetry, profiler/rollback metrics, performance harness |
| **WP4: Documentation & Surface Cleanup** | P3 | End-of-cycle polish | Remaining docstring/export cleanups |

## Work Package Details

### WP0: Safety & Strict Failure Enforcement *(Risk Tier P0, Priority P0)*
Focus: eliminate silent fallbacks, enforce strict dependences, and close data-loss paths before RC2 validation.

**Progress (2024-02-02):** Kasmina blend enforcement (items #2/#3), gate hardening (#4/#5), emergency/command validation (Kasmina #19/#20, Arch #10), and integration validation (Step 1.4 confidence/channel rejection coverage) are complete; Tamiyo command ID strictness (#13) also landed. Remaining scope continues to track the bullets below.

- **Kasmina**: ~~#19~~, ~~#20~~ (terminal fallback gating + shared validator); Arch #2 (emergency bypass routing), ~~Arch #10~~ (command validation surface).
- **Tamiyo**: #1, #3, #4, #7, #11, #12, ~~#14~~, ~~#15~~, #19; Arch items underpinning WP0 are already captured via #1/#3/#4/#7/#11/#12/#14/#15/#19 (architectural list has no additional P0 beyond these code fixes).
- **Tolaria**: #5, #8, #14, #21, #22, #24, #25.

### WP1: Concurrency & Async Correctness *(Risk Tier P1, Priority P1)*
Focus: stabilise asynchronous execution, cancellation, and shared-state access to prevent latent hangs or corruption.

- **Kasmina**: #10, #11, #12, #16, #17, #18; Arch #8 (blend-mode consolidation for runtime parity), Arch #9 (KernelLoadManager convergence).
- **Tamiyo**: #2, #6, #18, #20; Arch #2 (unified async worker), Arch #4 (blueprint metadata cache).
- **Tolaria**: #1 (purge legacy PyTorch shims), #2, #3, #13, #19, #20; Arch #1 (shared async mediator), Arch #3 (consolidated rollback/state manager).

### WP2: Configuration & Interface Hygiene *(Risk Tier P2, Priority P1/P2 mix)*
Focus: converge on deterministic configuration surfaces and retire legacy toggles/interfaces before expanding prototype scope.

- **Kasmina**: #1, #9; Arch #11 (shared Kasmina settings).
- **Tamiyo**: #16, #17, #21; Arch #1 (shared Tamiyo configuration), Arch #3 (risk/policy separation), Arch #5 (storage manager for normalisers/registries).
- **Tolaria**: #9, #10, #11, #15, #23; Arch #2 (trainer decomposition) and Arch #4 (typed Tolaria config schema).

### WP3: Telemetry, Observability & Tooling *(Risk Tier P2/P3, Priority P2)*
Focus: ensure incidents, rollbacks, and performance signals reach operators with actionable context.

- **Kasmina**: #13, #14, #15; Arch #1 (blend-mode telemetry), Arch #3 (torch.compile fallback metrics), Arch #6 (rollback SLA instrumentation), Arch #7 (distributed barrier documentation).
- **Tamiyo**: #5 (normaliser flush telemetry).
- **Tolaria**: #16, #18, #20; Arch #5 (central telemetry emitter).

### WP4: Documentation & Surface Cleanup *(Risk Tier P3, Priority P3)*
Focus: align public surfaces and guidance with the prototype-delta after the higher-risk fixes land.

- **Kasmina**: #6, #7, #8.
- **Tamiyo**: #8, #9, #10.
- **Tolaria**: #6, #7, #12, #17.

---

Execution order: land WP0 first, start WP1 in parallel with WP2 as soon as safety fixes stabilise, and address WP3 telemetry alongside the WP1/WP2 efforts. WP4 can trail as polish once regression tests are green.
