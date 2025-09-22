# Tamiyo 3A — Step‑Level Tight Coupling Plan (ADR‑001 Aligned)

Status: Accepted (prototype). Weatherlight remains a whole‑of‑system orchestrator; no Weatherlight expansion. We do NOT support a bus‑subscriber model for Tamiyo decisions (3B).

Non‑Goals
- No Weatherlight changes beyond existing supervision/telemetry.
- No Oona subscriber path for Tamiyo decisions; decisions occur via in‑process calls from Tolaria.

Overview
- Tolaria calls Tamiyo on every training step (tight coupling; ADR‑001) with a lean `SystemStatePacket`. Tamiyo returns a fully signed `AdaptationCommand` under strict per‑step deadlines. Kasmina applies commands; Kasmina emits per‑seed telemetry via its existing finalize‑step flush.

Interfaces (no new schemas)
- Tamiyo
  - `TamiyoService.evaluate_step(state: SystemStatePacket) -> AdaptationCommand`
  - Signs commands (HMAC) and sets `command_id`/`issued_at` before signing.
  - Applies short deadlines to inference and Urza metadata lookup; degrades safely on timeout.
- Tolaria (call site after global step increments)
  - `src/esper/tolaria/trainer.py:960` — call `evaluate_step(...)` with timeout; on timeout → no‑op.
  - Apply command to Kasmina with a small timeout; then advance α for seeds in BLENDING; then flush Kasmina per‑seed telemetry.
- Kasmina (already present)
  - `src/esper/kasmina/seed_manager.py:312` — `finalize_step(step_index=...)` flushes per‑seed packets.

Work Packages (with acceptance)
- T‑A1: Command Signing + Freshness (must‑have)
  - Add HMAC signing in Tamiyo; set `command_id` and `issued_at` before signing. Add nonce/freshness acceptance.
  - Acceptance: Kasmina accepts commands; no `command_rejected` due to `missing_signature`, `nonce_replayed`, or `stale_command`.
  - Refs: src/esper/kasmina/security.py:43, src/esper/kasmina/seed_manager.py:1265; src/esper/tamiyo/service.py:75.

- T‑A2: Step‑Level Evaluate + Deadlines (must‑have)
  - Implement `evaluate_step` in Tamiyo; keep inference p95 < 45 ms, per‑step budget 2–5 ms (timeout → no‑op command or defer).
  - Bound Urza metadata lookup (e.g., 10–20 ms) and skip enrichment on timeout.
  - Emit telemetry events for `timeout_inference`/`timeout_urza` with priorities.
  - Refs: src/esper/tamiyo/service.py:75, src/esper/tolaria/trainer.py:960.

- T‑A3: Risk Engine + Breakers (must‑have)
  - Multi‑signal gating: loss deltas, Kasmina isolation/breaker, kernel latency budget, Tolaria hook/epoch budgets, blueprint risk/quarantine, Tamiyo inference timing.
  - Circuit breakers around inference/Urza/IO; automatic conservative‑mode transitions; clear resume conditions.
  - Acceptance: Decisions explainable via events; CRITICAL for quarantine/rollback deadline; WARNING for loss spikes/timeouts.
  - Refs: src/esper/tamiyo/service.py:118, 151; docs/design/decisions/ADR-001-performance-optimized-tight-coupling.md.

- T‑A4: Tolaria Step Integration + α Advance + Kasmina Flush (must‑have)
  - Insert step‑level Tamiyo call at `src/esper/tolaria/trainer.py:960` (after `_global_step += 1`).
  - Apply returned command with a short timeout; then:
    - α ramp: use `export_seed_states()` and `advance_alpha(seed_id)` for seeds in BLENDING.
    - Per‑seed telemetry: call `finalize_step(step_index)` on Kasmina.
  - Acceptance: No stalls; per‑seed packets emitted each step with up‑to‑date stage/alpha.
  - Refs: src/esper/tolaria/trainer.py:596, src/esper/kasmina/seed_manager.py:312, src/esper/kasmina/seed_manager.py:777.

- T‑A5: Field Reports Per Decision (should‑have)
  - Emit a FieldReport for each decision; maintain a small observation window counter; WAL append + retention (existing) and batch publish.
  - Acceptance: Reports present for every decision; WAL rewrites on retention; consumed in integration tests.
  - Refs: src/esper/tamiyo/persistence.py:32, src/esper/tamiyo/service.py:232.

- T‑A6: Performance/Latency Tests (should‑have)
  - Enforce inference budget and per‑step evaluate budget via tests; assert no stall of the trainer loop; ensure CRITICAL/WARNING priorities route to Oona emergency.
  - Acceptance: p95 within target; Oona emergency routing observed for high‑priority events.
  - Refs: src/esper/oona/messaging.py:239, src/esper/weatherlight/service_runner.py:367.

Deadlines & Timeouts (prototype)
- Tamiyo per‑step evaluate: 2–5 ms timeout (no stall; safe no‑op on breach).
- Urza metadata lookup: 10–20 ms timeout (skip enrichment on breach).
- Policy inference overall: p95 < 45 ms under load; bounded in tests.
- Kasmina apply_command: same timeout as Tamiyo call path; fallback to conservative if exceeded.

Telemetry (expectations)
- Metrics: `tamiyo.validation_loss`, `tamiyo.loss_delta`, `tamiyo.inference.latency_ms`, `tamiyo.conservative_mode`, `tamiyo.blueprint.risk`.
- Events: `bp_quarantine`, `pause_triggered`, `timeout_inference`, `timeout_urza`, `policy_update_{applied|rejected}`, `conservative_{entered|exited}`.
- Priority: CRITICAL for quarantine/rollback‑deadline; HIGH for loss spikes/breaker opens/timeouts; NORMAL otherwise.

Verification Checklist
- Commands are signed and fresh; Kasmina accepts them.
- Decisions incorporate blueprint risk/quarantine and isolation/breaker state; conservative mode auto‑transitions.
- No step stalls; on timeouts, safe no‑op/PAUSE with telemetry.
- Per‑seed telemetry flushes each step (seed_id, stage, alpha, latency, fallback flags).
- Field reports present per decision in WAL with retention.

Risks & Mitigations
- Overhead per step: keep Tamiyo step‑evaluate lean; use tight timeouts and short enrichment path.
- Telemetry volume: per‑seed packets each step; Kasmina already caps queues (see `MAX_PENDING_EVENTS`).
- Cross‑process ambiguity: not applicable; 3A mandates in‑process triad for decisions.

Design References
- ADR‑001 Tight Coupling: `docs/design/decisions/ADR-001-performance-optimized-tight-coupling.md`
- Tamiyo detailed design series: `docs/design/detailed_design/03-*`
