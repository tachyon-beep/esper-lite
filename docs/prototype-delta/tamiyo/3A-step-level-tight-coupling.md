# Tamiyo 3A — Step‑Level Tight Coupling Plan (ADR‑001 Aligned)

Status: Accepted (prototype). Weatherlight remains a whole‑of‑system orchestrator; no Weatherlight expansion. We do NOT support a bus‑subscriber model for Tamiyo decisions (3B).

Non‑Goals
- No Weatherlight changes beyond existing supervision/telemetry.
- No Oona subscriber path for Tamiyo decisions; decisions occur via in‑process calls from Tolaria.

What’s Already In Place
- Kasmina per‑seed/per‑step telemetry: `finalize_step(step_index)` exists and is called every optimizer step. src/esper/kasmina/seed_manager.py:312
- Tolaria insertion point: immediately after `_global_step += 1` is the natural call site to invoke Tamiyo, apply the command, advance α, and flush Kasmina. src/esper/tolaria/trainer.py:960
- Kasmina gate: commands must be signed (HMAC), nonces fresh, and timestamps within window or they are rejected. src/esper/kasmina/security.py, src/esper/kasmina/seed_manager.py:1265

Overview
- Tolaria calls Tamiyo on every training step (tight coupling; ADR‑001) with a lean `SystemStatePacket`. Tamiyo returns a fully signed `AdaptationCommand` under strict per‑step deadlines. Kasmina applies commands; Kasmina emits per‑seed telemetry via its existing finalize‑step flush.

How To Use (for PR owners)
- Choose one task (T‑A1..T‑A6) below and reference the relevant companion spec(s) in your PR.
- Copy the spec’s "Checklist (for PR)" into your PR and tick items.
- Include:
  - File:line anchors for main edits (e.g., `src/esper/tamiyo/service.py:123`)
  - Test commands and expected telemetry/events
  - Budget observations (step evaluate p95, inference p95)
  - Confirmation that Weatherlight remains unchanged and no new contracts were introduced


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

Work Packages vs Current State
1) T‑A1 — Command signing (Must‑have)
- Current: Tamiyo emits raw commands without `command_id`, `issued_at`, or HMAC.
- Change: set fields and sign before return. Inject a `SignatureContext` from `ESPER_LEYLINE_SECRET`.
- Acceptance: Kasmina accepts; no rejects for `missing_signature`, `nonce_replayed`, `stale_command`.

2) T‑A2 — Step‑level evaluate + deadlines (Must‑have)
- Current: only `evaluate_epoch`; no per‑step API; no timeouts; epoch‑keyed telemetry.
- Change: add `evaluate_step`; wrap policy inference and Urza lookups with timeouts; on breach → safe no‑op/PAUSE; emit `timeout_*` telemetry.

3) T‑A3 — Risk engine/breakers (Must‑have)
- Current: loss deltas and conservative flag only; no breaker state, no multi‑signal gating, no CRITICAL telemetry for quarantine; no auto transitions.
- Change: add multi‑signal risk scoring and breakers; auto conservative transitions; CRITICAL/HIGH events.

4) T‑A4 — Tolaria integration (Must‑have)
- Current: α advance and Kasmina flush already occur; epoch‑level Tamiyo call exists.
- Change: replace epoch‑level call with per‑step `evaluate_step`; enforce timeouts on Tamiyo and Kasmina apply; reuse conservative fallback logic.

5) T‑A5 — Field reports per decision (Should‑have)
- Current: WAL store exists; not emitted per decision.
- Change: serialize every decision to a report; append to WAL; honor retention.

6) T‑A6 — Performance/latency tests (Should‑have)
- Current: no targeted step‑budget tests.
- Change: add tests to enforce per‑step deadline and Oona priority routing.

Implementation Steps (PR‑sized)
- PR1: T‑A1 signing + unit tests (sign/verify/replay/stale)
- PR2: T‑A2 deadlines around inference and Urza lookup + unit tests
- PR3: T‑A4 Tolaria hook: per‑step call → apply → α advance → finalize_step; integration test for step cadence
- PR4: T‑A4 Tamiyo `evaluate_step`: signing, deadlines, reason annotations
- PR5: T‑A3 Risk engine v1 + breakers; table‑driven unit tests; conservative automation
- PR6: T‑A5 Field reports per decision; WAL retention; integration test
- PR7: T‑A6 Perf/latency tests; telemetry taxonomy check

Timeout & Degradation Matrix (summary)
- Tamiyo per‑step evaluate: 2–5 ms timeout → no‑op/PAUSE; never stall trainer
- Urza metadata lookup: 10–20 ms timeout → skip enrichment; event `timeout_urza`
- Kasmina apply_command: small timeout (similar to Tamiyo step call) → log `tolaria.kasmina_timeout`
- Overall policy inference p95: < 45 ms budget (enforced by tests)

Telemetry & Events (summary)
- Metrics: tamiyo.validation_loss, tamiyo.loss_delta, tamiyo.inference.latency_ms, tamiyo.conservative_mode, tamiyo.blueprint.risk
- Events: bp_quarantine (CRITICAL), pause_triggered (HIGH/WARNING), timeout_inference (HIGH), timeout_urza (HIGH), conservative_entered/exited, policy_update_applied/rejected
- Indicators: reason, priority, step_index (when applicable), policy version

Acceptance & Test Plan
- Unit: command signing/freshness/replay; timeout fallbacks; risk mapping; breaker transitions; policy update accept/reject
- Integration: step‑level call path (no stall); Kasmina receives signed commands; per‑seed finalize_step packets each step; blueprint quarantine → PAUSE; isolation breaker → no SEED actions; Oona routes HIGH/CRITICAL
- Performance: inference p95 < 45 ms; per‑step evaluate within budget under load

Risks & Mitigations
- Step overhead: keep `evaluate_step` minimal; strict timeouts; avoid Urza on hot path if budget is tight
- Telemetry volume: per‑seed packets each step; Kasmina queues capped; emit only on events
- State drift: avoid dependence on bus; rely on in‑process step state and seed_states exporter

Interfaces & Pseudocode (Tolaria hook)
```python
# src/esper/tolaria/trainer.py (after self._global_step += 1)
step_state = self._emit_step_state(...)  # reuse SystemStatePacket; lean fields
command = None
with ThreadPoolExecutor(max_workers=1) as ex:
    fut = ex.submit(self._tamiyo.evaluate_step, step_state)
    try:
        command = fut.result(timeout=self._config.tamiyo_timeout_s_step)
    except FuturesTimeout:
        command = None
# Apply command quickly, then alpha ramp and flush Kasmina
if command is not None:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(self._kasmina.apply_command, command)
        try: fut.result(timeout=self._config.tamiyo_timeout_s_step)
        except FuturesTimeout: pass
# Alpha and flush
if callable(exporter) and callable(advancer):
    for st in exporter():
        if st.stage == leyline_pb2.SEED_STAGE_BLENDING:
            advancer(st.seed_id)
if callable(finalize):
    finalize(step_index=self._global_step)
```

Design References
- ADR‑001 Tight Coupling: `docs/design/decisions/ADR-001-performance-optimized-tight-coupling.md`
- Tamiyo detailed design series: `docs/design/detailed_design/03-*`
