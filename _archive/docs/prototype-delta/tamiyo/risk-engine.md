# Tamiyo — Risk Engine & Breakers (v1)

Purpose
- Gate decisions using multiple signals and enforce conservative behavior under risk.

Inputs (per step)
- Loss/accuracy deltas vs last known
- Tolaria budgets: epoch/hook overrun flags (if provided in step state)
- Kasmina: isolation breaker state/violations; kernel fetch latency vs budget (from seed states or trainer flags)
- Blueprint metadata: risk/quarantine (cached Urza lookup with deadlines)
- Tamiyo timing: inference latency, timeouts counters

Scoring & Mapping
- Nominal/Degraded/Unhealthy categories per signal; weighted mapping to actions:
  - SEED only if nominal; OPTIMIZER if drift but safe; PAUSE if high risk/quarantine or repeated violations

Breakers
- Wrap inference, Urza, Oona IO; thresholds and cool‑down; transitions: CLOSED → OPEN → HALF‑OPEN
- Conservative mode: enter on breaker OPEN or repeated high‑risk signals; exit after cool‑down and consecutive successes

Telemetry
- Events: `bp_quarantine` (CRITICAL), `loss_spike` (HIGH), `isolation_violations` (HIGH), `timeout_inference`/`timeout_urza` (HIGH), `conservative_entered`/`conservative_exited`
- Metrics: breaker state, failure counts, last latencies

Acceptance
- Every decision includes a reason code aligned with mapping
- Conservative transitions happen automatically; transitions are visible in telemetry

Tests
- Table‑driven mapping; breaker transitions; conservative enter/exit

## Checklist (for PR)
- [ ] Inputs wired: loss delta, budgets, isolation/breakers, kernel latency, blueprint risk, timing
- [ ] Action mapping returns reason codes
- [ ] Breaker transitions visible; conservative auto-enter/exit
