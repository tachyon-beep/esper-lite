# Tamiyo — Timeout & Degradation Matrix (Prototype 3A)

Purpose
- Define strict per-step and per-call budgets and the mandated degradation behavior. Trainer must never stall.

Scope
- Applies to Tamiyo step-level decision path (`evaluate_step`) and metadata enrichment within Tamiyo.
- Weatherlight unchanged; no additional orchestration here.

Budgets (targets)
- Step evaluate budget: 2–5 ms per training step (hard timeout; no stall)
- Urza metadata lookup: 10–20 ms (timeout → skip enrichment)
- Kasmina apply_command (from Tolaria): same class as step evaluate (2–5 ms)
- Policy inference overall (p95 under load): <45 ms

Degrade Paths
- `timeout_inference` → return safe no-op or PAUSE (depending on risk); emit HIGH event
- `timeout_urza` → skip blueprint enrichment; emit HIGH event
- `timeout_apply` (Kasmina) → log warning in Tolaria (`tolaria.kasmina_timeout`) and proceed

Implementation Notes
- Use worker thread + `future.result(timeout=...)` to bound blocking calls
- Never block the trainer thread; treat all deadlines as “fail‑open” to safe behavior

Acceptance
- Timeouts enforced; no trainer stall
- Events logged on each timeout with correct priority

Open Questions
- Exact numbers per hardware profile (tune in perf tests)

## Checklist (for PR)
- [ ] Step evaluate timeout enforced (2–5 ms)
- [ ] Urza lookup deadline enforced (10–20 ms)
- [ ] Apply timeout handled in Tolaria; no trainer stall
- [ ] Timeout events logged with HIGH priority
