# Leyline — Implementation Roadmap (Canonical, No Tech Debt)

Goal: one batched, breaking schema change that aligns the prototype with the full lifecycle and gates, then adopt consistently across subsystems.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | Lifecycle enums | Add `SEED_STAGE_DORMANT`, `SEED_STAGE_EMBARGOED`, `SEED_STAGE_RESETTING`, `SEED_STAGE_TERMINATED` to `SeedLifecycleStage`; remove `SEED_STAGE_CANCELLED` and update all usages | Canonical 11‑stage lifecycle |
| 2 | Gates | Add `enum SeedLifecycleGate { GATE_G0..GATE_G5 }` and optional `GateEvent { gate, passed, reason }` used in telemetry | Gate‑level observability |
| 3 | Security envelope | Standardise HMAC + nonce + freshness fields in `BusEnvelope` or message metadata; define enforcement guidance | Authenticated messaging |
| 4 | Validation & budgets | Add test harness to assert serialisation time/size budgets for key messages (Option B) | Budget guarantees verified |
| 5 | Adoption | Regenerate bindings and update Kasmina/Tolaria/Tamiyo/Simic/Oona/Nissa to the new enums and optional GateEvent | Consistent usage |

Notes
- Single breaking commit pre‑1.0 to avoid deprecation paths.
- Keep messages compact; prefer native maps where appropriate.

Acceptance Criteria
- New enums present and used; repository tests pass; budget tests included; producers adopt HMAC/nonce/freshness enforcement plan.

