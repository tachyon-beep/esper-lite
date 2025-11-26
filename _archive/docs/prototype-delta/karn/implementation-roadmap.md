# Karn — Implementation Roadmap (Phase‑1 Only)

Goal: complete Phase‑1 template system behaviours without introducing Phase‑2 generation.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | Request handling | Add `BlueprintQuery` handler using Leyline; return `BlueprintIR` with approvals/quarantine flags | Clear contract boundary |
| 2 | Circuit breaker + conservative mode | Track selection latency and validation failures; on repeated faults return conservative pool only | Robust safety behaviour |
| 3 | Approval enforcement | Enforce `approval_required` and `quarantine_only` flags during selection | Strong tier safety |
| 4 | Telemetry | Emit `karn.generation.duration_ms`, `karn.blueprint.tier`, breaker state via Oona | Operator visibility |
| 5 | TTL/refresh | Optional TTL on cached metadata; refresh hooks for template updates | Avoid drift |

Notes
- Keep Leyline descriptors/contracts as the only data model.
- Do not implement neural/generative features in Phase‑1.

Acceptance Criteria
- Leyline request/response path exists and is covered by tests.
- Breaker/conservative behaviour and approval enforcement are observable in telemetry.

