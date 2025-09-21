# Karn — Delta Matrix (Phase‑1)

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Static template library | `05-karn.md`, `05.1` | 50+ blueprints partitioned by tier | `src/esper/karn/templates.py` | Implemented | Must‑have | SAFE/EXPERIMENTAL/ADVERSARIAL provided. |
| Parameter bounds validation | `05.1` | Enforce approved parameter ranges | `catalog.validate_request()` | Implemented | Must‑have | Raises on out‑of‑bounds/missing. |
| Tier enforcement & conservative pool | `05.1` | Filter by tier; conservative fallback set | `catalog.choose_template()` | Partially Implemented | Should‑have | Conservative flag supported; not breaker‑driven. |
| Request handling via Leyline | `05.1` | Accept `BlueprintQuery`, return `BlueprintIR` | Blueprint handled via `urza.pipeline` | Partially Implemented | Should‑have | No dedicated `BlueprintQuery` handler in Karn. |
| Circuit breaker & quotas | `05.1` | Breaker protects selection latency/validation failures | — | Missing | Should‑have | Not implemented. |
| TTL/metadata refresh | `05.1` | TTL cleanup for cached metadata | — | Missing | Nice‑to‑have | Not required for in‑memory defaults. |
| Telemetry on selections | `05.1` | Emit selection latency, tier metrics | — | Missing | Should‑have | No telemetry path in Karn. |
| Approval/quarantine flags | `05.1` | Store approval and quarantine metadata | `templates.py` includes flags | Partially Implemented | Should‑have | Flags present in descriptors; enforcement logic not in Karn. |
| Leyline as canonical | `00-leyline` | Use Leyline for descriptors/contracts | `catalog.py` uses Leyline types | Implemented | Must‑have | Canonical. |

