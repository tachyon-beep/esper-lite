# Karn Combined Design

---

File: docs/design/detailed_design/05-karn-unified-design.md
---

# Karn Unified Design (v3.1 – Esper-Lite scope)

## Snapshot

- **Role**: Provides a vetted library of static blueprints that Tamiyo/Kasmina can request during training. No neural generation or dynamic mutation in Esper-Lite.
- **Scope**: Template selection, safety tiering, metadata management, and Urza integration for distribution.
- **Status**: Production-ready; design trimmed to Phase‑1 templates with C‑016 safety features kept intact.

## Responsibilities

- Serve 50 pre-built blueprints grouped by risk tier (Production / Experimental / Adversarial-for-training).
- Enforce three-tier safety architecture and circuit-breaker protections; adversarial tier never exposed to production.
- Handle blueprint lookups, metadata versioning, and publication through Leyline `BlueprintIR` contracts.
- Coordinate with Urza (storage/catalogue) using Leyline envelopes; validation assumed complete upstream.

## Component Map

| Component | Focus | Reference |
| --- | --- | --- |
| KarnBlueprintGenerator | Template selection + conservative fallback | `05.1-karn-template-system.md` |
| ThreeTierSafetyArchitecture | Risk tier enforcement, quarantine | `05.1` |
| RewardModel (lightweight) | Scores template usage for telemetry | `05.1` |
| Policy hooks | Accepts Tamiyo requests, returns blueprint ids | `05.1` |

## Eliminated Scope

- Neural/G2G generation, semantic mutation engines, GPU resource sharing, and other Phase‑2 features are **removed** for Esper-Lite. Karn does not invent new blueprints.

## Workflow

1. Receive request (optionally with `AdaptationCommand` context) via Leyline.
2. Select best matching static template; optionally apply minor parameterisation (e.g., scaling factors) within approved ranges.
3. Run safety checks (tier validation, cached approval status, circuit breaker state).
4. Return `BlueprintIR` metadata and notify Urza/Oona for bookkeeping.
5. Log usage metrics and feed lightweight reward stats for future template tuning.

## Safety & Operations

- Circuit breaker protects generation path; when open, Karn returns conservative fallback templates.
- Conservative mode triggers on repeated validation failures, slow generation (>400 ms), or telemetry alerts; limits selection to lowest-risk set.
- Memory manager performs TTL cleanup for cached metadata; prevents creep in long-running deployments.
- Telemetry exports `karn.template.selection_ms`, `karn.blueprint.tier`, `karn.breaker.state`.

## Performance Targets (static templates)

| Metric | Target | Notes |
| --- | --- | --- |
| Template lookup | 50–100 ms | Includes Urza metadata fetch. |
| Safety verification | 50–120 ms | Cached approval checks; no live validation. |
| Total generation latency | ≤300 ms | Conservative mode engages above budget. |
| Daily throughput | 10–50 blueprints | Library supports up to 9 k/day headroom. |

## Integrations (Leyline Option B)

- Contracts: `BlueprintQuery`, `BlueprintIR`, `EventEnvelope`, `TelemetryPacket`, `AdaptationCommand` (context only).
- Messages signed and versioned; stored approval flags indicate readiness for use.
- Oona transports events; Nissa captures metrics.

## References

- `docs/design/detailed_design/05.1-karn-template-system.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

### Mission-Critical Behaviours (Authoritative Reference)

`docs/design/detailed_design/05-karn.md` remains the canonical blueprint for Karn. The lightweight build must still honour the following responsibilities:

- **Tiered Template Library:** All 50 blueprints are partitioned into Production / Experimental / Adversarial tiers with approval metadata and quarantine rules (Old §"Blueprint Library" and §"Three-Tier Safety").
- **Request Handling:** Karn processes `BlueprintQuery` requests, enforces safety gates (approval flags, conservative pool when necessary), and returns `BlueprintIR` payloads matching Leyline contracts (Old §"Generation Flow").
- **Usage Logging & Telemetry:** Each selection emits telemetry (template id, tier, latency) for auditability and policy training (Old §"Reward & Metrics").
- **Safety Breakers:** Even if we simplify runtime breakers, the logical safeguards—quarantine of adversarial templates, fallback to conservative pool on repeated failures—must remain active (Old §"Safety Architecture").

These behaviours ensure Karn continues to function as the static blueprint authority for Esper-Lite.

---

File: docs/design/detailed_design/05.1-karn-template-system.md
---

# Karn Template System (Esper-Lite)

## Scope

Maintains the static blueprint library used in Esper-Lite. Only template selection, metadata validation, and safety-tier enforcement remain; dynamic mutations and neural generation are out of scope.

## Blueprint Library

- 50 pre-approved templates grouped by risk tier:
  - **Production (BP001–BP035)** – safe, proven patterns (identity, residual blocks, attention variants, etc.).
  - **Experimental (BP036–BP042)** – higher-risk structures used only when Tamiyo explicitly requests exploratory behaviour.
  - **Adversarial (BP043–BP050)** – quarantine-only blueprints for offline evaluation/training; never deployed to production.
- Metadata per template: id, description, curriculum stage, risk score, allowed parameters, approval stamp, last-updated timestamp.

## Generation Flow

```python
def generate_blueprint(context):
    if breaker.open:
        return conservative_pool.pick()
    # Optional adaptation command context
    cmd = context.get('adaptation_command')
    if cmd:
        template = select_by_command(cmd)
    else:
        template = select_best_template(context)
    if not validate(template):
        breaker.record_failure(); return conservative_pool.pick()
    return attach_metadata(template)
```

- `select_best_template` uses features such as target task, performance goals, hardware constraints (from `SystemStatePacket`).
- Validation checks tier rules, cached approval status, and ensures template parameters remain within approved ranges.
- Circuit breaker enters conservative mode when repeated failures or latency overruns occur; fallback set limited to lowest-risk templates.

## Safety Architecture

- **Tier enforcement**: experimental/adversarial tiers require explicit flags; adversarial tier blocked in production builds.
- **Validation pipeline**: Schema version check, approval flag verification, telemetry logging.
- **Conservative mode**: triggered by breaker; returns small safe subset (e.g., BP001–BP010).
- **Memory/TTL**: Cached template metadata refreshed every 6 h; older cache entries purged to avoid drift.

## Reward & Metrics (lightweight)

- Track blueprint usage and post-deployment outcomes to score templates; results feed reporting only (no auto-mutation).
- Metrics: `karn.blueprint.request_total`, `karn.blueprint.tier`, `karn.generation.duration_ms`, breaker state.

## Integrations

- Urza: template lookup & storage; reads/writes via Leyline `BlueprintQuery/BlueprintIR`.
- Approval Store: cached blueprint approvals and quarantine status.
- Oona/Nissa: event publication & telemetry.

Esper-Lite’s Karn remains a deterministic template server with strong safety guarantees and zero dynamic blueprint generation.
