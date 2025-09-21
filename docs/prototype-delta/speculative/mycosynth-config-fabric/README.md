# Speculative Delta — Mycosynth (Configuration Fabric)

Design anchor: docs/design/detailed_design/research_concepts/mycosynth-config-fabric.md

Summary
- Centralised, real‑time configuration fabric (distinct from Leyline). Defines operational behaviour (timeouts, budgets, flags) with schemas, HA service, and client libraries for runtime updates.

Proposed Behaviour (Design‑style)
- Service: gRPC API with schema validation, versioning, environment sets; backed by HA KV store.
- Client: fetch at startup, cache LKG (last‑known‑good), subscribe for updates, expose typed getters.
- Authority: Mycosynth owns operational config; Leyline remains the source of data contracts.

Status (Prototype)
- Implemented: None (subsystems read static env/config). Risk if configs diverge.

Adoption (Low‑risk, prototype)
- Define schemas for a few high‑value knobs (Tolaria epoch budget, Oona thresholds, Kasmina breaker settings) and simulate “client library” by reading a JSON/YAML file reloaded at epoch boundaries (no new service).

Cross‑Subsystem Impact
- Touches all subsystems at config boundaries; best adopted incrementally via client abstraction, not by adding a new service right now.
