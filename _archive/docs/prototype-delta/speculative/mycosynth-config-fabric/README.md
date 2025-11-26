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

Implementation Tasks (Speculative)
- Schema: Define a small set of high‑value config schemas (YAML/JSON) for epoch budgets, Oona thresholds, breaker settings.
- Client shim: Add a shared helper to load+validate configs and hot‑reload on a timer; expose typed getters per subsystem.
- Subsystem wiring: Replace hardcoded constants with config lookups (Tolaria hook budget; Oona thresholds; Kasmina breaker params).
- Telemetry: Emit `config.version` and key values in telemetry for traceability.
- Ops: Draft rotation/runbook for config changes; add a dry‑run validation command.
- Future: Spec a gRPC service if the shim proves insufficient (out of prototype scope).
