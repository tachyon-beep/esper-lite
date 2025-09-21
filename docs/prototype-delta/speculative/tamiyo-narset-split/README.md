# Speculative Delta — Tamiyo/Narset Split (Strategic vs Tactical Controllers)

Design anchor: docs/design/detailed_design/research_concepts/strategic_control/Draft Proposal - tamiyo_narset_split.md

Summary
- Elevate Tamiyo to global strategist; introduce Narset as regional tactical controllers. Add protected zones and inverse throttle coupling Emrakul’s pruning budget to scaffold/seed activity.

Proposed Behaviour (Design‑style)
- Contracts: `PolicyEnvelope`, `BudgetEnvelope`, `ProtectedZoneGrant`, `ScaffoldStatus`, `ScaffoldSunsetNotice` (Leyline additions).
- Timing: Narset decisions bind at next checkpoint; protected zones embargo Emrakul in/near active scaffolds.
- Inverse throttle: `B_prune` decreases with scaffold/seed spend EMA; quiet periods raise compaction allowance.

Status (Prototype)
- Implemented: None (single Tamiyo; no Narset; no zones). Partial: Tamiyo already annotates blueprint risk metadata.
- Risk: Requires Leyline extensions; out of current prototype scope for code.

Adoption (Low‑risk, prototype)
- Document the envelopes and zones in Leyline RFC; optionally pilot “zone” annotations in Tamiyo telemetry without functional impact.
- Use Oona topics for zone/scaffold lifecycle logs (telemetry only).

Cross‑Subsystem Impact
- Tamiyo/Narset split (controllers), Emrakul (planning windows), Elesh (importance), Kasmina (no lifecycle change), Oona (topics).

Implementation Tasks (Speculative)
- Leyline RFC: Define `PolicyEnvelope`, `BudgetEnvelope`, `ProtectedZoneGrant`, `ScaffoldStatus`, `ScaffoldSunsetNotice` messages; add enums/fields for zones.
- Controllers: Create a minimal `Narset` process (or Tamiyo submodule) that consumes envelopes and produces `AdaptationCommand`s scoped by region.
- Tamiyo: Produce envelopes; compute inverse‑throttle budget; track protected zones and their TTL.
- Emrakul: Honour `ProtectedZoneGrant` (embargo in zones); plan only outside active zones; add global abort on conflict.
- Oona: Add topics for `zones.*` and `scaffold.*` lifecycle; integrate with existing streams.
- Nissa: Dashboards for zone counts, embargoed regions, and inverse‑throttle budget over time.
- Docs: Decision log for when to shard Emrakul; conflict resolution procedures.
