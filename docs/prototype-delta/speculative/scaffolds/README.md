# Speculative Delta — Scaffolds (Temporary, Foldable Blueprints)

Design anchor: docs/design/detailed_design/research_concepts/scaffolds/kasmina_scaffold.md

Summary
- Treat “scaffold” as a deployment profile for regular blueprints: temporary, self‑sunsetting, policy‑driven alpha schedules, and exit gates (fuse/prune/persist). No change to Leyline lifecycle states; overlays policies during existing stages.

Proposed Behaviour (Design‑style)
- Metadata flags in blueprint descriptors: `supports_scaffold`, `sunset_policy`, `ttl_ckpts`, `success_gates`, `alpha_policy`, `foldability_descriptor`.
- Kasmina: honour overlay (alpha warm/hold/decay; host branch .detach() preserved) and emit extra gauges in telemetry.
- Tamiyo: budget scaffolds, gate on success metrics, request “persist” or “sunset” via annotations.
- Urza/Tezzeret: paired manifests carry foldability descriptors to inform fuse vs prune decisions.

Status (Prototype)
- Implemented: None (policy overlay not modelled). Partial metadata exists (risk/stage) in `BlueprintDescriptor`.
- Dependencies: Leyline remains unchanged; this is metadata + policy only.

Adoption (Low‑risk, prototype)
- Start with metadata only (Karn/Urza descriptors); allow Tamiyo to request scaffold class via annotation, and Kasmina to log alpha gauges without changing behaviour. Add telemetry names only (no functional gates yet).

Cross‑Subsystem Impact
- Karn/Urza/Tezzeret: metadata + manifests.
- Tamiyo: gating + annotations.
- Kasmina: telemetry overlay.
- Emrakul/Elesh (future): use foldability descriptors at checkpoint.

Implementation Tasks (Speculative)
- Leyline RFC: Extend `BlueprintDescriptor` with scaffold fields (`supports_scaffold`, `ttl_ckpts`, `alpha_policy`, `success_gates`, `foldability_descriptor`).
- Karn: Populate new fields in templates where applicable; add conservative defaults.
- Urza: Persist and expose scaffold metadata; include in `UrzaRecord` JSON.
- Tezzeret: Accept paired manifests; add foldability descriptor to artifact manifest without changing compilation.
- Tamiyo: Allow requesting scaffold class via command annotations; surface persist/sunset hints; log scaffold budgets in telemetry.
- Kasmina: Emit scaffold gauges (alpha, weaning flags) in telemetry; ensure host branch uses `.detach()` during blending (already required).
- Oona/Nissa: Add scaffold lifecycle topics and dashboards for `scaffold.*` events and gauges.
- Docs: Add operator runbook notes for scaffold TTL, exit gates, and telemetry interpretation.
