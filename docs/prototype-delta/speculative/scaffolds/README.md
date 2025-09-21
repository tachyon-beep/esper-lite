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
