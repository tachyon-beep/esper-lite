# Speculative Delta — BSDS‑Lite (Blueprint Safety Metadata)

Design anchor: docs/design/detailed_design/research_concepts/bsds/bsds.md

Summary
- Adopt a lightweight subset of the Blueprint Safety Data Sheet (BSDS) within existing `BlueprintDescriptor` metadata (Urza/Karn), enabling Tamiyo/Kasmina to apply basic risk‑aware policies without introducing Urabrask.

Proposed Behaviour (Design‑style)
- Extend descriptors with fields mirroring BSDS essentials: `overall_risk_score`, `gradient_instability_flag`, `memory_consumption_class`, `numerical_stability_notes`, `recommended_grafting_strategy`, `resource_requirements`.
- Tamiyo consults these fields for initial gating and annotations; Kasmina logs warnings for known hazard classes in telemetry; Tezzeret includes BSDS‑Lite in artifact manifests.

Status (Prototype)
- Partial: `risk`, `stage`, `approval_required`, `quarantine_only` already exist; Tamiyo reads Urza metadata and can pause on high risk.

Adoption (Low‑risk, prototype)
- Add enum/score fields at descriptor level (no new service). Keep usage to annotations/telemetry; do not hard‑gate flows yet.

Cross‑Subsystem Impact
- Urza/Karn (descriptor shape), Tamiyo (gating hints), Kasmina (telemetry), Nissa (observability).

Implementation Tasks (Speculative)
- Leyline RFC: Extend `BlueprintDescriptor` with BSDS‑Lite fields (`overall_risk_score`, `gradient_instability_flag`, `memory_consumption_class`, `numerical_stability_notes`, `recommended_grafting_strategy`, `resource_requirements`).
- Karn: Populate BSDS‑Lite fields in templates where known; default unknowns to safe values.
- Urza: Persist and surface the new fields; expose in queries.
- Tamiyo: Read BSDS‑Lite in Urza metadata; adjust risk gating/annotations; pause high‑risk blueprints by policy.
- Kasmina: Emit warnings in telemetry when deploying high‑risk categories (telemetry only).
- Nissa: Add dashboards/alerts for high BSDS risk usage.
