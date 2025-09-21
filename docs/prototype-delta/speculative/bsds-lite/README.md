# Speculative Delta — BSDS‑Lite (Blueprint Safety Metadata)

Design anchor: docs/design/detailed_design/research_concepts/bsds/bsds.md

Intent
- Provide a lightweight, immediately useful subset of Urabrask’s Blueprint Safety Data Sheet (BSDS) as additional fields on Leyline `BlueprintDescriptor`, so controllers have consistent safety signals before full Urabrask exists.
- Be clear that full fidelity ultimately requires “implement Urabrask” (evaluation harness + risk characterisation). BSDS‑Lite is an interim, metadata‑first path.

Scope (v1)
- Descriptive, low‑entropy safety fields and recommended handling patterns for each blueprint.
- Consumers: Tamiyo (gating hints), Kasmina (warnings/telemetry), Nissa (dashboards/alerts). Urza/Karn own storage/sourcing.
- No hard runtime gates in v1; controllers use the metadata for policy hints and operator visibility.

Data Model (Leyline additions)
- Extend `BlueprintDescriptor` with BSDS‑Lite fields; all optional with sane defaults:
  - `overall_risk_score: float` (0.0–1.0)
  - `hazard_gradient_instability: enum {LOW, MODERATE, HIGH, CRITICAL}`
  - `hazard_memory_consumption: enum {LOW, MODERATE, HIGH, CRITICAL}`
  - `hazard_numerical_stability: enum {LOW, MODERATE, HIGH, CRITICAL}`
  - `perf_latency_p95_ms: float` (reference batch profile)
  - `perf_throughput_units: float` (domain‑specific units)
  - `recommended_grafting_strategy: enum {STANDARD, DRIFT_CONTROLLED, SLOW_RAMP, CANARY_ONLY}`
  - `resource_requirements: map<string,string>` (e.g., `{"vram_gb_min": "24", "grad_clip_norm": "1.0"}`)
  - `incompatible_pairings: repeated string` (ids/tags)
  - `notes_numerical_precision: string` (e.g., “avoid pure FP16; use bfloat16”)
  - `bsds_provenance: enum {CURATED, HEURISTIC, URABRASK}` (source of the data)
  - `bsds_version: string` (schema/curation version)

Scoring & Categories
- Overall risk score bands: LOW(≤0.3), MODERATE(≤0.6), HIGH(≤0.8), CRITICAL(>0.8).
- Hazard enums are categorical summaries for fast policy checks; do not replace raw metrics if available later.

Sourcing (before Urabrask)
- CURATED: manual entries per template (Karn templates) by subsystem owners; reviewed alongside design docs.
- HEURISTIC: short, scripted soak tests during Tezzeret compilation or CI (latency, simple stress patterns) to fill perf fields; kept conservative.
- URABRASK (future): authoritative risk sheets from Urabrask crucible; will supersede curated/heuristic entries.

Subsystem Behaviours (v1)
- Karn: populate BSDS‑Lite for known templates; default unknowns to LOW/MODERATE with clear provenance.
- Urza: persist fields as part of descriptor JSON; allow queries by risk bands/hazard classes.
- Tezzeret: pass through descriptor fields; optionally attach perf numbers (if collected) to artifact manifests.
- Tamiyo: use bands for policy hints (e.g., HIGH/CRITICAL → prefer DRIFT_CONTROLLED or PAUSE under loss spikes); annotate AdaptationCommands with risk and chosen mitigation; no hard block.
- Kasmina: add telemetry warnings on deploy if CRITICAL hazards present; surface recommended grafting strategy in metrics/events; do not alter lifecycle.
- Nissa: dashboards for risk distributions, deployments by hazard class, and alerts when CRITICAL deployments occur.

Operator Guidance (v1)
- HIGH/CRITICAL risk or CRITICAL hazard flags should create operator visibility and require a rationale for deployment in production.
- Use recommended grafting strategy when present (e.g., DRIFT_CONTROLLED implies slower alpha schedule).

Migration Path (towards full Urabrask)
1) Add BSDS‑Lite fields (CURATED/HEURISTIC provenance); wire consumers and dashboards.
2) Build Urabrask v0: minimal evaluation harness to produce perf/hazard snippets for a subset of templates; provenance=URABRASK for those.
3) Expand coverage; make URABRASK the default provenance; CURATED entries shrink over time.
4) Optional: add a separate Leyline `BSDS` message if sheets become large; keep BSDS‑Lite on descriptors for fast policy checks.

Telemetry & Alerts (Nissa)
- Metrics: `karn.blueprint.bsds.overall_risk`, `deployments_by_hazard{class}`, `recommended_grafting_strategy{strategy}`.
- Alerts: CRITICAL hazard deployments, repeated deployments of HIGH risk without improvements, mismatch between recommended strategy and observed Kasmina alpha policy.

Security & Provenance
- Include `bsds_provenance` and `bsds_version` in descriptors; surface in telemetry for auditability.
- If/when Urabrask exists, sign sheets (payload signatures via Leyline security plan) and mark descriptors as URABRASK‑derived.

Acceptance Criteria (v1)
- Fields present in Leyline `BlueprintDescriptor` and populated for top 20 templates.
- Urza stores/exposes fields; Tamiyo annotates commands; Kasmina emits warnings; Nissa dashboards/alerts live.
- No hard runtime gate; deployment behaviour unchanged besides visibility and suggested strategies.

Out‑of‑Scope (but relevant)
- Full “implement Urabrask”: evaluation crucible, hazard prediction models, full BSDS generation. BSDS‑Lite anticipates this by aligning field names and bands with the intended Urabrask schema.

Implementation Tasks (Speculative)
- Leyline RFC: Extend `BlueprintDescriptor` with BSDS‑Lite fields (`overall_risk_score`, `gradient_instability_flag`, `memory_consumption_class`, `numerical_stability_notes`, `recommended_grafting_strategy`, `resource_requirements`).
- Karn: Populate BSDS‑Lite fields in templates where known; default unknowns to safe values.
- Urza: Persist and surface the new fields; expose in queries.
- Tamiyo: Read BSDS‑Lite in Urza metadata; adjust risk gating/annotations; pause high‑risk blueprints by policy.
- Kasmina: Emit warnings in telemetry when deploying high‑risk categories (telemetry only).
- Nissa: Add dashboards/alerts for high BSDS risk usage.
