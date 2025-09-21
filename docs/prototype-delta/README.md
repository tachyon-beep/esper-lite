# Prototype Delta (Esper‑Lite)

Purpose
- Capture the differences between the current prototype and the full design, subsystem by subsystem. This folder is documentation only; no source code is modified.

Method
- Extract requirements from `docs/design/detailed_design/*`.
- Compare against `src/esper/*` and `tests/*`.
- Record status per requirement using the rubric in `rubric.md` and link evidence.

Principles
- Leyline is the single authoritative source of truth for all data classes and enums (no local mappings).
- UK English and concise, actionable writing.
- Out‑of‑scope components are marked explicitly with placeholders.

Subsystem Index (in scope unless noted)
- `kasmina/` — Execution layer: lifecycle, gates, isolation, safety.
- `tolaria/` — Training orchestrator: epoch budget, LR controller, rollback.
- `tamiyo/` — Strategic controller: policy, risk engine, field reports.
- `simic/` — Offline trainer: replay, PPO/IMPALA, validation, updates.
- `karn/` — Phase‑1 static templates: catalog, selection, enforcement.
- `tezzeret/` — Compilation forge: torch.compile pipelines, WAL, telemetry.
- `urza/` — Kernel library: catalog, caching, integrity, query SLOs.
- `oona/` — Messaging bus: Redis Streams, priority routing, backpressure.
- `nissa/` — Observability: telemetry ingestion, alerts, SLOs, API.
- `leyline/` — Canonical contracts (schema evolution plan included).
- `security/` — Cross‑cut: HMAC signing, replay protection (adoption roadmap).

Out‑of‑Scope Placeholders (prototype excludes these)
- `urabrask/` — Safety validation and performance benchmarking (not in prototype).
- `jace/` — Testing frameworks and SLO framework (not in prototype).
- `emrakul/` — System‑level orchestration (not in prototype).
- `elesh/` — Importance tracking and analysis (not in prototype).

Speculative (Research‑Driven Additions)
- `speculative/` — Curated proposals framed as design‑style deltas with low‑risk adoption notes and concrete implementation task lists:
  - `scaffolds/` — Temporary, foldable blueprints as a policy overlay
  - `tamiyo-narset-split/` — Strategic vs tactical controllers, protected zones
  - `mycosynth-config-fabric/` — Centralised runtime configuration fabric
  - `future-pruning/` — Hardware‑aware pruning & torch.ao integration
  - `bsds-lite/` — Lightweight Blueprint Safety Data Sheet metadata

Quick Status Snapshot
- Implemented core deltas for Kasmina, Tolaria, Tamiyo, Simic, Karn (Phase‑1), Tezzeret, Urza, Oona, Nissa.
- Leyline delta defines a single, breaking update to the lifecycle (11 stages + gates) with no tech debt.
- Security delta documents current HMAC coverage (Oona) and the adoption plan for nonce/freshness and command‑path signing.

Notes
- Evidence uses repository‑relative paths; these are clickable in most editors.
- This folder is for engineering coordination; keep research paper content separate under `docs/paper/`.
