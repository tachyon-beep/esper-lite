## Task: Telemetry Topology Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/topology-findings.md`

Read-only scope:

- `src/esper/leyline/telemetry.py`
- `src/esper/nissa/`
- `src/esper/scripts/train.py`
- telemetry-related exports in `src/esper/leyline/__init__.py`, `src/esper/nissa/__init__.py`, and `src/esper/simic/telemetry/__init__.py`

Goal:

- Inventory `TelemetryEventType`, payload dataclasses, event envelope shape, Nissa hub/backends, CLI telemetry flags, and the high-level producer -> hub -> backend/store/UI graph.
- Identify event types with no obvious producer or no obvious consumer.
- Identify payloads whose fields are not preserved by obvious backends.
- Produce a Mermaid data-flow diagram.

Required output:

- Feed inventory table.
- Data-flow diagram.
- Findings with failure mode, severity, evidence file/line, and tracker-ready title.
- Note uncertainties separately from confirmed defects.

Constraints:

- Do not edit source.
- Do not rely on stale docs without verifying current files.
- Avoid defensive-programming recommendations that hide bugs; prefer fail-loud schema/contract fixes.

