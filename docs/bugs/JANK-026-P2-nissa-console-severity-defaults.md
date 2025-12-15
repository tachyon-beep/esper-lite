# JANK Template

- **Title:** Nissa ConsoleOutput severity handling defaults to info, can drop important events
- **Category:** observability / ergonomics
- **Symptoms:** ConsoleOutput uses `_SEVERITY_ORDER` and defaults `min_severity="info"`. Many telemetry events use `severity=None` or omit severity, mapping to default order 1, and debug-level events (e.g., REWARD_COMPUTED telemetry) are suppressed. Users may miss critical signals unless they manually set verbose/min_severity, but CLI/config doesn’t expose this for the console backend.
- **Impact:** Medium – important warnings/debug telemetry can be silently hidden; inconsistent behavior across backends.
- **Triggers:** Default console backend usage (common in scripts/train).
- **Root-Cause Hypothesis:** Console defaults chosen to reduce noise; not configurable via telemetry profiles.
- **Remediation Options:**
  - A) Wire min_severity/verbose through TelemetryConfig/profile to console backend.
  - B) Set sensible defaults per profile (normal/debug).
  - C) Ensure missing severity defaults to info or higher consistently and log when events are dropped.
- **Validation Plan:** Add tests for severity filtering; ensure profile changes affect console backend.
- **Status:** Open
- **Links:** `src/esper/nissa/output.py::ConsoleOutput`, `profiles.yaml`
