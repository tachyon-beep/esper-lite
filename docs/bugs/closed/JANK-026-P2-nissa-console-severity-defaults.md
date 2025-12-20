# JANK Template

- **Title:** Nissa ConsoleOutput severity handling defaults to info, can drop important events
- **Category:** observability / ergonomics
- **Symptoms:** ConsoleOutput filters by `_SEVERITY_ORDER` and defaults `min_severity="info"`, suppressing debug events unless configured.
- **Impact:** Medium – important warnings/debug telemetry can be silently hidden; inconsistent behavior across backends.
- **Triggers:** Default console backend usage (common in scripts/train).
- **Root-Cause Hypothesis:** Console defaults chosen to reduce noise; not configurable via telemetry profiles.
- **Remediation Options:**
  - A) Wire min_severity/verbose through TelemetryConfig/profile to console backend.
  - B) Set sensible defaults per profile (normal/debug).
  - C) Ensure missing severity defaults to info or higher consistently and log when events are dropped.
- **Validation Plan:** Add tests for severity filtering; ensure profile changes affect console backend.
- **Status:** Closed (Resolved)
- **Resolution:** `scripts/train` maps `--telemetry-level` to `ConsoleOutput(min_severity=...)`, so users can opt into debug-level console output (`--telemetry-level debug`) without code changes. `TelemetryEvent.severity` defaults to `"info"`, so missing severity does not silently drop events.
- **Links:** `src/esper/nissa/output.py` (`ConsoleOutput`), `src/esper/scripts/train.py` (`--telemetry-level` → `min_severity`), `src/esper/leyline/telemetry.py` (`TelemetryEvent.severity`)
