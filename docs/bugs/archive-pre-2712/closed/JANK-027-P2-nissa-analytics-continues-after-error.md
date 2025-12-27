# JANK Template

- **Title:** BlueprintAnalytics swallows malformed events and continues with partial state
- **Category:** observability / resilience
- **Symptoms:** `BlueprintAnalytics.emit` assumes well-formed `TelemetryEvent.data`; malformed fields could lead to mis-attribution (e.g., `"unknown"` blueprint_id) or backend exceptions.
- **Impact:** Low/Medium – malformed events can degrade analytics fidelity; training should continue.
- **Triggers:** Missing/incorrect fields in SEED_* events; unexpected event types.
- **Root-Cause Hypothesis:** Minimal validation for performance; assumes well-formed events.
- **Remediation Options:** Add schema validation/logging; skip bad events with warnings; enforce event contracts upstream.
- **Validation Plan:** Add tests emitting malformed events to ensure analytics logs/ignores safely; ensure state remains consistent.
- **Status:** Closed (Mitigated)
- **Resolution:** `NissaHub.emit` isolates backend exceptions and logs errors per-backend, so malformed events can’t crash training or silently stop other telemetry. Lifecycle event emitters use stable internal schemas; treat malformation as an upstream emitter bug rather than adding heavy validation in the analytics backend.
- **Links:** `src/esper/nissa/analytics.py` (`BlueprintAnalytics.emit`), `src/esper/nissa/output.py` (`NissaHub.emit` exception isolation)
