# JANK Template

- **Title:** BlueprintAnalytics swallows malformed events and continues with partial state
- **Category:** observability / resilience
- **Symptoms:** `BlueprintAnalytics.emit` assumes certain fields and does no validation; if an event is malformed, it may raise and stop emitting or silently mis-update stats. There’s no isolation/logging for bad events, so analytics state can become inconsistent without clear signals.
- **Impact:** Medium – dashboards and analytics can drift silently or crash, hiding telemetry issues.
- **Triggers:** Missing/incorrect fields in SEED_* events; unexpected event types.
- **Root-Cause Hypothesis:** Minimal validation for performance; assumes well-formed events.
- **Remediation Options:** Add schema validation/logging; skip bad events with warnings; enforce event contracts upstream.
- **Validation Plan:** Add tests emitting malformed events to ensure analytics logs/ignores safely; ensure state remains consistent.
- **Status:** Open
- **Links:** `src/esper/nissa/analytics.py` emit method
