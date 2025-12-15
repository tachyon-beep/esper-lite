# BUG Template

- **Title:** Nissa BlueprintAnalytics assumes env_id/params in seed events that aren’t guaranteed
- **Context:** `BlueprintAnalytics.emit` (`src/esper/nissa/analytics.py`) expects `env_id`, `params`, `params_added`, `epochs_total` in SEED_* events. Upstream emitters sometimes omit these fields or use different keys, leading to KeyError/default zeros and incorrect stats.
- **Impact:** P1 – analytics may miscount compute cost, ages, or fail when missing fields; dashboards/telemetry become misleading.
- **Environment:** Main branch; telemetry emitters in Kasmina/Simic vary per event; env_id often missing.
- **Reproduction Steps:** Emit SEED_FOSSILIZED without `env_id` or `params_added`; analytics will use default 0/env=0, skewing stats or printing wrong info.
- **Expected Behavior:** Analytics should handle missing fields robustly and/or validate contract; Leyline contract should guarantee required fields.
- **Observed Behavior:** Defaults to env_id=0 and params=0 silently; can KeyError if data missing (e.g., `event.data.get("params")` returns None and formatting expects number).
- **Hypotheses:** Event payloads not standardized; analytics assumes presence.
- **Fix Plan:** Align SEED_* telemetry with a defined Leyline schema; add validation/default handling in BlueprintAnalytics; enforce env_id and param fields at emit time.
- **Validation Plan:** Add tests emitting minimal seed events and ensure analytics handles gracefully; fix emitters to supply required fields.
- **Status:** Open
- **Links:** `src/esper/nissa/analytics.py` seed event handling; seed emitters in `kasmina/slot.py` and vectorized training
