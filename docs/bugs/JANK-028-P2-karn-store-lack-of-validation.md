# JANK Template

- **Title:** Karn store dataclasses accept arbitrary data with no validation/schema enforcement
- **Category:** correctness-risk / data integrity
- **Symptoms:** Karn store models (`store.py`) are plain dataclasses with no validation. Emitters can populate fields with wrong types/units (e.g., percentages vs fractions), missing env_ids, etc., and analytics/outputs consume them unchecked. No schema/contract enforcement beyond type hints.
- **Impact:** Medium – research telemetry can become inconsistent or misleading; downstream tools may break on unexpected types.
- **Triggers:** Malformed telemetry emission, changes in upstream payloads.
- **Root-Cause Hypothesis:** Simplicity; validation deferred.
- **Remediation Options:**
  - A) Add pydantic/dataclass validation or explicit `validate()` step before ingesting events into store.
  - B) Introduce lightweight schema checks and logging for invalid fields.
  - C) Provide canonical constructors from Leyline telemetry to ensure consistency.
- **Validation Plan:** Add tests feeding malformed data to store constructors and ensure validation catches issues; log warnings instead of silently accepting.
- **Status:** Open
- **Links:** `src/esper/karn/store.py` dataclasses
