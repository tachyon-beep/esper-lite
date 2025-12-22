# BUG-025: Unvalidated stage ints allow retired/reserved SeedStage values into telemetry/features

- **Title:** `SeedTelemetry.from_dict()` accepts any `stage` int (including retired/reserved values), enabling invalid stage values to enter runtime telemetry and derived feature vectors
- **Context:** Leyline `SeedTelemetry` is a core contract feeding Tamiyo/Simic observability and (optionally) feature construction. Stage is represented as an `int` in telemetry for speed/compactness.
- **Impact:** P2 – correctness + debugging risk. Invalid stage values can silently poison:
  - stage-derived telemetry features,
  - dashboards/analytics (stage counts, stage timelines),
  - any future offline replay tooling that reconstructs observations from telemetry.
- **Environment:** HEAD @ workspace; any code path that deserializes telemetry from dict/JSONL
- **Reproduction Steps:**
  1. Construct a telemetry dict with a reserved stage value, e.g. `{"seed_id": "...", "stage": 5, ...}` (5 is a retired/reserved gap in `SeedStage`).
  2. Call `SeedTelemetry.from_dict(data)` and observe it succeeds.
  3. Observe `to_features()` produces a plausible “stage feature” for an impossible stage value.
- **Expected Behavior:** Untrusted ingestion should validate that `stage` is a valid `SeedStage` value (and explicitly decide whether `UNKNOWN` is permitted). Reserved/out-of-range values should fail fast with an actionable error.
- **Observed Behavior:** `SeedTelemetry.from_dict()` uses `stage=data.get("stage", 1)` with no validation and therefore accepts any integer.
- **Logs/Telemetry:** None today; invalid values are silently accepted.
- **Hypotheses:** Telemetry was treated as “trusted internal” data; however, dict/JSON ingestion is untrusted by default (telemetry files, dashboards, replay tools, hand-written fixtures).
- **Fix Plan:**
  - Phase 0 (risk reduction): add strict validation in `SeedTelemetry.from_dict()`:
    - reject any `stage` that is not in the allowed `SeedStage` set (and handle `UNKNOWN` per policy),
    - add tests that explicitly reject the retired/reserved stage value and an out-of-range value.
  - (Follow-on) centralize stage validity + encoding under a Leyline StageSchema contract.
- **Validation Plan:**
  - Unit test: `stage=5` (reserved) is rejected.
  - Unit test: `stage=999` is rejected.
  - Unit test: valid `SeedStage` values round-trip via `to_dict()`/`from_dict()`.
- **Status:** Resolved
- **Links:**
  - Ingestion: `src/esper/leyline/telemetry.py:253`
  - Enum: `src/esper/leyline/stages.py:7`
  - Plan: `docs/plans/2025-12-22-stage-schema-hardening.md`

## Fix Implemented

- `SeedTelemetry.from_dict()` now validates `stage` is a valid `SeedStage` value and raises `ValueError` on reserved/out-of-range values.
- Added tests covering `stage=5` (reserved gap) and an out-of-range stage value.
