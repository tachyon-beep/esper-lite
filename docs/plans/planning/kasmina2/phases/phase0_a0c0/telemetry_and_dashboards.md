# Phase 0 Telemetry & Dashboards (Sensors Match Capabilities)

Phase 0 adds an internal lever (`internal_level`) and must ship:

- a typed telemetry event for every lever movement
- an Obs V3 scalar so Tamiyo can learn the lever

## 1) New telemetry event

**Contract:** `src/esper/leyline/telemetry.py`

- `TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED`
- `SeedInternalLevelChangedPayload` fields:
  - `slot_id: str`
  - `env_id: int` (Kasmina emits `-1` sentinel; replaced by `emit_with_env_context`)
  - `blueprint_id: str`
  - `from_level: int`
  - `to_level: int`
  - `max_level: int`
  - optional: `active_params: int | None`

**Emission point:** `src/esper/kasmina/slot.py` (when internal op succeeds).

**Env context injection:** `src/esper/simic/telemetry/emitters.py:emit_with_env_context` (existing pattern for seed lifecycle events).

## 2) Obs V3 support for the lever

**Contract fields:** `src/esper/leyline/reports.py:SeedStateReport`

- `internal_level`
- `internal_max_level`

**Consumption point:** `src/esper/tamiyo/policy/features.py`

- add per-slot scalar: `internal_level_norm`

## 3) Dashboards/views to watch (Phase 0)

Primary “learnability” views (Karn/Sanctum/Overwatch):

- Internal lever usage:
  - count/rate of `SEED_INTERNAL_LEVEL_CHANGED` over time
  - thrash metric: level changes per 100 steps
- Decision quality:
  - invalid-action rate (`last_action_success` false frequency)
  - op head entropy / collapse signals
- Economy:
  - effective seed params (rent proxy) vs accuracy gain (ROI curve)
- Safety:
  - `TelemetryEventType.GOVERNOR_ROLLBACK` frequency and trigger reasons

## 4) Minimal instrumentation checklist (Phase 0)

- Event type is present in storage and UI (at minimum: visible in raw event tables).
- Payload fields are queryable and consistent with slot/seed identity.
- Obs includes `internal_level_norm` and its value matches telemetry changes (spot check).

## 5) Review notes from `docs/plans/planning/kasmina2/chatgpt_feedback.md` (useful, but optional)

The feedback doc mostly overlaps with the roadmap, but two points are worth carrying forward explicitly:

- **C0 learnability risk:** `internal_level_norm` alone may be insufficient for the policy/critic to learn when grow/shrink is “doing work”. If internal ops are rarely used after Pack 0, the first upgrade should be *one more aggregated sensor* (not per-subtarget):
  - **Option A (cheap):** add `internal_active_params` so rent changes are immediately observable per slot (this is already listed as an optional report field in Phase 0 contracts).
  - **Option B (strong signal, watch overhead):** add a single activation-space “effect” norm (e.g., `seed_effect_norm = ||y - x||₂ / (||x||₂ + eps)`) emitted by Kasmina and surfaced as one extra Obs V3 per-slot scalar.
- **Thrash visibility:** make internal-level thrash visible as a derived metric (“level_changes_per_100_steps”) even if we keep Phase 0 contract changes minimal.
