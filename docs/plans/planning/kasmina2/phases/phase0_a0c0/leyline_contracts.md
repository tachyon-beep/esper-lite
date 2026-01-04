# Phase 0 Leyline Contract Deltas (A0 + C0)

This phase is contract-driven: **Leyline changes first**, then all consumers updated in the same PR (no backcompat).

## 1) Action space changes (Leyline)

### `src/esper/leyline/factored_actions.py`

**BlueprintAction (add one CNN blueprint):**

- Add enum member:
  - `CONV_LADDER = 13`
- Update `BlueprintAction.to_blueprint_id()` mapping:
  - `13: "conv_ladder"`
- Update `CNN_BLUEPRINTS` to include `BlueprintAction.CONV_LADDER`.

**LifecycleOp (add two internal microstructure ops):**

- Add enum members (append to preserve existing indices):
  - `GROW_INTERNAL = 6`
  - `SHRINK_INTERNAL = 7`
- Update derived constants and lookup tables:
  - `NUM_OPS = len(LifecycleOp)`
  - `OP_NAMES`, `OP_*` constants (add `OP_GROW_INTERNAL`, `OP_SHRINK_INTERNAL` if we keep the hot-path constant pattern).

**Causal relevance masks:**

- `src/esper/leyline/causal_masks.py` must treat internal ops as **slot-only** decisions:
  - relevant heads: `{ "op", "slot" }`.
- `compute_causal_masks()` already makes `slot` relevant for all non-WAIT ops; ensure `HEAD_RELEVANCE_BY_OP` explicitly includes:
  - `"GROW_INTERNAL": frozenset({"op", "slot"})`
  - `"SHRINK_INTERNAL": frozenset({"op", "slot"})`

## 2) Seed state reporting contract (Kasmina → Tamiyo)

### `src/esper/leyline/reports.py`

Add a shared enum for internal microstructure identity (Leyline owns shared contracts):

- `SeedInternalKind (IntEnum)`
  - `NONE = 0`
  - `CONV_LADDER = 1`
  - (reserved for later phases) `LORA_RANK_LADDER = 2`, `CHANNEL_GROUP_LADDER = 3`, `ATTN_HEAD_GROUP_LADDER = 4`

Extend `SeedStateReport` with internal microstructure fields:

- `internal_kind: SeedInternalKind = SeedInternalKind.NONE`
- `internal_level: int = 0`
- `internal_max_level: int = 1`

**Invariants (fail-fast, no feature-side guards):**

- `internal_max_level >= 1` always (even when `internal_kind=NONE`)
- `0 <= internal_level <= internal_max_level`
- If `internal_kind == NONE`: `internal_level == 0` and `internal_max_level == 1`

Optional (Phase 0.1, only if learnability needs it):

- `internal_active_params: int = 0` (active trainable params under current level mask)

## 3) Telemetry contracts (sensors match capabilities)

### `src/esper/leyline/telemetry.py`

Add a new event type:

- `TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED`

Add a typed payload dataclass (same pattern as existing seed lifecycle payloads):

- `SeedInternalLevelChangedPayload` (slots=True, frozen=True)
  - required:
    - `slot_id: str`
    - `env_id: int` (Kasmina emits `-1` sentinel, replaced by `emit_with_env_context`)
    - `blueprint_id: str`
    - `from_level: int`
    - `to_level: int`
    - `max_level: int`
  - optional:
    - `active_params: int | None = None`

## 4) Obs V3 constants (Leyline is the source of truth)

### `src/esper/leyline/__init__.py`

Phase 0 changes **both** op count and per-slot features:

- `NUM_OPS` increases by 2 → base feature vector must grow because of last_action_op one-hot.
- `OBS_V3_SLOT_FEATURE_SIZE` increases by 1 for `internal_level_norm`.

Required constant semantics:

- `OBS_V3_BASE_FEATURE_SIZE = 17 + NUM_OPS`
  - (Because base features are: 17 scalars + last_action_op one-hot of width NUM_OPS.)
- `OBS_V3_SLOT_FEATURE_SIZE = 31`
- `OBS_V3_NON_BLUEPRINT_DIM = OBS_V3_BASE_FEATURE_SIZE + (OBS_V3_SLOT_FEATURE_SIZE * DEFAULT_NUM_SLOTS)`

## 5) Downstream consumers that must be updated (no partial updates)

When Phase 0 contracts change, update all downstream call sites:

**Kasmina (mechanics):**

- `src/esper/kasmina/slot.py` (`SeedState.to_report()` populates new fields; internal ops update internal state; emit new event type)
- `src/esper/kasmina/blueprints/cnn.py` (add `conv_ladder` blueprint implementation)

**Tamiyo (decision surface):**

- `src/esper/tamiyo/policy/features.py` (add `internal_level_norm` feature; update feature-size comments/assumptions)
- `src/esper/tamiyo/policy/action_masks.py` (mask GROW/SHRINK_INTERNAL when physically impossible)
- `src/esper/tamiyo/networks/factored_lstm.py` (op head size changes automatically via `get_action_head_sizes`; verify any shape assertions)

**Simic (training/reward):**

- `src/esper/simic/training/vectorized.py` (execute new ops; validate slot/seed supports internal levers; maintain last_action_* parity)
- `src/esper/simic/rewards/rewards.py` (ensure internal ops have defined intervention-cost semantics; update `INTERVENTION_COSTS`)

**Tolaria/runtime (execution surface):**

- `src/esper/scripts/train.py` (no new flags needed; but Phase 0 experiment pack uses `--slots` + `--task`/`--preset` combos)

**Karn/Nissa (memory/senses):**

- Any event-type enums/tables that assume TelemetryEventType is closed.
- Dashboards/views that group lifecycle events by type should include the new event.

