# Phase 0 Tamiyo: Obs V3 + Action Masks (A0 + C0)

Phase 0 adds one new per-slot observation scalar and two new lifecycle ops. Tamiyo must be able to (1) perceive the lever and (2) avoid physically impossible uses.

## 1) Obs V3 additions (features)

**Consumption point:** `src/esper/tamiyo/policy/features.py`

Add one per-slot scalar feature:

- `internal_level_norm: float`
  - derived from `SeedStateReport.internal_level / SeedStateReport.internal_max_level`
  - expected range: `[0.0, 1.0]` by contract (no feature-side clamping)

**Contract prerequisites (fail fast):**

- `SeedStateReport` includes:
  - `internal_level: int`
  - `internal_max_level: int` (must be ≥1 always)

**Dimensional consequences:**

- `OBS_V3_SLOT_FEATURE_SIZE` increases from 30 → 31.
- `OBS_V3_BASE_FEATURE_SIZE` also changes because `NUM_OPS` grows by 2 (last_action_op one-hot).
- Update Leyline constants (`src/esper/leyline/__init__.py`) and ensure all shape assertions in Simic/Tamiyo are updated in lockstep.

## 2) Action masking changes (only physically impossible actions)

**Consumption point:** `src/esper/tamiyo/policy/action_masks.py`

Add masking rules for:

- `LifecycleOp.GROW_INTERNAL`
- `LifecycleOp.SHRINK_INTERNAL`

Mask should be **optimistic across enabled slots** (consistent with existing masking):

- `GROW_INTERNAL` is valid if **any enabled slot** has a seed where:
  - `internal_kind != NONE`
  - `internal_level < internal_max_level`
  - seed stage is not terminal/frozen (recommended: disallow `FOSSILIZED`)
- `SHRINK_INTERNAL` is valid if **any enabled slot** has a seed where:
  - `internal_kind != NONE`
  - `internal_level > 0`
  - stage not terminal/frozen

To support that without importing Kasmina, extend the masking view of seed state:

- `MaskSeedInfo` should include:
  - `internal_kind: int` (store `SeedInternalKind.value`)
  - `internal_level: int`
  - `internal_max_level: int`

## 3) Causal relevance (credit assignment)

No new action heads are introduced in Phase 0.

- internal ops should be treated as **slot-only** decisions:
  - causally relevant heads: `op`, `slot`
  - irrelevant heads: `blueprint`, `style`, `tempo`, `alpha_*`

Single source of truth remains `src/esper/leyline/causal_masks.py`.

## 4) Policy scalability (A0 slot count increase)

Phase 0 A0 can increase the number of enabled slots (e.g., 5 slots on `cifar_scale`):

- `src/esper/tamiyo/networks/factored_lstm.py` already sizes heads via `get_action_head_sizes(slot_config)`.
- Feature dims already support dynamic slot counts via `get_feature_size(slot_config)` in `src/esper/tamiyo/policy/features.py`.

Risk to track (but not solved in Phase 0):

- As slot count rises, flat concatenation + LSTM may saturate; this is the motivation for Phase α (Slot Transformer) in `docs/plans/planning/kasmina2/phases/phase4_slot_transformer/`.

