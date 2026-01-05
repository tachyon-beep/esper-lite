# Phase 0 Kasmina Mechanics (A0 + C0)

Phase 0 introduces internal microstructure inside a single seed family and exposes it as two lifecycle ops. The mechanics must remain host-agnostic and DDP-symmetric.

## 1) Where internal microstructure state lives

**Source of truth:** `src/esper/kasmina/slot.py` (`SeedState` + `SeedSlot`).

Add to `SeedState` (or to `SeedSlot` and mirrored into `SeedState.to_report()`):

- `internal_kind: SeedInternalKind`
- `internal_level: int`
- `internal_max_level: int`

Recommended invariant enforcement:

- initialize `internal_kind=NONE`, `internal_level=0`, `internal_max_level=1` for all non-ladder seeds.
- for `conv_ladder`, set `internal_kind=CONV_LADDER`, `internal_max_level=L`, and pick an initial `internal_level` that makes the seed non-degenerate (recommended: `1`).

## 2) C0 seed family: `conv_ladder`

**Location:** `src/esper/kasmina/blueprints/cnn.py`

**Definition:** a CNN seed that is shape-preserving and whose capacity scales with `internal_level`.

Minimal level ladder (recommended for C0):

- `internal_max_level = 2`
  - level 0: identity/no-op (parameters exist but inactive, or no active blocks)
  - level 1: equivalent compute tier to `conv_light` (one conv micro-block)
  - level 2: equivalent compute tier to `conv_heavy` (two conv micro-blocks)

Rationale:

- Matches the existing “light vs heavy” decision boundary but makes it *incremental and reversible*.
- Keeps `torch.compile` specialization bounded (≤3 variants).

## 3) Internal ops semantics (executed by SeedSlot)

**New lifecycle ops:** in `src/esper/leyline/factored_actions.py:LifecycleOp`

- `GROW_INTERNAL`: `internal_level += 1` (clamped to `internal_max_level`)
- `SHRINK_INTERNAL`: `internal_level -= 1` (clamped to `0`)

**Physical validity rules (mechanics-level, fail-fast):**

- op is invalid if slot has no active seed (`SeedSlot.state is None`)
- op is invalid if `internal_kind == NONE`
- op is invalid if it would not change state (e.g., GROW at max level, SHRINK at 0)
- op is invalid if seed is terminal/frozen (recommended: disallow for `SeedStage.FOSSILIZED`)

**Parameter accounting hook (rent learnability):**

- On every internal level change, update:
  - `SeedState.metrics.seed_param_count` to reflect *active* trainable params at the new level.
  - (Optional) a per-seed telemetry field / report field `internal_active_params` if we need explicit dashboards.

Implementation constraint:

- Do not do optimizer param-group surgery. If trainable set changes, use:
  - “active blocks participate in forward”
  - and optionally `requires_grad` toggles on inactive blocks.

## 4) Telemetry emission (sensors match capabilities)

Emit a new typed telemetry event from `src/esper/kasmina/slot.py` when an internal op succeeds:

- `TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED`
- payload: `SeedInternalLevelChangedPayload(from_level, to_level, max_level, ...)`
- `env_id=-1` sentinel (replaced by `emit_with_env_context` in `src/esper/simic/telemetry/emitters.py`).

This event is how we debug:

- action thrash (grow/shrink oscillation)
- internal lever usage rates
- correlations between internal growth and contribution/ROI

## 5) Fossilization semantics (Phase 0)

Keep Phase 0 conservative:

- **FOSSILIZED** seeds become **internally frozen**:
  - internal ops are masked and/or rejected.
  - `internal_level` remains visible in reports/telemetry.

Deferral (explicit):

- “merge microstructure into host” vs “keep as a fossilized seed with internal masking” is deferred to Phase 5. In Phase 0, fossilization is “keep”.

## 6) Safety constraints

- **DDP symmetry:** internal ops must go through the same synchronized action pathway as other lifecycle ops. No rank-local mutation.
- **torch.compile behavior:** internal_level changes are allowed to create a small number of specialized graphs. Keep `internal_max_level` small.
- **Throughput:** internal ops are epoch-boundary operations; avoid per-sample overhead inside the forward path beyond the additional active micro-blocks.

