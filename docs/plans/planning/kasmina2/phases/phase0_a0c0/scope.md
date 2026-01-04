# Phase 0 Scope (A0 + C0)

## Objective

Make it possible for Tamiyo to “turn a dial” instead of “buying a whole module”:

- **A0:** enable more injection points *without* changing host routing semantics (no new surfaces yet).
- **C0:** introduce one microstructured seed family (`conv_ladder`) with a single internal state (`internal_level`) controlled by two new lifecycle ops.

The target symptom to fix is **conv_heavy dominance**: the policy currently prefers heavyweight seeds because they produce the most immediate signal per intervention given coarse surfaces and step-function blueprint capacity.

## In scope (Phase 0)

**Track A0 (surfaces, minimal):**

- Use existing host injection boundaries (`src/esper/kasmina/host.py`).
- Increase surface availability by:
  - choosing deeper presets (e.g., `cifar_scale` in `src/esper/runtime/tasks.py`), and/or
  - enabling more slots via CLI `--slots` (SlotConfig remains dynamic).

**Track C0 (microstructure, minimal):**

- Add `conv_ladder` blueprint (CNN topology) whose capacity can be adjusted without re-germinating a different blueprint.
- Add two lifecycle ops:
  - `LifecycleOp.GROW_INTERNAL`
  - `LifecycleOp.SHRINK_INTERNAL`
- Add one Obs V3 per-slot scalar:
  - `internal_level_norm` (computed as `internal_level / internal_max_level`).
- Add one typed telemetry event:
  - `TelemetryEventType.SEED_INTERNAL_LEVEL_CHANGED`.

## Out of scope (explicit)

- No new host injection surfaces (`InjectionSurface`, `InjectionSpec.order`) — that starts Phase 1.
- No per-subtarget addressing (Phase 3).
- No Slot Transformer policy pivot (Phase α / Phase 4 folder).
- No new action heads (Phase 3).
- No backcompat shims for old enum values, obs dims, or checkpoint formats.

## Architectural boundaries (where each change belongs)

- **Leyline (contracts):** new enums/fields/constants (`src/esper/leyline/*`).
- **Kasmina (mechanics):** internal ops execution + ladder seed module (`src/esper/kasmina/*`).
- **Tamiyo (decisions):** features and masks reflect new levers (`src/esper/tamiyo/policy/*`).
- **Simic (training/reward):** executes new ops and accounts for them in reward/telemetry (`src/esper/simic/*`).
- **Tolaria/runtime:** exposes experiment presets and constructs models (`src/esper/runtime/tasks.py`, `src/esper/scripts/train.py`).
- **Nissa/Karn:** stores and displays new event type and derived analytics.

## Definition of Done (Phase 0)

End-to-end, for CIFAR tasks:

- Tamiyo can **germinate `conv_ladder`** and later **grow/shrink internal_level**.
- Obs V3 includes `internal_level_norm` and action masking does not explode invalid-action rates.
- Telemetry captures every internal level change with enough context to debug and learn from it.
- Rent/reward reflects the cost/benefit of internal growth (ROI-style evaluation is possible).

## Phase gate to unlock Phase 1

On a small cohort of runs (see `experiments_and_gates.md`):

- Internal ops are used non-trivially (≥10% of non-WAIT decisions **when** ladder seed is present).
- Conv-heavy selection frequency drops without a worse ROI curve (accuracy gain per added params/compute).
- No sustained increase in invalid-action rates or governor rollback events.

