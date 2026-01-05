# Phase 0 (A0 + C0): Minimal Submodule Intervention

**Objective:** Ship the smallest end-to-end slice where Tamiyo can make **more frequent, less weighty** interventions without new host routing:

- **A0 (surfaces):** use *existing* injection surfaces, but allow **more** of them via deeper presets + more enabled slots.
- **C0 (microstructure):** add a single **ladder seed** (`conv_ladder`) whose capacity is controlled via **two new lifecycle ops** and one new Obs scalar.

This phase is the “dev on-ramp”: it is intentionally contract-heavy (no backcompat), but mechanically minimal.

## Canonical references

- Roadmap source of truth: `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md` (Phase 0 section)
- Cross-domain sprawl map: `docs/plans/planning/kasmina2/planning_footprint.md`

## Phase 0 artifacts (implementation checklists)

- `docs/plans/planning/kasmina2/phases/phase0_a0c0/scope.md`
- `docs/plans/planning/kasmina2/phases/phase0_a0c0/leyline_contracts.md`
- `docs/plans/planning/kasmina2/phases/phase0_a0c0/kasmina_mechanics.md`
- `docs/plans/planning/kasmina2/phases/phase0_a0c0/tamiyo_obs_and_masks.md`
- `docs/plans/planning/kasmina2/phases/phase0_a0c0/simic_execution_and_reward.md`
- `docs/plans/planning/kasmina2/phases/phase0_a0c0/tolaria_runtime_tasks.md`
- `docs/plans/planning/kasmina2/phases/phase0_a0c0/telemetry_and_dashboards.md`
- `docs/plans/planning/kasmina2/phases/phase0_a0c0/experiments_and_gates.md`

## Non-negotiables (must remain true)

- **Sensors match capabilities** (`ROADMAP.md` commandment #1): internal levers require telemetry + Obs V3 support.
- **No legacy/backcompat** (`CLAUDE.md`): changing enums/dims requires updating all call sites in the same PR; no dual paths.
- **Fail fast, don’t mask bugs:** avoid `.get()`, `getattr()`, `hasattr()`, silent excepts as “fixes”.
- **GPU-first inverted control flow:** no new Python bottlenecks in `src/esper/simic/training/vectorized.py`.
- **DDP symmetry invariants:** internal ops must not allow rank divergence (`src/esper/kasmina/slot.py`).
- **Metaphors:** organism for domains; botanical only for seed lifecycle.

## Breaking-change warning (explicit)

Phase 0 adds **two new `LifecycleOp`s**. That increases `NUM_OPS`, which changes:

- the **Obs V3 base feature width** (last_action_op one-hot),
- the **op head size** in Tamiyo’s policy network,
- and therefore **checkpoint compatibility** (policy weights, rollout buffers, and any cached dim constants).

This phase must be shipped as a single contract update across Leyline → Kasmina → Tamiyo → Simic → Tolaria/runtime → Karn/Nissa.

