# Kasmina2 Planning Workspace

This folder is the *working planning surface* for moving Kasmina + Tamiyo from **module-level** intervention to **submodule-level** intervention along two tracks:

- **Track A (surfaces):** more injection surfaces (denser / finer injection points)
- **Track C (microstructure):** lighter seeds with internal levers (so Tamiyo isn’t forced to “buy a conv_heavy” to get signal)

This effort is intentionally cross-domain (Leyline + Kasmina + Tamiyo + Simic + Tolaria, plus Nissa/Karn for sensors).

## Contents

- `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md`: the current expanded roadmap (copied from `docs/research/kasmina_tamiyo_submodule_intervention_roadmap.md`).
- `docs/plans/planning/kasmina2/planning_footprint.md`: the “plan-of-plans” / impact map (what changes where, and how we keep it shippable).
- `docs/plans/planning/kasmina2/phases/`: phase-by-phase execution workspace; Phase 0 is decomposed into per-domain checklists.

## How this becomes “dev-ready”

- Use this folder to iterate and converge on **Phase 0 (A0+C0)** as a minimal, implementation-ready first slice.
- Once a phase’s scope is accepted, promote it to a single-file implementation plan under `docs/plans/ready/` (no backcompat, so the plan must list *all* call sites that change).
- Keep `docs/research/` as the research archive; keep `docs/plans/` as the execution surface.
