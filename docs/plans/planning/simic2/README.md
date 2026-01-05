# Simic2 Planning Workspace

This folder is the *working planning surface* for a maintainability refactor of **Simic** (Evolution) with an emphasis on:

- `src/esper/simic/training/vectorized.py` (4.4k LOC, nested helpers, mixed responsibilities)
- Simic’s large modules (`agent/ppo.py`, `rewards/rewards.py`) that are hard to test and modify safely

This is a **mechanical refactor** plan: preserve behavior, throughput, and telemetry semantics while making the code easier to change.

## Contents

- `docs/plans/planning/simic2/simic_vectorized_refactor_roadmap.md`: the phased roadmap (what we do and why).
- `docs/plans/planning/simic2/planning_footprint.md`: the “plan-of-plans” / sprawl map (what changes where, and how we keep it shippable).
- `docs/plans/planning/simic2/phases/`: phase-by-phase execution workspace (checklists and acceptance criteria).

## How this becomes “dev-ready”

- Use this folder to converge on **Phase 1** (vectorized split) as the first implementation slice.
- Once a phase’s scope is accepted, promote it to a single-file implementation plan under `docs/plans/ready/`.
- **No backcompat**: if a phase changes a contract or entrypoint, the plan must list *all* call sites that change and remove the old path in the same PR.

