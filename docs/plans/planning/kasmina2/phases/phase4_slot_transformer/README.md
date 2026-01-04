# Phase 4 (Phase α): Slot Transformer Policy Pivot

This folder is a working plan surface. The phase details live in:

- `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md` (Phase 4 / Phase α section)

## Objective

As slot count and/or subtarget inventories grow, flat concatenation becomes a bottleneck.

Phase α replaces the “flat features + LSTM” representation with a tokenized slot encoder:

- per-slot tokens (features + metadata)
- attention over slots (slot transformer)
- optional recurrence over timesteps

This is the scalability escape hatch for Track A density.

## Docs to write before coding Phase α

- `representation_contract.md` (what a “slot token” contains; how metadata is embedded)
- `leyline_contracts.md` (any new constants, dims, masks)
- `tamiyo_network_design.md` (architecture; throughput constraints)
- `experiments_and_gates.md` (scaling curves; memory/latency budgets; stability)

