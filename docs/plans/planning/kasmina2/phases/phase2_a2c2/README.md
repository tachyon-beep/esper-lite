# Phase 2 (A2 + C2): CNN Pre/Post-Pool Surfaces + Channel-Group Ladder

This folder is a working plan surface. The phase details live in:

- `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md` (Phase 2 section)

## Objective

- **A2:** add semantically meaningful CNN surfaces without hooks (e.g., `PRE_POOL`, `POST_POOL` per block).
- **C2:** add a channel-group ladder seed so Tamiyo can grow/shrink capacity *within* one seed.

## Expected risks

- Slot explosion (more surfaces) increases obs size and credit assignment difficulty.
- Must preserve channels_last routing invariants in `src/esper/kasmina/host.py:CNNHost`.

## Docs to write before coding Phase 2

- `leyline_contracts.md` (InjectionSurface additions; SlotConfig ordering; Obs V3 fields)
- `cnn_host_routing.md` (explicit split points; no hooks; memory format handling)
- `seed_design.md` (channel-group ladder invariants; parameter accounting; fossilization semantics)
- `experiments_and_gates.md` (ROI per FLOP/param; throughput checks)

