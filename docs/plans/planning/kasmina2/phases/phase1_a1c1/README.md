# Phase 1 (A1 + C1): Transformer Sublayer Surfaces + LoRA Ladder

This folder is a working plan surface. The phase details live in:

- `docs/plans/planning/kasmina2/kasmina_tamiyo_submodule_intervention_roadmap.md` (Phase 1 section)

## Objective

- **A1:** expose submodule-meaningful transformer injection surfaces per layer (e.g., `POST_ATTN`, `POST_MLP`) without hooks.
- **C1:** make transformer adaptation incremental via a LoRA rank ladder (avoid “jump to LORA_LARGE”).

## Anticipated contract deltas (Leyline-first)

- `src/esper/leyline/injection_spec.py`: add `InjectionSurface` + deterministic `InjectionSpec.order`.
- `src/esper/leyline/slot_config.py`: sort by `order`, carry surface/order metadata.
- `src/esper/leyline/reports.py`: surface metadata exposure (or via SlotConfig at env construction).
- `src/esper/leyline/__init__.py`: Obs V3 slot feature size updates for surface/order signals.

## Anticipated mechanics deltas

- `src/esper/kasmina/host.py:TransformerHost`: explicit routing at sublayer boundaries.
- `src/esper/kasmina/blueprints/transformer.py`: `lora_ladder` seed family with `internal_level`.

## Docs to write before coding Phase 1

- `scope.md` (what we will/won’t touch; compile/DDP assumptions)
- `leyline_contracts.md` (exact enum/field lists + consumer checklist)
- `kasmina_host_routing.md` (TransformerHost routing plan; deterministic slot IDs)
- `tamiyo_features_and_masks.md` (surface/order obs; masks for new surfaces)
- `experiments_and_gates.md` (ROI metrics; when to pivot to Phase α)

