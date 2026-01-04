# Phase 0 Tolaria/runtime: Task & Slot Surfaces (A0)

Phase 0 does not change host routing. It uses existing topology knobs and slot enablement to increase surfaces.

## A0 surface density choices (no host changes)

### CNN: deeper host + more enabled slots

Ground truth:

- Injection points are block boundaries from `src/esper/kasmina/host.py:CNNHost.injection_specs()`.
- The deeper CNN preset already exists: `src/esper/runtime/tasks.py` → `cifar_scale` uses `n_blocks=5`.

To actually expose more injection points to Tamiyo:

- pass more slots via CLI `--slots` (e.g., 5 slots: `r0c0 r0c1 r0c2 r0c3 r0c4`)
- ensure the training config/preset matches the slot list (Simic derives SlotConfig from enabled injection specs).

### Transformer: per-layer segments (optional A0)

Ground truth:

- Transformer segments are controlled by `num_segments` in `src/esper/kasmina/host.py:TransformerHost`.
- Setting `num_segments = n_layer` yields per-layer boundaries (A0 candidate).

Phase 0 recommendation:

- Keep transformer A0 as optional; primary Phase 0 experiment pack is CIFAR.
- If we want A0 transformer smoke tests, add a new TaskSpec (e.g., `tinystories_layerwise`) that sets `num_segments=n_layer` in `src/esper/runtime/tasks.py`. (Implementation is deferred until Phase 0 is accepted for coding.)

## Stable slot ordering (must not drift)

Slot ordering is derived from host injection specs:

- `src/esper/kasmina/host.py:MorphogeneticModel` uses `host.injection_specs()` order.
- `src/esper/simic/training/vectorized.py` derives `SlotConfig.from_specs(...)` after filtering to enabled slots.

Phase 0 does not introduce float tie risk because positions are unique for segment boundaries. Tie-proof ordering (`InjectionSpec.order`) is a Phase 1 contract change.

