# Karn — Parameter Bounds Audit (Prototype)

Summary
- Karn provides 50 static blueprints split across SAFE (35), EXPERIMENTAL (7), and ADVERSARIAL (8), consistent with tests.
- Blueprints use Leyline `BlueprintDescriptor.allowed_parameters` to declare safe ranges for tunables. These bounds are consumed by Tezzeret (compile‑time validation) and available for Kasmina G0 sanity checks.
- No changes required for PyTorch 2.8; bounds remain descriptive (Tezzeret owns compile/guards).

Observed parameter keys (by category)
- SAFE (sample of keys): `scale`, `epsilon`, `dropout`, `momentum`, `kernel`, `depth`, `branches`, `groups`, `gate_bias`, `reduction`, `heads`, `radius`, `sequence_length`, `hidden_multiple`, `layers`, `experts`, `output_size`, `levels`, `growth_rate`, `hops`, `dilation`, `offset_groups`, `window`, `frequency`.
- EXPERIMENTAL (sample): `experts`, `capacity`, `levels`, `memory_slots`, `steps`, `temperature`, `adapt_steps`, `depth`.
- ADVERSARIAL (sample): `scale`, `decay`, `frequency`, `iterations`, `growth_rate`, `threads`, `window_ns`, `epsilon`.

Notes
- Blending alpha is internal to Kasmina; do not pass `alpha` as a blueprint parameter unless a specific template requires it. The default library does not include `alpha` in SAFE/EXPERIMENTAL blueprints.
- Where tests require `alpha` (e.g., to drive a custom residual MLP), they set bounds on a bespoke descriptor at test time (see tests/karn/test_catalog.py).
- G0 gate in Kasmina currently validates presence/NaN only; range enforcement happens earlier when Tezzeret compiles from Karn → Urza. This is acceptable for the prototype.

Recommendations
- Keep template parameter keys stable across SAFE/EXPERIMENTAL; prefer existing names (`dropout`, `momentum`, `kernel`, `heads`, etc.).
- If a new template needs a tunable used by Kasmina for schedule‑like behaviours, give it explicit bounds in `allowed_parameters` and document the range in the template description.
- Do not overload `stage`: it is a curriculum/maturity marker, not a Kasmina lifecycle state.

References
- Source: `src/esper/karn/templates.py` (definitions), `src/esper/karn/catalog.py` (registry/validation)
- Tests: `tests/karn/test_catalog.py`
