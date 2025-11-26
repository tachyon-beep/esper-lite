# Karn — PyTorch 2.8 Review (Blueprint Definitions)

Summary
- No schema or code changes are required in Karn for PyTorch 2.8. Blueprint definitions remain Leyline `BlueprintDescriptor`s (id, name, tier, allowed_parameters, risk, stage, quarantine/approval flags, description). Tezzeret owns compilation; any 2.8 specifics (torch.compile/export, guards, pre‑warm) live in Tezzeret and Urza artifact metadata, not in Karn.

Clarifications (to avoid ambiguity)
- Stage semantics: `stage` in `BlueprintDescriptor` is a curriculum/progression marker for template maturity, not a Kasmina lifecycle state. Keep this distinct from the lifecycle in Leyline (`SeedLifecycleStage`).
- Parameter bounds: ensure templates that drive blending or runtime behaviour expose bounds in `allowed_parameters` (e.g., `alpha`, `dropout`, etc.) so G0 sanity checks in Kasmina can validate requests.

Optional (deferred; do not implement pre‑1.0)
- Compile strategy hints: if future builds want blueprint‑level hints (e.g., preferred compilation mode), attach them as Urza artifact metadata produced by Tezzeret, not as new fields in Karn.
- Input guard specs: shape/dtype/dynamic dim guards should be produced during Tezzeret export and stored with the Urza artifact; Karn remains purely descriptive of template identity and safe ranges.

References
- Blueprints: `src/esper/karn/templates.py` and `src/esper/karn/catalog.py`
- Tezzeret 2.8 pipeline: `docs/prototype-delta/tezzeret/pytorch-2.8-upgrades.md`
- Kasmina G0 checks use `allowed_parameters` and blueprint id for sanity validation.
