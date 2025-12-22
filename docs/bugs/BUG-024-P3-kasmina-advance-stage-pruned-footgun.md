# BUG-024: `advance_stage(PRUNED)` can create zombie states (skips prune side-effects)

- **Title:** `SeedSlot.advance_stage(target_stage=SeedStage.PRUNED)` would transition stage without running `prune()` cleanup (seed removal, alpha reset, freeze reset)
- **Context:** `advance_stage()` is the public API for lifecycle advancement; `prune()` is the public API for removal/cleanup
- **Impact:** P3 – latent API footgun; future callers could accidentally create a “PRUNED but still active” seed (state says PRUNED, but `seed` module and optimizers may still exist)
- **Environment:** HEAD @ workspace
- **Reproduction Steps:**
  1. Call `slot.advance_stage(SeedStage.PRUNED)` on a non-fossilized seed.
  2. Stage can transition (per `VALID_TRANSITIONS`) but `slot.seed` will not be cleared because cleanup lives in `prune()`.
- **Expected Behavior:** Failure-stage transitions should be routed through the dedicated APIs (`prune()`/cooldown mechanics), or explicitly rejected from `advance_stage()`.
- **Observed Behavior:** `advance_stage()` accepts an explicit target stage and does not guard against failure stages.
- **Logs/Telemetry:** Would manifest as inconsistent slot reports (stage PRUNED but `is_active==True`) if invoked.
- **Hypotheses:** No current call sites target PRUNED via `advance_stage()`, so this hasn’t surfaced; risk grows with future refactors.
- **Fix Plan:** In `advance_stage()`, reject `target_stage` in failure stages (`PRUNED/EMBARGOED/RESETTING`) with a clear error, and require `prune()`/cooldown to manage those transitions.
- **Validation Plan:** Unit test that `advance_stage(SeedStage.PRUNED)` fails and does not mutate state.
- **Status:** Resolved
- **Links:**
  - API: `src/esper/kasmina/slot.py:1176`
  - Cleanup path: `src/esper/kasmina/slot.py:1273`

## Fix Implemented

- `advance_stage()` now rejects explicit failure-stage targets with a clear error.
- Added a unit test ensuring `advance_stage(SeedStage.PRUNED)` raises and does not mutate state.
