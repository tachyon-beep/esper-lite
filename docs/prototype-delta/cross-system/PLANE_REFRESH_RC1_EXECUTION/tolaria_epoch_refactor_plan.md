# Tolaria Epoch Runner Refactor Plan (R4a)

## Objective
Reduce `TolariaTrainer._train_single_epoch` complexity to ≤ C by introducing a structured epoch runner with testable helper functions, while preserving existing telemetry, Tamiyo/Kasmina coordination, and rollback behaviour.

## Target Structure

### Epoch Runner
- Introduce `_EpochRunner` helper encapsulating per-epoch state:
  - Receives `TolariaTrainer` reference plus immutable config snapshots (model, optimizer, aggregation knobs).
  - Exposes `run() -> EpochStats` orchestrating the epoch lifecycle.

### State Containers
- `EpochContext`: loss/sample totals, gradient norm sums, conservative-mode flags, telemetry aggregates.
- `MicrobatchState`: inputs/targets tensors, seed IDs, timing metrics (`input_wait_ms`, `h2d_copy_ms`), forward outputs.
- `FenceState`: flattened gradients, seed masks, attribution weights, per-layer accumulators.

## Helper Responsibilities

1. `_prepare_microbatch()` — take a raw dataloader batch, apply device transfers, and compute timing metadata; returns a `MicrobatchState`.
2. `_iterate_microbatches()` — iterate the dataloader and yield `MicrobatchState` objects using `_prepare_microbatch()`.
3. `_forward_backward()` — perform forward/backward pass with autocast/compile handling; return loss tensor, correct count, gradient snapshot.
4. `_accumulate_microbatch()` — update attribution/per-layer buffers, store flattened grads, track seed state.
5. `_should_step_optimizer()` — evaluate accumulation fence condition.
6. `_optimizer_step()` — aggregate gradients (PCGrad/seed-aware), apply optimizer/scaler, compute gradient norms, return telemetry snippets.
7. `_invoke_control_loop()` — call Tamiyo/Kasmina, manage conservative-mode transitions, emit step telemetry metrics/events.
8. `_update_step_metrics()` — refresh EWMA/loss deltas, populate `training_metrics` fields, update metric counters.
9. `_finalize_epoch()` — snapshot fast rollback state, compute teacher/seed aggregation summaries, populate `EpochStats`.

## Regression Safeguards
- Capture golden artefacts (pre-refactor): epoch stats, step `training_metrics`, Tamiyo/Kasmina command sequences for a deterministic dummy run.
- Add tests replaying the same scenario post-refactor to ensure telemetry and metrics match.
- Unit-test extracted utilities (`_build_seed_masks`, optimizer aggregation) separately with synthetic inputs.

## Documentation / Tooling
- Drop `_train_single_epoch` complexity score in `lint_static_analysis.md` once refactor lands.
- Append Tolaria entry to `CHANGELOG_RC1.md` summarising the structural change and tests run.
- Update shared foundations doc with a note about the epoch runner for future contributors.

## Progress
- 2025-09-27: `_EpochRunner`, `EpochContext`, and the golden epoch fixture landed; legacy `_train_single_epoch` now delegates through the runner wrapper.
- 2025-09-27: Gradient aggregation and optimizer stepping migrated into `_EpochRunner._optimizer_step`, removing the monolithic fence block and keeping telemetry parity (`PYTHONPATH=src pytest tests/tolaria/test_tolaria_trainer.py`).
- 2025-09-27: Post-aggregation metrics and the Tamiyo/Kasmina control loop now flow through `_EpochRunner._update_step_metrics` and `_invoke_control_loop`, trimming another ~250 lines from the legacy body without behavioural drift.
- 2025-09-27: `_finalize_epoch` aggregates seed telemetry/metrics inside the runner and the golden fixture (`tests/fixtures/tolaria_epoch_fixture.json`) was refreshed via `python scripts/capture_tolaria_epoch_fixture.py`.
- 2025-09-27: `_EpochRunner.run` now orchestrates the full training loop; the legacy `_train_single_epoch_legacy` path has been removed and a regression test (`test_tolaria_epoch_fixture_parity`) guards the golden fixture.

## Next Steps
1. Monitor the parity test coverage and extend it to track additional telemetry fields as R4b/R4c land.
2. Update lint/static-analysis docs (`lint_static_analysis.md`) with the new Cyclomatic score once CI confirms the reduction.
3. Finalise R4a documentation/changelog entries and plan the follow-on refactors (Tamiyo risk engine, Kasmina dispatcher).
