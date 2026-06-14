# Training Loop Correctness Health Report

Date: 2026-06-13

Scope:

- PPO loop: `train_ppo_vectorized()` -> `VectorizedPPOTrainer.run()` -> `execute_actions()` -> `PPOCoordinator`.
- Heuristic loop: `train_heuristic()` -> `run_heuristic_episode()`.

Verdict:

- PPO is the primary signal-of-life loop. After this pass it has the stronger correctness posture, with recurrent rollback, finiteness-skip advancement, validation-tail retention, seeded policy initialization, causal reward attribution, and checkpoint-kind consistency repaired.
- Heuristic is still not a full scientific control. It now exposes `max_seeds` and `reward_mode`, no longer fossilizes without counterfactual proof, and excludes fossilized seeds from optimizer participation. It still lacks PPO-equivalent governor preflight/rollback and live counterfactual measurement.

## Severity Summary

| Severity | Status | Finding |
| --- | --- | --- |
| P1 | Fixed | PPO terminal rollback now zeroes recurrent hidden/cell state for rolled-back envs before later same-batch use. |
| P1 | Fixed | PPO finiteness-gate skips now mark degraded telemetry and allow the batch to finish unless the repeated-failure abort threshold is reached. |
| P1 | Fixed | Contribution reward no longer pays pre-counterfactual proxy attribution from host accuracy drift. |
| P1 | Fixed | Karn batch-only fallback now stores unknown validation loss as `None`, not `0.0`. |
| P1 | Partially fixed | Heuristic no longer fossilizes without counterfactual proof and no longer optimizes fossilized seed parameters; governor rollback parity remains open. |
| P2 | Fixed | CPU PPO validation iterator keeps held-out tail samples via `drop_last=False`. |
| P2 | Fixed | PPO applies the configured seed before policy construction. |
| P2 | Fixed | PPO checkpoints are now explicit `checkpoint_kind="last"` instead of best weights plus final optimizer moments. |
| P2 | Open | Exact-continuation resume is not guaranteed; RNG/DataLoader/augmentation generator state is still not persisted. |
| P2 | Fixed | Tamiyo observation features reject malformed current `gradient_health` instead of encoding it as healthy. |
| P3 | Fixed | Nissa exposes queue-pressure health through `NissaHub.get_health_snapshot()`. |
| P3 | Open | CIFAR `train=False` stream is still operationally used as validation; there is no final-only holdout split. |
| P4 | Fixed | CLI `--episode-length` help derives its default from `DEFAULT_EPISODE_LENGTH`. |

## PPO Loop Health

Current status: usable for signal-of-life work, with remaining caution around exact replay/resume semantics.

Fixed:

- Recurrent rollback isolation: `_reset_hidden_for_terminal_envs()` zeroes LSTM hidden/cell slices for terminal rollback envs, and the trainer applies it after `execute_actions()` before bootstrap or later same-batch transitions can reuse stale context. Source: `src/esper/simic/training/vectorized_trainer.py:94`, `src/esper/simic/training/vectorized_trainer.py:1570`.
- Finiteness-skip progress: `PPOCoordinator.check_finiteness_gate()` still marks `run_governor_status="degraded"` on first/second finiteness failures, but returns `should_continue=True` so batch telemetry and counters advance. It still raises on the third consecutive failure. Source: `src/esper/simic/training/ppo_coordinator.py`.
- Validation tail retention: CPU validation `SharedBatchIterator` now passes `drop_last=False`, matching the GPU validation path and preserving held-out tail samples. Source: `src/esper/simic/training/vectorized.py:1208`.
- Seeded initialization: non-resume PPO now calls `torch.manual_seed(seed)` and CUDA manual seeding immediately before Tamiyo policy construction. Source: `src/esper/simic/training/vectorized.py:1025`.
- Checkpoint semantics: save metadata now declares `checkpoint_kind="last"` and no longer reloads best policy weights before saving final optimizer state. Source: `src/esper/simic/training/vectorized_trainer.py:1835`.
- Reward causality: shaped contribution reward pays zero attribution when a seed exists but no counterfactual proof exists. Source: `src/esper/simic/rewards/contribution.py:558`.

Still open:

- Resume is restart-from-checkpoint, not exact replay. RNG states, DataLoader iterator/generator state, and augmentation generator state are not serialized yet.
- Horizon semantics are still “episode length / rollout timeout” in practice. The report should not claim full host-convergence terminal semantics for `max_epochs`.
- CIFAR validation is a repeated `train=False` stream, not a held-out final test split.

Verification added:

- `tests/simic/training/test_recurrent_rollback.py`
- `tests/simic/training/test_ppo_coordinator.py::test_check_finiteness_gate_marks_batch_degraded_but_allows_telemetry`
- `tests/simic/test_vectorized_correctness.py`
- `tests/simic/test_rewards.py` proxy attribution regression

## Heuristic Loop Health

Current status: runtime smoke/control-like baseline only. It is better bounded after this pass, but still not a valid Phase 2.5 scientific control.

Fixed:

- Reward-mode parity surface: `run_heuristic_episode()` and `train_heuristic()` accept `reward_mode`, use `RewardMode(reward_mode)`, emit the selected mode in `TRAINING_STARTED`, and CLI heuristic exposes `--reward-mode`. Source: `src/esper/simic/training/helpers.py:542`, `src/esper/scripts/train.py`.
- Max-seed parity surface: `run_heuristic_episode()` and `train_heuristic()` accept `max_seeds`, validate it, wire it from CLI, and block germination when occupied slots reach the limit. Source: `src/esper/simic/training/helpers.py:787`, `src/esper/simic/training/helpers.py:911`.
- Fossilization proof: `HeuristicTamiyo` now waits when a HOLDING seed has no `counterfactual_contribution`; positive `total_improvement` alone cannot fossilize. Source: `src/esper/tamiyo/heuristic.py:291`.
- Fossilized optimizer exclusion: heuristic seed optimizer construction now skips `SeedStage.FOSSILIZED` slots and is rebuilt after germination, prune, and successful fossilize. Source: `src/esper/simic/training/helpers.py:64`.

Still open:

- No PPO-equivalent governor preflight and rollback execution path is wired into `run_heuristic_episode()`.
- The heuristic loop still does not compute live counterfactual contribution; it now refuses to fossilize without that proof, which means fossilization may stall until a counterfactual source is added.
- Reward parity is interface-level and component-level for the shared reward function, not full behavioral parity with PPO action execution.

Verification added:

- `tests/tamiyo/test_heuristic_decisions.py::TestFossilizeDecisions`
- `tests/tamiyo/test_regressions.py::TestCounterfactualPreferredForFossilize`
- `tests/tamiyo/properties/test_decision_antigaming.py::TestCounterfactualGuard`
- `tests/simic/test_training_helper.py::TestTrainOneEpoch::test_heuristic_seed_optimizer_excludes_fossilized_slots`
- `tests/scripts/test_train.py::TestTrainMainWiring::test_main_heuristic_wires_outputs_and_calls_train`

## Shared Runtime And Telemetry Health

Fixed:

- Karn validation-loss unknowns: `HostSnapshot.val_loss` is optional, batch-only fallback sets it to `None`, store import preserves `None`, and loss-trigger/health code skips loss analysis when unknown. Source: `src/esper/karn/store.py:163`, `src/esper/karn/collector.py`, `src/esper/karn/health.py`.
- Observation feature contract: active slot telemetry with missing, NaN, or Inf `gradient_health` now raises `ValueError` instead of encoding `1.0`. Source: `src/esper/tamiyo/policy/features.py`.
- Nissa queue pressure: `NissaHub.get_health_snapshot()` reports aggregate, hub, and backend dropped-event counts with `healthy`/`degraded` status. Source: `src/esper/nissa/output.py:813`.
- CLI help: PPO `--episode-length` help now uses `DEFAULT_EPISODE_LENGTH`. Source: `src/esper/scripts/train.py`.

Still open:

- Karn/Overwatch do not yet automatically poll and render `NissaHub.get_health_snapshot()`; the read surface exists for the UI/reporting layer to consume.
- CIFAR naming remains ambiguous: the operational validation stream is the CIFAR `train=False` split, not a separate final holdout.

## Verification Status

Focused and broad verification passed:

```bash
PYTHONPATH=src uv run pytest tests/simic/training/test_ppo_coordinator.py::test_check_finiteness_gate_marks_batch_degraded_but_allows_telemetry tests/simic/training/test_recurrent_rollback.py tests/karn/test_collector_multienv.py::TestMinimalTelemetryFallback::test_batch_only_creates_epoch_snapshot tests/tamiyo/policy/test_features.py::test_batch_obs_to_features_rejects_malformed_gradient_health tests/simic/test_vectorized_correctness.py tests/scripts/test_train.py::test_episode_length_help_uses_leyline_default -q
PYTHONPATH=src uv run pytest tests/simic/test_rewards.py -k pre_blending_seed_requires_counterfactual_for_attribution -q
PYTHONPATH=src uv run pytest tests/simic/test_training_helper.py::TestTrainOneEpoch::test_heuristic_seed_optimizer_excludes_fossilized_slots tests/tamiyo/test_heuristic_decisions.py::TestFossilizeDecisions tests/tamiyo/test_regressions.py::TestCounterfactualPreferredForFossilize tests/scripts/test_train.py::TestTrainMainWiring::test_main_heuristic_wires_outputs_and_calls_train -q
PYTHONPATH=src uv run pytest tests/nissa/test_output.py::TestNissaHubHealth::test_health_snapshot_surfaces_hub_and_backend_drops -q
PYTHONPATH=src uv run pytest tests/simic/training tests/simic/agent tests/tamiyo tests/karn -q
HYPOTHESIS_PROFILE=ci PYTHONPATH=src uv run pytest -m property -q --hypothesis-show-statistics
uv run python scripts/lint_defensive_patterns.py
uv run python scripts/lint_gpu_sync.py
uv run python scripts/lint_leyline_types.py
MYPYPATH=src uv run mypy -p esper
```

Results:

- Broad Simic/Tamiyo/Karn slice: 1361 passed, 1 skipped, 114 deselected.
- Property suite: 362 passed, 4451 deselected.
- Defensive-pattern lint: passed with 0 violations and 0 stale whitelist entries.
- GPU-sync lint: passed with 0 violations and 0 stale whitelist entries after removing a stale `mark_terminal_with_penalty` whitelist key.
- Leyline type-definition guard: passed.
- Mypy: passed with no issues in 202 source files.

Wardline status:

```bash
wardline scan . --fail-on ERROR
```

Wardline was attempted because CLI argument handling changed. The scan produced no output and did not complete after roughly four minutes, so the foreground scan process was terminated. Treat Wardline as incomplete for this pass, not clean.
