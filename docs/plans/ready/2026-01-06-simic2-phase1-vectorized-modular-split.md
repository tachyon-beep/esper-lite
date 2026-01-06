# Simic2 Phase 1: Vectorized Trainer Modular Split (No Behavior Change)

## Intent
Make `src/esper/simic/training/vectorized.py` readable and testable by extracting coherent subsystems while preserving behavior, throughput, and telemetry semantics.

## Scope
- Split the vectorized training loop into:
  - `vectorized.py` (public entrypoint + thin orchestration)
  - `vectorized_trainer.py` (VectorizedPPOTrainer.run)
  - `env_factory.py`, `batch_ops.py`, `counterfactual_eval.py`, `action_execution.py`
- Keep `train_ppo_vectorized(...)` signature and semantics stable.

## Non-goals
- No algorithm changes (PPO math, reward design, action masks).
- No telemetry schema changes.
- No throughput optimization beyond "do not regress".

## Preconditions (from Phase 0)
- Baselines captured in `docs/plans/planning/simic2/phases/phase0_baseline_and_tests/baseline_capture.md`.
- Fast guardrail suite passes.
- Telemetry event-count baseline recorded.

## Call sites and seams (no changes expected)

Call sites (17 total, from baseline capture):
- `src/esper/simic/agent/ppo.py:135`
- `src/esper/simic/training/dual_ab.py:196`
- `src/esper/simic/training/config.py:26`
- `src/esper/simic/training/vectorized.py:21`
- `src/esper/simic/training/vectorized.py:549`
- `src/esper/scripts/train.py:842`
- `tests/cuda/test_vectorized_multi_gpu_smoke.py:97`
- `tests/integration/test_optimizer_lifecycle.py:146`
- `tests/integration/test_phase6_regression_baseline.py:37`
- `tests/integration/test_sparse_training.py:21`
- `tests/integration/test_sparse_training.py:40`
- `tests/integration/test_vectorized_determinism.py:143`
- `tests/scripts/test_train.py:481`
- `tests/scripts/test_train.py:606`
- `tests/simic/test_gpu_preload_batch_size.py:31`
- `tests/simic/test_reward_normalizer_checkpoint.py:178`
- `tests/telemetry/test_training_metrics.py:100`

Monkeypatch seams to preserve (keep symbols referenced in `vectorized.py`):
- `get_hub`, `RewardNormalizer`, `SharedBatchIterator`, `PPOAgent.load`
- `torch_amp.autocast`, `collect_per_layer_gradients`, `check_numerical_stability`

## Implementation plan

1) Move nested helpers to module scope (no new modules yet):
   - `make_telemetry_callback`
   - `configure_slot_telemetry`
   - `create_env_state`
   - `_collect_gradient_telemetry_for_batch`
   - `_parse_sampled_action`
   - `process_train_batch`
   - `process_val_batch`
   - `process_fused_val_batch`
   - `batch_signals_to_features`

2) Extract modules (mechanical moves, preserve logic):
   - `env_factory.py`: telemetry callback, slot telemetry config, env creation
   - `batch_ops.py`: train/val batch ops, gradient telemetry, signals->features
   - `counterfactual_eval.py`: fused validation batch + attribution helpers
   - `action_execution.py`: action decode/validate/execute, buffer writes
   - `vectorized_trainer.py`: `VectorizedPPOTrainer.run`

3) Wire `VectorizedPPOTrainer`:
   - `vectorized.py` constructs context, instantiates trainer, calls `.run()`.
   - Keep lazy `get_task_spec` import inside `train_ppo_vectorized`.
   - Do not add new imports to `src/esper/simic/training/__init__.py`.

4) Preserve hot-path fenced regions:
   - Training default-stream waits, epoch-end sync, validation waits/sync.
   - Single batched D2H transfers for actions/op_probs/head confidences.
   - Preserve stream usage ordering and avoid new `.cpu()`/`.item()` in per-env loops.

5) Action execution contract:
   - `action_execution` returns a small explicit structure (no ad-hoc dicts):
     - `truncated_bootstrap_targets`
     - `post_action_signals`
     - `post_action_slot_reports`
     - `post_action_masks`
   - Mutable lists are passed in explicitly and mutated in place.

## Validation
- Fast guardrail suite:
  - `UV_CACHE_DIR=.uv-cache uv run pytest -q tests/test_import_isolation.py tests/meta/test_factored_action_contracts.py tests/simic/test_vectorized.py tests/simic/training/test_entropy_annealing.py tests/simic/rewards/escrow/test_escrow_wiring.py tests/simic/test_reward_normalizer_checkpoint.py tests/simic/test_gpu_preload_batch_size.py tests/scripts/test_train.py`
- Minimal training run:
  - `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar_baseline --task cifar_baseline --rounds 1 --envs 1 --episode-length 5`
- Telemetry event counts:
  - Compare event_type counts against Phase 0 baseline (short run with `--telemetry-dir`).

## Acceptance criteria
- No nested helper functions remain in `vectorized.py`.
- `VectorizedPPOTrainer.run()` contains the main loop and reads top-to-bottom.
- Extracted modules have explicit dependencies (no closure capture).
- Import isolation test remains green (no new heavy imports at import-time).
- Telemetry event counts and throughput signals match baseline (sanity-level).
