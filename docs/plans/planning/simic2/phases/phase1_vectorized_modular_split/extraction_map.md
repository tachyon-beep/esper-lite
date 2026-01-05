# Phase 1 Extraction Map (Vectorized Trainer Decomposition)

This document is a “move list” to reduce surprises when we split `src/esper/simic/training/vectorized.py`.

## 0) Non-negotiable constraints (tests enforce these)

- **Import isolation:** `tests/test_import_isolation.py` must stay green (do not import vectorized training from lightweight modules).
- **Public entrypoint signature:** `tests/meta/test_factored_action_contracts.py` asserts `train_ppo_vectorized` parameters.
- **CLI wiring:** `tests/scripts/test_train.py` monkeypatches `esper.simic.training.vectorized.train_ppo_vectorized`.
- **Monkeypatch seams:** these tests patch names inside the `vectorized` module:
  - `tests/simic/test_gpu_preload_batch_size.py` (`get_hub`)
  - `tests/simic/test_reward_normalizer_checkpoint.py` (`get_hub`, `RewardNormalizer`, `SharedBatchIterator`, `PPOAgent.load`)

**Phase 1 mitigation:** keep those names referenced from `src/esper/simic/training/vectorized.py` even if the implementation moves into a trainer object; pass dependencies into the trainer rather than importing inside the new modules.

## 1) Current nested helpers inside `train_ppo_vectorized` (closure-captured)

These `def`s are nested under `train_ppo_vectorized` today and capture a large implicit context:

- `make_telemetry_callback` (`src/esper/simic/training/vectorized.py:1175`)
- `configure_slot_telemetry` (`src/esper/simic/training/vectorized.py:1187`)
- `create_env_state` (`src/esper/simic/training/vectorized.py:1236`)
- `_collect_gradient_telemetry_for_batch` (`src/esper/simic/training/vectorized.py:1384`) — decorated with `@torch.compiler.disable`
- `_parse_sampled_action` (`src/esper/simic/training/vectorized.py:1440`)
- `process_train_batch` (`src/esper/simic/training/vectorized.py:1522`)
- `process_val_batch` (`src/esper/simic/training/vectorized.py:1717`)
- `process_fused_val_batch` (`src/esper/simic/training/vectorized.py:1766`)
- `batch_signals_to_features` (`src/esper/simic/training/vectorized.py:1856`)

**Phase 1 step 1:** move these to module scope *without changing logic*, replacing implicit closure capture with an explicit “context object” parameter (dataclass or an explicit arg list).

## 2) Module-level helpers (tests import these today)

These helpers are already module-level and directly imported by tests:

- `_resolve_target_slot` (`tests/simic/rewards/escrow/test_escrow_wiring.py`)
- `_calculate_entropy_anneal_steps` (`tests/simic/training/test_entropy_annealing.py`)
- `_aggregate_ppo_metrics` (`tests/simic/test_vectorized.py`)
- `_run_ppo_updates` (`tests/simic/test_vectorized.py`)

**Phase 1 mitigation:** do not move/rename these unless we’re ready to update all call sites/tests in the same PR.

## 3) Performance-sensitive hotspots (do not regress)

These areas have explicit “don’t do N syncs” comments and are easy to accidentally break:

- Batched action transfer and per-step telemetry precompute:
  - `src/esper/simic/training/vectorized.py:2863` (stack heads → one `.cpu()` transfer)
  - `src/esper/simic/training/vectorized.py:2883` (op probs computed once pre-loop)
- Stream synchronization rules (data-loader transfers happen on default stream):
  - The “wait for default stream” pattern inside the epoch training loop (keep single epoch-end sync).

**Phase 1 mitigation:** treat these blocks as “fenced regions” in review; extracted functions must preserve the same device/stream sync structure.

## 4) Proposed extracted modules (Phase 1 target)

The goal is “coherent subsystems per file”, not premature abstraction.

- `env_factory.py`
  - env creation/reset, slot wiring, per-env stream/scaler setup
- `batch_ops.py`
  - train/val batch functions (stream-safe), including gradient collection helper
- `counterfactual_eval.py`
  - fused validation pass + ablation config construction + attribution matrix aggregation
- `action_execution.py`
  - decode/validate/execute actions per env; update env counters/optimizers; return an outcome struct
- `vectorized_trainer.py`
  - `VectorizedPPOTrainer.run()` — the main loop moved out of `train_ppo_vectorized`

