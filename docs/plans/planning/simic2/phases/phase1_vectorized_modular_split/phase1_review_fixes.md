# Phase 1 Review Fixes (Vectorized Trainer)

## Intent
Capture fixes discovered during the Phase 1 review so Phase 2 contracts do not
lock in incorrect behavior or misleading telemetry.

## Fix list

1) SUCCESS_THRESHOLD scale mismatch (percent vs fraction)
- Status: Implemented
- Fix: set SUCCESS_THRESHOLD to 80.0 (percent scale) or standardize accuracy to 0-1.
- Files: `src/esper/simic/training/action_execution.py`

2) Throughput timing only captures last epoch
- Status: Implemented
- Fix: accumulate `epoch` timings and `dataloader_wait_ms_epoch` inside the epoch loop
  using `time.perf_counter()` around the epoch body.
- Files: `src/esper/simic/training/vectorized_trainer.py`

3) CUDATimer stream/device mismatch in multi-stream / multi-GPU
- Status: Implemented
- Fix: replace CUDA-event timing with epoch wall-clock timing and rely on existing
  per-epoch stream syncs to bound GPU completion.
- Files: `src/esper/simic/training/vectorized_trainer.py`

4) Batch training correct/total telemetry uses placeholders
- Status: Implemented
- Fix: propagate real `train_corrects` and `train_totals` from the training loop;
  do not use percent * 100 or PPO step counts.
- Files: `src/esper/simic/training/vectorized_trainer.py`

5) LM fused validation totals use batch size, not token count
- Status: Implemented
- Fix: use `total_per_config` (tokens for LM) from `process_fused_val_batch` when
  accumulating `val_totals`.
- Files: `src/esper/simic/training/counterfactual_eval.py`, `src/esper/simic/training/vectorized_trainer.py`

6) ADVANCE is dynamic (auto-forward vs manual)
- Status: Implemented
- Fix: enable OP_ADVANCE only when no auto-forward gates are enabled. When enabled,
  OP_ADVANCE must progress GERMINATED -> TRAINING -> BLENDING -> HOLDING via `advance_stage()`;
  otherwise it is masked out.
- Files: `src/esper/simic/training/vectorized.py`, `src/esper/simic/training/action_execution.py`

7) chunk_length must match max_epochs
- Status: Implemented
- Fix: enforced in `TrainingConfig._validate()` and checked in `train_ppo_vectorized()`
  for direct callers.
- Files: `src/esper/simic/training/config.py`, `src/esper/simic/training/vectorized.py`

8) Normalizer update comment drift
- Status: Implemented
- Fix: update comments to reflect that the normalizer updates before PPO updates.
- Files: `src/esper/simic/training/vectorized_trainer.py`

9) Unused `use_amp` parameter in `process_train_batch`
- Status: Implemented
- Fix: remove the unused parameter to match runtime behavior.
- Files: `src/esper/simic/training/batch_ops.py`, `src/esper/simic/training/vectorized_trainer.py`

10) Redundant metric aggregation
- Status: Implemented
- Fix: remove `aggregate_ppo_metrics([metrics])` when metrics are already aggregated.
- Files: `src/esper/simic/training/vectorized_trainer.py`

11) Per-env reward config should not share a mutable instance
- Status: Implemented
- Fix: create per-env copies (e.g., `dataclasses.replace(reward_config)`) instead of
  `[reward_config] * n_envs`, even if the dataclass is currently immutable.
- Files: `src/esper/simic/training/vectorized.py`

## Non-issues
- Action counters are pre-initialized each episode; no KeyError risk.
  File: `src/esper/simic/training/parallel_env_state.py`
