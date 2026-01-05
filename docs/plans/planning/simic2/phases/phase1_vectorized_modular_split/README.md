# Phase 1: Split `vectorized.py` into Trainer + Modules (No Behavior Change)

**Intent:** Make the training loop readable and testable by moving coherent subsystems out of the giant module, while keeping runtime behavior and hot-path performance intact.

## Target shape

- Keep `train_ppo_vectorized(...)` as the public entrypoint (for now).
- Move the main loop into `VectorizedPPOTrainer.run()`.
- Replace nested closures with module-level functions or dedicated modules.

## Extraction order (low-risk first)

1) **Move nested helpers to module scope** (still in `vectorized.py`)
- `create_env_state`
- `process_train_batch`
- `process_fused_val_batch`
- `batch_signals_to_features`

2) **Create modules and move code without changing semantics**
- `env_factory.py`: env creation/reset, slot wiring, stream/scaler setup
- `batch_ops.py`: train/val batch functions (stream-safe; preserve `torch.compiler.disable` where needed)
- `counterfactual_eval.py`: fused validation and attribution matrix computation
- `action_execution.py`: decode/validate/execute actions; return structured outcomes

3) **Introduce `VectorizedPPOTrainer`**
- `vectorized.py` becomes “construct context → run trainer → return results”.

## Hot-path guardrails (do not regress)

- Preserve existing synchronization structure:
  - stream sync before consuming DataLoader transfers
  - single stream synchronize per epoch boundary
  - no per-env `.item()` calls before sync
- Preserve batched D2H transfers:
  - stacked head actions transfer once
  - stacked log prob confidences transfer once
  - avoid new `.cpu()` calls inside the per-env loop

## Validation (beyond unit tests)

- Run a minimal training invocation (as small as practical):
  - `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar_baseline --task cifar_baseline --rounds 1 --envs 1 --episode-length 5`
- Compare telemetry event counts before/after for a short run (sanity, not statistical equivalence).

## Done means

- `src/esper/simic/training/vectorized.py` no longer contains nested helper functions.
- The main loop is in `VectorizedPPOTrainer` and reads top-to-bottom.
- All extracted modules have explicit dependency surfaces (no implicit closure capture).
