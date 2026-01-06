# Phase 1: Split `vectorized.py` into Trainer + Modules (No Behavior Change)

**Intent:** Make the training loop readable and testable by moving coherent subsystems out of the giant module, while keeping runtime behavior and hot-path performance intact.

## Pre-flight (risk reduction)

- Confirm Phase 0 baselines are captured: `docs/plans/planning/simic2/phases/phase0_baseline_and_tests/baseline_capture.md`
- Re-run the fast guardrail suite before starting (use workspace-local UV cache if needed):
  - `UV_CACHE_DIR=.uv-cache uv run pytest -q tests/test_import_isolation.py tests/meta/test_factored_action_contracts.py tests/simic/test_vectorized.py tests/simic/training/test_entropy_annealing.py tests/simic/rewards/escrow/test_escrow_wiring.py tests/simic/test_reward_normalizer_checkpoint.py tests/simic/test_gpu_preload_batch_size.py tests/scripts/test_train.py`

Supporting docs:
- Extraction map: `docs/plans/planning/simic2/phases/phase1_vectorized_modular_split/extraction_map.md`
- Risk register: `docs/plans/planning/simic2/phases/phase1_vectorized_modular_split/risk_register.md`
- Preflight audits: `docs/plans/planning/simic2/phases/phase1_vectorized_modular_split/preflight_audit.md`
- Review fix list: `docs/plans/planning/simic2/phases/phase1_vectorized_modular_split/phase1_review_fixes.md`

## Target shape

- Keep `train_ppo_vectorized(...)` as the public entrypoint (for now).
- Move the main loop into `VectorizedPPOTrainer.run()` in `vectorized_trainer.py`.
- Replace nested closures with module-level functions or dedicated modules.

## Extraction order (low-risk first)

1) **Move nested helpers to module scope** (still in `vectorized.py`)
- `make_telemetry_callback`
- `configure_slot_telemetry`
- `create_env_state`
- `_collect_gradient_telemetry_for_batch`
- `_parse_sampled_action`
- `process_train_batch`
- `process_val_batch`
- `process_fused_val_batch`
- `batch_signals_to_features`

2) **Create modules and move code without changing semantics**
- `env_factory.py`: env creation/reset, slot wiring, stream/scaler setup
- `batch_ops.py`: train/val batch functions (stream-safe; preserve `torch.compiler.disable` where needed)
- `counterfactual_eval.py`: fused validation and attribution matrix computation
- `action_execution.py`: decode/validate/execute actions; return structured outcomes

3) **Introduce `VectorizedPPOTrainer` (in `vectorized_trainer.py`)**
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
  - For throughput, prefer the training-emitted throughput telemetry (CUDATimer-based) over `time.time()` sampling.
  - Use the telemetry event-count procedure in Phase 0 baseline capture.

## Done means

- `src/esper/simic/training/vectorized.py` no longer contains nested helper functions.
- The main loop is in `VectorizedPPOTrainer` and reads top-to-bottom.
- All extracted modules have explicit dependency surfaces (no implicit closure capture).
