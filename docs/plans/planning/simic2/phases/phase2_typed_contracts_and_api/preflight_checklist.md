# Phase 2 Preflight Checklist (Typed Boundaries)

## Objective
Prepare Phase 2 execution by locking scope, risks, guardrails, and validation so the
typed-contract refactor can ship without behavior, telemetry, or performance drift.

## Pre-Phase Activities (must complete before Phase 2 starts)

### Entry gates (readiness)
- [ ] Phase 1 extraction is stable and merged (vectorized_trainer, action_execution, batch_ops).
      (Merge deferred; stability validated on env-refactor.)
- [ ] Phase 1 review fixes list is implemented, merged, and re-baselined
      (SUCCESS_THRESHOLD scale, throughput timing, correct/total telemetry,
      CUDATimer decision, reward config copy).
      (Implemented + re-baselined on env-refactor; merge deferred.)
- [ ] Full test suite passes on main branch (no local-only fixes).
      (Deferred until merge.)
- [x] Telemetry event counts baseline is captured for a short PPO run (Phase 0 method).
- [x] Performance baseline captured for a short PPO run (throughput, per-epoch time).
- [x] Type checking gate decided and enforced (mypy or pyright) for touched modules;
      new typed containers have no `Any` fields unless explicitly justified at boundaries.

### Contract inventory and planning
- [x] Enumerate dict surfaces to be replaced and their call sites:
  - action decode/execute inputs and outputs:
    - `src/esper/simic/training/action_execution.py`: `_parse_sampled_action` tuple return
      (target_slot, slot_is_enabled, slot_state, seed_state, action_valid_for_reward,
      action_for_reward, blend_algorithm_id, alpha_algorithm, alpha_target).
    - `src/esper/simic/training/action_execution.py`: `action_dict` (head indices for telemetry).
    - `src/esper/simic/training/action_execution.py`: `ActionExecutionResult.post_action_slot_reports`
      (list of slot_id → SeedStateReport dicts) and `post_action_masks` (dict head → tensor) used
      to bootstrap truncated rollouts.
  - reward plumbing input arguments:
    - `src/esper/simic/training/action_execution.py`: `compute_reward(...)` and
      `compute_loss_reward(...)` long parameter lists.
    - `src/esper/simic/rewards/rewards.py`: `compute_reward_dispatch(...)` forwards long lists.
  - batch summary accumulation and history serialization:
    - `src/esper/simic/training/vectorized_trainer.py`: `reward_summary_accum` list of dicts.
    - `src/esper/simic/training/vectorized_trainer.py`: `history.append({ ... **metrics })` merges
      `metrics` dict into batch summary.
    - `src/esper/simic/training/action_execution.py`: `episode_history` list of dicts
      (EpisodeRecord schema).
- [x] Map each dict surface to a proposed typed container (field list + ownership):
  - `ActionSpec` (new, training): raw head indices + derived values
    (slot_idx, blueprint_idx, style_idx, tempo_idx, alpha_target_idx, alpha_speed_idx,
    alpha_curve_idx, op_idx, target_slot, slot_is_enabled, action_valid_for_reward,
    action_for_reward, blend_algorithm_id, alpha_algorithm, alpha_target).
    Ownership: CPU-only primitives/strings/enums; no tensors or model references.
  - `ActionOutcome` (new, training): execution result + telemetry fields
    (action_success, action_name, reward_raw, reward_normalized, reward_components,
    rollback_occurred, episode_reward, final_accuracy, episode_outcome, truncated).
    Ownership: CPU-only primitives + RewardComponentsTelemetry (CPU-only dataclass).
  - `ContributionRewardInputs` / `LossRewardInputs` (new, rewards):
    match current compute_reward/compute_loss_reward arguments; typed discriminators
    avoid optional-key dicts. Ownership: CPU-only primitives + SeedInfo.
  - `RewardSummaryAccumulator` (new, training): bounded_attribution, compute_rent,
    alpha_shock, hindsight_credit, total_reward, count, scaffold_count, scaffold_delay_total.
  - `BatchSummary` (new, training): batch, episodes, avg_accuracy, rolling_avg_accuracy,
    ppo_metrics (typed), reward_summary (typed), episode_history snapshot.
- [x] Dicts are allowed only as data bags where schema is owned elsewhere (e.g., telemetry
      details maps), not as inter-module contracts.
- [x] Decide where new dataclasses live:
  - `src/esper/simic/training/vectorized_types.py`: ActionSpec, ActionOutcome, RewardSummaryAccumulator, BatchSummary.
  - `src/esper/simic/rewards/types.py`: ContributionRewardInputs, LossRewardInputs.
- [x] Decide external API stance:
  - **Option A:** keep `train_ppo_vectorized(...)` signature stable.
  - **Option B:** config-only entrypoint; enumerate and update all call sites in one PR.
- [x] Record the API decision in the phase’s promoted ready plan (no dual paths).
  (Recorded in Phase 2 README; will copy when plan is promoted.)

### Risk reduction: correctness
- [x] Action decode/validation rules are documented and covered by unit tests
      (ADVANCE vs FOSSILIZE, slot validity, mask invariants).
  - ADVANCE vs FOSSILIZE: `tests/leyline/test_lifecycle_fix.py`, `tests/simic/test_vectorized.py`
  - Mask invariants: `tests/tamiyo/policy/test_action_masks.py`, `tests/tamiyo/properties/test_mask_properties.py`
  - MIN_PRUNE_AGE enforcement: `tests/simic/training/test_min_prune_age_enforcement.py`
- [x] Reward inputs are defined once, with a single construction site and tests.
      (ContributionRewardInputs/LossRewardInputs used end-to-end + tests updated.)
- [x] BatchSummary fields are explicit, serialized only at history boundary.
      (BatchSummary.to_dict used only when appending history.)
- [x] All new dataclasses use `slots=True` to avoid accidental attribute drift.
- [x] Invariants documented and tested:
      - HEAD_NAMES ordering and head tensor shapes:
        `tests/simic/properties/test_factored_action_properties.py`,
        `tests/simic/test_tamiyo_buffer.py`
      - SlotConfig slot-id ordering invariants:
        `tests/leyline/test_slot_config.py`,
        `tests/meta/test_slot_config_contracts.py`
      - rollout buffer tensor shapes and indexing semantics:
        `tests/simic/test_tamiyo_buffer.py`,
        `tests/simic/properties/test_buffer_properties.py`
      - accuracy unit convention (percent vs fraction):
        `tests/leyline/test_seed_telemetry_features.py`,
        `tests/simic/test_reward_simplified.py`
      - reward normalizer semantics (raw vs normalized storage):
        `tests/simic/test_normalization.py`,
        `tests/simic/test_reward_normalizer_checkpoint.py`

### Risk reduction: telemetry
- [x] Event payload shapes remain unchanged (field names and types).
      Coverage: `tests/telemetry/test_decision_metrics.py`,
      `tests/simic/telemetry/test_emitters.py`,
      `tests/karn/sanctum/test_schema.py`.
- [x] Telemetry emission order preserved (no reordering in hot path).
      Guarded by `tests/simic/telemetry/test_emitters.py` (batch tail ordering).
- [x] Telemetry baseline includes event sequence constraints for a short run
      (per-epoch ordering for key events such as EPOCH_COMPLETED, LAST_ACTION,
      COUNTERFACTUAL matrix, PPO_UPDATE, BATCH_COMPLETED).
- [x] Validate payload keys and key field semantics (accuracy scale, success threshold logic).
      `EPISODE_SUCCESS_THRESHOLD=80.0` enforced in `tests/simic/test_episode_outcome_emission.py`.
- [x] Add or update tests for:
  - event payload keys
  - op/slot head telemetry consistency
  - action success/failure accounting
  - event payload keys: `tests/telemetry/test_decision_metrics.py`,
    `tests/simic/telemetry/test_emitters.py`
  - op/slot head telemetry consistency: `tests/leyline/test_action_schema_drift.py`,
    `tests/karn/sanctum/test_schema.py`
  - action success/failure accounting: `tests/telemetry/test_decision_metrics.py`,
    `tests/telemetry/test_tele_env_state.py`

### Risk reduction: performance and GPU semantics
- [x] Confirm no new `.cpu()` or `.item()` calls are introduced in the hot path.
      Audit: call sites remain in `src/esper/simic/training/vectorized_trainer.py`
      and `src/esper/simic/training/helpers.py` only (no new sites in action execution).
- [x] No per-env/per-step dataclass allocations in the inner loop unless measured negligible;
      prefer batched construction or reuse where feasible.
      Plan: construct ActionSpec/Outcome in batched form; benchmark if per-env allocation
      becomes unavoidable.
- [x] Typed containers must not retain large tensors longer than needed
      (no accidental GPU tensor retention across epochs).
- [x] Ensure typed objects do not capture device tensors across streams incorrectly.
- [x] Validate that batched D2H transfers remain batched (no per-head sync).
      Batched transfer sites: `actions_stacked.cpu()` and `op_probs_all.cpu()` in
      `src/esper/simic/training/vectorized_trainer.py`.
- [x] Capture a before/after timing budget target (no regression beyond noise).
      Target: no >5% regression vs baseline episodes/sec and per-epoch deltas.

### Risk reduction: torch.compile compatibility
- [x] Typed objects do not cross into `torch.compile` regions; compiled functions accept
      tensors/primitives only.
- [x] No new imports from typed modules are pulled into compiled policy / PPO update paths.
      Compile boundary: `src/esper/simic/training/vectorized.py` (`loss_and_correct`,
      PPO update path).

### Risk reduction: typed container ownership
- [x] For each typed container, document field ownership and residency:
      - ActionSpec: CPU-only primitives/strings/enums; no tensors; do not hold SeedState objects.
      - ActionOutcome: CPU-only primitives + RewardComponentsTelemetry; no tensors; ephemeral per-step.
      - ContributionRewardInputs/LossRewardInputs: CPU-only primitives + SeedInfo; no tensors.
      - RewardSummaryAccumulator: CPU-only floats/ints; safe across epochs.
      - BatchSummary: CPU-only primitives + typed metrics; serialize to dict at history boundary.

### Risk reduction: multiprocessing / pickling
- [x] Typed containers are not captured by DataLoader worker processes
      (no references stored in dataset / collate fns / iterators).
- [x] If any typed object must cross process boundaries, add a targeted pickling test.
      Not required for Phase 2 scope; add if a worker boundary appears.

### Implementation planning artifacts
- [x] Per-file change list (modules touched, new files added).
  - New: `src/esper/simic/training/vectorized_types.py` (ActionSpec, ActionOutcome,
    RewardSummaryAccumulator, BatchSummary).
  - New: `src/esper/simic/rewards/types.py` (ContributionRewardInputs, LossRewardInputs).
  - Update: `src/esper/simic/training/action_execution.py` (ActionSpec/Outcome + RewardInputs).
  - Update: `src/esper/simic/training/vectorized_trainer.py`
    (typed container preallocation + BatchSummary/history serialization).
  - Update: `src/esper/simic/rewards/rewards.py` (accept RewardInputs, dispatch).
  - Update: `src/esper/simic/rewards/__init__.py` (export RewardInputs).
  - Tests: new coverage for RewardInputs, ActionSpec/Outcome, BatchSummary.
- [x] Call-site update list (train script, configs, tests, helper modules).
  - Reward plumbing: `action_execution.py` → `compute_reward`/`compute_loss_reward`,
    `rewards.py` → `compute_reward_dispatch`.
  - Action execution: `execute_actions` call in `vectorized_trainer.py` uses ActionSpec/Outcome.
  - History contract: `vectorized_trainer.py` history entries; `tests/simic/training/test_dual_ab.py`
    expects `history[0]["avg_accuracy"]` and `history[0]["batch"]`.
- [x] Test plan (unit tests + small PPO run + lint/mypy).
  - Add unit tests: RewardInputs construction + dispatch, ActionSpec decode invariants,
    ActionOutcome reward/telemetry fields, BatchSummary serialization.
  - Run fast guardrail suite (Phase 0 list) + telemetry decision tests:
    `tests/telemetry/test_decision_metrics.py`, `tests/simic/training/test_dual_ab.py`.
  - Run `uv run ruff check src/ tests/` and `uv run mypy -p esper`.
  - Re-run short PPO telemetry baseline on CPU (same command as Notes).
- [x] Rollback strategy (if regression found, revert whole PR; no partial shims).
  - Revert entire change set; re-run fast guardrail suite + telemetry baseline.

### Acceptance criteria (Phase 2 done means)
- [x] No dict-based optional-key contracts remain at the Phase 2 seams.
- [x] Reward plumbing uses a single typed container.
- [x] Action execution uses typed ActionSpec/ActionOutcome with unit tests.
- [x] Batch summaries are typed and serialized at the history boundary only.
- [x] Telemetry baselines match (event counts and payload keys).
- [x] Throughput within baseline tolerance (no measurable regression).

## Execution notes (Phase 2 completion)
- Full test suite: `UV_CACHE_DIR=.uv-cache uv run pytest`
  - 4240 passed, 34 skipped, 69 deselected, 4 xfailed
  - Warnings: CUDA init on this host, flex_attention compile warning, PPO recurrent_n_epochs notice
- Lint: `UV_CACHE_DIR=.uv-cache uv run ruff check src/ tests/` (clean)
- Types: `UV_CACHE_DIR=.uv-cache uv run mypy src/` (clean, 166 files)
- Telemetry xfails (TELE-600–603) are tracked as separate work and do not gate Phase 2

## Notes
- No compatibility shims. If a contract changes, update all call sites in the same PR.
- Avoid defensive programming: missing fields must fail fast.
- Baseline run: `telemetry/telemetry_2026-01-06_173034`
- Command: `PYTHONPATH=src UV_CACHE_DIR=.uv-cache uv run python -m esper.scripts.train ppo --preset cifar_baseline --task cifar_baseline --rounds 1 --envs 1 --episode-length 5 --telemetry-dir telemetry --device cpu --devices cpu --num-workers 0`
- Event counts: ANALYTICS_SNAPSHOT=7, EPOCH_COMPLETED=5, SEED_STAGE_CHANGED=3, TRAINING_STARTED=1, SEED_GERMINATED=1, SEED_GATE_EVALUATED=1, SEED_PRUNED=1, EPISODE_OUTCOME=1, VALUE_COLLAPSE_DETECTED=1, PPO_UPDATE_COMPLETED=1, BATCH_EPOCH_COMPLETED=1
- Analytics kinds: last_action=5, batch_stats=1, action_distribution=1
- Ordering constraints: per-epoch EPOCH_COMPLETED precedes ANALYTICS_SNAPSHOT:last_action; batch tail sequence VALUE_COLLAPSE_DETECTED → PPO_UPDATE_COMPLETED → ANALYTICS_SNAPSHOT:batch_stats → BATCH_EPOCH_COMPLETED → ANALYTICS_SNAPSHOT:action_distribution
- Performance baseline: total episode time 44.754748s, episodes/sec 0.022344, per-epoch deltas [7.395514s, 7.369687s, 7.77129s, 7.367458s], avg 7.475987s
- Throughput fields (fps/step_time_ms/dataloader_wait_ms) were null in snapshots; baseline derived from timestamps.
- Phase 3 preflight rerun: `telemetry/telemetry_2026-01-06_205739` total 39.277398s, 0.025460 episodes/sec, per-epoch deltas [7.500104s, 7.575927s, 7.99248s, 7.513749s].
- Type-check gate: CI runs `uv run mypy -p esper` (see `.github/workflows/test-suite.yml`).
- Episode outcome threshold: `EPISODE_SUCCESS_THRESHOLD=80.0` (percent scale),
  validated in `tests/simic/test_episode_outcome_emission.py`.
- Branch status: `env-refactor` ahead of `origin/env-refactor` with local changes; merge to main still pending for entry gates that require it.
