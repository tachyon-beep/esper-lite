# Phase 1 Preflight Audits (Vectorized Split)

## Objectives
Document risk-reduction audits required before the Phase 1 extraction.

## Planned activities (completed)
- [x] Dependency map for nested helpers in vectorized.py.
- [x] Move list and target module mapping.
- [x] Hot-path fenced region audit (stream sync + batched D2H).
- [x] Import-direction and isolation audit.
- [x] Telemetry event-count baseline procedure.
- [x] Monkeypatch seam audit.
- [x] Decisions: VectorizedPPOTrainer location and action-execution contract.

## Findings

### 1) Nested helper dependency map (vectorized.py)

make_telemetry_callback (vectorized.py:1176)
- Inputs: env_idx, device.
- Captures: hub, emit_with_env_context.
- Side effects: emits telemetry events via callback.
- Target module: env_factory.py.

configure_slot_telemetry (vectorized.py:1188)
- Inputs: env_state, inner_epoch, global_epoch.
- Captures: apply_slot_telemetry, ops_telemetry_enabled, telemetry_lifecycle_only.
- Side effects: mutates slot telemetry + fast_mode flags.
- Target module: env_factory.py.

create_env_state (vectorized.py:1237)
- Inputs: env_idx, base_seed.
- Captures: env_device_map, create_model, task_spec, slots, permissive_gates, auto_forward_gates,
  analytics, ActionEnum, use_telemetry, amp_enabled, resolved_amp_dtype, use_grad_scaler,
  gpu_preload_augment, DEFAULT_GOVERNOR_* constants.
- Side effects: seeds RNG, constructs model/optimizers/streams, updates analytics host params,
  mutates slot flags, initializes env_state accumulators and telemetry.
- Target module: env_factory.py.

_collect_gradient_telemetry_for_batch (vectorized.py:1384)
- Inputs: model, slots_with_active_seeds, env_dev.
- Captures: collect_host_gradients_async, collect_seed_gradients_only_async,
  collect_seed_gradients_async, torch.compiler.disable, SeedStage.
- Side effects: none (returns async stats).
- Target module: batch_ops.py (retain torch.compiler.disable decorator).

_parse_sampled_action (vectorized.py:1441)
- Inputs: env_idx, op_idx, slot_idx, style_idx, alpha_target_idx, slots, slot_config, model.
- Captures: _resolve_target_slot, STYLE_BLEND_IDS, STYLE_ALPHA_ALGORITHMS, ALPHA_TARGET_VALUES,
  OP_* constants, SeedStage, AlphaMode, MIN_PRUNE_AGE, LifecycleOp.
- Side effects: none (pure decode/validation).
- Target module: action_execution.py.

process_train_batch (vectorized.py:1523)
- Inputs: env_state, inputs, targets, criterion, use_telemetry, slots, use_amp, max_grad_norm.
- Captures: task_spec.task_type, task_spec.seed_lr, resolved_amp_dtype, torch_amp.autocast,
  _compiled_loss_and_correct, _collect_gradient_telemetry_for_batch.
- Side effects: creates seed optimizers, zero/backward/step, gradient clipping, stream usage.
- Target module: batch_ops.py.

process_val_batch (vectorized.py:1718)
- Inputs: env_state, inputs, targets, criterion.
- Captures: task_spec.task_type, _compiled_loss_and_correct.
- Side effects: model.eval() + inference.
- Target module: batch_ops.py.

process_fused_val_batch (vectorized.py:1767)
- Inputs: env_state, inputs, targets, criterion, alpha_overrides, num_configs.
- Captures: task_spec.task_type, task_spec.topology, _compiled_loss_and_correct.
- Side effects: model.eval(), fused_forward, stream record usage.
- Target module: counterfactual_eval.py.

batch_signals_to_features (vectorized.py:1857)
- Inputs: batch_signals, batch_slot_reports, slot_config, env_states, device.
- Captures: batch_obs_to_features, max_epochs.
- Side effects: none.
- Target module: batch_ops.py.

### 2) Action execution block dependency summary (vectorized.py:2948+)

The per-env action loop uses and mutates:
- Inputs/config: slot_config, slots, task_spec, reward_config/env_reward_configs,
  reward_family_enum, loss_reward_config, telemetry_config, disable_advance.
- Systems: analytics scoreboard, reward_normalizer, emitters, agent.buffer, obs_normalizer.
- Computations: compute_reward, compute_loss_reward, compute_scaffold_hindsight_credit,
  compute_rent_and_shock_inputs, compute_action_masks, build_slot_states.
- Mutable outputs used after the loop: truncated_bootstrap_targets,
  all_post_action_signals, all_post_action_slot_reports, all_post_action_masks,
  env_final_accs, env_total_rewards, episode_history, episode_outcomes,
  env_rollback_occurred.

### 3) Hot-path fenced regions (do not change sync semantics)

- Training stream sync (wait_stream before batch loop):
  vectorized.py:2009-2053.
- Epoch-end stream synchronize before .item() calls:
  vectorized.py:2117-2124.
- Validation default-stream waits:
  vectorized.py:2309-2317.
- Validation single sync + batched CPU transfer:
  vectorized.py:2431-2441.
- Batched action D2H transfer (single .cpu()):
  vectorized.py:2880-2911.
- Batched head confidence D2H transfer:
  vectorized.py:2913-2931.

### 4) Import-direction and isolation rules

- Keep the lazy get_task_spec import inside train_ppo_vectorized (no new runtime imports
  in extracted modules).
- Do not add new imports in src/esper/simic/training/__init__.py.
- New modules must not import esper.simic.training.vectorized; vectorized.py owns the
  public entrypoint and wires dependencies into the trainer object.
- Keep monkeypatch seams in vectorized.py by passing dependencies into the trainer
  (get_hub, RewardNormalizer, SharedBatchIterator, PPOAgent.load, torch_amp.autocast,
  collect_per_layer_gradients, check_numerical_stability).

### 5) Telemetry event count baseline procedure

Use a short run with --telemetry-dir and count events.jsonl by event_type.
Procedure is documented in baseline_capture.md (Phase 0).

### 6) Action execution contract decision

VectorizedPPOTrainer will live in src/esper/simic/training/vectorized_trainer.py.
action_execution.py should expose a single function that returns a small, explicit
result structure (no ad-hoc dicts). Recommended minimal result fields:
- truncated_bootstrap_targets: list[tuple[int, int]]
- post_action_signals: list[TrainingSignals]
- post_action_slot_reports: list[dict[str, SeedStateReport]]
- post_action_masks: list[dict[str, torch.Tensor]]

Mutable lists (env_final_accs, env_total_rewards, episode_history, episode_outcomes,
env_rollback_occurred) should be passed in explicitly and mutated in place to avoid
hidden side effects.
