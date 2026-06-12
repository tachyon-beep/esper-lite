import type {
  BestRunRecord,
  CounterfactualSnapshot,
  EnvState,
  EventLogEntry,
  RewardComponents,
  RunConfig,
  SanctumSnapshot,
  SeedState,
  ShapleySnapshot,
  SystemVitals,
  TamiyoState
} from '../../types/sanctum'

function merge<T extends object>(defaults: T, overrides: Partial<T>): T {
  return Object.assign({}, defaults, overrides)
}

export function createRewardComponents(
  overrides: Partial<RewardComponents> = {}
): RewardComponents {
  const defaults: RewardComponents = {
    total: 0.5,
    base_acc_delta: 0.1,
    bounded_attribution: 0.2,
    seed_contribution: 0.1,
    escrow_credit_prev: 0,
    escrow_credit_target: 0,
    escrow_delta: 0,
    escrow_credit_next: 0,
    escrow_forfeit: 0,
    compute_rent: -0.05,
    alpha_shock: 0,
    ratio_penalty: 0,
    stage_bonus: 0.05,
    fossilize_terminal_bonus: 0,
    blending_warning: 0,
    holding_warning: 0,
    hindsight_credit: 0,
    scaffold_count: 0,
    avg_scaffold_delay: 0,
    env_id: 0,
    val_acc: 0.87,
    stable_val_acc: null,
    last_action: 'OBSERVE'
  }

  return merge(defaults, overrides)
}

export function createCounterfactualSnapshot(
  overrides: Partial<CounterfactualSnapshot> = {}
): CounterfactualSnapshot {
  const defaults: CounterfactualSnapshot = {
    slot_ids: [],
    configs: [],
    strategy: 'ablation',
    compute_time_ms: 10
  }

  return merge(defaults, overrides)
}

export function createShapleySnapshot(
  overrides: Partial<ShapleySnapshot> = {}
): ShapleySnapshot {
  const defaults: ShapleySnapshot = {
    slot_ids: [],
    values: {},
    epoch: 0,
    timestamp: null
  }

  return merge(defaults, overrides)
}

export function createSeedState(overrides: Partial<SeedState> = {}): SeedState {
  const defaults: SeedState = {
    slot_id: 'slot_0',
    stage: 'TRAINING',
    blueprint_id: null,
    alpha: 0.0,
    accuracy_delta: 0.01,
    seed_params: 1000,
    grad_ratio: 0.5,
    has_vanishing: false,
    has_exploding: false,
    epochs_in_stage: 10,
    improvement: 0.05,
    prune_reason: '',
    auto_pruned: false,
    epochs_total: 50,
    counterfactual: 0.0,
    blend_tempo_epochs: 0,
    alpha_curve: 'LINEAR',
    contribution_velocity: 0.0,
    interaction_sum: 0.0,
    boost_received: 0.0,
    upstream_alpha_sum: 0.0,
    downstream_alpha_sum: 0.0
  }

  return merge(defaults, overrides)
}

export function createEnvState(overrides: Partial<EnvState> = {}): EnvState {
  const defaults: EnvState = {
    env_id: 0,
    current_epoch: 100,
    host_accuracy: 0.873,
    host_loss: 0.45,
    host_params: 1000000,
    seeds: {},
    active_seed_count: 3,
    fossilized_count: 2,
    pruned_count: 1,
    fossilized_params: 50000,
    blueprint_spawns: {},
    blueprint_prunes: {},
    blueprint_fossilized: {},
    reward_components: createRewardComponents(),
    counterfactual_matrix: createCounterfactualSnapshot(),
    shapley_snapshot: createShapleySnapshot(),
    reward_history: [0.1, 0.2, 0.3],
    accuracy_history: [0.8, 0.85, 0.87],
    cumulative_reward: 0.6,
    best_reward: 0.5,
    best_reward_epoch: 80,
    best_accuracy: 0.89,
    best_accuracy_epoch: 90,
    best_accuracy_episode: 5,
    peak_cumulative_reward: 0.6,
    best_seeds: {},
    best_reward_components: null,
    best_counterfactual_matrix: null,
    best_shapley_snapshot: null,
    best_action_history: [],
    best_blueprint_spawns: {},
    best_blueprint_fossilized: {},
    best_blueprint_prunes: {},
    lifecycle_events: [],
    best_lifecycle_events: [],
    action_history: ['OBSERVE', 'GERMINATE', 'OBSERVE'],
    action_counts: { OBSERVE: 50, GERMINATE: 10 },
    total_actions: 60,
    last_action_success: true,
    gaming_trigger_count: 0,
    total_reward_steps: 0,
    status: 'healthy',
    last_update: '2024-01-01T00:00:00Z',
    epochs_since_improvement: 5,
    stall_counter: 0,
    degraded_counter: 0,
    reward_mode: 'standard',
    rolled_back: false,
    rollback_reason: '',
    rollback_timestamp: null
  }

  return merge(defaults, overrides)
}

export function createTamiyoState(overrides: Partial<TamiyoState> = {}): TamiyoState {
  const defaults: TamiyoState = {
    entropy: 0.5,
    clip_fraction: 0.1,
    kl_divergence: 0.01,
    explained_variance: 0.7,
    policy_loss: 0.05,
    value_loss: 0.1,
    entropy_loss: -0.02,
    grad_norm: 1.5,
    learning_rate: 0.0003,
    entropy_coef: 0.01,
    ratio_mean: 1.0,
    ratio_min: 0.8,
    ratio_max: 1.2,
    ratio_std: 0.1,
    advantage_mean: 0.0,
    advantage_std: 1.0,
    advantage_skewness: 0.0,
    advantage_kurtosis: 0.0,
    advantage_min: -2.0,
    advantage_max: 2.0,
    advantage_raw_mean: 0.0,
    advantage_raw_std: 1.0,
    advantage_positive_ratio: 0.5,
    log_prob_min: -1.0,
    log_prob_max: 0.0,
    decision_density: 1.0,
    forced_step_ratio: 0.0,
    advantage_std_floored: false,
    pre_norm_advantage_std: null,
    decision_density_history: [],
    dead_layers: 0,
    exploding_layers: 0,
    nan_grad_count: 0,
    inf_grad_count: 0,
    head_nan_latch: {},
    head_inf_latch: {},
    layer_gradient_health: { 'policy.0': 0.95, 'policy.1': 0.88, 'value.0': 0.92 },
    entropy_collapsed: false,
    lstm_h_l2_total: null,
    lstm_c_l2_total: null,
    lstm_h_rms: null,
    lstm_c_rms: null,
    lstm_h_env_rms_mean: null,
    lstm_h_env_rms_max: null,
    lstm_c_env_rms_mean: null,
    lstm_c_env_rms_max: null,
    lstm_h_max: null,
    lstm_c_max: null,
    lstm_has_nan: false,
    lstm_has_inf: false,
    update_time_ms: 150,
    early_stop_epoch: null,
    head_slot_entropy: 0.5,
    head_blueprint_entropy: 0.6,
    head_style_entropy: 0.4,
    head_tempo_entropy: 0.55,
    head_alpha_target_entropy: 0.45,
    head_alpha_speed_entropy: 0.5,
    head_alpha_curve_entropy: 0.48,
    head_op_entropy: 0.52,
    head_slot_grad_norm: 0.8,
    head_blueprint_grad_norm: 0.9,
    head_style_grad_norm: 0.7,
    head_tempo_grad_norm: 0.85,
    head_alpha_target_grad_norm: 0.75,
    head_alpha_speed_grad_norm: 0.8,
    head_alpha_curve_grad_norm: 0.78,
    head_op_grad_norm: 0.82,
    head_slot_grad_norm_prev: 0.8,
    head_blueprint_grad_norm_prev: 0.9,
    head_style_grad_norm_prev: 0.7,
    head_tempo_grad_norm_prev: 0.85,
    head_alpha_target_grad_norm_prev: 0.75,
    head_alpha_speed_grad_norm_prev: 0.8,
    head_alpha_curve_grad_norm_prev: 0.78,
    head_op_grad_norm_prev: 0.82,
    head_slot_ratio_max: 1.0,
    head_blueprint_ratio_max: 1.0,
    head_style_ratio_max: 1.0,
    head_tempo_ratio_max: 1.0,
    head_alpha_target_ratio_max: 1.0,
    head_alpha_speed_ratio_max: 1.0,
    head_alpha_curve_ratio_max: 1.0,
    head_op_ratio_max: 1.0,
    joint_ratio_max: 1.0,
    episode_return_history: [0.1, 0.2, 0.3],
    current_episode_return: 0.3,
    current_episode: 5,
    policy_loss_history: [0.1, 0.08, 0.05],
    value_loss_history: [0.2, 0.15, 0.1],
    grad_norm_history: [2.0, 1.8, 1.5],
    entropy_history: [0.6, 0.55, 0.5],
    explained_variance_history: [0.5, 0.6, 0.7],
    kl_divergence_history: [0.02, 0.015, 0.01],
    clip_fraction_history: [0.15, 0.12, 0.1],
    inner_epoch: 3,
    ppo_batch: 10,
    action_counts: { OBSERVE: 50, GERMINATE: 10 },
    total_actions: 60,
    cumulative_action_counts: { OBSERVE: 50, GERMINATE: 10 },
    cumulative_total_actions: 60,
    ppo_data_received: true,
    recent_decisions: [],
    group_id: null,
    entropy_velocity: 0.0,
    collapse_risk_score: 0.0,
    _previous_risk: 0.0,
    entropy_clip_correlation: 0.0,
    value_mean: 0.0,
    value_std: 0.0,
    value_min: 0.0,
    value_max: 0.0,
    initial_value_spread: null,
    op_q_values: [],
    op_valid_mask: [],
    q_variance: 0.0,
    q_spread: 0.0,
    last_action_success: true,
    last_action_op: 'WAIT',
    infrastructure: {
      cuda_memory_allocated_gb: 0,
      cuda_memory_reserved_gb: 0,
      cuda_memory_peak_gb: 0,
      cuda_memory_fragmentation: 0,
      dataloader_wait_ratio: 0,
      compile_enabled: false,
      compile_backend: '',
      compile_mode: ''
    },
    gradient_quality: {
      gradient_cv: 0,
      clip_fraction_positive: 0,
      clip_fraction_negative: 0
    },
    value_function: {
      v_return_correlation: 0,
      td_error_mean: 0,
      td_error_std: 0,
      bellman_error: 0,
      return_p10: 0,
      return_p50: 0,
      return_p90: 0,
      return_skewness: 0,
      return_variance: 0,
      value_predictions: [],
      actual_returns: [],
      td_errors: []
    }
  }

  return merge(defaults, overrides)
}

export function createSystemVitals(overrides: Partial<SystemVitals> = {}): SystemVitals {
  const defaults: SystemVitals = {
    gpu_stats: {},
    gpu_memory_used_gb: 8.0,
    gpu_memory_total_gb: 24.0,
    gpu_utilization: 85,
    gpu_temperature: 65,
    cpu_percent: 45,
    ram_used_gb: 16.0,
    ram_total_gb: 64.0,
    epochs_per_second: 2.5,
    batches_per_hour: 100,
    host_params: 1000000
  }

  return merge(defaults, overrides)
}

export function createEventLogEntry(
  overrides: Partial<EventLogEntry> = {}
): EventLogEntry {
  const defaults: EventLogEntry = {
    timestamp: '2024-01-01T00:00:00Z',
    event_type: 'seed_germinated',
    env_id: 0,
    message: 'Seed germinated in slot_0',
    episode: 5,
    relative_time: '2m ago',
    metadata: {}
  }

  return merge(defaults, overrides)
}

export function createBestRunRecord(
  overrides: Partial<BestRunRecord> = {}
): BestRunRecord {
  const defaults: BestRunRecord = {
    env_id: 0,
    episode: 5,
    peak_accuracy: 0.92,
    final_accuracy: 0.90,
    epoch: 100,
    seeds: {},
    slot_ids: ['slot_0', 'slot_1'],
    growth_ratio: 1.05,
    record_id: 'record-001',
    cumulative_reward: 0.3,
    peak_cumulative_reward: 0.3,
    reward_components: createRewardComponents(),
    counterfactual_matrix: createCounterfactualSnapshot(),
    shapley_snapshot: null,
    action_history: ['OBSERVE', 'GERMINATE'],
    reward_history: [0.1, 0.2],
    accuracy_history: [0.85, 0.90],
    host_loss: 0.3,
    host_params: 1000000,
    fossilized_count: 2,
    pruned_count: 1,
    reward_mode: 'standard',
    blueprint_spawns: {},
    blueprint_fossilized: {},
    blueprint_prunes: {},
    end_seeds: {},
    end_reward_components: null,
    best_lifecycle_events: [],
    end_lifecycle_events: []
  }

  return merge(defaults, overrides)
}

export function createRunConfig(overrides: Partial<RunConfig> = {}): RunConfig {
  const defaults: RunConfig = {
    seed: 42,
    n_episodes: 100,
    lr: 0.0003,
    clip_ratio: 0.2,
    entropy_coef: 0.01,
    param_budget: 1000000,
    resume_path: '',
    entropy_anneal: {}
  }

  return merge(defaults, overrides)
}

export function createSnapshot(
  overrides: Partial<SanctumSnapshot> = {}
): SanctumSnapshot {
  const defaults: SanctumSnapshot = {
    envs: {
      0: createEnvState({ env_id: 0 }),
      1: createEnvState({ env_id: 1, status: 'stalled' })
    },
    tamiyo: createTamiyoState(),
    vitals: createSystemVitals(),
    rewards: createRewardComponents(),
    slot_ids: ['slot_0', 'slot_1', 'slot_2', 'slot_3'],
    current_episode: 5,
    current_batch: 10,
    max_batches: 100,
    current_epoch: 100,
    max_epochs: 1000,
    run_id: 'run-001',
    task_name: 'test-task',
    reward_mode: 'standard',
    run_config: createRunConfig(),
    start_time: '2024-01-01T00:00:00Z',
    connected: true,
    runtime_seconds: 3600,
    staleness_seconds: 0.5,
    captured_at: '2024-01-01T01:00:00Z',
    total_events_received: 100,
    poll_count: 50,
    training_thread_alive: true,
    aggregate_mean_accuracy: 0.85,
    aggregate_mean_reward: 0.4,
    batch_avg_reward: 0.45,
    batch_total_episodes: 8,
    mean_accuracy_history: [0.8, 0.82, 0.85],
    event_log: [
      createEventLogEntry({ event_type: 'seed_germinated', message: 'Seed germinated' }),
      createEventLogEntry({ event_type: 'anomaly_detected', message: 'Gradient explosion detected' })
    ],
    best_runs: [
      createBestRunRecord({ record_id: 'record-001', peak_accuracy: 0.92 }),
      createBestRunRecord({ record_id: 'record-002', peak_accuracy: 0.88 })
    ],
    cumulative_germinated: 15,
    cumulative_fossilized: 10,
    cumulative_pruned: 5,
    cumulative_blueprint_spawns: {},
    cumulative_blueprint_fossilized: {},
    cumulative_blueprint_prunes: {},
    slot_stage_counts: { DORMANT: 1, TRAINING: 2, FOSSILIZED: 1 },
    total_slots: 4,
    active_slots: 3,
    avg_epochs_in_stage: 25,
    last_ppo_update: '2024-01-01T00:59:00Z',
    last_reward_update: '2024-01-01T00:59:30Z',
    seed_lifecycle: {
      germination_count: 15,
      prune_count: 5,
      fossilize_count: 10,
      active_count: 3,
      total_slots: 4,
      germination_rate: 1.5,
      prune_rate: 0.5,
      fossilize_rate: 1.0,
      blend_success_rate: 0.67,
      avg_lifespan_epochs: 20,
      germination_trend: 'stable',
      prune_trend: 'stable',
      fossilize_trend: 'stable'
    },
    observation_stats: {
      slot_features_mean: 0,
      slot_features_std: 1,
      host_features_mean: 0,
      host_features_std: 1,
      context_features_mean: 0,
      context_features_std: 1,
      outlier_pct: 0,
      near_clip_pct: 0,
      clip_pct: 0,
      nan_count: 0,
      inf_count: 0,
      nan_pct: 0,
      inf_pct: 0,
      normalization_drift: 0,
      batch_size: 2
    },
    episode_stats: {
      length_mean: 0,
      length_std: 0,
      length_min: 0,
      length_max: 0,
      total_episodes: 0,
      episodes_per_second: 0,
      timeout_count: 0,
      success_count: 0,
      early_termination_count: 0,
      timeout_rate: 0,
      success_rate: 0,
      early_termination_rate: 0,
      steps_per_germinate: 0,
      steps_per_prune: 0,
      steps_per_fossilize: 0,
      action_entropy: 0,
      yield_rate: 0,
      slot_utilization: 0,
      completion_trend: 'stable'
    },
    focused_env_id: 0,
    last_action_env_id: null,
    last_action_timestamp: null
  }

  return merge(defaults, overrides)
}
