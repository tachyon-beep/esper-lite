// Auto-generated from Python schema - DO NOT EDIT
// Run: npm run generate:types from src/esper/karn/overwatch/web
// Generated from: esper.karn.sanctum.schema

export type SeedStage = "UNKNOWN" | "DORMANT" | "GERMINATED" | "TRAINING" | "BLENDING" | "HOLDING" | "FOSSILIZED" | "PRUNED" | "EMBARGOED" | "RESETTING";

export interface ShapleyEstimate {
  mean: number;
  std: number;
  n_samples: number;
}

export interface ShapleySnapshot {
  slot_ids: string[];
  values: Record<string, ShapleyEstimate>;
  epoch: number;
  timestamp: string | null;
}

export interface CounterfactualConfig {
  seed_mask: boolean[];
  accuracy: number;
}

export interface CounterfactualSnapshot {
  slot_ids: string[];
  configs: CounterfactualConfig[];
  strategy: string;
  compute_time_ms: number;
}

export interface GPUStats {
  device_id: number;
  memory_used_gb: number;
  memory_total_gb: number;
  utilization: number;
  temperature: number;
}

export interface InfrastructureMetrics {
  cuda_memory_allocated_gb: number;
  cuda_memory_reserved_gb: number;
  cuda_memory_peak_gb: number;
  cuda_memory_fragmentation: number;
  dataloader_wait_ratio: number;
  compile_enabled: boolean;
  compile_backend: string;
  compile_mode: string;
}

export interface GradientQualityMetrics {
  gradient_cv: number;
  clip_fraction_positive: number;
  clip_fraction_negative: number;
}

export interface ValueFunctionMetrics {
  v_return_correlation: number;
  td_error_mean: number;
  td_error_std: number;
  bellman_error: number;
  return_p10: number;
  return_p50: number;
  return_p90: number;
  return_skewness: number;
  return_variance: number;
  value_predictions: number[];
  actual_returns: number[];
  td_errors: number[];
}

export interface SeedLifecycleEvent {
  epoch: number;
  action: string;
  from_stage: string;
  to_stage: string;
  blueprint_id: string;
  slot_id: string;
  alpha: number | null;
  accuracy_delta: number | null;
  morphology_proposal_id: string | null;
  morphology_verdict_id: string | null;
  morphology_mutation_id: string | null;
  rng_stream: string | null;
  rng_seed: number | null;
}

export interface SeedLifecycleStats {
  germination_count: number;
  prune_count: number;
  fossilize_count: number;
  active_count: number;
  total_slots: number;
  germination_rate: number;
  prune_rate: number;
  fossilize_rate: number;
  blend_success_rate: number;
  avg_lifespan_epochs: number;
  germination_trend: string;
  prune_trend: string;
  fossilize_trend: string;
}

export interface MorphologyCausalLogEntry {
  phase: string;
  env_id: number;
  slot_id: string;
  operation: string;
  action_id: string;
  proposal_id: string;
  verdict_id: string;
  mutation_id: string;
  observation_hash: string;
  rng_stream: string;
  rng_seed: number;
  topology: string;
  blueprint_id: string | null;
  governor_approved: boolean | null;
  governor_reason: string | null;
  governor_blocked_factor: string | null;
  watch_window_evidence: number | null;
  linked_event_id: string | null;
}

export interface ObservationStats {
  slot_features_mean: number;
  slot_features_std: number;
  host_features_mean: number;
  host_features_std: number;
  context_features_mean: number;
  context_features_std: number;
  outlier_pct: number;
  near_clip_pct: number;
  clip_pct: number;
  nan_count: number;
  inf_count: number;
  nan_pct: number;
  inf_pct: number;
  normalization_drift: number;
  batch_size: number;
}

export interface EpisodeStats {
  length_mean: number;
  length_std: number;
  length_min: number;
  length_max: number;
  total_episodes: number;
  episodes_per_second: number;
  timeout_count: number;
  success_count: number;
  early_termination_count: number;
  timeout_rate: number;
  success_rate: number;
  early_termination_rate: number;
  steps_per_germinate: number;
  steps_per_prune: number;
  steps_per_fossilize: number;
  action_entropy: number;
  yield_rate: number;
  slot_utilization: number;
  completion_trend: string;
}

export interface SeedState {
  slot_id: string;
  stage: string;
  blueprint_id: string | null;
  alpha: number;
  accuracy_delta: number;
  seed_params: number;
  grad_ratio: number;
  has_vanishing: boolean;
  has_exploding: boolean;
  epochs_in_stage: number;
  improvement: number;
  prune_reason: string;
  auto_pruned: boolean;
  epochs_total: number;
  counterfactual: number;
  blend_tempo_epochs: number;
  alpha_curve: string;
  contribution_velocity: number;
  interaction_sum: number;
  boost_received: number;
  upstream_alpha_sum: number;
  downstream_alpha_sum: number;
}

export interface RewardComponents {
  total: number;
  base_acc_delta: number;
  bounded_attribution: number;
  seed_contribution: number;
  escrow_credit_prev: number;
  escrow_credit_target: number;
  escrow_delta: number;
  escrow_credit_next: number;
  escrow_forfeit: number;
  compute_rent: number;
  alpha_shock: number;
  ratio_penalty: number;
  stage_bonus: number;
  fossilize_terminal_bonus: number;
  blending_warning: number;
  holding_warning: number;
  hindsight_credit: number;
  scaffold_count: number;
  avg_scaffold_delay: number;
  env_id: number;
  val_acc: number;
  stable_val_acc: number | null;
  last_action: string;
}

export interface DecisionSnapshot {
  timestamp: string;
  slot_states: Record<string, string>;
  host_accuracy: number;
  chosen_action: string;
  chosen_slot: string | null;
  confidence: number;
  expected_value: number;
  actual_reward: number | null;
  alternatives: [string, number][];
  action_success: boolean | null;
  decision_id: string;
  env_id: number;
  episode: number;
  epoch: number;
  batch: number;
  value_residual: number;
  td_advantage: number | null;
  decision_entropy: number | null;
  chosen_blueprint: string | null;
  chosen_tempo: string | null;
  chosen_style: string | null;
  chosen_curve: string | null;
  chosen_alpha_target: string | null;
  chosen_alpha_speed: string | null;
  op_confidence: number;
  slot_confidence: number;
  blueprint_confidence: number;
  style_confidence: number;
  tempo_confidence: number;
  alpha_target_confidence: number;
  alpha_speed_confidence: number;
  curve_confidence: number;
  op_entropy: number | null;
  slot_entropy: number | null;
  blueprint_entropy: number | null;
  style_entropy: number | null;
  tempo_entropy: number | null;
  alpha_target_entropy: number | null;
  alpha_speed_entropy: number | null;
  curve_entropy: number | null;
}

export interface EventLogEntry {
  timestamp: string;
  event_type: string;
  env_id: number | null;
  message: string;
  episode: number;
  relative_time: string;
  metadata: Record<string, string | number>;
}

export interface RunConfig {
  seed: number | null;
  n_episodes: number;
  lr: number;
  clip_ratio: number;
  entropy_coef: number;
  param_budget: number;
  resume_path: string;
  entropy_anneal: Record<string, number>;
}

export interface BestRunRecord {
  env_id: number;
  episode: number;
  peak_accuracy: number;
  final_accuracy: number;
  epoch: number;
  seeds: Record<string, SeedState>;
  slot_ids: string[];
  growth_ratio: number;
  record_id: string;
  cumulative_reward: number;
  peak_cumulative_reward: number;
  reward_components: RewardComponents | null;
  counterfactual_matrix: CounterfactualSnapshot | null;
  shapley_snapshot: ShapleySnapshot | null;
  action_history: string[];
  reward_history: number[];
  accuracy_history: number[];
  host_loss: number;
  host_params: number;
  fossilized_count: number;
  pruned_count: number;
  reward_mode: string | null;
  blueprint_spawns: Record<string, number>;
  blueprint_fossilized: Record<string, number>;
  blueprint_prunes: Record<string, number>;
  end_seeds: Record<string, SeedState>;
  end_reward_components: RewardComponents | null;
  best_lifecycle_events: SeedLifecycleEvent[];
  end_lifecycle_events: SeedLifecycleEvent[];
}

export interface TamiyoState {
  entropy: number;
  clip_fraction: number;
  kl_divergence: number;
  explained_variance: number;
  value_nrmse: number;
  ev_low_return_variance: boolean;
  ev_return_variance: number | null;
  rollback_attempt_count: number;
  rollback_unattributed_count: number;
  policy_loss: number;
  value_loss: number;
  entropy_loss: number;
  grad_norm: number;
  learning_rate: number | null;
  entropy_coef: number;
  ratio_mean: number;
  ratio_min: number;
  ratio_max: number;
  ratio_std: number;
  advantage_mean: number;
  advantage_std: number;
  advantage_skewness: number;
  advantage_kurtosis: number;
  advantage_min: number;
  advantage_max: number;
  advantage_raw_mean: number;
  advantage_raw_std: number;
  advantage_positive_ratio: number;
  log_prob_min: number;
  log_prob_max: number;
  decision_density: number;
  forced_step_ratio: number;
  advantage_std_floored: boolean;
  pre_norm_advantage_std: number | null;
  decision_density_history: number[];
  dead_layers: number;
  exploding_layers: number;
  nan_grad_count: number;
  inf_grad_count: number;
  head_nan_latch: Record<string, boolean>;
  head_inf_latch: Record<string, boolean>;
  layer_gradient_health: Record<string, number> | null;
  entropy_collapsed: boolean;
  lstm_h_l2_total: number | null;
  lstm_c_l2_total: number | null;
  lstm_h_rms: number | null;
  lstm_c_rms: number | null;
  lstm_h_env_rms_mean: number | null;
  lstm_h_env_rms_max: number | null;
  lstm_c_env_rms_mean: number | null;
  lstm_c_env_rms_max: number | null;
  lstm_h_max: number | null;
  lstm_c_max: number | null;
  lstm_has_nan: boolean;
  lstm_has_inf: boolean;
  update_time_ms: number;
  early_stop_epoch: number | null;
  head_slot_entropy: number;
  head_blueprint_entropy: number;
  head_style_entropy: number;
  head_tempo_entropy: number;
  head_alpha_target_entropy: number;
  head_alpha_speed_entropy: number;
  head_alpha_curve_entropy: number;
  head_op_entropy: number;
  head_slot_grad_norm: number;
  head_blueprint_grad_norm: number;
  head_style_grad_norm: number;
  head_tempo_grad_norm: number;
  head_alpha_target_grad_norm: number;
  head_alpha_speed_grad_norm: number;
  head_alpha_curve_grad_norm: number;
  head_op_grad_norm: number;
  head_q_grad_norm: number;
  head_slot_grad_norm_prev: number;
  head_blueprint_grad_norm_prev: number;
  head_style_grad_norm_prev: number;
  head_tempo_grad_norm_prev: number;
  head_alpha_target_grad_norm_prev: number;
  head_alpha_speed_grad_norm_prev: number;
  head_alpha_curve_grad_norm_prev: number;
  head_op_grad_norm_prev: number;
  head_q_grad_norm_prev: number;
  head_slot_ratio_max: number;
  head_blueprint_ratio_max: number;
  head_style_ratio_max: number;
  head_tempo_ratio_max: number;
  head_alpha_target_ratio_max: number;
  head_alpha_speed_ratio_max: number;
  head_alpha_curve_ratio_max: number;
  head_op_ratio_max: number;
  joint_ratio_max: number;
  episode_return_history: number[];
  current_episode_return: number;
  current_episode: number;
  policy_loss_history: number[];
  value_loss_history: number[];
  grad_norm_history: number[];
  entropy_history: number[];
  explained_variance_history: number[];
  kl_divergence_history: number[];
  clip_fraction_history: number[];
  inner_epoch: number;
  ppo_batch: number;
  action_counts: Record<string, number>;
  total_actions: number;
  cumulative_action_counts: Record<string, number>;
  cumulative_total_actions: number;
  ppo_data_received: boolean;
  recent_decisions: DecisionSnapshot[];
  group_id: string | null;
  entropy_velocity: number;
  collapse_risk_score: number;
  _previous_risk: number;
  entropy_clip_correlation: number;
  value_mean: number;
  value_std: number;
  value_min: number;
  value_max: number;
  initial_value_spread: number | null;
  op_q_values: number[];
  op_valid_mask: boolean[];
  q_variance: number;
  q_spread: number;
  q_aux_loss: number;
  head_q_gradient_state: string;
  last_action_success: boolean;
  last_action_op: string;
  infrastructure: InfrastructureMetrics;
  gradient_quality: GradientQualityMetrics;
  value_function: ValueFunctionMetrics;
}

export interface SystemVitals {
  gpu_stats: Record<number, GPUStats>;
  gpu_memory_used_gb: number;
  gpu_memory_total_gb: number;
  gpu_utilization: number;
  gpu_temperature: number;
  gpu_data_present: boolean;
  cpu_percent: number | null;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  epochs_per_second: number;
  batches_per_hour: number;
  throughput_present: boolean;
  host_params: number;
}

export interface EnvState {
  env_id: number;
  current_epoch: number;
  host_accuracy: number;
  host_loss: number;
  host_params: number;
  seeds: Record<string, SeedState>;
  active_seed_count: number;
  fossilized_count: number;
  pruned_count: number;
  fossilized_params: number;
  blueprint_spawns: Record<string, number>;
  blueprint_prunes: Record<string, number>;
  blueprint_fossilized: Record<string, number>;
  reward_components: RewardComponents;
  counterfactual_matrix: CounterfactualSnapshot;
  shapley_snapshot: ShapleySnapshot;
  reward_history: number[];
  accuracy_history: number[];
  cumulative_reward: number;
  best_reward: number;
  best_reward_epoch: number;
  best_accuracy: number;
  best_accuracy_epoch: number;
  best_accuracy_episode: number;
  peak_cumulative_reward: number;
  best_seeds: Record<string, SeedState>;
  best_reward_components: RewardComponents | null;
  best_counterfactual_matrix: CounterfactualSnapshot | null;
  best_shapley_snapshot: ShapleySnapshot | null;
  best_action_history: string[];
  best_blueprint_spawns: Record<string, number>;
  best_blueprint_fossilized: Record<string, number>;
  best_blueprint_prunes: Record<string, number>;
  lifecycle_events: SeedLifecycleEvent[];
  best_lifecycle_events: SeedLifecycleEvent[];
  action_history: string[];
  action_counts: Record<string, number>;
  total_actions: number;
  last_action_success: boolean;
  gaming_trigger_count: number;
  total_reward_steps: number;
  status: string;
  last_update: string | null;
  epochs_since_improvement: number;
  stall_counter: number;
  degraded_counter: number;
  reward_mode: string | null;
  rolled_back: boolean;
  rollback_reason: string;
  rollback_timestamp: string | null;
}

export interface SanctumSnapshot {
  envs: Record<number, EnvState>;
  tamiyo: TamiyoState;
  vitals: SystemVitals;
  rewards: RewardComponents;
  slot_ids: string[];
  current_episode: number;
  current_batch: number;
  max_batches: number;
  current_epoch: number;
  max_epochs: number;
  run_id: string;
  task_name: string;
  reward_mode: string | null;
  run_config: RunConfig;
  start_time: string | null;
  connected: boolean;
  runtime_seconds: number;
  staleness_seconds: number;
  captured_at: string;
  total_events_received: number;
  poll_count: number;
  training_thread_alive: boolean | null;
  aggregate_mean_accuracy: number;
  aggregate_mean_reward: number;
  batch_avg_reward: number;
  batch_total_episodes: number;
  mean_accuracy_history: number[];
  event_log: EventLogEntry[];
  best_runs: BestRunRecord[];
  cumulative_germinated: number;
  cumulative_fossilized: number;
  cumulative_pruned: number;
  cumulative_blueprint_spawns: Record<string, number>;
  cumulative_blueprint_fossilized: Record<string, number>;
  cumulative_blueprint_prunes: Record<string, number>;
  slot_stage_counts: Record<string, number>;
  total_slots: number;
  active_slots: number;
  avg_epochs_in_stage: number;
  last_ppo_update: string | null;
  last_reward_update: string | null;
  seed_lifecycle: SeedLifecycleStats;
  observation_stats: ObservationStats;
  episode_stats: EpisodeStats;
  morphology_causal_log: MorphologyCausalLogEntry[];
  focused_env_id: number;
  last_action_env_id: number | null;
  last_action_timestamp: string | null;
}
