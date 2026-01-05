// Auto-generated from Python schema - DO NOT EDIT
// Run: python scripts/generate_overwatch_types.py
// Generated from: esper.karn.sanctum.schema

export type SeedStage = "UNKNOWN" | "DORMANT" | "GERMINATED" | "TRAINING" | "BLENDING" | "HOLDING" | "FOSSILIZED" | "PRUNED" | "EMBARGOED" | "RESETTING";

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
  decision_id: string;
  env_id: number;
  epoch: number;
  batch: number;
  value_residual: number;
  td_advantage: number | null;
  decision_entropy: number;
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
  op_entropy: number;
  slot_entropy: number;
  blueprint_entropy: number;
  style_entropy: number;
  tempo_entropy: number;
  alpha_target_entropy: number;
  alpha_speed_entropy: number;
  curve_entropy: number;
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
}

export interface TamiyoState {
  entropy: number;
  clip_fraction: number;
  kl_divergence: number;
  explained_variance: number;
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
  head_slot_grad_norm_prev: number;
  head_blueprint_grad_norm_prev: number;
  head_style_grad_norm_prev: number;
  head_tempo_grad_norm_prev: number;
  head_alpha_target_grad_norm_prev: number;
  head_alpha_speed_grad_norm_prev: number;
  head_alpha_curve_grad_norm_prev: number;
  head_op_grad_norm_prev: number;
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
  q_germinate: number;
  q_advance: number;
  q_fossilize: number;
  q_prune: number;
  q_wait: number;
  q_set_alpha: number;
  q_variance: number;
  q_spread: number;
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
  cpu_percent: number | null;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  epochs_per_second: number;
  batches_per_hour: number;
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
  best_seeds: Record<string, SeedState>;
  best_reward_components: RewardComponents | null;
  best_counterfactual_matrix: CounterfactualSnapshot | null;
  best_shapley_snapshot: ShapleySnapshot | null;
  best_action_history: string[];
  action_history: string[];
  action_counts: Record<string, number>;
  total_actions: number;
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
  focused_env_id: number;
  last_action_env_id: number | null;
  last_action_timestamp: string | null;
}

