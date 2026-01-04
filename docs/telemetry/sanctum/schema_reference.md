# Sanctum Schema Reference

Complete reference for all dataclasses in `src/esper/karn/sanctum/schema.py`.

This schema defines the telemetry data structures used by Sanctum (TUI/web dashboards) to display training state.

---

## Top-Level: SanctumSnapshot

**Purpose:** Complete snapshot of Sanctum state for rendering. This is the single source of truth passed to all widgets.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `envs` | `dict[int, EnvState]` | `{}` | Per-environment state, keyed by env_id |
| `tamiyo` | `TamiyoState` | factory | Policy agent (Tamiyo) state |
| `vitals` | `SystemVitals` | factory | System resource metrics (GPU, RAM, throughput) |
| `rewards` | `RewardComponents` | factory | Focused env reward breakdown |
| `slot_ids` | `list[str]` | `[]` | Slot configuration (dynamic based on config) |
| `current_episode` | `int` | `0` | Current episode/batch number |
| `current_batch` | `int` | `0` | Current batch number |
| `max_batches` | `int` | `0` | Total episodes/batches in run (from CLI) |
| `current_epoch` | `int` | `0` | Current epoch within episode |
| `max_epochs` | `int` | `0` | Max epochs per episode |
| `run_id` | `str` | `""` | Unique run identifier |
| `task_name` | `str` | `""` | Task being trained (e.g., "CIFAR10") |
| `run_config` | `RunConfig` | factory | Training hyperparameters |
| `start_time` | `datetime \| None` | `None` | Training start timestamp |
| `connected` | `bool` | `False` | Whether telemetry connection is active |
| `runtime_seconds` | `float` | `0.0` | Elapsed training time |
| `staleness_seconds` | `float` | `inf` | Seconds since last telemetry update |
| `captured_at` | `str` | `""` | ISO timestamp when snapshot was captured |
| `total_events_received` | `int` | `0` | Debug: total events received by backend |
| `poll_count` | `int` | `0` | Debug: number of UI poll cycles |
| `training_thread_alive` | `bool \| None` | `None` | Debug: is training thread running? |
| `aggregate_mean_accuracy` | `float` | `0.0` | Mean accuracy across all envs |
| `aggregate_mean_reward` | `float` | `0.0` | Mean reward across all envs |
| `batch_avg_reward` | `float` | `0.0` | Average reward for last batch |
| `batch_total_episodes` | `int` | `0` | Total episodes in training run |
| `mean_accuracy_history` | `deque[float]` | maxlen=50 | Rolling mean accuracy over time |
| `event_log` | `list[EventLogEntry]` | `[]` | Event log (most recent last) |
| `best_runs` | `list[BestRunRecord]` | `[]` | Top 10 runs by peak accuracy |
| `cumulative_germinated` | `int` | `0` | Total germinations across run |
| `cumulative_fossilized` | `int` | `0` | Total fossilizations across run |
| `cumulative_pruned` | `int` | `0` | Total prunes across run |
| `cumulative_blueprint_spawns` | `dict[str, int]` | `{}` | Per-blueprint spawn counts |
| `cumulative_blueprint_fossilized` | `dict[str, int]` | `{}` | Per-blueprint fossilize counts |
| `cumulative_blueprint_prunes` | `dict[str, int]` | `{}` | Per-blueprint prune counts |
| `slot_stage_counts` | `dict[str, int]` | `{}` | Aggregate slot counts by stage |
| `total_slots` | `int` | `0` | n_envs * n_slots_per_env |
| `active_slots` | `int` | `0` | Slots not in DORMANT state |
| `avg_epochs_in_stage` | `float` | `0.0` | Mean epochs_in_stage for non-dormant slots |
| `last_ppo_update` | `datetime \| None` | `None` | Timestamp of last PPO update |
| `last_reward_update` | `datetime \| None` | `None` | Timestamp of last reward update |
| `seed_lifecycle` | `SeedLifecycleStats` | factory | Seed lifecycle aggregate metrics |
| `observation_stats` | `ObservationStats` | factory | Observation health metrics |
| `episode_stats` | `EpisodeStats` | factory | Episode-level aggregate metrics |
| `focused_env_id` | `int` | `0` | Focused env for detail panel |
| `last_action_env_id` | `int \| None` | `None` | Env ID of last action (for highlighting) |
| `last_action_timestamp` | `datetime \| None` | `None` | Timestamp of last action (5s decay) |

**Property:**
- `is_stale` -> `bool`: Returns `True` if `staleness_seconds > 5.0`

---

## EnvState

**Purpose:** Per-environment state for multi-env tracking. Contains seeds, metrics, and history for one training environment.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `env_id` | `int` | required | Environment identifier |
| `current_epoch` | `int` | `0` | Current epoch within episode |
| `host_accuracy` | `float` | `0.0` | Current host model accuracy |
| `host_loss` | `float` | `0.0` | Current host model loss |
| `host_params` | `int` | `0` | Host model parameter count |
| `seeds` | `dict[str, SeedState]` | `{}` | Seed slots keyed by slot_id |
| `active_seed_count` | `int` | `0` | Number of active (non-dormant) seeds |
| `fossilized_count` | `int` | `0` | Number of fossilized seeds |
| `pruned_count` | `int` | `0` | Number of pruned seeds |
| `fossilized_params` | `int` | `0` | Total params in FOSSILIZED seeds |
| `blueprint_spawns` | `dict[str, int]` | `{}` | Per-blueprint spawn count |
| `blueprint_prunes` | `dict[str, int]` | `{}` | Per-blueprint prune count |
| `blueprint_fossilized` | `dict[str, int]` | `{}` | Per-blueprint fossilize count |
| `reward_components` | `RewardComponents` | factory | Reward breakdown from REWARD_COMPUTED |
| `counterfactual_matrix` | `CounterfactualSnapshot` | factory | Counterfactual attribution matrix |
| `shapley_snapshot` | `ShapleySnapshot` | factory | Shapley value attribution |
| `reward_history` | `deque[float]` | maxlen=50 | Recent rewards for sparkline |
| `accuracy_history` | `deque[float]` | maxlen=50 | Recent accuracies for sparkline |
| `cumulative_reward` | `float` | `0.0` | Sum of all rewards (entire episode) |
| `best_reward` | `float` | `-inf` | Best reward achieved |
| `best_reward_epoch` | `int` | `0` | Epoch when best reward achieved |
| `best_accuracy` | `float` | `0.0` | Best accuracy achieved |
| `best_accuracy_epoch` | `int` | `0` | Epoch when best accuracy achieved |
| `best_accuracy_episode` | `int` | `0` | Episode when best accuracy achieved |
| `best_seeds` | `dict[str, SeedState]` | `{}` | Seeds contributing at peak accuracy |
| `best_reward_components` | `RewardComponents \| None` | `None` | Reward breakdown at peak |
| `best_counterfactual_matrix` | `CounterfactualSnapshot \| None` | `None` | Counterfactual at peak |
| `best_shapley_snapshot` | `ShapleySnapshot \| None` | `None` | Shapley values at peak |
| `best_action_history` | `list[str]` | `[]` | Actions leading to peak |
| `action_history` | `deque[str]` | maxlen=10 | Recent actions (normalized) |
| `action_counts` | `dict[str, int]` | preset | Action counts by type |
| `total_actions` | `int` | `0` | Total action count |
| `gaming_trigger_count` | `int` | `0` | Steps with anti-gaming penalties |
| `total_reward_steps` | `int` | `0` | Total steps with reward computed |
| `status` | `str` | `"initializing"` | Env status (initializing/healthy/excellent/stalled/degraded) |
| `last_update` | `datetime \| None` | `None` | Last update timestamp |
| `epochs_since_improvement` | `int` | `0` | Epochs since accuracy improved |
| `stall_counter` | `int` | `0` | Hysteresis counter for stall detection |
| `degraded_counter` | `int` | `0` | Hysteresis counter for degraded detection |
| `reward_mode` | `str \| None` | `None` | A/B test cohort (shaped/simplified/sparse) |
| `rolled_back` | `bool` | `False` | Governor rollback state |
| `rollback_reason` | `str` | `""` | Rollback reason (nan/lobotomy/divergence) |
| `rollback_timestamp` | `datetime \| None` | `None` | When rollback occurred |

**Properties:**
- `current_reward` -> `float`: Most recent reward from history
- `mean_reward` -> `float`: Mean reward over history
- `gaming_rate` -> `float`: Fraction of steps with anti-gaming penalties
- `growth_ratio` -> `float`: `(host_params + fossilized_params) / host_params`

**Methods:**
- `add_reward(reward, epoch)`: Add reward, update cumulative/best tracking
- `add_accuracy(accuracy, epoch, episode)`: Add accuracy, update best/status, snapshot seeds at peak
- `add_action(action_name)`: Track action (normalizes factored actions to base types)
- `_update_status(prev_acc, curr_acc)`: Update status with hysteresis

**Action Normalization:** `GERMINATE_CONV_LIGHT` -> `GERMINATE`, `FOSSILIZE_R0C0` -> `FOSSILIZE`, etc.

---

## SeedState

**Purpose:** State of a single seed slot. Uses `slots=True` for memory efficiency.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `slot_id` | `str` | required | Slot identifier (e.g., "r0c0") |
| `stage` | `str` | `"DORMANT"` | Lifecycle stage |
| `blueprint_id` | `str \| None` | `None` | Blueprint type (e.g., "conv_light") |
| `alpha` | `float` | `0.0` | Blend coefficient (0-1) |
| `accuracy_delta` | `float` | `0.0` | Accuracy contribution |
| `seed_params` | `int` | `0` | Seed parameter count |
| `grad_ratio` | `float` | `0.0` | Gradient health ratio |
| `has_vanishing` | `bool` | `False` | Vanishing gradient flag |
| `has_exploding` | `bool` | `False` | Exploding gradient flag |
| `epochs_in_stage` | `int` | `0` | Epochs in current stage |
| `improvement` | `float` | `0.0` | Accuracy improvement when fossilized |
| `prune_reason` | `str` | `""` | Why seed was pruned |
| `auto_pruned` | `bool` | `False` | True if auto-pruned vs policy decision |
| `epochs_total` | `int` | `0` | Total epochs seed was alive |
| `counterfactual` | `float` | `0.0` | Causal attribution score |
| `blend_tempo_epochs` | `int` | `5` | Integration speed (3=FAST, 5=STANDARD, 8=SLOW) |
| `alpha_curve` | `str` | `"LINEAR"` | Alpha curve shape |
| `contribution_velocity` | `float` | `0.0` | EMA of contribution changes |
| `interaction_sum` | `float` | `0.0` | Total synergy from interactions |
| `boost_received` | `float` | `0.0` | Strongest interaction partner value |
| `upstream_alpha_sum` | `float` | `0.0` | Sum of alphas for slots j < i |
| `downstream_alpha_sum` | `float` | `0.0` | Sum of alphas for slots j > i |

---

## TamiyoState

**Purpose:** Tamiyo policy agent state - all metrics from existing TUI.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `entropy` | `float` | `0.0` | Policy entropy |
| `clip_fraction` | `float` | `0.0` | PPO clip fraction |
| `kl_divergence` | `float` | `0.0` | KL divergence from old policy |
| `explained_variance` | `float` | `0.0` | Value function explained variance |
| `policy_loss` | `float` | `0.0` | Policy loss |
| `value_loss` | `float` | `0.0` | Value function loss |
| `entropy_loss` | `float` | `0.0` | Entropy loss |
| `grad_norm` | `float` | `0.0` | Gradient norm |
| `learning_rate` | `float \| None` | `None` | Current learning rate |
| `entropy_coef` | `float` | `0.0` | Entropy coefficient (adaptive) |
| `ratio_mean` | `float` | `1.0` | Mean PPO ratio |
| `ratio_min` | `float` | `1.0` | Min PPO ratio |
| `ratio_max` | `float` | `1.0` | Max PPO ratio |
| `ratio_std` | `float` | `0.0` | Std of PPO ratio |
| `advantage_mean` | `float` | `0.0` | Post-norm advantage mean |
| `advantage_std` | `float` | `0.0` | Post-norm advantage std |
| `advantage_skewness` | `float` | `NaN` | Advantage skewness |
| `advantage_kurtosis` | `float` | `NaN` | Advantage kurtosis |
| `advantage_min` | `float` | `0.0` | Advantage min |
| `advantage_max` | `float` | `0.0` | Advantage max |
| `advantage_raw_mean` | `float` | `0.0` | Pre-norm advantage mean |
| `advantage_raw_std` | `float` | `0.0` | Pre-norm advantage std |
| `advantage_positive_ratio` | `float` | `NaN` | Fraction of positive advantages |
| `log_prob_min` | `float` | `NaN` | Most negative log prob (NaN predictor) |
| `log_prob_max` | `float` | `NaN` | Highest log prob |
| `dead_layers` | `int` | `0` | Layers with vanishing gradients |
| `exploding_layers` | `int` | `0` | Layers with exploding gradients |
| `nan_grad_count` | `int` | `0` | NaN gradient count |
| `inf_grad_count` | `int` | `0` | Inf gradient count |
| `head_nan_latch` | `dict[str, bool]` | preset | Per-head NaN latch (stays True once triggered) |
| `head_inf_latch` | `dict[str, bool]` | preset | Per-head Inf latch (stays True once triggered) |
| `layer_gradient_health` | `dict[str, float] \| None` | `None` | Per-layer gradient health |
| `entropy_collapsed` | `bool` | `False` | Entropy collapse detected |
| `update_time_ms` | `float` | `0.0` | PPO update duration (ms) |
| `early_stop_epoch` | `int \| None` | `None` | KL early stopping epoch |
| `head_slot_entropy` | `float` | `0.0` | Slot head entropy |
| `head_blueprint_entropy` | `float` | `0.0` | Blueprint head entropy |
| `head_style_entropy` | `float` | `0.0` | Style head entropy |
| `head_tempo_entropy` | `float` | `0.0` | Tempo head entropy |
| `head_alpha_target_entropy` | `float` | `0.0` | Alpha target head entropy |
| `head_alpha_speed_entropy` | `float` | `0.0` | Alpha speed head entropy |
| `head_alpha_curve_entropy` | `float` | `0.0` | Alpha curve head entropy |
| `head_op_entropy` | `float` | `0.0` | Op head entropy |
| `head_slot_grad_norm` | `float` | `0.0` | Slot head gradient norm |
| `head_blueprint_grad_norm` | `float` | `0.0` | Blueprint head gradient norm |
| `head_style_grad_norm` | `float` | `0.0` | Style head gradient norm |
| `head_tempo_grad_norm` | `float` | `0.0` | Tempo head gradient norm |
| `head_alpha_target_grad_norm` | `float` | `0.0` | Alpha target head gradient norm |
| `head_alpha_speed_grad_norm` | `float` | `0.0` | Alpha speed head gradient norm |
| `head_alpha_curve_grad_norm` | `float` | `0.0` | Alpha curve head gradient norm |
| `head_op_grad_norm` | `float` | `0.0` | Op head gradient norm |
| `head_*_grad_norm_prev` | `float` | `0.0` | Previous grad norms (for trend detection) |
| `head_*_ratio_max` | `float` | `1.0` | Per-head max PPO ratios |
| `joint_ratio_max` | `float` | `1.0` | Product of per-head ratios |
| `episode_return_history` | `deque[float]` | maxlen=20 | Episode return history |
| `current_episode_return` | `float` | `0.0` | Current episode return |
| `current_episode` | `int` | `0` | Current episode number |
| `policy_loss_history` | `deque[float]` | maxlen=10 | Policy loss history |
| `value_loss_history` | `deque[float]` | maxlen=10 | Value loss history |
| `grad_norm_history` | `deque[float]` | maxlen=10 | Gradient norm history |
| `entropy_history` | `deque[float]` | maxlen=10 | Entropy history |
| `explained_variance_history` | `deque[float]` | maxlen=10 | Explained variance history |
| `kl_divergence_history` | `deque[float]` | maxlen=10 | KL divergence history |
| `clip_fraction_history` | `deque[float]` | maxlen=10 | Clip fraction history |
| `inner_epoch` | `int` | `0` | Current inner optimization epoch |
| `ppo_batch` | `int` | `0` | Current batch within PPO update |
| `action_counts` | `dict[str, int]` | `{}` | Current batch action counts |
| `total_actions` | `int` | `0` | Current batch total actions |
| `cumulative_action_counts` | `dict[str, int]` | `{}` | All-time action counts |
| `cumulative_total_actions` | `int` | `0` | All-time total actions |
| `ppo_data_received` | `bool` | `False` | PPO data received flag |
| `recent_decisions` | `list[DecisionSnapshot]` | `[]` | Recent decisions (up to 8, 2min expiry) |
| `group_id` | `str \| None` | `None` | A/B testing group |
| `entropy_velocity` | `float` | `0.0` | d(entropy)/d(batch), negative = declining |
| `collapse_risk_score` | `float` | `0.0` | 0.0-1.0, >0.7 = high risk |
| `_previous_risk` | `float` | `0.0` | For hysteresis (not serialized) |
| `entropy_clip_correlation` | `float` | `0.0` | Entropy-clip correlation |
| `value_mean` | `float` | `0.0` | Value function mean |
| `value_std` | `float` | `0.0` | Value function std |
| `value_min` | `float` | `0.0` | Value function min |
| `value_max` | `float` | `0.0` | Value function max |
| `initial_value_spread` | `float \| None` | `None` | Set after warmup for relative thresholds |
| `q_germinate` | `float` | `0.0` | Q(s, GERMINATE) |
| `q_advance` | `float` | `0.0` | Q(s, ADVANCE) |
| `q_fossilize` | `float` | `0.0` | Q(s, FOSSILIZE) |
| `q_prune` | `float` | `0.0` | Q(s, PRUNE) |
| `q_wait` | `float` | `0.0` | Q(s, WAIT) |
| `q_set_alpha` | `float` | `0.0` | Q(s, SET_ALPHA) |
| `q_variance` | `float` | `0.0` | Variance across op Q-values |
| `q_spread` | `float` | `0.0` | max(Q) - min(Q) across ops |
| `last_action_success` | `bool` | `True` | Whether previous action succeeded |
| `last_action_op` | `str` | `"WAIT"` | Previous operation |
| `infrastructure` | `InfrastructureMetrics` | factory | PyTorch infrastructure metrics |
| `gradient_quality` | `GradientQualityMetrics` | factory | Gradient quality diagnostics |
| `value_function` | `ValueFunctionMetrics` | factory | Value function diagnostics |

---

## DecisionSnapshot

**Purpose:** Snapshot of a single Tamiyo decision for display. Captures what Tamiyo saw, what she chose, and the outcome.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `timestamp` | `datetime` | required | When decision was made |
| `slot_states` | `dict[str, str]` | required | slot_id -> "Training 12%" or "Empty" |
| `host_accuracy` | `float` | required | Host accuracy at decision time |
| `chosen_action` | `str` | required | Chosen action (GERMINATE/ADVANCE/etc.) |
| `chosen_slot` | `str \| None` | required | Target slot (None for WAIT) |
| `confidence` | `float` | required | Action probability (0-1) |
| `expected_value` | `float` | required | Value estimate before action |
| `actual_reward` | `float \| None` | required | Actual reward (None if pending) |
| `alternatives` | `list[tuple[str, float]]` | required | [(action_name, probability), ...] |
| `decision_id` | `str` | `""` | Unique ID for click targeting |
| `env_id` | `int` | `0` | Environment that made decision |
| `epoch` | `int` | `0` | Epoch when decision was made |
| `batch` | `int` | `0` | Batch when decision was made |
| `value_residual` | `float` | `0.0` | r - V(s): immediate reward minus value |
| `td_advantage` | `float \| None` | `None` | r + gamma*V(s') - V(s) (None until next step) |
| `decision_entropy` | `float` | `0.0` | -sum(p*log(p)) for this action distribution |
| `chosen_blueprint` | `str \| None` | `None` | e.g., "conv_light" |
| `chosen_tempo` | `str \| None` | `None` | "FAST", "STANDARD", "SLOW" |
| `chosen_style` | `str \| None` | `None` | "LINEAR_ADD", "GATED_GATE", etc. |
| `chosen_curve` | `str \| None` | `None` | "LINEAR", "COSINE", "SIGMOID" |
| `chosen_alpha_target` | `str \| None` | `None` | "HALF", "SEVENTY", "FULL" |
| `chosen_alpha_speed` | `str \| None` | `None` | "INSTANT", "FAST", "MEDIUM", "SLOW" |
| `op_confidence` | `float` | `0.0` | Probability of chosen operation |
| `slot_confidence` | `float` | `0.0` | Probability of chosen slot |
| `blueprint_confidence` | `float` | `0.0` | Probability of chosen blueprint |
| `style_confidence` | `float` | `0.0` | Probability of chosen style |
| `tempo_confidence` | `float` | `0.0` | Probability of chosen tempo |
| `alpha_target_confidence` | `float` | `0.0` | Probability of chosen alpha target |
| `alpha_speed_confidence` | `float` | `0.0` | Probability of chosen alpha speed |
| `curve_confidence` | `float` | `0.0` | Probability of chosen curve |
| `op_entropy` | `float` | `0.0` | Op head entropy |
| `slot_entropy` | `float` | `0.0` | Slot head entropy |
| `blueprint_entropy` | `float` | `0.0` | Blueprint head entropy |
| `style_entropy` | `float` | `0.0` | Style head entropy |
| `tempo_entropy` | `float` | `0.0` | Tempo head entropy |
| `alpha_target_entropy` | `float` | `0.0` | Alpha target head entropy |
| `alpha_speed_entropy` | `float` | `0.0` | Alpha speed head entropy |
| `curve_entropy` | `float` | `0.0` | Curve head entropy |

---

## RewardComponents

**Purpose:** Esper-specific reward signal breakdown. Documents ALL reward component keys.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `total` | `float` | `0.0` | Total reward |
| `base_acc_delta` | `float` | `0.0` | Legacy shaped signal based on accuracy improvement |
| `bounded_attribution` | `float` | `0.0` | Contribution-primary attribution (replaces seed_contribution) |
| `seed_contribution` | `float` | `0.0` | Seed contribution percentage (older format) |
| `compute_rent` | `float` | `0.0` | Cost of active seeds (always negative) |
| `alpha_shock` | `float` | `0.0` | Convex penalty on alpha deltas |
| `ratio_penalty` | `float` | `0.0` | Penalty for extreme policy ratios |
| `stage_bonus` | `float` | `0.0` | Bonus for reaching advanced stages (BLENDING+) |
| `fossilize_terminal_bonus` | `float` | `0.0` | Large terminal bonus for successful fossilization |
| `blending_warning` | `float` | `0.0` | Warning signal during blending phase |
| `holding_warning` | `float` | `0.0` | Warning signal during holding period |
| `hindsight_credit` | `float` | `0.0` | Retroactive credit when beneficiary fossilizes |
| `scaffold_count` | `int` | `0` | Number of scaffolds that contributed |
| `avg_scaffold_delay` | `float` | `0.0` | Average epochs since scaffolding |
| `env_id` | `int` | `0` | Environment ID |
| `val_acc` | `float` | `0.0` | Validation accuracy context (metadata) |
| `last_action` | `str` | `""` | Last action taken |

---

## ValueFunctionMetrics

**Purpose:** Value function quality diagnostics for DRL training health. Per DRL review: V-Return correlation is THE primary diagnostic.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `v_return_correlation` | `float` | `0.0` | Pearson correlation V(s) vs actual returns. <0.5 = value network not learning |
| `td_error_mean` | `float` | `0.0` | Mean TD error. High = biased value estimates |
| `td_error_std` | `float` | `0.0` | TD error std. High = noisy gradient targets |
| `bellman_error` | `float` | `0.0` | \|V(s) - (r + gamma*V(s'))\|^2. Spikes precede NaN losses |
| `return_p10` | `float` | `0.0` | 10th percentile of returns |
| `return_p50` | `float` | `0.0` | Median return (more robust than mean) |
| `return_p90` | `float` | `0.0` | 90th percentile of returns |
| `return_skewness` | `float` | `0.0` | >0 = right-skewed (few big wins) |
| `return_variance` | `float` | `0.0` | High = inconsistent policy |
| `value_predictions` | `deque[float]` | maxlen=100 | Historical value predictions |
| `actual_returns` | `deque[float]` | maxlen=100 | Historical actual returns |
| `td_errors` | `deque[float]` | maxlen=100 | Historical TD errors |

---

## GradientQualityMetrics

**Purpose:** Gradient quality diagnostics for DRL training health.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `gradient_cv` | `float` | `0.0` | Gradient Coefficient of Variation. <0.5 = high signal, >2.0 = noisy |
| `clip_fraction_positive` | `float` | `0.0` | r > 1+epsilon (probability increases capped) |
| `clip_fraction_negative` | `float` | `0.0` | r < 1-epsilon (probability decreases capped) |

---

## InfrastructureMetrics

**Purpose:** PyTorch infrastructure health metrics. Collected every N batches to amortize CPU-GPU sync overhead.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `cuda_memory_allocated_gb` | `float` | `0.0` | torch.cuda.memory_allocated() |
| `cuda_memory_reserved_gb` | `float` | `0.0` | torch.cuda.memory_reserved() |
| `cuda_memory_peak_gb` | `float` | `0.0` | torch.cuda.max_memory_allocated() |
| `cuda_memory_fragmentation` | `float` | `0.0` | 1 - (allocated/reserved), >0.3 = pressure |
| `compile_enabled` | `bool` | `False` | Whether torch.compile is enabled |
| `compile_backend` | `str` | `""` | "inductor", "eager", etc. |
| `compile_mode` | `str` | `""` | "default", "reduce-overhead", "max-autotune" |

**Property:**
- `memory_usage_percent` -> `float`: `(allocated / reserved) * 100`

---

## SystemVitals

**Purpose:** System resource metrics - GPU, RAM, throughput.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `gpu_stats` | `dict[int, GPUStats]` | `{}` | Per-GPU statistics |
| `gpu_memory_used_gb` | `float` | `0.0` | Convenience: gpu_stats[0] memory used |
| `gpu_memory_total_gb` | `float` | `0.0` | Convenience: gpu_stats[0] memory total |
| `gpu_utilization` | `float` | `0.0` | Convenience: gpu_stats[0] utilization |
| `gpu_temperature` | `float` | `0.0` | Convenience: gpu_stats[0] temperature |
| `cpu_percent` | `float \| None` | `0.0` | CPU usage percentage |
| `ram_used_gb` | `float \| None` | `0.0` | RAM used (GB) |
| `ram_total_gb` | `float \| None` | `0.0` | RAM total (GB) |
| `epochs_per_second` | `float` | `0.0` | Training throughput |
| `batches_per_hour` | `float` | `0.0` | Batch throughput |
| `host_params` | `int` | `0` | Host network parameter count |

**Properties:**
- `has_memory_alarm` -> `bool`: True if any device >90% memory usage
- `memory_alarm_devices` -> `list[str]`: Devices exceeding 90% usage

---

## GPUStats

**Purpose:** Per-GPU statistics for multi-GPU support.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `device_id` | `int` | `0` | CUDA device ID |
| `memory_used_gb` | `float` | `0.0` | Memory used (GB) |
| `memory_total_gb` | `float` | `0.0` | Memory total (GB) |
| `utilization` | `float` | `0.0` | GPU utilization (0-100) |
| `temperature` | `float` | `0.0` | GPU temperature (Celsius) |

---

## RunConfig

**Purpose:** Training run configuration captured at TRAINING_STARTED.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `n_episodes` | `int` | `0` | Total episodes to train |
| `lr` | `float` | `0.0` | Initial learning rate |
| `clip_ratio` | `float` | `0.2` | PPO clip ratio |
| `entropy_coef` | `float` | `0.01` | Initial entropy coefficient |
| `param_budget` | `int` | `0` | Seed parameter budget |
| `resume_path` | `str` | `""` | Checkpoint resume path |
| `entropy_anneal` | `dict[str, float]` | `{}` | Entropy schedule config |

---

## BestRunRecord

**Purpose:** Historical record of a best run for the leaderboard. Supports left-click (detail modal).

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `env_id` | `int` | required | Environment ID |
| `episode` | `int` | required | Batch/episode number (0-indexed) |
| `peak_accuracy` | `float` | required | Best accuracy during run |
| `final_accuracy` | `float` | required | Accuracy at batch end |
| `epoch` | `int` | `0` | Epoch when best achieved |
| `seeds` | `dict[str, SeedState]` | `{}` | Seeds at peak |
| `slot_ids` | `list[str]` | `[]` | All slot IDs |
| `growth_ratio` | `float` | `1.0` | Model size ratio |
| `record_id` | `str` | `""` | Unique ID for click targeting |
| `reward_components` | `RewardComponents \| None` | `None` | Reward breakdown at peak |
| `counterfactual_matrix` | `CounterfactualSnapshot \| None` | `None` | Counterfactual at peak |
| `shapley_snapshot` | `ShapleySnapshot \| None` | `None` | Shapley at peak |
| `action_history` | `list[str]` | `[]` | Recent actions at peak |
| `reward_history` | `list[float]` | `[]` | Reward history to peak |
| `accuracy_history` | `list[float]` | `[]` | Accuracy history to peak |
| `host_loss` | `float` | `0.0` | Host loss at peak |
| `host_params` | `int` | `0` | Host params at peak |
| `fossilized_count` | `int` | `0` | Fossilized count at peak |
| `pruned_count` | `int` | `0` | Pruned count at peak |
| `reward_mode` | `str \| None` | `None` | A/B cohort |
| `blueprint_spawns` | `dict[str, int]` | `{}` | Per-blueprint spawns at peak |
| `blueprint_fossilized` | `dict[str, int]` | `{}` | Per-blueprint fossilized at peak |
| `blueprint_prunes` | `dict[str, int]` | `{}` | Per-blueprint prunes at peak |

---

## EventLogEntry

**Purpose:** Single event log entry for Event Log panel. Message is GENERIC; specific values go in metadata.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `timestamp` | `str` | required | Formatted as HH:MM:SS |
| `event_type` | `str` | required | REWARD_COMPUTED, SEED_GERMINATED, etc. |
| `env_id` | `int \| None` | required | None for global events (PPO, BATCH) |
| `message` | `str` | required | Generic message (specifics in metadata) |
| `episode` | `int` | `0` | Episode number for grouping |
| `relative_time` | `str` | `""` | "(2s ago)" relative time string |
| `metadata` | `dict[str, str \| int \| float]` | `{}` | Structured data for detail view |

---

## SeedLifecycleStats

**Purpose:** Seed lifecycle aggregate metrics for TamiyoBrain display.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `germination_count` | `int` | `0` | Cumulative germinations |
| `prune_count` | `int` | `0` | Cumulative prunes |
| `fossilize_count` | `int` | `0` | Cumulative fossilizations |
| `active_count` | `int` | `0` | Current active count |
| `total_slots` | `int` | `0` | Total slots available |
| `germination_rate` | `float` | `0.0` | Germinations per episode |
| `prune_rate` | `float` | `0.0` | Prunes per episode |
| `fossilize_rate` | `float` | `0.0` | Fossilizations per episode |
| `blend_success_rate` | `float` | `0.0` | fossilized / (fossilized + pruned) |
| `avg_lifespan_epochs` | `float` | `0.0` | Mean epochs before terminal state |
| `germination_trend` | `str` | `"stable"` | "rising", "stable", "falling" |
| `prune_trend` | `str` | `"stable"` | "rising", "stable", "falling" |
| `fossilize_trend` | `str` | `"stable"` | "rising", "stable", "falling" |

---

## ObservationStats

**Purpose:** Observation space health metrics. Catches input distribution issues before NaN gradients.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `slot_features_mean` | `float` | `0.0` | Mean across slot features |
| `slot_features_std` | `float` | `0.0` | Std across slot features |
| `host_features_mean` | `float` | `0.0` | Mean across host features |
| `host_features_std` | `float` | `0.0` | Std across host features |
| `context_features_mean` | `float` | `0.0` | Mean across context features |
| `context_features_std` | `float` | `0.0` | Std across context features |
| `outlier_pct` | `float` | `0.0` | % of observations outside 3 sigma |
| `nan_count` | `int` | `0` | NaN values in observations |
| `inf_count` | `int` | `0` | Inf values in observations |
| `normalization_drift` | `float` | `0.0` | How much running mean/std has shifted |

---

## EpisodeStats

**Purpose:** Episode-level aggregate metrics. Helps diagnose timeout issues, early termination, and success rates.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `length_mean` | `float` | `0.0` | Mean episode length |
| `length_std` | `float` | `0.0` | Episode length std |
| `length_min` | `int` | `0` | Minimum episode length |
| `length_max` | `int` | `0` | Maximum episode length |
| `total_episodes` | `int` | `0` | Total episodes tracked |
| `timeout_count` | `int` | `0` | Episodes that hit max steps |
| `success_count` | `int` | `0` | Episodes that achieved goal |
| `early_termination_count` | `int` | `0` | Episodes terminated early |
| `timeout_rate` | `float` | `0.0` | timeout_count / total_episodes |
| `success_rate` | `float` | `0.0` | success_count / total_episodes |
| `early_termination_rate` | `float` | `0.0` | early_termination_count / total_episodes |
| `steps_per_germinate` | `float` | `0.0` | Avg steps between GERMINATE |
| `steps_per_prune` | `float` | `0.0` | Avg steps between PRUNE |
| `steps_per_fossilize` | `float` | `0.0` | Avg steps between FOSSILIZE |
| `completion_trend` | `str` | `"stable"` | "improving", "stable", "declining" |

---

## ShapleySnapshot

**Purpose:** Shapley value attribution for all slots. Computed via permutation sampling at episode boundaries.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `slot_ids` | `tuple[str, ...]` | `()` | e.g., ("r0c0", "r0c1", "r0c2") |
| `values` | `dict[str, ShapleyEstimate]` | `{}` | Per-slot Shapley estimates |
| `epoch` | `int` | `0` | Epoch when computed |
| `timestamp` | `datetime \| None` | `None` | Computation timestamp |

**Methods:**
- `get_mean(slot_id)` -> `float`: Get mean Shapley value for slot
- `get_significance(slot_id, z=1.96)` -> `bool`: Check if contribution is statistically significant (95% CI)
- `ranked_slots()` -> `list[tuple[str, float]]`: Slots ranked by mean contribution (descending)

---

## ShapleyEstimate

**Purpose:** Shapley value estimate for a single slot.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `mean` | `float` | `0.0` | Expected marginal contribution |
| `std` | `float` | `0.0` | Standard deviation (uncertainty) |
| `n_samples` | `int` | `0` | Number of permutation samples used |

---

## CounterfactualSnapshot

**Purpose:** Full factorial counterfactual matrix for an environment. Contains all 2^n configurations for n active seeds.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `slot_ids` | `tuple[str, ...]` | `()` | e.g., ("r0c0", "r0c1", "r0c2") |
| `configs` | `list[CounterfactualConfig]` | `[]` | All evaluated configurations |
| `strategy` | `str` | `"unavailable"` | "full_factorial" or "unavailable" |
| `compute_time_ms` | `float` | `0.0` | Computation time in milliseconds |

**Properties:**
- `baseline_accuracy` -> `float`: Accuracy with all seeds disabled
- `combined_accuracy` -> `float`: Accuracy with all seeds enabled

**Methods:**
- `get_accuracy(mask)` -> `float | None`: Get accuracy for specific seed configuration
- `individual_contributions()` -> `dict[str, float]`: Each seed's solo contribution over baseline
- `pair_contributions()` -> `dict[tuple[str, str], float]`: Each pair's contribution over baseline
- `total_synergy()` -> `float`: combined - baseline - sum(individual contributions)

---

## CounterfactualConfig

**Purpose:** Single configuration result from factorial evaluation.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `seed_mask` | `tuple[bool, ...]` | required | Which seeds are enabled |
| `accuracy` | `float` | `0.0` | Validation accuracy for this config |

---

## Utility Functions

### compute_entropy_velocity

```python
compute_entropy_velocity(entropy_history: deque[float] | list[float]) -> float
```

Compute rate of entropy change (d(entropy)/d(batch)). Uses numpy linear regression over last 10 samples. Returns velocity in entropy units per batch; negative = declining.

### compute_correlation

```python
compute_correlation(x_values, y_values) -> float
```

Compute Pearson correlation between two metric histories. Returns 0.0 if insufficient data or zero variance.

### compute_collapse_risk

```python
compute_collapse_risk(
    entropy_history,
    critical_threshold=0.3,
    warning_threshold=0.5,
    max_healthy_entropy=1.39,
    previous_risk=0.0,
    hysteresis=0.08
) -> float
```

Compute entropy collapse risk score (0.0 to 1.0). Based on proximity to critical threshold and velocity. Includes hysteresis to prevent flapping.

### detect_trend

```python
detect_trend(values, metric_name="default", metric_type="loss") -> str
```

Detect trend pattern with RL-appropriate thresholds. Returns: "improving", "stable", "volatile", "warning".

Uses 10-sample windows and metric-specific thresholds from `TREND_THRESHOLDS`.

### trend_to_indicator

```python
trend_to_indicator(trend: str) -> tuple[str, str]
```

Convert trend label to display indicator and Rich style. Returns (char, color).

### make_sparkline

```python
make_sparkline(values, width=8) -> str
```

Create a sparkline from values using block characters.

---

## Constants

### TREND_THRESHOLDS

Metric-specific thresholds for trend detection:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| `episode_return` | 15% | Returns vary naturally |
| `entropy` | 8% | Entropy changes are meaningful |
| `policy_loss` | 20% | Policy loss is noisy |
| `value_loss` | 20% | Value loss is noisy |
| `kl_divergence` | 30% | KL varies widely |
| `clip_fraction` | 30% | Clip fraction is variable |
| `grad_norm` | 25% | Gradients vary batch-to-batch |
| `expl_var` | 15% | Explained variance |
| `default` | 15% | Fallback |
