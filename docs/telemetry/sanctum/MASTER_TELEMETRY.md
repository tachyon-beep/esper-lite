# Sanctum TUI Telemetry Master Reference

> **Purpose:** Consolidated reference for all telemetry consumed by Sanctum TUI widgets.
> **Generated:** From comprehensive audit of 24 widget files.

---

## Table of Contents

1. [Widget Overview](#1-widget-overview)
2. [Schema Hierarchy](#2-schema-hierarchy)
3. [Field Usage Matrix](#3-field-usage-matrix)
4. [Threshold Reference](#4-threshold-reference)
5. [Color Coding Standards](#5-color-coding-standards)
6. [Data Flow](#6-data-flow)

---

## 1. Widget Overview

### 1.1 Global Status Widgets

| Widget | File | Purpose |
|--------|------|---------|
| **RunHeader** | `run_header.py` | Single-line status bar with connection, thread, progress, runtime, throughput, alarms |
| **AnomalyStrip** | `anomaly_strip.py` | Single-line anomaly surfacing: "ALL CLEAR" or count of issues by category |
| **EsperStatus** | `esper_status.py` | System vitals: seed stage counts, params, throughput, GPU/RAM/CPU usage |
| **EventLog** | `event_log.py` | Scrolling event log with drip-feed carousel (shows 1 event at a time) |
| **EventLogDetail** | `event_log_detail.py` | Modal: Raw event log entries in table format |
| **ThreadDeathModal** | `thread_death_modal.py` | Modal: Static notification when training thread dies |

### 1.2 Environment Monitoring Widgets

| Widget | File | Purpose |
|--------|------|---------|
| **EnvOverview** | `env_overview.py` | Per-environment table with accuracy, rewards, slots, status |
| **EnvDetailScreen** | `env_detail_screen.py` | Modal: Full environment diagnostics with seed cards, metrics, history |
| **HistoricalEnvDetail** | `historical_env_detail.py` | Modal: Frozen env state from Best Runs leaderboard |
| **Scoreboard** | `scoreboard.py` | Two-panel leaderboard: Best Runs (top 5) and Worst Trajectory (bottom 5) |

### 1.3 Tamiyo Brain Panel Widgets (Policy Agent)

| Widget | File | Purpose |
|--------|------|---------|
| **NarrativePanel** | `tamiyo/narrative_panel.py` | Now/Why/Next guidance + one-line health summary (replaces StatusBanner) |
| **SlotsPanel** | `tamiyo/slots_panel.py` | Slot stage distribution bars + lifecycle aggregate statistics |
| **HealthStatusPanel** | `tamiyo/health_status_panel.py` | Comprehensive health: advantage, gradients, KL, entropy, value range |
| **PPOLossesPanel** | `tamiyo/ppo_losses_panel.py` | PPO gauges (EV, entropy, clip) + loss sparklines with trends |
| **ActionHeadsPanel** | `tamiyo/action_heads_panel.py` | Per-head entropy/gradient/ratio grids + decision heatmap carousel |
| **ActionContext** | `tamiyo/action_distribution.py` | Consolidated decision context (critic, reward, returns, actions, sequence) |
| **EpisodeMetricsPanel** | `tamiyo/episode_metrics_panel.py` | Episode health: warmup baseline or training outcomes/trends |
| **ValueDiagnosticsPanel** | `tamiyo/value_diagnostics_panel.py` | Value function diagnostics: GAE, TD errors, explained variance |
| **DecisionsColumn** | `tamiyo/decisions_column.py` | Vertical stack of decision cards with throttled updates (30s swap) |
| **DecisionDetailScreen** | `tamiyo/decision_detail_screen.py` | Modal: Full decision details with all head choices |

### 1.4 Attribution Widgets

| Widget | File | Purpose |
|--------|------|---------|
| **RewardHealth** | `reward_health.py` | Reward component breakdown with PBRS fraction indicator |
| **CounterfactualPanel** | `counterfactual_panel.py` | Factorial attribution matrix with synergy/interference indicators |
| **ShapleyPanel** | `shapley_panel.py` | Shapley value attribution for seed contributions |

---

## 2. Schema Hierarchy

### 2.1 Primary Data Flow

```
SanctumSnapshot (root)
├── connected: bool                    # WebSocket connection status
├── staleness_seconds: float           # Time since last update
├── training_thread_alive: bool | None # Thread health
├── task_name: str                     # Run identifier
├── current_episode: int               # Episode counter
├── current_epoch: int                 # Epoch counter
├── max_epochs: int                    # Epoch limit (0 = unbounded)
├── current_batch: int                 # Batch counter
├── max_batches: int                   # Batches per epoch
├── runtime_seconds: float             # Elapsed time
├── total_slots: int                   # Slot count
├── slot_ids: list[str]                # Slot identifiers (e.g., "r0c0")
├── slot_stage_counts: dict[str, int]  # Stage distribution
├── cumulative_fossilized: int         # Total fossilizations
├── cumulative_pruned: int             # Total prunes
├── last_action_env_id: int | None     # Last action target
├── last_action_timestamp: datetime | None
├── aggregate_mean_accuracy: float     # Cross-env accuracy
├── aggregate_mean_reward: float       # Cross-env reward
│
├── envs: dict[int, EnvState]          # Per-environment state
├── tamiyo: TamiyoState                # Policy agent state
├── vitals: SystemVitals               # System metrics
├── seed_lifecycle: SeedLifecycleStats # Aggregate lifecycle
├── observation_stats: ObservationStats # Observation health
├── event_log: list[EventLogEntry]     # Event history
└── best_runs: list[BestRunRecord]     # Leaderboard entries
```

### 2.2 EnvState (Per-Environment)

```
EnvState
├── env_id: int
├── status: str                        # "initializing"|"healthy"|"stalled"|"degraded"|"excellent"
├── reward_mode: str | None            # A/B test cohort
├── rolled_back: bool                  # Rollback indicator
├── rollback_reason: str
├── last_update: datetime | None
│
├── host_accuracy: float               # Current accuracy
├── best_accuracy: float               # Peak accuracy
├── accuracy_history: deque[float]     # Last 50 values
├── epochs_since_improvement: int      # Stall counter
│
├── current_reward: float              # Latest reward
├── mean_reward: float                 # Rolling mean
├── cumulative_reward: float           # Episode total
├── reward_history: deque[float]       # Last 50 values
├── reward_components: RewardComponents # Reward breakdown
│
├── host_loss: float                   # Training loss
├── host_params: int                   # Parameter count
├── fossilized_params: int             # Fossilized params
├── growth_ratio: float                # Size ratio
│
├── seeds: dict[str, SeedState]        # Per-slot seed state
├── action_history: deque[str]         # Last 10 actions
├── counterfactual_matrix: CounterfactualSnapshot | None
└── shapley_snapshot: ShapleySnapshot | None
```

### 2.3 TamiyoState (Policy Agent)

```
TamiyoState
├── ppo_data_received: bool            # Gate for metrics display
├── group_id: str | None               # A/B test group
│
├── # PPO Core Metrics
├── entropy: float
├── explained_variance: float
├── clip_fraction: float
├── policy_loss: float
├── value_loss: float
├── kl_divergence: float
├── grad_norm: float
├── advantage_mean: float
├── advantage_std: float
├── advantage_skewness: float
├── advantage_kurtosis: float
├── advantage_positive_ratio: float
│
├── # History Deques (for sparklines/trends)
├── entropy_history: deque[float]
├── policy_loss_history: deque[float]
├── value_loss_history: deque[float]
├── kl_divergence_history: deque[float]
├── grad_norm_history: deque[float]
├── episode_return_history: deque[float]
│
├── # Entropy Collapse Tracking
├── entropy_velocity: float            # d(entropy)/d(batch)
├── collapse_risk_score: float         # 0.0-1.0
├── entropy_clip_correlation: float
├── entropy_collapsed: bool
│
├── # Per-Head Metrics (8 heads each)
├── head_*_entropy: float              # op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve
├── head_*_grad_norm: float
├── head_*_grad_norm_prev: float
├── head_*_ratio_max: float
├── head_nan_latch: dict[str, bool]
├── head_inf_latch: dict[str, bool]
│
├── # Gradient Health
├── dead_layers: int
├── exploding_layers: int
├── nan_grad_count: int
├── inf_grad_count: int
├── gradient_quality: GradientQualityMetrics
│
├── # Value Function
├── value_mean: float
├── value_std: float
├── value_min: float
├── value_max: float
├── initial_value_spread: float | None
├── log_prob_min: float
├── log_prob_max: float
│
├── # Joint Ratio
├── joint_ratio_max: float
│
├── infrastructure: InfrastructureMetrics
└── recent_decisions: list[DecisionSnapshot]
```

### 2.4 SeedState (Per-Slot)

```
SeedState
├── slot_id: str                       # Position identifier
├── stage: str                         # DORMANT|GERMINATED|TRAINING|HOLDING|BLENDING|FOSSILIZED|PRUNED
├── blueprint_id: str | None           # Architecture name
├── alpha: float                       # Blend ratio (0.0-1.0)
├── alpha_curve: str                   # LINEAR|COSINE|SIGMOID*
├── epochs_in_stage: int               # Stage duration
├── blend_tempo_epochs: int            # Blend speed (3=fast, 5=std, >5=slow)
├── has_vanishing: bool                # Gradient health
├── has_exploding: bool                # Gradient health
└── accuracy_contribution: float       # Attribution
```

### 2.5 Supporting Dataclasses

| Dataclass | Purpose | Key Fields |
|-----------|---------|------------|
| **SystemVitals** | Hardware metrics | `host_params`, `epochs_per_second`, `batches_per_hour`, `gpu_stats`, `ram_*`, `cpu_percent` |
| **GPUStats** | Per-GPU metrics | `memory_used_gb`, `memory_total_gb`, `utilization` |
| **RewardComponents** | Reward breakdown | `total`, `base_acc_delta`, `compute_rent`, `stage_bonus`, `bounded_attribution`, `hindsight_credit` |
| **SeedLifecycleStats** | Aggregate lifecycle | `active_count`, `fossilize_count`, `prune_count`, `germination_rate`, `blend_success_rate` |
| **ObservationStats** | Obs health | `nan_count`, `inf_count`, `outlier_pct`, `normalization_drift` |
| **GradientQualityMetrics** | Gradient quality | `gradient_cv`, `clip_fraction_positive`, `clip_fraction_negative` |
| **InfrastructureMetrics** | Infra state | `compile_enabled`, `memory_usage_percent` |
| **EventLogEntry** | Log event | `timestamp`, `event_type`, `env_id`, `episode`, `message`, `metadata` |
| **DecisionSnapshot** | Decision record | All head choices + confidences + entropies + slot_states + alternatives |
| **BestRunRecord** | Leaderboard entry | `peak_accuracy`, `final_accuracy`, `seeds`, `reward_components`, history arrays |
| **CounterfactualSnapshot** | Attribution | `strategy`, `slot_ids`, `raw_scores`, `total_synergy()` method |
| **ShapleySnapshot** | Attribution | `values`, `uncertainties`, `meta` |
| **EpisodeStats** | Episode metrics | `length_*`, `timeout_rate`, `success_rate`, `early_termination_rate`, `completion_trend` |

---

## 3. Field Usage Matrix

### 3.1 SanctumSnapshot Root Fields

| Field | RunHeader | AnomalyStrip | EsperStatus | EventLog | EnvOverview | Scoreboard | NarrativePanel | SlotsPanel |
|-------|:---------:|:------------:|:-----------:|:--------:|:-----------:|:----------:|:------------:|:----------:|
| `connected` | ✓ | | | | | | | |
| `staleness_seconds` | ✓ | | | | | | | |
| `training_thread_alive` | ✓ | | | | | | | |
| `task_name` | ✓ | | | | | | | |
| `current_episode` | ✓ | | | | | | | |
| `current_epoch` | ✓ | | | | | | | |
| `current_batch` | ✓ | | | | | | ✓ | |
| `runtime_seconds` | ✓ | | ✓ | | | | | |
| `total_slots` | | | | | | | | ✓ |
| `slot_ids` | | | | | ✓ | | | |
| `slot_stage_counts` | | | | | | | | ✓ |
| `envs` | | ✓ | ✓ | | ✓ | ✓ | | ✓ |
| `best_runs` | | | | | | ✓ | | |
| `event_log` | | | | ✓ | | | | |
| `aggregate_mean_accuracy` | | | | | ✓ | | | |
| `cumulative_fossilized` | | | | | | ✓ | | |
| `cumulative_pruned` | | | | | | ✓ | | |

### 3.2 TamiyoState Fields

| Field | NarrativePanel | PPOLossesPanel | HealthStatusPanel | ActionHeadsPanel | DecisionsColumn |
|-------|:------------:|:--------------:|:-----------------:|:----------------:|:---------------:|
| `ppo_data_received` | ✓ | | | | |
| `group_id` | ✓ | | | | ✓ |
| `entropy` | ✓ | ✓ | ✓ | | |
| `explained_variance` | ✓ | ✓ | | | |
| `clip_fraction` | ✓ | ✓ | ✓ | | |
| `policy_loss` | | ✓ | | | |
| `value_loss` | | ✓ | | | |
| `kl_divergence` | ✓ | | ✓ | | |
| `grad_norm` | ✓ | | ✓ | | |
| `advantage_mean/std` | ✓ | | ✓ | | |
| `entropy_velocity` | | ✓ | ✓ | | |
| `collapse_risk_score` | | ✓ | ✓ | | |
| `nan_grad_count` | ✓ | | | ✓ | |
| `inf_grad_count` | ✓ | | | ✓ | |
| `dead_layers` | | | | ✓ | |
| `exploding_layers` | | | | ✓ | |
| `head_*_entropy` | | | | ✓ | |
| `head_*_grad_norm` | | | | ✓ | |
| `head_*_ratio_max` | | | | ✓ | |
| `head_nan_latch` | | | | ✓ | |
| `head_inf_latch` | | | | ✓ | |
| `recent_decisions` | | | | ✓ | ✓ |
| `infrastructure.compile_enabled` | ✓ | | | | |
| `infrastructure.memory_usage_percent` | ✓ | | | | |

### 3.3 EnvState Fields

| Field | EnvOverview | EnvDetailScreen | HistoricalEnvDetail | AnomalyStrip | EsperStatus |
|-------|:-----------:|:---------------:|:-------------------:|:------------:|:-----------:|
| `env_id` | ✓ | ✓ | ✓ | | |
| `status` | ✓ | ✓ | | ✓ | |
| `host_accuracy` | ✓ | ✓ | | | |
| `best_accuracy` | ✓ | ✓ | | | ✓ |
| `accuracy_history` | ✓ | ✓ | ✓ | | |
| `epochs_since_improvement` | ✓ | | | | |
| `current_reward` | ✓ | ✓ | | | |
| `cumulative_reward` | ✓ | | | | |
| `reward_history` | ✓ | ✓ | ✓ | | |
| `reward_components` | ✓ | ✓ | ✓ | | |
| `host_loss` | ✓ | ✓ | | | |
| `growth_ratio` | ✓ | ✓ | ✓ | | |
| `seeds` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `action_history` | ✓ | ✓ | ✓ | | |
| `counterfactual_matrix` | ✓ | ✓ | ✓ | | |

### 3.4 SeedState Fields

| Field | EnvOverview | EnvDetailScreen | SlotsPanel | EsperStatus | ActionHeadsPanel |
|-------|:-----------:|:---------------:|:----------:|:-----------:|:----------------:|
| `stage` | ✓ | ✓ | ✓ | ✓ | |
| `blueprint_id` | ✓ | ✓ | | | |
| `alpha` | ✓ | ✓ | | | |
| `alpha_curve` | ✓ | ✓ | | | |
| `epochs_in_stage` | ✓ | ✓ | | | |
| `blend_tempo_epochs` | ✓ | ✓ | | | |
| `has_vanishing` | ✓ | ✓ | | | |
| `has_exploding` | ✓ | ✓ | | | |

---

## 4. Threshold Reference

### 4.1 PPO Health Thresholds (from `TUIThresholds`)

| Metric | Warning | Critical | Source |
|--------|---------|----------|--------|
| **Entropy** | < 0.3 | < 0.1 | leyline `DEFAULT_ENTROPY_*_THRESHOLD` |
| **Explained Variance** | < 0.3 | <= 0.0 | karn.constants |
| **Clip Fraction** | > 0.25 | > 0.30 | karn.constants |
| **KL Divergence** | > 0.015 | > 0.03 | karn.constants |
| **Gradient Norm** | > 5.0 | > 10.0 | karn.constants |

### 4.2 Advantage Statistics Thresholds

| Metric | Low Warning | OK Range | Warning | Critical |
|--------|-------------|----------|---------|----------|
| **advantage_std** | < 0.5 | 0.5-2.0 | > 2.0 | < 0.1 or > 3.0 |
| **advantage_skewness** | < -0.5 | -0.5 to 1.0 | > 1.0 | < -1.0 or > 2.0 |
| **advantage_kurtosis** | < -1.0 | -1.0 to 3.0 | > 3.0 | < -2.0 or > 6.0 |
| **advantage_positive_ratio** | < 0.4 | 0.4-0.6 | > 0.6 | < 0.2 or > 0.8 |

### 4.3 Memory Thresholds

| Resource | Green | Yellow | Red |
|----------|-------|--------|-----|
| **GPU Memory** | < 75% | 75-90% | > 90% |
| **RAM** | < 75% | 75-90% | > 90% |
| **GPU Utilization** | < 80% | 80-95% | > 95% |
| **CPU** | < 90% | - | > 90% (alarm) |

### 4.4 Environment Health Thresholds

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| **Host Loss** | < 0.1 | 0.5-1.0 | >= 1.0 |
| **Growth Ratio** | 1.0-2.0 | 2.0-5.0 | >= 5.0 |
| **Cumulative Reward** | > 0 | - | < -5 |
| **Current Reward** | > 0 | - | < -0.5 |
| **Epochs Since Improvement** | 0 | 6-15 | > 15 |

### 4.5 Scoreboard Thresholds

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| **Epoch (peak timing)** | < 25 | 50-64 | >= 65 |
| **Trajectory Delta** | > +0.5% | -2% to -5% | < -5% |
| **Growth Ratio** | 1.0-1.1 | - | >= 1.1 (bold) |

### 4.6 ActionHeadsPanel Thresholds

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| **Normalized Entropy** | > 0.5 | 0.25-0.5 | < 0.25 |
| **Gradient Norm** | 0.1-2.0 | 0.01-0.1 or 2.0-5.0 | < 0.01 or > 5.0 |
| **PPO Ratio** | 0.8-1.2 | 0.5-1.5 | < 0.5 or > 1.5 |
| **Gradient CV** | < 0.5 | 0.5-2.0 | >= 2.0 |

### 4.7 Observation Health Thresholds

| Metric | OK | Warning | Critical |
|--------|-----|---------|----------|
| **Outlier %** | < 5% | 5-10% | > 10% |
| **Normalization Drift** | < 1.0 sigma | 1.0-2.0 sigma | > 2.0 sigma |
| **NaN/Inf Count** | 0 | - | > 0 |

### 4.8 Staleness Thresholds

| Condition | Age | Style |
|-----------|-----|-------|
| **LIVE** | < 2.0s | green |
| **SLOW** | 2.0-5.0s | yellow |
| **STALE** | >= 5.0s | red |
| **BAD** | None or > 5.0s | red |

### 4.9 Blend Success Rate Thresholds

| Rate | Style | Meaning |
|------|-------|---------|
| >= 70% | green | Excellent |
| 50-70% | yellow | Moderate |
| < 50% | red | Poor |

---

## 5. Color Coding Standards

### 5.1 Seed Stage Colors (from leyline `STAGE_COLORS`)

| Stage | Color | Abbreviation |
|-------|-------|--------------|
| DORMANT | `dim` | Dorm |
| GERMINATED | `bright_blue` | Germ |
| TRAINING | `cyan` | Train |
| HOLDING | `magenta` | Hold |
| BLENDING | `yellow` | Blend |
| FOSSILIZED | `green` | Foss |
| PRUNED | `red` | Prune |
| EMBARGOED | `bright_red` | Embar |
| RESETTING | `dim` | Reset |

### 5.2 Action Colors

| Action | Color |
|--------|-------|
| GERMINATE | `green` |
| SET_ALPHA_TARGET | `cyan` |
| FOSSILIZE | `blue` |
| PRUNE | `red` |
| WAIT | `dim` |
| ADVANCE | `cyan` |

### 5.3 Status Colors

| Status | Style | Icon |
|--------|-------|------|
| excellent | `bold green` | star |
| healthy | `green` | filled circle |
| initializing | `dim` | half circle |
| stalled | `yellow` | half circle |
| degraded | `red` | empty circle |

### 5.4 A/B Cohort Colors

| Cohort | Color |
|--------|-------|
| A | `cyan` |
| B | `magenta` |
| shaped | `bright_blue` |
| simplified | `bright_yellow` |
| sparse | `bright_white` |

### 5.5 Trend Arrows

| Trend | Arrow | Good Context | Bad Context |
|-------|-------|--------------|-------------|
| Improving | `↗` | green | - |
| Stable | `→` | dim | dim |
| Declining | `↘` | - | red |
| Volatile | `~` | yellow | yellow |

### 5.6 Alpha Curve Glyphs (from leyline `ALPHA_CURVE_GLYPHS`)

| Curve | Glyph |
|-------|-------|
| LINEAR | `/` |
| COSINE | `~` |
| SIGMOID_GENTLE | `)` |
| SIGMOID | `)` |
| SIGMOID_SHARP | `D` |

### 5.7 Tempo Indicators

| Tempo | Epochs | Arrows |
|-------|--------|--------|
| FAST | <= 3 | `>>>` |
| STANDARD | <= 5 | `>>` |
| SLOW | > 5 | `>` |

---

## 6. Data Flow

### 6.1 Aggregator Pipeline

```
Training Process (simic)
    │
    ├── PPO Updates ──────────────────────────────────────────────┐
    ├── Episode Completions ──────────────────────────────────────┤
    ├── Seed Lifecycle Events ────────────────────────────────────┤
    ├── Decision Snapshots ───────────────────────────────────────┤
    │                                                              │
    ▼                                                              ▼
TelemetryEmitter ───────────────────────────────────────────► Sanctum Aggregator
                     (WebSocket/Queue)                              │
                                                                    ▼
                                                            SanctumSnapshot
                                                                    │
                    ┌───────────────────────────────────────────────┼───────────────────────────────────────────────┐
                    ▼                     ▼                         ▼                     ▼                         ▼
              RunHeader           EnvOverview              TamiyoBrainPanel          Scoreboard               EventLog
                                      │                         │                         │
                           ┌──────────┴──────────┐    ┌─────────┴─────────┐               │
                           ▼                     ▼    ▼                   ▼               ▼
                    EnvDetailScreen      AnomalyStrip    NarrativePanel  ActionHeadsPanel  EventLogDetail
                                                         PPOLossesPanel  DecisionsColumn
                                                         HealthStatusPanel  SlotsPanel
```

### 6.2 Update Frequency

| Widget Type | Update Trigger | Throttling |
|-------------|----------------|------------|
| **RunHeader** | Every snapshot | None |
| **EnvOverview** | Every snapshot | Row refresh only on change |
| **TamiyoBrain panels** | Every snapshot | Border title updates |
| **Scoreboard** | Every snapshot | State preservation on refresh |
| **DecisionsColumn** | Every snapshot | 30s swap interval |
| **ActionHeadsPanel** | Every snapshot | 5s decision swap interval |
| **EventLog** | Event arrival | 3.3s drip-feed carousel |
| **Modals** | On open | Static snapshot |

### 6.3 Carousel/Drip-Feed Mechanisms

| Widget | Max Items | Swap Interval | Age Tracking |
|--------|-----------|---------------|--------------|
| **DecisionsColumn** | 4 cards | 30s | Display timestamp |
| **ActionHeadsPanel** | 6 decisions | 5s | Display timestamp |
| **EventLog** | 1 visible | 3.3s (autoplay) | Relative time |

---

## Appendix: Individual Audit Documents

For detailed per-widget audits, see:

| Widget | Audit File |
|--------|------------|
| RunHeader | `run_header.md` |
| AnomalyStrip | `anomaly_strip.md` |
| EsperStatus | `esper_status.md` |
| EnvOverview | `env_overview.md` |
| EnvDetailScreen | `env_detail_screen.md` |
| HistoricalEnvDetail | `historical_env_detail.md` |
| Scoreboard | `scoreboard.md` |
| NarrativePanel | `narrative_panel.md` |
| SlotsPanel | `slots_panel.md` |
| HealthStatusPanel | `health_status_panel.md` |
| PPOLossesPanel | `ppo_losses_panel.md` |
| ActionHeadsPanel | `action_heads_panel.md` |
| ActionContext | `action_context.md` |
| EpisodeMetricsPanel | `episode_metrics_panel.md` |
| ValueDiagnosticsPanel | `value_diagnostics_panel.md` |
| DecisionsColumn | `decisions_column.md` |
| DecisionDetailScreen | `decision_detail_screen.md` |
| RewardHealth | `reward_health.md` |
| CounterfactualPanel | `counterfactual_panel.md` |
| ShapleyPanel | `shapley_panel.md` |
| EventLog | `event_log.md` |
| EventLogDetail | `event_log_detail.md` |
| ThreadDeathModal | `thread_death_modal.md` |
| Schema Reference | `schema_reference.md` |
