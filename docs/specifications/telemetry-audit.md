# Telemetry Metrics Audit

Reference of what telemetry we capture, how it is produced, and where it flows. Covers runtime emissions (PPO/heuristic), lifecycle events, analytics, and optional diagnostics.

## Backends and Configuration
- **Hub/backends:** `nissa.output.NissaHub` fan-outs to Console/File/Directory/TUI/Dashboard/Karn backends wired in `scripts/train.py`. Severity filter follows CLI telemetry level (debug shows all, normal hides debug).
- **Profiles (Nissa):** `nissa/config.py` + `profiles.yaml` control heavy diagnostics (gradient stats, per-class, loss landscape, weight norms). Default `standard` enables gradient stats for select layers.
- **Levels (Simic):** `simic/telemetry_config.py` levels OFF/MINIMAL/NORMAL/DEBUG with auto-escalation on anomalies. Slot telemetry uses `use_telemetry` flag and level; `fast_mode` disables slot events entirely.
- **Contracts:** Event types in `leyline/telemetry.py`; RL observation schema in `leyline/signals.py` (35-dim base + SeedTelemetry when enabled).

## Seed Lifecycle Telemetry (Kasmina)
- **Emitter:** `kasmina/slot.py` via `on_telemetry` callback (skipped when `fast_mode=True`).
- **Events:**
  - `SEED_GERMINATED` (blueprint_id, seed_id, params)
  - `SEED_STAGE_CHANGED` (from, to)
  - `SEED_FOSSILIZED` (blueprint_id, seed_id, improvement, blending_delta, counterfactual, params_added, epochs_total, epochs_in_stage)
  - `SEED_CULLED` (reason, blueprint_id, seed_id, improvement, blending_delta, counterfactual, epochs_total, epochs_in_stage)
- **SeedTelemetry:** 10-dim normalized snapshot (grad_norm/health, vanish/explode flags, accuracy/Δ, epochs_in_stage, stage, alpha, epoch/max_epochs) kept on `SeedState` and updated via `sync_telemetry`.

## PPO Vectorized Telemetry (Simic)
- **Setup:** `vectorized.py` adds `BlueprintAnalytics` backend and emits `TRAINING_STARTED` (devices, task, reward mode, dataloader config, budgets).
- **Lifecycle/Counterfactual:** Slot events from Kasmina (with injected `env_id`), `COUNTERFACTUAL_COMPUTED` per slot when baselines available (real_acc, baseline_acc, Δ).
- **Batch cadence:**
  - `BATCH_EPOCH_COMPLETED` (batch_idx, episodes_completed/total, env_accuracies, avg/rolling acc, avg reward, train/val loss/acc, n_envs, skipped_update, plateau_detected, inner_epoch).
  - `EPOCH_COMPLETED` (per-env) with val loss/acc and seed telemetry (env_id scoped).
  - Progress markers: `PLATEAU_DETECTED` / `DEGRADATION_DETECTED` / `IMPROVEMENT_DETECTED` based on rolling_avg_accuracy deltas vs thresholds.
- **PPO updates:** `PPO_UPDATE_COMPLETED` per batch epoch: policy/value loss, entropy + coef, KL, clip_fraction, ratio_max/min/std, explained_variance, batch + inner_epoch ids, avg/rolling acc/reward, entropy coef. Skips with reason when governor rollback clears buffer.
- **Rewards:** When telemetry level DEBUG, `REWARD_COMPUTED` carries full `RewardComponentsTelemetry` (bounded_attribution, compute_rent, stage/pbrs bonuses, blending/probation penalties, terminal bonuses, action_success, seed_stage, val_acc, baselines, growth_ratio). Governor punishment also emits `REWARD_COMPUTED` with reason.
- **Anomalies:** `RATIO_EXPLOSION/COLLAPSE`, `VALUE_COLLAPSE`, `NUMERICAL_INSTABILITY`, fallback `GRADIENT_ANOMALY`. Optional payloads include per-layer grad stats and numerical stability report when debug gradients enabled.
- **Governor:** `GOVERNOR_ROLLBACK` (env_id, reason, loss_at_panic, threshold, consecutive_panics) and injects negative reward; snapshots every 5 epochs (no event) and panic detection uses loss stats.
- **Analytics sync:** `ANALYTICS_SNAPSHOT` every batch (rolling accuracy, entropy/KL/EV, seeds created/fossilized, skipped flag) plus summary/scoreboard strings every 5 batches when `quiet_analytics` is False.
- **Checkpointing:** `CHECKPOINT_LOADED` (resume/best state) and `CHECKPOINT_SAVED` (path, avg_accuracy).
- **Telemetry gating:** Slot telemetry and reward components are disabled when `use_telemetry` is False or level below ops-normal/debug respectively; SeedTelemetry features are omitted from state vector when `use_telemetry` is False.

## Heuristic Telemetry (Simic training.py)
- `TRAINING_STARTED` at run start.
- Per-epoch `EPOCH_COMPLETED` (train/val loss/acc, available_slots, seeds_active, action/op chosen, reward, seed targets, episode_id/mode).
- Episode summaries: `ANALYTICS_SNAPSHOT` with config at start and per-episode completion (final accuracy, total_reward, action_counts).
- Uses same slot telemetry callbacks; `env_id` fixed to 0.

## Decision Signals (Tamiyo)
- `SignalTracker.update` computes `TrainingSignals` (loss/acc deltas, plateau count, host_stabilized latch, histories, bests, seed summary). When stabilization latch trips, emits `TAMIYO_INITIATED` (env_id, epoch, stable_count, stabilization_epochs, val_loss). Signals feed PPO observations and heuristic policy; no other events.

## Governor (Tolaria)
- `TolariaGovernor.execute_rollback` emits `GOVERNOR_ROLLBACK` (reason, loss_at_panic, loss_threshold, consecutive_panics) and culls live seeds before restoring snapshot. No periodic events; vital signs check and snapshot are internal.

## Blueprint Analytics Backend (Nissa)
- **Source:** Receives lifecycle events.
- **Stats tracked:** Per-blueprint germinated/fossilized/culled counts; mean acc_delta, blending_delta, counterfactual, churn; fossilization rate. Per-env scoreboard: params added, compute_cost (multiplier table), fossilize/cull age, distribution by blueprint, params% of host.
- **Outputs:** `summary_table()` and `scoreboard_table()` strings included in periodic `ANALYTICS_SNAPSHOT`; snapshot dict attached to final history entry in PPO.

## Diagnostic Tracker (Nissa)
- **Opt-in:** Not auto-wired; used manually for rich per-epoch diagnostics.
- **Metrics:** Gradient per-layer stats (norm/std/mean/percentiles, vanish/explode pct), `GradientHealth` aggregate, per-class acc/loss variance, loss noise, sharpness estimate (perturbation), weight norms, narrative + red_flags/opportunities.
- **Events:** None by default—callers must emit their own `TelemetryEvent` with `EpochSnapshot.to_dict()` if they want it persisted.
- **Performance:** Expensive; profiles control scope. Hooks require `cleanup()` to remove.

## Unused/Defined-but-Unwired Events
- Defined but not emitted: `MEMORY_WARNING`, `REWARD_HACKING_SUSPECTED`, command events (`COMMAND_*`), `ISOLATION_VIOLATION`, `PERFORMANCE_DEGRADATION`. Hook points exist in telemetry contracts but no producers currently wire them.

## Notable Behaviors / Correctness Notes
- Slot telemetry is skipped when `fast_mode=True` or `use_telemetry=False`; PPO state vector drops SeedTelemetry in that case.
- Reward breakdowns only surface at DEBUG level; ops-normal runs see only aggregate reward.
- Counterfactual events originate from (a) per-slot baseline in vectorized loop (final epoch) and (b) optional Karn helper (multi-slot matrix). Missing baselines yield `seed_contribution=None` and proxy signals in reward computation.
- BlueprintAnalytics expects `env_id` in lifecycle events; vectorized path injects it via callback, heuristic fixes it to 0—other emitters must include it for correct aggregation.
