# Observation Deep Dive (Working Notes)

Scope: what the Simic PPO agent feeds into Tamiyo’s policy network (“Obs V3”), where the numbers come from, and what transforms/normalization are applied before the LSTM sees them.

This is a scratchpad. The cleaned summary is in `docs/scratch/obs_deep_dive/FINAL_REPORT.md`.

## 1) Primary data contracts (sources)

Obs V3 is assembled from three sources per environment:

- **TrainingSignals** (`src/esper/leyline/signals.py`, built by `SignalTracker.update()` in `src/esper/tamiyo/tracker.py`)
  - Used fields (Obs V3 base): `metrics.epoch`, `metrics.val_loss`, `metrics.val_accuracy`, plus `loss_history`, `accuracy_history`.
- **SeedStateReport** per slot (`src/esper/leyline/reports.py`, produced by Kasmina via `SeedState.to_report()` in `src/esper/kasmina/slot.py`, surfaced as `MorphogeneticModel.get_slot_reports()` in `src/esper/kasmina/host.py`)
  - Used fields (Obs V3 per-slot): `stage`, alpha schedule scaffolding (`alpha_*`), `blend_tempo_epochs`, `metrics.*`, plus `telemetry.*` when present.
- **ParallelEnvState** (`src/esper/simic/training/parallel_env_state.py`)
  - Used fields (Obs V3): `last_action_success`, `last_action_op`, plus per-slot trackers `gradient_health_prev` and `epochs_since_counterfactual`.

Blueprint identity is intentionally *not* one-hot in the observation tensor; it is passed separately as `blueprint_indices` and embedded inside the network.

## 2) When Obs V3 is captured (per-epoch “sensory” loop)

In the vectorized PPO loop (`src/esper/simic/training/vectorized.py`), for each epoch step:

1. **Tolaria executes train + validation** for each env, with fused counterfactual configs.
2. **Kasmina per-seed telemetry** is synced (`seed_state.sync_telemetry(...)`) and **counterfactual contribution** is updated (solo ablation path).
3. **TrainingSignals** are updated via `env_state.signal_tracker.update(...)`.
4. **Slot reports** collected via `model.get_slot_reports()`.
5. **Action masks** built from slot states (`compute_action_masks(...)`).
6. **Obs V3 features** built (batched) via `batch_obs_to_features(...)`.
7. **ObservationStatsTelemetry** is computed on raw (unnormalized) Obs V3 when ops telemetry is enabled.
8. **RunningMeanStd normalization** is applied (`obs_normalizer.normalize(...)`) to produce the actual policy input.
9. **Tamiyo policy** samples actions (`agent.policy.get_action(...)`) and transitions are stored into the rollout buffer.

## 3) Obs V3 tensor layout (the actual numbers)

### 3.1 Output shapes

From `batch_obs_to_features(...)` (`src/esper/tamiyo/policy/features.py`):

- `obs`: `float32[n_envs, obs_dim]`
  - `obs_dim = 23 + 30 * num_slots` (default 3 slots → 113).
- `blueprint_indices`: `int64[n_envs, num_slots]`
  - `-1` means slot inactive (mapped to a trainable null embedding in the network).

The network concatenates `obs` with learned blueprint embeddings:

- `BlueprintEmbedding` (`src/esper/tamiyo/networks/factored_lstm.py`) maps indices `[-1, 0..12] → R^4`.
- Total network input (default): `113 + 3*4 = 125`.

### 3.2 Base features (23 dims)

All base features live at fixed indices `0..22`:

- `0` `epoch_norm = epoch / max_epochs`
- `1` `val_loss_symlog = symlog(val_loss) / 7`
- `2` `val_acc_norm = val_accuracy / 100`
- `3..7` `loss_history_5 = symlog(loss_history[i]) / 7` (left-padded with `0.0`)
- `8..12` `acc_history_5 = acc_history[i] / 100` (left-padded with `0.0`)
- `13..15` stage distribution: `num_training/num_slots`, `num_blending/num_slots`, `num_holding_or_fossilized/num_slots`
- `16` `last_action_success` (0/1)
- `17..22` `last_action_op_one_hot` (6 dims, `NUM_OPS`)

### 3.3 Per-slot features (30 dims × num_slots)

For slot index `s` with offset:

- `slot_offset = 23 + 30*s`

Fields:

- `+0` `is_active` (0 if slot empty, 1 if seed present)
- `+1..+10` stage one-hot (`NUM_STAGES=10`, via `StageSchema` / `STAGE_TO_INDEX`)
- `+11` `current_alpha` (0..1)
- `+12` `contribution_norm = clamp(contribution / 10, -1, 1)`
  - `contribution = counterfactual_contribution` when available, else `improvement_since_stage_start`
- `+13` `contribution_velocity_norm = clamp(contribution_velocity / 10, -1, 1)`
- `+14` `blend_tempo_norm = blend_tempo_epochs / 12` (not clamped)
- `+15..+22` alpha schedule scaffolding (8 dims):
  - `alpha_target` (0..1)
  - `alpha_mode_norm = alpha_mode / max(AlphaMode)`
  - `alpha_steps_total_norm = min(alpha_steps_total, max_epochs) / max_epochs`
  - `alpha_steps_done_norm = min(alpha_steps_done, max_epochs) / max_epochs`
  - `time_to_target_norm = min(time_to_target, max_epochs) / max_epochs`
  - `alpha_velocity = clamp(alpha_velocity, -1, 1)`
  - `alpha_algorithm_norm = (alpha_algorithm - min) / (max-min)`
  - `interaction_sum_norm = min(interaction_sum/10, 1)` (**upper-clamped only**)
- `+23..+26` merged per-seed telemetry (4 dims, if `report.telemetry` present):
  - `gradient_norm_symlog = symlog(gradient_norm) / 7` else `0.0`
  - `gradient_health = safe(gradient_health, default=1.0, max_val=1.0)`
  - `has_vanishing` (0/1)
  - `has_exploding` (0/1)
- `+27` `gradient_health_prev` (from `ParallelEnvState`, default 1.0 if missing)
- `+28` `epochs_in_stage_norm = min(epochs_in_current_stage, max_epochs) / max_epochs`
- `+29` `counterfactual_fresh = DEFAULT_GAMMA ** epochs_since_counterfactual`

Inactive slots remain all-zero in `obs` and keep `blueprint_indices=-1`.

## 4) Preprocessing and normalization

### 4.1 Local transforms (feature-wise)

Obs V3 uses “baked in” transforms intended to bound values before the global normalizer:

- `symlog(x)/7` for potentially large magnitudes (`val_loss`, loss history, `gradient_norm`).
- `x/100` for accuracies.
- `x/max_epochs` for timing-like scalars.
- `clamp(x/10, -1, 1)` for contribution-like scalars (percentage points).

### 4.2 Global observation normalization (RunningMeanStd)

- `obs_normalizer = RunningMeanStd((state_dim,), momentum=0.99)` in `vectorized.py`.
- During rollout collection, `states_batch_normalized = obs_normalizer.normalize(states_batch)` and the **normalized** states are what go into the rollout buffer.
- Raw (unnormalized) `states_batch` are accumulated and used to update the normalizer once per batch (in `_run_ppo_updates()`).
- `normalize()` applies `(x - mean) / sqrt(var + eps)` with `eps=1e-4`, then clamps to `[-10, 10]`.

Implication: PPO updates operate on states normalized with the statistics that were active during rollout collection (consistent actor/critic inputs), while the normalizer update affects the *next* rollout’s normalization.

## 5) Observation telemetry (“Obs Health”)

`compute_observation_stats()` in `src/esper/simic/telemetry/observation_stats.py` is computed on **raw** Obs V3 (`states_batch`) when ops telemetry is enabled.

It reports:

- Mean/std over all features (currently reused for slot/host/context groups).
- `outlier_pct`: fraction of elements with per-feature `|z| > 3` (computed across the batch).
- `nan_count` / `inf_count`
- `normalization_drift`: mean absolute change in running mean from epoch-0 snapshot.

Caveat: one-hot and sparse binary features can legitimately register as “outliers” when the rare value (`1`) appears in a mostly-zero batch. Treat `outlier_pct` as an anomaly-ish indicator, not a universal “damage” gauge.

## 6) Potential correctness risks / open questions

- **`interaction_sum_norm` is not lower-clamped.** `interaction_sum` can be negative (pair interaction term `I_ij`), so this feature can leave the intended `[-1, 1]` band. This is likely harmless under `RunningMeanStd` + clipping, but it can make the raw observation telemetry look alarming and can distort normalizer statistics if interactions are occasionally large.
- **Normaliser update semantics:** the current design stores already-normalized states in the buffer, so updating the normaliser “before PPO” does not retroactively change batch-N’s training inputs; it changes the next rollout’s inputs.
- **“Per-group” obs stats are placeholders:** `slot_features_*`, `host_features_*`, `context_features_*` are currently all populated with overall mean/std.

## 7) Key files (quick links)

- Obs V3 feature extraction: `src/esper/tamiyo/policy/features.py`
- Obs V3 normalization: `src/esper/simic/control/normalization.py`
- Vectorized collection loop + timing: `src/esper/simic/training/vectorized.py`
- TrainingSignals contract: `src/esper/leyline/signals.py`, `src/esper/tamiyo/tracker.py`
- Seed reports contract: `src/esper/leyline/reports.py`, `src/esper/kasmina/slot.py`, `src/esper/kasmina/host.py`
- Stage schema (one-hot mapping): `src/esper/leyline/stage_schema.py`
- Blueprint embedding: `src/esper/tamiyo/networks/factored_lstm.py`
- Observation stats telemetry: `src/esper/simic/telemetry/observation_stats.py`

