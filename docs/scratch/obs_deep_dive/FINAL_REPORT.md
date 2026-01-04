# Observation Deep Dive (Final Report)

Goal: document how Simic constructs the Obs V3 “sensory surface” that feeds Tamiyo’s policy network, what preprocessing happens, and what looks correct vs risky.

## Executive Summary

- Obs V3 is a **113‑dim float vector** (default 3 slots) plus **per-slot blueprint indices** that are embedded inside the network (total input 125 dims).
- Most scalars are already bounded or compressed before global normalization (symlog, `[0,1]` scaling, or `[-1,1]` clamps), then a `RunningMeanStd` normaliser (EMA, clip) produces stable inputs for the LSTM.
- The “large numbers” humans see in the UI are usually **instrumentation semantics**, not “the LSTM going feral”: raw L2 norms and raw outlier counters scale with tensor size and feature sparsity.
- The primary correctness risk in Obs V3 today is one feature: **`interaction_sum_norm` is only upper-clamped**, even though the interaction term can be negative; this can push raw observations outside the intended range.

## What the agent actually sees (Obs V3)

### Shapes

Obs V3 is produced by `batch_obs_to_features()` (`src/esper/tamiyo/policy/features.py`):

- `obs`: `float32[n_envs, 23 + 30*num_slots]` (default: 113)
- `blueprint_indices`: `int64[n_envs, num_slots]` (`-1` for inactive slots)

The network uses `BlueprintEmbedding` (`src/esper/tamiyo/networks/factored_lstm.py`) to convert `blueprint_indices` into learned vectors (`R^4` per slot) and concatenates them with `obs` before the feature net and LSTM. `BlueprintEmbedding` maps exactly `-1` to a dedicated trainable “null” embedding for inactive slots and (when validation is enabled) raises if it receives invalid indices outside `-1` or `[0, num_blueprints)`.

### Content (high-level)

Obs V3 has:

- **23 base features**: episode progress, host validation loss/accuracy + short history, stage counts, and last-action feedback.
- **30 per-slot features** per enabled slot: stage one-hot, alpha schedule state, counterfactual contribution signals, gradient telemetry, and a small amount of “staleness memory”.

Full index-by-index mapping is in `docs/scratch/obs_deep_dive/WORKING_NOTES.md`.

## How observations are collected (Kasmina → Simic → Tamiyo)

In `src/esper/simic/training/vectorized.py`, each epoch step per environment:

1. Tolaria runs training and the fused validation pass (including solo/pair ablation configs when enabled).
2. During validation result processing, solo ablation updates `SeedMetrics.counterfactual_contribution` and resets the Obs V3 staleness tracker (`epochs_since_counterfactual`).
3. Kasmina syncs per-seed telemetry (`seed_state.sync_telemetry(...)`) so per-slot reports reflect the current epoch’s metrics (and gradient stats when available).
4. `SignalTracker.update()` builds `TrainingSignals` (Leyline contract) from the epoch metrics and short histories.
5. Kasmina exports per-slot `SeedStateReport`s (`model.get_slot_reports()`).
6. Simic builds action masks and then calls `batch_obs_to_features(...)` to produce Obs V3.

This means Obs V3 always reflects the latest validated epoch metrics and latest per-seed telemetry snapshot.

## Preprocessing and normalization

### Feature-wise transforms (before global normalization)

Obs V3 applies local transforms to keep values in sane bands before the global normalizer:

- `symlog(x) / 7` for potentially high-magnitude scalars (validation loss, loss history, seed gradient norm).
- `x/100` for accuracies.
- `x/max_epochs` for time-like scalars (epoch progress, schedule steps, epochs-in-stage).
- `clamp(x/10, -1, 1)` for contribution-like scalars (percentage points).

This protects the LSTM early in training (before `RunningMeanStd` converges) and reduces sensitivity to rare spikes.

### Global normalization (RunningMeanStd)

The policy does not consume raw Obs V3; it consumes:

`obs_normalizer.normalize(obs_v3)` where:

- `RunningMeanStd` uses EMA (`momentum=0.99`) for long-run stability.
- Normalized values are clipped to `[-10, 10]`.
- Rollout buffer stores the **normalized** observations used during action sampling.
- The normalizer updates once per batch using the raw Obs V3 collected during rollout.

This keeps actor/critic inputs consistent within a batch while still adapting slowly as the policy (and therefore the observation distribution) evolves.

## Assessment: what looks correct vs what is risky

### Looks correct / appropriate

- **Value scale management:** symlog + bounded scaling + global normalization is a robust combination for PPO with an LSTM.
- **Blueprint embeddings:** avoids a sparse blueprint one-hot explosion and gives the network a compact trainable representation.
- **Stage one-hot via Leyline schema:** avoids invalid stage values and keeps categorical encoding stable.
- **Counterfactual “freshness” (`gamma ** age`)** is a sensible way to tell the policy when a contribution number might be stale without hiding it.

### Primary correctness risk: `interaction_sum_norm`

`interaction_sum` is computed from pair counterfactuals (`I_ij = f({i,j}) - f({i}) - f({j}) + f(empty)`), and can be negative or positive.

Obs V3 currently computes:

- `interaction_sum_norm = min(interaction_sum/10, 1)`

That upper-clamps only. A sufficiently negative interaction can produce values less than `-1`, violating the “bounded feature” intent and potentially destabilizing raw observation telemetry (even if `RunningMeanStd` + clipping prevents catastrophic effects).

Recommended fix (once you’re ready to change behaviour): symmetric clamp to `[-1, 1]` or switch to a symlog-style compression for this specific scalar.

### Secondary “interpretation” risks (telemetry/UI)

- **Outlier percent can flag sparse one-hots:** a rare `1` in a mostly-zero feature produces high z-scores; treat outliers as anomaly telemetry, not a universal danger gauge.
- **“Group stats” are coarse aggregates:** observation stats compute separate host/context/slot mean/std, but each is averaged over heterogeneous feature groups; use these as broad drift signals, not a substitute for per-feature inspection.

## Suggested next validations (no-code)

If you want to sanity-check Obs V3 without changing behaviour:

- Confirm `interaction_sum` typical magnitudes (it should usually be small, single-digit percentage points).
- Compare `obs_normalizer.mean/var` drift across runs with different `n_envs` to ensure the normalizer isn’t being dominated by inactive-slot zeros.
- If “Obs Health” is being used as a status color, base it on non-finite counts + saturation/near-clip indicators rather than raw outlier counts alone.
