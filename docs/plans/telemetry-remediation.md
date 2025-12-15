# Telemetry Remediation Plan

## Master To-Add Checklist
- Emit unwired event types: `MEMORY_WARNING`, `REWARD_HACKING_SUSPECTED`, `COMMAND_*`, `ISOLATION_VIOLATION`, `PERFORMANCE_DEGRADATION`.
- Add warning/event when telemetry is disabled; consider lifecycle-only minimal mode (slot events even in fast runs).
- Surface reward breakdowns at ops-normal (summary fields or sampled debug).
- Integrate DiagnosticTracker optionally into PPO/heuristic cadence and emit snapshots.
- Enforce env_id injection for lifecycle events (helper/assertion in BlueprintAnalytics path).
- Emit counterfactual-unavailable markers; optionally lightweight per-slot ablation at final epoch.
- Add governor periodic health/memory snapshots and `GOVERNOR_ROLLBACK` env_id/device context.
- Reintroduce action distribution summaries in analytics snapshots.
- Per-slot additions: include alpha in lifecycle events; add inner_epoch + global epoch; emit gate-evaluated pass/fail with reasons; include gradient health/vanish/explode, isolation_violations, seed_gradient_norm_ratio.
- Per-env additions: emit full last-action detail (slot, blueprint, blend, masked?, success?); add step time/fps/dataloader-wait telemetry.
- Policy pulse: ensure `PPO_UPDATE_COMPLETED` carries lr, grad_norm/surrogate, update_time_ms; add per-head/mask hit rates for op/slot/blueprint/blend.

## Triage (Priority / Difficulty)

### Quick, High-Value
- Warn/event when telemetry disabled; consider minimal lifecycle-only mode. *(infra)*
- Enforce `env_id` on lifecycle events; add env_id/device to `GOVERNOR_ROLLBACK`. *(analytics/governor)*
- Add alpha + epoch (inner + global) fields to lifecycle events; include full last-action details (slot/blueprint/blend/masked?/success?). *(slots/policy)*
- Add lr, grad_norm/surrogate, update_time_ms to `PPO_UPDATE_COMPLETED`; reintroduce action distribution summary. *(policy pulse)*
- Emit counterfactual-unavailable markers; small per-slot baseline at final epoch if cheap. *(counterfactual)*
- Add step time/fps/dataloader wait per env to detect stragglers. *(throughput)*

### Medium Effort / Medium Value
- Emit gate-evaluated events with pass/fail reasons. *(slots)*
- Add per-slot health fields: gradient health/vanish/explode, isolation_violations, seed_gradient_norm_ratio. *(slots)*
- Reward breakdown at ops-normal (summarized or sampled). *(rewards)*
- Optional DiagnosticTracker cadence + snapshot emission hooks. *(diagnostics)*
- Per-head/mask hit rates for op/slot/blueprint/blend. *(policy pulse)*

### Heavier / Marginal
- Wire `MEMORY_WARNING` via allocator stats and `PERFORMANCE_DEGRADATION` heuristic; add governor periodic health/memory snapshots. *(governor/health)*
- Reward hacking detector (`REWARD_HACKING_SUSPECTED`) from attribution/improvement ratio spikes. *(rewards)*
- Command events (`COMMAND_*`) in CLI orchestration. *(infra)*
- Lightweight per-slot ablation at final epoch to guarantee counterfactual coverage (compute-heavy on large models). *(counterfactual)*

### Logical Groupings
- **Slots/Lifecycle:** alpha/epoch fields, gate events, per-slot health, env_id enforcement, action details.
- **Policy/Rewards:** reward breakdown visibility, PPO vitals (lr/grad_norm/time), mask hit rates, action distribution, reward hacking detector.
- **Counterfactuals:** unavailable markers, per-slot baseline/ablation.
- **Health/Throughput:** governor context + snapshots, memory/perf warnings, step time/fps/wait.
- **Infra/Backends:** telemetry-disable warning, command events, analytics env_id helper, DiagnosticTracker hooks.

Targeting gaps noted in the telemetry audit. Each item includes intent and proposed action.

## Unwired Event Types
- **Events:** `MEMORY_WARNING`, `REWARD_HACKING_SUSPECTED`, `COMMAND_*`, `ISOLATION_VIOLATION`, `PERFORMANCE_DEGRADATION`.
- **Action:** Identify producers and wire emissions:
  - Memory: hook into `torch.cuda.memory_stats` or allocator callbacks in PPO loop and governor.
  - Reward hacking: flag extreme `seed_contribution/total_improvement` ratios in `rewards.py` and emit.
  - Commands: emit on CLI command start/finish/fail in `scripts/train.py`.
  - Isolation: emit from `GradientIsolationMonitor` when violations increment.
  - Performance degradation: emit on rolling accuracy drops (reuse `DEGRADATION_DETECTED` thresholds).

## Telemetry Suppression
- **Issue:** `use_telemetry=False` or `fast_mode=True` disables slot lifecycle events and SeedTelemetry (dropped from PPO state).
- **Action:** Add explicit warning/event when telemetry is disabled; consider minimal lifecycle-only mode so critical events still emit even in fast runs.

## Reward Breakdown Visibility
- **Issue:** `REWARD_COMPUTED` breakdown only at DEBUG level.
- **Action:** Add ops-normal summary (e.g., bounded_attribution + rent + total) or configurable debug-once-per-N-batches sampling.

## DiagnosticTracker Integration
- **Issue:** Heavy diagnostics not emitted anywhere by default.
- **Action:** Provide optional hook in PPO/heuristic loops to run DiagnosticTracker on a cadence (e.g., every K epochs) and emit `ANALYTICS_SNAPSHOT` with `EpochSnapshot.to_dict()` when enabled.

## Blueprint Analytics Env Attribution
- **Issue:** New lifecycle emitters must set `env_id` or analytics mis-attributes.
- **Action:** Add assertion/helper to inject `env_id` before hub emission; document requirement in `BlueprintAnalytics`.

## Counterfactual Coverage
- **Issue:** Counterfactual telemetry only when baselines exist; missing baselines fall back to proxy with no event.
- **Action:** Emit “counterfactual unavailable” metric when baseline missing; optionally add lightweight per-slot ablation in final epoch to guarantee coverage.

## Governor Observability
- **Issue:** Only emits on rollback; no periodic health/memory telemetry.
- **Action:** Add periodic `GOVERNOR_SNAPSHOT`/`PERFORMANCE_DEGRADATION` with loss stats and optional memory usage; emit `MEMORY_WARNING` when thresholds exceed profile.

## Action Distribution Reporting
- **Issue:** Action distribution removed from vectorized telemetry.
- **Action:** Reintroduce as a low-cost summary in `ANALYTICS_SNAPSHOT` or every N batches to aid policy debugging.

## UI Truthfulness Gaps (Overwatch layout)
- **Per-slot chips/inspector**
  - **Alpha missing:** Add `alpha` to lifecycle events (`SEED_STAGE_CHANGED`, `SEED_FOSSILIZED`, `SEED_CULLED`) or emit periodic slot snapshots.
  - **Epoch context:** Include both inner_epoch and monotonic batch/global epoch on lifecycle events so slot age/stage timelines are accurate.
  - **Gate visibility:** Emit “gate evaluated” events with gate id, pass/fail, and reasons for failures.
  - **Health fields:** Add gradient health/vanish/explode flags, isolation_violations, seed_gradient_norm_ratio to per-slot telemetry payloads.
- **Per-env flight board**
  - **Last action detail:** Emit sampled slot, blueprint, blend id, mask-hit/masked status, and action_success for each step.
  - **Throughput:** Emit per-env step time/fps/dataloader wait to diagnose stalls/stragglers.
- **Governor**
  - **Context:** Include env_id/device in `GOVERNOR_ROLLBACK` (and future snapshots) for attribution.
- **Policy pulse**
  - **Vitals:** Ensure `PPO_UPDATE_COMPLETED` always carries lr, grad_norm (or surrogate), update_time_ms; add per-head/mask hit rates for op/slot/blueprint/blend.
