# JANK Template

- **Title:** PPO approx KL early-stop lacks telemetry/logging, making it invisible
- **Category:** observability / maintainability
- **Symptoms:** `PPOAgent.update` computes `approx_kl` and can set `early_stopped`, but never logs or emits telemetry when KL exceeds `target_kl`. With `recurrent_n_epochs=1` the stop is a no-op (BUG-003), and even when multi-epoch is used there’s no visibility into stop events.
- **Impact:** Medium – KL guardrails can silently disengage or never trigger, leaving policy updates unchecked and making tuning/diagnosis difficult. Users can’t tell whether KL thresholds are active.
- **Triggers:** PPO training with `target_kl` set; default recurrent_n_epochs=1 masks the stop, and logging is absent for any path.
- **Root-Cause Hypothesis:** Early-stop logic added without accompanying telemetry; default epochs hide the effect.
- **Remediation Options:**
  - A) Emit telemetry/analytics snapshot fields when early_stop triggers, including approx_kl and epoch_i; add a warning log.
  - B) Surface a counter in metrics so dashboards/TUI can display stop frequency.
  - Coordinate with BUG-003 fix to ensure stops actually occur when configured.
- **Validation Plan:** Unit/integration test forcing low target_kl to trigger stop; assert telemetry/log entry appears and update loop breaks as expected with recurrent_n_epochs>1.
- **Status:** Open
- **Links:** `src/esper/simic/ppo.py` (approx_kl, early_stopped handling), BUG-003
