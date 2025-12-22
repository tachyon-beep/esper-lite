# JANK Template

- **Title:** PPO approx KL early-stop lacks telemetry/logging, making it invisible
- **Category:** observability / maintainability
- **Symptoms:** `PPOAgent.update` computes `approx_kl` and can early-stop when KL exceeds `target_kl`; the concern was lack of visibility into stop events and the `recurrent_n_epochs=1` no-op behavior.
- **Impact:** Medium – KL guardrails can silently disengage or never trigger, leaving policy updates unchecked and making tuning/diagnosis difficult. Users can’t tell whether KL thresholds are active.
- **Triggers:** PPO training with `target_kl` set; default recurrent_n_epochs=1 masks the stop, and logging is absent for any path.
- **Root-Cause Hypothesis:** Early-stop logic originally lacked telemetry/metrics emission; default epochs hid the effect.
- **Remediation Options:**
  - A) Emit telemetry/analytics snapshot fields when early_stop triggers, including approx_kl and epoch_i; add a warning log.
  - B) Surface a counter in metrics so dashboards/TUI can display stop frequency.
  - Coordinate with BUG-003 fix to ensure stops actually occur when configured.
- **Validation Plan:** Unit/integration test forcing low target_kl to trigger stop; assert telemetry/log entry appears and update loop breaks as expected with recurrent_n_epochs>1.
- **Status:** Closed (Resolved)
- **Resolution:** `approx_kl` is recorded into PPO metrics and emitted via Simic telemetry (`kl_divergence`), and early-stops record `early_stop_epoch`. The KL check now happens before the optimizer step, making the stop effective even with `recurrent_n_epochs=1`, and vectorized training breaks out of multi-update loops when the threshold is exceeded.
- **Links:** `src/esper/simic/agent/ppo.py` (KL + early stop), `src/esper/simic/telemetry/emitters.py` (`kl_divergence`, `early_stop_epoch`), `src/esper/simic/training/vectorized.py` (`_run_ppo_updates`)
