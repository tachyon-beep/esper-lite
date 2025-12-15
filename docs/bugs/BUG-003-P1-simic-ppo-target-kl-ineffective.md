# BUG Template

- **Title:** PPO `target_kl` early-stop is effectively disabled with default settings
- **Context:** Simic / `PPOAgent.update` (`src/esper/simic/ppo.py`)
- **Impact:** P1 – `target_kl` is advertised/defaulted (0.015) but does nothing for the default `recurrent_n_epochs=1`; oversize policy updates are never curtailed, allowing runaway updates or ratio explosions without backoff.
- **Environment:** Main branch; PPO runs with default recurrent settings (n_epochs=1 for recurrent); GPU/CPU.
- **Reproduction Steps:**
  1. Run `PYTHONPATH=src uv run python -m esper.scripts.train ppo --episodes 1 --n-envs 1 --max-epochs 1 --target-kl 0.0001`.
  2. Observe updates proceed even when `approx_kl` exceeds the threshold; no early exit or retry occurs because the loop has a single epoch.
- **Expected Behavior:** When `approx_kl` crosses the threshold, the update should halt the current optimization step (or skip remaining epochs) and surface the stop in metrics; `target_kl` should be a functioning safety valve.
- **Observed Behavior:** `approx_kl` is computed and `early_stopped` is set, but with `recurrent_n_epochs=1` there is nothing to stop; the flag is only consulted at the top of the next epoch, so `target_kl` is a no-op in default configs (`src/esper/simic/ppo.py` around the approx_kl block).
- **Logs/Telemetry:** `approx_kl` shows large values; no indication of stop beyond a metrics field.
- **Hypotheses:** Early-stop logic was ported from multi-epoch PPO; recurrent default of 1 epoch neuters the feature, leaving advertised safety unfulfilled.
- **Fix Plan:** Either (a) move the KL check earlier to break out of the current epoch immediately, or (b) apply gradient/ratio guardrails when the threshold is exceeded; ensure a metric/telemetry flag indicates a stop.
- **Validation Plan:** Force `target_kl` low and confirm update exits early (no optimizer step after threshold) and metrics flag the stop; add unit test toggling recurrent_n_epochs=1 and >1 to cover both paths.
- **Status:** Open
- **Links:** `src/esper/simic/ppo.py` approx_kl block (~520-560), DRL expert finding “KL stopping disabled”
