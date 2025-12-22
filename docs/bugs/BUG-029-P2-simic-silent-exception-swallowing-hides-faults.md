# BUG-029: Broad exception swallowing hides counterfactual/health faults and yields partial analytics

- **Title:** Multiple `except Exception: pass/continue` blocks silently suppress failures in counterfactual evaluation and training telemetry collection
- **Context:** Simic vectorized training + counterfactual attribution; failures become invisible and results can be silently partial
- **Impact:** P2 – debugging/observability correctness risk. Training can proceed with degraded/partial analytics (counterfactual matrices, health warnings, gradient-health telemetry) with no signal that data is missing or wrong.
- **Environment:** HEAD @ workspace; any run computing counterfactual matrices or collecting debug telemetry
- **Reproduction Steps:**
  1. In Python, call `CounterfactualEngine.compute_matrix(...)` with an `evaluate_fn` that raises for some configs.
  2. Observe the failing configs are skipped with no warning/telemetry, producing a partial `CounterfactualMatrix` (and downstream Shapley estimates computed from incomplete data).
  3. In vectorized PPO, force `HealthMonitor._check_memory_and_warn(...)` or `collect_per_layer_gradients(...)` to raise (e.g., via monkeypatch) and observe training continues with no indication that monitoring/telemetry collection failed.
- **Expected Behavior:** Failures should be surfaced at least once per env/epoch (throttled) via logging or telemetry markers (e.g., `ANALYTICS_SNAPSHOT` with an error counter), so missing analytics can be detected and correlated.
- **Observed Behavior:** Exceptions are swallowed and evaluation/telemetry silently degrades:
  - Counterfactual evaluation: `except Exception: continue  # Skip failed evaluations`
  - Health monitor: `except Exception: pass  # Non-critical monitoring`
  - Gradient health collection: `except Exception: pass  # Graceful degradation`
- **Logs/Telemetry:** None; failures are currently invisible.
- **Hypotheses:** “Don’t crash training” robustness was implemented without a corresponding error signal, violating the “sensors match capabilities” principle in `ROADMAP.md`.
- **Fix Plan:** Replace silent catches with:
  - Throttled warning logs (once per error type per env per N steps), and/or
  - A lightweight telemetry marker containing `env_id`, exception type, and a counter for dropped counterfactual configs / failed collectors.
  - For counterfactual evaluation: optionally record `failed_configs` count on `CounterfactualMatrix` so downstream consumers can detect partial data.
- **Validation Plan:**
  - Unit test: an `evaluate_fn` that raises increments a failure counter and emits a warning marker (or logs once), and Shapley computation is skipped or annotated when the matrix is too sparse.
  - Smoke PPO run with forced collector failure to confirm the training loop continues but emits a visible warning event.
- **Status:** Open
- **Links:**
  - Counterfactual swallow: `src/esper/simic/attribution/counterfactual.py:244`
  - Health monitor swallow: `src/esper/simic/training/vectorized.py:1671`
  - Gradient telemetry swallow: `src/esper/simic/training/vectorized.py:2985`

