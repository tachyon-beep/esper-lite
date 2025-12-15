# BUG Template

- **Title:** Counterfactual validation uses mutation-based `force_alpha`, unsafe for multi-thread/stream eval
- **Context:** Tolaria `validate_with_attribution` uses `seed_slot.force_alpha(0.0)` (mutation) to compute baseline, sharing state with concurrent operations.
- **Impact:** P1 – in multi-env/stream or future DDP settings, mutating slot state for baseline pass can corrupt in-flight forwards or compiled graphs, producing incorrect attribution or crashes.
- **Environment:** Main branch; GPU runs with multiple streams/envs or any concurrent validation.
- **Reproduction Steps:** Run validation concurrently with another forward (e.g., in vectorized setup) while counterfactual runs; observe alpha leakage or mismatched outputs.
- **Expected Behavior:** Counterfactual path should be functional (pass override to forward) or isolated per-call without mutating shared slot state.
- **Observed Behavior:** `force_alpha` toggles shared state; docstring warns not thread-safe.
- **Hypotheses:** Convenience use of `force_alpha`; aligns with Kasmina TODO on thread safety.
- **Fix Plan:** Refactor counterfactual validation to use functional alpha override (no shared mutation) or single-threaded isolation guard; coordinate with Kasmina force_alpha fix.
- **Validation Plan:** Add test ensuring baseline computation doesn’t affect concurrent forwards; ensure counterfactual results match functional override.
- **Status:** Open
- **Links:** `src/esper/tolaria/trainer.py::validate_with_attribution`, `src/esper/kasmina/slot.py::force_alpha`
