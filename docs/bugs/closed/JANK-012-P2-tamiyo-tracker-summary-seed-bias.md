# JANK Template

- **Title:** SignalTracker summary seed selection biases toward highest stage, ignoring contribution
- **Category:** correctness-risk / observability
- **Symptoms:** `SignalTracker.update` picks a “summary seed” for metrics using a key (stage, alpha, -counterfactual, seed_id). This can overrepresent a late-stage but low-contribution seed, suppressing signals from a training-stage seed that’s actually improving. With multi-slot, summary metrics (seed_stage/alpha/improvement) may not reflect the most relevant seed.
- **Impact:** Medium – heuristic and downstream telemetry consumers may react to misleading summary signals, affecting decisions and dashboards.
- **Triggers:** Multiple active seeds with differing stages/alphas; e.g., a PROBATIONARY seed with low/negative contribution and a TRAINING seed with strong improvement.
- **Root-Cause Hypothesis:** The summary key was tuned for single-slot and safety (highest stage + alpha), not for fairness across seeds in multi-slot settings.
- **Remediation Options:**
  - A) Make summary selection configurable (e.g., choose seed with highest counterfactual contribution or highest improvement).
  - B) Emit per-slot summary metrics instead of a single summary seed, or include both highest-stage and highest-contribution identifiers.
  - C) At minimum, document the bias and ensure Simic/heuristic consumers use per-slot metrics.
- **Validation Plan:** Add tests with multiple seeds to ensure summary selection matches the chosen policy; verify heuristics/telemetry aren’t misled in multi-slot scenarios.
- **Status:** Closed (By design)
- **Resolution:** The summary seed selection is deterministic and explicitly documented in `SignalTracker` (favoring highest stage/alpha with a safety tie-break). Current controllers operate on per-seed/per-slot state (`active_seeds` / `SeedState` lists and slot reports), so the single-summary `TrainingSignals.seed_*` fields are informational only; consumers needing per-slot fidelity should use per-slot metrics rather than the summary seed.
- **Links:** `src/esper/tamiyo/tracker.py` (summary seed selection), `src/esper/tamiyo/heuristic.py` (per-seed decisions)
