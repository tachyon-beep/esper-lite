# JANK Template

- **Title:** Heuristic Tamiyo confidence scores are static heuristics, not grounded in signals
- **Category:** observability / ergonomics
- **Symptoms:** Confidence in `TamiyoDecision` is set using simple ratios (e.g., plateau_epochs/5, improvement/5) and defaults to 1.0 elsewhere. This doesn’t reflect uncertainty from signals (variance, plateau trend) and can mislead consumers/telemetry that display confidence as meaningful.
- **Impact:** Medium – downstream analytics/Karn/TUI may misinterpret confidence; comparisons with RL policy confidence become apples-to-oranges.
- **Triggers:** Any heuristic decision; confidence rarely varies beyond the simple ratios.
- **Root-Cause Hypothesis:** Confidence was stubbed in for display; not tied to signal statistics.
- **Remediation Options:**
  - A) Derive confidence from normalized signal health (plateau trend, stabilization, variance).
  - B) Set confidence=None or a fixed value and document that heuristic confidence is not meaningful.
  - C) Add configurable confidence strategy to avoid misleading telemetry.
- **Validation Plan:** Add tests asserting confidence strategy is documented or computed from signals; ensure telemetry consumers handle None/flagged confidence.
- **Status:** Closed (Superseded)
- **Resolution:** Heuristic `TamiyoDecision.confidence` is not used for telemetry/dashboards; operator UIs consume PPO `action_confidence` (policy probability) rather than heuristic “confidence”. Treat the heuristic scalar as an internal display hint, not a probabilistic measure.
- **Links:** `src/esper/tamiyo/heuristic.py` (heuristic confidence assignments), `src/esper/simic/training/vectorized.py` (`action_confidence` emission), `src/esper/karn/sanctum/aggregator.py` (`action_confidence` ingestion)
