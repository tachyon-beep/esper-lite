# JANK Template

- **Title:** Probationary WAIT penalty dwarfs other reward components
- **Category:** correctness-risk / reward shaping ergonomics
- **Symptoms:** `probation_warning` in `src/esper/simic/rewards.py` applies an exponential penalty up to -10.0 for WAIT in PROBATIONARY, while other shaping terms are ~[-0.5, 0.5]. This dominates reward scale and can overshadow contribution/rent signals.
- **Impact:** Rewards become lopsided; policy may learn to avoid PROBATIONARY WAIT regardless of context, reducing sensitivity to actual contribution signals. Harder to tune entropy/clip because one term drives variance.
- **Triggers:** Seeds lingering in PROBATIONARY with counterfactual available (epochs_in_stage ≥ 2, bounded_attribution > 0); common in sparse reward or cautious policies.
- **Root-Cause Hypothesis:** Penalty was steepened per DRL review without rebalancing other components or renormalizing reward scale.
- **Remediation Options:** 
  - A) Reduce cap (e.g., -3.0) or soften base multiplier.
  - B) Normalize reward components to a common scale and log to telemetry.
  - C) Anneal penalty strength over episodes instead of per-epoch exponential.
- **Risks of Change:** Altering reward magnitudes changes learned policy; needs re-baselining and may affect prior checkpoint compatibility.
- **Stopgap Mitigation:** Add telemetry field for component magnitudes; adjust PPO reward normalization (RewardNormalizer) to counter spikes.
- **Validation Plan:** Inspect component magnitudes in a short PPO run; ensure penalties stay within target range and overall reward variance remains stable. Optionally add a unit test that clamps component magnitude.
- **Status:** Open
- **Links:** `src/esper/simic/rewards.py:530-575`, DRL specialist finding on reward scale asymmetry
