# JANK Template

- **Title:** Holding WAIT penalty dwarfs other reward components
- **Category:** correctness-risk / reward shaping ergonomics
- **Symptoms:** `holding_warning` in `src/esper/simic/rewards/rewards.py` applies an exponential penalty up to -10.0 for WAIT in HOLDING. The concern is that this can dominate per-step reward scale and overshadow contribution/rent signals.
- **Impact:** Rewards become lopsided; policy may learn to avoid HOLDING WAIT regardless of context, reducing sensitivity to actual contribution signals. Harder to tune entropy/clip because one term drives variance.
- **Triggers:** Seeds lingering in HOLDING with counterfactual available (epochs_in_stage ≥ 2, bounded_attribution > 0); common in sparse reward or cautious policies.
- **Root-Cause Hypothesis:** Penalty was steepened per DRL review to counterbalance large positive attribution and prevent WAIT-farming; the key risk is reward-scale dominance if not gated/observable.
- **Remediation Options:** 
  - A) Reduce cap (e.g., -3.0) or soften base multiplier.
  - B) Normalize reward components to a common scale and log to telemetry.
  - C) Anneal penalty strength over episodes instead of per-epoch exponential.
- **Risks of Change:** Altering reward magnitudes changes learned policy; needs re-baselining and may affect prior checkpoint compatibility.
- **Stopgap Mitigation:** Emit per-component telemetry (including `holding_warning`) and use `RewardNormalizer` to stabilize reward magnitudes.
- **Validation Plan:** Inspect component magnitudes in a short PPO run; ensure penalties stay within target range and overall reward variance remains stable. Optionally add a unit test that clamps component magnitude.
- **Status:** Closed (Superseded)
- **Resolution:** The current reward implementation intentionally gates and caps `holding_warning` at the PPO clip boundary (-10) to prevent HOLDING WAIT farming, and it is now observable (`RewardComponentsTelemetry.holding_warning`) and magnitude-stabilized via `RewardNormalizer`. The original ticket’s “dwarfs other components” framing no longer matches the current contribution-primary reward scale.
- **Links:** `src/esper/simic/rewards/rewards.py`, `src/esper/simic/rewards/reward_telemetry.py`, `src/esper/simic/control/normalization.py`
