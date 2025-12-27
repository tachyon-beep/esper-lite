# JANK Template

- **Title:** Heuristic Tamiyo ransomware detection is TODO and can mis-fossilize harmful seeds
- **Category:** correctness-risk / resilience
- **Symptoms:** In HOLDING, heuristic Tamiyo only checks counterfactual_contribution/total_improvement and has a TODO for ransomware detection (seeds that increase dependence but hurt overall performance). Reward code handles this, but heuristic decisions can still fossilize such seeds.
- **Impact:** Medium – heuristic baseline can lock in harmful seeds, skewing comparisons and potentially misleading users who rely on heuristic runs for sanity checks.
- **Triggers:** Seeds with positive counterfactual but negative total improvement (ransomware pattern) in HOLDING under heuristic control.
- **Root-Cause Hypothesis:** TODO left in place; heuristic path not updated to mirror Simic’s ransomware safeguards.
- **Remediation Options:**
  - A) Implement ransomware guard: require both counterfactual > threshold and total improvement ≥ 0 before fossilizing; otherwise favor cull.
  - B) Emit telemetry when ransomware pattern detected to inform users.
  - C) Add config knob to align heuristic behavior with reward anti-ransomware logic.
- **Validation Plan:** Add property test mirroring `TestRansomwareDetection` to ensure heuristic culls ransomware seeds; run heuristic smoke to confirm no regressions.
- **Status:** Closed (Resolved)
- **Resolution:** Heuristic Tamiyo now implements ransomware detection in the HOLDING branch (cull when counterfactual contribution is positive but total improvement is negative, with configurable thresholds), removing the earlier TODO gap.
- **Links:** `src/esper/tamiyo/heuristic.py` (HOLDING ransomware detection), `src/esper/simic/rewards/rewards.py` (anti-ransomware reward shaping)
