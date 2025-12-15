# BUG Template

- **Title:** Counterfactual feature is unbounded, blows up observation normalization
- **Context:** Leyline / `FastTrainingSignals.to_features()` (`src/esper/leyline/signals.py`)
- **Impact:** P0 – observation vector can spike to unnormalized magnitudes when `seed_counterfactual` is large, destabilizing PPO and masking gradients; violates stated ~[-1, 1] normalization contract.
- **Environment:** Main branch; any PPO run that reports large counterfactual deltas; GPU/CPU alike.
- **Reproduction Steps:**
  1. Simulate a signal with `seed_counterfactual=1e6` and call `FastTrainingSignals.to_features()`.
  2. Observe the returned feature element is `100000.0` (1e6 / 10) instead of being clamped.
- **Expected Behavior:** Counterfactual is clamped symmetrically to [-10, 10] before division, matching `host_grad_norm` handling and keeping the feature within [-1, 1].
- **Observed Behavior:** No clamping; `self.seed_counterfactual / 10.0` emits arbitrarily large values (`src/esper/leyline/signals.py:150-160`).
- **Logs/Telemetry:** N/A (pure observation transform).
- **Hypotheses:** Clamp step was omitted when adding counterfactual; host_grad_norm already clamps so asymmetry is accidental.
- **Fix Plan:** Clamp `seed_counterfactual` to `[-10.0, 10.0]` before dividing by 10; add a unit test to enforce the bound.
- **Validation Plan:** New test: construct `FastTrainingSignals(seed_counterfactual=1e6).to_features()` and assert value is 1.0; rerun a short PPO smoke (`--episodes 1 --n-envs 1 --max-epochs 1`) to ensure no regression.
- **Status:** Open
- **Links:** `src/esper/leyline/signals.py:150-160`, architecture review P0 item
