# BUG Template

- **Title:** Counterfactual feature normalization missing clamp (legacy V3 vector path)
- **Context:** Leyline / `FastTrainingSignals.to_vector()` (`src/esper/leyline/signals.py`), populated by Tamiyo `SignalTracker.update()` (`src/esper/tamiyo/tracker.py`)
- **Impact:** P0 (contract) – without clamping, this feature could spike far outside ~[-1, 1]. Current PPO uses the multi-slot feature path (`src/esper/simic/features.py`), but Leyline should still honor the normalization contract.
- **Environment:** Main branch; device-agnostic (CPU/GPU).
- **Reproduction Steps:**
  1. `PYTHONPATH=src python - <<'PY'\nfrom esper.leyline.signals import FastTrainingSignals, TensorSchema\nvec = FastTrainingSignals.empty()._replace(seed_counterfactual=1e6).to_vector()\nprint(vec[TensorSchema.SEED_COUNTERFACTUAL])\nPY`
  2. Verify the printed value is `1.0` (clamped from 1e6 → 10 → 1.0).
- **Expected Behavior:** Counterfactual is clamped symmetrically to [-10, 10] before division, matching `host_grad_norm` handling and keeping the feature within [-1, 1].
- **Observed Behavior (pre-fix):** No clamping; `self.seed_counterfactual / 10.0` could emit arbitrarily large values.
- **Observed Behavior (post-fix):** Clamped + normalized in `FastTrainingSignals.to_vector()`, and `TrainingSignals.seed_counterfactual` is populated from the summary seed’s `counterfactual_contribution` when available.
- **Logs/Telemetry:** N/A (pure observation transform).
- **Hypotheses / Root Cause:** Clamp step was omitted when adding counterfactual; `host_grad_norm` already clamps so asymmetry was accidental. Additionally, `TrainingSignals.seed_counterfactual` existed but was not populated by the tracker, making impact on PPO runs ambiguous.
- **Fix Plan:** Clamp `seed_counterfactual` to `[-10.0, 10.0]` before dividing by 10; populate `TrainingSignals.seed_counterfactual` from the summary seed’s `counterfactual_contribution`; add unit tests.
- **Validation Plan:** `pytest tests/test_simic_features.py::TestCounterfactualInObservation::test_to_vector_clamps_counterfactual` and `pytest tests/tamiyo/test_tracker_unit.py::TestSignalTrackerUpdate::test_update_populates_seed_counterfactual_when_available`.
- **Status:** Resolved
- **Links:** `src/esper/leyline/signals.py`, `src/esper/tamiyo/tracker.py`, `src/esper/simic/features.py`
