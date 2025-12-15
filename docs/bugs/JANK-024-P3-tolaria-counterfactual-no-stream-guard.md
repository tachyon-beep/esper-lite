# JANK Template

- **Title:** Counterfactual validation in Tolaria runs on default stream, ignores potential stream context
- **Category:** performance / maintainability
- **Symptoms:** `validate_with_attribution` and `_run_validation_pass` run on the default stream with no option to integrate CUDA streams or non_blocking transfers beyond basic `.to(device)`. In multi-stream setups (used elsewhere in vectorized Simic), this can serialize validation and counterfactual passes, reducing overlap and potentially causing unintended syncs.
- **Impact:** Lower but relevant for throughput; could matter when integrating with multi-stream training.
- **Triggers:** GPU runs with concurrent stream usage.
- **Root-Cause Hypothesis:** Tolaria designed for simpler single-stream flow; not updated when vectorized multi-stream approach landed.
- **Remediation Options:**
  - A) Add optional stream context parameter to validation functions; default to current behavior.
  - B) Document serialization; ensure callers in multi-stream contexts are aware.
- **Validation Plan:** Add a test or benchmark demonstrating stream-enabled validation doesn’t break correctness and can overlap transfers.
- **Status:** Open
- **Links:** `src/esper/tolaria/trainer.py` validation/counterfactual functions
