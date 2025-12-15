# JANK Template

- **Title:** Validation/training metrics sample only 10 train batches, can mislead telemetry
- **Category:** correctness-risk / observability
- **Symptoms:** `validate_and_get_metrics` samples at most 10 train batches to compute train_loss/train_accuracy. For large datasets or early epochs, this small sample may not reflect true training performance, yet telemetry and decisions may consume these metrics.
- **Impact:** Medium – misleading train metrics in telemetry/dashboards; could influence stabilization/heuristic behavior if reused.
- **Triggers:** Tasks with large/imbalanced datasets; noisy early training.
- **Root-Cause Hypothesis:** Perf optimization to avoid full pass; sampling hard-coded to 10.
- **Remediation Options:**
  - A) Make sample size configurable or proportional to dataset size; allow full pass when needed.
  - B) Emit telemetry noting sampled batches; avoid using sampled train metrics for decisions.
  - C) Separate “quick sample” from “true train metrics” to avoid misinterpretation.
- **Validation Plan:** Add test ensuring configurable sample count; verify telemetry marks sampling; optionally compare sampled vs full metrics in CI.
- **Status:** Open
- **Links:** `src/esper/tolaria/trainer.py::validate_and_get_metrics` (itertools.islice(trainloader, 10))
