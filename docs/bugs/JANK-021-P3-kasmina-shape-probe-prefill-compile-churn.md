# JANK Template

- **Title:** Shape probe creation inside germinate may cause torch.compile churn
- **Category:** performance / maintainability
- **Symptoms:** `_get_shape_probe` creates random tensors on-demand during germinate/validation. Under torch.compile, new probe shapes/devices can trigger graph specialization/guard churn. Cache helps, but device/topology changes (see BUG-014) or frequent germinations can create new probes mid-run.
- **Impact:** Lower but real—compile guard churn and potential perf hit during warmup; less of a correctness bug, more of a perf/compile stability concern.
- **Triggers:** Frequent germination with changing channels/topology; torch.compile active.
- **Root-Cause Hypothesis:** Probe creation interleaved with compiled regions; caching partially mitigates.
- **Remediation Options:** Pre-create probes per topology/device at init, keep deterministic shapes, and avoid creating new tensors inside compiled code paths; ensure cache invalidation is minimal.
- **Validation Plan:** Measure guard churn before/after prefill; ensure no new probes are created during compiled forwards.
- **Status:** Open
- **Links:** `src/esper/kasmina/slot.py::_get_shape_probe`, torch.compile behavior notes
