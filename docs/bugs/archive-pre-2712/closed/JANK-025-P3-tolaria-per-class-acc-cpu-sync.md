# JANK Template

- **Title:** Per-class accuracy computation forces CPU sync per validation epoch
- **Category:** performance
- **Symptoms:** `validate_and_get_metrics` computes per-class accuracy using `torch.bincount` on GPU tensors then calls `.cpu()` and iterates to build a Python dict. For tasks with many classes or frequent validation, this CPU sync and Python loop can be a bottleneck; no batching/async path or option to skip dict materialization when not needed.
- **Impact:** Lower but present; wastes time on frequent validations and large class counts; might stall GPU pipeline.
- **Triggers:** `compute_per_class=True` (likely in diagnostic mode); large num_classes.
- **Root-Cause Hypothesis:** Simplicity over performance; per-class was added for diagnostics without optimization.
- **Remediation Options:** 
  - A) Make per-class aggregation optional (default off) and document cost.
  - B) Return tensors instead of Python dict to avoid per-class iteration; leave dict building to callers when needed.
  - C) Use async CPU transfer or deferred aggregation.
- **Validation Plan:** Benchmark per-class mode on large class count; ensure optional path avoids CPU sync.
- **Status:** Closed (By design)
- **Resolution:** Per-class accuracy is already optional (`compute_per_class=False` by default) and is intended for diagnostic/research runs (typically with small class counts like CIFAR-10). The CPU materialization is an acceptable tradeoff for readability/telemetry in that mode.
- **Links:** `src/esper/tolaria/trainer.py` (`validate_and_get_metrics` per-class section), `src/esper/nissa/config.py` (`PerClassConfig`)
