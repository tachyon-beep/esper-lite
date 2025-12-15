# JANK Template

- **Title:** GPU preload path hardcodes batch_size=512, ignores task/CLI config
- **Category:** config honesty / maintainability
- **Symptoms:** In `train_ppo_vectorized`, when `gpu_preload=True`, `load_cifar10_gpu` is called with a hardcoded `batch_size=512` (lines ~600). This bypasses `task_spec.dataloader_defaults` and any intended batch sizing, leading to unexpected VRAM usage and divergence between GPU-preload vs SharedBatchIterator modes.
- **Impact:** Medium – users enabling gpu_preload may OOM on smaller GPUs or get different optimization dynamics than configured. CLI/task batch size is silently ignored in this mode.
- **Triggers:** Running PPO with `--gpu-preload` (or equivalent config).
- **Root-Cause Hypothesis:** GPU preload path optimized for throughput on large GPUs; configurability was skipped.
- **Remediation Options:** 
  - A) Thread through a configurable batch size (respect task defaults/CLI), with a documented default override.
  - B) Emit telemetry/warning when overriding batch size; make override opt-in.
  - C) Add VRAM-aware heuristic/flag to cap batch size.
- **Validation Plan:** Add tests ensuring gpu_preload path uses provided batch size; run a smoke with a small batch to verify no silent override.
- **Status:** Open
- **Links:** `src/esper/simic/vectorized.py` gpu_preload branch (~600), `src/esper/utils/data.py::load_cifar10_gpu`
