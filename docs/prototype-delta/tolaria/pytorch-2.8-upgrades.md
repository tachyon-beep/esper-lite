# Tolaria — PyTorch 2.8 Mandatory Upgrades (No Tech Debt)

Scope
- Optimise the training orchestrator for PyTorch 2.8. Apply a single, guarded code path (no feature flags); keep eager fallback only on hard failure.

Mandatory changes

1) Compile the training step closure
- What: Wrap a pure `train_step(inputs, targets)` closure (forward → loss → backward → optimizer.step → zero_grad) with `torch.compile` to reduce Python overhead.
- How:
  - In `src/esper/tolaria/trainer.py`, factor out a `_train_step` method or closure that accepts preloaded tensors.
  - At initialisation, try `self._compiled_step = torch.compile(self._train_step, dynamic=True)`.
  - On exception, log once and use the eager `_train_step` (no feature flag, just a guarded fallback).
- Acceptance: Per‑batch latency drops; correctness preserved; failure path uses eager and emits a telemetry WARNING once.

2) AMP with bfloat16 (or float16) and GradScaler
- What: Use autocast + GradScaler during training to reduce bandwidth and speed up matmuls.
- How:
  - In the training loop, wrap forward/loss in `torch.autocast('cuda', dtype=torch.bfloat16)` on Ampere+ (A100), otherwise float16 as appropriate.
  - Use `torch.cuda.amp.GradScaler()` with scaled backward/step.
- Acceptance: Throughput improves; no NaN/Inf regressions; scaler adjusts automatically.

3) Set global matmul precision / TF32
- What: Optimise float32 matmul on A100‑class hardware in line with 2.8 guidance.
- How:
  - At process init: `torch.set_float32_matmul_precision('high')`.
  - Enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`; `torch.backends.cudnn.allow_tf32 = True`.
- Acceptance: Higher throughput with stable numerics.

4) Validation path hygiene
- What: Use `torch.inference_mode()` for validation/inference epochs to avoid autograd tracking.
- How: Wrap the end‑of‑epoch validation (wherever applicable) in `inference_mode()`.
- Acceptance: Lower validation overhead and memory.

5) Data transfer improvements (when CUDA is used)
- What: Reduce H2D overheads.
- How: Use DataLoader with `pin_memory=True`, set `.to(device, non_blocking=True)` on tensors.
- Acceptance: Measurable reduction in batch copy time on GPU runs.

6) SGD foreach path (if retained)
- What: Use the foreach implementation for SGD to speed multi‑tensor operations.
- How: Construct `torch.optim.SGD(..., foreach=True)` if CUDA is used; otherwise default.
- Acceptance: Same numerics; improved step time with many parameter tensors.

Optional next (post‑upgrade)
- CUDA Graphs capture for steady‑state training steps (fixed shapes); guarded fallback to normal path on shape change.
- Compile separate eval step for batched validation if it’s a bottleneck.

References
- Training loop: `src/esper/tolaria/trainer.py`
- Device selection / matmul precision are typically set at process start
