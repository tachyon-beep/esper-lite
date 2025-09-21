# Tolaria — PyTorch 2.8 Mandatory Upgrades (No Tech Debt)

Scope
- Optimise the training orchestrator for PyTorch 2.8. Apply a single, guarded code path (no feature flags); keep eager fallback only on hard failure.

Mandatory changes

1) Compile the training step closure — **Implemented**
   - `_eager_train_step` in `src/esper/tolaria/trainer.py` encapsulates the forward/backward path.
   - When CUDA + `torch.compile` are available, the trainer compiles the step once and falls back on failure (telemetry event `tolaria.compile_*`).
   - CPU environments skip compilation to avoid toolchain dependencies; metrics expose whether compilation is active.

2) AMP with bfloat16 (or float16) and GradScaler — **Implemented**
   - CUDA runs now wrap forward/loss in `torch.cuda.amp.autocast` and use `GradScaler` for scaled updates.
   - CPU paths remain eager; telemetry reports `tolaria.train.amp_enabled` for observability.

3) Set global matmul precision / TF32 — **Implemented**
   - The trainer initialisation calls `_initialise_pytorch_defaults()` to set `torch.set_float32_matmul_precision('high')` and enable TF32 once per process (guarded on CUDA).

4) Validation path hygiene — **Implemented**
   - The Tamiyo/Kasmina handshake and catalog interactions now run under `torch.inference_mode()` so end-of-epoch work avoids autograd tracking.

5) Data transfer improvements — **Implemented**
   - CUDA runs attempt to enable DataLoader `pin_memory` and copy tensors with `non_blocking=True` before training steps.

6) SGD foreach path — **Implemented**
   - When running on CUDA, SGD optimisers are configured with `foreach=True` for faster multi-tensor updates.

Optional next (post-upgrade)
- CUDA Graphs capture for steady-state training steps (fixed shapes); guarded fallback to normal path on shape change.
- Compile separate eval step for batched validation if it becomes a bottleneck.

References
- Training loop: `src/esper/tolaria/trainer.py`
- Device selection / matmul precision are typically set at process start
