# Simic — PyTorch 2.8 Mandatory Upgrades (PPO/IMPALA Training)

Scope
- Optimise Simic’s offline training for PyTorch 2.8 with a single, guarded code path (no feature flags). Keep eager fallback only on hard failure.

Mandatory changes

1) Compile the training step closure
- What: Wrap a pure training step `(features, metric_sequence, seed_idx, blueprint_idx, returns) -> loss` with `torch.compile` to reduce Python overhead.
- How:
  - In `src/esper/simic/trainer.py`, factor out a `_train_step(batch)` method that includes forward, loss build, backward, scaler step, zero_grad.
  - At init, try `self._compiled_step = torch.compile(self._train_step, dynamic=True)`.
  - On exception, log once and use eager `_train_step`.
- Acceptance: Per‑iteration latency drops; correctness preserved under eager fallback.

2) AMP + GradScaler for PPO/IL updates
- What: Reduce bandwidth and speed matmuls with mixed precision.
- How: Wrap forward/loss in `torch.autocast('cuda', dtype=torch.bfloat16)` on Ampere (float16 otherwise) and use `torch.cuda.amp.GradScaler()` for backward/step.
- Acceptance: Improved throughput; no NaN/Inf regressions.

3) Set TF32 / matmul precision
- What: Optimise float32 matmul.
- How: `torch.set_float32_matmul_precision('high')`; enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`; `torch.backends.cudnn.allow_tf32 = True`.
- Acceptance: Stable speedup.

4) Data transfer / DataLoader hygiene
- What: Reduce H2D overhead and ensure pinned buffers.
- How: Use `pin_memory=True` on loaders (if used) and `.to(device, non_blocking=True)` for tensors.
- Acceptance: Lower copy time on GPU runs.

5) Optimiser foreach path
- What: Use `foreach=True` for optimisers to accelerate multi‑tensor ops.
- How: For Adam/AdamW, set `fused=True` where supported; otherwise `foreach=True`.
- Acceptance: Same numerics; faster step time.

Optional next
- CUDA Graphs capture for steady‑state PPO updates with fixed shapes; guard and fall back as shapes change.

References
- Trainer: `src/esper/simic/trainer.py`
- Policy network: `_PolicyNetwork` in the same file; adjust forward to support compiled training step where needed.
