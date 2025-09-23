# Tamiyo — PyTorch 2.8 Mandatory Upgrades (Hetero‑GNN Inference + Optional Training)

Scope
- Optimise Tamiyo’s policy inference (4‑layer hetero‑GNN in the unified design) for PyTorch 2.8. Apply a single, guarded code path (no feature flags). Keep eager fallback only on hard failures. If policy training is enabled in‑process (PPO), mirror Simic’s 2.8 changes for training.

Mandatory changes (inference)

1) Compile the policy inference step
- What: Wrap the policy’s forward callable with `torch.compile` to reduce Python overhead.
- How:
  - In the policy module (e.g., `TamiyoGNN.forward()`): at service init, try `self._compiled = torch.compile(self.forward, dynamic=True, mode='reduce-overhead')` and route inference through it.
  - On exception, log once and fall back to eager.
- Acceptance: Per‑inference latency drops; failure path uses eager and emits a single WARNING.

2) Use `torch.inference_mode()` and autocast
- What: Prevent autograd book‑keeping and reduce precision costs where safe.
- How: Wrap inference in `torch.inference_mode()` and `torch.autocast('cuda', dtype=torch.bfloat16)` on Ampere+ (fall back to float16 or disable autocast if CPU).
- Acceptance: Lower memory and improved throughput without accuracy regressions.

3) Set global matmul precision / TF32
- What: Optimise float32 matmul on A100‑class hardware.
- How: At process init, set `torch.set_float32_matmul_precision('high')`; enable TF32: `torch.backends.cuda.matmul.allow_tf32 = True`; `torch.backends.cudnn.allow_tf32 = True`.
- Acceptance: Throughput improves with stable numerics.

4) Data transfer improvements (when CUDA is used)
- What: Reduce H2D overheads.
- How: Pin input buffers where applicable and use `.to(device, non_blocking=True)` before inference.
- Acceptance: Lower copy time; end‑to‑end latency reduction.

Optional (if policy training runs in Tamiyo)
- Compile the training step closure (see Simic upgrades). Use AMP + GradScaler for PPO updates.
- Consider CUDA Graphs for steady‑state inference if shapes are fully stable.

References
- Tamiyo service: `src/esper/tamiyo/service.py` (evaluate_epoch)
- Policy stub: `src/esper/tamiyo/policy.py` (replace with hetero‑GNN for the unified design before adopting the compile path)

## Operator/Dev Notes

- Compile fallback counter
  - Tamiyo exports `tamiyo.gnn.compile_fallback_total` (count) when the compile path is disabled at init or falls back at runtime due to backend issues. `tamiyo.gnn.compile_enabled` reflects the current state (1.0=compiled, 0.0=eager).
  - Expect occasional fallbacks with some PyG kernels or exotic CUDA stacks; Tamiyo continues in eager mode without crashing.

- Running CUDA perf checks locally
  - Ensure a CUDA‑capable GPU and drivers are available (`torch.cuda.is_available()` returns True).
  - Run the CUDA perf test (skipped on CPU or on unsupported backends):
    - `pytest -q tests/tamiyo/test_policy_gnn.py::test_policy_inference_perf_budget_cuda_compile`
  - The test compares p95 latency of eager vs compile‑on for a representative graph. It skips if the compile path falls back or raises during warmup/measurement. When it runs, the expectation is compile p95 ≤ max(45 ms, eager p95 × 1.10).
  - Troubleshooting compile performance:
    - Set `TORCH_LOGS=recompiles` to diagnose excessive dynamo recompiles.
    - Verify TF32 is enabled and autocast is on (see policy init). Disable downstream ops that mutate module parameters during inference.
