# Kasmina — Blending Upgrades (Executor-Side)

Purpose: extend Kasmina’s grafting blend beyond a single scalar α to improve stability and control while preserving isolation and compile stability. This guide specifies concrete, low‑risk modes Tamiyo can learn and how to implement them in Kasmina without changing Leyline contracts.

Scope and invariants
- Isolation invariant: always detach the host branch in blend paths.
  - out = f(seed_out, host_out.detach(), α, …)
- α is a runtime buffer/tensor (not an nn.Parameter). Keep shapes stable to avoid retraces.
- No Leyline schema changes: choose blend mode via internal config/state; do not add new enums to Leyline.
- PyTorch 2.8 assumptions: torch.compile available; prefer shape‑stable, fused elementwise ops; avoid .item() in hot paths.

Policy selection boundary
- Blending applies to seed integration phases only (grafting: TRAINING → BLENDING → SHADOWING/PROBATIONARY), not general training.
- Tamiyo selects the blending mechanism from a small, approved set based on policy/risk. Kasmina executes the requested mode safely; it does not choose the mode.
- The selection signal should be conveyed by Tamiyo via `AdaptationCommand` annotations or parameters. If omitted, Kasmina uses its safe default (convex blend with host.detach()).

Safety rails and budgets
- α bounds: clamp α to [0, 1]. For maps: cap fraction of elements > τ (e.g., τ=0.8) to ≤ p% (e.g., 20%).
- Hysteresis: apply a small dead‑band (±0.02) around α thresholds to reduce flapping and retraces.
- Telemetry: record α summary (mean, p95), mode, and any clamping/sparsity applied.

Recommended near‑term modes (implement now)
1) Residual‑Gated Injection (scalar α)
- Formula (detached host):
  - out = host + α · seed, where host := host_out.detach()
- Rationale: keeps host as baseline; injects seed improvements proportionally. More stable than convex mixing for some tasks.
- PyTorch 2.8 kernel (shape‑agnostic):
  ```python
  @torch.compile(fullgraph=False, dynamic=True)
  def blend_residual(host, seed, alpha):
      # host, seed: same shape; alpha: scalar tensor on same device/dtype
      return host + alpha * seed
  ```
- Notes:
  - α is a rank‑0 tensor on the device (e.g., `alpha = torch.tensor(α_float, device=host.device, dtype=host.dtype)`).
  - Maintain dtype parity (TF32/FP16 as configured) to keep fusion stable.
- Acceptance:
  - Host branch is detached; isolation hooks see zero gradients in host path.
  - No Dynamo retraces when α changes per‑batch.
  - Latency within baseline ±0.05 ms vs convex blend.

2) Channel/Group‑Wise α (vector α)
- Formula (NCHW as example):
  - out[n, c, h, w] = α[c] · seed[n, c, h, w] + (1 − α[c]) · host[n, c, h, w]
- API/data shape:
  - α has shape [C] or [G] for group‑wise, broadcast to activation shape. Store α in seed state as a runtime buffer; update per‑batch or per‑step schedule.
- PyTorch 2.8 kernel (channels‑last ok):
  ```python
  @torch.compile(fullgraph=False, dynamic=True)
  def blend_channelwise(host, seed, alpha_vec):
      # alpha_vec: shape [C] broadcast to activation shape
      while alpha_vec.dim() < host.dim():
          alpha_vec = alpha_vec.unsqueeze(0)
      # Broadcast over N and spatial dims
      return alpha_vec * seed + (1 - alpha_vec) * host
  ```
- Notes:
  - Prefer contiguous channels‑last for 2D to maximise fusion.
  - For group‑wise, precompute a per‑channel α by repeating per‑group weights to the channel dimension.
  - Clamp α per‑channel to [α_min, α_max]; optionally L1 penalise the deviation from 0/1 to encourage sparsity.
- Acceptance:
  - Broadcast shapes correct for 1D/2D/3D features.
  - Isolation maintained; gradient only through seed and α buffers if α requires grad=False.
  - Perf: ≤ 1.15× cost of scalar blend for typical C.

3) Confidence‑Aware Scalar α (adaptive gate)
- Idea: modulate α based on seed confidence or stability signals to reduce risk under uncertainty.
- Example (classification logits):
  - margin = top1(seed_logits) − top2(seed_logits)
  - gate = sigmoid(k · (margin − τ))  # k≈4–8, τ≈0.1–0.3
  - α_eff = clamp(α_base · gate, α_lo, α_hi)
- Implementation sketch:
  ```python
  @torch.compile(fullgraph=False, dynamic=True)
  def compute_margin_gate(seed_logits, k: float, tau: float):
      top2 = torch.topk(seed_logits, k=2, dim=1).values  # [N, 2]
      margin = top2[:, 0] - top2[:, 1]                   # [N]
      return torch.sigmoid(k * (margin - tau))           # [N]

  @torch.compile(fullgraph=False, dynamic=True)
  def blend_confidence(host, seed, alpha_base, gate):
      # gate: [N] broadcast to activation shape; alpha_base: scalar tensor
      # α_eff = clamp(alpha_base * gate_n)
      while gate.dim() < host.dim():
          gate = gate.unsqueeze(-1)
      alpha_eff = (alpha_base * gate).clamp_(0.0, 1.0)
      return alpha_eff * seed + (1 - alpha_eff) * host
  ```
- Notes:
  - When not in a classification regime, substitute a drift/stability signal (e.g., moving std of outputs) to compute gate.
  - Add hysteresis around τ to avoid flapping.
- Acceptance:
  - α_eff remains in [0,1], mean α decreases on low‑confidence batches.
  - No retrace across varying batch gates; verify with compile cache hit rate.

Optional modes (defer unless needed)
A) Delta‑Blending (scalar α)
- out = host + α · (seed − host). Equivalent to convex blend but emphasises the correction term; can improve drift control and telemetry interpretability for “injected delta”.
- Implementation is identical cost to convex blend; consider if residual metrics are preferred.

B) Logit‑Space Blending (classification)
- Blend in logit space then softmax:
  - z = α · z_seed + (1 − α) · z_host
  - p = softmax(z)
- Alternative is probability‑space mixture p = α·softmax(z_seed) + (1−α)·softmax(z_host) (avoid if you need logit continuity).

Integration with Tamiyo (selection and safety)
- Tamiyo chooses the mode and (optionally) hyper‑parameters from the approved list.
- Kasmina validates the requested mode against safety constraints (isolation, shapes, dtype), applies it, or falls back to default on mismatch.
- Telemetry includes `blend_mode`, α summary, and any clamping applied; on fallback, emit a warning event.

Defaults & prototype behaviour
- Default mode: convex blend `α·seed + (1−α)·host.detach()` (implemented by `AlphaBlender`).
- Advanced modes above are design‑complete and intended to be selected by Tamiyo; prototype keeps default unless explicitly requested.
- Ensure host logits come from a detached forward.

C) Two‑Expert Soft Gate (seed vs host)
- out = w_seed·seed + w_host·host, with [w_seed, w_host] = softmax(g(x)) from a tiny MLP.
- Guardrails: limit depth/width of g, add L2 and entropy regularisers; apply hysteresis and cap |Δw| per step.

Executor wiring plan (no code committed here)
- State/config
  - Add an internal BlendMode enum in Kasmina (executor scope only): {CONVEX, RESIDUAL, CHANNEL, CONFIDENCE, DELTA, LOGIT, SOFT_GATE}.
  - Add BlenderConfig to seed state: scalar α_base, optional α_vec (C or G), mode, hysteresis ε, clamps, and gate hyper‑params (k, τ).
  - α buffers live on the same device/dtype as activations; never Parameters; requires_grad=False.
- Forward integration
  - In the grafting branch, select the compiled blend kernel by mode.
  - For CHANNEL mode, pre‑broadcast α_vec once per batch/step to avoid repeated unsqueeze.
  - For CONFIDENCE mode, compute gate from the current batch’s logits or stability metric before blending and cache per batch.
- Scheduling α
  - Continue the existing α(t) ramp; for CHANNEL, ramp the vector towards target α_vec; for CONFIDENCE, α_eff = α_ramp · gate.
  - Add per‑batch advance helper so α progresses smoothly during BLENDING.

PyTorch 2.8 / Inductor guidance
- Keep α as runtime buffers; never embed α as a constant that could trigger retrace.
- Prefer elementwise, broadcasted math; avoid data‑dependent Python branching in hot path.
- Maintain shape‑class stability (N, C, H, W consistent; channels‑last for 2D if configured).
- Avoid `.item()` on tensors during forward; use tensor ops for clamps/hysteresis.
- Compile blend kernels separately; keep the number of modes small to maximise cache reuse.

Safety and tests (must add)
- Isolation: unit tests assert no gradients flow into host branch under each mode (backward hooks remain silent).
- Numeric bounds: α in [0,1]; for CHANNEL, α_vec clamped; for CONFIDENCE, α_eff in [α_lo, α_hi].
- Stability: hysteresis prevents oscillation on synthetic sequences; verify ≤ 1 recompilation over 1k α changes.
- Performance: micro‑bench compare latency against current convex blend; budget regressions ≤ +15% for CHANNEL, ≤ +5% for RESIDUAL/CONFIDENCE.
- Correctness: broadcast tests for 1D/2D/3D shapes; classification tests for LOGIT blend (probabilities sum to 1, calibration sanity).

Telemetry (suggested)
- Report: blend_mode, alpha_mean, alpha_p95, alpha_sparsity (CHANNEL), gate_mean (CONFIDENCE), clamped_fraction, kernel_ms.
- Route CRITICAL events on violations (e.g., α NaN, broadcast mismatch) via emergency path when available.

Traceability
- Parent design: 02‑kasmina‑unified‑design.md (Lifecycle and safety invariant); 02.1‑kasmina‑kernel‑execution.md (blend branch and α schedule).
- This upgrade refines the blending function only; it does not alter lifecycle, gates, or Leyline contracts.

Implementation order of operations (coder checklist)
1. Add BlendMode and BlenderConfig to Kasmina executor state; ensure default = existing convex blend.
2. Implement the three kernels above (RESIDUAL, CHANNEL, CONFIDENCE) with torch.compile wrappers.
3. Wire mode selection into the grafting forward path; keep host branch detached.
4. Add per‑batch α advance helper and call during BLENDING.
5. Add tests per Safety and tests section; extend telemetry with blend summaries.
6. Benchmark to confirm budgets and compile cache stability.
