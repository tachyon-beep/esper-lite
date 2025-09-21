# Tezzeret — PyTorch 2.8 Mandatory Upgrades (Compilation Pipeline)

Scope
- Tezzeret owns all kernel compilation. Kasmina must never call `torch.compile` at runtime. This document defines the mandatory 2.8 compilation flow for Esper‑Lite.

Mandatory changes

1) Compile blueprints with `torch.compile`
- What: For each blueprint, construct a representative module and compile it with `torch.compile` to eliminate first‑run latency in Kasmina.
- How:
  - In `src/esper/tezzeret/compiler.py` and/or `runner.py`:
    - Build the module for the blueprint.
    - Wrap with `compiled = torch.compile(module, dynamic=True)` (consider `mode='reduce-overhead'` for inference‑dominant kernels).
    - Pre‑warm with one or more representative input shapes to populate caches.
    - Persist the artifact (weights + exported graph and/or metadata needed for guard checks). Prefer `torch.export` to capture guards for shape validation, alongside state dict.
    - Persist Inductor cache (set `TORCHINDUCTOR_CACHE_DIR`) to enable cross‑process reuse when feasible.
  - On any compilation failure, store a fallback eager artifact and record the failure in metadata for Kasmina to surface via telemetry when used.

2) Attach shape/guard metadata to artifacts
- What: Provide Kasmina with the information to verify input compatibility without recompiling.
- How: Save guards from `torch.export` or a schema describing dynamic dims and constraints; store with the artifact in Urza.

3) Validate numeric and latency budgets at compile time
- What: Ensure compiled kernels meet basic inference/training budget expectations before publishing to Urza.
- How: After pre‑warm, record p50/p95 latencies on a small set of shapes; write results into artifact metadata.

4) Versioning and fallback routing
- What: Version compiled artifacts per kernel+shape profile; include a flag signalling eager fallback when compiled path is unavailable at runtime.
- How: Add version/flags to the Urza descriptor; Kasmina reads and logs metadata; if eager fallback is used, surface a WARNING event.

Kasmina (for completeness)
- Load the compiled artifact; verify shape guards; pre‑warm once (no compile calls).
- If compiled path is unavailable, run eager and emit telemetry indicating fallback.

Acceptance criteria
- No `torch.compile` in Kasmina; all compilation occurs in Tezzeret.
- Compiled artifacts published via Urza include guards and basic performance metadata.
- First‑use latency in Kasmina is dominated by cache warm‑up, not compilation.

