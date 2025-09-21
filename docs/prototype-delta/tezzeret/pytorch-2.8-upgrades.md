# Tezzeret — PyTorch 2.8 Mandatory Upgrades (Compilation Pipeline)

Scope
- Tezzeret owns all kernel compilation. Kasmina must never call `torch.compile` at runtime. This document defines the mandatory 2.8 compilation flow for Esper‑Lite.

Mandatory changes

1) Compile blueprints with `torch.compile`
- Status: ✅ Implemented in `src/esper/tezzeret/compiler.py`.
- How:
  - Build representative modules per blueprint, invoke `torch.compile(..., dynamic=True)`, and execute a pre‑warm pass to hydrate caches.
  - Persist the eager module plus metadata while priming Inductor cache (honouring `TEZZERET_INDUCTOR_CACHE_DIR`).
  - Capture fallback when compilation fails and surface the strategy in artifact metadata.

2) Attach shape/guard metadata to artifacts
- Status: ✅ Guard specs (shape/dtype/stride) persisted via Urza extras; digest stored in `KernelCatalogUpdate`.
- How: Guard spec derived from representative inputs accompanies each artifact and is available to Kasmina for verification.

3) Validate numeric and latency budgets at compile time
- Status: ✅ Compile and pre‑warm timings are captured per artifact and stored in Urza extras (baseline percentiles forthcoming).

4) Versioning and fallback routing
- Status: Partially complete — eager fallback flag and strategy recorded; semantic versioning & signing remain TODO.
- Next: Extend extras with semantic version identifiers and signatures once contract decided.

Kasmina (for completeness)
- Load the compiled artifact; verify shape guards; pre‑warm once (no compile calls).
- If compiled path is unavailable, run eager and emit telemetry indicating fallback.

Acceptance criteria
- No `torch.compile` in Kasmina; all compilation occurs in Tezzeret.
- Compiled artifacts published via Urza include guards and basic performance metadata.
- First‑use latency in Kasmina is dominated by cache warm‑up, not compilation.
