# JANK Template

- **Title:** Karn counterfactual helper emits telemetry without Leyline contracts
- **Category:** correctness-risk / boundary hygiene
- **Symptoms:** `CounterfactualHelper._emit_events` (`src/esper/karn/counterfactual_helper.py`) constructs `TelemetryEvent` directly in Karn and emits via `get_hub()`. This couples Karn to Nissa and Leyline at runtime, bypassing shared contract helpers and blurring the “Karn consumes Nissa” boundary. Any change to TelemetryEvent fields/types requires Karn updates; failures are silently swallowed in a broad except.
- **Impact:** Medium – cross-subsystem coupling and silent failures. Changes to telemetry schema or hub behavior can break counterfactual emission without type safety. Violates the intended direction (producers emit to Nissa; Karn consumes).
- **Triggers:** Using `CounterfactualHelper` with `emit_events=True` (default) during training/eval.
- **Root-Cause Hypothesis:** Helper added a convenience emit path instead of routing through existing producers in Simic/Tolaria; no contract wrapper exists.
- **Remediation Options:**
  - A) Remove emission from Karn; have the training loop (Simic/Tolaria) emit `COUNTERFACTUAL_COMPUTED` via Nissa using Leyline contracts. Karn consumes the stored telemetry only.
  - B) If emission remains, wrap TelemetryEvent construction in a shared Leyline helper and add typed errors/logging (no broad except), keeping Karn’s dependency surface minimal.
  - C) Default `emit_events=False` in `CounterfactualHelper` to keep Karn as a pure consumer; expose a separate, explicitly training-side helper for emission.
- **Validation Plan:** Add a test asserting counterfactual telemetry is emitted from the training side (or via a Karn helper that uses Leyline contracts) and fails loudly on schema mismatch; ensure Karn can operate without Nissa import.
- **Status:** Open
- **Links:** `src/esper/karn/counterfactual_helper.py::_emit_events`, `src/esper/nissa/__init__.py`
