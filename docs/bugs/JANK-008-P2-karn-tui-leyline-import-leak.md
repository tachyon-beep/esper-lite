# JANK Template

- **Title:** Karn TUI type-check imports create a soft dependency on Leyline/Nissa
- **Category:** boundary hygiene / maintainability
- **Symptoms:** `src/esper/karn/tui.py` TYPE_CHECKING block imports `TelemetryEvent` from `esper.leyline.telemetry`. TUI is intended as a Nissa backend/consumer, but this import (and implicit reliance on event shape) tightens coupling. Any Leyline telemetry change requires Karn edits, and accidental runtime import could occur if TYPE_CHECKING guards are removed or tools evaluate annotations at runtime.
- **Impact:** Medium – cross-layer coupling; complicates treating Karn as an optional consumer separate from Leyline/Nissa. Risk of import churn when telemetry contracts evolve.
- **Triggers:** Static typing/IDE checks, or runtime annotation evaluation tools; evolving telemetry schema.
- **Root-Cause Hypothesis:** Convenience type hints pulled from Leyline instead of defining a local Protocol/dataclass for the minimal event fields Karn consumes.
- **Remediation Options:**
  - A) Define a local `TelemetryEventLike` Protocol/dataclass in Karn with only the fields TUI reads; avoid importing Leyline.
  - B) Route all event typing through a stable Leyline consumer-facing contract module to decouple from producer internals.
  - C) Keep TYPE_CHECKING but add clear boundary docs; ensure no runtime imports of Leyline from Karn.
- **Validation Plan:** Add a mypy/pyright check ensuring Karn builds without importing Leyline; run TUI as a backend in a minimal stub hub with dummy events implementing the Protocol.
- **Status:** Open
- **Links:** `src/esper/karn/tui.py` (TYPE_CHECKING TelemetryEvent), `src/esper/nissa/output.py` (backend contract)
