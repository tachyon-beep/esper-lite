# JANK Template

- **Title:** Karn collector lifecycle not tied to Nissa hub reset; duplicate/backlogged events across runs
- **Category:** maintainability / boundary hygiene
- **Symptoms:** Karn collector backend (e.g., `Collector`/`websocket_output`) is added to Nissa hub but lacks a defined lifecycle/reset. When runs occur in the same process (tests/notebooks), collectors persist, causing duplicate emission and backlog growth.
- **Impact:** Medium – telemetry duplication, memory use, and stale output across runs.
- **Triggers:** Multiple runs without process restart; hub singleton reuse.
- **Root-Cause Hypothesis:** Collector treated as stateless backend; no reset hook.
- **Remediation Options:** Add close/reset hooks for Karn backends and integrate with Nissa hub reset (ties to JANK-009/Nissa lifecycle); ensure collectors clear buffers and threads on reset.
- **Validation Plan:** Add test running two training sessions in one process and assert collectors don't double-emit and buffers are cleared on reset.
- **Status:** Open
- **Links:** `src/esper/karn/collector.py`, `src/esper/nissa/output.py` hub lifecycle
