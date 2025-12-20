# JANK Template

- **Title:** SignalTracker reset does not clear stabilization latch consistently
- **Category:** correctness-risk / reproducibility
- **Symptoms:** SignalTracker.reset() recreates histories and zeroes metrics but relies on `_is_stabilized` and `_stable_count` manual reset. If the tracker is reused across runs (not re-instantiated), stabilization state can persist unexpectedly; there’s no guard to re-run stabilization detection after reset beyond zeroing fields. In long-lived processes, accidental reuse may skip the gating phase.
- **Impact:** Medium – reused trackers (e.g., in integration tests or multi-episode heuristic runs without re-instantiation) may start with `_is_stabilized=True`, permitting germination immediately and altering behavior.
- **Triggers:** Reusing a SignalTracker instance across runs/episodes without creating a new one; incomplete reset due to missed fields or future additions.
- **Root-Cause Hypothesis:** Reset is manual and could drift as new stabilization fields are added; no invariant or assert to ensure stabilization gating restarts.
- **Remediation Options:**
  - A) Ensure reset explicitly clears `_is_stabilized`/`_stable_count` (currently done) and add a self-test/invariant; consider a fresh instance per run.
  - B) Add a `clear()` or context manager that guarantees full state reset, with a unit test covering all fields; optionally make tracker dataclass frozen and reconstruct.
  - C) Emit telemetry when stabilization re-locks to catch reuse errors.
- **Validation Plan:** Add a unit test reusing a tracker across two synthetic runs to ensure stabilization gating fires again (no latent True flag); assert reset covers all fields.
- **Status:** Closed (Resolved)
- **Resolution:** `SignalTracker.reset()` recreates histories and explicitly clears the stabilization latch (`_is_stabilized=False`, `_stable_count=0`), so reuse across runs/episodes does not retain stabilized state.
- **Links:** `src/esper/tamiyo/tracker.py` (`reset`)
