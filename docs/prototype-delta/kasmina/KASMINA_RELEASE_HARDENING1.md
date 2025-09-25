# Kasmina Release Hardening Plan 1 (Housekeeping Line)

## Context & Rationale
Kasmina’s prototype implementation is feature-complete for the current delta, but several housekeeping issues increase ongoing maintenance cost and obscure real reliability signals. Key cleanliness goals for this iteration:

- Align lifecycle telemetry and memory governance with accurate statistics so GC alerts reflect actual behaviour.
- Remove redundant or confusing imports/state to keep the seed manager easier to scan.
- Trim per-step overhead in blending and event queues to reduce the chance of soft performance regressions when the continuous runners go live.
- Strengthen background service diagnostics so Weatherlight/Kasmina operators only page on true failures.

These changes are strictly internal; no behaviour flags or compatibility shims are required.

## Work Packages

### KAS-H1-001 — Memory GC Telemetry Accuracy
- **Scope:** Update `TTLMemoryCache.cleanup()` (`src/esper/kasmina/memory.py:43-58`) to return the count of entries evicted. Propagate the value through `KasminaMemoryManager.periodic_gc()` (`src/esper/kasmina/memory.py:83-122`) and `KasminaSeedManager.update_epoch()` (`src/esper/kasmina/seed_manager.py:944-974`) so telemetry events reflect real eviction counts.
- **Acceptance:** GC telemetry emits integer counts matching cache size changes; unit tests cover cleanup return value; integration smoke (`tests/kasmina/test_memory.py`) verifies event payload.
- **Risk:** Low — pure housekeeping; easy rollback.

### KAS-H1-002 — Pending Event Queue Efficiency
- **Scope:** Replace the O(n²) drop loop in `_queue_seed_events` (`src/esper/kasmina/seed_manager.py:244-263`) with an efficient drop strategy (slice or `collections.deque`). Ensure telemetry event `seed_queue_dropped` still fires with accurate counts.
- **Acceptance:** New helper keeps queue size bounded with O(n) behaviour; tests cover overflow logic.
- **Risk:** Low — focused data-structure tweak.

### KAS-H1-003 — Import / State Consolidation
- **Scope:** Collapse duplicated Leyline imports in `seed_manager.py` (`leyline_pb2` and alias `pb`). Remove unused `_request_counter` and any comments referencing the removed state. Update call sites accordingly.
- **Acceptance:** Single import path; static analysis confirms no unused variables; existing tests pass.
- **Risk:** Low.

### KAS-H1-004 — Alpha Schedule Lightweight Implementation
- **Scope:** Reimplement `AlphaSchedule.value()` in `src/esper/kasmina/blending.py:20-26` using `math` rather than `torch` tensor construction to avoid per-call tensor allocations and autograd baggage.
- **Acceptance:** Unit tests (`tests/kasmina/test_blending.py`) cover new implementation and confirm values match previous behaviour within tolerance.
- **Risk:** Low.

### KAS-H1-005 — Isolation Hook Refactor
- **Scope:** Refactor `_make_host_hook` / `_make_seed_hook` in `src/esper/kasmina/isolation.py:48-118` to share projection and dot-product logic while keeping behaviour identical. Add targeted unit tests to confirm dot-product/Norm accumulation still matches existing metrics.
- **Acceptance:** Shared helper with clear structure and test coverage; no behaviour change flagged by isolation tests.
- **Risk:** Medium — changed structure in safety-critical monitoring; mitigated by unit tests.

### KAS-H1-006 — Prefetch Task Diagnostics
- **Scope:** Adjust `KasminaPrefetchCoordinator.poll_task_issue()` (`src/esper/kasmina/prefetch.py:63-103`) so it distinguishes between tasks cancelled via `close()` and true unexpected exits, preventing false alarms during cleanup.
- **Acceptance:** Unit/integration tests confirm graceful shutdown produces no spurious `RuntimeError` while hard failures still surface.
- **Risk:** Low.

## Risk Assessment
- Overall plan risk: Low-to-Medium. Most work is minor refactoring; the isolation hook cleanup touches risk-sensitive code but includes comprehensive tests.
- Potential failure modes:
  1. Isolation refactor introduces subtle numerical differences → guard with before/after test harness.
  2. GC telemetry miscounts due to incorrect return handling → covered by regression tests.
  3. Prefetch diagnostics misclassify errors → exercise through async unit tests.

## Risk Reduction Activities
1. **Isolation Module Snapshot Test:** Before refactoring, capture current isolation statistics for synthetic gradients; rerun after change to ensure identical dot-products/norms.
2. **Telemetry Smoke Run:** After KAS-H1-001, execute `pytest tests/kasmina/test_memory.py` and inspect emitted telemetry payloads for correct eviction counts.
3. **Prefetch Lifecycle Test:** Add/execute an async test that starts/stops the prefetch coordinator to confirm `poll_task_issue()` returns `None` on clean shutdown.

## Validation Strategy
- Run targeted unit suites: `pytest tests/kasmina` plus specific async prefetched tests.
- Manual inspection (debug logs) for telemetry events post-GC update in staging.
- Commit only after linters (pylint for `src/esper/kasmina`) and tests pass.

## Residual Risk and Confidence
- **Residual Risk:** Low once the above tasks land; remaining exposure is limited to potential isolation refactor mistakes, which are bounded by tests.
- **Confidence:** High — the code surfaces are small, and the existing test coverage gives strong support.
