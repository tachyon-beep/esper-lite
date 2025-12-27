# JANK Template

- **Title:** No standardized lifecycle/reset for Nissa backends (incl. Karn)
- **Category:** maintainability / boundary hygiene
- **Symptoms:** Nissa backends are plain objects with `emit(event)`; there is no interface for initialization/teardown. Long-lived processes/tests that add Karn/Directory/Console backends cannot cleanly reset them without replacing the singleton hub. This blurs lifecycle boundaries and leaves resource management (file handles, threads) ad hoc.
- **Impact:** Lower-severity but persistent: duplicate emissions, stale file paths, dangling threads in Karn TUI, and inability to reuse the hub cleanly across runs.
- **Triggers:** Multiple runs in one process (notebooks/tests) adding backends repeatedly; absence of `reset`/`close` hooks.
- **Root-Cause Hypothesis:** Backend contract was kept minimal; lifecycle concerns were deferred. As Karn became a standard backend, missing teardown surfaced.
- **Remediation Options:**
  - A) Define a BackendProtocol with optional `start()`/`close()` and have Nissa manage lifecycle; call close() on reset.
  - B) Add a `reset_hub()` that clears backends and calls close where present (ties to JANK-006 singleton leak).
  - C) Ensure Karn backends expose close/cleanup methods for files/threads.
- **Validation Plan:** Add a test that registers a mock backend with a `closed` flag, resets the hub, and asserts cleanup occurred; run two training invocations in one process and check no duplicate emissions/FD leaks.
- **Status:** Closed (Fixed)
- **Resolution:** Fixed by adding `OutputBackend.start()` to the backend contract and implementing `NissaHub.reset()` (and global `reset_hub()`) to properly manage backend lifecycles. Backends are now started on addition and closed on reset/shutdown.
- **Links:** `src/esper/nissa/output.py` (backend API), `src/esper/nissa/__init__.py` (singleton), `src/esper/karn` backends (store/TUI)
