# JANK Template

- **Title:** NissaHub singleton retains backends across runs/tests, causing duplicate telemetry
- **Category:** maintainability / correctness-risk
- **Symptoms:** `nissa.get_hub()` returns a module-level singleton; backends added during one training run (e.g., ConsoleOutput, KarnCollector, DirectoryOutput) persist for the process lifetime. Subsequent runs/tests emit duplicate events or write to stale paths unless manually cleared.
- **Impact:** Medium – confusing duplicate logs/JSONL entries, noisy tests, and potential file descriptor leaks in long-lived processes or notebooks that call train multiple times.
- **Triggers:** Invoking `get_hub()` in successive runs without process restart (common in interactive sessions or test suites). No API to reset/clear backends.
- **Root-Cause Hypothesis:** Singleton pattern lacks reset logic; `get_hub()` caches the instance but never clears outputs.
- **Remediation Options:** 
  - A) Add a `reset_hub()` to clear backends/state between runs/tests and use it in scripts/tests.
  - B) Make `get_hub()` idempotent per “session” by tracking run IDs or using weakrefs for backends.
  - C) Provide context manager to isolate telemetry configuration per run.
- **Risks of Change:** Potentially breaks code relying on process-global hub; must coordinate with Karn collector lifecycle.
- **Stopgap Mitigation:** Document the leak and call `hub.reset()` (if added) in tests; avoid reusing processes for multiple runs.
- **Validation Plan:** Add a test that runs two sequential training snippets in one process and asserts no duplicate backend emission after reset.
- **Status:** Closed (Fixed)
- **Resolution:** `reset_hub()` already existed in `src/esper/nissa/output.py` but wasn't exported from `esper.nissa`; now exported and covered by regression test so interactive/test reuse can reliably clear the singleton’s backends.
- **Links:** `src/esper/nissa/output.py`, `src/esper/nissa/__init__.py`, `tests/nissa/test_global_hub_reset.py`
