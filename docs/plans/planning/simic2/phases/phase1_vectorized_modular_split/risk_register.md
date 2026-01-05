# Phase 1 Risk Register (Vectorized Refactor)

This is a short, tactical risk register for the Phase 1 decomposition.

## Risks

### R1: Performance regression from extra GPU↔CPU syncs

- **Severity:** High
- **Likelihood:** Medium
- **Where it bites:** per-step action/telemetry blocks (`src/esper/simic/training/vectorized.py:2863`)
- **Mitigation:**
  - Preserve the “one batched transfer” pattern (stack heads/log-probs → one `.cpu()`).
  - Keep `.item()` calls behind an epoch-end stream sync barrier.
  - Add a “throughput smoke” manual check (short run) after the refactor.

### R2: Correctness drift from subtle ordering changes

- **Severity:** High
- **Likelihood:** Medium
- **Where it bites:** buffer write ordering, bootstrap targets, reward attribution wiring
- **Mitigation:**
  - Split work into two mechanical steps:
    1) move nested helpers to module scope (no new modules yet),
    2) move those helpers into new modules.
  - Run the fast guardrail suite after each step (`docs/plans/planning/simic2/phases/phase0_baseline_and_tests/baseline_capture.md:73`).

### R3: Import-cycle regression / import-isolation failure

- **Severity:** Medium
- **Likelihood:** Medium
- **Where it bites:** new modules importing runtime/simic at import-time
- **Mitigation:**
  - Keep `esper.runtime.get_task_spec` lookup lazy (see `docs/plans/planning/simic2/phases/phase0_baseline_and_tests/baseline_capture.md:112`).
  - Do not add imports of vectorized modules to `src/esper/simic/training/__init__.py`.
  - Keep `tests/test_import_isolation.py` in the always-run PR suite.

### R4: Test breakage from lost monkeypatch seams

- **Severity:** Medium
- **Likelihood:** High
- **Where it bites:** tests patch `vectorized.get_hub`, `vectorized.RewardNormalizer`, `vectorized.SharedBatchIterator` (see baseline capture).
- **Mitigation:**
  - Keep these symbols referenced in `src/esper/simic/training/vectorized.py` and pass dependencies into the new trainer object.
  - If we decide to move the seam, update the tests in the same PR (no dual path).

