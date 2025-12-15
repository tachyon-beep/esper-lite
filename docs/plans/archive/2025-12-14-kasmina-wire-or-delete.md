# Kasmina: “Wire or Delete” + Bug Fixes Plan (2025-12-14)

**Status:** ✅ Completed (2025-12-15)

## Context

Kasmina (`src/esper/kasmina/`) matches the intended architecture (slots, lifecycle, blueprints, gradient isolation), but the audit found three concrete issues:

1. **Dead-but-important contracts**: `SeedStateReport` / `SeedMetrics` in Leyline exist as the intended “Kasmina → Tamiyo” interface, but are not used anywhere, and are missing critical fields (e.g. counterfactual contribution, gradient activity).
2. **True dead code**: a small set of methods/functions with **zero call sites** (not used by runtime *or* tests) that should be deleted under the repo’s “no legacy code” policy.
3. **Real bugs / inconsistencies**: CNN segmented forwarding currently ignores registered injection-point modules, making segmented execution diverge from full `CNNHost.forward()` semantics.

This plan turns the findings into small, reviewable PRs that:
- **Wire** existing Kasmina capabilities into real consumers where it improves correctness/observability, and
- **Delete** code that is demonstrably dead or a no-op knob, and
- **Fix** correctness bugs discovered during wiring.

## Ground Rules

- **No legacy / no backwards compatibility**: when we remove or change an API, we delete the old code completely and update all call sites and tests in the same PR.
- **Subsumption rule (no redundancy)**: if a new system or interface *subsumes* an existing one, we **remove the older system** immediately. No feature flags, no “keep both for safety”, no deprecated aliases, no parallel code paths.
- **One seed per slot** remains the invariant; avoid adding APIs that suggest multi-seed-per-slot unless we’re explicitly changing that model.
- **No `hasattr()` additions** without explicit operator authorization (see `CLAUDE.md`).

## Deliverable

After completing this plan:

1. **CNN segmented forward is semantics-preserving**: `forward_to_segment()` + `forward_from_segment()` matches `CNNHost.forward()` even when injection-point slots are registered.
2. **Leyline reporting is real and complete**: `SeedStateReport` is usable for external consumers and includes the fields Tamiyo/Simic actually reason about (counterfactual contribution, gradient activity, etc.).
3. **At least one runtime consumer uses reports**: Simic feature extraction + masking (and optionally Tamiyo) consumes `SeedStateReport` instead of reaching into Kasmina internals.
4. **No true dead Kasmina code remains**: every module/class/function in `src/esper/kasmina/` is either:
   - used by runtime, or
   - used by tests as an intentional contract, or
   - deleted.
5. **Tests lock in invariants**: add/adjust tests so future drift reintroducing dead code or segment/forward inconsistency is caught immediately.

## Work Plan (small PRs)

### Risk / Complexity Scale

- **Complexity**
  - **LOW**: localized change, ≤2 modules, minimal call-site updates.
  - **MED**: touches core logic or shared contracts; requires coordinated updates across several files/tests.
  - **HIGH**: cross-subsystem refactor affecting hot paths or invariants; requires careful sequencing and stronger test coverage.
- **Risk**
  - **LOW**: low likelihood of regressions; failures are obvious and caught by existing tests.
  - **MED**: moderate regression potential; may change behavior in subtle ways but is testable.
  - **HIGH**: high regression potential or hard-to-debug failure modes (e.g., masking correctness, training dynamics, hot-path performance).

### PR0 — Fix CNNHost segmented forward to apply injection-point slots (bug fix)

**Why**
- `TransformerHost` segmented forward applies registered injection-point modules; `CNNHost` does not.
- This causes full forward vs segmented forward to diverge whenever `host.register_slot()` is used.

**Changes**
- In `src/esper/kasmina/host.py` (CNNHost):
  - Update `forward_to_segment()` and `forward_from_segment()` to apply `self.slots[...]` at the same injection points as `forward()`.
  - Ensure semantics: when `from_segment` is set, inputs are assumed to already be “post-slot” for that segment boundary (mirroring current segment index skipping).

**Acceptance criteria**
- Add a test similar to `tests/test_host.py::test_segment_round_trip_matches_forward` but for CNN:
  - Register a non-identity slot at `"block2_post"` (e.g., multiply by 2).
  - Assert `host(x)` equals `host.forward_from_segment("mid", host.forward_to_segment("mid", x))` in `eval()` mode.

**Tests**
- `uv run pytest tests/kasmina/test_host.py -m "not slow"`
- `uv run pytest tests/test_host_protocol.py -m "not slow"` (ensure register/unregister tests still pass)

**Risk / Complexity**
- **Complexity: MED** — touches core forward-path semantics and segment boundary logic.
- **Risk: MED** — failures can be subtle (wrong slot application order), but are deterministic and testable.
- **Primary risks**
  - Segment round-trip diverges from `forward()` due to missing/double slot application.
  - Off-by-one errors when `from_segment` is set (applying slots at wrong boundary).
  - `n_blocks > 3` behavior drift (segment mapping assumes early/mid/late only).
- **Mitigations**
  - Add the “slot registered” round-trip equality test (eval mode).
  - Add at least one test that uses `from_segment="early"` to hit the partial-range path.
  - Keep the implementation as close as possible to the existing `forward()` loop (shared indices/keys).

---

### PR1 — Make Leyline `SeedStateReport` / `SeedMetrics` include the metrics we actually use (activate the contract)

**Why**
- `SeedStateReport` is currently unused partly because it cannot represent key lifecycle decisions:
  - **Counterfactual contribution** (required for G5 fossilization) is missing.
  - **Gradient activity** (`seed_gradient_norm_ratio`) that gates TRAINING→BLENDING is missing.
  - Some “seed identity + blending progress” fields are missing or split across internal-only Kasmina structures.

**Changes**
- In `src/esper/leyline/reports.py`:
  - Extend `SeedMetrics` to include:
    - `counterfactual_contribution: float | None`
    - `seed_gradient_norm_ratio: float`
    - `seed_param_count: int`
    - `host_param_count: int`
    - (optional) `accuracy_at_blending_start: float` if used elsewhere
    - (optional) `blending_steps_done: int`, `blending_steps_total: int` if we want progress visibility without reaching into Kasmina state
- In `src/esper/kasmina/slot.py`:
  - Update `SeedMetrics.to_leyline()` to populate the new Leyline fields.
  - Ensure any fields that are authoritative in Kasmina remain authoritative (no duplicate sources of truth).

**Acceptance criteria**
- New fields are populated end-to-end for a germinated seed in tests (set values, convert to report, assert equality).
- No other subsystem needs to touch `kasmina.SeedState` to get counterfactual/gradient readiness metrics.

**Tests**
- Add focused unit tests in `tests/kasmina/` or `tests/test_seed_slot.py`:
  - “Report includes counterfactual contribution”
  - “Report includes seed_gradient_norm_ratio / param counts”
- Run: `uv run pytest -m "not slow"`

**Risk / Complexity**
- **Complexity: MED** — updates a shared contract and a conversion path.
- **Risk: MED** — contract changes can ripple into consumers (now and future), but are mostly structural and easy to validate.
- **Primary risks**
  - Incomplete/incorrect mapping in `SeedMetrics.to_leyline()` (fields silently default).
  - Contract drift between Leyline and Kasmina (two sources of truth).
  - Unintended downstream effects if any code serializes/pickles Leyline reports.
- **Mitigations**
  - Add explicit “populate and assert” unit tests for every new field.
  - Treat Kasmina as the only source of truth; Leyline report remains a snapshot.
  - Prefer additive-only changes in this PR; do wiring/deletion in PR2.

---

### PR2 — Wire reports into Simic observation + action masking (functional improvement + dead-code elimination)

**Why**
- Simic currently reaches into `model.seed_slots[slot].state.metrics...` in multiple places.
- This couples Simic to Kasmina’s mutable internal dataclasses and makes it easier for “report contracts” to rot.
- Wiring reports will also make the currently-dead `SeedSlot.get_state_report()` / `SeedState.to_report()` / `SeedMetrics.to_leyline()` real.

**Changes**
- In `src/esper/kasmina/host.py`:
  - Replace the unused `MorphogeneticModel.get_slot_states()` with a useful, report-shaped API, e.g.:
    - `get_slot_reports(enabled_slots: list[str] | None = None) -> dict[str, SeedStateReport | None]`
  - Delete `get_slot_states()` (no call sites today) — do not keep both APIs.
- In `src/esper/simic/ppo.py`:
  - Update `signals_to_features()` to build per-slot dicts from `model.get_slot_reports()` instead of direct state access, and then delete any legacy “read directly from kasmina state” branches (no dual paths).
  - When computing “improvement” for features:
    - prefer `counterfactual_contribution` if available, else fall back to `improvement_since_stage_start`.
- In `src/esper/simic/action_masks.py`:
  - Update `build_slot_states()` to use `SeedStateReport` rather than direct Kasmina state access, and remove the old internal-state read path (no dual paths).

**Acceptance criteria**
- `signals_to_features()` no longer reads `slot.state.metrics.*` directly.
- `build_slot_states()` no longer reads `seed_slot.state.*` directly.
- `rg -n "get_state_report\\(|to_report\\(|to_leyline\\(" src/esper` shows these methods now have runtime call sites.

**Tests**
- `uv run pytest tests/simic/test_ppo.py -m "not slow"`
- `uv run pytest tests/simic/test_action_masks.py -m "not slow"` (or the closest existing masking test module)
- `uv run pytest -m "not slow"`

**Risk / Complexity**
- **Complexity: HIGH** — cross-subsystem refactor (Kasmina → Simic), touches hot-path feature extraction + masking.
- **Risk: HIGH** — masking/feature bugs can be subtle and may change training dynamics without obvious crashes.
- **Primary risks**
  - Report API introduces extra per-step allocations or Python overhead on the vectorized path (throughput regression).
  - Incomplete report data causes masking to under-mask (invalid actions) or over-mask (degenerate policies).
  - Removing direct-state access reveals missing fields that were relied on implicitly.
  - Ordering hazard: doing PR2 before PR1 can cause “report missing fields” failures.
- **Mitigations**
  - Enforce PR order: PR1 → PR2.
  - Add a regression test that `compute_action_masks()` never produces an all-false op mask for sane states.
  - Keep report construction O(#enabled_slots) and avoid scanning all parameters/telemetry inside it.
  - Add a targeted “feature vector stability” test: same model state yields identical feature vector pre/post refactor.

---

### PR3 — Delete truly dead Kasmina code (no wireable value)

**Why**
- A few APIs have zero call sites and do not meaningfully improve the system if “wired”.
- Keeping them violates the repo’s “no legacy code” stance and increases audit surface area.

**Targets (confirm with `rg` before deleting)**
- `src/esper/kasmina/blueprints/registry.py`:
  - `BlueprintRegistry.reset()` (unused)
- `src/esper/kasmina/host.py`:
  - `MorphogeneticModel.count_seeds_in_slot()` (unused and redundant with one-seed-per-slot invariant)
- `src/esper/kasmina/slot.py`:
  - `SeedState.increment_epoch()` (unused)
  - `SeedState.record_epoch()` (unused)
- `src/esper/kasmina/slot.py`:
  - Remove the unused `temperature` parameter on `SeedSlot.start_blending(...)` **or** wire it meaningfully.
    - Default action: delete the param (today it is always passed as `1.0` and never used).

**Acceptance criteria**
- `rg -n "BlueprintRegistry\\.reset\\(|count_seeds_in_slot\\(|increment_epoch\\(|record_epoch\\(|start_blending\\(.*temperature" src/ tests/` returns nothing.
- All tests pass.

**Tests**
- `uv run pytest -m "not slow"`
- `ruff check src/ tests/` (should also drop any “unused import” fallouts)

**Risk / Complexity**
- **Complexity: LOW** — mostly deletions and signature cleanup.
- **Risk: LOW→MED** — low runtime risk if call sites are correctly updated; medium risk if something was “quietly” relied upon (e.g., tests, checkpoints).
- **Primary risks**
  - Hidden call sites (dynamic imports/tests) for `BlueprintRegistry.reset()` or `count_seeds_in_slot()`.
  - Changing `SeedSlot.start_blending()` signature breaks internal calls or tests.
  - Removing methods used by pickled state in old checkpoints (explicitly not supported, but must be acknowledged).
- **Mitigations**
  - Pre-delete `rg` confirmation for each symbol and run the full non-slow test suite.
  - In the PR description: explicitly state that old checkpoints are not supported if applicable.

---

### PR4 — Optional: Wire `fast_mode` as an explicit performance knob (only if it improves real training throughput)

**Why**
- `SeedSlot.fast_mode` exists but is never set `True` by runtime; it is currently dead as a capability.
- If the PPO vectorized loop is throughput-bound on Python-side telemetry, this can materially improve speed.

**Design decision required (pick one, don’t keep both)**
1. **Keep `fast_mode`** but expose it explicitly:
   - Add CLI/config wiring for PPO vectorized runs: e.g. `--slot-fast-mode` or infer from telemetry profile.
   - Ensure the G2 gradient-readiness signal still works under fast mode (vectorized already computes `seed_gradient_norm_ratio` without `GradientIsolationMonitor`).
2. **Split responsibilities**:
   - Replace `fast_mode` with two explicit booleans: `telemetry_enabled` and `isolation_monitor_enabled`.
   - This avoids conflating “no telemetry” with “no gradient monitor”.

**Acceptance criteria**
- There is a clear, documented knob, and it measurably changes runtime behavior (less telemetry overhead and/or fewer sync points).
- The alternative approach (the system not chosen) is deleted completely (no redundant knobs or parallel code paths retained “just in case”).
- G2 gate still functions in both non-vectorized (Tolaria) and vectorized (Simic) training paths.

**Tests**
- Add a regression test to ensure G2 can still pass in vectorized mode when fast mode is enabled (mock/short run).
- Run: `uv run pytest -m "not slow"`

**Risk / Complexity**
- **Complexity: MED→HIGH** — depends on the chosen design; touches training loop behavior and potentially telemetry/monitoring semantics.
- **Risk: MED→HIGH** — easy to accidentally disable a signal required for correct lifecycle progression (G2 gating) or observability.
- **Primary risks**
  - Turning on fast mode disables `GradientIsolationMonitor` in non-vectorized training, breaking `capture_gradient_telemetry()` and making G2 always fail.
  - Confusing semantics: “fast mode” conflates telemetry suppression with safety/health monitoring.
  - If implemented via CLI flags, risk of new “dead knobs” if not used/covered by tests.
- **Mitigations**
  - Only proceed if we can demonstrate measurable throughput impact (otherwise delete `fast_mode` entirely).
  - Ensure *both* training paths have a valid G2 signal:
    - vectorized: already sets `seed_gradient_norm_ratio`
    - non-vectorized: must still have a monitor/capture path if G2 is enforced there
  - Add a test that explicitly asserts “G2 can pass with fast mode on” for the chosen path(s).

---

### PR5 — Hardening: fix blueprint edge cases discovered during wiring (small correctness PR)

**Candidates**
- `src/esper/kasmina/blueprints/transformer.py`:
  - Add explicit validation that `dim % n_head == 0` for attention/flex-attention blueprints (avoid silent shape errors if config changes).
- `src/esper/kasmina/host.py`:
  - Remove conditional `override` import logic by always importing `override` from `typing_extensions` (keeps Python 3.11 support without try/except compatibility code).

**Acceptance criteria**
- New validation is covered by a unit test that would previously crash with an unclear reshape error.
- No behavior changes for default configs.

**Tests**
- `uv run pytest tests/kasmina/test_blueprints.py -m "not slow"` (or add a new focused test file)
- `uv run pytest -m "not slow"`

**Risk / Complexity**
- **Complexity: LOW** — input validation and minor import cleanup.
- **Risk: LOW** — failures should be immediate and self-explanatory; mostly reduces debug time.
- **Primary risks**
  - Over-eager validation could reject currently-valid but unusual configs (rare).
  - Changing `override` import behavior could affect type checking or runtime in unexpected environments (unlikely given `typing_extensions` dependency).
- **Mitigations**
  - Validate only where the downstream code would otherwise crash with an unclear shape/reshape error.
  - Add a single negative test that asserts the new error message is clear and specific.

---

## Final Verification Checklist

Run these at the end of the series:

```bash
uv run pytest -m "not slow"
ruff check src/ tests/
mypy src/
rg -n \"get_slot_states\\(|SeedStateReport\\(|to_report\\(|to_leyline\\(\" src/esper
```

Expected:
- Test suite green.
- No unused Kasmina APIs remain unless they are explicitly “test-only contracts”.
- Report contract is used by runtime (at minimum: Simic observation + masking).

**Risk / Complexity**
- **Complexity: LOW** — verification only.
- **Risk: LOW** — main risk is discovering unexpected coupling late; treat failures as signals to split PRs further.
