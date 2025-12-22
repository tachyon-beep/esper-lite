# BUG-023: Slot “emptiness” semantics differ (mask uses `stage==DORMANT`, env uses `state is None`)

- **Title:** `build_slot_states()` treats `report.stage==DORMANT` as empty, but Kasmina/Simic treat a slot as available only when `slot.state is None`
- **Context:** Phase 4 cooldown pipeline retains `state` in `PRUNED/EMBARGOED/RESETTING` while `seed=None`; availability is “no state”, not “no active seed”
- **Impact:** P3 – latent mismatch in corrupted/unusual states (e.g., checkpoint with `state.stage==DORMANT` but `state` present); masks could allow `GERMINATE` that execution will reject
- **Environment:** HEAD @ workspace
- **Reproduction Steps:**
  1. Construct a slot report with `stage=DORMANT` but with a non-`None` state present (e.g., via a bad checkpoint or manual test mutation).
  2. `build_slot_states` returns `None` (empty), enabling `GERMINATE`.
  3. Execution rejects germination because `slot.state is not None`.
- **Expected Behavior:** Masking should align with the real availability contract: “empty” iff the slot has no state (`slot.state is None`), not by stage label.
- **Observed Behavior:** `build_slot_states` special-cases `stage==DORMANT` as empty.
- **Logs/Telemetry:** None; typically appears as repeated failed germinate attempts.
- **Hypotheses:** Legacy/defensive handling from pre-Phase-4 semantics where “DORMANT stage” might have been represented as an explicit state.
- **Fix Plan:** Treat a slot as empty iff there is no report (i.e., `report is None`). If a report exists with `stage==DORMANT`, treat as non-empty and optionally assert/log because it should not happen in healthy runs.
- **Validation Plan:** Unit test that a non-`None` report (even if stage==DORMANT) is treated as occupied for germination masking.
- **Status:** Open
- **Links:**
  - Mask builder: `src/esper/tamiyo/policy/action_masks.py:85`
  - Execution availability: `src/esper/simic/training/vectorized.py:2342`
  - Kasmina state clear: `src/esper/kasmina/slot.py:2133`

