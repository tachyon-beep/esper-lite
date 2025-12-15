# JANK Template

- **Title:** SeedSlot extra_state stores live objects (SeedState/alpha_schedule), risking mutation/stale devices
- **Category:** correctness-risk / checkpoint integrity
- **Symptoms:** `get_extra_state`/`set_extra_state` (`src/esper/kasmina/slot.py`) persist `seed_state` and `alpha_schedule` by reference. If `set_extra_state` is called with objects tied to a different device/graph, or if callers mutate the returned dict, slot state can become inconsistent (wrong device, shared state across slots) after load.
- **Impact:** Medium – checkpoint restores may reattach stale/live objects with incorrect device; mutations outside slot can affect restored state; alpha_schedule may not move to slot device.
- **Triggers:** Loading checkpoints across devices, reusing extra_state dicts, or using deep-copy-unaware serialization.
- **Root-Cause Hypothesis:** Extra state stored as live objects for convenience; no deep copy or device reconciliation.
- **Remediation Options:**
  - A) Deep-copy or re-instantiate SeedState/alpha_schedule on restore; rebind to slot device.
  - B) Validate device compatibility and detach references when setting extra state.
  - C) Document that extra_state must not be mutated and add tests for device-safe restore.
- **Validation Plan:** Add a test loading extra_state on a different device and ensure seed/alpha are correctly placed and isolated.
- **Status:** Open
- **Links:** `src/esper/kasmina/slot.py::get_extra_state/set_extra_state`, device move concerns
