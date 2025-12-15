# BUG Template

- **Title:** Heuristic embargo/cull cooldown tracked globally, not per slot
- **Context:** Tamiyo heuristic (`src/esper/tamiyo/heuristic.py`) uses `_last_cull_epoch` as a single scalar across all slots.
- **Impact:** P1 – in multi-slot runs, culling a seed in one slot blocks germination in all slots for `embargo_epochs_after_cull`, even if other slots remain empty/healthy. This is unintended global coupling and can stall growth.
- **Environment:** Main branch; heuristic training with `--slots` including multiple slots.
- **Reproduction Steps:**
  1. Run heuristic with `slots=["early","mid"]`, short epochs.
  2. Cull in `early` (e.g., via poor seed); observe `_last_cull_epoch` set.
  3. Germination in `mid` is embargoed despite no cull there.
- **Expected Behavior:** Embargo/cooldown should be per-slot, allowing other slots to germinate if they weren’t just culled.
- **Observed Behavior:** Single global `_last_cull_epoch` throttles all slots equally.
- **Hypotheses:** State is scalar because early versions were single-slot; not updated for multi-slot semantics.
- **Fix Plan:** Track `_last_cull_epoch` per slot (dict) and apply embargo per target slot; ensure germination decision uses slot-aware embargo and blueprint penalties stay scoped.
- **Validation Plan:** Add a multi-slot heuristic test ensuring cull in one slot doesn’t block germination in others; verify embargo still applies per slot.
- **Status:** Open
- **Links:** `src/esper/tamiyo/heuristic.py` (embargo logic in `_decide_germination`, `_cull_seed`)
