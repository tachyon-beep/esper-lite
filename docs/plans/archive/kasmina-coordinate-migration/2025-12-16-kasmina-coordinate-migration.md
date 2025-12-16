# Work Package: Kasmina Stability Fixes + Migration to Seed Coordinate System

**Status:** Draft (pending implementation plan)
**Priority:** High
**Affects:** Kasmina, Leyline, Tolaria, Simic, Tamiyo, Nissa, Karn

---

## Goal

Make Kasmina seed lifecycle robust under PyTorch 2.9 and Python 3.13, eliminate known correctness traps, and migrate slot addressing from `early/mid/late` to a canonical 2D coordinate system (`"r{row}c{col}"`).

## Architectural Decisions (Resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Migration strategy | **Clean break** | Pre-release software, no tech debt accumulation |
| Grid topology | **Full 2D grid (rXcY)** | Enables future multi-row hierarchical seeds |
| Gated blending | **Blending-only, then integrate** | Clear lifecycle: gate active during BLENDING, discarded on completion |
| CLI format | **Canonical IDs only** | `--slots r0c0 r0c1` — no legacy aliases |
| Action space | **Dynamic** | `SlotAction` becomes runtime-configured, not fixed enum |

## Non-goals

* No new seed algorithms beyond what exists.
* No DDP redesign beyond preserving existing behaviour.
* No backwards compatibility with `early/mid/late` naming (clean break).

## Constraints

* PyTorch 2.9
* Python 3.13
* Pre-1.0: remove "pretend support" paths rather than leaving broken fallbacks.

---

## Migration Surface (32 Source Files + 15 Test Files)

The `early/mid/late` naming is embedded across the codebase. This is the complete inventory:

### Leyline (Contracts)
- `src/esper/leyline/factored_actions.py` — **CRITICAL: SlotAction enum, NUM_SLOTS constant**
- `src/esper/leyline/signals.py`
- `src/esper/leyline/stages.py`
- `src/esper/leyline/__init__.py`

### Kasmina (Core)
- `src/esper/kasmina/slot.py`
- `src/esper/kasmina/host.py` — `segment_channels`, `_slot_order`
- `src/esper/kasmina/blending.py`
- `src/esper/kasmina/isolation.py`
- `src/esper/kasmina/blueprints/registry.py`
- `src/esper/kasmina/blueprints/cnn.py`

### Simic (RL Infrastructure)
- `src/esper/simic/features.py` — `TaskConfig`
- `src/esper/simic/ppo.py` — checkpoint save/load
- `src/esper/simic/training.py`
- `src/esper/simic/vectorized.py` — checkpoint resume
- `src/esper/simic/rewards.py`
- `src/esper/simic/slots.py`
- `src/esper/simic/action_masks.py`
- `src/esper/simic/config.py`
- `src/esper/simic/anomaly_detector.py`
- `src/esper/simic/telemetry_config.py`

### Tamiyo (Heuristic Controller)
- `src/esper/tamiyo/heuristic.py`
- `src/esper/tamiyo/tracker.py`

### Tolaria (Training Engine)
- `src/esper/tolaria/environment.py`
- `src/esper/tolaria/trainer.py`

### Nissa (Telemetry)
- `src/esper/nissa/tracker.py`

### Karn (Analytics/Dashboard)
- `src/esper/karn/tui.py`
- `src/esper/karn/collector.py`
- `src/esper/karn/analytics.py`
- `src/esper/karn/counterfactual.py`
- `src/esper/karn/counterfactual_helper.py`
- `src/esper/karn/health.py`
- `src/esper/karn/store.py`

### Scripts
- `src/esper/scripts/train.py` — CLI `--slots` argument

### Tests (15+ files)
- `tests/conftest.py`
- `tests/strategies.py`
- `tests/test_simic_rewards.py`
- `tests/test_nissa_analytics.py`
- `tests/test_tolaria_governor.py`
- `tests/test_kasmina_telemetry.py`
- `tests/test_utils_loss.py`
- `tests/test_morphogenetic_model.py`
- `tests/properties/test_pbrs_telescoping.py`
- `tests/integration/test_sparse_training.py`
- `tests/integration/test_vectorized_factored.py`
- `tests/integration/test_tamiyo_kasmina.py`
- `tests/integration/test_tamiyo_tolaria.py`
- `tests/integration/test_tamiyo_simic.py`
- `tests/integration/test_max_seeds_wiring.py`

---

## Milestone 0: Sanity and File Hygiene (Hard Fails)

### M0.1 Confirm module boundaries and eliminate paste-merge hazards

**Problem:** If any of these pasted chunks are physically in one `.py`, `from __future__ import annotations` will not be valid and there is at least one obvious syntax collision (`]"""Kasmina Protocol...`).

**Tasks:**

- [ ] Verify each logical module is a separate file:
  - `esper/kasmina/__init__.py`
  - `esper/kasmina/blending.py`
  - `esper/kasmina/host.py`
  - `esper/kasmina/isolation.py`
  - `esper/kasmina/protocol.py`
  - `esper/kasmina/slot.py`
- [ ] Ensure `from __future__ import annotations` appears only at the top of files where used.
- [ ] Run `python -m py_compile` (or `python -m compileall`) on the package.

**Acceptance:**

- No syntax errors.
- Imports succeed cleanly.

---

## Milestone 1: Slot Identity Migration to Coordinates

### M1.1 Introduce canonical slot id format

**Decision:** Full 2D grid support with canonical string IDs: `"r{row}c{col}"` (example `"r1c3"`).

**Tasks:**

- [ ] Create `esper/leyline/slot_id.py` with:
  - `def format_slot_id(row: int, col: int) -> str`
  - `def parse_slot_id(slot_id: str) -> tuple[int, int]`
  - `def slot_sort_key(slot_id: str) -> tuple[int, int]` — for deterministic ordering
  - `def validate_slot_id(slot_id: str) -> bool`
- [ ] **NO legacy alias mapping** — clean break, `"early"` is an error not a redirect.
- [ ] Add validation at module boundaries (CLI, config loading) with clear error messages.

**Acceptance:**

- `SeedState.slot_id` always stores canonical `"rXcY"`.
- Attempting to use `"early"` raises `ValueError` with guidance to use `"r0c0"`.

---

### M1.2 Host API: expose injection points using canonical slot ids

Right now there are two overlapping concepts: "segments" and "injection points". The coordinate system should make injection points the primary concept.

**Tasks:**

- [ ] Define `HostProtocol` (or ABC) with required interface:
  - `injection_points: dict[str, int]` — canonical slot id → channel dimension
  - `slot_order: tuple[str, ...]` — canonical ids in forward execution order
- [ ] Update `CNNHost`:
  - Replace `segment_channels` with `injection_points` using canonical IDs
  - Add `slot_order` property
  - Remove `forward_to_segment` / `forward_from_segment`
- [ ] Update `TransformerHost` similarly.
- [ ] Grid-aware injection point generation:
  - For n_blocks, compute valid (row, col) coordinates
  - Row = depth tier (0 = shallow, 1 = mid, 2 = deep)
  - Col = position within tier

**Acceptance:**

- No host method accepts `early/mid/late`.
- Hosts provide deterministic `slot_order`.
- Protocol enforces interface contract.

---

### M1.3 Replace segment-based routing with slot-based routing

**Tasks:**

- [ ] Replace host methods with:
  - `forward_to(slot_id: str, x: Tensor, from_slot_id: str | None = None) -> Tensor`
  - `forward_from(slot_id: str, x: Tensor) -> Tensor`
- [ ] Implement for CNNHost and TransformerHost:
  - `forward_to` runs from network input (or from after `from_slot_id`) up to the injection boundary for `slot_id`
  - `forward_from` runs from after `slot_id` to output
- [ ] **Delete** `forward_to_segment` / `forward_from_segment` entirely (clean break).

**Acceptance:**

- MorphogeneticModel does not reference "segments".
- All routing uses canonical slot ids.

---

### M1.4 MorphogeneticModel: dynamic slot support

**Problem:** Currently hardcodes `["early", "mid", "late"]` and assumes exactly 3 slots.

**Tasks:**

- [ ] In `MorphogeneticModel.__init__`:
  - Accept `slots: list[str]` with canonical IDs
  - Validate each ID against `host.injection_points`
  - Compute active slot order via `slot_sort_key()` (row-major: r0c0 < r0c1 < r1c0)
- [ ] Update `MorphogeneticModel.forward` to iterate slot ids in sorted order.
- [ ] Remove hardcoded `_slot_order = ["early", "mid", "late"]`.

**Acceptance:**

- Works with arbitrary grid slots.
- Adding a slot like `"r1c3"` requires no code changes (just host support).

---

### M1.5 Remove broken fallbacks

**Problem:** CNNHost has fallback segment naming that is internally inconsistent.

**Tasks:**

- [ ] Delete the `n_blocks < 3` segment fallback code path.
- [ ] Enforce minimum block requirements clearly:
  - `n_blocks >= 2` required for at least one injection point
  - Document which canonical slot ids exist for each block count

**Acceptance:**

- No dead code paths that appear to support something but crash or silently no-op.

---

## Milestone 1.5: FactoredActions / RL Action Space Migration (NEW)

**Problem:** The current `SlotAction` is a fixed 3-element enum. Full 2D grid requires dynamic action space.

### M1.5.1 Replace SlotAction enum with dynamic slot configuration

**Tasks:**

- [ ] Remove `SlotAction(IntEnum)` from `factored_actions.py`.
- [ ] Add `SlotConfig` class:
  ```python
  @dataclass
  class SlotConfig:
      slot_ids: tuple[str, ...]  # e.g., ("r0c0", "r0c1", "r0c2")

      @property
      def num_slots(self) -> int:
          return len(self.slot_ids)

      def slot_id_for_index(self, idx: int) -> str:
          return self.slot_ids[idx]
  ```
- [ ] Update `FactoredAction` to accept slot index + config reference.
- [ ] Replace `NUM_SLOTS = len(SlotAction)` with `slot_config.num_slots`.

### M1.5.2 Update PPO policy network for dynamic action dimensions

**Tasks:**

- [ ] Modify `TamiyoNetwork` to accept `slot_config` at construction.
- [ ] Policy head output dimension becomes `slot_config.num_slots` (not hardcoded 3).
- [ ] Update action sampling and log_prob computation.

### M1.5.3 Update all FactoredActions consumers

**Files to update:**
- `simic/ppo.py`
- `simic/training.py`
- `simic/vectorized.py`
- `simic/action_masks.py`
- `simic/slots.py`

**Acceptance:**

- PPO training works with 2, 3, or N slots.
- Action masking respects dynamic slot count.
- No hardcoded assumptions about 3 slots.

---

## Milestone 2: Lifecycle and Blending Correctness

### M2.1 Fix gated blending vs lifecycle bookkeeping

**Decision:** Gated blending is **BLENDING-only, then integrates**.

**Problem:** `GatedBlend.get_alpha(step)` returns meaningless 0.5 but lifecycle uses it to update `state.alpha`.

**Tasks:**

- [ ] Implement BLENDING-only semantics:
  - During BLENDING: `alpha = gate(x)` computes per-sample blend weight
  - On BLENDING completion (→ PROBATIONARY):
    - Set `alpha_schedule = None`
    - Set `state.alpha = 1.0`
    - **Discard** the gate module (don't serialize it)
- [ ] Update `GatedBlend.get_alpha()` to raise `NotImplementedError` or return `state.alpha` for inspection.
- [ ] Ensure G3 gate logic uses actual integration state, not gated alpha.

**Acceptance:**

- Whatever is reported as "alpha" matches what forward actually uses.
- Stage transitions reflect real behaviour.
- Gated blend module not persisted after BLENDING completes.

---

### M2.2 Unify `advance_stage()` with `step_epoch()` special handling

**Problem:** `advance_stage()` can move TRAINING to BLENDING without initialising blending schedule or setting `_blending_started`.

**Tasks:**

- [ ] Create `_on_enter_stage(new_stage: SeedStage, old_stage: SeedStage)` method.
- [ ] Route both `advance_stage()` and `step_epoch()` through this method.
- [ ] Ensure TRAINING → BLENDING always:
  - Sets `_blending_started = True`
  - Snapshots `accuracy_at_blending_start`
  - Initialises blending schedule and `blending_steps_total/done`
  - Calls `start_blending()`

**Acceptance:**

- Stage transitions behave identically regardless of which method triggers them.

---

## Milestone 3: Gradient Isolation Monitoring Correctness

### M3.1 Define and measure isolation correctly

**Problem:** Current logic flags a violation if `host_grad_norm > 1e-6` while `isolate_gradients=True`. But host gradients via direct loss path are expected.

**Tasks:**

- [ ] Define isolation invariant: "No gradients from seed path flow into host parameters" (structural via detach).
- [ ] Remove numeric violation detection (detach is the guarantee).
- [ ] Rename monitoring to reflect actual purpose:
  - `GradientHealthMonitor` (not "isolation")
  - Report: seed grad norm, host grad norm, ratio (for G2 gate health)
- [ ] Remove `violations` counter from `GradientIsolationMonitor`.

**Acceptance:**

- No spurious violations during normal training.
- Monitoring outputs remain useful for G2 gate decisions.

---

### M3.2 Harden `torch._foreach_norm` usage

**Problem:** Private API risk.

**Tasks:**

- [ ] Create `_batched_grad_norm(grads: list[Tensor]) -> Tensor` helper.
- [ ] Try `torch._foreach_norm`, fallback to `torch.stack([g.norm() for g in grads])`.
- [ ] Single implementation point, no duplication.

**Acceptance:**

- Works regardless of private API availability.

---

## Milestone 4: Checkpoint Compatibility (PyTorch 2.9 weights-only)

### M4.1 Make checkpoints loadable with default `torch.load` behaviour

**Problem:** PyTorch 2.9 defaults `weights_only=True`, rejecting arbitrary Python objects.

**Current problematic state in `SeedSlot.get_extra_state()`:**
- `SeedState` dataclass with `datetime`, `deque`, `SeedStage` enum
- `alpha_schedule` (potentially a module)

**Tasks:**

- [ ] Redesign `SeedSlot.get_extra_state()` to return only primitives:
  ```python
  {
      "seed_state": {
          "seed_id": str,
          "blueprint_id": str,
          "slot_id": str,
          "stage": str,  # enum name, not enum
          "stage_entered_at": float,  # timestamp, not datetime
          "alpha": float,
          "metrics": {...},  # all primitives
          "blending_steps_done": int,
          "blending_steps_total": int,
      },
      "alpha_schedule_config": {
          "algorithm_id": str,
          "total_steps": int,
          "current_step": int,
          # steepness etc. as needed
      } | None,
      "isolate_gradients": bool,
  }
  ```
- [ ] For gated blending: gate module weights live in `state_dict` (registered submodule), not `extra_state`.
- [ ] Update `set_extra_state()` to reconstruct from primitives.
- [ ] Remove `weights_only=False` from checkpoint loading in `simic/ppo.py` and `simic/vectorized.py`.

**Acceptance:**

- Save/load roundtrip works with default `torch.load(path)`.
- No need for `weights_only=False` in normal workflows.

---

## Milestone 5: Test Suite Migration

**Scope:** 15+ test files need updates. This is not "add new tests" — it's a full test suite migration.

### M5.1 Update test fixtures and strategies

- [ ] Update `tests/conftest.py` — replace all `"early"/"mid"/"late"` with canonical IDs
- [ ] Update `tests/strategies.py` — Hypothesis strategies for slot generation

### M5.2 Update unit tests

- [ ] `tests/test_morphogenetic_model.py`
- [ ] `tests/test_kasmina_telemetry.py`
- [ ] `tests/test_simic_rewards.py`
- [ ] `tests/test_nissa_analytics.py`
- [ ] `tests/test_tolaria_governor.py`
- [ ] `tests/test_utils_loss.py`
- [ ] `tests/properties/test_pbrs_telescoping.py`

### M5.3 Update integration tests

- [ ] `tests/integration/test_vectorized_factored.py` — **CRITICAL: RL action space**
- [ ] `tests/integration/test_tamiyo_kasmina.py`
- [ ] `tests/integration/test_tamiyo_tolaria.py`
- [ ] `tests/integration/test_tamiyo_simic.py`
- [ ] `tests/integration/test_max_seeds_wiring.py`
- [ ] `tests/integration/test_sparse_training.py`

### M5.4 New tests for coordinate system

- [ ] `tests/leyline/test_slot_id.py` — format, parse, validate, sort_key
- [ ] `tests/kasmina/test_dynamic_slots.py` — 2, 3, 4+ slot configurations
- [ ] `tests/simic/test_dynamic_action_space.py` — PPO with variable slot counts

### M5.5 Checkpoint compatibility tests (PyTorch 2.9)

- [ ] Save checkpoint
- [ ] Load with default `torch.load` (no `weights_only=False`)
- [ ] Resume training for one step
- [ ] Verify state reconstruction correctness

**Acceptance:**

- All tests pass.
- CI covers new test files.

---

## Milestone 6: CLI and Documentation

### M6.1 Update CLI arguments

**File:** `src/esper/scripts/train.py`

**Tasks:**

- [ ] Change `--slots` from `choices=["early", "mid", "late"]` to validated canonical IDs.
- [ ] Add helpful error message: `"Invalid slot 'early'. Use canonical format: r0c0, r0c1, r0c2, ..."`
- [ ] Update `--help` text with examples.

### M6.2 Update documentation

- [ ] Update `README.md` CLI examples.
- [ ] Update any docstrings referencing `early/mid/late`.

**Acceptance:**

- CLI rejects legacy slot names with clear guidance.
- Documentation reflects new coordinate system.

---

## Recommended Implementation Order

1. **M0** — hygiene (so you are not debugging ghosts)
2. **M4** — checkpoint fixes (biggest practical footgun under PyTorch 2.9)
3. **M2** — gated blending semantics + transition consistency
4. **M3** — isolation monitoring correctness
5. **M1** — coordinate system in Kasmina core
6. **M1.5** — FactoredActions / RL action space migration
7. **M6** — CLI and documentation
8. **M5** — test suite migration (runs throughout, finalized last)

---

## Confidence Assessment (WEP)

| Issue | Confidence | Notes |
|-------|------------|-------|
| Checkpoint extra_state needs redesign for PyTorch 2.9 | Almost certain | Default `weights_only=True` will reject current format |
| Gated blending misrepresents integration state | Very likely | Unless lifecycle is aligned with effective alpha |
| Isolation violations are noisy/wrong | Very likely | Unless host is frozen during isolation |
| FactoredActions hardcodes 3 slots | Confirmed | `SlotAction` enum is fixed, blocks grid expansion |
| Migration surface is 32+ files | Confirmed | Grep found 32 source + 15 test files |

---

## Dependencies

```
M0 (hygiene)
 └── M4 (checkpoints) ─────────────────┐
      └── M2 (gated blending) ─────────┤
           └── M3 (isolation) ─────────┤
                └── M1 (coordinates) ──┤
                     └── M1.5 (RL) ────┤
                          └── M6 (CLI) ┤
                               └── M5 (tests) ← runs throughout
```

M2 (gated blending decision) affects M4 (what to serialize).
M1 (coordinate system) must complete before M1.5 (RL action space).
M5 (tests) is iterative — update tests as each milestone completes.
