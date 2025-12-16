# Dynamic Slots Spike Results

**Date:** 2025-12-16
**Spike:** WP-B (Dynamic Action Space)
**Verdict:** GREEN LIGHT for M1.5

---

## Summary

| Test Category | Result | Notes |
|---------------|--------|-------|
| Network Construction | PASS (12/12) | num_slots parameter works for N=1,2,3,4,5,8 |
| Forward/Backward | PASS (2/2) | Gradients flow correctly to slot_head |
| Action Masking | PASS (4/4) | Network handles dynamic mask shapes |
| Action Sampling | PASS (4/4) | get_action works with N slots |
| Action Evaluation | PASS (4/4) | evaluate_actions works with N slots |
| Entropy Computation | PASS (5/5) | max_entropy scales correctly with log(N) |
| Checkpoint Compatibility | PASS (2/2) | Breaking change documented |
| LSTM Hidden State | PASS (4/4) | Hidden state persists and resets correctly |
| Minimal PPO Training | PASS (2/2) | Network-only training loop works |
| Integration (action_masks) | XFAIL (2/2) | Expected blockers documented |
| PPOAgent Integration | SKIP (1/1) | Known blocker, deferred to M1.5 |

**Total: 39 passed, 1 skipped, 2 xfailed in 0.91s**

**Network-level support for dynamic slots: CONFIRMED**

---

## Policy Compatibility Matrix

| Checkpoint Config | Load with N=3 | Load with N=5 | Load with N=8 |
|-------------------|---------------|---------------|---------------|
| Trained N=3       | YES           | NO (arch mismatch) | NO       |
| Trained N=5       | NO            | YES           | NO            |
| Trained N=8       | NO            | NO            | YES           |

**BREAKING CHANGE:** Checkpoints are NOT portable across different num_slots values.
This is expected and documented. Release notes must mention this.

### PyTorch 2.9 Safety Improvement

In PyTorch 2.9+, `strict=False` **still raises RuntimeError** for size mismatches.
This is safer than older PyTorch versions where mismatched tensors were silently skipped.

To do a partial load (NOT RECOMMENDED), you must manually filter the state dict:
```python
filtered = {k: v for k, v in state_dict.items() if "slot_head" not in k}
net.load_state_dict(filtered, strict=False)
```

---

## Hyperparameter Sensitivity (from DRL Specialist)

When `num_slots` changes, these hyperparameters may need adjustment:

| Hyperparameter | Reason | Recommendation |
|----------------|--------|----------------|
| Entropy coefficient | H_max scales with log(num_slots) | Re-tune for N>3 |
| Learning rate | Larger action space | May need adjustment |
| PPO clip epsilon | Factored action ratios | Monitor during training |

**Note:** These are NOT blockers, just tuning considerations for M1.5.

---

## Credit Assignment Note

With more slots (N > 3), the credit assignment problem becomes harder:
- More possible actions = sparser reward signal
- May require longer training or denser reward shaping
- Consider per-slot reward decomposition if needed

---

## Blockers Identified

### Blocker 1: action_masks.py hardcodes NUM_SLOTS

**File:** `src/esper/simic/action_masks.py`
**Lines:** 33-36, 134

```python
from esper.leyline.factored_actions import NUM_SLOTS  # Line 33
...
slot_mask = torch.zeros(NUM_SLOTS, dtype=torch.bool, device=device)  # Line 134
```

**Fix Required:**
- Add `num_slots` parameter to `compute_action_masks()`
- Add `num_slots` parameter to `compute_batch_masks()`
- Pass through from environment/agent configuration

**Complexity:** Low (parameter threading)

---

### Blocker 2: _SLOT_ID_TO_INDEX is static

**File:** `src/esper/simic/action_masks.py`
**Lines:** 43-47

```python
_SLOT_ID_TO_INDEX: dict[str, int] = {
    "early": SlotAction.EARLY.value,
    "mid": SlotAction.MID.value,
    "late": SlotAction.LATE.value,
}
```

**Fix Required:**
- Accept canonical slot IDs (r0c0, r0c1, r0c2, ...)
- Build index mapping from slot_config.slot_ids
- Remove dependency on SlotAction enum

**Complexity:** Medium (depends on WP-C slot_id module)

---

### Blocker 3: factored_actions.py SlotAction enum

**File:** `src/esper/leyline/factored_actions.py`
**Lines:** 16-23, 117

```python
class SlotAction(IntEnum):
    EARLY = 0
    MID = 1
    LATE = 2
...
NUM_SLOTS = len(SlotAction)  # Always 3
```

**Fix Required (M1.5):**
- Replace SlotAction enum with SlotConfig dataclass
- Make NUM_SLOTS dynamic: `slot_config.num_slots`
- Update all consumers to accept slot_config

**Complexity:** High (many consumers)

**Consumers to update:**
```bash
# Run this to find all SlotAction usages:
grep -r "SlotAction" src/ --include="*.py"
```

---

### Blocker 4: PPOAgent mask propagation

**File:** `src/esper/simic/ppo.py`
**Indirect:** Masks come from environment with hardcoded shape

**Fix Required:**
- Add `num_slots` (or `slot_config`) to PPOAgent.__init__
- Pass through to buffer and mask computation
- Update checkpoint save/load for slot_config

**Complexity:** Medium (parameter threading + checkpoint)

---

## Files Requiring Changes for M1.5

| File | Change Type | Complexity |
|------|-------------|------------|
| `leyline/factored_actions.py` | Replace SlotAction enum | High |
| `simic/action_masks.py` | Add num_slots parameter | Medium |
| `simic/ppo.py` | Thread num_slots through | Medium |
| `simic/vectorized.py` | Thread num_slots through | Medium |
| `simic/tamiyo_buffer.py` | Dynamic mask shapes | Low |
| `simic/slots.py` | Use canonical slot IDs | Low |

---

## Recommendation

**Proceed with M1.5 implementation.**

The network layer (FactoredRecurrentActorCritic) already fully supports dynamic slot counts. The blockers are all in the integration layer (action masks, parameter threading) and are straightforward refactoring tasks.

**Risk Level:** LOW — no architectural changes needed, just parameter threading.

**Breaking Change:** Document in release notes that checkpoints are not portable across num_slots changes.

---

## Test Artifacts

Spike tests saved to: `tests/spike_dynamic_slots.py`

To re-run spike:
```bash
PYTHONPATH=src uv run pytest tests/spike_dynamic_slots.py -v
```

Delete after M1.5 implementation is complete and verified.

---

## Appendix: Actual Consumer List for M1.5

**Generated:** 2025-12-16 (from `grep -r "SlotAction|NUM_SLOTS" src/`)

### SlotAction Consumers

| File | Line | Usage |
|------|------|-------|
| `leyline/factored_actions.py` | 16 | `class SlotAction(IntEnum):` — **Definition** |
| `leyline/factored_actions.py` | 67 | `slot: SlotAction` — FactoredAction dataclass field |
| `leyline/factored_actions.py` | 109 | `slot=SlotAction(slot_idx)` — from_indices() |
| `simic/action_masks.py` | 33 | Import |
| `simic/action_masks.py` | 45-47 | `_SLOT_ID_TO_INDEX` mapping uses `SlotAction.EARLY/MID/LATE.value` |
| `simic/action_masks.py` | 228 | Docstring reference |

### NUM_SLOTS Consumers

| File | Line | Usage |
|------|------|-------|
| `leyline/factored_actions.py` | 117 | `NUM_SLOTS = len(SlotAction)` — **Definition** |
| `simic/tamiyo_network.py` | 29 | Import |
| `simic/tamiyo_network.py` | 53 | `num_slots: int = NUM_SLOTS` — Default parameter |
| `simic/tamiyo_buffer.py` | 26 | Import |
| `simic/tamiyo_buffer.py` | 97 | `num_slots: int = NUM_SLOTS` — Default parameter |
| `simic/action_masks.py` | 34 | Import |
| `simic/action_masks.py` | 121 | Docstring reference |
| `simic/action_masks.py` | 131 | `torch.zeros(NUM_SLOTS, ...)` — **HARDCODED USAGE** |

### M1.5 Migration Order (Recommended)

1. **Create `SlotConfig` dataclass** in `leyline/` (WP-C dependency)
2. **Update `leyline/factored_actions.py`**: Deprecate `SlotAction` enum, keep `NUM_SLOTS` as legacy alias
3. **Update `simic/tamiyo_network.py`**: Already parameterized (✅ done)
4. **Update `simic/tamiyo_buffer.py`**: Already parameterized (✅ done)
5. **Update `simic/action_masks.py`**: Add `num_slots` parameter, use `slot_config.slot_ids`
6. **Thread through PPOAgent**: Pass `slot_config` to buffer and mask computation

### Key Insight

The network (`tamiyo_network.py`) and buffer (`tamiyo_buffer.py`) already accept `num_slots` as a parameter with `NUM_SLOTS` as default. The actual work is:
- `action_masks.py` refactoring (hardcoded usage on line 131)
- PPOAgent parameter threading
- Removing `SlotAction` enum dependency from `_SLOT_ID_TO_INDEX`
