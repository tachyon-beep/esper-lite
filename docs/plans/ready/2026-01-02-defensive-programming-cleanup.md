# Defensive Programming Cleanup Sprint

**Created:** 2026-01-02
**Status:** ✅ Ready for Implementation (All Sign-Offs Complete)
**Scope:** Remove bug-hiding patterns that mask integration issues and schema drift

> **Context:** This sprint was triggered after significant debugging time was spent on faulty telemetry masked by a div/0 guard protecting a value that should never have been zero. These defensive patterns create "silent corruption" where training appears healthy while data is fabricated.

## Executive Summary

An audit of the Esper codebase identified **~120 instances** of defensive programming patterns. After triage by specialized Explore agents, we've refined this to **19 confirmed bugs** requiring fixes and **~100 legitimate patterns** that should NOT be changed.

**Key principle:** If a defensive pattern would prevent a crash, that crash is a bug to fix upstream, not a symptom to suppress downstream.

---

## Specialist Sign-Off Status

| Specialist | Bugs Reviewed | Status | Date |
|------------|---------------|--------|------|
| **DRL Expert** | DRL-01 to DRL-05 | ✅ 5/5 CONFIRMED | 2026-01-02 |
| **PyTorch Expert** | PT-01 to PT-07 | ✅ 7/7 CONFIRMED | 2026-01-02 |

**DRL Expert Summary:** *"These bugs collectively create a 'silent corruption' pattern where training appears healthy while phantom zero rewards pollute statistics, TD advantages use fabricated bootstrap values, and NaN gradients go unreported."*

**PyTorch Expert Summary:** *"If data is malformed, it's a bug that should surface immediately. All torch.optim.Optimizer subclasses MUST have `param_groups[0]["lr"]` - if they don't, something is fundamentally broken."*

---

## Triage Results Summary

| Category | Count | Action |
|----------|-------|--------|
| **Confirmed Bugs** | 19 | Fix in this sprint |
| **Legitimate Patterns** | ~100 | No action needed |
| **Framework Quirks** | 2 | Refine exception type only |
| **Redundant Code** | 2 | Clean up |

---

## CONFIRMED BUGS (All Sign-Offs Complete ✅)

### Category A: RL-Critical Telemetry Bugs ✅ (DRL Expert Signed Off)

These affect training signal integrity. **All confirmed as genuine bugs.**

| ID | File | Line | Pattern | Impact | Fix |
|----|------|------|---------|--------|-----|
| **DRL-01** | `karn/sanctum/aggregator.py` | 1304 | `total_reward or 0.0` | Conflates 0.0 reward with missing reward; corrupts TD calculations | Use explicit None check |
| **DRL-02** | `karn/sanctum/aggregator.py` | 1346 | `value_estimate or 0.0` | Conflates 0.0 value with missing estimate; corrupts advantage calculation | Use explicit None check |
| **DRL-03** | `leyline/telemetry.py` | 754 | `nan_grad_count` uses `.get()` | Required field masked by default; violates fail-fast on NaN | Use direct key access |
| **DRL-04** | `leyline/telemetry.py` | 755 | `pre_clip_grad_norm` inconsistent | Dataclass has default but `from_dict()` uses direct access | Align dataclass and deserializer |
| **DRL-05** | `leyline/telemetry.py` | 810 | `ppo_updates_count` inconsistent | Dataclass has default but `from_dict()` uses direct access | Align dataclass and deserializer |

### Category B: Schema/Data Integrity Bugs ✅ (PyTorch Expert Signed Off)

These affect checkpoint/telemetry data integrity. **All confirmed as genuine bugs.**

| ID | File | Line | Pattern | Impact | Fix |
|----|------|------|---------|--------|-----|
| **PT-01** | `karn/store.py` | 741 | `coerce_str_or_none(...) or ""` | Masks validation failure for `action_op` | Remove `or ""` |
| **PT-02** | `karn/store.py` | 847 | `coerce_str_or_none(...) or ""` | Masks validation failure for `gate_id` | Remove `or ""` |
| **PT-03** | `karn/store.py` | 848 | `coerce_str_or_none(...) or ""` | Masks validation failure for `slot_id` | Remove `or ""` |
| **PT-04** | `karn/store.py` | 851 | `coerce_str_or_none(...) or ""` | Masks validation failure for `reason` | Remove `or ""` |
| **PT-05** | `karn/store.py` | 950 | `epoch or data.get("epoch", 0)` | Truthiness masks epoch=0 vs None | Use explicit None check |
| **PT-06** | `karn/store.py` | 655-662 | Conditional `if "initial_checkpoint_path" in data` | Migration debt; violates no-legacy policy | Always coerce field |
| **PT-07** | `simic/telemetry/emitters.py` | 754-756 | 4-exception broad catch for optimizer LR | Over-defensive; masks real bugs | Replace with structure validation |

### Category C: UI/Widget Bugs (General Review)

| ID | File | Line | Pattern | Impact | Fix |
|----|------|------|---------|--------|-----|
| **UI-01** | `karn/sanctum/widgets/scoreboard.py` | 175-176 | `except Exception: return` | Over-broad; should catch `NoMatches` only | Narrow to `NoMatches` |
| **UI-02** | `karn/sanctum/widgets/scoreboard.py` | 208-209 | `except Exception: pass` | Over-broad; should catch `NoMatches` only | Narrow to `NoMatches` |
| **UI-03** | `karn/sanctum/widgets/tamiyo_brain/action_context.py` | 294 | `getattr(d, "success", True)` | Field doesn't exist on DecisionSnapshot | Add field to schema or remove access |

### Category D: Redundant Defensive Code (Cleanup)

| ID | File | Line | Pattern | Impact | Fix |
|----|------|------|---------|--------|-----|
| **RD-01** | `simic/agent/ppo.py` | 842 | `setdefault()` on defaultdict | Redundant; defaultdict already handles | Use direct key access |
| **RD-02** | `kasmina/slot.py` | 2522-2533 | 6× `setdefault()` on fresh dict | Possibly dead code path (all callers use typed payloads) | Verify and simplify |

---

## LEGITIMATE PATTERNS (No Action Needed)

### Telemetry Deserialization - LEGITIMATE

| File | Lines | Pattern | Why Legitimate |
|------|-------|---------|----------------|
| `leyline/telemetry.py` | 480-491 | `TrainingStartedPayload` `.get()` fields | All fields have matching dataclass defaults; distributed training defaults (world_size=1, rank=0) are correct for single-GPU |
| `leyline/telemetry.py` | 521-524 | `EpochCompletedPayload` optional fields | Explicitly Optional; None means "not computed" vs 0.0 means "computed as zero" |

### Checkpoint Deserialization - LEGITIMATE

| File | Lines | Pattern | Why Legitimate |
|------|-------|---------|----------------|
| `karn/store.py` | 760-826 | Silent `.pop()` for Optional fields | Fields are Optional in schema; `.pop()` lets dataclass factory provide default |
| `karn/store.py` | 640-707 | Context/Snapshot coercion | Proper Optional field handling with explicit None assignment |
| `karn/store.py` | 902-911 | `isinstance(data, dict)` checks | Defensive typing for untrusted JSONL input |

### Numeric Truthiness - LEGITIMATE

| File | Line | Pattern | Why Legitimate |
|------|------|---------|----------------|
| `karn/collector.py` | 284 | `n_envs or 1` | 0 envs is invalid; 1 is correct single-env default |
| `karn/collector.py` | 370-372 | `train_loss/accuracy or 0.0` | Optional fields in sum aggregation; None means "not reported" |
| `karn/sanctum/aggregator.py` | 1094 | `seed_params or 0` | Field default is 0; defensive for corrupted state |
| `karn/sanctum/aggregator.py` | 1381 | `decision_entropy or 0.0` | Display/telemetry only; not used in RL calculations |
| `env_detail_screen.py` | 450-525 | Various `or 0`/`or 1.0` | Pure display code; cosmetic fallbacks appropriate |
| `wandb_backend.py` | 245 | `epoch or 0` | Logging/telemetry; 0 is reasonable epoch default |
| `rewards.py` | 1625 | `achievable_range or 1.0` | @property fallback; never actually None |

### Framework/PyTorch Patterns - LEGITIMATE

| File | Line | Pattern | Why Legitimate |
|------|------|---------|----------------|
| `tolaria/governor.py` | 142, 300 | `hasattr(self.model, 'seed_slots')` | Feature detection for MorphogeneticModel vs base model |
| `nissa/output.py` | 538, 550 | `hasattr(self, '_file')` | Cleanup guard in `__del__`; standard Python pattern |
| `tamiyo/policy/lstm_bundle.py` | 334, 341, 481 | `getattr(net, '_orig_mod', net)` | PyTorch torch.compile unwrapping |
| `kasmina/blueprints/initialization.py` | 52 | `getattr(layer, "bias", None)` | NN module initialization; not all layers have bias |
| All defaultdict uses | Various | `defaultdict(list/int/etc)` | Intentional aggregation patterns |
| All `next()` patterns | Various | `next(iter(...), default)` | All properly handled with explicit checks |

---

## Implementation Checklist (By Bug ID)

### Phase 1: DRL-Critical Fixes ✅ (DRL Expert Signed Off)

| Bug ID | Task | File:Line | Est. |
|--------|------|-----------|------|
| DRL-01 | Fix `total_reward or 0.0` truthiness | `aggregator.py:1304` | 15m |
| DRL-02 | Fix `value_estimate or 0.0` truthiness | `aggregator.py:1346` | 15m |
| DRL-03 | Remove `.get()` default on `nan_grad_count` | `telemetry.py:754` | 10m |
| DRL-04 | Align `pre_clip_grad_norm` dataclass/deserializer | `telemetry.py:755` | 20m |
| DRL-05 | Align `ppo_updates_count` dataclass/deserializer | `telemetry.py:810` | 20m |

**Total Phase 1:** ~1.5 hours

### Phase 2: Data Integrity Fixes ✅ (PyTorch Expert Signed Off)

| Bug ID | Task | File:Line | Est. |
|--------|------|-----------|------|
| PT-01 | Remove `or ""` after `action_op` coercion | `store.py:741` | 10m |
| PT-02 | Remove `or ""` after `gate_id` coercion | `store.py:847` | 5m |
| PT-03 | Remove `or ""` after `slot_id` coercion | `store.py:848` | 5m |
| PT-04 | Remove `or ""` after `reason` coercion | `store.py:851` | 5m |
| PT-05 | Fix `epoch or ...` truthiness | `store.py:950` | 10m |
| PT-06 | Remove conditional `initial_checkpoint_path` check | `store.py:655-662` | 15m |
| PT-07 | Replace 4-exception catch with validation | `emitters.py:754-756` | 20m |

**Total Phase 2:** ~1.5 hours

### Phase 3: UI/Widget Fixes (General Review)

| Bug ID | Task | File:Line | Est. |
|--------|------|-----------|------|
| UI-01 | Narrow exception to `NoMatches` | `scoreboard.py:175-176` | 10m |
| UI-02 | Narrow exception to `NoMatches` | `scoreboard.py:208-209` | 10m |
| UI-03 | Add `success` field to DecisionSnapshot OR remove access | `action_context.py:294` | 30m |

**Total Phase 3:** ~1 hour

### Phase 4: Cleanup (No Sign-Off Required)

| Bug ID | Task | File:Line | Est. |
|--------|------|-----------|------|
| RD-01 | Remove redundant `setdefault()` | `ppo.py:842` | 5m |
| RD-02 | Verify/simplify `setdefault()` calls | `slot.py:2522-2533` | 20m |

**Total Phase 4:** ~30 minutes

---

## Specialist Review Protocol ✅ COMPLETE

**DRL Expert reviewed DRL-01 through DRL-05:**
- All patterns confirmed as corrupting training signals
- Recommended fixes: fail-fast on None, remove dataclass defaults for required fields
- No downstream consumers expect fabricated 0.0 values

**PyTorch Expert reviewed PT-01 through PT-07:**
- No checkpoint compatibility impact (these are telemetry ingestion bugs)
- No migration path needed
- Recommended: fail hard on malformed data, create `coerce_str()` that raises on invalid type

---

## Testing Strategy

1. **Before changes:** Run full test suite to establish baseline
2. **After DRL fixes:** Run `pytest tests/simic/ tests/karn/`
3. **After PT fixes:** Run `pytest tests/karn/` + checkpoint load tests
4. **After UI fixes:** Manual Sanctum TUI smoke test
5. **Regression:** Full test suite after all phases

---

## Success Criteria

- [ ] All 19 confirmed bugs fixed
- [x] DRL expert signs off on DRL-01 through DRL-05 ✅ (2026-01-02)
- [x] PyTorch expert signs off on PT-01 through PT-07 ✅ (2026-01-02)
- [ ] Test suite passes with no new failures
- [ ] No silent exception handlers remaining (`except Exception: pass`)
