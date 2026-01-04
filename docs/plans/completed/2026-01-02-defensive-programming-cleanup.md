# Defensive Programming Cleanup Sprint

**Created:** 2026-01-02
**Updated:** 2026-01-02 (All Sign-Offs Complete)
**Status:** ✅ Ready for Implementation
**Scope:** Remove bug-hiding patterns that mask integration issues and schema drift

> **Context:** This sprint was triggered after significant debugging time was spent on faulty telemetry masked by a div/0 guard protecting a value that should never have been zero. These defensive patterns create "silent corruption" where training appears healthy while data is fabricated.

## Executive Summary

An audit of the Esper codebase identified **~120 instances** of defensive programming patterns. After triage by specialized Explore agents and specialist sign-off, we've refined this to **30 confirmed bugs** requiring fixes and **~100 legitimate patterns** that should NOT be changed.

**Second audit (2026-01-02):** A comprehensive 6-agent sweep discovered **14 additional candidates** not caught in the initial audit. After specialist review, **11 were confirmed** and **3 were rejected** as legitimate patterns (CRIT-03, HIGH-04, HIGH-05).

**Key principle:** If a defensive pattern would prevent a crash, that crash is a bug to fix upstream, not a symptom to suppress downstream.

---

## Specialist Sign-Off Status

| Specialist | Bugs Reviewed | Status | Date |
|------------|---------------|--------|------|
| **DRL Expert** | DRL-01 to DRL-05 | ✅ 5/5 CONFIRMED | 2026-01-02 |
| **PyTorch Expert** | PT-01 to PT-07 | ✅ 7/7 CONFIRMED | 2026-01-02 |
| **DRL Expert** | CRIT-01 to CRIT-03 | ✅ 2/3 CONFIRMED, 1 REJECTED | 2026-01-02 |
| **PyTorch Expert** | HIGH-01 to HIGH-05 | ✅ 3/5 CONFIRMED, 2 REJECTED | 2026-01-02 |

**DRL Expert Summary (Original):** *"These bugs collectively create a 'silent corruption' pattern where training appears healthy while phantom zero rewards pollute statistics, TD advantages use fabricated bootstrap values, and NaN gradients go unreported."*

**DRL Expert Summary (CRIT Review):** *"CRIT-01 and CRIT-02 are genuine bugs — Q-value defaults defeat NaN sentinel values and episode stats corrupt A/B testing. CRIT-03 is REJECTED: this is a first-observation bootstrap problem, not defensive programming. The first time we observe a slot, there IS no 'previous' gradient health — defaulting to 1.0 (healthy) is the semantically correct prior."*

**PyTorch Expert Summary (Original):** *"If data is malformed, it's a bug that should surface immediately. All torch.optim.Optimizer subclasses MUST have `param_groups[0]["lr"]` - if they don't, something is fundamentally broken."*

**PyTorch Expert Summary (HIGH Review):** *"HIGH-01, HIGH-02, HIGH-03 are genuine bugs that hide numerical errors and mask telemetry failures. HIGH-04 is REJECTED: alpha_overrides is intentionally a sparse override dict — callers only provide overrides for slots they want to modify. HIGH-05 is REJECTED: typed dataclass payloads are validated at construction time by the dataclass machinery — the code correctly skips enrichment for already-complete typed payloads."*

---

## Triage Results Summary

| Category | Count | Action |
|----------|-------|--------|
| **CRITICAL Bugs (NEW)** | 2 | Fix first — affect telemetry/A-B testing |
| **HIGH Bugs (NEW)** | 3 | Fix in Phase 1A |
| **DRL-Critical Bugs** | 5 | Fix in Phase 1 |
| **Data Integrity Bugs** | 7 | Fix in Phase 2 |
| **UI/Widget Bugs** | 3 | Fix in Phase 3 |
| **MEDIUM Bugs (NEW)** | 5 | Fix in Phase 3A |
| **Cleanup/Redundant** | 2 | Fix in Phase 4 |
| **Legitimate Patterns** | ~103 | No action needed (incl. CRIT-03, HIGH-04, HIGH-05) |
| **TOTAL BUGS** | **30** | - |

---

## NEW FINDINGS (Second Audit Pass)

### Category CRIT: CRITICAL Path Bugs ✅ (DRL Expert Signed Off — 2/3 Confirmed)

These affect telemetry and A/B testing integrity.

| ID | File | Line | Pattern | Impact | Fix | Status |
|----|------|------|---------|--------|-----|--------|
| **CRIT-01** | `simic/telemetry/emitters.py` | 818-825, 852-860 | Q-value metrics `.get(..., 0.0)` | Missing Q-values default to 0.0; hides policy evaluation failures; dashboard shows "normal" when rollouts are corrupted | Use NaN for missing, fail-fast if required | ✅ CONFIRMED |
| **CRIT-02** | `simic/training/vectorized.py` | 3699-3702 | Episode stats `.get(..., 0)` | Missing episode_reward/final_accuracy default to 0; A/B test corruption; can't distinguish "zero reward" from "missing data" | Require fields with direct access | ✅ CONFIRMED |
| **CRIT-03** | `tamiyo/policy/features.py` | 262, 280 | `.get(slot_id, default)` in active features | *(First-observation bootstrap — no prior data exists for new slots)* | **No fix needed** | ❌ REJECTED |

### Category HIGH: Training Signal Bugs ✅ (PyTorch Expert Signed Off — 3/5 Confirmed)

These affect training signal integrity and debugging capability.

| ID | File | Line | Pattern | Impact | Fix | Status |
|----|------|------|---------|--------|-----|--------|
| **HIGH-01** | `simic/training/vectorized.py` | 2362-2369, 3344-3353 | Broad `except Exception` in counterfactual | NaN/Inf in Shapley computation silently caught; next epoch uses stale counterfactual values | Narrow to `KeyError`, `ZeroDivisionError` | ✅ CONFIRMED (P1) |
| **HIGH-02** | `simic/telemetry/emitters.py` | 333-334 | Broad `except Exception` in layer gradient collection | Debug telemetry intermittently disabled without signal | Narrow to `RuntimeError`, `AttributeError` | ✅ CONFIRMED (P2) |
| **HIGH-03** | `karn/sanctum/aggregator.py` | 624-638 | Seed telemetry `.get(field, seed.field)` cascade | Missing telemetry fields silently keep previous epoch's values; UI shows "frozen" seeds | Replace with direct access, fail-fast on missing | ✅ CONFIRMED (P1) |
| **HIGH-04** | `kasmina/host.py` | 640 | Alpha override `.get(slot_id)` returns None | *(Intentional sparse override design — callers provide only slots they want to modify)* | **No fix needed** | ❌ REJECTED |
| **HIGH-05** | `kasmina/slot.py` | 2518-2534 | Unvalidated typed payloads in telemetry | *(Typed payloads validated at construction by dataclass machinery)* | **No fix needed** | ❌ REJECTED |

### Category MED: Medium Severity Bugs (General Review)

| ID | File | Line | Pattern | Impact | Fix |
|----|------|------|---------|--------|-----|
| **MED-01** | `simic/agent/ppo.py` | 177 | `entropy_coef_per_head or default` | Empty dict `{}` is falsy; silently reverts to defaults when caller passed explicit empty | Use `is not None` check |
| **MED-02** | `simic/agent/ppo.py` | 1032-1035 | Ratio max defaults to 1.0 | No data returns 1.0 (neutral) instead of NaN; can't distinguish "healthy ratio" from "no data" | Document or change to NaN |
| **MED-03** | `simic/training/vectorized.py` | 755 | Batch size `.get()` default 128 | Missing batch_size in task_spec silently defaults; masks schema violations | Fail if task spec incomplete |
| **MED-04** | `karn/sanctum/aggregator.py` | 567-568 | `compile_backend or ""` | Truthiness bug on string field; empty string conflated with None | Use explicit None check |
| **MED-05** | `karn/sanctum/aggregator.py` | 1367 | `alpha_target_map.get()` | Invalid alpha_target silently becomes None; decision cards show "NONE" | Validate in map or raise |

---

## CONFIRMED BUGS (Original Audit — All Sign-Offs Complete ✅)

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

## Implementation Checklist (By Phase)

### Phase 0: CRITICAL Path Fixes ✅ (DRL Expert Signed Off)

| Bug ID | Task | File:Line | Est. | Status |
|--------|------|-----------|------|--------|
| CRIT-01 | Replace Q-value `.get(..., 0.0)` with NaN propagation | `emitters.py:818-825, 852-860` | 30m | ⏳ TODO |
| CRIT-02 | Replace episode stats `.get(..., 0)` with direct access | `vectorized.py:3699-3702` | 20m | ⏳ TODO |
| ~~CRIT-03~~ | ~~Replace policy feature `.get()`~~ | ~~`features.py:262, 280`~~ | - | ❌ REJECTED |

**Total Phase 0:** ~50 minutes

### Phase 1: DRL-Critical Fixes ✅ (DRL Expert Signed Off)

| Bug ID | Task | File:Line | Est. |
|--------|------|-----------|------|
| DRL-01 | Fix `total_reward or 0.0` truthiness | `aggregator.py:1304` | 15m |
| DRL-02 | Fix `value_estimate or 0.0` truthiness | `aggregator.py:1346` | 15m |
| DRL-03 | Remove `.get()` default on `nan_grad_count` | `telemetry.py:754` | 10m |
| DRL-04 | Align `pre_clip_grad_norm` dataclass/deserializer | `telemetry.py:755` | 20m |
| DRL-05 | Align `ppo_updates_count` dataclass/deserializer | `telemetry.py:810` | 20m |

**Total Phase 1:** ~1.5 hours

### Phase 1A: HIGH Training Signal Fixes ✅ (PyTorch Expert Signed Off)

| Bug ID | Task | File:Line | Est. | Status |
|--------|------|-----------|------|--------|
| HIGH-01 | Narrow exception to `KeyError`, `ZeroDivisionError` | `vectorized.py:2362-2369, 3344-3353` | 30m | ⏳ TODO |
| HIGH-02 | Narrow exception to `RuntimeError`, `AttributeError` | `emitters.py:333-334` | 20m | ⏳ TODO |
| HIGH-03 | Replace seed telemetry `.get()` cascade with direct access | `aggregator.py:624-638` | 45m | ⏳ TODO |
| ~~HIGH-04~~ | ~~Validate alpha_overrides dict completeness~~ | ~~`host.py:640`~~ | - | ❌ REJECTED |
| ~~HIGH-05~~ | ~~Add typed payload validation in slot telemetry~~ | ~~`slot.py:2518-2534`~~ | - | ❌ REJECTED |

**Total Phase 1A:** ~1.5 hours

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

### Phase 3A: MEDIUM Fixes (General Review)

| Bug ID | Task | File:Line | Est. |
|--------|------|-----------|------|
| MED-01 | Use `is not None` for entropy_coef_per_head | `ppo.py:177` | 10m |
| MED-02 | Document or change ratio_max to NaN | `ppo.py:1032-1035` | 15m |
| MED-03 | Validate task_spec has batch_size or fail | `vectorized.py:755` | 15m |
| MED-04 | Use explicit None check for compile_backend/mode | `aggregator.py:567-568` | 10m |
| MED-05 | Validate alpha_target in map before using | `aggregator.py:1367` | 10m |

**Total Phase 3A:** ~1 hour

### Phase 4: Cleanup (No Sign-Off Required)

| Bug ID | Task | File:Line | Est. |
|--------|------|-----------|------|
| RD-01 | Remove redundant `setdefault()` | `ppo.py:842` | 5m |
| RD-02 | Verify/simplify `setdefault()` calls | `slot.py:2522-2533` | 20m |

**Total Phase 4:** ~30 minutes

---

## Specialist Review Protocol

### Original Reviews ✅ COMPLETE

**DRL Expert reviewed DRL-01 through DRL-05:**
- All patterns confirmed as corrupting training signals
- Recommended fixes: fail-fast on None, remove dataclass defaults for required fields
- No downstream consumers expect fabricated 0.0 values

**PyTorch Expert reviewed PT-01 through PT-07:**
- No checkpoint compatibility impact (these are telemetry ingestion bugs)
- No migration path needed
- Recommended: fail hard on malformed data, create `coerce_str()` that raises on invalid type

### New Reviews ✅ COMPLETE

**DRL Expert reviewed CRIT-01 through CRIT-03:**
- ✅ CRIT-01: Q-value defaults defeat NaN sentinel values — CONFIRMED
- ✅ CRIT-02: Episode stats corrupt A/B testing statistics — CONFIRMED
- ❌ CRIT-03: First-observation bootstrap is legitimate — REJECTED

**PyTorch Expert reviewed HIGH-01 through HIGH-05:**
- ✅ HIGH-01: Narrow counterfactual exception to specific types — CONFIRMED (P1)
- ✅ HIGH-02: Narrow gradient telemetry exception to RuntimeError/AttributeError — CONFIRMED (P2)
- ✅ HIGH-03: Seed telemetry stale fallbacks are bug-hiding — CONFIRMED (P1)
- ❌ HIGH-04: Sparse override dict is intentional design — REJECTED
- ❌ HIGH-05: Typed payloads validated by dataclass machinery — REJECTED

---

## Testing Strategy

1. **Before changes:** Run full test suite to establish baseline
2. **After CRIT fixes:** Run `pytest tests/simic/ tests/tamiyo/` — these touch policy network
3. **After DRL fixes:** Run `pytest tests/simic/ tests/karn/`
4. **After HIGH fixes:** Run `pytest tests/simic/ tests/kasmina/ tests/karn/`
5. **After PT fixes:** Run `pytest tests/karn/` + checkpoint load tests
6. **After UI fixes:** Manual Sanctum TUI smoke test
7. **Regression:** Full test suite after all phases

---

## Success Criteria

- [ ] All 30 confirmed bugs fixed
- [x] DRL expert signs off on DRL-01 through DRL-05 ✅ (2026-01-02)
- [x] PyTorch expert signs off on PT-01 through PT-07 ✅ (2026-01-02)
- [x] DRL expert signs off on CRIT-01 through CRIT-03 ✅ (2026-01-02) — 2 confirmed, 1 rejected
- [x] PyTorch expert signs off on HIGH-01 through HIGH-05 ✅ (2026-01-02) — 3 confirmed, 2 rejected
- [ ] Test suite passes with no new failures
- [ ] No silent exception handlers remaining (`except Exception: pass`)
