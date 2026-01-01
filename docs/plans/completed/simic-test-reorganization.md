# Simic Test Reorganization Plan

**Date:** 2025-12-16
**Status:** Draft for Review
**Risk:** LOW - File moves only, no code changes

---

## Objective

Consolidate simic-related tests into `tests/simic/` to:
1. Mirror source structure (`src/esper/simic/` → `tests/simic/`)
2. Enable running domain tests easily: `pytest tests/simic/`
3. Reduce confusion about where to add new tests
4. Clean up legacy top-level test placement

---

## Phase 1: Clean Moves (No Conflicts)

These files have no naming conflicts and can be moved directly.

### 1.1 Top-level → `tests/simic/`

| Source | Destination | Size | Tests |
|--------|-------------|------|-------|
| `tests/test_simic_rewards.py` | `tests/simic/test_rewards.py` | 58KB | ~50 |
| `tests/test_simic_normalization.py` | `tests/simic/test_normalization.py` | 6.5KB | ~15 |

**Commands:**
```bash
git mv tests/test_simic_rewards.py tests/simic/test_rewards.py
git mv tests/test_simic_normalization.py tests/simic/test_normalization.py
```

### 1.2 `tests/properties/` → `tests/simic/properties/`

| Source | Destination | Size |
|--------|-------------|------|
| `tests/properties/test_reward_properties.py` | `tests/simic/properties/test_reward_properties.py` | 6.6KB |
| `tests/properties/test_gradient_properties.py` | `tests/simic/properties/test_gradient_properties.py` | 17KB |
| `tests/properties/test_normalization_properties.py` | `tests/simic/properties/test_normalization_properties.py` | 11.7KB |

**Commands:**
```bash
git mv tests/properties/test_reward_properties.py tests/simic/properties/
git mv tests/properties/test_gradient_properties.py tests/simic/properties/
git mv tests/properties/test_normalization_properties.py tests/simic/properties/
```

### 1.3 `tests/properties/` → `tests/leyline/properties/`

| Source | Destination | Reason |
|--------|-------------|--------|
| `tests/properties/test_action_properties.py` | `tests/leyline/properties/test_action_properties.py` | Tests `esper.leyline.actions` |

**Commands:**
```bash
mkdir -p tests/leyline/properties
git mv tests/properties/test_action_properties.py tests/leyline/properties/
```

**After Phase 1:** `tests/properties/` should be empty and can be removed.

---

## Phase 2: Merges (Naming Conflicts)

These files have existing counterparts in `tests/simic/`. Requires manual merge.

### 2.1 Vectorized Tests

| File | Lines | Test Count | Content |
|------|-------|------------|---------|
| `tests/test_simic_vectorized.py` | 660 | 16 | Unit tests for `_advance_active_seed`, `_calculate_entropy_anneal_steps`, `_emit_*`, etc. |
| `tests/simic/test_vectorized.py` | 179 | 10 | Tests for `_build_simic_action_mask`, normalization helpers |

**Analysis:**
- Tests are **complementary** (different functions tested)
- No duplicate test names found

**Recommendation:** Append `test_simic_vectorized.py` content to `test_vectorized.py`

**Merge Strategy:**
1. Read both files
2. Combine imports (deduplicate)
3. Append test classes from `test_simic_vectorized.py`
4. Delete `test_simic_vectorized.py`

### 2.2 Config Tests

| File | Lines | Content |
|------|-------|---------|
| `tests/test_simic_config.py` | 75 | `TestTrainingConfig` class |
| `tests/simic/test_config.py` | 30 | `TestTrainingConfigBasics` class |

**Analysis:**
- Both test `TrainingConfig` dataclass
- Top-level file has more tests

**Recommendation:** Merge into `tests/simic/test_config.py`

### 2.3 Gradient Collector Tests

| File | Tests | Content |
|------|-------|---------|
| `tests/test_simic_gradient_collector.py` | 2 | `test_gradient_collector_vectorized`, `test_gradient_collector_empty` |
| `tests/simic/test_gradient_collector_enhanced.py` | 8 | Enhanced metrics, dual stats, normalized ratio |

**Analysis:**
- Top-level has 2 basic tests
- Simic folder has comprehensive enhanced tests

**Recommendation:** Append 2 basic tests to enhanced file, then rename to `test_gradient_collector.py`

### 2.4 PBRS Property Tests

| File | Lines | Content |
|------|-------|---------|
| `tests/properties/test_pbrs_telescoping.py` | 566 | Comprehensive telescoping tests, lifecycle comparisons |
| `tests/simic/properties/test_pbrs_properties.py` | 236 | Monotonicity, basic telescoping |

**Analysis:**
- `test_pbrs_telescoping.py` is more comprehensive (20 tests vs 4)
- Some overlap in monotonicity and telescoping tests
- `test_pbrs_telescoping.py` has unique tests: roundtrip cancellation, cull cycles, lifecycle comparisons

**Recommendation:**
1. Merge unique tests from `test_pbrs_properties.py` into `test_pbrs_telescoping.py`
2. Rename combined file to `test_pbrs_properties.py` (shorter, follows naming pattern)
3. Delete original `test_pbrs_properties.py`

---

## Phase 3: Cleanup

### 3.1 Remove Empty Directory
```bash
rmdir tests/properties  # After all files moved
```

### 3.2 Update Any Import Paths

Search for hardcoded test paths in:
- CI/CD configs (`.github/workflows/`)
- `pytest.ini` or `pyproject.toml`
- Documentation

**Likely no changes needed** - pytest discovers tests by pattern, not explicit paths.

---

## Verification Plan

After each phase:

```bash
# Run all tests to verify nothing broke
PYTHONPATH=src uv run pytest tests/ -x -q

# Run simic tests specifically
PYTHONPATH=src uv run pytest tests/simic/ -v

# Check for any orphaned imports
grep -r "from tests.test_simic" tests/
grep -r "from tests.properties" tests/
```

---

## Summary Table

| Action | Files | Risk |
|--------|-------|------|
| **Phase 1: Clean Moves** | 6 files | LOW |
| **Phase 2: Merges** | 4 merge operations | MEDIUM |
| **Phase 3: Cleanup** | Remove empty dir | LOW |

**Total files affected:** 10
**Estimated time:** 30-45 minutes
**Rollback:** `git reset --hard HEAD~1` (if done in single commit)

---

## Questions for Review

1. **Naming convention:** Should merged files drop the `simic_` prefix entirely? (e.g., `test_rewards.py` vs `test_simic_rewards.py`)
   - **Recommendation:** Yes, drop prefix since location (`tests/simic/`) already indicates domain

2. **PBRS merge:** Keep `test_pbrs_properties.py` name or `test_pbrs_telescoping.py`?
   - **Recommendation:** `test_pbrs_properties.py` (consistent with other property test naming)

3. **Gradient collector:** Rename `test_gradient_collector_enhanced.py` → `test_gradient_collector.py`?
   - **Recommendation:** Yes, "enhanced" is legacy naming from an upgrade

4. **Commit strategy:** Single commit or one per phase?
   - **Recommendation:** One commit per phase for easier review/rollback

---

## Appendix: Current vs Target Structure

### Current Structure (Simic-related only)
```
tests/
├── test_simic_rewards.py          # 58KB
├── test_simic_vectorized.py       # 24KB
├── test_simic_config.py           # 2.8KB
├── test_simic_normalization.py    # 6.5KB
├── test_simic_gradient_collector.py # 1.3KB
├── properties/
│   ├── test_reward_properties.py
│   ├── test_gradient_properties.py
│   ├── test_normalization_properties.py
│   ├── test_pbrs_telescoping.py
│   └── test_action_properties.py  # → leyline
└── simic/
    ├── test_vectorized.py         # Merge target
    ├── test_config.py             # Merge target
    ├── test_gradient_collector_enhanced.py  # Merge target
    └── properties/
        └── test_pbrs_properties.py  # Merge target
```

### Target Structure
```
tests/
├── leyline/
│   └── properties/
│       └── test_action_properties.py
└── simic/
    ├── test_rewards.py            # From test_simic_rewards.py
    ├── test_vectorized.py         # Merged
    ├── test_config.py             # Merged
    ├── test_normalization.py      # From test_simic_normalization.py
    ├── test_gradient_collector.py # Merged + renamed
    └── properties/
        ├── test_reward_properties.py
        ├── test_gradient_properties.py
        ├── test_normalization_properties.py
        └── test_pbrs_properties.py  # Merged
```
