# Test Suite Remediation Plan

**Date:** 2025-12-16
**Status:** Draft for Review
**Scope:** Comprehensive test suite quality improvement

---

## Executive Summary

The Esper test suite has grown organically to 1,091 tests but suffers from:
- **Uneven coverage** (Karn severely undertested, Kasmina lacks property tests)
- **Missing seam tests** (5 of 7 critical integration points untested)
- **Insufficient property-based testing** (only 13% of tests use Hypothesis)
- **Duplication and low-value tests** (13 duplicates, 678 trivial tests)

This plan provides a phased approach to systematically address these issues.

---

## Phase 1: Critical Seam Coverage (Week 1)

### Objective
Add integration tests for the 5 missing critical seams. These are the highest-risk gaps.

### 1.1 simic ↔ kasmina Integration

**File:** `tests/integration/test_simic_kasmina.py`

**What to test:**
- Reward signals correctly flow to seed slot alpha updates
- PBRS shaping respects seed stage transitions
- Gradient isolation between host and seed during blending
- Fossilization triggers correct reward mode switch

**Key scenarios:**
```python
class TestSimicKasminaIntegration:
    """Integration tests for RL rewards affecting seed lifecycle."""

    def test_positive_reward_increases_alpha(self):
        """Consistent positive rewards should advance seed toward blending."""
        # Create model with active seed
        # Run PPO steps with positive rewards
        # Assert alpha increased

    def test_negative_reward_triggers_cull_consideration(self):
        """Sustained negative rewards should move seed toward culling."""
        # Create model with active seed
        # Run PPO steps with negative rewards
        # Assert cull signals generated

    def test_pbrs_shaping_matches_stage_potential(self):
        """PBRS potential should align with current seed stage."""
        # For each stage transition
        # Assert PBRS delta matches expected potential difference

    def test_fossilization_disables_seed_gradients(self):
        """After fossilization, seed parameters should not receive gradients."""
        # Fossilize a seed
        # Run backward pass
        # Assert seed.grad is None for all parameters
```

**Estimated tests:** 8-12

---

### 1.2 tolaria ↔ simic Integration

**File:** `tests/integration/test_tolaria_simic.py`

**What to test:**
- Training loop correctly invokes PPO update
- Loss values flow correctly to optimizer
- Gradient clipping respects configured bounds
- Learning rate scheduling affects PPO correctly

**Key scenarios:**
```python
class TestTolariaSimicIntegration:
    """Integration tests for trainer invoking RL updates."""

    def test_train_epoch_calls_ppo_update(self):
        """train_epoch should invoke PPO update with correct arguments."""

    def test_loss_gradient_flow(self):
        """Loss from PPO should produce gradients in model parameters."""

    def test_gradient_clipping_applied(self):
        """Configured gradient clipping should limit gradient norms."""

    def test_lr_schedule_affects_ppo(self):
        """Learning rate changes should propagate to PPO optimizer."""
```

**Estimated tests:** 6-8

---

### 1.3 tolaria ↔ kasmina Integration

**File:** `tests/integration/test_tolaria_kasmina.py`

**What to test:**
- Trainer correctly handles MorphogeneticModel
- Validation runs with seeds in various stages
- Checkpoint save/load preserves seed state
- Attribution validation produces correct counterfactuals

**Key scenarios:**
```python
class TestTolariaKasminaIntegration:
    """Integration tests for trainer with morphogenetic models."""

    def test_train_with_active_seed(self):
        """Training should work with model containing active seed."""

    def test_validation_with_blending_seed(self):
        """Validation accuracy should reflect seed contribution."""

    def test_checkpoint_preserves_seed_state(self):
        """Save/load cycle should preserve seed stage and alpha."""

    def test_attribution_isolates_seed_contribution(self):
        """Attribution validation should correctly measure seed impact."""
```

**Estimated tests:** 6-10

---

### 1.4 nissa ↔ simic Integration

**File:** `tests/integration/test_nissa_simic.py`

**What to test:**
- Telemetry events correctly capture training metrics
- Reward components are logged accurately
- Gradient health metrics are recorded
- Anomaly detection triggers on pathological training

**Key scenarios:**
```python
class TestNissaSimicIntegration:
    """Integration tests for telemetry during RL training."""

    def test_ppo_update_emits_telemetry(self):
        """PPO update should emit training telemetry events."""

    def test_reward_components_logged(self):
        """All reward components should appear in telemetry."""

    def test_gradient_health_recorded(self):
        """Gradient norm and health should be telemetered."""

    def test_anomaly_detection_on_nan_loss(self):
        """NaN loss should trigger anomaly event."""
```

**Estimated tests:** 6-8

---

### 1.5 karn ↔ nissa Integration

**File:** `tests/integration/test_karn_nissa.py`

**What to test:**
- Analytics correctly aggregate telemetry events
- TUI state reflects live telemetry
- Export formats contain all telemetry data
- Multi-environment collection works correctly

**Key scenarios:**
```python
class TestKarnNissaIntegration:
    """Integration tests for analytics consuming telemetry."""

    def test_analytics_aggregate_events(self):
        """Analytics should correctly summarize telemetry stream."""

    def test_tui_reflects_live_telemetry(self):
        """TUI state should update from telemetry events."""

    def test_export_contains_all_events(self):
        """JSONL export should contain all emitted events."""
```

**Estimated tests:** 4-6

---

## Phase 2: Property-Based Testing Expansion (Week 2)

### Objective
Add property-based tests to modules with mathematical invariants or state machine behavior.

### 2.1 Kasmina Property Tests (CRITICAL)

**File:** `tests/kasmina/properties/test_seed_slot_properties.py`

Currently kasmina has **195 tests but only 1 property test**. This is the biggest gap.

**Properties to test:**

```python
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule

class TestSeedSlotInvariants:
    """Property-based tests for SeedSlot state machine."""

    @given(st.floats(0.0, 1.0))
    def test_alpha_always_in_bounds(self, alpha):
        """Alpha should always be in [0, 1] regardless of updates."""

    @given(st.lists(st.sampled_from(['advance', 'cull', 'fossilize'])))
    def test_stage_transitions_are_monotonic(self, actions):
        """Stage can only increase (DORMANT → GERMINATED → ... → FOSSILIZED)."""

    @given(st.floats(-100, 100))
    def test_alpha_update_respects_stage(self, delta):
        """Alpha updates should only work in appropriate stages."""


class SeedSlotStateMachine(RuleBasedStateMachine):
    """Stateful testing of SeedSlot lifecycle."""

    def __init__(self):
        super().__init__()
        self.slot = SeedSlot(channels=64, device="cpu")

    @rule()
    def germinate(self):
        """Germinate a seed if dormant."""

    @rule(delta=st.floats(-0.1, 0.1))
    def update_alpha(self, delta):
        """Update alpha and verify bounds."""

    @rule()
    def advance_stage(self):
        """Advance to next stage."""

    @invariant()
    def alpha_in_bounds(self):
        assert 0.0 <= self.slot.alpha <= 1.0

    @invariant()
    def stage_is_valid(self):
        assert self.slot.stage in SeedStage
```

**Estimated tests:** 15-20

---

**File:** `tests/kasmina/properties/test_blending_properties.py`

```python
class TestBlendingInvariants:
    """Property tests for alpha blending mathematics."""

    @given(st.floats(0, 1), host_tensor(), seed_tensor())
    def test_blend_interpolates_correctly(self, alpha, host, seed):
        """blend(alpha) = (1-alpha)*host + alpha*seed"""

    @given(host_tensor(), seed_tensor())
    def test_alpha_zero_returns_host(self, host, seed):
        """With alpha=0, output should equal host exactly."""

    @given(host_tensor(), seed_tensor())
    def test_alpha_one_returns_seed(self, host, seed):
        """With alpha=1, output should equal seed exactly."""

    @given(st.floats(0, 1), st.floats(0, 1))
    def test_blend_is_monotonic_in_alpha(self, alpha1, alpha2):
        """Higher alpha should produce output closer to seed."""
```

**Estimated tests:** 8-10

---

### 2.2 Tolaria Property Tests

**File:** `tests/tolaria/properties/test_trainer_properties.py`

```python
class TestTrainerInvariants:
    """Property tests for training loop correctness."""

    @given(st.integers(1, 100), st.floats(0.0001, 0.1))
    def test_loss_decreases_on_simple_task(self, epochs, lr):
        """On a simple task, loss should generally decrease."""

    @given(st.floats(0.1, 10.0))
    def test_gradient_clipping_bounds_norm(self, max_norm):
        """Gradient norm should never exceed max_norm after clipping."""

    @given(st.integers(1, 10))
    def test_validation_deterministic(self, runs):
        """Multiple validation runs should produce identical results."""
```

**File:** `tests/tolaria/properties/test_governor_properties.py`

```python
class TestGovernorInvariants:
    """Property tests for TolariaGovernor watchdog."""

    @given(st.floats(0.1, 60.0), st.floats(0.1, 60.0))
    def test_timeout_triggers_correctly(self, timeout, elapsed):
        """Governor should trigger exactly when elapsed > timeout."""

    @given(st.lists(st.floats(0, 100)))
    def test_loss_spike_detection(self, losses):
        """Sudden loss increases should trigger intervention."""
```

**Estimated tests:** 10-12

---

### 2.3 Leyline Property Tests

**File:** `tests/leyline/properties/test_factored_actions_properties.py`

```python
class TestFactoredActionInvariants:
    """Property tests for factored action space."""

    @given(valid_action_tuples())
    def test_encode_decode_roundtrip(self, action):
        """encode(decode(action)) == action"""

    @given(st.integers(0, TOTAL_ACTIONS - 1))
    def test_flat_to_factored_bijection(self, flat_idx):
        """Flat index should uniquely map to factored action."""

    @given(valid_slot_configs())
    def test_action_mask_respects_config(self, config):
        """Masked actions should only allow valid operations."""
```

**Estimated tests:** 8-10

---

### 2.4 Simic Property Test Additions

**File:** `tests/simic/properties/test_advantages_properties.py`

```python
class TestGAEInvariants:
    """Property tests for Generalized Advantage Estimation."""

    @given(reward_sequences(), st.floats(0.9, 0.999), st.floats(0.9, 0.999))
    def test_gae_bounded_by_mc_and_td(self, rewards, gamma, lam):
        """GAE should interpolate between MC and TD estimates."""

    @given(st.floats(0.9, 0.999))
    def test_lambda_one_equals_mc(self, gamma):
        """With lambda=1, GAE should equal Monte Carlo returns."""

    @given(st.floats(0.9, 0.999))
    def test_lambda_zero_equals_td(self, gamma):
        """With lambda=0, GAE should equal TD(0) estimates."""
```

**Estimated tests:** 6-8

---

## Phase 3: Duplicate Removal & Consolidation (Week 3)

### Objective
Remove the 13 duplicate tests and consolidate related tests.

### 3.1 Duplicates to Remove

| Test Name | Keep In | Remove From |
|-----------|---------|-------------|
| `test_components_sum_to_total` | `test_reward_invariants.py` | `test_rewards.py`, `test_reward_telemetry.py` |
| `test_default_values` | Single location in `test_leyline.py` | Other 2 in same file |
| `test_embargo_blocks_germination` | `test_tamiyo_properties.py` | `test_heuristic_unit.py` |
| `test_first_epoch_no_crash` | `test_edge_cases.py` | `test_tamiyo_properties.py` |
| `test_fossilize_maps_to_advance_stage` | `test_command_properties.py` | `test_decisions_unit.py` |
| `test_fossilized_has_highest_potential` | `test_pbrs_properties.py` | `test_rewards.py` |
| `test_initial_values` | Single location in `test_analytics.py` | Duplicate in same file |
| `test_lm_task_type` | `test_trainer.py` | `test_training_helper.py` |
| `test_penalized_blueprint_avoided` | `test_decision_antigaming.py` | `test_heuristic_decisions.py` |
| `test_reset_clears_all_state` | `test_governor.py` | `test_tracker_unit.py` |
| `test_validate_and_get_metrics_returns_tuple` | `test_environment.py` | `test_tamiyo_tolaria.py` |
| `test_wait_maps_to_request_state` | `test_command_properties.py` | `test_decisions_unit.py` |
| `test_zero_loss_no_division_error` | `test_edge_cases.py` | `test_tamiyo_properties.py` |

### 3.2 Consolidation Strategy

**Principle:** Property-based tests supersede example-based tests for the same invariant.

When a property test covers the same ground as multiple example tests:
1. Keep the property test
2. Remove example tests that are strict subsets
3. Keep example tests that test specific edge cases not covered by property

---

## Phase 4: Karn Test Coverage Sprint (Week 4)

### Objective
Bring Karn from 3:1 test ratio to 15:1 minimum.

### Current State
- **11 source files** in `src/esper/karn/`
- **31 tests** in `tests/karn/`
- **Ratio: 3:1** (lowest in codebase)

### Files Needing Tests

| Source File | Current Tests | Target Tests | Priority |
|-------------|---------------|--------------|----------|
| `collector.py` | ~5 | 15 | HIGH |
| `store.py` | ~5 | 15 | HIGH |
| `analytics.py` | ~8 | 20 | HIGH |
| `tui.py` | ~8 | 15 | MEDIUM |
| `export.py` | ~5 | 10 | MEDIUM |
| `events.py` | 0 | 10 | HIGH |
| `aggregator.py` | 0 | 10 | HIGH |

### Key Test Scenarios

**collector.py:**
- Multi-environment event collection
- Event ordering guarantees
- Buffer overflow handling
- Thread safety (if applicable)

**store.py:**
- Persistence correctness
- Query performance
- Concurrent access
- Corruption recovery

**analytics.py:**
- Aggregation correctness
- Statistical calculations
- Time window handling
- Memory efficiency with large datasets

---

## Phase 5: Trivial Test Review (Week 5)

### Objective
Review the 678 tests with ≤2 lines and either enhance or remove them.

### Triage Categories

1. **Delete** - Tests that only call a function without assertions
2. **Enhance** - Tests with assertions but missing edge cases
3. **Keep** - Legitimate smoke tests or regression tests

### Review Process

For each trivial test:

```
1. Does it have an assertion?
   - NO → Delete or add assertion
   - YES → Continue

2. Is there a property test covering the same invariant?
   - YES → Consider deleting (property test is stronger)
   - NO → Continue

3. Is this a smoke test for a complex function?
   - YES → Keep (smoke tests have value)
   - NO → Continue

4. Does this test a specific bug fix?
   - YES → Keep and add comment linking to issue
   - NO → Consider deleting or enhancing
```

### Files to Prioritize

| File | Trivial Tests | Action |
|------|---------------|--------|
| `nissa/test_output.py` | 5 | Review - may be legitimate smoke tests |
| `nissa/test_analytics.py` | 6 | Review - statistical tests may be thin |
| `integration/test_sparse_training.py` | 3 | Review - integration smoke tests |

---

## Phase 6: CI/Quality Gates (Week 6)

### Objective
Prevent regression by adding automated quality checks.

### 6.1 Property Test Coverage Gate

Add to CI:
```yaml
- name: Check property test coverage
  run: |
    PROP_COUNT=$(grep -r "@given" tests/ --include="*.py" | wc -l)
    if [ $PROP_COUNT -lt 150 ]; then
      echo "Property test count ($PROP_COUNT) below threshold (150)"
      exit 1
    fi
```

### 6.2 Seam Test Manifest

Create `tests/integration/SEAMS.md`:
```markdown
# Required Integration Tests

All cross-domain interfaces must have integration tests.

| Seam | Test File | Status |
|------|-----------|--------|
| tamiyo ↔ simic | test_tamiyo_simic.py | ✓ |
| tamiyo ↔ kasmina | test_tamiyo_kasmina.py | ✓ |
| simic ↔ kasmina | test_simic_kasmina.py | ✓ |
| tolaria ↔ simic | test_tolaria_simic.py | ✓ |
| tolaria ↔ kasmina | test_tolaria_kasmina.py | ✓ |
| nissa ↔ simic | test_nissa_simic.py | ✓ |
| karn ↔ nissa | test_karn_nissa.py | ✓ |
```

### 6.3 New Module Checklist

When adding a new source module:
- [ ] Unit tests with ≥15:1 test ratio
- [ ] Property tests if stateful or mathematical
- [ ] Integration tests if cross-domain

---

## Success Metrics

### Phase 1 Complete
- [ ] 5 new integration test files
- [ ] 30-44 new integration tests
- [ ] All critical seams covered

### Phase 2 Complete
- [ ] Property tests ≥200 (from 140)
- [ ] Kasmina property tests ≥20 (from 1)
- [ ] Tolaria property tests ≥10 (from 0)

### Phase 3 Complete
- [ ] 0 duplicate test names
- [ ] Test count reduced by ~15-20

### Phase 4 Complete
- [ ] Karn test ratio ≥15:1 (from 3:1)
- [ ] All karn source files have tests

### Phase 5 Complete
- [ ] Trivial tests reduced by ≥50%
- [ ] Remaining trivial tests documented

### Phase 6 Complete
- [ ] CI gates active
- [ ] Seam manifest enforced
- [ ] New module checklist in CONTRIBUTING.md

---

## Appendix: Quick Reference

### Property Test Template

```python
from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

class TestMyModuleProperties:
    """Property-based tests for MyModule."""

    @given(st.floats(allow_nan=False, allow_infinity=False))
    def test_invariant_name(self, value):
        """Describe the invariant being tested."""
        result = my_function(value)
        assert invariant_holds(result)

    @settings(max_examples=200)  # More examples for critical paths
    @given(complex_strategy())
    def test_critical_invariant(self, data):
        """Critical invariants get more examples."""
        pass
```

### Integration Test Template

```python
import pytest
from esper.domain1 import Component1
from esper.domain2 import Component2

class TestDomain1Domain2Integration:
    """Integration tests for Domain1 ↔ Domain2 interface."""

    @pytest.fixture
    def integrated_system(self):
        """Create integrated components."""
        c1 = Component1()
        c2 = Component2()
        return c1, c2

    def test_data_flows_correctly(self, integrated_system):
        """Data from Component1 should reach Component2 correctly."""
        c1, c2 = integrated_system
        c1.produce_data()
        assert c2.received_data_matches_expected()

    def test_error_propagation(self, integrated_system):
        """Errors in Component1 should be handled by Component2."""
        c1, c2 = integrated_system
        c1.trigger_error()
        assert c2.handled_error_gracefully()
```

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Seam Coverage | 5 integration test files |
| 2 | Property Tests | 50+ new property tests |
| 3 | Consolidation | Duplicate removal |
| 4 | Karn Sprint | 70+ new Karn tests |
| 5 | Trivial Review | Test cleanup |
| 6 | CI Gates | Automation |

**Total estimated new tests:** 150-200
**Total estimated removed tests:** 30-50
**Net improvement:** Higher quality, better coverage of critical paths
