# Seed Scaffolding Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** PARTIALLY IMPLEMENTED (~75%)

**Goal:** Enable Tamiyo to learn and exploit seed scaffolding behavior - where one seed "primes" another, making the second seed more powerful, then gets pruned after its scaffolding role is complete.

**Architecture:** Four-phase incremental approach: (1) Fix feature redundancy, (2) Add scaffolding observations, (3) Add reward shaping, (4) Optional auxiliary value head. Each phase is independently valuable and builds on the previous.

**Tech Stack:** Python 3.11+, PyTorch 2.x, PPO with factored recurrent actor-critic (LSTM), counterfactual validation via fused forward passes.

---

## Implementation Status (Gap Analysis 2025-12-24)

| Phase | Task | Status |
|-------|------|--------|
| **Phase 1** | Use raw contribution_velocity | ✅ Done |
| **Phase 2.1** | Add interaction fields to SeedMetrics | ✅ Done |
| **Phase 2.2** | Update feature extraction (SLOT_FEATURE_SIZE=39) | ✅ Done |
| **Phase 2.3** | Compute interaction metrics in vectorized | ✅ Done |
| **Phase 3.1** | Add synergy bonus to rewards | ✅ Done |
| **Phase 3.2** | Scaffold hindsight credit | ❌ **NOT IMPLEMENTED** |
| **Phase 3.3** | Increase GAE lambda to 0.98 | ❓ Unverified |
| **Phase 4** | Optional aux value head | ⏸️ Skipped (by design) |

### Gap: Scaffold Hindsight Credit (Phase 3.2)

**What's missing:** `compute_scaffold_hindsight_credit()` function that gives retroactive reward to scaffolds when their beneficiary succeeds.

**Impact:** Scaffolds that get pruned *before* their beneficiary fossilizes won't receive credit. The real-time synergy bonus (Phase 3.1 ✅) partially compensates, but full credit assignment for delayed scaffolding payoff is missing.

**Decision needed:** Is this blocking for Phase 3 Transformers, or is the synergy bonus sufficient?

---

## Background: The Scaffolding Phenomenon

During training, an emergent behavior was observed where Tamiyo "changed" a blueprint by:
1. Germinating seed A in one slot
2. Germinating seed B in another slot
3. Seed A modifies the host in a way that makes seed B more effective
4. Pruning seed A once seed B has "absorbed" its influence

This is analogous to biological scaffolding proteins that temporarily support other structures during development. We want Tamiyo to:
- **Recognize** when a seed is acting as a scaffold (low solo contribution, high interaction with others)
- **Reward** the scaffolding behavior even when the scaffold seed gets pruned
- **Learn** to intentionally deploy scaffolding strategies

## Phase 1: Feature Redundancy Fix

**Why:** The DRL expert identified that `fossilize_value = contribution + velocity * 3` is highly correlated with `contribution`, reducing its informational value. Replacing with raw `contribution_velocity` provides orthogonal signal.

### Task 1.1: Update features.py to use raw contribution_velocity

**Files:**
- Modify: `src/esper/tamiyo/policy/features.py:279-283`
- Test: `tests/tamiyo/policy/test_features.py`

**Step 1: Write the failing test**

```python
# tests/tamiyo/policy/test_features.py - Add after test_stage_one_hot_all_valid_stages

def test_contribution_velocity_raw_not_fossilize_value():
    """Verify contribution_velocity is passed through raw, not as fossilize_value."""
    from esper.tamiyo.policy.features import obs_to_multislot_features, SLOT_FEATURE_SIZE

    obs = _make_test_obs(epoch=5)
    obs["slots"] = {
        "r0c0": {
            "is_active": True,
            "stage": 3,  # TRAINING
            "alpha": 0.5,
            "improvement": 2.0,  # contribution
            "contribution_velocity": 0.8,  # raw velocity
            "blueprint_id": "conv_light",
        }
    }

    features = obs_to_multislot_features(obs, total_seeds=1, max_seeds=3)

    # Slot 0 state features start at index 23 + 11 = 34 (after base + is_active + stage one-hot)
    # [0] is_active, [1-10] stage one-hot, [11] alpha, [12] contribution, [13] velocity
    velocity_idx = 23 + 13  # offset 36
    velocity_feature = features[velocity_idx]

    # Velocity should be 0.8 / 10.0 (normalized by _IMPROVEMENT_CLAMP_PCT_PTS)
    expected = 0.8 / 10.0  # 0.08
    assert abs(velocity_feature - expected) < 1e-6, (
        f"Expected raw velocity {expected}, got {velocity_feature}"
    )
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py::test_contribution_velocity_raw_not_fossilize_value -v`
Expected: FAIL (currently computes `contribution + velocity * 3` instead of raw velocity)

**Step 3: Update obs_to_multislot_features**

```python
# src/esper/tamiyo/policy/features.py - Replace lines 278-283

        # Compute normalized contribution and velocity (raw, not lookahead)
        contribution = safe(slot.get("improvement", 0.0), 0.0, max_val=_IMPROVEMENT_CLAMP_PCT_PTS)
        contribution_velocity = safe(slot.get("contribution_velocity", 0.0), 0.0, max_val=_IMPROVEMENT_CLAMP_PCT_PTS)

        # is_active (1 dim)
        slot_features.append(float(slot.get('is_active', 0)))
        # Stage one-hot (10 dims)
        slot_features.extend(stage_one_hot)
        # Other state features (11 dims)
        slot_features.extend([
            alpha,
            contribution / _IMPROVEMENT_CLAMP_PCT_PTS,
            contribution_velocity / _IMPROVEMENT_CLAMP_PCT_PTS,  # Raw velocity, not fossilize_value
            # ... rest unchanged
```

**Step 4: Update batch_obs_to_features (same pattern)**

```python
# src/esper/tamiyo/policy/features.py - Replace lines 392-397

                # Get contribution velocity (raw, not lookahead)
                velocity = report.metrics.contribution_velocity
                velocity_norm = max(-1.0, min(velocity / _IMPROVEMENT_CLAMP_PCT_PTS, 1.0))

                # ... in the feature assignment section, replace fossilize_value with velocity:
                features[env_idx, offset + 13] = velocity_norm  # Was: fossilize_value / _IMPROVEMENT_CLAMP_PCT_PTS
```

**Step 5: Update docstrings**

```python
# Update per-slot feature documentation in obs_to_multislot_features docstring:
#    - fossilize_value: estimated long-term value (contribution + velocity * lookahead)
# Replace with:
#    - contribution_velocity: EMA of contribution delta (for trend detection)
```

**Step 6: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py::test_contribution_velocity_raw_not_fossilize_value -v`
Expected: PASS

**Step 7: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/esper/tamiyo/policy/features.py tests/tamiyo/policy/test_features.py
git commit -m "feat(tamiyo): use raw contribution_velocity instead of fossilize_value

DRL expert recommendation: fossilize_value (contribution + velocity * 3) is
highly correlated with contribution, providing little orthogonal signal.
Raw velocity enables trend detection without redundancy.

BREAKING: Observation index 13 per slot now contains normalized velocity,
not fossilize_value. Checkpoints trained on old schema are incompatible."
```

---

## Phase 2: Scaffolding Observations

**Why:** Tamiyo needs to see inter-slot interactions to learn scaffolding. The counterfactual engine already computes `InteractionTerm` with synergy/interference classification; we expose this in observations.

### Task 2.1: Add interaction tracking to SeedMetrics

**Files:**
- Modify: `src/esper/leyline/reports.py:31-70`
- Test: `tests/leyline/test_reports.py` (create if needed)

**Step 1: Write the failing test**

```python
# tests/leyline/test_reports.py

import pytest
from esper.leyline.reports import SeedMetrics


def test_seed_metrics_has_interaction_fields():
    """SeedMetrics should have interaction tracking fields."""
    metrics = SeedMetrics()

    # Interaction sum: total synergy received from other seeds
    assert hasattr(metrics, "interaction_sum")
    assert metrics.interaction_sum == 0.0

    # Boost received: max single interaction from any other seed
    assert hasattr(metrics, "boost_received")
    assert metrics.boost_received == 0.0

    # Upstream alpha sum: total alpha of seeds in earlier slots
    assert hasattr(metrics, "upstream_alpha_sum")
    assert metrics.upstream_alpha_sum == 0.0

    # Downstream alpha sum: total alpha of seeds in later slots
    assert hasattr(metrics, "downstream_alpha_sum")
    assert metrics.downstream_alpha_sum == 0.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_reports.py::test_seed_metrics_has_interaction_fields -v`
Expected: FAIL (fields don't exist)

**Step 3: Add interaction fields to SeedMetrics**

```python
# src/esper/leyline/reports.py - Add after line 56 (_prev_contribution field)

    # Inter-slot interaction tracking (set by counterfactual engine)
    # interaction_sum: Σ I_ij for all j ≠ i (total synergy from interactions)
    interaction_sum: float = 0.0
    # boost_received: max(I_ij) for j ≠ i (strongest interaction partner)
    boost_received: float = 0.0
    # upstream_alpha_sum: Σ alpha_j for slots j < i (position-aware blending)
    upstream_alpha_sum: float = 0.0
    # downstream_alpha_sum: Σ alpha_j for slots j > i (position-aware blending)
    downstream_alpha_sum: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_reports.py::test_seed_metrics_has_interaction_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/leyline/reports.py tests/leyline/test_reports.py
git commit -m "feat(leyline): add interaction tracking fields to SeedMetrics

Scaffolding support Phase 2.1: Add fields for inter-slot influence:
- interaction_sum: total synergy from interactions with other seeds
- boost_received: strongest interaction from any single partner
- upstream_alpha_sum: total alpha of seeds in earlier slots
- downstream_alpha_sum: total alpha of seeds in later slots

These will be populated by counterfactual engine in Task 2.3."
```

### Task 2.2: Update feature extraction for interaction features

**Files:**
- Modify: `src/esper/tamiyo/policy/features.py:93, 144-311, 377-434`
- Test: `tests/tamiyo/policy/test_features.py`

**Step 1: Write the failing test**

```python
# tests/tamiyo/policy/test_features.py

def test_slot_features_include_interactions():
    """Verify slot features include interaction and topology fields."""
    from esper.tamiyo.policy.features import SLOT_FEATURE_SIZE

    # New layout: 1 is_active + 10 stage + 15 state + 13 blueprint = 39 dims
    # (was 35: 1 + 10 + 11 + 13)
    # Added 4 dims: interaction_sum, boost_received, upstream_alpha, downstream_alpha
    assert SLOT_FEATURE_SIZE == 39, f"Expected 39 dims/slot, got {SLOT_FEATURE_SIZE}"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py::test_slot_features_include_interactions -v`
Expected: FAIL (SLOT_FEATURE_SIZE is 35, not 39)

**Step 3: Update SLOT_FEATURE_SIZE constant**

```python
# src/esper/tamiyo/policy/features.py - Replace lines 86-98

# Per-slot features (schema v2 - one-hot stage + scaffolding):
# 1 is_active
# + 10 stage one-hot (SeedStage categorical encoding via StageSchema)
# + 15 state (alpha, contribution, velocity, tempo, alpha_target, alpha_mode,
#            alpha_steps_total, alpha_steps_done, time_to_target, alpha_velocity, alpha_algorithm,
#            interaction_sum, boost_received, upstream_alpha, downstream_alpha)
# + 13 blueprint one-hot
# Total: 1 + 10 + 15 + 13 = 39 dims per slot
SLOT_FEATURE_SIZE = 39

# Feature size (with telemetry off): 23 base + 3 slots * 39 features per slot = 140
# With telemetry on: + 3 slots * SeedTelemetry.feature_dim() (26) = 218 total
# NOTE: Default for 3-slot configuration. Use get_feature_size(slot_config) for dynamic slot counts.
MULTISLOT_FEATURE_SIZE = 140
```

**Step 4: Update obs_to_multislot_features slot feature extraction**

```python
# src/esper/tamiyo/policy/features.py - Add after alpha_algorithm_norm, before stage extraction (~line 260)

        # Interaction and topology features (scaffolding support)
        interaction_sum = safe(slot.get("interaction_sum", 0.0), 0.0, max_val=10.0) / 10.0
        boost_received = safe(slot.get("boost_received", 0.0), 0.0, max_val=5.0) / 5.0
        upstream_alpha = safe(slot.get("upstream_alpha_sum", 0.0), 0.0, max_val=3.0) / 3.0
        downstream_alpha = safe(slot.get("downstream_alpha_sum", 0.0), 0.0, max_val=3.0) / 3.0
```

```python
# In the slot_features.extend() call, add 4 new features after alpha_algorithm_norm:

        slot_features.extend([
            alpha,
            contribution / _IMPROVEMENT_CLAMP_PCT_PTS,
            contribution_velocity / _IMPROVEMENT_CLAMP_PCT_PTS,
            float(slot.get('blend_tempo_epochs', 5)) / 12.0,
            alpha_target,
            alpha_mode_norm,
            alpha_steps_total_norm,
            alpha_steps_done_norm,
            time_to_target_norm,
            alpha_velocity,
            alpha_algorithm_norm,
            # NEW: Scaffolding features
            interaction_sum,
            boost_received,
            upstream_alpha,
            downstream_alpha,
        ])
```

**Step 5: Update batch_obs_to_features**

```python
# src/esper/tamiyo/policy/features.py - In batch_obs_to_features, update slot feature section

                # Scaffolding features (offset + 22 to offset + 25, 4 dims)
                features[env_idx, offset + 22] = min(report.metrics.interaction_sum / 10.0, 1.0)
                features[env_idx, offset + 23] = min(report.metrics.boost_received / 5.0, 1.0)
                features[env_idx, offset + 24] = min(report.metrics.upstream_alpha_sum / 3.0, 1.0)
                features[env_idx, offset + 25] = min(report.metrics.downstream_alpha_sum / 3.0, 1.0)

                # Blueprint one-hot (offset + 26 to offset + 38, 13 dims) - shifted by 4
                bp_idx = _BLUEPRINT_TO_INDEX.get(report.blueprint_id, -1)
                if 0 <= bp_idx < _NUM_BLUEPRINT_TYPES:
                    features[env_idx, offset + 26 + bp_idx] = 1.0
```

**Step 6: Update docstrings**

Update the docstring in `obs_to_multislot_features` to document the new feature layout.

**Step 7: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py::test_slot_features_include_interactions -v`
Expected: PASS

**Step 8: Update all dimension assertions in tests**

Update `tests/tamiyo/policy/test_features.py` to expect new dimensions:
- SLOT_FEATURE_SIZE: 35 → 39
- MULTISLOT_FEATURE_SIZE: 128 → 140
- Per-slot feature indices shift for blueprint one-hot

**Step 9: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add src/esper/tamiyo/policy/features.py tests/tamiyo/policy/test_features.py
git commit -m "feat(tamiyo): add scaffolding observation features

Scaffolding support Phase 2.2: Extend slot observations with 4 new dims:
- interaction_sum: total synergy with other seeds (normalized 0-1)
- boost_received: strongest single interaction (normalized 0-1)
- upstream_alpha_sum: alpha of seeds in earlier slots (normalized 0-1)
- downstream_alpha_sum: alpha of seeds in later slots (normalized 0-1)

BREAKING: SLOT_FEATURE_SIZE 35→39, MULTISLOT_FEATURE_SIZE 128→140.
Existing checkpoints are incompatible."
```

### Task 2.3: Populate interaction metrics in vectorized training

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:1976-1985`
- Test: `tests/integration/test_vectorized_scaffolding.py` (new)

**Step 1: Write the failing test**

```python
# tests/integration/test_vectorized_scaffolding.py

import pytest
from esper.leyline.reports import SeedMetrics


def test_interaction_metrics_populated_after_counterfactual():
    """Verify interaction metrics are computed from counterfactual matrix."""
    # This test will verify the integration once implemented
    metrics = SeedMetrics()

    # After counterfactual validation with two interacting seeds,
    # the interaction metrics should be non-zero
    # For now, just verify the fields exist
    assert metrics.interaction_sum == 0.0
    assert metrics.boost_received == 0.0


def test_alpha_topology_computed():
    """Verify upstream/downstream alpha sums are computed from slot positions."""
    metrics = SeedMetrics()

    # After step() with multiple active seeds, topology features should be set
    assert metrics.upstream_alpha_sum == 0.0
    assert metrics.downstream_alpha_sum == 0.0
```

**Step 2: Run test to verify baseline**

Run: `PYTHONPATH=src uv run pytest tests/integration/test_vectorized_scaffolding.py -v`
Expected: PASS (tests are minimal stubs)

**Step 3: Add interaction computation in vectorized.py**

```python
# src/esper/simic/training/vectorized.py - After counterfactual matrix reporting (~line 1985)

                # Compute interaction terms and populate scaffolding metrics
                if len(active_slots) >= 2 and i in pair_accs:
                    all_off_acc = all_disabled_accs.get(i, 0.0)
                    for (slot_a, slot_b), pair_acc in pair_accs[i].items():
                        solo_a = baseline_accs[i].get(slot_a, 0.0)
                        solo_b = baseline_accs[i].get(slot_b, 0.0)
                        # I_ij = f({i,j}) - f({i}) - f({j}) + f(empty)
                        interaction = pair_acc - solo_a - solo_b + all_off_acc

                        # Update metrics for both seeds
                        if env_state.model.has_active_seed_in_slot(slot_a):
                            seed_a = env_state.model.seed_slots[slot_a].state
                            if seed_a and seed_a.metrics:
                                seed_a.metrics.interaction_sum += interaction
                                seed_a.metrics.boost_received = max(
                                    seed_a.metrics.boost_received, interaction
                                )

                        if env_state.model.has_active_seed_in_slot(slot_b):
                            seed_b = env_state.model.seed_slots[slot_b].state
                            if seed_b and seed_b.metrics:
                                seed_b.metrics.interaction_sum += interaction
                                seed_b.metrics.boost_received = max(
                                    seed_b.metrics.boost_received, interaction
                                )

                # Compute topology features (upstream/downstream alpha sums)
                for slot_idx, slot_id in enumerate(ordered_slots):
                    if not env_state.model.has_active_seed_in_slot(slot_id):
                        continue
                    seed_state = env_state.model.seed_slots[slot_id].state
                    if seed_state is None or seed_state.metrics is None:
                        continue

                    upstream_sum = 0.0
                    downstream_sum = 0.0
                    for other_idx, other_id in enumerate(ordered_slots):
                        if other_id == slot_id:
                            continue
                        if not env_state.model.has_active_seed_in_slot(other_id):
                            continue
                        other_state = env_state.model.seed_slots[other_id].state
                        if other_state is None:
                            continue

                        other_alpha = other_state.metrics.current_alpha if other_state.metrics else 0.0
                        if other_idx < slot_idx:
                            upstream_sum += other_alpha
                        else:
                            downstream_sum += other_alpha

                    seed_state.metrics.upstream_alpha_sum = upstream_sum
                    seed_state.metrics.downstream_alpha_sum = downstream_sum
```

**Step 4: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/integration/test_vectorized_scaffolding.py -v`
Expected: PASS

**Step 5: Run smoke test**

Run: `PYTHONPATH=src timeout 60 uv run python -m esper.scripts.train ppo --episodes 2 --n-envs 2 --max-epochs 5`
Expected: Training completes without errors

**Step 6: Commit**

```bash
git add src/esper/simic/training/vectorized.py tests/integration/test_vectorized_scaffolding.py
git commit -m "feat(simic): compute scaffolding metrics from counterfactual matrix

Scaffolding support Phase 2.3: Populate interaction metrics after
counterfactual validation:
- interaction_sum: computed from pair accuracy matrix using I_ij formula
- boost_received: max interaction with any partner
- upstream_alpha_sum: sum of alpha for seeds in earlier slots
- downstream_alpha_sum: sum of alpha for seeds in later slots

These metrics flow to observations via SeedMetrics → SeedStateReport."
```

---

## Phase 3: Reward Shaping for Scaffolding

**Why:** Scaffolding creates delayed credit assignment: seed A's value is only realized when seed B succeeds later. We need (1) synergy bonus for positive interactions, (2) scaffold hindsight credit when beneficiary succeeds, and (3) higher GAE lambda (0.98) for longer credit horizon.

### Task 3.1: Add synergy bonus to reward computation

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py:143-200`
- Test: `tests/simic/rewards/test_scaffolding_rewards.py` (new)

**Step 1: Write the failing test**

```python
# tests/simic/rewards/test_scaffolding_rewards.py

import pytest
from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig, SeedInfo
from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp


def test_synergy_bonus_added_for_positive_interaction():
    """Verify synergy bonus is added when interaction_sum > 0."""
    config = ContributionRewardConfig()

    seed_info = SeedInfo(
        stage=SeedStage.BLENDING,
        epochs_in_stage=3,
        seed_param_count=10000,
        host_param_count=100000,
        total_params=110000,
        param_budget=500000,
        interaction_sum=2.5,  # Strong positive interaction
        boost_received=1.2,
    )

    reward_with_synergy = compute_contribution_reward(
        action=LifecycleOp.CONTINUE,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        config=config,
    )

    # Same but with no interaction
    seed_info_no_synergy = SeedInfo(
        stage=SeedStage.BLENDING,
        epochs_in_stage=3,
        seed_param_count=10000,
        host_param_count=100000,
        total_params=110000,
        param_budget=500000,
        interaction_sum=0.0,
        boost_received=0.0,
    )

    reward_no_synergy = compute_contribution_reward(
        action=LifecycleOp.CONTINUE,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info_no_synergy,
        epoch=10,
        max_epochs=25,
        config=config,
    )

    assert reward_with_synergy > reward_no_synergy, (
        f"Expected synergy bonus: {reward_with_synergy} > {reward_no_synergy}"
    )
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/rewards/test_scaffolding_rewards.py::test_synergy_bonus_added_for_positive_interaction -v`
Expected: FAIL (SeedInfo doesn't have interaction fields yet, or bonus not computed)

**Step 3: Add interaction fields to SeedInfo**

```python
# src/esper/simic/rewards/rewards.py - Add to SeedInfo NamedTuple (around line 180)

class SeedInfo(NamedTuple):
    """Seed state for reward computation."""
    stage: SeedStage
    epochs_in_stage: int
    seed_param_count: int
    host_param_count: int
    total_params: int
    param_budget: int
    # Stage transition info (for PBRS)
    previous_stage: SeedStage = SeedStage.UNKNOWN
    previous_epochs_in_stage: int = 0
    # Scaffolding support
    interaction_sum: float = 0.0
    boost_received: float = 0.0
```

**Step 4: Add synergy bonus computation**

```python
# src/esper/simic/rewards/rewards.py - Add synergy bonus function (after PBRS functions)

def _compute_synergy_bonus(
    interaction_sum: float,
    boost_received: float,
    synergy_weight: float = 0.1,
) -> float:
    """Compute synergy bonus for scaffolding behavior.

    Rewards seeds that have positive interactions with others.
    Uses tanh to bound the bonus and prevent reward hacking.

    Args:
        interaction_sum: Total interaction I_ij with all other seeds
        boost_received: Maximum single interaction
        synergy_weight: Scaling factor for bonus (default 0.1)

    Returns:
        Bounded synergy bonus in [0, synergy_weight]
    """
    if interaction_sum <= 0:
        return 0.0

    # Tanh bounds the bonus, preventing runaway positive feedback
    raw_bonus = math.tanh(interaction_sum * 0.5)
    return raw_bonus * synergy_weight
```

**Step 5: Integrate synergy bonus into compute_contribution_reward**

```python
# In compute_contribution_reward, after PBRS and before return:

    # Scaffolding: synergy bonus for positive interactions
    synergy_bonus = _compute_synergy_bonus(
        seed_info.interaction_sum,
        seed_info.boost_received,
    )
    total_reward += synergy_bonus
```

**Step 6: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/rewards/test_scaffolding_rewards.py::test_synergy_bonus_added_for_positive_interaction -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/esper/simic/rewards/rewards.py tests/simic/rewards/test_scaffolding_rewards.py
git commit -m "feat(simic): add synergy bonus for scaffolding interactions

Scaffolding support Phase 3.1: Add synergy bonus to reward computation:
- _compute_synergy_bonus(): bounded bonus for positive interaction_sum
- Uses tanh to prevent reward hacking from artificial interactions
- Integrated into compute_contribution_reward

Seeds that enhance other seeds now receive credit proportional to
the strength of their interaction."
```

### Task 3.2: Add scaffold hindsight credit

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py`
- Modify: `src/esper/simic/training/vectorized.py`
- Test: `tests/simic/rewards/test_scaffolding_rewards.py`

**Step 1: Write the failing test**

```python
# tests/simic/rewards/test_scaffolding_rewards.py

def test_scaffold_credit_on_beneficiary_success():
    """Verify scaffold credit is added when beneficiary fossilizes."""
    # When seed B fossilizes successfully after having received boost from seed A,
    # seed A (if still alive) should receive retroactive credit
    from esper.simic.rewards import compute_scaffold_hindsight_credit

    # Seed A provided boost of 1.5 to seed B
    # Seed B just fossilized with 3% improvement
    credit = compute_scaffold_hindsight_credit(
        boost_given=1.5,
        beneficiary_improvement=3.0,
        credit_weight=0.2,
    )

    assert credit > 0, f"Expected positive credit, got {credit}"
    assert credit <= 0.2, f"Credit should be bounded by weight"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/rewards/test_scaffolding_rewards.py::test_scaffold_credit_on_beneficiary_success -v`
Expected: FAIL (function doesn't exist)

**Step 3: Add scaffold hindsight credit function**

```python
# src/esper/simic/rewards/rewards.py

def compute_scaffold_hindsight_credit(
    boost_given: float,
    beneficiary_improvement: float,
    credit_weight: float = 0.2,
) -> float:
    """Compute retroactive credit for scaffold seeds.

    When a beneficiary seed fossilizes successfully, the scaffold seed
    that boosted it receives credit proportional to its contribution.

    This implements Hindsight Credit Assignment for scaffolding:
    the scaffold's value is only known after the beneficiary succeeds.

    Args:
        boost_given: The interaction term I_ij that scaffold provided
        beneficiary_improvement: The improvement the beneficiary achieved
        credit_weight: Maximum credit amount (default 0.2)

    Returns:
        Credit in [0, credit_weight]
    """
    if boost_given <= 0 or beneficiary_improvement <= 0:
        return 0.0

    # Credit is proportional to boost given and beneficiary success
    raw_credit = math.tanh(boost_given * beneficiary_improvement * 0.1)
    return raw_credit * credit_weight
```

**Step 4: Add __all__ export**

```python
# src/esper/simic/rewards/__init__.py - Add export

    "compute_scaffold_hindsight_credit",
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/rewards/test_scaffolding_rewards.py::test_scaffold_credit_on_beneficiary_success -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/rewards/rewards.py src/esper/simic/rewards/__init__.py tests/simic/rewards/test_scaffolding_rewards.py
git commit -m "feat(simic): add scaffold hindsight credit function

Scaffolding support Phase 3.2: Add compute_scaffold_hindsight_credit()
for retroactive reward relabeling when beneficiary succeeds.

This enables credit assignment for scaffolding behavior where the
scaffold's value is only realized after the beneficiary fossilizes.
Integration with training loop deferred to later task."
```

### Task 3.3: Increase GAE lambda for longer credit horizon

**Files:**
- Modify: `src/esper/leyline/__init__.py:89`
- Modify: `src/esper/simic/training/config.py` (add gae_lambda field)
- Test: `tests/simic/test_training_config.py`

**Step 1: Write the failing test**

```python
# tests/simic/test_training_config.py (add to existing)

def test_gae_lambda_configurable():
    """Verify GAE lambda can be configured for scaffolding."""
    from esper.simic.training.config import TrainingConfig

    config = TrainingConfig(gae_lambda=0.98)
    assert config.gae_lambda == 0.98


def test_gae_lambda_default():
    """Verify default GAE lambda is suitable for scaffolding (0.98)."""
    from esper.simic.training.config import TrainingConfig

    config = TrainingConfig()
    # DRL expert recommendation: 0.98 for longer credit horizon
    assert config.gae_lambda == 0.98
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_training_config.py::test_gae_lambda_configurable -v`
Expected: FAIL (field doesn't exist)

**Step 3: Update DEFAULT_GAE_LAMBDA in leyline**

```python
# src/esper/leyline/__init__.py - Update line 89

# DRL expert recommendation for scaffolding: 0.98 for longer credit horizon
# (was 0.97; increased to capture delayed scaffold effects)
DEFAULT_GAE_LAMBDA = 0.98
```

**Step 4: Add gae_lambda to TrainingConfig**

```python
# src/esper/simic/training/config.py - Add field after gamma

    gae_lambda: float = DEFAULT_GAE_LAMBDA
```

**Step 5: Update to_ppo_kwargs**

```python
# src/esper/simic/training/config.py - Add to to_ppo_kwargs return dict

            "gae_lambda": self.gae_lambda,
```

**Step 6: Update imports**

```python
# src/esper/simic/training/config.py - Add to imports

from esper.leyline import (
    DEFAULT_GAMMA,
    DEFAULT_GAE_LAMBDA,  # NEW
    # ... other imports
)
```

**Step 7: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_training_config.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add src/esper/leyline/__init__.py src/esper/simic/training/config.py tests/simic/test_training_config.py
git commit -m "feat(simic): increase GAE lambda to 0.98 for scaffolding

Scaffolding support Phase 3.3: DRL expert recommendation for longer
credit horizon. Higher lambda (0.98 vs 0.97) captures delayed scaffold
effects where value is only realized 5-10 epochs later.

Also adds gae_lambda to TrainingConfig for configurability."
```

---

## Phase 4: Optional Enhancements

**Why:** These are optional improvements that can be added if Phase 1-3 prove effective.

### Task 4.1: Add auxiliary scaffold_value_head (Optional)

**Files:**
- Modify: `src/esper/simic/agent/network.py`
- Modify: `src/esper/simic/agent/ppo.py`
- Test: `tests/simic/test_scaffold_value_head.py` (new)

**Note:** This task is marked optional. Only implement if Phase 1-3 results show that Tamiyo struggles to attribute scaffolding credit accurately.

**Design:**
- Add a second value head that predicts "scaffold value" (expected value if this seed is used as a scaffold)
- Train it with auxiliary loss on episodes where scaffolding occurred
- Use it to bias action selection toward scaffold-compatible configurations

**Step 1: Write the failing test (stub)**

```python
# tests/simic/test_scaffold_value_head.py

import pytest

@pytest.mark.skip(reason="Optional Phase 4 - implement if needed")
def test_scaffold_value_head_exists():
    """Verify network has auxiliary scaffold value head."""
    from esper.simic.agent.network import FactoredRecurrentActorCritic

    network = FactoredRecurrentActorCritic(
        state_dim=140,
        slot_config=None,  # Use default
        include_scaffold_head=True,
    )

    assert hasattr(network, "scaffold_value_head")
```

**Step 2: Commit placeholder**

```bash
git add tests/simic/test_scaffold_value_head.py
git commit -m "chore(simic): add placeholder for optional scaffold value head

Scaffolding support Phase 4.1 placeholder: auxiliary value head for
explicit scaffold value prediction. Implement only if Phase 1-3 results
show credit assignment difficulties.

Currently skipped - will be enabled if needed."
```

### Task 4.2: Add temporal precedence features (Optional)

**Files:**
- Modify: `src/esper/leyline/reports.py`
- Modify: `src/esper/tamiyo/policy/features.py`
- Test: `tests/tamiyo/policy/test_temporal_features.py` (new)

**Note:** This task is marked optional. Only implement if scaffolding behavior requires explicit temporal ordering information.

**Design:**
- Add `germination_order` field to SeedMetrics (which seed germinated first)
- Add `relative_age` feature (epochs_this_seed / max_epochs_any_seed)
- These help Tamiyo learn "older seed scaffolds younger seed" patterns

**Step 1: Write the failing test (stub)**

```python
# tests/tamiyo/policy/test_temporal_features.py

import pytest

@pytest.mark.skip(reason="Optional Phase 4 - implement if needed")
def test_germination_order_in_observations():
    """Verify germination order is available in observations."""
    pass
```

**Step 2: Commit placeholder**

```bash
git add tests/tamiyo/policy/test_temporal_features.py
git commit -m "chore(tamiyo): add placeholder for optional temporal precedence features

Scaffolding support Phase 4.2 placeholder: germination order and
relative age features for explicit temporal precedence learning.

Currently skipped - will be enabled if scaffolding requires
explicit temporal ordering beyond what LSTM captures."
```

---

## Verification Checklist

After completing all phases, verify:

- [ ] `PYTHONPATH=src uv run pytest tests/tamiyo/policy/test_features.py -v` - All feature tests pass
- [ ] `PYTHONPATH=src uv run pytest tests/leyline/ -v` - All leyline tests pass
- [ ] `PYTHONPATH=src uv run pytest tests/simic/rewards/ -v` - All reward tests pass
- [ ] `PYTHONPATH=src timeout 120 uv run python -m esper.scripts.train ppo --episodes 5 --n-envs 2` - Training runs without errors
- [ ] Observation dimension matches network input dimension (no shape errors)
- [ ] Telemetry shows interaction metrics being populated

---

## Breaking Changes Summary

| Phase | Change | Migration |
|-------|--------|-----------|
| 1 | `fossilize_value` → `contribution_velocity` | Retrain from scratch |
| 2 | SLOT_FEATURE_SIZE 35→39, MULTISLOT 128→140 | Retrain from scratch |
| 3 | DEFAULT_GAE_LAMBDA 0.97→0.98 | Retrain from scratch |

**All phases require fresh training runs. Existing checkpoints are incompatible.**
