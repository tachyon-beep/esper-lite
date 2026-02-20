# Post-Fossilization Drip Reward for BASIC Mode

```yaml
# Plan Metadata
id: post-fossilization-drip-reward
title: Post-Fossilization Drip Reward for BASIC Mode
type: ready
created: 2026-01-12
updated: 2026-01-12
owner: Claude

# Prioritization
urgency: high
value: Prevents seeds from gaming by fossilizing at peak metrics then degrading; enforces long-term accountability for fossilized seeds

# Constraints
complexity: M
risk: medium
risk_notes: |
  - PPO variance could increase if drip magnitude is miscalibrated
  - Credit assignment becomes more complex (rewards arrive after action)
  - Potential for negative drip spiral if seed genuinely degrades post-fossilization
  - Must ensure drip normalization prevents early-fossilization gaming

# Dependencies
depends_on: []
soft_depends:
  - simic2-phase1-vectorized-split  # Clean module structure helps

blocks: []

# Status
status_notes: Design complete after DRL Expert review. Ready for implementation.
percent_complete: 0

# Expert Review (REQUIRED before promotion to ready)
reviewed_by:
  - reviewer: drl-expert
    date: 2026-01-12
    verdict: approved
    notes: |
      70/30 split is appropriate for credit assignment vs accountability tradeoff.
      Epoch-normalized drip scale is critical to prevent early-fossilization gaming.
      Drip clipping at +/-0.1 prevents variance explosion.
      Contribution-only for drip (not harmonic mean) is correct since improvement is fixed at fossilization.
```

---

## Problem Statement

In BASIC reward mode, the FOSSILIZE action gives an immediate reward based on `f(improvement, contribution)` using harmonic mean. The problem: **seeds can game by fossilizing when metrics look good, then degrading post-fossilization**. There's no accountability after fossilization.

### Observed Gaming Pattern

1. Seed trains to a local optimum where contribution looks high
2. Policy fossilizes immediately to capture the bonus
3. Post-fossilization, the frozen seed parameters cause interference
4. Host accuracy degrades, but the seed already got its reward
5. Net effect: Policy learns to "lock in gains" at peak metrics regardless of long-term value

### Why Current Design Fails

The current design pays 100% of the fossilize bonus immediately at the FOSSILIZE action (P0-style credit assignment). While this is correct for credit assignment (the action that caused fossilization should receive the reward), it creates a **moral hazard**: the policy has no incentive to ensure the fossilized seed remains beneficial.

---

## Solution Design

### Core Concept: Drip Reward

Split the fossilize reward into two components:
1. **Immediate reward (30%)**: Paid at FOSSILIZE action for credit assignment
2. **Drip reward (70%)**: Paid over remaining epochs based on continued contribution

The drip reward creates **post-fossilization accountability**: if the seed degrades after fossilization, the drip becomes negative (penalty).

### Key Design Decisions

#### D1: 70/30 Split Ratio

**Decision**: 30% immediate, 70% drip

**Rationale** (DRL Expert review):
- 30% immediate preserves credit assignment for the FOSSILIZE decision
- 70% drip provides strong accountability signal
- More aggressive splits (90/10) would make FOSSILIZE action appear low-value in early training
- Less aggressive splits (50/50) provide insufficient accountability

**Config field**: `drip_fraction: float = 0.7`

#### D2: Epoch-Normalized Drip Scale

**Decision**: Normalize drip by remaining epochs to prevent early-fossilization gaming

**Formula**:
```python
remaining_epochs = max_epochs - fossilize_epoch
drip_scale = drip_total / remaining_epochs
per_epoch_drip = drip_scale * current_contribution
```

**Rationale**:
- Without normalization, fossilizing at epoch 10 (140 remaining epochs) would receive much more total drip than fossilizing at epoch 140 (10 remaining epochs)
- This would incentivize early fossilization to maximize drip opportunities
- Epoch normalization makes expected total drip equal regardless of fossilization timing

**Edge case**: If `remaining_epochs <= min_drip_epochs`, use `min_drip_epochs` as denominator to prevent division by near-zero.

#### D3: Contribution-Only for Drip (Not Harmonic Mean)

**Decision**: Use `seed_contribution` directly for drip calculation, not `f(improvement, contribution)`

**Rationale**:
- Improvement is **fixed at fossilization** - the seed's total_improvement doesn't change after fossilization
- Only contribution can change post-fossilization (counterfactual validation recomputes it each epoch)
- Using harmonic mean would always use the stale improvement value
- Contribution alone captures "is this seed still helping?"

**Formula**:
```python
drip_this_epoch = drip_scale * seed_contribution  # Can be negative if contribution < 0
```

#### D4: Drip Clipping for Variance Control

**Decision**: Clip drip to `[-max_drip_per_epoch, +max_drip_per_epoch]`

**Default**: `max_drip_per_epoch = 0.1`

**Rationale**:
- Prevents variance explosion from extreme contribution values
- +/-0.1 is ~10% of typical episode reward magnitude
- Allows meaningful signal without destabilizing PPO

#### D5: Diminishing Returns for Multiple Fossilized Seeds

**Decision**: Apply diminishing returns discount to drip based on fossilized seed count

**Formula**:
```python
# Discount based on number of fossilized seeds
diminishing_factor = 1.0 / (1.0 + 0.2 * (num_fossilized - 1))
effective_drip = base_drip * diminishing_factor
```

**Rationale**:
- Prevents "fossil farming" where policy fossilizes many mediocre seeds
- First fossil gets full drip, subsequent fossils get diminishing drip
- Coefficient 0.2 means: 2 fossils = 83%, 3 fossils = 71%, 5 fossils = 56%

#### D6: Optional min_drip_epochs Guard

**Decision**: Allow optional configuration to block fossilization if fewer than `min_drip_epochs` remain

**Default**: `min_drip_epochs = 5` (disabled by default, set to 0)

**Rationale**:
- Last-minute fossilization can game the system (capture immediate bonus, minimal drip exposure)
- When enabled, FOSSILIZE is blocked if `max_epochs - epoch < min_drip_epochs`
- This is optional because some training runs may legitimately fossilize late

---

## Implementation Tasks

### Task 1: Add Drip Configuration Fields

**File**: `/home/john/esper-lite/src/esper/simic/training/config.py`

Add to `TrainingConfig`:

```python
# === Drip Reward Configuration (BASIC mode only) ===
# Post-fossilization accountability: drip reward paid over remaining epochs
# based on continued seed contribution.

# Fraction of fossilize reward paid as drip (vs immediate)
# 0.7 = 70% drip, 30% immediate; 0.0 = disable drip entirely
drip_fraction: float = 0.7

# Maximum drip per epoch (prevents variance explosion)
max_drip_per_epoch: float = 0.1

# Minimum epochs required for drip payout (blocks last-minute fossilization)
# Set to 0 to disable this guard
min_drip_epochs: int = 0

# Diminishing returns coefficient for multiple fossilized seeds
# Higher = faster diminishing; 0.0 = no diminishing returns
drip_diminishing_coef: float = 0.2
```

Add validation in `_validate()`:

```python
if self.drip_fraction < 0.0 or self.drip_fraction > 1.0:
    raise ValueError("drip_fraction must be in [0.0, 1.0]")
if self.max_drip_per_epoch <= 0:
    raise ValueError("max_drip_per_epoch must be positive")
if self.min_drip_epochs < 0:
    raise ValueError("min_drip_epochs must be non-negative")
if self.drip_diminishing_coef < 0:
    raise ValueError("drip_diminishing_coef must be non-negative")
```

### Task 2: Add Drip Configuration to ContributionRewardConfig

**File**: `/home/john/esper-lite/src/esper/simic/rewards/contribution.py`

Add to `ContributionRewardConfig`:

```python
# === Drip Reward Configuration (BASIC mode only) ===
# Fraction of fossilize reward paid as drip (vs immediate)
drip_fraction: float = 0.7

# Maximum drip per epoch (clips to prevent variance explosion)
max_drip_per_epoch: float = 0.1

# Minimum remaining epochs to allow fossilization (0 = disabled)
min_drip_epochs: int = 0

# Diminishing returns coefficient for multiple fossils
drip_diminishing_coef: float = 0.2
```

### Task 3: Add Drip State Tracking Dataclass

**File**: `/home/john/esper-lite/src/esper/simic/rewards/contribution.py`

Add new dataclass to track drip state per fossilized seed:

```python
@dataclass(slots=True)
class FossilizedSeedDripState:
    """Tracks drip reward state for a fossilized seed.

    Created at fossilization, updated each epoch until episode end.
    """
    seed_id: str
    slot_id: str
    fossilize_epoch: int
    max_epochs: int

    # Total drip pool (70% of original fossilize bonus)
    drip_total: float

    # Per-epoch drip scale (normalized by remaining epochs)
    drip_scale: float

    # Accumulated drip paid so far
    drip_paid: float = 0.0

    @property
    def remaining_epochs(self) -> int:
        """Epochs remaining when seed was fossilized."""
        return self.max_epochs - self.fossilize_epoch

    def compute_epoch_drip(
        self,
        current_contribution: float,
        num_fossilized: int,
        max_drip: float,
        diminishing_coef: float,
    ) -> float:
        """Compute drip for this epoch.

        Args:
            current_contribution: Counterfactual contribution this epoch
            num_fossilized: Total fossilized seeds (for diminishing returns)
            max_drip: Maximum absolute drip per epoch
            diminishing_coef: Diminishing returns coefficient

        Returns:
            Clipped drip amount for this epoch
        """
        # Base drip from contribution
        base_drip = self.drip_scale * current_contribution

        # Apply diminishing returns for multiple fossils
        if diminishing_coef > 0 and num_fossilized > 1:
            diminishing_factor = 1.0 / (1.0 + diminishing_coef * (num_fossilized - 1))
            base_drip *= diminishing_factor

        # Clip to prevent variance explosion
        clipped_drip = max(-max_drip, min(max_drip, base_drip))

        return clipped_drip
```

### Task 4: Modify compute_basic_reward for Drip

**File**: `/home/john/esper-lite/src/esper/simic/rewards/contribution.py`

Modify `compute_basic_reward` to:
1. Split fossilize bonus into immediate and drip pool
2. Return drip state for newly fossilized seeds
3. Accept and process existing drip states

Updated signature and implementation:

```python
def compute_basic_reward(
    *,
    acc_delta: float,
    effective_seed_params: float | None,
    total_params: int,
    host_params: int,
    config: ContributionRewardConfig,
    epoch: int,
    max_epochs: int,
    seed_info: SeedInfo | None = None,
    action: LifecycleOp = LifecycleOp.WAIT,
    seed_contribution: float | None = None,
    # NEW: Drip-related parameters
    fossilized_drip_states: list[FossilizedSeedDripState] | None = None,
    fossilized_contributions: dict[str, float] | None = None,  # seed_id -> contribution
) -> tuple[float, float, float, float, float, FossilizedSeedDripState | None, float]:
    """Compute BASIC reward with drip support.

    Returns:
        tuple of (total_reward, rent_penalty, growth_ratio, pbrs_bonus,
                  fossilize_bonus, new_drip_state, drip_this_epoch)

        new_drip_state: FossilizedSeedDripState if FOSSILIZE action created one, else None
        drip_this_epoch: Total drip reward from all fossilized seeds this epoch
    """
    # ... existing code for PBRS and rent ...

    # =========================================================================
    # Component 3: Contribution-Scaled Fossilization with Drip Split
    # =========================================================================
    fossilize_bonus = 0.0
    new_drip_state: FossilizedSeedDripState | None = None

    if action == LifecycleOp.FOSSILIZE and seed_info is not None:
        is_valid = seed_info.stage == STAGE_HOLDING

        # Check min_drip_epochs guard
        remaining_epochs = max_epochs - epoch
        if config.min_drip_epochs > 0 and remaining_epochs < config.min_drip_epochs:
            # Block fossilization - treat as invalid
            fossilize_bonus = config.basic_fossilize_invalid_penalty
        elif is_valid:
            improvement = seed_info.total_improvement
            meets_threshold = (
                seed_contribution is not None
                and seed_contribution >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
            )

            if improvement > 0 and meets_threshold:
                # Calculate full bonus
                combined = _compute_attributed_value(
                    progress=improvement,
                    seed_contribution=seed_contribution,
                    formula=config.attribution_formula,
                )
                legitimacy_discount = min(1.0, seed_info.epochs_in_stage / MIN_HOLDING_EPOCHS)
                full_bonus = (
                    config.basic_fossilize_base_bonus
                    + config.basic_contribution_scale * combined
                ) * legitimacy_discount

                # Split into immediate and drip
                immediate_fraction = 1.0 - config.drip_fraction
                fossilize_bonus = full_bonus * immediate_fraction

                # Create drip state for the drip portion
                if config.drip_fraction > 0 and remaining_epochs > 0:
                    drip_total = full_bonus * config.drip_fraction
                    drip_scale = drip_total / remaining_epochs
                    new_drip_state = FossilizedSeedDripState(
                        seed_id=seed_info.seed_id if hasattr(seed_info, 'seed_id') else "unknown",
                        slot_id=seed_info.slot_id if hasattr(seed_info, 'slot_id') else "unknown",
                        fossilize_epoch=epoch,
                        max_epochs=max_epochs,
                        drip_total=drip_total,
                        drip_scale=drip_scale,
                    )
            elif improvement <= 0 and seed_contribution is not None and seed_contribution > 0.1:
                fossilize_bonus = config.basic_fossilize_invalid_penalty
            else:
                fossilize_bonus = config.basic_fossilize_noncontributing_penalty
        else:
            fossilize_bonus = config.basic_fossilize_invalid_penalty

    reward += fossilize_bonus

    # =========================================================================
    # Component 5: Drip Payout for Existing Fossilized Seeds
    # =========================================================================
    drip_this_epoch = 0.0
    if fossilized_drip_states and fossilized_contributions:
        num_fossilized = len(fossilized_drip_states)
        for drip_state in fossilized_drip_states:
            contribution = fossilized_contributions.get(drip_state.seed_id, 0.0)
            epoch_drip = drip_state.compute_epoch_drip(
                current_contribution=contribution,
                num_fossilized=num_fossilized,
                max_drip=config.max_drip_per_epoch,
                diminishing_coef=config.drip_diminishing_coef,
            )
            drip_this_epoch += epoch_drip
            drip_state.drip_paid += epoch_drip

    reward += drip_this_epoch

    # ... existing terminal bonus code ...

    return reward, rent_penalty, growth_ratio, pbrs_bonus, fossilize_bonus, new_drip_state, drip_this_epoch
```

### Task 5: Add Drip Telemetry Fields

**File**: `/home/john/esper-lite/src/esper/simic/rewards/reward_telemetry.py`

Add to `RewardComponentsTelemetry`:

```python
# Drip reward fields (BASIC mode post-fossilization accountability)
drip_this_epoch: float = 0.0  # Total drip paid this epoch
drip_immediate_bonus: float = 0.0  # Immediate portion of fossilize bonus (30%)
drip_deferred_total: float = 0.0  # Total deferred drip pool created this epoch
num_drip_sources: int = 0  # Number of fossilized seeds contributing drip
```

Update `to_dict()` and `from_dict()` to include these fields.

### Task 6: Update Telemetry Payload

**File**: `/home/john/esper-lite/src/esper/leyline/telemetry.py`

Add drip fields to `AnalyticsSnapshotPayload` for `kind="reward_summary"`:

```python
# Drip reward fields (BASIC mode)
drip_this_epoch: float | None = None
drip_sources_count: int | None = None
```

### Task 7: Add Seed ID/Slot ID to SeedInfo

**File**: `/home/john/esper-lite/src/esper/simic/rewards/types.py`

Ensure `SeedInfo` includes seed_id and slot_id for drip tracking:

```python
@dataclass(slots=True)
class SeedInfo:
    # ... existing fields ...

    # Identity fields for drip tracking
    seed_id: str = ""
    slot_id: str = ""
```

### Task 8: Integrate Drip State Management in Vectorized Trainer

**File**: `/home/john/esper-lite/src/esper/simic/training/vectorized.py`

Add drip state tracking per environment:

```python
# In environment state tracking
self._drip_states: list[list[FossilizedSeedDripState]] = [[] for _ in range(n_envs)]

# In reward computation loop, after fossilization:
if new_drip_state is not None:
    self._drip_states[env_id].append(new_drip_state)

# Pass drip states to compute_basic_reward:
reward, rent, growth, pbrs, foss_bonus, new_drip, drip_epoch = compute_basic_reward(
    # ... existing args ...
    fossilized_drip_states=self._drip_states[env_id],
    fossilized_contributions=fossilized_contributions,  # from counterfactual validation
)

# Reset drip states at episode boundary
def _reset_env(self, env_id: int) -> None:
    # ... existing reset code ...
    self._drip_states[env_id].clear()
```

### Task 9: Add Golden Tests for Drip Reward

**File**: `/home/john/esper-lite/tests/simic/rewards/test_reward_golden.py`

```python
def test_basic_mode_fossilize_drip_split() -> None:
    """FOSSILIZE in BASIC mode splits reward into immediate and drip pool.

    With drip_fraction=0.7, only 30% of the bonus is paid immediately.
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        drip_fraction=0.7,
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
    )
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,  # Full legitimacy
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    reward, _, _, _, foss_bonus, new_drip, _ = compute_basic_reward(
        acc_delta=5.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=20,
        max_epochs=150,
        seed_info=seed_info,
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
    )

    # Full bonus would be (0.3 + 0.5 * 5.0) * 1.0 = 2.8 (harmonic(5,5)=5)
    # Immediate = 30% = 0.84
    assert foss_bonus == pytest.approx(0.84, abs=0.01)

    # Drip state should be created
    assert new_drip is not None
    assert new_drip.drip_total == pytest.approx(1.96, abs=0.01)  # 70% of 2.8
    assert new_drip.remaining_epochs == 130  # 150 - 20


def test_basic_mode_drip_payout_positive() -> None:
    """Drip payout rewards continued positive contribution."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=20,
        max_epochs=150,
        drip_total=1.96,
        drip_scale=1.96 / 130,  # ~0.015 per epoch
    )

    reward, _, _, _, _, _, drip_epoch = compute_basic_reward(
        acc_delta=0.1,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=25,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": 3.0},  # Still contributing
    )

    # Drip = drip_scale * contribution = 0.015 * 3.0 = 0.045
    assert drip_epoch == pytest.approx(0.045, abs=0.01)


def test_basic_mode_drip_payout_negative() -> None:
    """Drip payout penalizes negative contribution (seed now hurting)."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=20,
        max_epochs=150,
        drip_total=1.96,
        drip_scale=1.96 / 130,
    )

    reward, _, _, _, _, _, drip_epoch = compute_basic_reward(
        acc_delta=-1.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=25,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": -2.0},  # Now hurting!
    )

    # Drip = drip_scale * contribution = 0.015 * (-2.0) = -0.03
    assert drip_epoch == pytest.approx(-0.03, abs=0.01)
    assert drip_epoch < 0, "Negative contribution should produce negative drip"


def test_basic_mode_drip_clipping() -> None:
    """Drip is clipped to max_drip_per_epoch to prevent variance explosion."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=140,  # Late fossilization = high drip_scale
        max_epochs=150,
        drip_total=1.96,
        drip_scale=1.96 / 10,  # ~0.196 per epoch (high)
    )

    reward, _, _, _, _, _, drip_epoch = compute_basic_reward(
        acc_delta=0.1,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=145,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": 5.0},  # High contribution
    )

    # Uncapped: 0.196 * 5.0 = 0.98, but should be clipped to 0.1
    assert drip_epoch == pytest.approx(0.1, abs=0.001)
```

### Task 10: Add Property Tests for Drip Anti-Gaming

**File**: `/home/john/esper-lite/tests/simic/properties/test_reward_antigaming.py`

```python
@pytest.mark.property
class TestDripAntiGaming:
    """Drip reward prevents post-fossilization gaming."""

    @given(
        fossilize_epoch=st.integers(min_value=10, max_value=140),
        max_epochs=st.just(150),
        contribution_at_foss=st.floats(min_value=1.0, max_value=10.0),
    )
    @settings(max_examples=100)
    def test_epoch_normalization_prevents_early_gaming(
        self, fossilize_epoch, max_epochs, contribution_at_foss
    ):
        """Expected total drip is roughly equal regardless of fossilization timing.

        Without epoch normalization, early fossilization would capture more
        total drip. Normalization by remaining_epochs should equalize this.
        """
        config = ContributionRewardConfig(
            reward_mode=RewardMode.BASIC,
            drip_fraction=0.7,
            max_drip_per_epoch=0.1,
            drip_diminishing_coef=0.0,  # Disable diminishing for this test
        )

        remaining = max_epochs - fossilize_epoch

        # Simulate constant contribution post-fossilization
        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=fossilize_epoch,
            max_epochs=max_epochs,
            drip_total=2.0,  # Fixed for comparison
            drip_scale=2.0 / remaining,
        )

        # Sum drip over all remaining epochs with constant contribution
        total_drip = 0.0
        for e in range(fossilize_epoch + 1, max_epochs + 1):
            epoch_drip = drip_state.compute_epoch_drip(
                current_contribution=contribution_at_foss,
                num_fossilized=1,
                max_drip=config.max_drip_per_epoch,
                diminishing_coef=0.0,
            )
            # Clip can reduce total, but per-epoch is bounded
            total_drip += epoch_drip

        # Expected total (without clipping): drip_total * contribution
        expected_max = 2.0 * contribution_at_foss

        # Total should be bounded regardless of timing
        assert total_drip <= expected_max + 0.1, (
            f"Total drip {total_drip} exceeds expected max {expected_max} "
            f"for fossilize_epoch={fossilize_epoch}"
        )

    @given(
        contribution_sequence=st.lists(
            st.floats(min_value=-5.0, max_value=5.0),
            min_size=10,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_negative_drip_for_degrading_seeds(self, contribution_sequence):
        """Seeds that degrade post-fossilization receive negative drip (penalty)."""
        config = ContributionRewardConfig(
            reward_mode=RewardMode.BASIC,
            drip_fraction=0.7,
            max_drip_per_epoch=0.1,
        )

        drip_state = FossilizedSeedDripState(
            seed_id="test",
            slot_id="r0c1",
            fossilize_epoch=100,
            max_epochs=150,
            drip_total=2.0,
            drip_scale=2.0 / 50,
        )

        total_drip = 0.0
        negative_epochs = 0
        for contrib in contribution_sequence:
            epoch_drip = drip_state.compute_epoch_drip(
                current_contribution=contrib,
                num_fossilized=1,
                max_drip=config.max_drip_per_epoch,
                diminishing_coef=0.0,
            )
            total_drip += epoch_drip
            if contrib < 0:
                negative_epochs += 1
                # Negative contribution should produce negative drip
                assert epoch_drip < 0 or epoch_drip == 0, (
                    f"Negative contribution {contrib} produced positive drip {epoch_drip}"
                )

        # If mostly negative contributions, total drip should be negative
        if negative_epochs > len(contribution_sequence) * 0.7:
            assert total_drip < 0, (
                f"Mostly negative contributions ({negative_epochs}/{len(contribution_sequence)}) "
                f"should produce negative total drip, got {total_drip}"
            )
```

---

## Config File Updates

Update existing config files to include drip parameters (for BASIC mode):

**File**: `/home/john/esper-lite/configs/config-3slot-3seed-baseline-basic.json` (new)

```json
{
  "reward_mode": "basic",
  "drip_fraction": 0.7,
  "max_drip_per_epoch": 0.1,
  "min_drip_epochs": 0,
  "drip_diminishing_coef": 0.2,
  "basic_fossilize_base_bonus": 0.3,
  "basic_contribution_scale": 0.5
}
```

---

## Success Criteria

1. **Immediate reward reduced**: FOSSILIZE action pays only 30% of total bonus immediately
2. **Drip state created**: `FossilizedSeedDripState` created with correct `drip_total` and `drip_scale`
3. **Positive drip for contributing seeds**: Seeds with positive post-fossilization contribution receive positive drip each epoch
4. **Negative drip for degrading seeds**: Seeds with negative post-fossilization contribution receive negative drip (penalty)
5. **Clipping works**: Drip per epoch never exceeds `max_drip_per_epoch`
6. **Diminishing returns**: Multiple fossils receive diminished per-fossil drip
7. **Epoch normalization**: Expected total drip is independent of fossilization timing (given constant contribution)
8. **Telemetry captured**: `drip_this_epoch`, `drip_immediate_bonus`, etc. appear in reward telemetry
9. **All existing tests pass**: No regressions in shaped/escrow/sparse modes
10. **Golden tests pass**: New golden tests for drip mechanics pass

---

## File Checklist

| File | Action | Status |
|------|--------|--------|
| `src/esper/simic/training/config.py` | Add drip config fields | [ ] |
| `src/esper/simic/rewards/contribution.py` | Add `ContributionRewardConfig` fields | [ ] |
| `src/esper/simic/rewards/contribution.py` | Add `FossilizedSeedDripState` dataclass | [ ] |
| `src/esper/simic/rewards/contribution.py` | Modify `compute_basic_reward` | [ ] |
| `src/esper/simic/rewards/types.py` | Add seed_id/slot_id to SeedInfo | [ ] |
| `src/esper/simic/rewards/reward_telemetry.py` | Add drip telemetry fields | [ ] |
| `src/esper/leyline/telemetry.py` | Add drip to AnalyticsSnapshotPayload | [ ] |
| `src/esper/simic/training/vectorized.py` | Integrate drip state management | [ ] |
| `tests/simic/rewards/test_reward_golden.py` | Add drip golden tests | [ ] |
| `tests/simic/properties/test_reward_antigaming.py` | Add drip property tests | [ ] |
| `configs/config-3slot-3seed-baseline-basic.json` | Create basic mode config | [ ] |

---

## Rollout Plan

1. **Phase 1**: Implement core drip logic in `contribution.py` (Tasks 1-4)
2. **Phase 2**: Add telemetry (Tasks 5-6)
3. **Phase 3**: Integrate with vectorized trainer (Task 8)
4. **Phase 4**: Add tests (Tasks 9-10)
5. **Phase 5**: Run training with `reward_mode=basic` and verify telemetry shows drip
6. **Phase 6**: Compare training curves: with drip vs without (drip_fraction=0.0)

---

## Appendix: Mathematical Analysis

### Expected Value Analysis

With 70/30 split and epoch normalization:

```
E[total_reward | FOSSILIZE at epoch t] =
    0.3 * full_bonus                           # Immediate
    + sum_{e=t+1}^{T} (drip_scale * E[contribution_e])  # Drip

where drip_scale = 0.7 * full_bonus / (T - t)

If E[contribution_e] = c (constant), then:
    Drip total = (T - t) * drip_scale * c
              = (T - t) * (0.7 * full_bonus / (T - t)) * c
              = 0.7 * full_bonus * c

For c = 1 (contribution maintains level at fossilization):
    E[total_reward] = 0.3 * full_bonus + 0.7 * full_bonus = full_bonus
```

The epoch normalization ensures expected total equals full_bonus regardless of fossilization timing, assuming contribution maintains its level.

### Variance Considerations

Drip introduces additional variance into episode rewards:
- Without drip: Var(episode_reward) = Var(FOSSILIZE_bonus)
- With drip: Var(episode_reward) = 0.09 * Var(full_bonus) + Var(drip_sum)

Clipping at +/-0.1 per epoch bounds drip contribution variance. With max 130 drip epochs:
- Max drip contribution: 130 * 0.1 = 13.0 (unrealistic, would require max clip every epoch)
- Typical drip contribution: ~0.015 * 3.0 * 100 = 4.5 (for average contribution)

This is within acceptable PPO variance bounds given the reward scale.
