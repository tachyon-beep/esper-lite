# Reward Telemetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-step reward component telemetry so operators can see exactly what is rewarding/penalizing Tamiyo at each decision point.

**Architecture:** Add `REWARD_COMPUTED` event type, extend `compute_contribution_reward` with `return_components` support matching `compute_shaped_reward`, emit telemetry event after each reward computation in vectorized training loop. Gate telemetry on debug level for performance.

**Tech Stack:** Python dataclasses, esper.leyline telemetry, esper.simic.reward_telemetry

**Specialist Review:** 2025-12-07
- DRL Expert: CONDITIONAL GO - add missing diagnostic fields, fix hasattr usage
- PyTorch Expert: CONDITIONAL GO - fix performance issues (asdict, imports), make gating mandatory

---

## Task 1: Add REWARD_COMPUTED Event Type

**Files:**
- Modify: `src/esper/leyline/telemetry.py:56` (add new event type)

**Step 1: Add the event type**

In `src/esper/leyline/telemetry.py`, add after `REWARD_HACKING_SUSPECTED`:

```python
    REWARD_COMPUTED = auto()  # Per-step reward breakdown for debugging
```

**Step 2: Verify import works**

Run: `PYTHONPATH=src python3 -c "from esper.leyline import TelemetryEventType; print(TelemetryEventType.REWARD_COMPUTED)"`

Expected: `TelemetryEventType.REWARD_COMPUTED`

**Step 3: Commit**

```bash
git add src/esper/leyline/telemetry.py
git commit -m "feat(leyline): add REWARD_COMPUTED telemetry event type"
```

---

## Task 2: Extend RewardComponentsTelemetry for Contribution Reward

**Files:**
- Modify: `src/esper/simic/reward_telemetry.py`

**Step 1: Add contribution-specific fields and fix performance**

The existing `RewardComponentsTelemetry` has `seed_contribution` and `bounded_attribution` but we need to track more components from `compute_contribution_reward`.

Per DRL Expert: Add `val_acc`, `acc_at_germination`, `host_baseline_acc`, `growth_ratio`, `action_success` for diagnostics.
Per PyTorch Expert: Replace `asdict()` with explicit dict construction (3-5x faster).

Update the dataclass:

```python
@dataclass(slots=True)
class RewardComponentsTelemetry:
    """Breakdown of reward components for debugging.

    Each field represents one component of the total reward.
    All components should sum to total_reward.
    """

    # Base signal (legacy shaped reward)
    base_acc_delta: float = 0.0

    # Contribution-primary signal
    seed_contribution: float | None = None
    bounded_attribution: float | None = None
    progress_since_germination: float | None = None

    # Penalties
    compute_rent: float = 0.0

    # Bonuses
    stage_bonus: float = 0.0
    pbrs_bonus: float = 0.0
    action_shaping: float = 0.0
    terminal_bonus: float = 0.0

    # Context (for debugging) - DRL Expert recommended fields
    action_name: str = ""
    action_success: bool = True
    seed_stage: int | None = None
    epoch: int = 0
    val_acc: float = 0.0
    acc_at_germination: float | None = None
    host_baseline_acc: float | None = None  # Counterfactual baseline
    growth_ratio: float = 0.0  # total_params / host_params

    # Total
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field.

        Uses explicit dict construction instead of asdict() for 3-5x performance
        improvement in hot path (PyTorch Expert recommendation).
        """
        return {
            "base_acc_delta": self.base_acc_delta,
            "seed_contribution": self.seed_contribution,
            "bounded_attribution": self.bounded_attribution,
            "progress_since_germination": self.progress_since_germination,
            "compute_rent": self.compute_rent,
            "stage_bonus": self.stage_bonus,
            "pbrs_bonus": self.pbrs_bonus,
            "action_shaping": self.action_shaping,
            "terminal_bonus": self.terminal_bonus,
            "action_name": self.action_name,
            "action_success": self.action_success,
            "seed_stage": self.seed_stage,
            "epoch": self.epoch,
            "val_acc": self.val_acc,
            "acc_at_germination": self.acc_at_germination,
            "host_baseline_acc": self.host_baseline_acc,
            "growth_ratio": self.growth_ratio,
            "total_reward": self.total_reward,
        }
```

**Step 2: Remove asdict import if present**

Check if `from dataclasses import dataclass, asdict` is at top - change to just `from dataclasses import dataclass`.

**Step 3: Verify dataclass works**

Run: `PYTHONPATH=src python3 -c "from esper.simic.reward_telemetry import RewardComponentsTelemetry; c = RewardComponentsTelemetry(action_name='WAIT', epoch=5, val_acc=72.5); print(c.to_dict())"`

Expected: Dict with all fields including new ones

**Step 4: Commit**

```bash
git add src/esper/simic/reward_telemetry.py
git commit -m "feat(simic): extend RewardComponentsTelemetry with diagnostic fields"
```

---

## Task 3: Add return_components to compute_contribution_reward

**Files:**
- Modify: `src/esper/simic/rewards.py` (function `compute_contribution_reward`)
- Test: `tests/test_simic_rewards.py`

**Step 0: Move import to module level (PyTorch Expert requirement)**

At the top of `src/esper/simic/rewards.py`, add:

```python
from esper.simic.reward_telemetry import RewardComponentsTelemetry
```

This avoids per-call import overhead.

**Step 1: Write the failing test**

Add to `tests/test_simic_rewards.py`:

```python
class TestContributionRewardComponents:
    """Tests for compute_contribution_reward return_components."""

    def test_return_components_returns_tuple(self):
        """Test that return_components=True returns (reward, components) tuple."""
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
        from esper.simic.reward_telemetry import RewardComponentsTelemetry
        from esper.leyline import SeedStage

        # Create a mock action enum
        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=1.0,
            total_improvement=1.0,
            epochs_in_stage=5,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            seed_age_epochs=5,
        )

        result = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=2.0,
            val_acc=70.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=1000,
            host_params=10000,
            acc_at_germination=65.0,
            return_components=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        reward, components = result
        assert isinstance(reward, float)
        # Use isinstance instead of hasattr (per CLAUDE.md policy)
        assert isinstance(components, RewardComponentsTelemetry)

    def test_components_sum_to_total(self):
        """Test that component values sum to total_reward."""
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
        from esper.leyline import SeedStage

        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=SeedStage.BLENDING.value,
            improvement_since_stage_start=2.0,
            total_improvement=3.0,
            epochs_in_stage=3,
            seed_params=5000,
            previous_stage=SeedStage.TRAINING.value,
            seed_age_epochs=8,
        )

        reward, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=3.0,
            val_acc=72.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=5000,
            host_params=20000,
            acc_at_germination=68.0,
            return_components=True,
        )

        # Components should sum to total (within floating point tolerance)
        component_sum = (
            (components.bounded_attribution or 0.0)
            + components.compute_rent
            + components.pbrs_bonus
            + components.action_shaping
            + components.terminal_bonus
        )
        assert abs(reward - component_sum) < 0.001, f"Sum {component_sum} != total {reward}"
        assert components.total_reward == reward

    def test_components_track_context(self):
        """Test that components include action and epoch context."""
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
        from esper.leyline import SeedStage

        from enum import IntEnum
        class MockAction(IntEnum):
            CULL = 1

        seed_info = SeedInfo(
            stage=SeedStage.TRAINING.value,
            improvement_since_stage_start=-1.0,
            total_improvement=-1.0,
            epochs_in_stage=10,
            seed_params=1000,
            previous_stage=SeedStage.GERMINATED.value,
            seed_age_epochs=10,
        )

        reward, components = compute_contribution_reward(
            action=MockAction.CULL,
            seed_contribution=-0.5,
            val_acc=68.0,
            seed_info=seed_info,
            epoch=12,
            max_epochs=25,
            return_components=True,
        )

        assert components.action_name == "CULL"
        assert components.epoch == 12
        assert components.seed_stage == SeedStage.TRAINING.value

    def test_components_include_diagnostic_fields(self):
        """Test that components include DRL Expert recommended diagnostic fields."""
        from esper.simic.rewards import compute_contribution_reward, SeedInfo
        from esper.leyline import SeedStage

        from enum import IntEnum
        class MockAction(IntEnum):
            WAIT = 0

        seed_info = SeedInfo(
            stage=SeedStage.BLENDING.value,
            improvement_since_stage_start=1.5,
            total_improvement=2.0,
            epochs_in_stage=5,
            seed_params=5000,
            previous_stage=SeedStage.TRAINING.value,
            seed_age_epochs=10,
        )

        reward, components = compute_contribution_reward(
            action=MockAction.WAIT,
            seed_contribution=2.5,
            val_acc=72.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=5000,
            host_params=20000,
            acc_at_germination=68.0,
            return_components=True,
        )

        # DRL Expert recommended fields
        assert components.val_acc == 72.0
        assert components.acc_at_germination == 68.0
        assert components.growth_ratio == 5000 / 20000  # 0.25
        assert components.progress_since_germination == 72.0 - 68.0  # 4.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_simic_rewards.py::TestContributionRewardComponents -v`

Expected: FAIL with `TypeError: compute_contribution_reward() got an unexpected keyword argument 'return_components'`

**Step 3: Implement return_components in compute_contribution_reward**

Modify `src/esper/simic/rewards.py` function `compute_contribution_reward`. Add the parameter and tracking.

Note: Import is already at module level from Step 0, so no in-function import needed.

```python
def compute_contribution_reward(
    action: IntEnum,
    seed_contribution: float | None,
    val_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int = 0,
    host_params: int = 1,
    config: ContributionRewardConfig | None = None,
    acc_at_germination: float | None = None,
    return_components: bool = False,
) -> float | tuple[float, "RewardComponentsTelemetry"]:
    """Compute reward using bounded attribution (ransomware-resistant).

    ... (existing docstring, add to Returns section:)

    Returns:
        Shaped reward value, or (reward, components) if return_components=True
    """
    if config is None:
        config = _DEFAULT_CONTRIBUTION_CONFIG

    # Track components if requested (no import needed - already at module level)
    components = RewardComponentsTelemetry() if return_components else None
    if components:
        components.action_name = action.name
        components.epoch = epoch
        components.seed_stage = seed_info.stage if seed_info else None
        # DRL Expert recommended diagnostic fields
        components.val_acc = val_acc
        components.acc_at_germination = acc_at_germination

    reward = 0.0

    # === 1. PRIMARY: Bounded Attribution (Ransomware-Resistant) ===
    bounded_attribution = 0.0
    progress = None
    if seed_contribution is not None:
        if seed_contribution < 0:
            bounded_attribution = config.contribution_weight * seed_contribution
        else:
            if acc_at_germination is not None:
                progress = val_acc - acc_at_germination
                attributed = min(max(0.0, progress), seed_contribution)
            else:
                attributed = seed_contribution * 0.5
            bounded_attribution = config.contribution_weight * attributed
        reward += bounded_attribution

    if components:
        components.seed_contribution = seed_contribution
        components.bounded_attribution = bounded_attribution
        components.progress_since_germination = progress

    # === 2. PBRS: Stage Progression ===
    pbrs_bonus = 0.0
    if seed_info is not None:
        pbrs_bonus = _contribution_pbrs_bonus(seed_info, config)
        reward += pbrs_bonus
    if components:
        components.pbrs_bonus = pbrs_bonus

    # === 3. RENT: Compute Cost ===
    rent_penalty = 0.0
    growth_ratio = 0.0
    if host_params > 0 and total_params > 0:
        growth_ratio = total_params / host_params
        scaled_cost = math.log(1.0 + max(0.0, growth_ratio))
        rent_penalty = min(config.rent_weight * scaled_cost, config.max_rent)
        reward -= rent_penalty
    if components:
        components.compute_rent = -rent_penalty  # Negative because it's a penalty
        components.growth_ratio = growth_ratio  # DRL Expert diagnostic field

    # === 4. ACTION SHAPING ===
    action_shaping = 0.0
    action_name = action.name

    if is_germinate_action(action):
        if seed_info is not None:
            action_shaping += config.germinate_with_seed_penalty
        else:
            phi_germinated = _CONTRIBUTION_STAGE_POTENTIALS.get(STAGE_GERMINATED, 0.0)
            phi_no_seed = 0.0
            pbrs_germinate = config.gamma * phi_germinated - phi_no_seed
            action_shaping += config.pbrs_weight * pbrs_germinate
        action_shaping += config.germinate_cost

    elif action_name == "FOSSILIZE":
        action_shaping += _contribution_fossilize_shaping(seed_info, seed_contribution, config)
        action_shaping += config.fossilize_cost

    elif action_name == "CULL":
        action_shaping += _contribution_cull_shaping(seed_info, seed_contribution, config)
        action_shaping += config.cull_cost

    reward += action_shaping
    if components:
        components.action_shaping = action_shaping

    # === 5. TERMINAL BONUS ===
    terminal_bonus = 0.0
    if epoch == max_epochs:
        terminal_bonus = val_acc * config.terminal_acc_weight
        reward += terminal_bonus
    if components:
        components.terminal_bonus = terminal_bonus

    if components:
        components.total_reward = reward
        return reward, components
    return reward
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/test_simic_rewards.py::TestContributionRewardComponents -v`

Expected: PASS (3 tests)

**Step 5: Run full reward test suite**

Run: `PYTHONPATH=src uv run pytest tests/test_simic_rewards.py -v`

Expected: All tests pass (existing + new)

**Step 6: Commit**

```bash
git add src/esper/simic/rewards.py tests/test_simic_rewards.py
git commit -m "feat(simic): add return_components to compute_contribution_reward"
```

---

## Task 4: Emit REWARD_COMPUTED Telemetry in Vectorized Training (with Gating)

**Files:**
- Modify: `src/esper/simic/vectorized.py:924-946` (reward computation section)

Per PyTorch Expert: Gating on debug level is MANDATORY for performance. We combine Task 4 and Task 5.

**Step 1: Add TelemetryEventType import if not present**

At top of vectorized.py, ensure import includes `TelemetryEventType`:

```python
from esper.leyline import TelemetryEvent, TelemetryEventType, SeedTelemetry
```

**Step 2: Modify reward computation with gated telemetry**

In `src/esper/simic/vectorized.py`, modify the reward computation block (~lines 922-946).

Key changes:
1. Gate component collection on debug level (PyTorch Expert: avoid allocation when not needed)
2. Include `host_baseline_acc` from counterfactual (DRL Expert)
3. Track `action_success` after action execution (DRL Expert)
4. Also add telemetry for fallback path (PyTorch Expert)

```python
                # Determine if we need reward components (only at debug level)
                collect_reward_telemetry = telemetry_config.should_collect("debug")

                # Use contribution-primary reward when counterfactual is available
                if seed_contribution is not None:
                    if collect_reward_telemetry:
                        reward, reward_components = compute_contribution_reward(
                            action=action,
                            seed_contribution=seed_contribution,
                            val_acc=env_state.val_acc,
                            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                            epoch=epoch,
                            max_epochs=max_epochs,
                            total_params=total_params,
                            host_params=host_params,
                            acc_at_germination=env_state.acc_at_germination,
                            return_components=True,
                        )
                        # Add host_baseline_acc from counterfactual (DRL Expert)
                        reward_components.host_baseline_acc = baseline_accs[env_idx]
                    else:
                        reward = compute_contribution_reward(
                            action=action,
                            seed_contribution=seed_contribution,
                            val_acc=env_state.val_acc,
                            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                            epoch=epoch,
                            max_epochs=max_epochs,
                            total_params=total_params,
                            host_params=host_params,
                            acc_at_germination=env_state.acc_at_germination,
                        )
                        reward_components = None
                else:
                    # Fallback to legacy reward (no counterfactual available)
                    if collect_reward_telemetry:
                        reward, reward_components = compute_shaped_reward(
                            action=action,
                            acc_delta=signals.metrics.accuracy_delta,
                            val_acc=env_state.val_acc,
                            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                            epoch=epoch,
                            max_epochs=max_epochs,
                            total_params=total_params,
                            host_params=host_params,
                            return_components=True,
                        )
                    else:
                        reward = compute_shaped_reward(
                            action=action,
                            acc_delta=signals.metrics.accuracy_delta,
                            val_acc=env_state.val_acc,
                            seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                            epoch=epoch,
                            max_epochs=max_epochs,
                            total_params=total_params,
                            host_params=host_params,
                        )
                        reward_components = None
```

**Step 3: Add action execution and telemetry emission (after action execution)**

After the action execution block (around line ~980, after `action_success` is set), add:

```python
                # Emit reward telemetry if collecting (after action execution so we have action_success)
                if reward_components is not None:
                    reward_components.action_success = action_success
                    hub.emit(TelemetryEvent(
                        event_type=TelemetryEventType.REWARD_COMPUTED,
                        seed_id=seed_state.seed_id if seed_state else None,
                        epoch=epoch,
                        data={
                            "env_id": env_idx,
                            **reward_components.to_dict(),
                        },
                        severity="debug",
                    ))
```

**Step 4: Verify syntax is correct**

Run: `python3 -m py_compile src/esper/simic/vectorized.py`

Expected: No output (success)

**Step 5: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/integration/test_ppo_integration.py -v -x`

Expected: All tests pass

**Step 6: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(simic): emit gated REWARD_COMPUTED telemetry in vectorized training"
```

---

## Task 5: Final Verification

**Step 1: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/ -x -q`

Expected: All tests pass

**Step 2: Manual smoke test**

Run a short training to verify telemetry appears:

```bash
PYTHONPATH=src python -m esper.scripts.train ppo --vectorized --n-envs 2 --episodes 4 --max-epochs 10 --telemetry-dir ./telemetry-test --telemetry-level debug
```

Then check for REWARD_COMPUTED events:

```bash
grep "REWARD_COMPUTED" ./telemetry-test/*/events.jsonl | head -5
```

Expected: JSON lines with reward component breakdown including diagnostic fields

**Step 3: Verify diagnostic fields are present**

```bash
grep "REWARD_COMPUTED" ./telemetry-test/*/events.jsonl | head -1 | python3 -m json.tool
```

Expected: Should include `val_acc`, `acc_at_germination`, `host_baseline_acc`, `growth_ratio`, `action_success`

**Step 4: Clean up test telemetry**

```bash
rm -rf ./telemetry-test
```

**Step 5: Commit final state**

```bash
git add -A
git commit -m "docs: add reward telemetry implementation plan"
```

---

## Summary

After completing all tasks, the telemetry will include per-step reward breakdowns with full diagnostic fields:

```json
{
  "event_type": "REWARD_COMPUTED",
  "epoch": 15,
  "seed_id": "env0_seed_1",
  "data": {
    "env_id": 0,
    "action_name": "WAIT",
    "action_success": true,
    "seed_contribution": 2.5,
    "bounded_attribution": 7.5,
    "progress_since_germination": 4.0,
    "compute_rent": -0.15,
    "pbrs_bonus": 0.3,
    "action_shaping": 0.0,
    "terminal_bonus": 0.0,
    "seed_stage": 4,
    "epoch": 15,
    "val_acc": 72.0,
    "acc_at_germination": 68.0,
    "host_baseline_acc": 69.5,
    "growth_ratio": 0.25,
    "total_reward": 7.65
  }
}
```

This enables debugging:
- **Why did Tamiyo get rewarded/penalized?** - See `bounded_attribution`, `action_shaping`
- **Is bounded attribution capping too aggressively?** - Compare `seed_contribution` vs `progress_since_germination`
- **Are PBRS bonuses balanced?** - Check `pbrs_bonus` magnitude
- **Is compute rent too aggressive?** - Check `compute_rent` vs `growth_ratio`
- **Is counterfactual working?** - Compare `val_acc` vs `host_baseline_acc`
- **Are actions succeeding?** - Check `action_success` field

### Key Insight from DRL Expert Review

> "Negative contributions pay full penalty, but positive contributions are bounded. If the seed is legitimately helping but progress is slow (host model stalling), the reward is artificially capped."

With this telemetry, you can now detect this pattern by checking:
- `seed_contribution > progress_since_germination` → bounded attribution is capping
- If this happens frequently, consider softening the bound
