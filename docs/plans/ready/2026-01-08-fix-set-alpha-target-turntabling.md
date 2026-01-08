# Fix SET_ALPHA_TARGET Turntabling Exploit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the reward exploit where Tamiyo spams SET_ALPHA_TARGET in HOLDING stage to avoid the indecision penalty while collecting dense positive rewards.

**Architecture:** The fix extends the `holding_warning` penalty to apply to ALL non-terminal actions in HOLDING (not just WAIT). Terminal actions that resolve the holding state (FOSSILIZE, PRUNE) remain exempt. This aligns incentives: if you're in HOLDING with positive attribution, you should commit to a decision, not turntable the alpha controller.

**Tech Stack:** Python, pytest, Esper reward system (simic domain)

---

## Background

### The Exploit
In SHAPED reward mode, Tamiyo discovered she can spam `SET_ALPHA_TARGET` every epoch in HOLDING to:
1. Avoid the `holding_warning` penalty (which only fires on `action == WAIT`)
2. Still collect action-agnostic dense positives (`bounded_attribution`, `pbrs_bonus`, `synergy_bonus`)

### Telemetry Evidence
```
HOLDING Stage: SET_ALPHA_TARGET 19 actions, holding_warning=0.0
HOLDING Stage: WAIT            3 actions,  holding_warning=-0.033
```

### Root Cause
`contribution.py:439-455` only applies holding_warning when `action == LifecycleOp.WAIT`.

---

## Task 1: Add Test for SET_ALPHA_TARGET Holding Warning

**Files:**
- Create: `tests/simic/rewards/test_holding_warning.py`

**Step 1: Write the failing test**

```python
"""Tests for holding_warning penalty in HOLDING stage.

Bug fixed (2026-01-08): SET_ALPHA_TARGET was exempt from holding_warning,
creating an exploit where Tamiyo could turntable alpha settings to avoid
the indecision penalty while collecting dense positive rewards.
"""

import pytest

from esper.leyline import LifecycleOp, MIN_PRUNE_AGE, SeedStage
from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig
from esper.simic.rewards.types import SeedInfo


def holding_config() -> ContributionRewardConfig:
    """Config that isolates holding_warning from other reward components."""
    return ContributionRewardConfig(
        # Zero out other shaping to isolate holding_warning
        contribution_weight=1.0,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        alpha_shock_coef=0.0,
        germinate_cost=0.0,
        fossilize_cost=0.0,
        prune_cost=0.0,
        set_alpha_target_cost=0.0,
        germinate_with_seed_penalty=0.0,
    )


def holding_seed_info(epochs_in_stage: int = 3) -> SeedInfo:
    """Create a SeedInfo for a seed in HOLDING stage."""
    return SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.0,
        total_improvement=5.0,
        epochs_in_stage=epochs_in_stage,
        seed_params=1000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=MIN_PRUNE_AGE + 5,
        interaction_sum=0.0,
        boost_received=0.0,
    )


class TestHoldingWarningPenalty:
    """Tests for holding_warning penalty in HOLDING stage."""

    def test_wait_in_holding_triggers_warning(self) -> None:
        """WAIT in HOLDING with positive attribution triggers holding_warning."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,  # Positive contribution
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,  # Positive progress
        )

        # WAIT should trigger holding_warning
        assert components.holding_warning < 0, "WAIT in HOLDING should incur penalty"

    def test_set_alpha_target_in_holding_triggers_warning(self) -> None:
        """SET_ALPHA_TARGET in HOLDING with positive attribution triggers holding_warning.

        Bug fix: Previously SET_ALPHA_TARGET was exempt, creating turntabling exploit.
        """
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=5.0,  # Positive contribution
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,  # Positive progress
        )

        # SET_ALPHA_TARGET should ALSO trigger holding_warning (the fix)
        assert components.holding_warning < 0, (
            "SET_ALPHA_TARGET in HOLDING should incur penalty (turntabling fix)"
        )

    def test_fossilize_in_holding_exempt_from_warning(self) -> None:
        """FOSSILIZE in HOLDING is exempt - it's a terminal decision."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # FOSSILIZE is exempt - it resolves the holding state
        assert components.holding_warning == 0.0, "FOSSILIZE should be exempt from penalty"

    def test_prune_in_holding_exempt_from_warning(self) -> None:
        """PRUNE in HOLDING is exempt - it's a terminal decision."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        reward, components = compute_contribution_reward(
            action=LifecycleOp.PRUNE,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # PRUNE is exempt - it resolves the holding state
        assert components.holding_warning == 0.0, "PRUNE should be exempt from penalty"

    def test_holding_warning_requires_positive_attribution(self) -> None:
        """Holding warning only fires when bounded_attribution > 0."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=3)

        # Negative contribution = no penalty (seed is hurting, agent should prune)
        reward, components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=-2.0,  # Negative contribution
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # No penalty when attribution is negative
        assert components.holding_warning == 0.0, (
            "No penalty when seed contribution is negative"
        )

    def test_holding_warning_requires_epochs_in_stage_ge_2(self) -> None:
        """Holding warning only fires after epochs_in_stage >= 2."""
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=1)  # Just entered HOLDING

        reward, components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # No penalty in first epoch of HOLDING
        assert components.holding_warning == 0.0, (
            "No penalty in first epoch of HOLDING stage"
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/simic/rewards/test_holding_warning.py -v`

Expected: `test_set_alpha_target_in_holding_triggers_warning` FAILS with assertion error (holding_warning == 0.0 instead of < 0)

**Step 3: Commit the failing test**

```bash
git add tests/simic/rewards/test_holding_warning.py
git commit -m "test: add failing test for SET_ALPHA_TARGET holding_warning (turntabling fix)"
```

---

## Task 2: Fix holding_warning to Apply to Non-Terminal Actions

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:439-455`

**Step 1: Read the current implementation**

The current code at lines 439-455:
```python
holding_warning = 0.0
if seed_info is not None and seed_info.stage == STAGE_HOLDING:
    if action == LifecycleOp.WAIT:
        if seed_info.epochs_in_stage >= 2 and bounded_attribution > 0:
            # ... penalty logic
```

**Step 2: Implement the fix**

Replace the condition `action == LifecycleOp.WAIT` with a check for non-terminal actions:

```python
    # === Holding indecision penalty ===
    # In HOLDING stage, penalize actions that don't resolve the decision.
    # Terminal actions (FOSSILIZE, PRUNE) are exempt - they commit to a decision.
    # Non-terminal actions (WAIT, SET_ALPHA_TARGET, etc.) incur penalty.
    #
    # Bug fix (2026-01-08): Previously only WAIT triggered this penalty, allowing
    # Tamiyo to "turntable" SET_ALPHA_TARGET to avoid penalty while collecting
    # dense positives. Now all non-terminal actions in HOLDING are penalized.
    holding_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_HOLDING:
        # Terminal actions that resolve HOLDING - exempt from penalty
        terminal_actions = (LifecycleOp.FOSSILIZE, LifecycleOp.PRUNE)
        if action not in terminal_actions:
            if seed_info.epochs_in_stage >= 2 and bounded_attribution > 0:
                has_counterfactual = (
                    seed_contribution is not None
                    or (config.proxy_confidence_factor > 0 and acc_delta is not None)
                )
                if has_counterfactual:
                    epochs_waiting = seed_info.epochs_in_stage - 1
                    base_penalty = 0.1
                    ramp_penalty = max(0, epochs_waiting - 1) * 0.05
                    per_epoch_penalty = min(base_penalty + ramp_penalty, 0.3)
                    holding_warning = -per_epoch_penalty
                    reward += holding_warning
    if components:
        components.holding_warning = holding_warning
```

**Step 3: Run test to verify it passes**

Run: `uv run pytest tests/simic/rewards/test_holding_warning.py -v`

Expected: All 6 tests PASS

**Step 4: Run full reward test suite**

Run: `uv run pytest tests/simic/rewards/ -v --tb=short`

Expected: All tests PASS (no regressions)

**Step 5: Commit the fix**

```bash
git add src/esper/simic/rewards/contribution.py
git commit -m "fix(rewards): extend holding_warning to all non-terminal actions

Previously only WAIT triggered holding_warning in HOLDING stage, allowing
Tamiyo to 'turntable' SET_ALPHA_TARGET every epoch to avoid the indecision
penalty while collecting dense positive rewards (bounded_attribution, pbrs).

Telemetry confirmed the exploit:
- SET_ALPHA_TARGET in HOLDING: 19 actions, holding_warning=0.0
- WAIT in HOLDING: 3 actions, holding_warning=-0.033

Fix: Now all non-terminal actions (WAIT, SET_ALPHA_TARGET, GERMINATE, ADVANCE)
trigger holding_warning. Terminal actions (FOSSILIZE, PRUNE) remain exempt
as they commit to resolving the holding state.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Integration Test for Reward Parity

**Files:**
- Add to: `tests/simic/rewards/test_holding_warning.py`

**Step 1: Add integration test**

Append to the test file:

```python
class TestHoldingWarningParity:
    """Integration tests ensuring WAIT and SET_ALPHA_TARGET have equal penalties."""

    def test_wait_and_set_alpha_target_same_penalty_in_holding(self) -> None:
        """WAIT and SET_ALPHA_TARGET should incur the same holding_warning.

        This prevents any action from being a 'free WAIT' loophole.
        """
        config = holding_config()
        seed_info = holding_seed_info(epochs_in_stage=5)

        # WAIT
        _, wait_components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # SET_ALPHA_TARGET
        _, set_alpha_components = compute_contribution_reward(
            action=LifecycleOp.SET_ALPHA_TARGET,
            seed_contribution=5.0,
            val_acc=50.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=100,
            config=config,
            return_components=True,
            acc_at_germination=45.0,
        )

        # Both should have the SAME holding_warning
        assert wait_components.holding_warning == set_alpha_components.holding_warning, (
            f"WAIT ({wait_components.holding_warning}) and SET_ALPHA_TARGET "
            f"({set_alpha_components.holding_warning}) should have same penalty"
        )
        assert wait_components.holding_warning < 0, "Both should be penalized"
```

**Step 2: Run test**

Run: `uv run pytest tests/simic/rewards/test_holding_warning.py::TestHoldingWarningParity -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/simic/rewards/test_holding_warning.py
git commit -m "test: add parity test for WAIT/SET_ALPHA_TARGET holding_warning"
```

---

## Task 4: Update BLENDING Stage Warning (Consistency)

**Files:**
- Modify: `src/esper/simic/rewards/contribution.py:424-437` (blending_warning section)

**Step 1: Check if BLENDING has the same issue**

Read the blending_warning code and verify it also needs the fix.

**Step 2: Apply same pattern if needed**

If blending_warning only fires on WAIT, apply the same fix:

```python
    # === Blending warning ===
    # Same pattern as holding_warning - penalize non-terminal actions
    blending_warning = 0.0
    if seed_info is not None and seed_info.stage == STAGE_BLENDING:
        terminal_actions = (LifecycleOp.FOSSILIZE, LifecycleOp.PRUNE)
        if action not in terminal_actions:
            # ... existing logic
```

**Step 3: Add test for BLENDING if needed**

**Step 4: Run tests and commit**

Run: `uv run pytest tests/simic/rewards/ -v`

```bash
git add src/esper/simic/rewards/contribution.py tests/simic/rewards/test_holding_warning.py
git commit -m "fix(rewards): extend blending_warning to non-terminal actions (consistency)"
```

---

## Task 5: Run Full Test Suite and Verify

**Step 1: Run all simic tests**

Run: `uv run pytest tests/simic/ -v --tb=short`

Expected: All tests PASS

**Step 2: Run property tests**

Run: `uv run pytest tests/simic/properties/ -v`

Expected: All PBRS and reward invariant tests PASS

**Step 3: Type check**

Run: `uv run mypy src/esper/simic/rewards/contribution.py --ignore-missing-imports`

Expected: No errors

---

## Task 6: Document the Fix

**Files:**
- Update: `src/esper/simic/rewards/contribution.py` (docstring)

**Step 1: Update module docstring or add comment**

Add a note in the contribution reward docstring about the turntabling fix.

**Step 2: Commit**

```bash
git add src/esper/simic/rewards/contribution.py
git commit -m "docs: document SET_ALPHA_TARGET turntabling fix"
```

---

## Verification Checklist

After implementation, verify:

- [ ] `test_set_alpha_target_in_holding_triggers_warning` passes
- [ ] `test_wait_and_set_alpha_target_same_penalty_in_holding` passes
- [ ] `test_fossilize_in_holding_exempt_from_warning` passes
- [ ] `test_prune_in_holding_exempt_from_warning` passes
- [ ] All existing reward tests still pass
- [ ] PBRS property tests still pass
- [ ] No mypy errors

---

## Follow-Up (Optional Enhancements)

These are lower-priority fixes identified in the diagnosis:

1. **Plan-reset shock**: Add penalty for consecutive SET_ALPHA_TARGET calls (Fix D)
2. **Progress=None fallback**: Tighten attribution when progress is unknown (Fix C)
3. **Consider ESCROW mode**: For control actions with delayed effects, ESCROW mode may be more appropriate than SHAPED

These can be addressed in separate PRs if the primary fix doesn't fully resolve the turntabling behavior.
