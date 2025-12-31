# Leyline Action Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the duplicate `TamiyoAction` enum and unify on a single `Action` type in leyline, fixing the P0 architectural violation.

**Architecture:** Rename `SimicAction` → `Action` in leyline with backwards-compat alias. Delete `TamiyoAction` entirely. Update `HeuristicTamiyo.decide()` to return `Action` directly. Remove all mapping code in simic.

**Tech Stack:** Python enums, dataclasses. No new dependencies.

**Audit Doc:** `docs/plans/2025-11-29-leyline-violations-audit.md`

---

## Task 1: Rename SimicAction → Action in Leyline

**Files:**
- Modify: `src/esper/leyline/actions.py:9-37`
- Modify: `src/esper/leyline/__init__.py:14,69`

**Step 1: Update the enum name and add alias**

In `src/esper/leyline/actions.py`, change:

```python
class SimicAction(Enum):
    """Discrete actions Tamiyo can take.
```

To:

```python
class Action(Enum):
    """Discrete actions for seed lifecycle control.

    This is the shared action space for all controllers (Tamiyo, Simic).
    Actions represent atomic decisions about seed lifecycle management.
```

**Step 2: Add backwards-compat alias at end of file**

Add after the class definition:

```python
# Backwards compatibility alias (deprecated)
SimicAction = Action
```

**Step 3: Update method references**

Change all `SimicAction` references inside the class to `Action`:

```python
    @classmethod
    def is_germinate(cls, action: "Action") -> bool:
        """Check if action is any germinate variant."""
        return action in (cls.GERMINATE_CONV, cls.GERMINATE_ATTENTION,
                         cls.GERMINATE_NORM, cls.GERMINATE_DEPTHWISE)

    @classmethod
    def get_blueprint_id(cls, action: "Action") -> str | None:
        """Get blueprint ID for germinate actions, None for others."""
```

**Step 4: Update leyline __init__.py exports**

In `src/esper/leyline/__init__.py`, change line 14:

```python
from esper.leyline.actions import Action, SimicAction  # SimicAction is alias
```

Change line 69 in `__all__`:

```python
    # Actions
    "Action",
    "SimicAction",  # deprecated alias
```

**Step 5: Run existing tests**

Run: `.venv/bin/python -m pytest tests/ -v -k "simic or leyline" --tb=short`

Expected: All tests PASS (alias maintains compatibility)

**Step 6: Commit**

```bash
git add src/esper/leyline/actions.py src/esper/leyline/__init__.py
git commit -m "refactor(leyline): rename SimicAction to Action with backwards-compat alias"
```

---

## Task 2: Update Tamiyo to Return Action Directly

**Files:**
- Modify: `src/esper/tamiyo/decisions.py` (delete TamiyoAction, update TamiyoDecision)
- Modify: `src/esper/tamiyo/heuristic.py` (update HeuristicTamiyo)
- Modify: `src/esper/tamiyo/__init__.py` (update exports)

**Step 1: Update TamiyoDecision to use Action**

In `src/esper/tamiyo/decisions.py`, replace the entire file with:

```python
"""Tamiyo Decisions - Strategic decision structures.

Defines the decisions made by strategic controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from esper.leyline import (
    Action,
    CommandType,
    RiskLevel,
    AdaptationCommand,
    SeedStage,
)


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


# Mapping from Action to leyline CommandType and target stage
_ACTION_TO_COMMAND: dict[Action, tuple[CommandType, SeedStage | None]] = {
    Action.WAIT: (CommandType.REQUEST_STATE, None),
    Action.GERMINATE_CONV: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.GERMINATE_ATTENTION: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.GERMINATE_NORM: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.GERMINATE_DEPTHWISE: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.ADVANCE: (CommandType.ADVANCE_STAGE, None),  # Target determined by current stage
    Action.CULL: (CommandType.CULL, SeedStage.CULLED),
}


@dataclass
class TamiyoDecision:
    """A decision made by Tamiyo.

    Uses the shared Action enum from leyline for compatibility with
    all controllers (heuristic, RL, etc.).
    """
    action: Action
    target_seed_id: str | None = None
    reason: str = ""
    confidence: float = 1.0

    def __str__(self) -> str:
        parts = [f"Action: {self.action.name}"]
        if self.target_seed_id:
            parts.append(f"Target: {self.target_seed_id}")
        if self.reason:
            parts.append(f"Reason: {self.reason}")
        return " | ".join(parts)

    @property
    def blueprint_id(self) -> str | None:
        """Get blueprint ID if this is a germinate action."""
        return Action.get_blueprint_id(self.action)

    def to_command(self) -> AdaptationCommand:
        """Convert to Leyline's canonical AdaptationCommand format."""
        command_type, target_stage = _ACTION_TO_COMMAND.get(
            self.action,
            (CommandType.REQUEST_STATE, None)
        )

        # Determine risk level based on action
        if self.action == Action.WAIT:
            risk = RiskLevel.GREEN
        elif Action.is_germinate(self.action):
            risk = RiskLevel.YELLOW
        elif self.action == Action.ADVANCE:
            risk = RiskLevel.YELLOW
        elif self.action == Action.CULL:
            risk = RiskLevel.ORANGE
        else:
            risk = RiskLevel.GREEN

        return AdaptationCommand(
            command_type=command_type,
            target_seed_id=self.target_seed_id,
            blueprint_id=self.blueprint_id,
            target_stage=target_stage,
            reason=self.reason,
            confidence=self.confidence,
            risk_level=risk,
        )


__all__ = [
    "TamiyoDecision",
]
```

**Step 2: Update HeuristicTamiyo to use Action**

In `src/esper/tamiyo/heuristic.py`, make these changes:

Change imports (lines 11-12):

```python
from esper.leyline import Action, SeedStage, is_terminal_stage, is_failure_stage, TrainingSignals
from esper.tamiyo.decisions import TamiyoDecision
```

Change all `TamiyoAction.WAIT` → `Action.WAIT`, etc.:

In `_decide_germination` (around line 97-117):

```python
    def _decide_germination(self, signals: TrainingSignals) -> TamiyoDecision:
        """Decide whether to germinate a new seed."""

        # Don't germinate too early
        if signals.metrics.epoch < self.config.min_epochs_before_germinate:
            return TamiyoDecision(
                action=Action.WAIT,
                reason=f"Too early (epoch {signals.metrics.epoch} < {self.config.min_epochs_before_germinate})"
            )

        # Check for plateau
        if signals.metrics.plateau_epochs >= self.config.plateau_epochs_to_germinate:
            # Select germinate action based on blueprint
            blueprint_id = self._get_next_blueprint()
            germinate_action = self._blueprint_to_action(blueprint_id)
            self._germination_count += 1
            return TamiyoDecision(
                action=germinate_action,
                reason=f"Plateau detected ({signals.metrics.plateau_epochs} epochs without improvement)",
                confidence=min(1.0, signals.metrics.plateau_epochs / 5.0),
            )

        # No action needed
        return TamiyoDecision(
            action=Action.WAIT,
            reason="Training progressing normally"
        )
```

Add helper method after `_get_next_blueprint`:

```python
    def _blueprint_to_action(self, blueprint_id: str) -> Action:
        """Convert blueprint ID to corresponding GERMINATE action."""
        blueprint_map = {
            "conv_enhance": Action.GERMINATE_CONV,
            "attention": Action.GERMINATE_ATTENTION,
            "norm": Action.GERMINATE_NORM,
            "depthwise": Action.GERMINATE_DEPTHWISE,
        }
        return blueprint_map.get(blueprint_id, Action.GERMINATE_CONV)
```

In `_decide_seed_management` (around line 132-153), change:

```python
            if seed.stage == SeedStage.GERMINATED:
                # Ready to start training - auto-advance
                return TamiyoDecision(
                    action=Action.ADVANCE,
                    target_seed_id=seed.seed_id,
                    reason="Seed germinated, starting isolated training",
                )
```

And the WAIT checks:

```python
                if decision.action != Action.WAIT:
                    return decision
```

In `_evaluate_training_seed` (around line 155-195), change all `TamiyoAction.X` to `Action.X`:

```python
    def _evaluate_training_seed(
        self,
        signals: TrainingSignals,
        seed: "SeedState",
    ) -> TamiyoDecision:
        """Evaluate a seed in TRAINING stage."""

        # Need minimum training time
        if seed.epochs_in_stage < self.config.min_training_epochs:
            return TamiyoDecision(
                action=Action.WAIT,
                target_seed_id=seed.seed_id,
                reason=f"Training epoch {seed.epochs_in_stage}/{self.config.min_training_epochs}"
            )

        # Check improvement
        improvement = seed.metrics.improvement_since_stage_start

        if improvement >= self.config.training_improvement_threshold:
            # Good improvement - advance to blending
            return TamiyoDecision(
                action=Action.ADVANCE,
                target_seed_id=seed.seed_id,
                reason=f"Good improvement ({improvement:.2f}%), advancing to blending",
                confidence=min(1.0, improvement / 5.0),
            )

        # Check if we should cull
        if seed.epochs_in_stage >= self.config.cull_after_epochs_without_improvement:
            if improvement < -self.config.cull_if_accuracy_drops_by:
                return TamiyoDecision(
                    action=Action.CULL,
                    target_seed_id=seed.seed_id,
                    reason=f"Seed hurting performance ({improvement:.2f}%)",
                )

        return TamiyoDecision(
            action=Action.WAIT,
            target_seed_id=seed.seed_id,
            reason=f"Training in progress, improvement: {improvement:.2f}%"
        )
```

In `_evaluate_blending_seed` (around line 197-226), change similarly:

```python
    def _evaluate_blending_seed(
        self,
        signals: TrainingSignals,
        seed: "SeedState",
    ) -> TamiyoDecision:
        """Evaluate a seed in BLENDING stage."""

        # Check if blending is complete
        if seed.epochs_in_stage >= self.config.blending_epochs:
            # Check final improvement
            improvement = seed.metrics.total_improvement

            if improvement > 0:
                return TamiyoDecision(
                    action=Action.ADVANCE,
                    target_seed_id=seed.seed_id,
                    reason=f"Blending complete, total improvement: {improvement:.2f}%",
                )
            else:
                return TamiyoDecision(
                    action=Action.CULL,
                    target_seed_id=seed.seed_id,
                    reason=f"Blending complete but no improvement ({improvement:.2f}%)",
                )

        return TamiyoDecision(
            action=Action.WAIT,
            target_seed_id=seed.seed_id,
            reason=f"Blending epoch {seed.epochs_in_stage}/{self.config.blending_epochs}"
        )
```

**Step 3: Update tamiyo __init__.py**

In `src/esper/tamiyo/__init__.py`, change:

```python
"""Tamiyo - Strategic decision-making for Esper.

Tamiyo observes training signals and makes strategic decisions
about seed lifecycle management.
"""

from esper.tamiyo.decisions import TamiyoDecision
from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.heuristic import (
    TamiyoPolicy,
    HeuristicPolicyConfig,
    HeuristicTamiyo,
)

__all__ = [
    "TamiyoDecision",
    "SignalTracker",
    "TamiyoPolicy",
    "HeuristicPolicyConfig",
    "HeuristicTamiyo",
]
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: Some tests may fail if they reference TamiyoAction - fix in next task.

**Step 5: Commit**

```bash
git add src/esper/tamiyo/
git commit -m "refactor(tamiyo): use Action from leyline, delete TamiyoAction"
```

---

## Task 3: Remove Mapping Code from Simic

**Files:**
- Modify: `src/esper/simic/episodes.py:673-706` (simplify action_from_decision)
- Modify: `src/esper/simic/comparison.py:597-620` (simplify heuristic_action_fn)
- Modify: `src/esper/simic/comparison.py:299-305` (fix live_comparison)

**Step 1: Simplify action_from_decision**

In `src/esper/simic/episodes.py`, replace `action_from_decision` function (lines 673-706):

```python
def action_from_decision(decision) -> ActionTaken:
    """Convert Tamiyo's TamiyoDecision to a Simic ActionTaken.

    Since TamiyoDecision now uses the shared Action enum from leyline,
    this is a simple wrapper that extracts the relevant fields.
    """
    from esper.leyline import Action

    return ActionTaken(
        action=decision.action,  # Already an Action enum
        blueprint_id=decision.blueprint_id,
        target_seed_id=decision.target_seed_id,
        confidence=decision.confidence,
        reason=decision.reason,
    )
```

**Step 2: Simplify heuristic_action_fn in head_to_head_comparison**

In `src/esper/simic/comparison.py`, find `heuristic_action_fn` (around line 597) and replace:

```python
    def heuristic_action_fn(signals, model, tracker, use_telemetry):
        """Get action from heuristic Tamiyo."""
        # Get active seeds from model
        active_seeds = [model.seed_state] if model.has_active_seed and model.seed_state else []
        decision = tamiyo.decide(signals, active_seeds)
        # Decision.action is already an Action enum from leyline
        return decision.action
```

Note: `tamiyo` must be defined in the outer scope (fix in Task 4).

**Step 3: Fix live_comparison action tracking**

In `src/esper/simic/comparison.py`, find the action tracking code (around line 299-305) and change:

```python
            # Track actions - both now use Action enum
            results['action_counts']['heuristic'][h_action.action.name] += 1
            results['action_counts']['iql'][iql_action] += 1

            if h_action.action.name == iql_action:
                ep_agreements += 1
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v --tb=short`

Expected: Tests pass

**Step 5: Commit**

```bash
git add src/esper/simic/episodes.py src/esper/simic/comparison.py
git commit -m "refactor(simic): remove action mapping code, use Action directly"
```

---

## Task 4: Fix HeuristicTamiyo Instantiation Scope

**Files:**
- Modify: `src/esper/simic/comparison.py` (move tamiyo to closure scope)

**Step 1: Move HeuristicTamiyo instantiation in head_to_head_comparison**

In `src/esper/simic/comparison.py`, find `head_to_head_comparison` function.

Add after the results dict initialization (around line 406):

```python
    # Create heuristic Tamiyo ONCE - maintains state across episodes
    tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())
```

The `heuristic_action_fn` closure will now capture this instance.

**Step 2: Add tamiyo.reset() between episodes**

In the episode loop (around line 663), after recording results, add:

```python
        # Reset tamiyo state for next episode (blueprint rotation, etc.)
        tamiyo.reset()
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v --tb=short`

Expected: Tests pass

**Step 4: Commit**

```bash
git add src/esper/simic/comparison.py
git commit -m "fix(simic): instantiate HeuristicTamiyo once per comparison, not per decision"
```

---

## Task 5: Fix Stage Constants in Rewards

**Files:**
- Modify: `src/esper/simic/rewards.py:133-136`

**Step 1: Replace hardcoded constants with leyline imports**

In `src/esper/simic/rewards.py`, change lines 133-136:

From:
```python
# Stage constants (match SeedStage IntEnum values)
STAGE_TRAINING = 3
STAGE_BLENDING = 4
STAGE_FOSSILIZED = 7
```

To:
```python
# Stage constants from leyline contract
from esper.leyline import SeedStage
STAGE_TRAINING = SeedStage.TRAINING.value
STAGE_BLENDING = SeedStage.BLENDING.value
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value
```

Note: Move the import to the top of the file with other imports.

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v --tb=short`

Expected: Tests pass (values unchanged, just sourced from leyline)

**Step 3: Commit**

```bash
git add src/esper/simic/rewards.py
git commit -m "refactor(simic): use leyline SeedStage for stage constants"
```

---

## Task 6: Update Simic Exports

**Files:**
- Modify: `src/esper/simic/__init__.py`

**Step 1: Export Action alongside SimicAction alias**

In `src/esper/simic/__init__.py`, change line 20:

```python
# Actions
from esper.leyline import Action, SimicAction  # SimicAction is deprecated alias
```

Update `__all__` (around line 84-86):

```python
__all__ = [
    # Actions
    "Action",
    "SimicAction",  # deprecated alias
```

**Step 2: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All tests pass

**Step 3: Commit**

```bash
git add src/esper/simic/__init__.py
git commit -m "refactor(simic): export Action from leyline"
```

---

## Task 7: Fix live_comparison Telemetry Zero-Padding

**Files:**
- Modify: `src/esper/simic/comparison.py:165-320` (live_comparison function)

**Context:** `live_comparison` is an observation-only mode - it doesn't execute actions or manage seeds. It just compares what heuristic vs IQL *would* decide given the same training trajectory. However, if a telemetry-mode model (37-dim or 54-dim) is loaded, it will receive zeros for telemetry features because:
1. No `SeedGradientCollector` is instantiated
2. No `seed_telemetry` is passed to `snapshot_to_features`
3. Seeds are never actually germinated (actions aren't executed)

**The Fix:** Since `live_comparison` doesn't execute actions, it can't have real seed telemetry. We should:
1. Warn if telemetry-mode model is loaded
2. Document this limitation
3. Optionally: refuse to run telemetry models in live_comparison mode

**Step 1: Add warning for telemetry models in live_comparison**

In `src/esper/simic/comparison.py`, find the `live_comparison` function (around line 165).

After loading the model (around line 176-178), add:

```python
    # Load IQL model (use_telemetry inferred from checkpoint)
    print(f"Loading IQL model from {model_path}...")
    iql, telemetry_mode = load_iql_model(model_path, device=device)
    use_telemetry = (telemetry_mode in ['seed', 'legacy'])
    print(f"  State dim: {iql.q_network.net[0].in_features} (telemetry mode: {telemetry_mode})")

    # Warn about telemetry limitation in live_comparison mode
    if use_telemetry:
        import warnings
        warnings.warn(
            f"live_comparison with telemetry_mode='{telemetry_mode}' will use zero-padded "
            "telemetry features. This mode doesn't execute actions, so no seeds are created "
            "and no gradient telemetry is collected. For accurate telemetry-aware comparison, "
            "use head_to_head_comparison instead.",
            UserWarning,
        )
```

**Step 2: Update docstring to document limitation**

Update the `live_comparison` docstring (around line 165):

```python
def live_comparison(
    model_path: str,
    n_episodes: int = 5,
    max_epochs: int = 25,
    device: str = "cpu",
) -> dict:
    """Compare IQL policy decisions against heuristic Tamiyo (observation only).

    This mode runs a single training trajectory and asks both policies what
    they would do at each step. Actions are NOT executed - both policies
    observe the same unmodified training run.

    WARNING: This mode does not support telemetry features properly. If the
    loaded model was trained with telemetry (37-dim or 54-dim), it will receive
    zeros for telemetry features since no seeds are created. Use
    head_to_head_comparison for telemetry-aware evaluation.

    Args:
        model_path: Path to saved IQL model checkpoint
        n_episodes: Number of episodes to run
        max_epochs: Maximum epochs per episode
        device: Device to run on

    Returns:
        Dictionary with comparison results including agreement rates
    """
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v --tb=short`

Expected: Tests pass

**Step 4: Commit**

```bash
git add src/esper/simic/comparison.py
git commit -m "fix(simic): warn when using telemetry model in live_comparison mode"
```

---

## Task 8: Clean Up snapshot_from_signals (Optional - Low Priority)

**Files:**
- Modify: `src/esper/simic/episodes.py:634-670`

**Step 1: Add deprecation warning or fix the function**

Option A - Delete (if truly unused):
```python
# Remove the function entirely and from exports
```

Option B - Fix to use signals.metrics:
```python
def snapshot_from_signals(
    signals,  # TrainingSignals from tamiyo
    seed_state=None,  # SeedState from kasmina, optional
) -> TrainingSnapshot:
    """Convert Tamiyo's TrainingSignals to a Simic TrainingSnapshot.

    DEPRECATED: Active code builds snapshots directly. This function
    is maintained for backwards compatibility.
    """
    import warnings
    warnings.warn(
        "snapshot_from_signals is deprecated. Build TrainingSnapshot directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Use signals.metrics.* for TrainingSignals
    metrics = signals.metrics

    # Pad history to 5 elements
    loss_hist = signals.loss_history[-5:] if signals.loss_history else []
    loss_hist = [0.0] * (5 - len(loss_hist)) + loss_hist

    acc_hist = signals.accuracy_history[-5:] if signals.accuracy_history else []
    acc_hist = [0.0] * (5 - len(acc_hist)) + acc_hist

    snapshot = TrainingSnapshot(
        epoch=metrics.epoch,
        global_step=metrics.global_step,
        train_loss=metrics.train_loss,
        val_loss=metrics.val_loss,
        loss_delta=metrics.loss_delta,
        train_accuracy=metrics.train_accuracy,
        val_accuracy=metrics.val_accuracy,
        accuracy_delta=metrics.accuracy_delta,
        plateau_epochs=metrics.plateau_epochs,
        best_val_accuracy=metrics.best_val_accuracy,
        loss_history_5=tuple(loss_hist),
        accuracy_history_5=tuple(acc_hist),
        available_slots=signals.available_slots,
    )

    # Add seed state if present
    if seed_state is not None:
        snapshot.has_active_seed = True
        snapshot.seed_stage = int(seed_state.stage)
        snapshot.seed_epochs_in_stage = seed_state.epochs_in_stage
        snapshot.seed_alpha = seed_state.alpha
        snapshot.seed_improvement = seed_state.metrics.improvement_since_stage_start

    return snapshot
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: Tests pass

**Step 3: Commit**

```bash
git add src/esper/simic/episodes.py
git commit -m "fix(simic): fix snapshot_from_signals to use signals.metrics, add deprecation warning"
```

---

## Task 9: Final Integration Test

**Files:**
- Create: `tests/test_action_unification.py`

**Step 1: Write integration test**

Create `tests/test_action_unification.py`:

```python
"""Integration tests for action unification.

Verifies that Tamiyo and Simic use the same Action enum from leyline.
"""

import pytest


class TestActionUnification:
    """Tests for unified action space."""

    def test_action_importable_from_leyline(self):
        """Action should be importable from leyline."""
        from esper.leyline import Action
        assert Action is not None
        assert len(Action) == 7  # WAIT + 4 GERMINATE + ADVANCE + CULL

    def test_simicaction_is_alias(self):
        """SimicAction should be an alias for Action."""
        from esper.leyline import Action, SimicAction
        assert SimicAction is Action

    def test_tamiyo_decision_uses_action(self):
        """TamiyoDecision.action should be an Action enum."""
        from esper.leyline import Action
        from esper.tamiyo import TamiyoDecision

        decision = TamiyoDecision(action=Action.WAIT)
        assert isinstance(decision.action, Action)

    def test_heuristic_tamiyo_returns_action(self):
        """HeuristicTamiyo.decide() should return decision with Action."""
        from esper.leyline import Action, TrainingSignals
        from esper.tamiyo import HeuristicTamiyo, HeuristicPolicyConfig

        tamiyo = HeuristicTamiyo(HeuristicPolicyConfig())
        signals = TrainingSignals()
        signals.metrics.epoch = 10
        signals.metrics.plateau_epochs = 5

        decision = tamiyo.decide(signals, active_seeds=[])

        assert isinstance(decision.action, Action)

    def test_no_tamiyoaction_exists(self):
        """TamiyoAction should not exist anymore."""
        import esper.tamiyo as tamiyo_module
        assert not hasattr(tamiyo_module, 'TamiyoAction')

    def test_action_from_decision_returns_action(self):
        """action_from_decision should return ActionTaken with Action."""
        from esper.leyline import Action
        from esper.tamiyo import TamiyoDecision
        from esper.simic.episodes import action_from_decision

        decision = TamiyoDecision(action=Action.GERMINATE_CONV)
        action_taken = action_from_decision(decision)

        assert action_taken.action == Action.GERMINATE_CONV

    def test_stage_constants_match_leyline(self):
        """Stage constants in rewards should match leyline."""
        from esper.leyline import SeedStage
        from esper.simic.rewards import STAGE_TRAINING, STAGE_BLENDING, STAGE_FOSSILIZED

        assert STAGE_TRAINING == SeedStage.TRAINING.value
        assert STAGE_BLENDING == SeedStage.BLENDING.value
        assert STAGE_FOSSILIZED == SeedStage.FOSSILIZED.value
```

**Step 2: Run the integration tests**

Run: `.venv/bin/python -m pytest tests/test_action_unification.py -v`

Expected: All tests PASS

**Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_action_unification.py
git commit -m "test: add integration tests for action unification"
```

---

## Summary

After completing all tasks:

1. **Single action space** - `Action` enum in leyline (with `SimicAction` alias)
2. **No TamiyoAction** - Deleted, Tamiyo uses `Action` directly
3. **No mapping code** - `action_from_decision` is trivial, `heuristic_action_fn` returns directly
4. **Stateful heuristic** - `HeuristicTamiyo` instantiated once per comparison
5. **No magic constants** - Stage values imported from leyline
6. **live_comparison telemetry** - Warns when telemetry model loaded (can't provide real telemetry)
7. **Fixed snapshot_from_signals** - Uses `signals.metrics.*` correctly (optional)

The leyline contract is now properly honored across all components.

## Task Summary

| Task | Priority | Description |
|------|----------|-------------|
| 1 | P0 | Rename SimicAction → Action in leyline |
| 2 | P0 | Update Tamiyo to return Action directly |
| 3 | P0 | Remove mapping code from Simic |
| 4 | P2 | Fix HeuristicTamiyo instantiation scope |
| 5 | P2 | Fix stage constants in rewards |
| 6 | P2 | Update Simic exports |
| 7 | P2 | Fix live_comparison telemetry warning |
| 8 | P3 | Clean up snapshot_from_signals (optional) |
| 9 | P1 | Final integration test |
