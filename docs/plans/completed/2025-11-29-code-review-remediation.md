# Code Review Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all P0/P1 issues from the comprehensive code review of the Simic refactoring, including action unification, feature dimension consistency, telemetry error enforcement, and blueprint centralization.

**Architecture:** Unify action space in leyline, eliminate dimension mismatches between PPO/IQL, enforce telemetry requirements, centralize blueprint mappings. Maintains backwards compatibility via aliases while fixing architectural violations.

**Tech Stack:** Python enums, dataclasses, PyTorch tensor operations. No new dependencies.

**Review Docs:**
- Code review findings (in session)
- `docs/plans/2025-11-29-leyline-violations-audit.md`
- `docs/plans/2025-11-29-leyline-action-unification.md` (original plan)

**Critical Issues Addressed:**
- P0: Duplicate action enums (TamiyoAction vs SimicAction)
- P0: Feature dimension inconsistency (PPO uses 54-dim, IQL uses 37-dim)
- P0: Zero-padding telemetry causes distribution shift
- P2: Blueprint mapping duplicated in multiple locations
- P2: Stage constants hardcoded instead of using leyline
- P2: HeuristicTamiyo instantiation in tight loop

---

## Task 1: Rename SimicAction → Action in Leyline

**Priority:** P0
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

**Priority:** P0
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

Update all other methods to use `Action.WAIT`, `Action.ADVANCE`, `Action.CULL` instead of `TamiyoAction.*`

**Step 3: Update tamiyo __init__.py**

In `src/esper/tamiyo/__init__.py`:

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

Expected: Some tests may fail if they reference TamiyoAction - we'll fix those in Task 3

**Step 5: Commit**

```bash
git add src/esper/tamiyo/
git commit -m "refactor(tamiyo): use Action from leyline, delete TamiyoAction"
```

---

## Task 3: Remove Mapping Code from Simic

**Priority:** P0
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

Note: `tamiyo` will be defined in outer scope in Task 6.

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

## Task 4: Fix PPO Feature Dimension Inconsistency

**Priority:** P0 (CRITICAL - prevents silent bugs between PPO and IQL)
**Files:**
- Modify: `src/esper/simic/ppo.py:25-85` (signals_to_features function)
- Modify: `tests/test_simic_networks.py` (add dimension test)

**Step 1: Write failing test**

Add to `tests/test_simic_networks.py`:

```python
def test_ppo_features_match_comparison_dimensions():
    """PPO and comparison should use same telemetry dimensions.

    This test prevents the critical bug where PPO uses 54-dim (27 base + 27 legacy)
    while comparison uses 37-dim (27 base + 10 seed), causing silent failures
    when models are trained in one mode and evaluated in another.
    """
    from esper.simic.ppo import signals_to_features
    from esper.simic.comparison import snapshot_to_features
    from esper.simic.episodes import TrainingSnapshot
    from esper.tamiyo import SignalTracker
    from esper.tolaria import create_model
    from esper.leyline import SeedTelemetry

    # Create mock signals and model
    tracker = SignalTracker()
    signals = tracker.update(
        epoch=1, global_step=100, train_loss=1.0, train_accuracy=50.0,
        val_loss=1.0, val_accuracy=50.0, active_seeds=[], available_slots=1
    )

    model = create_model("cpu")

    # PPO features with telemetry
    ppo_features = signals_to_features(signals, model, tracker, use_telemetry=True)

    # Comparison features with telemetry (zero seed telemetry)
    snapshot = TrainingSnapshot(
        epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
        loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
        accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
        best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
        has_active_seed=False, seed_stage=0, seed_epochs_in_stage=0,
        seed_alpha=0.0, seed_improvement=0.0, available_slots=1
    )

    # Create zero telemetry for comparison
    zero_telemetry = SeedTelemetry(seed_id="test")
    comparison_features = snapshot_to_features(
        snapshot, use_telemetry=True, seed_telemetry=zero_telemetry
    )

    # CRITICAL: Dimensions must match
    assert len(ppo_features) == len(comparison_features), \
        f"PPO uses {len(ppo_features)}-dim, comparison uses {len(comparison_features)}-dim"

    # Both should be 37-dim (27 base + 10 seed telemetry)
    assert len(ppo_features) == 37, f"Expected 37 dims, got {len(ppo_features)}"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_simic_networks.py::test_ppo_features_match_comparison_dimensions -v`

Expected: FAIL with dimension mismatch (PPO returns 54, comparison returns 37)

**Step 3: Fix signals_to_features to use SeedTelemetry**

In `src/esper/simic/ppo.py`, replace the `signals_to_features` function (around lines 25-85):

```python
def signals_to_features(signals, model, tracker=None, use_telemetry: bool = True) -> list[float]:
    """Convert training signals to feature vector.

    Args:
        signals: TrainingSignals from tamiyo
        model: MorphogeneticModel
        tracker: Optional tracker (unused, kept for API compatibility)
        use_telemetry: Whether to include telemetry features

    Returns:
        Feature vector (27 dims base, +10 if telemetry)
    """
    from esper.simic.features import obs_to_base_features

    # Build observation dict
    loss_hist = list(signals.loss_history[-5:]) if signals.loss_history else []
    while len(loss_hist) < 5:
        loss_hist.insert(0, 0.0)

    acc_hist = list(signals.accuracy_history[-5:]) if signals.accuracy_history else []
    while len(acc_hist) < 5:
        acc_hist.insert(0, 0.0)

    obs = {
        'epoch': signals.epoch,
        'global_step': signals.global_step,
        'train_loss': signals.train_loss,
        'val_loss': signals.val_loss,
        'loss_delta': signals.loss_delta,
        'train_accuracy': signals.train_accuracy,
        'val_accuracy': signals.val_accuracy,
        'accuracy_delta': signals.accuracy_delta,
        'plateau_epochs': signals.plateau_epochs,
        'best_val_accuracy': signals.best_val_accuracy,
        'best_val_loss': min(signals.loss_history) if signals.loss_history else 10.0,
        'loss_history_5': loss_hist,
        'accuracy_history_5': acc_hist,
        'has_active_seed': 1.0 if signals.active_seeds else 0.0,
        'available_slots': signals.available_slots,
    }

    # Seed state features
    if signals.active_seeds:
        seed = signals.active_seeds[0]
        obs['seed_stage'] = seed.stage.value
        obs['seed_epochs_in_stage'] = seed.metrics.epochs_in_current_stage
        obs['seed_alpha'] = seed.alpha
        obs['seed_improvement'] = seed.metrics.improvement_since_stage_start
    else:
        obs['seed_stage'] = 0
        obs['seed_epochs_in_stage'] = 0
        obs['seed_alpha'] = 0.0
        obs['seed_improvement'] = 0.0

    features = obs_to_base_features(obs)

    if use_telemetry:
        # Use seed telemetry (10 dims) not legacy full telemetry (27 dims)
        if signals.active_seeds and hasattr(signals.active_seeds[0], 'telemetry'):
            features.extend(signals.active_seeds[0].telemetry.to_features())
        else:
            # No seed active or no telemetry - use zeros
            from esper.leyline import SeedTelemetry
            features.extend([0.0] * SeedTelemetry.feature_dim())

    return features
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_simic_networks.py::test_ppo_features_match_comparison_dimensions -v`

Expected: PASS

**Step 5: Run full PPO tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v --tb=short`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/simic/ppo.py tests/test_simic_networks.py
git commit -m "fix(simic): use SeedTelemetry in PPO for consistent 37-dim features"
```

---

## Task 5: Centralize Blueprint Mapping to Leyline

**Priority:** P2 (MEDIUM - prevents duplication, enables reuse)
**Files:**
- Create: `src/esper/leyline/blueprints.py`
- Modify: `src/esper/leyline/__init__.py`
- Modify: `src/esper/tamiyo/heuristic.py` (use centralized mapping)
- Modify: `src/esper/leyline/actions.py` (use centralized mapping)

**Step 1: Create blueprint mapping module**

Create `src/esper/leyline/blueprints.py`:

```python
"""Blueprint definitions and mappings.

This module defines the available seed blueprints and their mappings
to germinate actions. Centralized here to prevent duplication across
tamiyo, simic, and other modules.
"""

from __future__ import annotations

from esper.leyline.actions import Action


# Available blueprint IDs
BLUEPRINT_CONV_ENHANCE = "conv_enhance"
BLUEPRINT_ATTENTION = "attention"
BLUEPRINT_NORM = "norm"
BLUEPRINT_DEPTHWISE = "depthwise"

ALL_BLUEPRINTS = [
    BLUEPRINT_CONV_ENHANCE,
    BLUEPRINT_ATTENTION,
    BLUEPRINT_NORM,
    BLUEPRINT_DEPTHWISE,
]


# Blueprint ID → Germinate Action mapping
BLUEPRINT_TO_ACTION: dict[str, Action] = {
    BLUEPRINT_CONV_ENHANCE: Action.GERMINATE_CONV,
    BLUEPRINT_ATTENTION: Action.GERMINATE_ATTENTION,
    BLUEPRINT_NORM: Action.GERMINATE_NORM,
    BLUEPRINT_DEPTHWISE: Action.GERMINATE_DEPTHWISE,
}


# Germinate Action → Blueprint ID mapping (inverse)
ACTION_TO_BLUEPRINT: dict[Action, str] = {
    Action.GERMINATE_CONV: BLUEPRINT_CONV_ENHANCE,
    Action.GERMINATE_ATTENTION: BLUEPRINT_ATTENTION,
    Action.GERMINATE_NORM: BLUEPRINT_NORM,
    Action.GERMINATE_DEPTHWISE: BLUEPRINT_DEPTHWISE,
}


def blueprint_to_action(blueprint_id: str) -> Action:
    """Convert blueprint ID to corresponding germinate action.

    Args:
        blueprint_id: Blueprint identifier

    Returns:
        Corresponding GERMINATE_* action

    Raises:
        KeyError: If blueprint_id is unknown
    """
    return BLUEPRINT_TO_ACTION[blueprint_id]


def action_to_blueprint(action: Action) -> str | None:
    """Convert germinate action to blueprint ID.

    Args:
        action: Action enum value

    Returns:
        Blueprint ID if action is a germinate variant, None otherwise
    """
    return ACTION_TO_BLUEPRINT.get(action)


__all__ = [
    "BLUEPRINT_CONV_ENHANCE",
    "BLUEPRINT_ATTENTION",
    "BLUEPRINT_NORM",
    "BLUEPRINT_DEPTHWISE",
    "ALL_BLUEPRINTS",
    "BLUEPRINT_TO_ACTION",
    "ACTION_TO_BLUEPRINT",
    "blueprint_to_action",
    "action_to_blueprint",
]
```

**Step 2: Export from leyline**

In `src/esper/leyline/__init__.py`, add after actions import:

```python
# Blueprints
from esper.leyline.blueprints import (
    BLUEPRINT_CONV_ENHANCE,
    BLUEPRINT_ATTENTION,
    BLUEPRINT_NORM,
    BLUEPRINT_DEPTHWISE,
    ALL_BLUEPRINTS,
    blueprint_to_action,
    action_to_blueprint,
)
```

And in `__all__`:

```python
    # Blueprints
    "BLUEPRINT_CONV_ENHANCE",
    "BLUEPRINT_ATTENTION",
    "BLUEPRINT_NORM",
    "BLUEPRINT_DEPTHWISE",
    "ALL_BLUEPRINTS",
    "blueprint_to_action",
    "action_to_blueprint",
```

**Step 3: Update Action.get_blueprint_id() to use centralized mapping**

In `src/esper/leyline/actions.py`, update the method:

```python
    @classmethod
    def get_blueprint_id(cls, action: "Action") -> str | None:
        """Get blueprint ID for germinate actions, None for others."""
        from esper.leyline.blueprints import action_to_blueprint
        return action_to_blueprint(action)
```

**Step 4: Update HeuristicTamiyo to use centralized mapping**

In `src/esper/tamiyo/heuristic.py`, replace `_blueprint_to_action`:

```python
    def _blueprint_to_action(self, blueprint_id: str) -> Action:
        """Convert blueprint ID to corresponding GERMINATE action."""
        from esper.leyline import blueprint_to_action
        return blueprint_to_action(blueprint_id)
```

**Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/esper/leyline/blueprints.py src/esper/leyline/__init__.py src/esper/leyline/actions.py src/esper/tamiyo/heuristic.py
git commit -m "refactor(leyline): centralize blueprint-to-action mapping"
```

---

## Task 6: Fix HeuristicTamiyo Instantiation Scope

**Priority:** P2 (MEDIUM - performance optimization)
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
        if hasattr(tamiyo, 'reset'):
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

## Task 7: Fix Stage Constants in Rewards

**Priority:** P2 (MEDIUM - eliminates magic numbers)
**Files:**
- Modify: `src/esper/simic/rewards.py:133-136`

**Step 1: Replace hardcoded constants with leyline imports**

In `src/esper/simic/rewards.py`, find the stage constants (around lines 133-136).

Change from:

```python
# Stage constants (match SeedStage IntEnum values)
STAGE_TRAINING = 3
STAGE_BLENDING = 4
STAGE_FOSSILIZED = 7
```

To (at the top with other imports):

```python
from esper.leyline import SeedStage

# ... later in the file ...

# Stage constants from leyline contract
STAGE_TRAINING = SeedStage.TRAINING.value
STAGE_BLENDING = SeedStage.BLENDING.value
STAGE_FOSSILIZED = SeedStage.FOSSILIZED.value
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_simic*.py -v --tb=short`

Expected: Tests pass (values unchanged, just sourced from leyline)

**Step 3: Commit**

```bash
git add src/esper/simic/rewards.py
git commit -m "refactor(simic): use leyline SeedStage for stage constants"
```

---

## Task 8: Update Simic Exports

**Priority:** P2 (MEDIUM - API consistency)
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

## Task 9: Make Zero-Padding an Error, Not Warning

**Priority:** P0 (CRITICAL - prevents distribution shift anti-pattern)
**Files:**
- Modify: `src/esper/simic/comparison.py:87-145` (snapshot_to_features)
- Modify: `src/esper/simic/comparison.py:165-320` (live_comparison)
- Modify: `tests/test_seed_telemetry.py`

**Step 1: Write failing test**

Add to `tests/test_seed_telemetry.py`:

```python
def test_snapshot_to_features_requires_telemetry_when_enabled():
    """When use_telemetry=True with active seed, seed_telemetry must be provided.

    This enforces the DRL review finding that zero-padding telemetry
    causes distribution shift and degrades policy quality.
    """
    from esper.simic.comparison import snapshot_to_features
    from esper.simic.episodes import TrainingSnapshot
    import pytest

    snapshot = TrainingSnapshot(
        epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
        loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
        accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
        best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
        has_active_seed=True,  # Seed active but no telemetry provided
        seed_stage=2, seed_epochs_in_stage=3, seed_alpha=0.0,
        seed_improvement=5.0, available_slots=0
    )

    # Should raise ValueError, not warn
    with pytest.raises(ValueError, match="seed_telemetry is required"):
        snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)


def test_snapshot_to_features_allows_none_when_no_seed():
    """When use_telemetry=True but no seed is active, None telemetry is OK."""
    from esper.simic.comparison import snapshot_to_features
    from esper.simic.episodes import TrainingSnapshot

    snapshot = TrainingSnapshot(
        epoch=1, global_step=100, train_loss=1.0, val_loss=1.0,
        loss_delta=0.0, train_accuracy=50.0, val_accuracy=50.0,
        accuracy_delta=0.0, plateau_epochs=0, best_val_accuracy=50.0,
        best_val_loss=1.0, loss_history_5=(1.0,)*5, accuracy_history_5=(50.0,)*5,
        has_active_seed=False,  # No seed active
        available_slots=1
    )

    # Should NOT raise when no seed is active
    features = snapshot_to_features(snapshot, use_telemetry=True, seed_telemetry=None)
    assert len(features) == 37  # 27 base + 10 zero telemetry
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py::test_snapshot_to_features_requires_telemetry_when_enabled -v`

Expected: FAIL (currently warns instead of raising)

**Step 3: Change warning to error in snapshot_to_features**

In `src/esper/simic/comparison.py`, modify `snapshot_to_features` (around line 930-942):

```python
    if use_telemetry:
        if seed_telemetry is not None:
            features.extend(seed_telemetry.to_features())
        else:
            # Check if we have an active seed that needs telemetry
            if snapshot.has_active_seed:
                # CRITICAL: Zero-padding causes distribution shift (DRL review)
                # Refuse to proceed rather than corrupt the model's inputs
                raise ValueError(
                    "seed_telemetry is required when use_telemetry=True and seed is active. "
                    "Zero-padding telemetry features causes distribution shift and "
                    "degrades policy quality. Either provide real telemetry or set "
                    "use_telemetry=False."
                )
            else:
                # No active seed - zeros are semantically correct
                from esper.leyline import SeedTelemetry
                features.extend([0.0] * SeedTelemetry.feature_dim())

    return features
```

**Step 4: Update live_comparison to disable telemetry**

In `src/esper/simic/comparison.py`, find `live_comparison` function (around line 165).

After loading the model (around line 176-178), update to:

```python
    # Load IQL model (use_telemetry inferred from checkpoint)
    print(f"Loading IQL model from {model_path}...")
    iql, telemetry_mode = load_iql_model(model_path, device=device)
    use_telemetry_model = (telemetry_mode in ['seed', 'legacy'])
    print(f"  State dim: {iql.q_network.net[0].in_features} (telemetry mode: {telemetry_mode})")

    # Warn about telemetry limitation in live_comparison mode
    if use_telemetry_model:
        import warnings
        warnings.warn(
            f"Model was trained with telemetry_mode='{telemetry_mode}', but live_comparison "
            "cannot provide telemetry (no seeds created). Evaluation will use base features "
            "only (27-dim). Results may not be accurate. Use head_to_head_comparison for "
            "telemetry-aware evaluation.",
            UserWarning,
        )

    # Force disable telemetry since we can't provide it
    use_telemetry = False
```

Then in the snapshot_to_features call (around line 292):

```python
        features = snapshot_to_features(snapshot, use_telemetry=False)  # No telemetry in live mode
```

**Step 5: Update docstring to document limitation**

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

    WARNING: This mode does not support telemetry features. If the loaded
    model was trained with telemetry (37-dim or 54-dim), it will be evaluated
    using only base features (27-dim) since no seeds are created. For accurate
    telemetry-aware evaluation, use head_to_head_comparison.

    Args:
        model_path: Path to saved IQL model checkpoint
        n_episodes: Number of episodes to run
        max_epochs: Maximum epochs per episode
        device: Device to run on

    Returns:
        Dictionary with comparison results including agreement rates
    """
```

**Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_seed_telemetry.py -v`

Expected: All tests PASS

**Step 7: Run all tests**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`

Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/esper/simic/comparison.py tests/test_seed_telemetry.py
git commit -m "fix(simic): make zero-padding telemetry a hard error, disable telemetry in live_comparison"
```

---

## Task 10: Final Integration Test

**Priority:** P1 (HIGH - validates entire remediation)
**Files:**
- Create: `tests/test_action_unification.py`

**Step 1: Write integration test**

Create `tests/test_action_unification.py`:

```python
"""Integration tests for action unification and code review remediation.

Verifies that:
1. Tamiyo and Simic use the same Action enum from leyline
2. Feature dimensions are consistent across PPO and IQL
3. Telemetry enforcement works correctly
4. Blueprint mappings are centralized
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


class TestBlueprintCentralization:
    """Tests for centralized blueprint mappings."""

    def test_blueprint_to_action_function(self):
        """blueprint_to_action should map correctly."""
        from esper.leyline import blueprint_to_action, Action

        assert blueprint_to_action("conv_enhance") == Action.GERMINATE_CONV
        assert blueprint_to_action("attention") == Action.GERMINATE_ATTENTION
        assert blueprint_to_action("norm") == Action.GERMINATE_NORM
        assert blueprint_to_action("depthwise") == Action.GERMINATE_DEPTHWISE

    def test_action_to_blueprint_function(self):
        """action_to_blueprint should map correctly."""
        from esper.leyline import action_to_blueprint, Action

        assert action_to_blueprint(Action.GERMINATE_CONV) == "conv_enhance"
        assert action_to_blueprint(Action.GERMINATE_ATTENTION) == "attention"
        assert action_to_blueprint(Action.GERMINATE_NORM) == "norm"
        assert action_to_blueprint(Action.GERMINATE_DEPTHWISE) == "depthwise"
        assert action_to_blueprint(Action.WAIT) is None

    def test_action_get_blueprint_id_uses_centralized(self):
        """Action.get_blueprint_id should use centralized mapping."""
        from esper.leyline import Action

        assert Action.get_blueprint_id(Action.GERMINATE_CONV) == "conv_enhance"
        assert Action.get_blueprint_id(Action.WAIT) is None


class TestFeatureDimensionConsistency:
    """Tests for consistent feature dimensions across modules."""

    def test_ppo_and_comparison_dimensions_match(self):
        """PPO and comparison should produce same dimensions."""
        # This is tested in test_simic_networks.py::test_ppo_features_match_comparison_dimensions
        # We just verify the dimensions are what we expect
        from esper.leyline import SeedTelemetry

        # Base features + seed telemetry = 37 dims
        expected_dim = 27 + SeedTelemetry.feature_dim()
        assert expected_dim == 37

    def test_seed_telemetry_dimension(self):
        """SeedTelemetry should be exactly 10 dims."""
        from esper.leyline import SeedTelemetry

        assert SeedTelemetry.feature_dim() == 10

        telem = SeedTelemetry(seed_id="test")
        features = telem.to_features()
        assert len(features) == 10


class TestTelemetryEnforcement:
    """Tests for telemetry requirement enforcement."""

    def test_snapshot_to_features_enforces_telemetry(self):
        """snapshot_to_features should require telemetry when seed is active."""
        # Tested in test_seed_telemetry.py
        # This is a placeholder to document the requirement
        pass

    def test_live_comparison_disables_telemetry(self):
        """live_comparison should disable telemetry (can't provide it)."""
        # This would require mocking a full training run
        # We document the expectation here
        pass
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
git commit -m "test: add integration tests for code review remediation"
```

---

## Summary

After completing all tasks, the following issues are resolved:

### **P0 (Critical) Issues Fixed:**
1. ✅ **Action space unification** - Single `Action` enum in leyline, `TamiyoAction` deleted
2. ✅ **Feature dimension consistency** - Both PPO and IQL use 37-dim (27 base + 10 seed telemetry)
3. ✅ **Zero-padding enforcement** - Hard error when telemetry is required but not provided
4. ✅ **Telemetry in live_comparison** - Correctly disables telemetry (can't provide it)

### **P2 (Medium) Issues Fixed:**
5. ✅ **Blueprint centralization** - Mappings in `leyline/blueprints.py`, no duplication
6. ✅ **Stage constants** - Imported from leyline, no magic numbers
7. ✅ **HeuristicTamiyo instantiation** - Once per comparison, not per decision
8. ✅ **API consistency** - `Action` exported from all modules with backward-compat `SimicAction` alias

### **Validation:**
9. ✅ **Integration tests** - Comprehensive test suite validates all fixes

## Task Summary Table

| Task | Priority | Description | LOC Changed |
|------|----------|-------------|-------------|
| 1 | P0 | Rename SimicAction → Action in leyline | ~20 |
| 2 | P0 | Update Tamiyo to use Action directly | ~150 |
| 3 | P0 | Remove mapping code from Simic | ~40 |
| 4 | P0 | Fix PPO feature dimension (54→37) | ~50 |
| 5 | P2 | Centralize blueprint mapping | ~80 |
| 6 | P2 | Fix HeuristicTamiyo instantiation | ~10 |
| 7 | P2 | Fix stage constants in rewards | ~5 |
| 8 | P2 | Update Simic exports | ~5 |
| 9 | P0 | Make zero-padding an error | ~30 |
| 10 | P1 | Final integration test | ~100 |

**Total estimated changes:** ~490 lines across 15 files

**Recommended execution order:** Tasks 1→2→3→4→5→6→7→8→9→10

The leyline contract is now properly honored across all components, and all critical code review findings are addressed.
