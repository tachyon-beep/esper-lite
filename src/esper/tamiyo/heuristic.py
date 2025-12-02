"""Tamiyo Heuristic - Rule-based strategic controller.

Implements a heuristic policy for seed lifecycle management.
Stages auto-advance via SeedSlot.step_epoch(); this controller only handles:
- GERMINATE: when to start a new seed
- FOSSILIZE: when to permanently integrate a seed (from PROBATIONARY only)
- CULL: when to abandon a failing seed
- WAIT: let the system proceed normally
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

from esper.leyline import SeedStage, is_terminal_stage, is_failure_stage, TrainingSignals
from esper.leyline.actions import build_action_enum
from esper.tamiyo.decisions import TamiyoDecision

if TYPE_CHECKING:
    from esper.kasmina import SeedState


class TamiyoPolicy(Protocol):
    """Protocol for Tamiyo policy implementations."""

    def decide(self, signals: TrainingSignals, active_seeds: list["SeedState"]) -> TamiyoDecision:
        """Make a decision based on current training signals."""
        ...

    def reset(self) -> None:
        """Reset policy state."""
        ...


@dataclass
class HeuristicPolicyConfig:
    """Configuration for the heuristic policy."""

    # Germination triggers
    plateau_epochs_to_germinate: int = 3
    min_epochs_before_germinate: int = 5

    # Culling thresholds
    cull_after_epochs_without_improvement: int = 5
    cull_if_accuracy_drops_by: float = 2.0  # % drop triggers cull

    # Fossilization threshold
    min_improvement_to_fossilize: float = 0.0  # Any positive improvement

    # Anti-thrashing: cooldown after cull
    embargo_epochs_after_cull: int = 5

    # Blueprint selection with penalty tracking
    blueprint_rotation: list[str] = field(
        default_factory=lambda: ["conv_enhance", "attention", "norm", "depthwise"]
    )
    blueprint_penalty_on_cull: float = 2.0
    blueprint_penalty_decay: float = 0.5
    blueprint_penalty_threshold: float = 3.0


class HeuristicTamiyo:
    """Heuristic-based Tamiyo policy.

    Simplified decision making since stages auto-advance:
    - Germinate when training plateaus (no active seed)
    - Fossilize when seed reaches PROBATIONARY with improvement
    - Cull when seed is failing
    - Wait otherwise (let auto-advance handle stage transitions)
    """

    def __init__(self, config: HeuristicPolicyConfig | None = None, topology: str = "cnn"):
        self.config = config or HeuristicPolicyConfig()
        self.topology = topology
        self._action_enum = build_action_enum(topology)
        self._blueprint_index = 0
        self._germination_count = 0
        self._decisions_made: list[TamiyoDecision] = []
        self._last_cull_epoch: int = -100  # Start with no embargo
        self._blueprint_penalties: dict[str, float] = {}

    def decide(self, signals: TrainingSignals, active_seeds: list["SeedState"]) -> TamiyoDecision:
        """Make a decision based on training signals."""
        Action = self._action_enum

        # Filter to only non-terminal, non-failure seeds
        live_seeds = [
            s for s in active_seeds
            if not is_terminal_stage(s.stage) and not is_failure_stage(s.stage)
        ]

        if not live_seeds:
            decision = self._decide_germination(signals)
        else:
            decision = self._decide_seed_management(signals, live_seeds)

        self._decisions_made.append(decision)
        return decision

    def _decide_germination(self, signals: TrainingSignals) -> TamiyoDecision:
        """Decide whether to germinate a new seed."""
        Action = self._action_enum

        # Embargo: prevent thrashing after cull
        epochs_since_cull = signals.metrics.epoch - self._last_cull_epoch
        if epochs_since_cull < self.config.embargo_epochs_after_cull:
            return TamiyoDecision(
                action=Action.WAIT,
                reason=f"Embargo ({epochs_since_cull}/{self.config.embargo_epochs_after_cull} since cull)"
            )

        # Too early in training
        if signals.metrics.epoch < self.config.min_epochs_before_germinate:
            return TamiyoDecision(
                action=Action.WAIT,
                reason=f"Too early (epoch {signals.metrics.epoch})"
            )

        # Check for plateau
        if signals.metrics.plateau_epochs >= self.config.plateau_epochs_to_germinate:
            blueprint_id = self._get_next_blueprint()
            germinate_action = getattr(Action, f"GERMINATE_{blueprint_id.upper()}")
            self._germination_count += 1
            return TamiyoDecision(
                action=germinate_action,
                reason=f"Plateau detected ({signals.metrics.plateau_epochs} epochs)",
                confidence=min(1.0, signals.metrics.plateau_epochs / 5.0),
            )

        return TamiyoDecision(
            action=Action.WAIT,
            reason="Training progressing normally"
        )

    def _decide_seed_management(
        self,
        signals: TrainingSignals,
        active_seeds: list["SeedState"],
    ) -> TamiyoDecision:
        """Decide how to manage active seeds."""
        Action = self._action_enum

        for seed in active_seeds:
            stage = seed.stage
            improvement = seed.metrics.improvement_since_stage_start
            epochs_in_stage = seed.epochs_in_stage

            # GERMINATED: wait for auto-advance to TRAINING
            if stage == SeedStage.GERMINATED:
                return TamiyoDecision(
                    action=Action.WAIT,
                    target_seed_id=seed.seed_id,
                    reason="Awaiting auto-advance to TRAINING"
                )

            # TRAINING: check for failure, otherwise wait for auto-advance
            if stage == SeedStage.TRAINING:
                if self._should_cull(improvement, epochs_in_stage):
                    return self._cull_seed(signals, seed, "Failing in TRAINING")
                return TamiyoDecision(
                    action=Action.WAIT,
                    target_seed_id=seed.seed_id,
                    reason=f"Training: {improvement:+.2f}% improvement"
                )

            # BLENDING: check for failure, otherwise wait for auto-advance
            if stage == SeedStage.BLENDING:
                if self._should_cull(improvement, epochs_in_stage):
                    return self._cull_seed(signals, seed, "Failing in BLENDING")
                return TamiyoDecision(
                    action=Action.WAIT,
                    target_seed_id=seed.seed_id,
                    reason=f"Blending: alpha={seed.alpha:.2f}"
                )

            # SHADOWING: check for failure, otherwise wait for auto-advance
            if stage == SeedStage.SHADOWING:
                if self._should_cull(improvement, epochs_in_stage):
                    return self._cull_seed(signals, seed, "Failing in SHADOWING")
                return TamiyoDecision(
                    action=Action.WAIT,
                    target_seed_id=seed.seed_id,
                    reason="Shadowing in progress"
                )

            # PROBATIONARY: decision point - fossilize or cull
            if stage == SeedStage.PROBATIONARY:
                total_improvement = seed.metrics.total_improvement

                if total_improvement > self.config.min_improvement_to_fossilize:
                    return TamiyoDecision(
                        action=Action.FOSSILIZE,
                        target_seed_id=seed.seed_id,
                        reason=f"Fossilizing: {total_improvement:+.2f}% total improvement",
                        confidence=min(1.0, total_improvement / 5.0),
                    )
                else:
                    return self._cull_seed(
                        signals, seed,
                        f"No improvement in PROBATIONARY ({total_improvement:+.2f}%)"
                    )

        return TamiyoDecision(
            action=Action.WAIT,
            reason="Seeds progressing normally"
        )

    def _should_cull(self, improvement: float, epochs_in_stage: int) -> bool:
        """Check if seed should be culled based on performance."""
        if epochs_in_stage < self.config.cull_after_epochs_without_improvement:
            return False
        return improvement < -self.config.cull_if_accuracy_drops_by

    def _cull_seed(
        self,
        signals: TrainingSignals,
        seed: "SeedState",
        reason: str,
    ) -> TamiyoDecision:
        """Create a cull decision and track for embargo."""
        Action = self._action_enum
        self._last_cull_epoch = signals.metrics.epoch
        self._apply_blueprint_penalty(seed.blueprint_id)
        return TamiyoDecision(
            action=Action.CULL,
            target_seed_id=seed.seed_id,
            reason=reason,
        )

    def _get_next_blueprint(self) -> str:
        """Get next blueprint, avoiding heavily penalized ones."""
        blueprints = self.config.blueprint_rotation

        # Decay penalties
        for bp in list(self._blueprint_penalties.keys()):
            self._blueprint_penalties[bp] *= self.config.blueprint_penalty_decay
            if self._blueprint_penalties[bp] < 0.1:
                del self._blueprint_penalties[bp]

        # Find blueprint below penalty threshold
        for _ in range(len(blueprints)):
            blueprint_id = blueprints[self._blueprint_index % len(blueprints)]
            self._blueprint_index += 1
            penalty = self._blueprint_penalties.get(blueprint_id, 0.0)
            if penalty < self.config.blueprint_penalty_threshold:
                return blueprint_id

        # All penalized - pick lowest
        return min(blueprints, key=lambda bp: self._blueprint_penalties.get(bp, 0.0))

    def _apply_blueprint_penalty(self, blueprint_id: str) -> None:
        """Penalize a blueprint that was culled."""
        current = self._blueprint_penalties.get(blueprint_id, 0.0)
        self._blueprint_penalties[blueprint_id] = current + self.config.blueprint_penalty_on_cull

    def reset(self) -> None:
        """Reset policy state."""
        self._blueprint_index = 0
        self._germination_count = 0
        self._decisions_made.clear()
        self._last_cull_epoch = -100
        self._blueprint_penalties.clear()

    @property
    def decisions(self) -> list[TamiyoDecision]:
        """Get history of decisions made."""
        return self._decisions_made.copy()


__all__ = [
    "TamiyoPolicy",
    "HeuristicPolicyConfig",
    "HeuristicTamiyo",
]
