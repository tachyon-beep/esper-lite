"""Tamiyo Heuristic - Heuristic-based strategic controller.

Implements a rule-based policy for seed lifecycle management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

from esper.leyline import Action, SeedStage, is_terminal_stage, is_failure_stage, TrainingSignals
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

    # When to germinate
    plateau_epochs_to_germinate: int = 3
    min_epochs_before_germinate: int = 5

    # Training phase
    min_training_epochs: int = 3
    training_improvement_threshold: float = 1.0  # Min % improvement to advance

    # Blending phase
    blending_epochs: int = 5

    # Culling
    cull_after_epochs_without_improvement: int = 5
    cull_if_accuracy_drops_by: float = 2.0  # % drop triggers cull

    # Blueprint selection
    blueprint_rotation: list[str] = field(
        default_factory=lambda: ["conv_enhance", "attention", "norm", "depthwise"]
    )


class HeuristicTamiyo:
    """Heuristic-based Tamiyo policy (Tiny Tamiyo).

    Makes decisions based on simple rules:
    - Germinate when training plateaus
    - Advance when seed shows improvement
    - Cull when seed hurts performance
    """

    def __init__(self, config: HeuristicPolicyConfig | None = None):
        self.config = config or HeuristicPolicyConfig()
        self._blueprint_index = 0
        self._germination_count = 0
        self._decisions_made: list[TamiyoDecision] = []

    def decide(self, signals: TrainingSignals, active_seeds: list["SeedState"]) -> TamiyoDecision:
        """Make a decision based on training signals.

        Args:
            signals: Training signals from the environment
            active_seeds: List of active seed states

        Returns:
            TamiyoDecision describing the action to take
        """

        # Check if we have any active seeds
        if not active_seeds:
            # No active seeds - should we germinate?
            decision = self._decide_germination(signals)
        else:
            # Have active seeds - manage them
            decision = self._decide_seed_management(signals, active_seeds)

        self._decisions_made.append(decision)
        return decision

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

    def _decide_seed_management(
        self,
        signals: TrainingSignals,
        active_seeds: list["SeedState"],
    ) -> TamiyoDecision:
        """Decide how to manage active seeds."""

        for seed in active_seeds:
            # Skip terminal and failure states using leyline helpers
            if is_terminal_stage(seed.stage) or is_failure_stage(seed.stage):
                continue

            # Handle based on current stage
            if seed.stage == SeedStage.GERMINATED:
                # Ready to start training
                return TamiyoDecision(
                    action=Action.ADVANCE,
                    target_seed_id=seed.seed_id,
                    reason="Seed germinated, starting isolated training",
                )

            elif seed.stage == SeedStage.TRAINING:
                decision = self._evaluate_training_seed(signals, seed)
                if decision.action != Action.WAIT:
                    return decision

            elif seed.stage == SeedStage.BLENDING:
                decision = self._evaluate_blending_seed(signals, seed)
                if decision.action != Action.WAIT:
                    return decision

            elif seed.stage == SeedStage.SHADOWING:
                # Shadowing stage - continue advancing toward fossilization
                return TamiyoDecision(
                    action=Action.ADVANCE,
                    target_seed_id=seed.seed_id,
                    reason="Shadowing complete, advancing to probationary",
                )

            elif seed.stage == SeedStage.PROBATIONARY:
                # Probationary stage - advance to fossilized
                return TamiyoDecision(
                    action=Action.ADVANCE,
                    target_seed_id=seed.seed_id,
                    reason="Probationary complete, fossilizing seed",
                )

        return TamiyoDecision(
            action=Action.WAIT,
            reason="Seeds progressing normally"
        )

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

    def _get_next_blueprint(self) -> str:
        """Get the next blueprint to try (round-robin)."""
        blueprints = self.config.blueprint_rotation
        blueprint_id = blueprints[self._blueprint_index % len(blueprints)]
        self._blueprint_index += 1
        return blueprint_id

    def _blueprint_to_action(self, blueprint_id: str) -> Action:
        """Convert blueprint ID to corresponding GERMINATE action."""
        from esper.leyline import blueprint_to_action
        return blueprint_to_action(blueprint_id)

    def reset(self) -> None:
        """Reset policy state."""
        self._blueprint_index = 0
        self._germination_count = 0
        self._decisions_made.clear()

    @property
    def decisions(self) -> list[TamiyoDecision]:
        """Get history of decisions made."""
        return self._decisions_made.copy()


__all__ = [
    "TamiyoPolicy",
    "HeuristicPolicyConfig",
    "HeuristicTamiyo",
]
