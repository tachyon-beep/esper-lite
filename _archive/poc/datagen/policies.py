"""Behavior policy wrapper with epsilon-greedy and probability logging."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from esper.datagen.configs import (
    ActionProbabilities,
    BehaviorPolicyConfig,
    POLICY_PRESETS,
)

if TYPE_CHECKING:
    from esper.tamiyo import TrainingSignals


class BehaviorPolicy:
    """Wraps Kasmina-style policy with ε-greedy and probability logging."""

    def __init__(self, config: BehaviorPolicyConfig):
        self.config = config

    def decide(self, signals: "TrainingSignals") -> tuple[str, ActionProbabilities]:
        """Make a decision with probability logging.

        Returns:
            Tuple of (action_name, action_probabilities)
        """
        # Compute greedy action probabilities (softmax over scores)
        scores = self._compute_action_scores(signals)
        greedy_probs = self._softmax(scores, temperature=self.config.temperature)

        # Apply epsilon-greedy
        if self.config.epsilon > 0 and random.random() < self.config.epsilon:
            # Random action
            action = random.choice(list(greedy_probs.keys()))
        else:
            # Greedy action
            action = max(greedy_probs, key=greedy_probs.get)

        # Build action probabilities
        probs = ActionProbabilities.from_decision(
            greedy_probs=greedy_probs,
            sampled_action=action,
            epsilon=self.config.epsilon,
        )

        return action, probs

    def _compute_action_scores(self, signals: "TrainingSignals") -> dict[str, float]:
        """Compute raw scores for each action based on signals.

        Higher score = more likely to be selected.
        """
        scores = {
            "WAIT": 0.0,
            "GERMINATE": -10.0,  # Default: don't germinate
            "ADVANCE": -10.0,
            "CULL": -10.0,
        }

        epoch = signals.epoch
        plateau = signals.plateau_epochs
        has_slots = signals.available_slots > 0
        has_seed = len(signals.active_seeds) > 0

        # GERMINATE scoring
        if has_slots and not has_seed:
            if epoch >= self.config.min_epochs_before_germinate:
                if plateau >= self.config.plateau_epochs_to_germinate:
                    scores["GERMINATE"] = 5.0 + plateau * 0.5

        # ADVANCE scoring (if we have an active seed)
        if has_seed:
            seed = signals.active_seeds[0]
            if hasattr(seed, 'metrics'):
                improvement = seed.metrics.improvement_since_stage_start
                epochs_in_stage = seed.metrics.epochs_in_current_stage

                # Score based on improvement
                if improvement > 0.5 and epochs_in_stage >= 3:
                    scores["ADVANCE"] = 3.0 + improvement

        # CULL scoring
        if has_seed:
            seed = signals.active_seeds[0]
            if hasattr(seed, 'metrics'):
                epochs_no_improve = seed.metrics.epochs_in_current_stage
                improvement = seed.metrics.improvement_since_stage_start

                if epochs_no_improve >= self.config.cull_after_epochs_without_improvement:
                    if improvement <= 0:
                        scores["CULL"] = 4.0 + epochs_no_improve * 0.3

        # WAIT is default - boost if nothing else is good
        if all(s < 0 for a, s in scores.items() if a != "WAIT"):
            scores["WAIT"] = 2.0

        return scores

    def _softmax(self, scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
        """Convert scores to probabilities via softmax."""
        max_score = max(scores.values())
        exp_scores = {
            a: math.exp((s - max_score) / temperature)
            for a, s in scores.items()
        }
        total = sum(exp_scores.values())
        return {a: e / total for a, e in exp_scores.items()}


def create_policy(preset_id: str, epsilon: float | None = None) -> BehaviorPolicy:
    """Create a behavior policy from preset.

    Args:
        preset_id: Name of preset (e.g., "baseline", "aggressive")
        epsilon: Optional override for epsilon-greedy exploration

    Returns:
        BehaviorPolicy instance
    """
    config = BehaviorPolicyConfig.from_preset(preset_id)
    if epsilon is not None:
        config = config.with_epsilon(epsilon)
    return BehaviorPolicy(config)
