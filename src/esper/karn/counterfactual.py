"""Karn Counterfactual Engine - Causal attribution for seeds.

Implements two types of attribution:
1. Removal Cost (fast, confounded) - ablation at epoch T
2. Causal Contribution (slow, unconfounded) - parallel control runs

The factorial matrix computes removal cost for all seed combinations,
enabling Shapley value estimation for fair attribution.

IMPORTANT: Removal cost measures "how much worse if we disable seed A now?"
This is NOT the same as "how much value did seed A add?" (causal contribution).
The host has adapted assuming seeds were present, creating confounding.
For publication-grade causal claims, use parallel control runs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Callable, Literal
import random

from esper.nissa import get_hub
from esper.leyline import TelemetryEvent, TelemetryEventType


@dataclass(frozen=True)
class CounterfactualConfig:
    """User-configurable counterfactual strategy.

    Immutable to prevent mid-run mutations.
    """

    # Strategy selection
    strategy: Literal["auto", "full_factorial", "shapley", "ablation_only"] = "auto"

    # Auto-strategy thresholds
    full_factorial_max_seeds: int = 4  # Switch to Shapley above this
    shapley_samples: int = 20  # Permutation samples for Shapley

    # Force full factorial regardless of seed count (use with caution)
    force_full_factorial: bool = False

    # Compute budget controls
    max_configurations: int | None = None  # Hard cap, None = unlimited
    timeout_seconds: float | None = None  # Abort if exceeded

    def effective_strategy(self, n_active_seeds: int) -> str:
        """Determine strategy based on config and seed count."""
        if self.force_full_factorial:
            return "full_factorial"
        if self.strategy != "auto":
            return self.strategy
        if n_active_seeds <= self.full_factorial_max_seeds:
            return "full_factorial"
        return "shapley"


@dataclass
class CounterfactualResult:
    """Result of evaluating a single seed configuration."""

    config: tuple[bool, ...]  # (True, False, True) = r0c0 on, r0c1 off, r0c2 on
    slot_ids: tuple[str, ...]  # ("r0c0", "r0c1", "r0c2")
    alpha_settings: dict[str, float] = field(default_factory=dict)  # {slot: alpha}
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    per_class_accuracy: dict[int, float] | None = None  # Research mode only


@dataclass
class CounterfactualMatrix:
    """Full counterfactual matrix for an epoch.

    Contains results for all evaluated configurations.
    """

    epoch: int = 0
    configs: list[CounterfactualResult] = field(default_factory=list)
    strategy_used: str = ""
    compute_time_seconds: float = 0.0

    # Derived metrics (computed lazily)
    _marginal_contributions: dict[str, float] | None = field(
        default=None, repr=False
    )
    _shapley_values: dict[str, ShapleyEstimate] | None = field(
        default=None, repr=False
    )

    @property
    def baseline_accuracy(self) -> float:
        """Accuracy with all seeds disabled (000...)."""
        for result in self.configs:
            if not any(result.config):  # All False
                return result.val_accuracy
        return 0.0

    @property
    def full_accuracy(self) -> float:
        """Accuracy with all seeds enabled (111...)."""
        for result in self.configs:
            if all(result.config):  # All True
                return result.val_accuracy
        return 0.0

    def marginal_contribution(self, slot_id: str) -> float:
        """Get marginal contribution for a slot.

        Marginal = avg(acc with slot on) - avg(acc with slot off)
        """
        if self._marginal_contributions is None:
            self._compute_marginal_contributions()
        return self._marginal_contributions.get(slot_id, 0.0)

    def _compute_marginal_contributions(self) -> None:
        """Compute marginal contributions for all slots."""
        if not self.configs:
            self._marginal_contributions = {}
            return

        slot_ids = self.configs[0].slot_ids
        self._marginal_contributions = {}

        for i, slot_id in enumerate(slot_ids):
            with_slot = [r.val_accuracy for r in self.configs if r.config[i]]
            without_slot = [r.val_accuracy for r in self.configs if not r.config[i]]

            if with_slot and without_slot:
                avg_with = sum(with_slot) / len(with_slot)
                avg_without = sum(without_slot) / len(without_slot)
                self._marginal_contributions[slot_id] = avg_with - avg_without
            else:
                self._marginal_contributions[slot_id] = 0.0


@dataclass
class ShapleyEstimate:
    """Shapley value estimate with uncertainty (P1 from expert review)."""

    mean: float = 0.0
    std: float = 0.0
    n_samples: int = 0
    algorithm: str = "permutation_antithetic"

    def is_significant(self, confidence: float = 0.95) -> bool:
        """True if contribution is significantly different from zero."""
        # Using normal approximation: mean - 2*std > 0
        z = 1.96 if confidence == 0.95 else 2.58  # 95% or 99%
        return abs(self.mean) > z * self.std if self.std > 0 else self.mean != 0


@dataclass
class InteractionTerm:
    """Pairwise interaction between two slots."""

    slot_pair: tuple[str, str]
    interaction: float = 0.0  # I_ij = f({i,j}) - f({i}) - f({j}) + f(empty)
    smoothed: float = 0.0  # EMA of interaction
    derivative: float = 0.0  # Change rate

    @property
    def regime(self) -> str:
        """Classify interaction regime."""
        if self.interaction > 0.5:  # Threshold for "significant"
            return "synergy"
        elif self.interaction < -0.5:
            if self.derivative > 0.001:
                return "recovering"  # Metamorphosis
            else:
                return "interference"  # Parasitism candidate
        return "independent"


class CounterfactualEngine:
    """Engine for computing counterfactual attribution.

    This is the core computation engine. It doesn't store state —
    that's the collector's job.
    """

    def __init__(self, config: CounterfactualConfig | None = None, emit_telemetry: bool = False):
        self.config = config or CounterfactualConfig()
        self.emit_telemetry = emit_telemetry

    def compute_matrix(
        self,
        slot_ids: list[str],
        evaluate_fn: Callable[[dict[str, float]], tuple[float, float]],
    ) -> CounterfactualMatrix:
        """Compute counterfactual matrix for given slots.

        Args:
            slot_ids: List of slot identifiers (e.g., ["r0c0", "r0c1", "r0c2"])
            evaluate_fn: Function that takes alpha settings and returns (val_loss, val_accuracy)
                        e.g., evaluate_fn({"r0c0": 0.0, "r0c1": 1.0}) -> (loss, acc)

        Returns:
            CounterfactualMatrix with all evaluated configurations.
        """
        n_seeds = len(slot_ids)
        strategy = self.config.effective_strategy(n_seeds)
        start_time = time.monotonic()

        matrix = CounterfactualMatrix(strategy_used=strategy)

        if strategy == "full_factorial":
            configs = self._generate_full_factorial(slot_ids)
        elif strategy == "shapley":
            configs = self._generate_shapley_configs(slot_ids)
        elif strategy == "ablation_only":
            configs = self._generate_ablation_configs(slot_ids)
        else:
            configs = self._generate_full_factorial(slot_ids)

        # Apply max_configurations limit
        if self.config.max_configurations:
            configs = configs[: self.config.max_configurations]

        # Evaluate each configuration
        for config_tuple in configs:
            # Check timeout BETWEEN configs (CUDA-safe pattern from design doc)
            if self.config.timeout_seconds:
                elapsed = time.monotonic() - start_time
                if elapsed > self.config.timeout_seconds:
                    break

            # Build alpha settings
            alpha_settings = {
                slot_id: 1.0 if enabled else 0.0
                for slot_id, enabled in zip(slot_ids, config_tuple)
            }

            # Evaluate
            try:
                val_loss, val_accuracy = evaluate_fn(alpha_settings)
            except Exception:
                continue  # Skip failed evaluations

            result = CounterfactualResult(
                config=config_tuple,
                slot_ids=tuple(slot_ids),
                alpha_settings=alpha_settings,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )
            matrix.configs.append(result)

        matrix.compute_time_seconds = time.monotonic() - start_time
        return matrix

    def _generate_full_factorial(
        self, slot_ids: list[str]
    ) -> list[tuple[bool, ...]]:
        """Generate all 2^n configurations."""
        n = len(slot_ids)
        configs = []
        for i in range(2**n):
            config = tuple(bool((i >> j) & 1) for j in range(n))
            configs.append(config)
        return configs

    def _generate_ablation_configs(
        self, slot_ids: list[str]
    ) -> list[tuple[bool, ...]]:
        """Generate O(n) ablation configs: all-on and each slot off."""
        n = len(slot_ids)
        configs = []

        # All on
        configs.append(tuple(True for _ in range(n)))

        # Each slot off (one at a time)
        for i in range(n):
            config = tuple(j != i for j in range(n))
            configs.append(config)

        # All off (baseline)
        configs.append(tuple(False for _ in range(n)))

        return configs

    def _generate_shapley_configs(
        self, slot_ids: list[str]
    ) -> list[tuple[bool, ...]]:
        """Generate configs for Shapley sampling with antithetic pairing."""
        n = len(slot_ids)
        n_samples = self.config.shapley_samples

        # Start with ablation configs (always useful)
        configs = self._generate_ablation_configs(slot_ids)
        seen = set(configs)

        # Add random permutation-based samples
        for _ in range(n_samples // 2):  # Antithetic: each perm generates 2
            perm = list(range(n))
            random.shuffle(perm)

            # Forward: add slots one by one
            for k in range(n + 1):
                config = tuple(i in perm[:k] for i in range(n))
                if config not in seen:
                    configs.append(config)
                    seen.add(config)

            # Antithetic: reverse permutation
            perm_rev = perm[::-1]
            for k in range(n + 1):
                config = tuple(i in perm_rev[:k] for i in range(n))
                if config not in seen:
                    configs.append(config)
                    seen.add(config)

        return configs

    def compute_shapley_values(
        self, matrix: CounterfactualMatrix
    ) -> dict[str, ShapleyEstimate]:
        """Compute Shapley values from a counterfactual matrix.

        Uses permutation sampling with the evaluated configurations.
        """
        if not matrix.configs:
            return {}

        slot_ids = list(matrix.configs[0].slot_ids)
        n = len(slot_ids)

        # Build lookup: config tuple -> accuracy
        lookup: dict[tuple[bool, ...], float] = {
            r.config: r.val_accuracy for r in matrix.configs
        }

        # Compute Shapley values
        shapley_values: dict[str, list[float]] = {sid: [] for sid in slot_ids}

        # Sample permutations
        n_perms = min(100, len(list(permutations(range(n)))))
        perms = list(permutations(range(n)))
        random.shuffle(perms)
        perms = perms[:n_perms]

        for perm in perms:
            for i, slot_idx in enumerate(perm):
                # Coalition before adding this slot
                before_set = set(perm[:i])
                after_set = set(perm[: i + 1])

                before_config = tuple(j in before_set for j in range(n))
                after_config = tuple(j in after_set for j in range(n))

                # Look up values (skip if not in matrix)
                if before_config in lookup and after_config in lookup:
                    marginal = lookup[after_config] - lookup[before_config]
                    shapley_values[slot_ids[slot_idx]].append(marginal)

        # Compute estimates with statistics
        result = {}
        for slot_id in slot_ids:
            values = shapley_values[slot_id]
            if values:
                mean = sum(values) / len(values)
                variance = (
                    sum((v - mean) ** 2 for v in values) / len(values)
                    if len(values) > 1
                    else 0.0
                )
                std = variance**0.5
                result[slot_id] = ShapleyEstimate(
                    mean=mean,
                    std=std,
                    n_samples=len(values),
                    algorithm="permutation_antithetic",
                )
            else:
                result[slot_id] = ShapleyEstimate()

        # Emit telemetry if enabled
        if self.emit_telemetry:
            hub = get_hub()
            if hub is not None:
                # Convert ShapleyEstimate objects to serializable dict
                shapley_dict = {
                    slot_id: {
                        "mean": estimate.mean,
                        "std": estimate.std,
                        "n_samples": estimate.n_samples,
                    }
                    for slot_id, estimate in result.items()
                }
                hub.emit(TelemetryEvent(
                    event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
                    data={
                        "kind": "shapley_computed",
                        "shapley_values": shapley_dict,
                        "num_slots": len(result),
                        "epoch": matrix.epoch,
                    }
                ))

        return result

    def compute_interaction_terms(
        self, matrix: CounterfactualMatrix
    ) -> dict[tuple[str, str], InteractionTerm]:
        """Compute pairwise interaction terms from counterfactual matrix.

        I_ij = f({i,j}) - f({i}) - f({j}) + f(empty)

        Only valid for n <= 3 active seeds (O(2^n) compute).
        """
        if not matrix.configs:
            return {}

        slot_ids = list(matrix.configs[0].slot_ids)
        n = len(slot_ids)

        if n > 3:
            return {}  # Too expensive for more than 3 seeds

        # Build lookup
        lookup: dict[tuple[bool, ...], float] = {
            r.config: r.val_accuracy for r in matrix.configs
        }

        # f(empty)
        empty = tuple(False for _ in range(n))
        f_empty = lookup.get(empty, 0.0)

        interactions = {}
        for i in range(n):
            for j in range(i + 1, n):
                # f({i})
                config_i = tuple(k == i for k in range(n))
                f_i = lookup.get(config_i, f_empty)

                # f({j})
                config_j = tuple(k == j for k in range(n))
                f_j = lookup.get(config_j, f_empty)

                # f({i,j})
                config_ij = tuple(k == i or k == j for k in range(n))
                f_ij = lookup.get(config_ij, f_empty)

                # Interaction term
                interaction = f_ij - f_i - f_j + f_empty

                pair = (slot_ids[i], slot_ids[j])
                interactions[pair] = InteractionTerm(
                    slot_pair=pair,
                    interaction=interaction,
                )

        return interactions
