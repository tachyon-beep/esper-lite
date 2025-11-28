"""Dataset health checks for offline RL data quality."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    passed: bool
    level: str  # "ok", "warning", "error"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def check_action_coverage(
    action_counts: dict[str, int],
    min_pct: float = 0.02,
    warn_pct: float = 0.05,
) -> HealthCheckResult:
    """Check that all actions have minimum coverage.

    Args:
        action_counts: Dict mapping action names to counts
        min_pct: Minimum percentage for error (blocking)
        warn_pct: Minimum percentage for warning

    Returns:
        HealthCheckResult
    """
    total = sum(action_counts.values())
    if total == 0:
        return HealthCheckResult(
            name="action_coverage",
            passed=False,
            level="error",
            message="No actions in dataset",
        )

    percentages = {a: c / total for a, c in action_counts.items()}
    low_actions = [a for a, p in percentages.items() if p < min_pct]
    warn_actions = [a for a, p in percentages.items() if min_pct <= p < warn_pct]

    if low_actions:
        return HealthCheckResult(
            name="action_coverage",
            passed=False,
            level="error",
            message=f"Actions below {min_pct*100:.0f}% coverage: {low_actions}",
            details={"percentages": percentages, "low_actions": low_actions},
        )
    elif warn_actions:
        return HealthCheckResult(
            name="action_coverage",
            passed=True,
            level="warning",
            message=f"Actions below {warn_pct*100:.0f}% coverage: {warn_actions}",
            details={"percentages": percentages, "warn_actions": warn_actions},
        )
    else:
        return HealthCheckResult(
            name="action_coverage",
            passed=True,
            level="ok",
            message="All actions have adequate coverage",
            details={"percentages": percentages},
        )


def check_action_entropy(
    action_counts: dict[str, int],
    min_entropy: float = 0.3,
    warn_entropy: float = 0.5,
) -> HealthCheckResult:
    """Check action distribution entropy.

    Args:
        action_counts: Dict mapping action names to counts
        min_entropy: Minimum entropy for error (blocking)
        warn_entropy: Minimum entropy for warning

    Returns:
        HealthCheckResult
    """
    total = sum(action_counts.values())
    if total == 0:
        return HealthCheckResult(
            name="action_entropy",
            passed=False,
            level="error",
            message="No actions in dataset",
        )

    probs = [c / total for c in action_counts.values() if c > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(action_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    if entropy < min_entropy:
        return HealthCheckResult(
            name="action_entropy",
            passed=False,
            level="error",
            message=f"Action entropy too low: {entropy:.3f} < {min_entropy}",
            details={"entropy": entropy, "normalized": normalized_entropy},
        )
    elif entropy < warn_entropy:
        return HealthCheckResult(
            name="action_entropy",
            passed=True,
            level="warning",
            message=f"Action entropy below recommended: {entropy:.3f} < {warn_entropy}",
            details={"entropy": entropy, "normalized": normalized_entropy},
        )
    else:
        return HealthCheckResult(
            name="action_entropy",
            passed=True,
            level="ok",
            message=f"Action entropy adequate: {entropy:.3f}",
            details={"entropy": entropy, "normalized": normalized_entropy},
        )


def check_policy_diversity(
    episodes: list[dict],
    min_policies: int = 5,
    warn_policies: int = 8,
) -> HealthCheckResult:
    """Check that multiple behavior policies are represented.

    Args:
        episodes: List of episode dicts with behavior_policy field
        min_policies: Minimum unique policies for error
        warn_policies: Minimum unique policies for warning

    Returns:
        HealthCheckResult
    """
    policy_ids = set()
    for ep in episodes:
        if "behavior_policy" in ep:
            policy_ids.add(ep["behavior_policy"].get("policy_id", "unknown"))

    num_policies = len(policy_ids)

    if num_policies < min_policies:
        return HealthCheckResult(
            name="policy_diversity",
            passed=False,
            level="error",
            message=f"Too few policies: {num_policies} < {min_policies}",
            details={"num_policies": num_policies, "policies": list(policy_ids)},
        )
    elif num_policies < warn_policies:
        return HealthCheckResult(
            name="policy_diversity",
            passed=True,
            level="warning",
            message=f"Policy diversity below recommended: {num_policies} < {warn_policies}",
            details={"num_policies": num_policies, "policies": list(policy_ids)},
        )
    else:
        return HealthCheckResult(
            name="policy_diversity",
            passed=True,
            level="ok",
            message=f"Good policy diversity: {num_policies} policies",
            details={"num_policies": num_policies, "policies": list(policy_ids)},
        )


def check_single_action_clusters(
    episodes: list[dict],
    max_single_action_pct: float = 0.50,
) -> HealthCheckResult:
    """Check percentage of state clusters with only one action.

    This is a simplified version - clusters states by (epoch, plateau_epochs, has_seed).

    Args:
        episodes: List of episode dicts
        max_single_action_pct: Maximum allowed percentage of single-action clusters

    Returns:
        HealthCheckResult
    """
    # Simple state clustering by key features
    state_actions: dict[tuple, set[str]] = {}

    for ep in episodes:
        for decision in ep.get("decisions", []):
            obs = decision.get("observation", {})
            # Simple state key
            state_key = (
                obs.get("epoch", 0) // 5,  # Bucket by 5 epochs
                min(obs.get("plateau_epochs", 0), 5),
                obs.get("has_active_seed", False),
            )
            action = decision.get("action", {}).get("action", "WAIT")

            if state_key not in state_actions:
                state_actions[state_key] = set()
            state_actions[state_key].add(action)

    if not state_actions:
        return HealthCheckResult(
            name="single_action_clusters",
            passed=False,
            level="error",
            message="No state-action data found",
        )

    single_action_clusters = sum(1 for actions in state_actions.values() if len(actions) == 1)
    total_clusters = len(state_actions)
    pct = single_action_clusters / total_clusters

    if pct > max_single_action_pct:
        return HealthCheckResult(
            name="single_action_clusters",
            passed=False,
            level="error",
            message=f"Too many single-action clusters: {pct*100:.1f}% > {max_single_action_pct*100:.0f}%",
            details={
                "single_action_clusters": single_action_clusters,
                "total_clusters": total_clusters,
                "percentage": pct,
            },
        )
    else:
        return HealthCheckResult(
            name="single_action_clusters",
            passed=True,
            level="ok",
            message=f"Single-action clusters: {pct*100:.1f}%",
            details={
                "single_action_clusters": single_action_clusters,
                "total_clusters": total_clusters,
                "percentage": pct,
            },
        )


class DatasetHealthCheck:
    """Run all health checks on a dataset."""

    def __init__(
        self,
        action_coverage_min: float = 0.02,
        action_coverage_warn: float = 0.05,
        entropy_min: float = 0.3,
        entropy_warn: float = 0.5,
        policy_diversity_min: int = 5,
        single_action_max: float = 0.50,
    ):
        self.action_coverage_min = action_coverage_min
        self.action_coverage_warn = action_coverage_warn
        self.entropy_min = entropy_min
        self.entropy_warn = entropy_warn
        self.policy_diversity_min = policy_diversity_min
        self.single_action_max = single_action_max

    def run_all(self, episodes: list[dict]) -> dict[str, HealthCheckResult]:
        """Run all health checks.

        Args:
            episodes: List of episode dicts

        Returns:
            Dict mapping check name to result
        """
        # Count actions across all episodes
        action_counts: Counter = Counter()
        for ep in episodes:
            for decision in ep.get("decisions", []):
                action = decision.get("action", {}).get("action", "WAIT")
                action_counts[action] += 1

        results = {}

        results["action_coverage"] = check_action_coverage(
            dict(action_counts),
            min_pct=self.action_coverage_min,
            warn_pct=self.action_coverage_warn,
        )

        results["action_entropy"] = check_action_entropy(
            dict(action_counts),
            min_entropy=self.entropy_min,
            warn_entropy=self.entropy_warn,
        )

        results["policy_diversity"] = check_policy_diversity(
            episodes,
            min_policies=self.policy_diversity_min,
        )

        results["single_action_clusters"] = check_single_action_clusters(
            episodes,
            max_single_action_pct=self.single_action_max,
        )

        return results

    def has_blocking_errors(self, results: dict[str, HealthCheckResult]) -> bool:
        """Check if any results are blocking errors."""
        return any(r.level == "error" and not r.passed for r in results.values())

    def print_report(self, results: dict[str, HealthCheckResult]) -> None:
        """Print a formatted health check report."""
        print("\n" + "=" * 60)
        print("Dataset Health Check Report")
        print("=" * 60)

        for name, result in results.items():
            status = "✓" if result.passed else "✗"
            level_str = f"[{result.level.upper()}]"
            print(f"{status} {name:25s} {level_str:10s} {result.message}")

        print("=" * 60)

        if self.has_blocking_errors(results):
            print("⚠️  BLOCKING ERRORS FOUND - Dataset may not be suitable for training")
        else:
            print("✓  All critical checks passed")
