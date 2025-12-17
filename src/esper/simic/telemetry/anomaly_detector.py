"""Anomaly Detection for PPO Training.

Detects training anomalies that should trigger escalation to DEBUG telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AnomalyReport:
    """Report of detected anomalies."""

    has_anomaly: bool = False
    anomaly_types: list[str] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)

    def add_anomaly(self, anomaly_type: str, detail: str | None = None) -> None:
        """Add an anomaly to the report."""
        self.has_anomaly = True
        self.anomaly_types.append(anomaly_type)
        if detail:
            self.details[anomaly_type] = detail


@dataclass(slots=True)
class AnomalyDetector:
    """Detects training anomalies for telemetry escalation.

    Thresholds are configurable but have sensible defaults based on
    typical PPO training behavior.

    Value collapse detection uses phase-dependent thresholds because:
    - Early training: Critic hasn't learned yet, low EV is expected
    - Late training: Critic should correlate with returns, low EV is concerning

    Phase boundaries are proportions of total_episodes (configurable):
    - Warmup (0-10%): Allow anti-correlated predictions (EV < -0.5)
    - Early (10-25%): Expect some signal (EV < -0.2)
    - Mid (25-75%): Expect positive correlation (EV < 0.0)
    - Late (75%+): Expect useful predictions (EV < 0.1)
    """

    # Ratio thresholds
    max_ratio_threshold: float = 5.0
    min_ratio_threshold: float = 0.1

    # Value function thresholds by training phase (proportions of total_episodes)
    # Phase boundaries as fractions of total training
    warmup_fraction: float = 0.10   # 0-10% of training
    early_fraction: float = 0.25    # 10-25% of training
    mid_fraction: float = 0.75      # 25-75% of training
    # Late is implicitly 75%+

    # EV thresholds for each phase
    ev_threshold_warmup: float = -0.5   # Allow anti-correlated (random init)
    ev_threshold_early: float = -0.2    # Expect some learning
    ev_threshold_mid: float = 0.0       # Expect positive correlation
    ev_threshold_late: float = 0.1      # Expect useful predictions

    def get_ev_threshold(self, current_episode: int, total_episodes: int) -> float:
        """Get explained variance threshold based on training progress.

        Args:
            current_episode: Current episode number (1-indexed)
            total_episodes: Total episodes configured for training

        Returns:
            Appropriate EV threshold for current training phase
        """
        if total_episodes <= 0:
            raise ValueError("total_episodes must be > 0 for phase-dependent EV thresholds")

        progress = current_episode / total_episodes

        if progress < self.warmup_fraction:
            return self.ev_threshold_warmup
        elif progress < self.early_fraction:
            return self.ev_threshold_early
        elif progress < self.mid_fraction:
            return self.ev_threshold_mid
        else:
            return self.ev_threshold_late

    def check_ratios(
        self,
        ratio_max: float,
        ratio_min: float,
    ) -> AnomalyReport:
        """Check for ratio explosion or collapse.

        Args:
            ratio_max: Maximum ratio in batch
            ratio_min: Minimum ratio in batch

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if ratio_max > self.max_ratio_threshold:
            report.add_anomaly(
                "ratio_explosion",
                f"ratio_max={ratio_max:.3f} > {self.max_ratio_threshold}",
            )

        if ratio_min < self.min_ratio_threshold:
            report.add_anomaly(
                "ratio_collapse",
                f"ratio_min={ratio_min:.3f} < {self.min_ratio_threshold}",
            )

        return report

    def check_value_function(
        self,
        explained_variance: float,
        current_episode: int = 0,
        total_episodes: int = 0,
    ) -> AnomalyReport:
        """Check for value function collapse with phase-dependent thresholds.

        Early in training, low explained variance is expected (critic learning).
        Later in training, low EV indicates the critic isn't providing useful
        signal for policy updates.

        Args:
            explained_variance: Explained variance metric (1 - Var(returns-values)/Var(returns))
            current_episode: Current episode number for phase detection (0 = use static threshold)
            total_episodes: Total configured episodes (0 = use static threshold)

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if current_episode <= 0 or total_episodes <= 0:
            raise ValueError("current_episode and total_episodes are required (> 0)")

        threshold = self.get_ev_threshold(current_episode, total_episodes)

        if explained_variance < threshold:
            progress_pct = current_episode / total_episodes * 100
            progress_str = f"{progress_pct:.0f}%"
            report.add_anomaly(
                "value_collapse",
                f"explained_variance={explained_variance:.3f} < {threshold} (at {progress_str} training)",
            )

        return report

    def check_numerical_stability(
        self,
        has_nan: bool,
        has_inf: bool,
    ) -> AnomalyReport:
        """Check for numerical instability.

        Args:
            has_nan: Whether NaN values were detected
            has_inf: Whether Inf values were detected

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if has_nan or has_inf:
            issues = []
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")
            report.add_anomaly(
                "numerical_instability",
                f"Detected: {', '.join(issues)}",
            )

        return report

    def check_all(
        self,
        ratio_max: float,
        ratio_min: float,
        explained_variance: float,
        has_nan: bool = False,
        has_inf: bool = False,
        current_episode: int = 0,
        total_episodes: int = 0,
    ) -> AnomalyReport:
        """Run all anomaly checks and combine results.

        Args:
            ratio_max: Maximum ratio in batch
            ratio_min: Minimum ratio in batch
            explained_variance: Explained variance metric
            has_nan: Whether NaN values were detected
            has_inf: Whether Inf values were detected
            current_episode: Current episode for phase-dependent value collapse threshold
            total_episodes: Total configured episodes for phase detection

        Returns:
            Combined AnomalyReport
        """
        combined = AnomalyReport()

        for check_report in [
            self.check_ratios(ratio_max, ratio_min),
            self.check_value_function(explained_variance, current_episode, total_episodes),
            self.check_numerical_stability(has_nan, has_inf),
        ]:
            if check_report.has_anomaly:
                combined.has_anomaly = True
                combined.anomaly_types.extend(check_report.anomaly_types)
                combined.details.update(check_report.details)

        return combined


__all__ = ["AnomalyDetector", "AnomalyReport"]
