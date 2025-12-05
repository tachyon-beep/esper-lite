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
    details: dict = field(default_factory=dict)

    def add_anomaly(self, anomaly_type: str, detail: str | None = None) -> None:
        """Add an anomaly to the report."""
        self.has_anomaly = True
        self.anomaly_types.append(anomaly_type)
        if detail:
            self.details[anomaly_type] = detail


@dataclass
class AnomalyDetector:
    """Detects training anomalies for telemetry escalation.

    Thresholds are configurable but have sensible defaults based on
    typical PPO training behavior.
    """

    # Ratio thresholds
    max_ratio_threshold: float = 5.0
    min_ratio_threshold: float = 0.1

    # Value function thresholds
    min_explained_variance: float = 0.1

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
    ) -> AnomalyReport:
        """Check for value function collapse.

        Args:
            explained_variance: Explained variance metric

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if explained_variance < self.min_explained_variance:
            report.add_anomaly(
                "value_collapse",
                f"explained_variance={explained_variance:.3f} < {self.min_explained_variance}",
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
    ) -> AnomalyReport:
        """Run all anomaly checks and combine results.

        Args:
            ratio_max: Maximum ratio in batch
            ratio_min: Minimum ratio in batch
            explained_variance: Explained variance metric
            has_nan: Whether NaN values were detected
            has_inf: Whether Inf values were detected

        Returns:
            Combined AnomalyReport
        """
        combined = AnomalyReport()

        for check_report in [
            self.check_ratios(ratio_max, ratio_min),
            self.check_value_function(explained_variance),
            self.check_numerical_stability(has_nan, has_inf),
        ]:
            if check_report.has_anomaly:
                combined.has_anomaly = True
                combined.anomaly_types.extend(check_report.anomaly_types)
                combined.details.update(check_report.details)

        return combined


__all__ = ["AnomalyDetector", "AnomalyReport"]
