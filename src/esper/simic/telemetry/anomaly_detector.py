"""Anomaly Detection for PPO Training.

Detects training anomalies that should trigger escalation to DEBUG telemetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from esper.leyline import (
    DEFAULT_ENTROPY_COLLAPSE_THRESHOLD,
    DEFAULT_ENTROPY_WARNING_THRESHOLD,
    DEFAULT_RATIO_EXPLOSION_THRESHOLD,
    DEFAULT_RATIO_COLLAPSE_THRESHOLD,
)


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

    # M21: Ratio thresholds from leyline (single source of truth)
    max_ratio_threshold: float = DEFAULT_RATIO_EXPLOSION_THRESHOLD
    min_ratio_threshold: float = DEFAULT_RATIO_COLLAPSE_THRESHOLD

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

        # C3 FIX: Check for NaN/Inf BEFORE comparison.
        # IEEE 754 NaN comparisons always return False, so NaN values would
        # silently pass through threshold checks without this guard.
        if not math.isfinite(ratio_max) or not math.isfinite(ratio_min):
            issues = []
            if not math.isfinite(ratio_max):
                issues.append(f"ratio_max={ratio_max}")
            if not math.isfinite(ratio_min):
                issues.append(f"ratio_min={ratio_min}")
            report.add_anomaly(
                "ratio_nan_inf",
                f"Non-finite ratio values: {', '.join(issues)}",
            )
            return report  # Skip threshold checks for non-finite values

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
        current_episode: int,
        total_episodes: int,
    ) -> AnomalyReport:
        """Check for value function collapse with phase-dependent thresholds.

        Early in training, low explained variance is expected (critic learning).
        Later in training, low EV indicates the critic isn't providing useful
        signal for policy updates.

        Args:
            explained_variance: Explained variance metric (1 - Var(returns-values)/Var(returns))
            current_episode: Current episode number for phase detection (required, must be > 0)
            total_episodes: Total configured episodes (required, must be > 0)

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

    # M11: Entropy collapse thresholds from leyline (single source of truth)
    entropy_collapse_threshold: float = DEFAULT_ENTROPY_COLLAPSE_THRESHOLD
    entropy_warning_threshold: float = DEFAULT_ENTROPY_WARNING_THRESHOLD

    # KL divergence thresholds
    kl_spike_threshold: float = 0.1  # KL above this indicates large policy update
    kl_extreme_threshold: float = 0.5  # KL above this indicates instability

    def check_entropy_collapse(
        self,
        entropy: float,
        head_name: str = "policy",
    ) -> AnomalyReport:
        """Check for entropy collapse (policy becoming deterministic).

        Entropy collapse is one of the most common PPO failure modes:
        - Policy converges to deterministic action selection
        - Exploration stops, learning stalls
        - Often caused by insufficient entropy coefficient or reward hacking

        Args:
            entropy: Current policy entropy (normalized, 0-1 range)
            head_name: Name of the action head (for multi-head policies)

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if entropy < self.entropy_collapse_threshold:
            report.add_anomaly(
                "entropy_collapse",
                f"{head_name} entropy={entropy:.4f} < {self.entropy_collapse_threshold} (collapsed)",
            )
        elif entropy < self.entropy_warning_threshold:
            report.add_anomaly(
                "entropy_low",
                f"{head_name} entropy={entropy:.4f} < {self.entropy_warning_threshold} (warning)",
            )

        return report

    def check_kl_divergence(
        self,
        kl: float,
    ) -> AnomalyReport:
        """Check for KL divergence anomalies.

        KL spikes are leading indicators of training instability that often
        precede ratio explosion. High KL means the policy is changing too
        quickly, violating PPO's trust region.

        Args:
            kl: KL divergence between old and new policy

        Returns:
            AnomalyReport with any detected issues
        """
        report = AnomalyReport()

        if kl > self.kl_extreme_threshold:
            report.add_anomaly(
                "kl_extreme",
                f"kl={kl:.4f} > {self.kl_extreme_threshold} (trust region violated)",
            )
        elif kl > self.kl_spike_threshold:
            report.add_anomaly(
                "kl_spike",
                f"kl={kl:.4f} > {self.kl_spike_threshold} (large policy update)",
            )

        return report

    def check_gradient_drift(
        self,
        norm_drift: float,
        health_drift: float,
        drift_threshold: float = 0.5,
    ) -> AnomalyReport:
        """Check for gradient drift anomaly (P4-9).

        Gradient drift indicates training is slowly diverging from stable behavior.
        Single-step checks miss this; EMA tracking catches gradual degradation.

        Args:
            norm_drift: Gradient norm drift indicator (|current - ema| / ema)
            health_drift: Gradient health drift indicator
            drift_threshold: Threshold for drift warning (0.5 = 50% deviation from EMA)

        Returns:
            AnomalyReport with any detected drift
        """
        report = AnomalyReport()

        if norm_drift > drift_threshold:
            report.add_anomaly(
                "gradient_norm_drift",
                f"norm_drift={norm_drift:.3f} > {drift_threshold}",
            )

        if health_drift > drift_threshold:
            report.add_anomaly(
                "gradient_health_drift",
                f"health_drift={health_drift:.3f} > {drift_threshold}",
            )

        return report

    # B7-DRL-04: LSTM hidden state health thresholds
    # These are RMS thresholds (scale-free w.r.t. batch size / hidden dim).
    # They intentionally do NOT use total L2 norms, which scale with sqrt(numel).
    lstm_max_rms: float = 10.0  # Explosion/saturation threshold (per-element magnitude)
    lstm_min_rms: float = 1e-6  # Vanishing threshold

    def check_lstm_health(
        self,
        h_rms: float,
        c_rms: float,
        h_env_rms_max: float,
        c_env_rms_max: float,
        has_nan: bool,
        has_inf: bool,
    ) -> AnomalyReport:
        """Check LSTM hidden state health (B7-DRL-04).

        LSTM hidden states can drift during training due to:
        - Explosion/saturation: RMS magnitude spikes above threshold
        - Vanishing: RMS magnitude collapses below threshold
        - Numerical instability: NaN/Inf propagates through hidden state

        This should be called after PPO updates, when gradient-induced
        corruption is most likely to occur (BPTT through LSTM layers).

        Args:
            h_rms: RMS magnitude of hidden state tensor (scale-free)
            c_rms: RMS magnitude of cell state tensor (scale-free)
            h_env_rms_max: Max per-env RMS across batch (outlier detection)
            c_env_rms_max: Max per-env RMS across batch (outlier detection)
            has_nan: Whether NaN was detected in hidden state
            has_inf: Whether Inf was detected in hidden state

        Returns:
            AnomalyReport with any detected LSTM health issues
        """
        report = AnomalyReport()

        # Critical: NaN/Inf in hidden state is irrecoverable
        if has_nan:
            report.add_anomaly(
                "lstm_nan",
                f"NaN in LSTM hidden state (h_rms={h_rms:.3f}, c_rms={c_rms:.3f})",
            )
        if has_inf:
            report.add_anomaly(
                "lstm_inf",
                f"Inf in LSTM hidden state (h_rms={h_rms:.3f}, c_rms={c_rms:.3f})",
            )

        # Critical: Hidden state explosion/saturation (use max per-env RMS to catch outliers)
        if h_env_rms_max > self.lstm_max_rms:
            report.add_anomaly(
                "lstm_h_explosion",
                f"h_env_rms_max={h_env_rms_max:.3f} > {self.lstm_max_rms} (h_rms={h_rms:.3f})",
            )
        if c_env_rms_max > self.lstm_max_rms:
            report.add_anomaly(
                "lstm_c_explosion",
                f"c_env_rms_max={c_env_rms_max:.3f} > {self.lstm_max_rms} (c_rms={c_rms:.3f})",
            )

        # Critical: Hidden state vanishing (global RMS)
        if h_rms < self.lstm_min_rms:
            report.add_anomaly(
                "lstm_h_vanishing",
                f"h_rms={h_rms:.6f} < {self.lstm_min_rms}",
            )
        if c_rms < self.lstm_min_rms:
            report.add_anomaly(
                "lstm_c_vanishing",
                f"c_rms={c_rms:.6f} < {self.lstm_min_rms}",
            )

        return report

    def check_all(
        self,
        ratio_max: float,
        ratio_min: float,
        explained_variance: float,
        current_episode: int,
        total_episodes: int,
        has_nan: bool = False,
        has_inf: bool = False,
        entropy: float | None = None,
        kl: float | None = None,
    ) -> AnomalyReport:
        """Run all anomaly checks and combine results.

        Args:
            ratio_max: Maximum ratio in batch
            ratio_min: Minimum ratio in batch
            explained_variance: Explained variance metric
            current_episode: Current episode for phase-dependent value collapse threshold (required)
            total_episodes: Total configured episodes for phase detection (required)
            has_nan: Whether NaN values were detected
            has_inf: Whether Inf values were detected
            entropy: Policy entropy (0-1 normalized). If provided, checks for entropy collapse.
            kl: KL divergence between old and new policy. If provided, checks for KL spikes.

        Returns:
            Combined AnomalyReport
        """
        combined = AnomalyReport()

        checks = [
            self.check_ratios(ratio_max, ratio_min),
            self.check_value_function(explained_variance, current_episode, total_episodes),
            self.check_numerical_stability(has_nan, has_inf),
        ]

        # Add entropy check if provided (H1: entropy collapse detection)
        if entropy is not None:
            checks.append(self.check_entropy_collapse(entropy))

        # Add KL check if provided (H2: KL divergence detection)
        if kl is not None:
            checks.append(self.check_kl_divergence(kl))

        for check_report in checks:
            if check_report.has_anomaly:
                combined.has_anomaly = True
                combined.anomaly_types.extend(check_report.anomaly_types)
                combined.details.update(check_report.details)

        return combined


__all__ = ["AnomalyDetector", "AnomalyReport"]
