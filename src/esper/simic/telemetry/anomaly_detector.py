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
    ENTROPY_COLLAPSE_PER_HEAD,
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

    Value collapse detection is ROBUST-ANCHORED ONLY: it fires solely on the
    scale-anchored robust signals (value_loss PRIMARY, bellman_error SECONDARY
    safety-net). explained_variance is artifact-prone under the op-marginal V(s)
    critic and is a PURE DIAGNOSTIC, never a gate trigger. The phase-dependent EV
    thresholds (get_ev_threshold, ev_threshold_*) remain available for diagnostic
    display but DO NOT participate in the value_collapse firing decision.
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

    # Robust value-collapse thresholds (EV-telemetry-robustness, GATE = ROBUST-ANCHORED
    # ONLY). The value-collapse gate fires SOLELY on the scale-anchored robust signals
    # below; explained_variance is artifact-prone under the op-marginal V(s) critic (EV
    # blows up on low-return-variance batches as a pure denominator artifact) and is a
    # PURE DIAGNOSTIC carried in telemetry, NEVER a gate trigger.
    #
    # Constants locked against 400 PPO_UPDATE_COMPLETED events from two P0-1 A/B runs
    # (K=4 n=200, K=1 n=200), drl-expert sign-off 2026-06-18.
    #
    # value_loss = 0.5 * mean((value - normalized_return)**2) on the value-normalizer
    # (unit-variance) target scale (ppo_update.py:354-360). PRIMARY trigger. Per-update
    # normalized-target variance is median ~0.38 (NOT 1.0), so the clean
    # "value_loss>=0.5 == zero skill" identity does NOT hold per update: 0.5 fires on
    # 25.5% of all updates and 11% of the strict-healthy band. STRICT-healthy (EV>0.1 AND
    # v_return_corr>0.3) value_loss has p99=1.01, max=3.0. value_loss_threshold=5.0 gives
    # 0 strict-healthy false-fires, 1.67x margin over the healthy max, and fires on the
    # 18 worst observed updates (all EV<=0.001 genuine zero-skill).
    value_loss_threshold: float = 5.0
    # bellman_error = mean |GAE-TD-residual| (value_metrics.py:71), bounded — max observed
    # 3.5 across BOTH arms, strict-healthy max 3.44. SECONDARY safety-net co-trigger (OR'd
    # with value_loss). bellman_error tracks advantage magnitude, NOT value-fit quality
    # (the 11 momentary fit-failures with high value_loss had LOW bellman_error 0.19-0.88),
    # so value_loss must remain PRIMARY; bellman_error alone would miss real fit failures.
    # bellman_error_threshold=5.0 sits just above the healthy max (3.44) as a pure safety
    # net for a true blow-up; 0% fire in these healthy runs.
    bellman_error_threshold: float = 5.0

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
        *,
        bellman_error: float = 0.0,
        value_loss: float = 0.0,
        value_nrmse: float = 0.0,
        v_return_correlation: float = 1.0,
        ev_low_return_variance: bool = False,
    ) -> AnomalyReport:
        """Check for value function collapse — ROBUST-ANCHORED gate (locked owner decision).

        GATE = ROBUST-ANCHORED ONLY. Under the op-marginal V(s) critic the
        explained_variance summary statistic blows up (EV << 0) on low-return-variance
        batches as a pure DENOMINATOR artifact, not a value-fit regression. Firing
        value_collapse on EV at all false-alarms on every K>1 run and invalidates the
        reward-efficiency verdict, so the EV arm is REMOVED entirely:

        - value_collapse fires SOLELY on the scale-anchored robust signal:
          value_loss (PRIMARY) OR bellman_error (SECONDARY safety-net), both over their
          calibrated thresholds. This is the only firing path.
        - explained_variance, value_nrmse, ev_low_return_variance, and
          v_return_correlation are PURE DIAGNOSTICS — carried in the detail string for
          observability, NEVER a gate trigger. An update whose ONLY symptom is
          artefactual low EV (any ev_low_return_variance state, healthy robust signals)
          does NOT fire.

        value_loss is the PRIMARY trigger (value-fit quality on the normalized scale);
        bellman_error is a SECONDARY safety-net co-trigger (OR'd) for a raw-scale
        blow-up. bellman_error tracks advantage magnitude, not value-fit quality, so it
        alone would miss real fit failures — value_loss must remain primary.

        Defaults (load-bearing, conservative): bellman_error/value_loss=0.0 keep an
        un-threaded caller's gate quiescent (never false-fires).

        Args:
            explained_variance: DIAGNOSTIC ONLY — explained variance (carried in detail).
            current_episode: Current episode number for the progress string (required, > 0).
            total_episodes: Total configured episodes (required, > 0).
            value_loss: PRIMARY trigger — value MSE on the normalized (unit-variance) scale.
            bellman_error: SECONDARY trigger — mean |TD error| (raw-scale safety net).
            value_nrmse: DIAGNOSTIC ONLY — floor-stabilized residual RMSE (carried in detail).
            v_return_correlation: DIAGNOSTIC ONLY (carried in detail).
            ev_low_return_variance: DIAGNOSTIC ONLY — EV denominator-floored flag (carried in detail).

        Returns:
            AnomalyReport with any detected issues.
        """
        report = AnomalyReport()

        if current_episode <= 0 or total_episodes <= 0:
            raise ValueError("current_episode and total_episodes are required (> 0)")

        progress_pct = current_episode / total_episodes * 100
        progress_str = f"{progress_pct:.0f}%"

        # ROBUST-ANCHORED gate: value_loss (PRIMARY) OR bellman_error (SECONDARY safety
        # net). EV and friends are diagnostics only — never part of the firing decision.
        robust_bad = (
            value_loss > self.value_loss_threshold
            or bellman_error > self.bellman_error_threshold
        )
        if robust_bad:
            report.add_anomaly(
                "value_collapse",
                (
                    f"robust signal collapse: value_loss={value_loss:.3f} "
                    f"(>{self.value_loss_threshold}) or bellman_error={bellman_error:.3f} "
                    f"(>{self.bellman_error_threshold}) "
                    f"[diagnostics: ev={explained_variance:.3f}, value_nrmse={value_nrmse:.3f}, "
                    f"v_return_corr={v_return_correlation:.3f}, "
                    f"ev_low_return_variance={ev_low_return_variance}] (at {progress_str} training)"
                ),
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

    # Per-head entropy collapse tracking (for hysteresis)
    # Mutable default requires field(default_factory=...)
    _head_collapse_streak: dict[str, int] = field(default_factory=dict)
    collapse_streak_threshold: int = 3  # Consecutive updates before warning
    recovery_multiplier: float = 1.5    # Entropy must exceed threshold * 1.5 to clear

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

    def check_per_head_entropy_collapse(
        self,
        head_entropies: dict[str, float],
        thresholds: dict[str, float] | None = None,
    ) -> AnomalyReport:
        """Check for per-head entropy collapse with hysteresis.

        Each head has its own threshold based on activation frequency.
        Uses hysteresis to prevent warning spam:
        - Requires N consecutive collapses before warning
        - Requires entropy > threshold * 1.5 to clear the streak

        Args:
            head_entropies: Dict of head_name -> mean entropy (normalized 0-1)
            thresholds: Optional custom thresholds per head (defaults to ENTROPY_COLLAPSE_PER_HEAD)

        Returns:
            AnomalyReport with any detected per-head collapse issues
        """
        report = AnomalyReport()
        thresholds = thresholds or ENTROPY_COLLAPSE_PER_HEAD

        for head, entropy in head_entropies.items():
            threshold = thresholds.get(head, 0.1)  # Default fallback
            current_streak = self._head_collapse_streak.get(head, 0)

            if not math.isfinite(entropy):
                report.add_anomaly(
                    f"entropy_nan_inf_{head}",
                    f"{head} entropy={entropy} is non-finite",
                )
                continue

            if entropy < threshold:
                # Increment streak
                current_streak += 1
                self._head_collapse_streak[head] = current_streak

                # Only report after sustained collapse (N consecutive)
                if current_streak >= self.collapse_streak_threshold:
                    report.add_anomaly(
                        f"entropy_collapse_{head}",
                        f"{head} entropy={entropy:.4f} < {threshold} "
                        f"({current_streak} consecutive updates)",
                    )
            elif entropy > threshold * self.recovery_multiplier:
                # Clear streak only when entropy is well above threshold
                self._head_collapse_streak[head] = 0

        return report

    def check_all(
        self,
        ratio_max: float,
        ratio_min: float,
        explained_variance: float,
        current_episode: int,
        total_episodes: int,
        value_collapse_applicable: bool,
        has_nan: bool = False,
        has_inf: bool = False,
        entropy: float | None = None,
        kl: float | None = None,
        *,
        bellman_error: float = 0.0,
        value_loss: float = 0.0,
        value_nrmse: float = 0.0,
        v_return_correlation: float = 1.0,
        ev_low_return_variance: bool = False,
    ) -> AnomalyReport:
        """Run all anomaly checks and combine results.

        Args:
            ratio_max: Maximum ratio in batch
            ratio_min: Minimum ratio in batch
            explained_variance: Explained variance metric (unflagged-only run-level mean)
            current_episode: Current episode for phase-dependent value collapse threshold (required)
            total_episodes: Total configured episodes for phase detection (required)
            value_collapse_applicable: Whether low EV can be interpreted as critic collapse
            has_nan: Whether NaN values were detected
            has_inf: Whether Inf values were detected
            entropy: Policy entropy (0-1 normalized). If provided, checks for entropy collapse.
            kl: KL divergence between old and new policy. If provided, checks for KL spikes.
            bellman_error: PRIMARY robust value-collapse signal (mean |TD|, B1).
            value_loss: PRIMARY robust value-collapse signal (MSE on trained scale, B1).
            value_nrmse: SECONDARY corroboration (floor-stabilized).
            v_return_correlation: SECONDARY corroboration (trustworthy only when not flagged).
            ev_low_return_variance: True when the EV denominator was floored (artifact flag).

        Returns:
            Combined AnomalyReport
        """
        combined = AnomalyReport()

        checks = [
            self.check_ratios(ratio_max, ratio_min),
            self.check_numerical_stability(has_nan, has_inf),
        ]
        if value_collapse_applicable:
            checks.append(
                self.check_value_function(
                    explained_variance,
                    current_episode,
                    total_episodes,
                    bellman_error=bellman_error,
                    value_loss=value_loss,
                    value_nrmse=value_nrmse,
                    v_return_correlation=v_return_correlation,
                    ev_low_return_variance=ev_low_return_variance,
                )
            )

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
