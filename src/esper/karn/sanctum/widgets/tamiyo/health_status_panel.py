"""HealthStatusPanel - Training health and status indicators.

Displays:
- Advantage stats with skewness/kurtosis
- Gradient norm with sparkline
- Log prob extremes (NaN risk)
- Entropy level + trend
- Policy state
- Value range
- Observation health

Note: Layer health (dead/exploding) shown in ActionHeads Flow footer.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

from .sparkline_utils import render_sparkline
from .trends import trend_arrow_for_history

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState


class HealthStatusPanel(Static):
    """Training health and status panel.

    Extends Static directly for minimal layout overhead.
    """

    SPARKLINE_WIDTH: ClassVar[int] = 10  # Compact sparkline for inline display

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "HEALTH"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Text:
        """Render health and status metrics."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Advantage stats with inline skewness/kurtosis and positive ratio
        adv_status = self._get_advantage_status(tamiyo.advantage_std)
        skew_status = self._get_skewness_status(tamiyo.advantage_skewness)
        kurt_status = self._get_kurtosis_status(tamiyo.advantage_kurtosis)
        adv_pos_status = self._get_adv_positive_status(tamiyo.advantage_positive_ratio)
        worst_status = max(
            [adv_status, skew_status, kurt_status, adv_pos_status],
            key=lambda s: ["ok", "warning", "critical"].index(s),
        )

        result.append("Advantage    ", style="dim")
        # Use .3f for mean to distinguish small values from true zero (e.g., -0.005 vs 0.000)
        result.append(
            f"{tamiyo.advantage_mean:+.3f}±{tamiyo.advantage_std:.2f}",
            style=self._status_style(adv_status),
        )
        result.append(" sk:", style="dim")
        # Show "---" for NaN skewness/kurtosis/positive_ratio (no data yet)
        if math.isnan(tamiyo.advantage_skewness):
            result.append("---", style="dim")
        else:
            result.append(
                f"{tamiyo.advantage_skewness:+.1f}",
                style=self._status_style(skew_status),
            )
        result.append(self._skewness_hint(tamiyo.advantage_skewness), style="dim")
        result.append(" kt:", style="dim")
        if math.isnan(tamiyo.advantage_kurtosis):
            result.append("---", style="dim")
        else:
            result.append(
                f"{tamiyo.advantage_kurtosis:+.1f}",
                style=self._status_style(kurt_status),
            )
        # Adv+ percentage (healthy: 40-60%)
        result.append(" +:", style="dim")
        if math.isnan(tamiyo.advantage_positive_ratio):
            result.append("---", style="dim")
        else:
            result.append(
                f"{tamiyo.advantage_positive_ratio:.0%}",
                style=self._status_style(adv_pos_status),
            )
        if worst_status != "ok":
            result.append(" !", style=self._status_style(worst_status))
        result.append("\n")

        # Grad norm with sparkline (rising is bad for gradients)
        # Fixed layout: Label(13) + Value(7) + Indicator(1) + Space(1) + Sparkline(10) + Trend(1)
        # Use space flag for sign alignment: "  0.123" vs " -0.123"
        gn_status = self._get_grad_norm_status(tamiyo.grad_norm)
        result.append("Grad Norm    ", style="dim")  # 13 chars
        result.append(f"{tamiyo.grad_norm: 7.3f}", style=self._status_style(gn_status))
        if gn_status != "ok":
            result.append("!", style=self._status_style(gn_status))
        else:
            result.append(" ", style="dim")
        result.append(" ")
        # Sparkline for gradient history
        if tamiyo.grad_norm_history:
            sparkline = render_sparkline(
                tamiyo.grad_norm_history,
                width=self.SPARKLINE_WIDTH,
                style=self._status_style(gn_status),
            )
            result.append(sparkline)
            arrow, arrow_style = trend_arrow_for_history(
                tamiyo.grad_norm_history,
                metric_name="grad_norm",
                metric_type="loss",
            )
            if arrow:
                result.append(arrow, style=arrow_style)
        else:
            result.append("─" * self.SPARKLINE_WIDTH, style="dim")
        result.append("\n")

        # Log prob extremes (NaN predictor)
        lp_status = self._get_log_prob_status(tamiyo.log_prob_min)
        result.append("Log Prob     ", style="dim")
        if math.isnan(tamiyo.log_prob_min):
            result.append("[---,---]", style="dim")
        else:
            result.append(
                f"[{tamiyo.log_prob_min:.1f},{tamiyo.log_prob_max:.1f}]",
                style=self._status_style(lp_status),
            )
            if lp_status == "critical":
                result.append(" NaN RISK", style="red bold")
            elif lp_status == "warning":
                result.append(" !", style="yellow")
        result.append("\n")

        # Entropy level
        ent_status = self._get_entropy_status(tamiyo.entropy)
        result.append("Entropy      ", style="dim")
        result.append(f"{tamiyo.entropy: 7.3f}", style=self._status_style(ent_status))
        if ent_status != "ok":
            result.append(" !", style=self._status_style(ent_status))
        result.append("\n")

        # Entropy trend
        result.append(self._render_entropy_trend())
        result.append("\n")

        # Policy state
        result.append(self._render_policy_state())
        result.append("\n")

        # Value range
        result.append(self._render_value_stats())
        result.append("\n")

        # Observation stats (input health)
        result.append(self._render_observation_stats())
        result.append("\n")

        # Behavioral efficiency (slots/yield/entropy)
        result.append(self._render_efficiency_stats())
        result.append("\n")

        # LSTM hidden state health (B7-DRL-04)
        result.append(self._render_lstm_health())

        return result

    def _render_entropy_trend(self) -> Text:
        """Render entropy trend with velocity and countdown."""
        if self._snapshot is None:
            return Text()

        tamiyo = self._snapshot.tamiyo
        velocity = tamiyo.entropy_velocity
        risk = tamiyo.collapse_risk_score

        result = Text()
        result.append("Entropy D    ", style="dim")

        EPSILON = 1e-6
        if abs(velocity) < 0.005:
            result.append("stable [--]", style="green")
            return result

        # Trend arrows
        if velocity < -0.03:
            arrow = "[vv]"
            arrow_style = "red bold"
        elif velocity < -0.01:
            arrow = "[v]"
            arrow_style = "yellow"
        elif velocity > 0.01:
            arrow = "[^]"
            arrow_style = "green"
        else:
            arrow = "[~]"
            arrow_style = "dim"

        result.append(f"{velocity:+.3f}/r ", style=arrow_style)
        result.append(arrow, style=arrow_style)

        # Countdown (only if declining toward critical)
        if velocity < -EPSILON and tamiyo.entropy > TUIThresholds.ENTROPY_CRITICAL:
            distance = tamiyo.entropy - TUIThresholds.ENTROPY_CRITICAL
            rounds_to_collapse = int(distance / abs(velocity))

            if rounds_to_collapse < 100:
                result.append(f" ~{rounds_to_collapse}r", style="yellow")

            if risk > 0.7:
                result.append(" [ALERT]", style="red bold")

        return result

    def _render_policy_state(self) -> Text:
        """Render policy state based on entropy/clip correlation."""
        if self._snapshot is None:
            return Text()

        tamiyo = self._snapshot.tamiyo
        corr = tamiyo.entropy_clip_correlation
        entropy = tamiyo.entropy
        clip = tamiyo.clip_fraction

        result = Text()
        result.append("Policy       ", style="dim")

        # The dangerous pattern: entropy falling + clip rising + both concerning
        if (
            corr < -0.5
            and entropy < TUIThresholds.ENTROPY_WARNING
            and clip > TUIThresholds.CLIP_WARNING
        ):
            result.append("COLLAPSE RISK", style="red bold")
            result.append(f" (r={corr:.2f})", style="dim")
        elif corr < -0.6 and entropy < TUIThresholds.ENTROPY_WARNING:
            result.append("collapsing", style="yellow")
            result.append(f" (r={corr:.2f})", style="dim")
        elif corr < -0.4 and clip < 0.15:
            result.append("narrowing", style="green")
        elif abs(corr) < 0.3:
            result.append("stable", style="green")
        else:
            result.append("drifting", style="yellow")
            result.append(f" (r={corr:.2f})", style="dim")

        return result

    def _render_value_stats(self) -> Text:
        """Render value function range and stability indicators.

        Designed to catch value collapse (constant), value explosion, and
        unstable coefficient-of-variation regimes.
        """
        if self._snapshot is None:
            return Text()

        tamiyo = self._snapshot.tamiyo
        result = Text()

        v_min = tamiyo.value_min
        v_max = tamiyo.value_max
        v_std = tamiyo.value_std

        status = self._get_value_status(tamiyo)

        result.append("Value Range  ", style="dim")
        value_range = abs(v_max - v_min)
        if value_range < 0.01:
            range_precision = 3
        elif value_range < 0.1:
            range_precision = 2
        else:
            range_precision = 1
        result.append(
            f"[{v_min:.{range_precision}f},{v_max:.{range_precision}f}]",
            style=self._status_style(status),
        )
        result.append(" ", style="dim")
        result.append(f"s={v_std:.2f}", style="dim")

        if status != "ok":
            result.append(" !", style=self._status_style(status))

        return result

    def _render_observation_stats(self) -> Text:
        """Render observation space health indicators.

        Shows input distribution health to catch numerical issues early.
        """
        if self._snapshot is None:
            return Text()

        obs = self._snapshot.observation_stats
        result = Text()

        # Check for NaN/Inf first (critical issue)
        if obs.nan_count > 0 or obs.inf_count > 0:
            result.append("Obs Health   ", style="dim")
            result.append(
                f"NaN:{obs.nan_pct:.2%} Inf:{obs.inf_pct:.2%}", style="red bold"
            )
            result.append(
                f" (n={obs.nan_count} i={obs.inf_count})",
                style="dim",
            )
            return result

        # Check outlier percentage
        outlier_status = self._get_outlier_status(obs.outlier_pct)
        sat_status = self._get_obs_saturation_status(obs.near_clip_pct)
        clip_status = self._get_obs_clip_status(obs.clip_pct)
        drift_status = self._get_drift_status(obs.normalization_drift)

        # Always show all metrics (dim when ok, colored when warning/critical)
        result.append("Obs Health   ", style="dim")
        result.append(
            f"Out:{obs.outlier_pct:.1%}", style=self._status_style(outlier_status)
        )
        result.append(" ", style="dim")
        result.append(
            f"Sat:{obs.near_clip_pct:.1%}", style=self._status_style(sat_status)
        )
        result.append(" ", style="dim")
        result.append(
            f"Drift:{obs.normalization_drift:.2f}",
            style=self._status_style(drift_status),
        )
        if obs.clip_pct > 0.0:
            result.append(" ", style="dim")
            result.append(
                f"Clip:{obs.clip_pct:.1%}", style=self._status_style(clip_status)
            )

        # Group std (now real, not placeholder)
        result.append("\n")
        result.append("Obs σ        ", style="dim")
        result.append(f"H:{obs.host_features_std:.2f}", style="dim")
        result.append(" ", style="dim")
        result.append(f"C:{obs.context_features_std:.2f}", style="dim")
        result.append(" ", style="dim")
        result.append(f"S:{obs.slot_features_std:.2f}", style="dim")

        return result

    def _render_lstm_health(self) -> Text:
        """Render LSTM hidden state health indicators (B7-DRL-04).

        LSTM hidden states can become corrupted during BPTT:
        - Explosion/saturation: RMS > threshold
        - Vanishing: RMS < threshold
        - NaN/Inf: numerical instability
        """
        if self._snapshot is None:
            return Text()

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # If no LSTM data, show placeholder (non-recurrent policy or no data yet)
        if tamiyo.lstm_h_rms is None:
            result.append("LSTM RMS     ", style="dim")
            result.append("---", style="dim")
            return result

        # Check for NaN/Inf first (critical issue)
        if tamiyo.lstm_has_nan or tamiyo.lstm_has_inf:
            result.append("LSTM RMS     ", style="dim")
            issues = []
            if tamiyo.lstm_has_nan:
                issues.append("NaN")
            if tamiyo.lstm_has_inf:
                issues.append("Inf")
            result.append(" ".join(issues), style="red bold")
            return result

        # Check h and c RMS magnitudes (scale-free across batch size / hidden dim)
        h_status = self._get_lstm_rms_status(tamiyo.lstm_h_rms)
        c_status = self._get_lstm_rms_status(tamiyo.lstm_c_rms)
        worst_status = max(
            [h_status, c_status],
            key=lambda s: ["ok", "warning", "critical"].index(s),
        )

        result.append("LSTM RMS     ", style="dim")
        result.append(
            f"h:{tamiyo.lstm_h_rms:.2f}",
            style=self._status_style(h_status),
        )
        result.append(" ", style="dim")
        result.append(
            f"c:{tamiyo.lstm_c_rms:.2f}",
            style=self._status_style(c_status),
        )

        env_rms_max = None
        if tamiyo.lstm_h_env_rms_max is not None and tamiyo.lstm_c_env_rms_max is not None:
            env_rms_max = max(tamiyo.lstm_h_env_rms_max, tamiyo.lstm_c_env_rms_max)
        elif tamiyo.lstm_h_env_rms_max is not None:
            env_rms_max = tamiyo.lstm_h_env_rms_max
        elif tamiyo.lstm_c_env_rms_max is not None:
            env_rms_max = tamiyo.lstm_c_env_rms_max
        if env_rms_max is not None:
            result.append(" ", style="dim")
            result.append(f"env\u2191:{env_rms_max:.2f}", style="dim")

        peak_abs = None
        if tamiyo.lstm_h_max is not None and tamiyo.lstm_c_max is not None:
            peak_abs = max(tamiyo.lstm_h_max, tamiyo.lstm_c_max)
        elif tamiyo.lstm_h_max is not None:
            peak_abs = tamiyo.lstm_h_max
        elif tamiyo.lstm_c_max is not None:
            peak_abs = tamiyo.lstm_c_max
        if peak_abs is not None:
            result.append(" ", style="dim")
            result.append(f"|x|max:{peak_abs:.1f}", style="dim")

        if worst_status != "ok":
            result.append(" !", style=self._status_style(worst_status))

        return result

    def _render_efficiency_stats(self) -> Text:
        """Render slot utilization, yield, and action entropy at a glance."""
        if self._snapshot is None:
            return Text()

        stats = self._snapshot.episode_stats
        result = Text()

        result.append("Efficiency  ", style="dim")
        if stats.total_episodes <= 0:
            result.append("waiting", style="dim")
            return result

        util_status = "ok"
        if (
            stats.slot_utilization < TUIThresholds.SLOT_UTILIZATION_LOW
            or stats.slot_utilization > TUIThresholds.SLOT_UTILIZATION_HIGH
        ):
            util_status = "warning"
        result.append("util:", style="dim")
        result.append(
            f"{stats.slot_utilization:.0%}",
            style=self._status_style(util_status),
        )

        yield_status = "ok"
        if (
            stats.yield_rate < TUIThresholds.YIELD_RATE_LOW
            or stats.yield_rate > TUIThresholds.YIELD_RATE_HIGH
        ):
            yield_status = "warning"
        result.append(" ", style="dim")
        result.append("yield:", style="dim")
        result.append(
            f"{stats.yield_rate:.0%}",
            style=self._status_style(yield_status),
        )

        entropy_status = "ok"
        if (
            stats.action_entropy < TUIThresholds.ACTION_ENTROPY_LOW
            or stats.action_entropy > TUIThresholds.ACTION_ENTROPY_HIGH
        ):
            entropy_status = "warning"
        result.append(" ", style="dim")
        result.append("H:", style="dim")
        result.append(
            f"{stats.action_entropy:.2f}",
            style=self._status_style(entropy_status),
        )

        return result

    def _get_lstm_rms_status(self, rms: float | None) -> str:
        """Check if LSTM hidden state RMS magnitude is healthy."""
        if rms is None:
            return "ok"  # No LSTM - neutral status
        if rms > 10.0:  # Explosion/saturation threshold
            return "critical"
        if rms > 5.0:  # Warning threshold
            return "warning"
        if rms < 1e-6:  # Vanishing threshold
            return "critical"
        if rms < 1e-4:  # Low warning
            return "warning"
        return "ok"

    def _get_outlier_status(self, outlier_pct: float) -> str:
        """Check if outlier percentage is healthy."""
        if outlier_pct > TUIThresholds.OBS_OUTLIER_CRITICAL:
            return "critical"
        if outlier_pct > TUIThresholds.OBS_OUTLIER_WARNING:
            return "warning"
        return "ok"

    def _get_obs_saturation_status(self, near_clip_pct: float) -> str:
        """Check if normalized observations are saturating near the clip bound."""
        if near_clip_pct > TUIThresholds.OBS_SAT_CRITICAL:
            return "critical"
        if near_clip_pct > TUIThresholds.OBS_SAT_WARNING:
            return "warning"
        return "ok"

    def _get_obs_clip_status(self, clip_pct: float) -> str:
        """Check if normalized observations are being clamped."""
        if clip_pct > TUIThresholds.OBS_CLIP_CRITICAL:
            return "critical"
        if clip_pct > TUIThresholds.OBS_CLIP_WARNING:
            return "warning"
        return "ok"

    def _get_drift_status(self, drift: float) -> str:
        """Check if normalization drift is healthy."""
        if drift > TUIThresholds.OBS_DRIFT_CRITICAL:
            return "critical"
        if drift > TUIThresholds.OBS_DRIFT_WARNING:
            return "warning"
        return "ok"

    def _get_value_status(self, tamiyo: "TamiyoState") -> str:
        """Check if value function is healthy using relative thresholds."""
        v_range = tamiyo.value_max - tamiyo.value_min
        v_mean = tamiyo.value_mean
        v_std = tamiyo.value_std
        initial = tamiyo.initial_value_spread

        # Collapse detection: values stuck at constant
        if v_range < 0.1 and v_std < 0.01:
            return "critical"

        # Coefficient of variation check (relative instability)
        if abs(v_mean) > 0.1:
            cov = v_std / abs(v_mean)
            if cov > 3.0:
                return "critical"
            if cov > 2.0:
                return "warning"

        # Relative threshold (if initial spread known)
        if initial is not None and initial > 0.1:
            ratio = v_range / initial
            if ratio > 10:
                return "critical"
            if ratio > 5:
                return "warning"
            return "ok"

        # Absolute fallback (during warmup or if initial unknown)
        if v_range > 1000 or abs(tamiyo.value_max) > 10000:
            return "critical"
        if v_range > 500 or abs(tamiyo.value_max) > 5000:
            return "warning"

        return "ok"

    # Status helpers
    def _get_advantage_status(self, adv_std: float) -> str:
        if adv_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:
            return "critical"
        if adv_std > TUIThresholds.ADVANTAGE_STD_WARNING:
            return "warning"
        if adv_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING:
            return "warning"
        return "ok"

    def _get_skewness_status(self, skewness: float) -> str:
        if math.isnan(skewness):
            return "ok"  # No data yet - neutral status
        if skewness < -1.0 or skewness > 2.0:
            return "critical"
        if skewness < -0.5 or skewness > 1.0:
            return "warning"
        return "ok"

    def _get_kurtosis_status(self, kurtosis: float) -> str:
        if math.isnan(kurtosis):
            return "ok"  # No data yet - neutral status
        if kurtosis < -2.0 or kurtosis > 6.0:
            return "critical"
        if kurtosis < -1.0 or kurtosis > 3.0:
            return "warning"
        return "ok"

    def _get_adv_positive_status(self, ratio: float) -> str:
        """Check if advantage positive ratio is healthy (40-60%)."""
        if math.isnan(ratio):
            return "ok"  # No data yet - neutral status
        if ratio < 0.2 or ratio > 0.8:
            return "critical"  # Severely imbalanced
        if ratio < 0.4 or ratio > 0.6:
            return "warning"  # Moderately imbalanced
        return "ok"

    def _get_log_prob_status(self, log_prob_min: float) -> str:
        """Check if log probs are in safe range (NaN predictor).

        Very negative log probs indicate actions becoming nearly impossible,
        which leads to numerical underflow and eventually NaN gradients.
        """
        if math.isnan(log_prob_min):
            return "ok"  # No data yet - neutral status
        if log_prob_min < -100:
            return "critical"  # Numerical underflow imminent
        if log_prob_min < -50:
            return "warning"  # Action nearly impossible
        return "ok"

    def _skewness_hint(self, skewness: float) -> str:
        if math.isnan(skewness):
            return ""  # No hint for missing data
        if abs(skewness) < 0.3:
            return "~"
        elif skewness > 1.0:
            return ">>"
        elif skewness > 0.3:
            return ">"
        elif skewness < -1.0:
            return "<<"
        else:
            return "<"

    def _get_ratio_status(self, ratio_min: float, ratio_max: float) -> str:
        if (
            ratio_max > TUIThresholds.RATIO_MAX_CRITICAL
            or ratio_min < TUIThresholds.RATIO_MIN_CRITICAL
        ):
            return "critical"
        if (
            ratio_max > TUIThresholds.RATIO_MAX_WARNING
            or ratio_min < TUIThresholds.RATIO_MIN_WARNING
        ):
            return "warning"
        return "ok"

    def _get_grad_norm_status(self, grad_norm: float) -> str:
        if grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            return "critical"
        if grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            return "warning"
        return "ok"

    def _get_entropy_status(self, entropy: float) -> str:
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "critical"
        if entropy < TUIThresholds.ENTROPY_WARNING:
            return "warning"
        return "ok"

    def _status_style(self, status: str) -> str:
        # Use cyan for ok (visible but not loud), yellow/red for problems
        return {"ok": "cyan", "warning": "yellow", "critical": "red bold"}[status]
