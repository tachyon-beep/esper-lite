"""HealthStatusPanel - Training health and status indicators.

Displays:
- Advantage stats with skewness/kurtosis
- Ratio bounds
- Gradient norm
- Layer health
- Entropy trend
- Policy state
- Value range
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds
from esper.leyline import DEFAULT_HOST_LSTM_LAYERS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState


class HealthStatusPanel(Static):
    """Training health and status panel.

    Extends Static directly for minimal layout overhead.
    """

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
        result.append(f"{tamiyo.advantage_mean:+.2f}±{tamiyo.advantage_std:.2f}", style=self._status_style(adv_status))
        result.append(" sk:", style="dim")
        # Show "---" for NaN skewness/kurtosis/positive_ratio (no data yet)
        if math.isnan(tamiyo.advantage_skewness):
            result.append("---", style="dim")
        else:
            result.append(f"{tamiyo.advantage_skewness:+.1f}", style=self._status_style(skew_status))
        result.append(self._skewness_hint(tamiyo.advantage_skewness), style="dim")
        result.append(" kt:", style="dim")
        if math.isnan(tamiyo.advantage_kurtosis):
            result.append("---", style="dim")
        else:
            result.append(f"{tamiyo.advantage_kurtosis:+.1f}", style=self._status_style(kurt_status))
        # Adv+ percentage (healthy: 40-60%)
        result.append(" +:", style="dim")
        if math.isnan(tamiyo.advantage_positive_ratio):
            result.append("---", style="dim")
        else:
            result.append(f"{tamiyo.advantage_positive_ratio:.0%}", style=self._status_style(adv_pos_status))
        if worst_status != "ok":
            result.append(" !", style=self._status_style(worst_status))
        result.append("\n")

        # Ratio bounds (joint ratio for multi-head)
        joint_status = self._get_joint_ratio_status(tamiyo.joint_ratio_max)
        result.append("Ratio Joint  ", style="dim")
        result.append(f"{tamiyo.joint_ratio_max:.2f}", style=self._status_style(joint_status))
        if joint_status != "ok":
            result.append(" !", style=self._status_style(joint_status))
        result.append("\n")

        # Per-head ratio max (condensed - only show if any head is concerning)
        head_ratios = [
            ("Op", tamiyo.head_op_ratio_max),
            ("Sl", tamiyo.head_slot_ratio_max),
            ("BP", tamiyo.head_blueprint_ratio_max),
            ("St", tamiyo.head_style_ratio_max),
            ("Te", tamiyo.head_tempo_ratio_max),
            ("αT", tamiyo.head_alpha_target_ratio_max),
            ("αS", tamiyo.head_alpha_speed_ratio_max),
            ("Cv", tamiyo.head_alpha_curve_ratio_max),
        ]
        worst_head_ratio = max(r for _, r in head_ratios)
        if worst_head_ratio > 1.5:  # Show per-head breakdown if any head is elevated
            result.append("  Per-head:  ", style="dim")
            for i, (label, ratio) in enumerate(head_ratios):
                if i > 0:
                    result.append(" ")
                color = "red" if ratio > 2.0 else ("yellow" if ratio > 1.5 else "dim")
                result.append(f"{label}:{ratio:.1f}", style=color)
            result.append("\n")

        # Grad norm
        gn_status = self._get_grad_norm_status(tamiyo.grad_norm)
        result.append("Grad Norm    ", style="dim")
        result.append(f"{tamiyo.grad_norm:>5.2f}", style=self._status_style(gn_status))
        if gn_status != "ok":
            result.append(" !", style=self._status_style(gn_status))
        result.append("\n")

        # Log prob extremes (NaN predictor)
        lp_status = self._get_log_prob_status(tamiyo.log_prob_min)
        result.append("Log Prob     ", style="dim")
        if math.isnan(tamiyo.log_prob_min):
            result.append("[---,---]", style="dim")
        else:
            result.append(f"[{tamiyo.log_prob_min:.1f},{tamiyo.log_prob_max:.1f}]", style=self._status_style(lp_status))
            if lp_status == "critical":
                result.append(" NaN RISK", style="red bold")
            elif lp_status == "warning":
                result.append(" !", style="yellow")
        result.append("\n")

        # Layer health
        result.append("Layers       ", style="dim")
        healthy = DEFAULT_HOST_LSTM_LAYERS - tamiyo.dead_layers - tamiyo.exploding_layers
        if tamiyo.dead_layers > 0 or tamiyo.exploding_layers > 0:
            result.append(f"{tamiyo.dead_layers}D/{tamiyo.exploding_layers}E", style="red")
        else:
            result.append(f"{healthy}/{DEFAULT_HOST_LSTM_LAYERS} ✓", style="green")
        result.append("\n")

        # Entropy trend
        result.append(self._render_entropy_trend())
        result.append("\n")

        # Policy state
        result.append(self._render_policy_state())
        result.append("\n")

        # Value range
        result.append(self._render_value_stats())

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

        result.append(f"{velocity:+.3f}/b ", style=arrow_style)
        result.append(arrow, style=arrow_style)

        # Countdown (only if declining toward critical)
        if velocity < -EPSILON and tamiyo.entropy > TUIThresholds.ENTROPY_CRITICAL:
            distance = tamiyo.entropy - TUIThresholds.ENTROPY_CRITICAL
            batches_to_collapse = int(distance / abs(velocity))

            if batches_to_collapse < 100:
                result.append(f" ~{batches_to_collapse}b", style="yellow")

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
        if (corr < -0.5 and
            entropy < TUIThresholds.ENTROPY_WARNING and
            clip > TUIThresholds.CLIP_WARNING):
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
        """Render op-conditioned Q-values (Policy V2).

        Shows Q(s,op) for each operation and Q-variance metric.
        Low variance indicates critic is ignoring op conditioning.
        """
        if self._snapshot is None:
            return Text()

        tamiyo = self._snapshot.tamiyo
        result = Text()

        # Q-values per operation (abbreviated for space)
        result.append("Q-Values     ", style="dim")

        # Define ops with colors
        ops = [
            ("G", tamiyo.q_germinate, "green"),
            ("A", tamiyo.q_advance, "cyan"),
            ("F", tamiyo.q_fossilize, "blue"),
            ("P", tamiyo.q_prune, "red"),
            ("V", tamiyo.q_set_alpha, "cyan"),  # V for set alpha (A is advance)
            ("W", tamiyo.q_wait, "dim"),
        ]

        for i, (label, q_val, color) in enumerate(ops):
            if i > 0:
                result.append(" ", style="dim")
            result.append(f"{label}:", style="dim")
            result.append(f"{q_val:+.1f}", style=color)

        result.append("\n")

        # Q-variance (op-sensitivity check)
        result.append("Q Variance   ", style="dim")

        var_status = self._get_q_variance_status(tamiyo.q_variance)
        result.append(f"{tamiyo.q_variance:.2f}", style=self._status_style(var_status))

        if var_status == "critical":
            result.append(" NO OP COND!", style="red bold")
        elif var_status == "warning":
            result.append(" weak", style="yellow")

        # Q-spread for context
        result.append(f"  spread:{tamiyo.q_spread:.1f}", style="dim")

        return result

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
        if ratio_max > TUIThresholds.RATIO_MAX_CRITICAL or ratio_min < TUIThresholds.RATIO_MIN_CRITICAL:
            return "critical"
        if ratio_max > TUIThresholds.RATIO_MAX_WARNING or ratio_min < TUIThresholds.RATIO_MIN_WARNING:
            return "warning"
        return "ok"

    def _get_grad_norm_status(self, grad_norm: float) -> str:
        if grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            return "critical"
        if grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            return "warning"
        return "ok"

    def _get_q_variance_status(self, q_variance: float) -> str:
        """Check if Q-variance indicates op conditioning is working.

        Low variance means Q(s, op) ≈ Q(s, op') for all ops → critic ignoring op input.
        High variance means different ops get different value estimates → healthy.
        """
        if q_variance < 0.01:
            return "critical"  # Essentially collapsed to V(s)
        if q_variance < 0.1:
            return "warning"  # Weak differentiation between ops
        return "ok"

    def _get_joint_ratio_status(self, joint_ratio: float) -> str:
        """Check joint ratio (product of per-head ratios).

        With 8 heads, individual ratios at 1.15 produce joint ratio ~3.06.
        Standard PPO clip range is [0.8, 1.2] → joint should be close to 1.0.
        """
        if joint_ratio > 3.0 or joint_ratio < 0.33:
            return "critical"  # Severe explosion/collapse
        if joint_ratio > 2.0 or joint_ratio < 0.5:
            return "warning"  # Elevated but not critical
        return "ok"

    def _status_style(self, status: str) -> str:
        # Use cyan for ok (visible but not loud), yellow/red for problems
        return {"ok": "cyan", "warning": "yellow", "critical": "red bold"}[status]
