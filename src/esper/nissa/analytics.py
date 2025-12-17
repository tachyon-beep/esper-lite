"""Nissa Analytics - Blueprint performance aggregation.

Aggregates telemetry events into strategic dashboards:
- BlueprintStats: Per-blueprint fossilization rates and accuracy metrics
- SeedScoreboard: Per-environment cumulative seed tracking
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa.output import OutputBackend

_logger = logging.getLogger(__name__)


# =============================================================================
# Compute Cost Multipliers
# =============================================================================

BLUEPRINT_COMPUTE_MULTIPLIERS: dict[str, float] = {
    # CNN blueprints
    "noop": 1.0,            # Identity seed - no compute impact
    "norm": 1.02,           # Minimal - normalization only
    "depthwise": 1.08,      # Cheap - depthwise separable
    "conv_light": 1.15,     # Moderate - single conv block
    "conv_heavy": 1.25,     # Heavier - double conv block
    # Transformer blueprints
    "lora": 1.05,           # Low-rank adapter - lightweight
    "attention": 1.35,      # Additional self-attention head (O(n²))
    "flex_attention": 1.35, # FlexAttention variant - similar cost envelope
    "mlp": 1.20,            # Extra MLP block
}


def compute_cost_for_blueprint(blueprint_id: str) -> float:
    """Return compute multiplier for a blueprint type."""
    return BLUEPRINT_COMPUTE_MULTIPLIERS.get(blueprint_id, 1.1)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BlueprintStats:
    """Performance statistics for a single blueprint type."""

    germinated: int = 0
    fossilized: int = 0
    culled: int = 0
    acc_deltas: list[float] = field(default_factory=list)
    churns: list[float] = field(default_factory=list)
    blending_deltas: list[float] = field(default_factory=list)  # Accuracy change during blending
    counterfactuals: list[float] = field(default_factory=list)  # True causal attribution

    @property
    def mean_acc_delta(self) -> float:
        """Mean accuracy improvement at terminal state."""
        return sum(self.acc_deltas) / len(self.acc_deltas) if self.acc_deltas else 0.0

    @property
    def mean_churn(self) -> float:
        """Mean accuracy change on cull (usually negative)."""
        return sum(self.churns) / len(self.churns) if self.churns else 0.0

    @property
    def mean_blending_delta(self) -> float:
        """Mean accuracy change during blending stages."""
        return sum(self.blending_deltas) / len(self.blending_deltas) if self.blending_deltas else 0.0

    @property
    def mean_counterfactual(self) -> float:
        """Mean counterfactual contribution (true causal attribution)."""
        return sum(self.counterfactuals) / len(self.counterfactuals) if self.counterfactuals else 0.0

    @property
    def fossilization_rate(self) -> float:
        """Percentage of terminal seeds that fossilized (not culled)."""
        total = self.fossilized + self.culled
        return (self.fossilized / total * 100) if total > 0 else 0.0


@dataclass
class SeedScoreboard:
    """Cumulative seed tracking for an environment."""

    total_germinated: int = 0
    total_fossilized: int = 0
    total_culled: int = 0
    fossilized_by_blueprint: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    live_blueprint: str | None = None
    params_added: int = 0
    host_params: int = 0
    total_fossilize_age_epochs: int = 0
    total_cull_age_epochs: int = 0

    @property
    def compute_cost(self) -> float:
        """Estimated compute cost relative to baseline (1.0)."""
        cost = 1.0
        for bp_id, count in self.fossilized_by_blueprint.items():
            cost += (compute_cost_for_blueprint(bp_id) - 1.0) * count
        return cost

    @property
    def params_percentage(self) -> float:
        """Params added as percentage of host."""
        return (self.params_added / self.host_params * 100) if self.host_params > 0 else 0.0

    @property
    def avg_fossilize_age_epochs(self) -> float:
        """Average total age (epochs) at fossilization."""
        return (
            self.total_fossilize_age_epochs / self.total_fossilized
            if self.total_fossilized > 0
            else 0.0
        )

    @property
    def avg_cull_age_epochs(self) -> float:
        """Average total age (epochs) at cull."""
        return (
            self.total_cull_age_epochs / self.total_culled
            if self.total_culled > 0
            else 0.0
        )


class BlueprintAnalytics(OutputBackend):
    """Aggregates blueprint performance from telemetry events.

    Implements OutputBackend to receive events from NissaHub.
    Tracks:
    - Per-blueprint stats (germinated, fossilized, culled, accuracy)
    - Per-environment scoreboards (params, compute cost, distribution)

    Args:
        quiet: Suppress console output (use when TUI is active).
    """

    def __init__(self, quiet: bool = False):
        self.stats: dict[str, BlueprintStats] = defaultdict(BlueprintStats)
        self.scoreboards: dict[int, SeedScoreboard] = {}
        self.quiet = quiet

    def set_host_params(self, env_id: int, host_params: int) -> None:
        """Initialize or update host parameter count for an environment."""
        sb = self._get_scoreboard(env_id)
        sb.host_params = host_params

    def emit(self, event: TelemetryEvent) -> None:
        """Process lifecycle events to update stats and print status."""
        if event.event_type == TelemetryEventType.SEED_GERMINATED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            seed_id = event.data.get("seed_id", "unknown")
            params = event.data.get("params", 0)

            self.stats[bp_id].germinated += 1
            sb = self._get_scoreboard(env_id)
            sb.total_germinated += 1
            sb.live_blueprint = bp_id

            if not self.quiet:
                print(f"    [env{env_id}] Germinated '{seed_id}' ({bp_id}, {params/1000:.1f}K params)")

        elif event.event_type == TelemetryEventType.SEED_FOSSILIZED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            seed_id = event.data.get("seed_id", "unknown")
            improvement = event.data.get("improvement", 0.0)
            blending_delta = event.data.get("blending_delta", 0.0)
            counterfactual = event.data.get("counterfactual")  # May be None
            params = event.data.get("params_added", 0)
            epochs_total = event.data.get("epochs_total", 0)

            self.stats[bp_id].fossilized += 1
            self.stats[bp_id].acc_deltas.append(improvement)
            self.stats[bp_id].blending_deltas.append(blending_delta)
            if counterfactual is not None:
                self.stats[bp_id].counterfactuals.append(counterfactual)

            sb = self._get_scoreboard(env_id)
            sb.total_fossilized += 1
            sb.fossilized_by_blueprint[bp_id] += 1
            sb.params_added += params
            sb.total_fossilize_age_epochs += int(epochs_total)
            sb.live_blueprint = None

            # Show total improvement, blending delta, and causal contribution
            if not self.quiet:
                causal_str = f", causal Δ {counterfactual:+.2f}%" if counterfactual is not None else ""
                print(f"    [env{env_id}] Fossilized '{seed_id}' ({bp_id}, "
                      f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%{causal_str})")

        elif event.event_type == TelemetryEventType.SEED_CULLED:
            bp_id = event.data.get("blueprint_id", "unknown")
            env_id = event.data.get("env_id", 0)
            seed_id = event.data.get("seed_id", "unknown")
            improvement = event.data.get("improvement", 0.0)
            blending_delta = event.data.get("blending_delta", 0.0)
            counterfactual = event.data.get("counterfactual")  # May be None
            reason = event.data.get("reason", "")
            epochs_total = event.data.get("epochs_total", 0)

            self.stats[bp_id].culled += 1
            self.stats[bp_id].churns.append(improvement)
            self.stats[bp_id].blending_deltas.append(blending_delta)
            if counterfactual is not None:
                self.stats[bp_id].counterfactuals.append(counterfactual)

            sb = self._get_scoreboard(env_id)
            sb.total_culled += 1
            sb.total_cull_age_epochs += int(epochs_total)
            sb.live_blueprint = None

            # Show total improvement, blending delta, and causal contribution
            if not self.quiet:
                reason_str = f" ({reason})" if reason else ""
                causal_str = f", causal Δ {counterfactual:+.2f}%" if counterfactual is not None else ""
                print(f"    [env{env_id}] Culled '{seed_id}' ({bp_id}, "
                      f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%{causal_str}){reason_str}")

        elif event.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT:
            kind = event.data.get("kind", "unknown")
            env_id = event.data.get("env_id", 0)

            if kind == "shapley_computed":
                shapley_values = event.data.get("shapley_values", {})
                num_slots = event.data.get("num_slots", 0)

                if not self.quiet and shapley_values:
                    # Format Shapley values compactly
                    values_str = ", ".join(
                        f"{slot}: {val.get('mean', 0):+.2f}%"
                        for slot, val in shapley_values.items()
                    )
                    print(f"    [env{env_id}] Shapley values ({num_slots} slots): {values_str}")
            else:
                _logger.debug(f"[env{env_id}] ANALYTICS_SNAPSHOT kind={kind}")

        # === Training Progress Events ===
        elif event.event_type == TelemetryEventType.SEED_STAGE_CHANGED:
            env_id = event.data.get("env_id", 0)
            slot_id = event.data.get("slot_id", "?")
            old_stage = event.data.get("old_stage", "?")
            new_stage = event.data.get("new_stage", "?")
            _logger.debug(f"[env{env_id}] Stage change {slot_id}: {old_stage} → {new_stage}")

        elif event.event_type == TelemetryEventType.SEED_GATE_EVALUATED:
            env_id = event.data.get("env_id", 0)
            slot_id = event.data.get("slot_id", "?")
            gate = event.data.get("gate", "?")
            passed = event.data.get("passed", False)
            _logger.debug(f"[env{env_id}] Gate {gate} for {slot_id}: {'PASSED' if passed else 'BLOCKED'}")

        elif event.event_type == TelemetryEventType.TAMIYO_INITIATED:
            env_id = event.data.get("env_id", 0)
            epoch = event.data.get("epoch", 0)
            _logger.debug(f"[env{env_id}] Tamiyo initiated at epoch {epoch}")

        # === Trend Detection Events ===
        elif event.event_type == TelemetryEventType.PLATEAU_DETECTED:
            env_id = event.data.get("env_id", 0)
            delta = event.data.get("delta", 0)
            if not self.quiet:
                print(f"    [env{env_id}] ⏸ Plateau detected (Δ={delta:+.3f}%)")

        elif event.event_type == TelemetryEventType.DEGRADATION_DETECTED:
            env_id = event.data.get("env_id", 0)
            delta = event.data.get("delta", 0)
            if not self.quiet:
                print(f"    [env{env_id}] 📉 Degradation detected (Δ={delta:+.3f}%)")

        elif event.event_type == TelemetryEventType.IMPROVEMENT_DETECTED:
            env_id = event.data.get("env_id", 0)
            delta = event.data.get("delta", 0)
            if not self.quiet:
                print(f"    [env{env_id}] 📈 Improvement detected (Δ={delta:+.3f}%)")

        # === Health/Warning Events ===
        elif event.event_type == TelemetryEventType.MEMORY_WARNING:
            gpu_util = event.data.get("gpu_utilization", 0)
            gpu_gb = event.data.get("gpu_allocated_gb", 0)
            if not self.quiet:
                print(f"    ⚠️ Memory warning: GPU {gpu_util:.0%} ({gpu_gb:.1f}GB)")

        elif event.event_type == TelemetryEventType.GRADIENT_ANOMALY:
            env_id = event.data.get("env_id", 0)
            anomaly_type = event.data.get("anomaly_type", "unknown")
            if not self.quiet:
                print(f"    [env{env_id}] ⚠️ Gradient anomaly: {anomaly_type}")

        elif event.event_type == TelemetryEventType.PERFORMANCE_DEGRADATION:
            env_id = event.data.get("env_id", 0)
            metric = event.data.get("metric", "unknown")
            if not self.quiet:
                print(f"    [env{env_id}] ⚠️ Performance degradation: {metric}")

        elif event.event_type == TelemetryEventType.REWARD_HACKING_SUSPECTED:
            env_id = event.data.get("env_id", 0)
            reason = event.data.get("reason", "unknown")
            if not self.quiet:
                print(f"    [env{env_id}] 🚨 Reward hacking suspected: {reason}")

        # === PPO Anomaly Events ===
        elif event.event_type == TelemetryEventType.RATIO_EXPLOSION_DETECTED:
            env_id = event.data.get("env_id", 0)
            max_ratio = event.data.get("max_ratio", 0)
            if not self.quiet:
                print(f"    [env{env_id}] 💥 Ratio explosion: max={max_ratio:.2f}")

        elif event.event_type == TelemetryEventType.RATIO_COLLAPSE_DETECTED:
            env_id = event.data.get("env_id", 0)
            min_ratio = event.data.get("min_ratio", 0)
            if not self.quiet:
                print(f"    [env{env_id}] 📉 Ratio collapse: min={min_ratio:.4f}")

        elif event.event_type == TelemetryEventType.VALUE_COLLAPSE_DETECTED:
            env_id = event.data.get("env_id", 0)
            std = event.data.get("value_std", 0)
            if not self.quiet:
                print(f"    [env{env_id}] 📉 Value collapse: std={std:.4f}")

        elif event.event_type == TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED:
            env_id = event.data.get("env_id", 0)
            details = event.data.get("details", "unknown")
            if not self.quiet:
                print(f"    [env{env_id}] ⚠️ Numerical instability: {details}")

        # === Governor Events ===
        elif event.event_type == TelemetryEventType.GOVERNOR_ROLLBACK:
            env_id = event.data.get("env_id", 0)
            reason = event.data.get("reason", "vital signs failure")
            if not self.quiet:
                print(f"    [env{env_id}] 🔄 Governor rollback: {reason}")

        # === Counterfactual Events ===
        elif event.event_type == TelemetryEventType.COUNTERFACTUAL_COMPUTED:
            env_id = event.data.get("env_id", 0)
            slot_id = event.data.get("slot_id", "?")
            contribution = event.data.get("contribution", 0)
            _logger.debug(f"[env{env_id}] Counterfactual {slot_id}: {contribution:+.2f}%")

    def _get_scoreboard(self, env_id: int) -> SeedScoreboard:
        """Get or create scoreboard for environment."""
        if env_id not in self.scoreboards:
            self.scoreboards[env_id] = SeedScoreboard()
        return self.scoreboards[env_id]

    def summary_table(self) -> str:
        """Pretty-print blueprint performance stats."""
        lines = ["Blueprint Stats:"]
        lines.append("  " + "-" * 110)
        lines.append(
            f"  {'Blueprint':<14} {'Germ':>5} {'Foss':>5} {'Cull':>5} "
            f"{'Rate':>6} {'ΔAcc':>8} {'BlendΔ':>8} {'CausalΔ':>8} {'Churn':>8}"
        )
        lines.append("  " + "-" * 110)

        for bp_id in sorted(self.stats.keys()):
            s = self.stats[bp_id]
            lines.append(
                f"  {bp_id:<14} {s.germinated:>5} {s.fossilized:>5} "
                f"{s.culled:>5} {s.fossilization_rate:>5.1f}% "
                f"{s.mean_acc_delta:>+7.2f}% {s.mean_blending_delta:>+7.2f}% "
                f"{s.mean_counterfactual:>+7.2f}% {s.mean_churn:>+7.2f}%"
            )
        return "\n".join(lines)

    def scoreboard_table(self, env_id: int = 0) -> str:
        """Pretty-print scoreboard for an environment."""
        sb = self._get_scoreboard(env_id)

        dist = ", ".join(
            f"{bp} x{count}" for bp, count in sb.fossilized_by_blueprint.items()
        )

        lines = [
            f"Seed Scoreboard (env {env_id}):",
            f"  Fossilized: {sb.total_fossilized} "
            f"(+{sb.params_added/1000:.1f}K params, +{sb.params_percentage:.1f}% of host)",
            f"  Culled: {sb.total_culled}",
            f"  Avg fossilize age: {sb.avg_fossilize_age_epochs:.1f} epochs",
            f"  Avg cull age: {sb.avg_cull_age_epochs:.1f} epochs",
            f"  Compute cost: {sb.compute_cost:.2f}x baseline",
            f"  Distribution: {dist or 'none'}",
        ]
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return serializable snapshot for history."""
        return {
            "stats": {
                bp: {
                    "germinated": s.germinated,
                    "fossilized": s.fossilized,
                    "culled": s.culled,
                    "mean_acc_delta": s.mean_acc_delta,
                    "mean_blending_delta": s.mean_blending_delta,
                    "mean_counterfactual": s.mean_counterfactual,
                    "mean_churn": s.mean_churn,
                    "fossilization_rate": s.fossilization_rate,
                }
                for bp, s in self.stats.items()
            },
            "scoreboards": {
                env_id: {
                    "total_germinated": sb.total_germinated,
                    "total_fossilized": sb.total_fossilized,
                    "total_culled": sb.total_culled,
                    "params_added": sb.params_added,
                    "compute_cost": sb.compute_cost,
                    "total_fossilize_age_epochs": sb.total_fossilize_age_epochs,
                    "total_cull_age_epochs": sb.total_cull_age_epochs,
                }
                for env_id, sb in self.scoreboards.items()
            },
        }


__all__ = [
    "BLUEPRINT_COMPUTE_MULTIPLIERS",
    "compute_cost_for_blueprint",
    "BlueprintStats",
    "SeedScoreboard",
    "BlueprintAnalytics",
]
