"""Nissa Analytics - Blueprint performance aggregation.

Aggregates telemetry events into strategic dashboards:
- BlueprintStats: Per-blueprint fossilization rates and accuracy metrics
- SeedScoreboard: Per-environment cumulative seed tracking
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    SeedGerminatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    SeedStageChangedPayload,
    AnalyticsSnapshotPayload,
    GovernorRollbackPayload,
    TrendDetectedPayload,
    TamiyoInitiatedPayload,
)
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
    pruned: int = 0
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
        """Mean accuracy change on prune (usually negative)."""
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
        """Percentage of terminal seeds that fossilized (not pruned)."""
        total = self.fossilized + self.pruned
        return (self.fossilized / total * 100) if total > 0 else 0.0


@dataclass
class SeedScoreboard:
    """Cumulative seed tracking for an environment."""

    total_germinated: int = 0
    total_fossilized: int = 0
    total_pruned: int = 0
    fossilized_by_blueprint: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    live_blueprint: str | None = None
    params_added: int = 0
    host_params: int = 0
    total_fossilize_age_epochs: int = 0
    total_prune_age_epochs: int = 0

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
    def avg_prune_age_epochs(self) -> float:
        """Average total age (epochs) at prune."""
        return (
            self.total_prune_age_epochs / self.total_pruned
            if self.total_pruned > 0
            else 0.0
        )


class BlueprintAnalytics(OutputBackend):
    """Aggregates blueprint performance from telemetry events.

    Implements OutputBackend to receive events from NissaHub.
    Tracks:
    - Per-blueprint stats (germinated, fossilized, pruned, accuracy)
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
            if isinstance(event.data, SeedGerminatedPayload):
                bp_id = event.data.blueprint_id
                env_id = event.data.env_id
                seed_id = event.seed_id or "unknown"
                params = event.data.params
            else:
                _logger.warning(f"Unexpected SEED_GERMINATED payload type: {type(event.data)}")
                return

            self.stats[bp_id].germinated += 1
            sb = self._get_scoreboard(env_id)
            sb.total_germinated += 1
            sb.live_blueprint = bp_id

            if not self.quiet:
                print(f"    [env{env_id}] Germinated '{seed_id}' ({bp_id}, {params/1000:.1f}K params)")

        elif event.event_type == TelemetryEventType.SEED_FOSSILIZED:
            if isinstance(event.data, SeedFossilizedPayload):
                bp_id = event.data.blueprint_id
                env_id = event.data.env_id
                seed_id = event.seed_id or "unknown"
                improvement = event.data.improvement
                blending_delta = event.data.blending_delta if event.data.blending_delta is not None else 0.0
                counterfactual = event.data.counterfactual
                params = event.data.params_added
                epochs_total = event.data.epochs_total
            else:
                _logger.warning(f"Unexpected SEED_FOSSILIZED payload type: {type(event.data)}")
                return

            self.stats[bp_id].fossilized += 1
            self.stats[bp_id].acc_deltas.append(improvement)
            self.stats[bp_id].blending_deltas.append(blending_delta)
            if counterfactual is not None:
                self.stats[bp_id].counterfactuals.append(counterfactual)

            sb = self._get_scoreboard(env_id)
            sb.total_fossilized += 1
            sb.fossilized_by_blueprint[bp_id] += 1
            sb.params_added += params
            if epochs_total is not None:
                sb.total_fossilize_age_epochs += int(epochs_total)
            sb.live_blueprint = None

            # Show total improvement, blending delta, and causal contribution
            if not self.quiet:
                causal_str = f", causal Δ {counterfactual:+.2f}%" if counterfactual is not None else ""
                print(f"    [env{env_id}] Fossilized '{seed_id}' ({bp_id}, "
                      f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%{causal_str})")

        elif event.event_type == TelemetryEventType.SEED_PRUNED:
            if isinstance(event.data, SeedPrunedPayload):
                bp_id = event.data.blueprint_id or "unknown"
                env_id = event.data.env_id
                seed_id = event.seed_id or "unknown"
                improvement = event.data.improvement
                blending_delta = event.data.blending_delta if event.data.blending_delta is not None else 0.0
                counterfactual = event.data.counterfactual
                reason = event.data.reason
                epochs_total = event.data.epochs_total
            else:
                _logger.warning(f"Unexpected SEED_PRUNED payload type: {type(event.data)}")
                return

            self.stats[bp_id].pruned += 1
            self.stats[bp_id].churns.append(improvement)
            self.stats[bp_id].blending_deltas.append(blending_delta)
            if counterfactual is not None:
                self.stats[bp_id].counterfactuals.append(counterfactual)

            sb = self._get_scoreboard(env_id)
            sb.total_pruned += 1
            if epochs_total is not None:
                sb.total_prune_age_epochs += int(epochs_total)
            sb.live_blueprint = None

            # Show total improvement, blending delta, and causal contribution
            if not self.quiet:
                reason_str = f" ({reason})" if reason else ""
                causal_str = f", causal Δ {counterfactual:+.2f}%" if counterfactual is not None else ""
                print(f"    [env{env_id}] Pruned '{seed_id}' ({bp_id}, "
                      f"total Δacc {improvement:+.2f}%, blending Δ {blending_delta:+.2f}%{causal_str}){reason_str}")

        elif event.event_type == TelemetryEventType.ANALYTICS_SNAPSHOT:
            if isinstance(event.data, AnalyticsSnapshotPayload):
                kind = event.data.kind
                env_id_snapshot = event.data.env_id
                if not self.quiet:
                    _logger.debug(f"[env{env_id_snapshot}] ANALYTICS_SNAPSHOT kind={kind}")
            else:
                _logger.warning(f"Unexpected ANALYTICS_SNAPSHOT payload type: {type(event.data)}")
                return

        # === Training Progress Events ===
        elif event.event_type == TelemetryEventType.SEED_STAGE_CHANGED:
            if isinstance(event.data, SeedStageChangedPayload):
                env_id = event.data.env_id
                slot_id = event.data.slot_id
                old_stage = event.data.from_stage
                new_stage = event.data.to_stage
                _logger.debug(f"[env{env_id}] Stage change {slot_id}: {old_stage} → {new_stage}")
            else:
                _logger.warning(f"Unexpected SEED_STAGE_CHANGED payload type: {type(event.data)}")
                return

        elif event.event_type == TelemetryEventType.SEED_GATE_EVALUATED:
            # SEED_GATE_EVALUATED uses typed payload (SeedGateEvaluatedPayload)
            # No action needed - gates don't affect blueprint stats
            return

        elif event.event_type == TelemetryEventType.TAMIYO_INITIATED:
            # TAMIYO_INITIATED signals host stabilization - no analytics aggregation needed.
            # Just validate the payload type and return.
            if not isinstance(event.data, TamiyoInitiatedPayload):
                _logger.warning(f"TAMIYO_INITIATED unexpected payload: {type(event.data).__name__}")
            return

        # === Trend Detection Events ===
        elif event.event_type in (
            TelemetryEventType.PLATEAU_DETECTED,
            TelemetryEventType.DEGRADATION_DETECTED,
            TelemetryEventType.IMPROVEMENT_DETECTED,
        ):
            # Trend events use TrendDetectedPayload for typed access.
            # No analytics aggregation needed - events are logged/output for monitoring.
            if not isinstance(event.data, TrendDetectedPayload):
                _logger.warning(f"{event.event_type.name} missing typed payload")
            return

        # === Health/Warning Events ===
        elif event.event_type == TelemetryEventType.MEMORY_WARNING:
            _logger.warning("MEMORY_WARNING event not yet migrated to typed payload")
            return

        elif event.event_type == TelemetryEventType.PERFORMANCE_DEGRADATION:
            _logger.warning("PERFORMANCE_DEGRADATION event not yet migrated to typed payload")
            return

        elif event.event_type == TelemetryEventType.REWARD_HACKING_SUSPECTED:
            _logger.warning("REWARD_HACKING_SUSPECTED event not yet migrated to typed payload")
            return

        # === PPO Anomaly Events (use AnomalyDetectedPayload) ===
        elif event.event_type in (
            TelemetryEventType.GRADIENT_ANOMALY,
            TelemetryEventType.RATIO_EXPLOSION_DETECTED,
            TelemetryEventType.RATIO_COLLAPSE_DETECTED,
            TelemetryEventType.VALUE_COLLAPSE_DETECTED,
            TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED,
            TelemetryEventType.GRADIENT_PATHOLOGY_DETECTED,
        ):
            # All anomaly events now use AnomalyDetectedPayload
            # No analytics processing needed - events are logged for debugging
            return

        # === Governor Events ===
        elif event.event_type == TelemetryEventType.GOVERNOR_ROLLBACK:
            # Governor rollbacks are operational safety events, not blueprint performance metrics.
            # The event is logged/output by ConsoleOutput; no aggregation needed here.
            if not isinstance(event.data, GovernorRollbackPayload):
                _logger.error("GOVERNOR_ROLLBACK event missing typed payload")
                return
            _logger.debug(
                "Governor rollback on env %d: %s (loss=%.4f)",
                event.data.env_id,
                event.data.reason,
                event.data.loss_at_panic if event.data.loss_at_panic is not None else float('nan'),
            )
            return

        # === Counterfactual Events ===
        elif event.event_type == TelemetryEventType.COUNTERFACTUAL_COMPUTED:
            _logger.warning("COUNTERFACTUAL_COMPUTED event not yet migrated to typed payload")
            return

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
            f"  {'Blueprint':<14} {'Germ':>5} {'Foss':>5} {'Prun':>5} "
            f"{'Rate':>6} {'ΔAcc':>8} {'BlendΔ':>8} {'CausalΔ':>8} {'Churn':>8}"
        )
        lines.append("  " + "-" * 110)

        for bp_id in sorted(self.stats.keys()):
            s = self.stats[bp_id]
            lines.append(
                f"  {bp_id:<14} {s.germinated:>5} {s.fossilized:>5} "
                f"{s.pruned:>5} {s.fossilization_rate:>5.1f}% "
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
            f"  Pruned: {sb.total_pruned}",
            f"  Avg fossilize age: {sb.avg_fossilize_age_epochs:.1f} epochs",
            f"  Avg prune age: {sb.avg_prune_age_epochs:.1f} epochs",
            f"  Compute cost: {sb.compute_cost:.2f}x baseline",
            f"  Distribution: {dist or 'none'}",
        ]
        return "\n".join(lines)

    def snapshot(self) -> dict[str, Any]:
        """Return serializable snapshot for history."""
        return {
            "stats": {
                bp: {
                    "germinated": s.germinated,
                    "fossilized": s.fossilized,
                    "pruned": s.pruned,
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
                    "total_pruned": sb.total_pruned,
                    "params_added": sb.params_added,
                    "compute_cost": sb.compute_cost,
                    "total_fossilize_age_epochs": sb.total_fossilize_age_epochs,
                    "total_prune_age_epochs": sb.total_prune_age_epochs,
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
