"""TamiyoBrain - Policy agent dashboard.

Policy agent diagnostics using Textual's widget composition for
maintainability, testability, and visual design.

Usage:
    from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain

    widget = TamiyoBrain()
    widget.update_snapshot(snapshot)
"""

from .tamiyo_brain import TamiyoBrain
from .ppo_losses_panel import PPOLossesPanel
from .action_heads_panel import ActionHeadsPanel
from .episode_metrics_panel import EpisodeMetricsPanel
from .value_diagnostics_panel import ValueDiagnosticsPanel

__all__ = [
    "TamiyoBrain",
    "PPOLossesPanel",
    "ActionHeadsPanel",
    "EpisodeMetricsPanel",
    "ValueDiagnosticsPanel",
]
