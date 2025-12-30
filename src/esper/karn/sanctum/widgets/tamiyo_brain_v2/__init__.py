"""TamiyoBrainV2 - Redesigned policy agent dashboard.

A clean-slate rebuild using Textual's widget composition for better
maintainability, testability, and visual design.

Usage:
    from esper.karn.sanctum.widgets.tamiyo_brain_v2 import TamiyoBrainV2

    # Drop-in replacement for TamiyoBrain
    widget = TamiyoBrainV2()
    widget.update_snapshot(snapshot)
"""

from .widget import TamiyoBrainV2
from .ppo_losses_panel import PPOLossesPanel
from .attention_heatmap import AttentionHeatmapPanel

__all__ = ["TamiyoBrainV2", "PPOLossesPanel", "AttentionHeatmapPanel"]
