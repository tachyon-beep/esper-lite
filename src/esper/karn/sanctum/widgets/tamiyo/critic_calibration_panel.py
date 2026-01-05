"""CriticCalibrationPanel - critic calibration diagnostics.

Shows whether the value function is aligned with actual returns:
- V-Return correlation + explained variance
- TD error mean/std (bias + noise)
- Bellman error + value span
- Value mean/std
- Calibration summary
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class CriticCalibrationPanel(Static):
    """Critic calibration diagnostics panel."""

    # Column layout: two sub-columns with label + value each
    LABEL_W = 10
    VAL_W = 8
    GAP = 2

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "CRITIC CALIBRATION"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        vf = tamiyo.value_function
        if not tamiyo.ppo_data_received:
            return self._render_placeholder()

        result = Text()

        # Line 1: V-Return Correlation | Explained Variance
        self._render_label(result, "V-Corr")
        corr_style, corr_icon = self._get_corr_style(vf.v_return_correlation)
        self._render_value(
            result, f"{vf.v_return_correlation:.2f}{corr_icon}", corr_style
        )

        self._render_label(result, "EV (fit)")
        ev_style = self._get_ev_style(tamiyo.explained_variance)
        self._render_value(
            result, f"{tamiyo.explained_variance:.2f}", ev_style, last=True
        )
        result.append("\n")

        # Line 2: TD Mean | TD Std
        self._render_label(result, "TD Mean")
        td_style = self._get_td_mean_style(vf.td_error_mean)
        self._render_value(result, f"{vf.td_error_mean:+.1f}", td_style)

        self._render_label(result, "TD Std")
        self._render_value(result, f"{vf.td_error_std:.1f}", "cyan", last=True)
        result.append("\n")

        # Line 3: Bellman | V-Span
        self._render_label(result, "Bellman")
        bellman_style = self._get_bellman_style(vf.bellman_error)
        self._render_value(result, f"{vf.bellman_error:.1f}", bellman_style)

        v_span = tamiyo.value_max - tamiyo.value_min
        self._render_label(result, "V-Span")
        self._render_value(result, f"{v_span:.1f}", "cyan", last=True)
        result.append("\n")

        # Line 4: V-Mean | V-Std
        self._render_label(result, "V-Mean")
        self._render_value(result, f"{tamiyo.value_mean:+.1f}", "cyan")
        self._render_label(result, "V-Std")
        self._render_value(result, f"{tamiyo.value_std:.1f}", "cyan", last=True)
        result.append("\n")

        # Line 5: Calibration summary
        self._render_label(result, "Calib")
        calib_label, calib_style = self._get_calib_summary(
            corr_style, ev_style, td_style, bellman_style
        )
        self._render_value(result, calib_label, calib_style, last=True)

        return result

    def _render_placeholder(self) -> Text:
        result = Text()
        for label_left, label_right in (
            ("V-Corr", "EV"),
            ("TD Mean", "TD Std"),
            ("Bellman", "V-Span"),
            ("V-Mean", "V-Std"),
            ("Calib", ""),
        ):
            self._render_label(result, label_left)
            self._render_value(result, "---", "dim")
            if label_right:
                self._render_label(result, label_right)
                self._render_value(result, "---", "dim", last=True)
            else:
                self._render_value(result, "", "dim", last=True)
            result.append("\n")
        return result

    def _render_label(self, result: Text, label: str) -> None:
        result.append(label.ljust(self.LABEL_W), style="dim")

    def _render_value(
        self, result: Text, value: str, style: str, *, last: bool = False
    ) -> None:
        result.append(value.ljust(self.VAL_W), style=style)
        if not last:
            result.append(" " * self.GAP)

    def _get_corr_style(self, corr: float) -> tuple[str, str]:
        if corr >= 0.8:
            return ("green bold", "↗")
        if corr >= TUIThresholds.V_RETURN_CORR_WARNING:
            return ("green", "→")
        if corr >= TUIThresholds.V_RETURN_CORR_CRITICAL:
            return ("yellow", "→")
        return ("red bold", "↘")

    def _get_ev_style(self, ev: float) -> str:
        if ev <= TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "red bold"
        if ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "yellow"
        return "green"

    def _get_td_mean_style(self, mean: float) -> str:
        abs_mean = abs(mean)
        if abs_mean < 5:
            return "green"
        if abs_mean < 15:
            return "yellow"
        return "red bold"

    def _get_bellman_style(self, bellman: float) -> str:
        if bellman < 20:
            return "green"
        if bellman < 50:
            return "yellow"
        return "red bold"

    def _get_calib_summary(
        self,
        corr_style: str,
        ev_style: str,
        td_style: str,
        bellman_style: str,
    ) -> tuple[str, str]:
        styles = [corr_style, ev_style, td_style, bellman_style]
        if "red" in " ".join(styles):
            return ("bad", "red bold")
        if "yellow" in " ".join(styles):
            return ("weak", "yellow")
        return ("ok", "green")
