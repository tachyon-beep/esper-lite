"""ExperimentPanel - side-by-side A/B leg comparison.

One column per policy group, rows = the EV-stab acceptance metrics, plus a
delta column when exactly two legs are running. Replaces compare-from-memory
via the 't' toggle for judging paired runs (the toggle still selects which leg
drives the other tabs).

Missing values render as an em dash — a leg that does not emit a metric must
not show a fabricated 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from rich.table import Table
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot

# Group identity colors, mirroring the TamiyoBrain border classes in styles.tcss
# (Rich color names here; the CSS uses the ansi_-prefixed Textual equivalents)
_GROUP_COLORS = {"a": "bright_green", "b": "bright_cyan", "c": "bright_magenta"}


def _fmt(value: float | None, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


class ExperimentPanel(Static):
    """A/B experiment comparison table."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._groups: dict[str, "SanctumSnapshot"] = {}
        self._primary_group_id: str | None = None
        self.classes = "panel"
        self.border_title = "EXPERIMENT"

    @property
    def group_ids(self) -> list[str]:
        return list(self._groups.keys())

    def update_groups(
        self,
        snapshots_by_group: dict[str, "SanctumSnapshot"],
        primary_group_id: str | None,
    ) -> None:
        """Update with all policy-group snapshots for this tick."""
        self._groups = dict(snapshots_by_group)
        self._primary_group_id = primary_group_id
        self.refresh()

    def _ordered_group_ids(self) -> list[str]:
        ordered = sorted(self._groups.keys())
        if "default" in ordered:
            ordered.remove("default")
            ordered.insert(0, "default")
        return ordered

    def _any_ev_stab_leg(self) -> bool:
        return any(
            s.tamiyo.hra_leg_active
            or s.tamiyo.rvt_leg_active
            or s.tamiyo.advantage_per_head_normalized
            for s in self._groups.values()
        )

    def render(self) -> Table | Text:
        if not self._groups:
            return Text("[no data]", style="dim")

        if len(self._groups) < 2 and not self._any_ev_stab_leg():
            return Text(
                "Single-leg run — this view populates in A/B mode "
                "or when an EV-stab leg (HRA/RVT/PHN) is active.",
                style="dim",
            )

        ordered = self._ordered_group_ids()

        # Metric rows: label, extractor, format spec, recent-window history attr
        # (None = no window). None from the extractor renders as an em dash.
        # Contaminated diagnostics (cov_rcf/r_main_cov) are absent (PDR-0028).
        # These are single-update POINT ESTIMATES from ONE seed per leg — the Δ is
        # not a seeded A/B verdict (see the caveat caption). Between-seed variance,
        # the dominant variance in RL, is entirely unobserved here, so NO cross-leg
        # CI/SEM/significance is computed — only a negative "within noise" filter.
        def t(group: "SanctumSnapshot") -> Any:
            return group.tamiyo

        rows: list[
            tuple[str, Callable[["SanctumSnapshot"], float | None], str, str | None]
        ] = [
            ("EV", lambda s: t(s).explained_variance if t(s).ppo_data_received else None, ".2f", "explained_variance_history"),
            ("EV main", lambda s: t(s).ev_main, ".2f", "ev_main_history"),
            ("EV cf", lambda s: t(s).ev_cf, ".2f", "ev_cf_history"),
            ("cf V.Loss", lambda s: t(s).cf_value_loss, ".3f", "cf_value_loss_history"),
            ("CF var share", lambda s: t(s).return_var_cf_share, ".2f", "return_var_cf_share_history"),
            ("Main var share", lambda s: t(s).return_var_main_share, ".2f", None),
            ("Entropy", lambda s: t(s).entropy if t(s).ppo_data_received else None, ".2f", "entropy_history"),
            ("KL", lambda s: t(s).kl_divergence if t(s).ppo_data_received else None, ".3f", "kl_divergence_history"),
            ("Clip frac", lambda s: t(s).clip_fraction if t(s).ppo_data_received else None, ".2f", "clip_fraction_history"),
            ("Grad norm", lambda s: t(s).grad_norm if t(s).ppo_data_received else None, ".2f", "grad_norm_history"),
            ("Mean accuracy", lambda s: s.aggregate_mean_accuracy, ".1f", None),
            ("Mean reward", lambda s: s.aggregate_mean_reward, ".2f", None),
        ]

        table = Table(expand=False, pad_edge=False, border_style="dim")
        table.add_column("metric", style="dim", min_width=14)
        for group_id in ordered:
            color = _GROUP_COLORS.get(group_id.lower(), "white")
            marker = " ●" if group_id == self._primary_group_id else ""
            table.add_column(f"{group_id}{marker}", style=color, min_width=10)
        show_delta = len(ordered) == 2
        if show_delta:
            # Not bold: a point-estimate delta must not read as a verdict.
            table.add_column(f"Δ ({ordered[0]}−{ordered[1]})", style="dim", min_width=12)

        for label, extract, spec, hist_attr in rows:
            values = [extract(self._groups[group_id]) for group_id in ordered]
            if all(v is None for v in values):
                continue  # neither leg emits this metric: no dead rows
            cells = [label] + [_fmt(v, spec) for v in values]
            if show_delta:
                if values[0] is None or values[1] is None:
                    cells.append("—")
                else:
                    delta = values[0] - values[1]
                    delta_str = format(delta, f"+{spec}")
                    if self._within_noise(delta, hist_attr, ordered):
                        delta_str += " ~noise"
                    cells.append(delta_str)
            table.add_row(*cells)

        table.caption = (
            "point estimates · 1 seed/leg — not a seeded A/B verdict; Δ needs N "
            "seeds for significance. ~noise = |Δ| within a leg's recent-window wobble."
        )
        table.caption_style = "dim italic"
        return table

    def _within_noise(
        self, delta: float, history_attr: str | None, ordered: list[str]
    ) -> bool:
        """Negative-only filter: is |Δ| inside either leg's recent-window wobble?

        Δ smaller than the update-to-update wobble is indistinguishable from noise.
        This is used ONLY to demote a Δ (never to promote one to significance): a
        single autocorrelated run cannot support a positive claim.
        """
        if history_attr is None:
            return False
        wobbles: list[float] = []
        for group_id in ordered:
            history = list(getattr(self._groups[group_id].tamiyo, history_attr))
            if len(history) >= 2:
                wobbles.append(max(history) - min(history))
        return bool(wobbles) and abs(delta) <= max(wobbles)
