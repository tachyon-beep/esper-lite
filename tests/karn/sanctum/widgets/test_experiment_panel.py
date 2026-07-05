"""ExperimentPanel: side-by-side A/B leg comparison of acceptance metrics."""

from io import StringIO

from rich.console import Console

from esper.karn.sanctum.schema import SanctumSnapshot
from esper.karn.sanctum.widgets.experiment_panel import ExperimentPanel


def _render(panel: ExperimentPanel) -> str:
    console = Console(file=StringIO(), width=140, legacy_windows=False)
    console.print(panel.render())
    return console.file.getvalue()


def _leg(
    ev: float,
    *,
    ev_main: float | None = None,
    cf_share: float | None = None,
    accuracy: float = 0.0,
) -> SanctumSnapshot:
    snapshot = SanctumSnapshot(aggregate_mean_accuracy=accuracy)
    snapshot.tamiyo.ppo_data_received = True
    snapshot.tamiyo.explained_variance = ev
    if ev_main is not None:
        snapshot.tamiyo.hra_leg_active = True
        snapshot.tamiyo.ev_main = ev_main
    if cf_share is not None:
        snapshot.tamiyo.rvt_leg_active = True
        snapshot.tamiyo.return_var_cf_share = cf_share
    return snapshot


def test_placeholder_when_single_plain_leg():
    panel = ExperimentPanel()
    panel.update_groups({"default": SanctumSnapshot()}, "default")
    output = _render(panel)
    assert "A/B" in output  # explains when this view populates


def test_two_groups_render_columns_and_delta():
    panel = ExperimentPanel()
    panel.update_groups(
        {"A": _leg(0.31, ev_main=0.42, accuracy=71.2), "B": _leg(0.18, accuracy=70.9)},
        "A",
    )
    output = _render(panel)

    assert "A" in output and "B" in output
    assert "0.31" in output and "0.18" in output
    assert "Δ" in output
    assert "+0.13" in output  # EV delta A-B
    assert "0.42" in output  # A-only metric renders
    assert "—" in output  # B lacks EV main -> em dash, not a fabricated 0


def test_cf_share_row_present_when_rvt_leg_active():
    panel = ExperimentPanel()
    panel.update_groups(
        {"A": _leg(0.30, cf_share=0.51), "B": _leg(0.28, cf_share=0.53)},
        "A",
    )
    output = _render(panel)
    assert "CF var share" in output
    assert "0.51" in output and "0.53" in output


def test_single_leg_with_ev_stab_flag_still_renders_table():
    """A lone RVT/HRA leg is still an experiment: show its column."""
    panel = ExperimentPanel()
    panel.update_groups({"default": _leg(0.25, cf_share=0.44)}, "default")
    output = _render(panel)
    assert "0.44" in output
