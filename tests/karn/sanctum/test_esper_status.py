from rich.console import Console
from rich.panel import Panel

from esper.karn.sanctum.schema import EnvState, GPUStats, SanctumSnapshot, SeedState, SystemVitals
from esper.karn.sanctum.widgets.base import SanctumWidget
from esper.karn.sanctum.widgets.esper_status import EsperStatus


def _render_text(panel: Panel) -> str:
    console = Console(width=100, record=True)
    console.print(panel)
    return console.export_text()


def test_esper_status_renders_no_data_panel() -> None:
    widget = EsperStatus()

    text = _render_text(widget.render())

    assert "ESPER STATUS" in text
    assert "No data" in text


def test_esper_status_renders_multigpu_ram_cpu_and_seed_counts() -> None:
    widget = EsperStatus()
    snapshot = SanctumSnapshot(
        envs={
            0: EnvState(
                env_id=0,
                seeds={
                    "r0c0": SeedState(slot_id="r0c0", stage="TRAINING"),
                    "r0c1": SeedState(slot_id="r0c1", stage="DORMANT"),
                },
            ),
            1: EnvState(
                env_id=1,
                seeds={"r1c0": SeedState(slot_id="r1c0", stage="FOSSILIZED")},
            ),
        },
        vitals=SystemVitals(
            gpu_stats={
                0: GPUStats(device_id=0, memory_used_gb=4.0, memory_total_gb=16.0, utilization=50.0),
                1: GPUStats(device_id=1, memory_used_gb=15.0, memory_total_gb=16.0, utilization=97.0),
            },
            ram_used_gb=12.0,
            ram_total_gb=32.0,
            cpu_percent=37.5,
            epochs_per_second=2.5,
            batches_per_hour=90.0,
            host_params=1_250_000,
        ),
        runtime_seconds=3661.0,
    )

    widget.update_snapshot(snapshot)
    text = _render_text(widget.render())

    assert "Foss" in text
    assert "Train" in text
    assert "1.2M" in text
    assert "2.50" in text
    assert "90" in text
    assert "1h 1m 1s" in text
    assert "GPU0:" in text
    assert "4.0/16.0GB" in text
    assert "GPU1:" in text
    assert "15.0/16.0GB" in text
    assert "97%" in text
    assert "12.0/32GB" in text
    assert "37.5%" in text


def test_esper_status_renders_legacy_single_gpu_and_empty_resources() -> None:
    widget = EsperStatus()
    snapshot = SanctumSnapshot(
        vitals=SystemVitals(
            gpu_memory_used_gb=8.0,
            gpu_memory_total_gb=10.0,
            gpu_utilization=85.0,
            ram_used_gb=None,
            ram_total_gb=None,
            cpu_percent=0.0,
            host_params=512,
        )
    )

    widget.update_snapshot(snapshot)
    text = _render_text(widget.render())

    assert "Host Params:" in text
    assert "512" in text
    assert "GPU:" in text
    assert "8.0/10.0GB" in text
    assert "GPU util:" in text
    assert "85%" in text
    assert "RAM:" in text
    assert "CPU:" in text


def test_esper_status_renders_k_params_and_no_gpu_fallback() -> None:
    widget = EsperStatus()
    snapshot = SanctumSnapshot(vitals=SystemVitals(host_params=50_000))

    widget.update_snapshot(snapshot)
    text = _render_text(widget.render())

    assert "50K" in text
    assert "GPU:" in text
    assert "RAM:" in text
    assert "CPU:" in text


def test_sanctum_widget_protocol_accepts_snapshot_receiver() -> None:
    class Receiver:
        def __init__(self) -> None:
            self.snapshot: SanctumSnapshot | None = None

        def update_snapshot(self, snapshot: SanctumSnapshot) -> None:
            self.snapshot = snapshot

    receiver: SanctumWidget = Receiver()
    snapshot = SanctumSnapshot()

    receiver.update_snapshot(snapshot)

    assert receiver.snapshot is snapshot
