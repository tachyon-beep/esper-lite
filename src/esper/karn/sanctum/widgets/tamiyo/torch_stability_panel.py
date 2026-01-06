"""TorchStabilityPanel - PyTorch stability and performance telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class TorchStabilityPanel(Static):
    """PyTorch stability and performance panel (compile + CUDA memory + timing)."""

    LABEL_W = 10

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "TORCH STABILITY"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Text:
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        tamiyo = self._snapshot.tamiyo
        infra = tamiyo.infrastructure
        result = Text()

        # Line 1: torch.compile status
        self._render_label(result, "Compile")
        if infra.compile_enabled:
            backend = infra.compile_backend if infra.compile_backend else "unknown"
            mode = infra.compile_mode if infra.compile_mode else "default"
            result.append(f"{backend}:{mode}", style="green")
            result.append(" OK", style="green bold")
        else:
            result.append("EAGER", style="red bold reverse")
        result.append("\n")

        # Line 2: CUDA memory allocated/reserved
        self._render_label(result, "CUDA Mem")
        alloc = infra.cuda_memory_allocated_gb
        reserved = infra.cuda_memory_reserved_gb
        if reserved > 0:
            usage = alloc / reserved
            usage_style = "red" if usage > 0.90 else "yellow" if usage > 0.75 else "green"
            result.append(f"{alloc:.1f}/{reserved:.1f}G", style=usage_style)
            result.append(f" {usage:.0%}", style=usage_style)
        else:
            result.append("--/--", style="dim")
        result.append("\n")

        # Line 3: CUDA peak memory
        self._render_label(result, "Peak")
        peak = infra.cuda_memory_peak_gb
        if peak > 0:
            peak_style = "red" if reserved > 0 and (peak / reserved) > 0.90 else "cyan"
            result.append(f"{peak:.1f}G", style=peak_style)
        else:
            result.append("---", style="dim")
        result.append("\n")

        # Line 4: CUDA fragmentation
        self._render_label(result, "Frag")
        frag = infra.cuda_memory_fragmentation
        if reserved > 0:
            frag_style = "red" if frag > 0.30 else "yellow" if frag > 0.20 else "green"
            result.append(f"{frag:.0%}", style=frag_style)
        else:
            result.append("---", style="dim")
        result.append("\n")

        # Line 5: NaN/Inf gradients
        self._render_label(result, "NaN/Inf")
        nan_style = "red bold" if tamiyo.nan_grad_count > 0 else "dim"
        inf_style = "red bold" if tamiyo.inf_grad_count > 0 else "dim"
        result.append(f"NaN:{tamiyo.nan_grad_count}", style=nan_style)
        result.append(" ", style="dim")
        result.append(f"Inf:{tamiyo.inf_grad_count}", style=inf_style)
        result.append("\n")

        # Line 6: DataLoader wait ratio
        self._render_label(result, "DL Wait")
        if tamiyo.ppo_data_received:
            ratio = infra.dataloader_wait_ratio
            ratio_style = "red" if ratio > 0.50 else "yellow" if ratio > 0.25 else "green"
            result.append(f"{ratio:.0%}", style=ratio_style)
        else:
            result.append("---", style="dim")
        result.append("\n")

        # Line 7: PPO update time (ms)
        self._render_label(result, "Update")
        update_ms = tamiyo.update_time_ms
        if update_ms > 0:
            update_style = "red" if update_ms > 3000 else "yellow" if update_ms > 1500 else "cyan"
            result.append(f"{update_ms:.0f}ms", style=update_style)
        else:
            result.append("---", style="dim")

        return result

    def _render_label(self, result: Text, label: str) -> None:
        result.append(label.ljust(self.LABEL_W), style="dim")
