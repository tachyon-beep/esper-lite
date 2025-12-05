"""GPU Memory Telemetry.

Tracks CUDA memory usage for OOM prevention and fragmentation detection.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass(slots=True)
class MemoryMetrics:
    """GPU memory statistics for training monitoring.

    Ops Normal level - collected periodically (every N epochs).
    """

    # CUDA memory (MB)
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float

    # Derived indicators
    fragmentation_ratio: float  # reserved / allocated
    utilization: float  # allocated / total
    headroom_mb: float  # total - reserved

    # Risk indicator
    oom_risk_score: float  # 0-1, higher = more risk

    # Optional: seed-specific
    seed_param_mb: float = 0.0

    def is_healthy(
        self,
        min_headroom_mb: float = 200.0,
        max_fragmentation: float = 2.5,
    ) -> bool:
        """Check if memory situation is healthy.

        Args:
            min_headroom_mb: Minimum free memory buffer
            max_fragmentation: Maximum fragmentation ratio

        Returns:
            True if memory is in healthy state
        """
        return (
            self.headroom_mb >= min_headroom_mb
            and self.fragmentation_ratio <= max_fragmentation
        )

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


def collect_memory_metrics(
    device: torch.device,
    seed_module: nn.Module | None = None,
) -> MemoryMetrics:
    """Collect GPU memory statistics.

    Call after backward() for accurate peak tracking.
    Returns zero metrics for CPU device.

    Args:
        device: PyTorch device to check
        seed_module: Optional seed module for param memory calculation

    Returns:
        MemoryMetrics instance
    """
    if not torch.cuda.is_available() or device.type != "cuda":
        return MemoryMetrics(
            allocated_mb=0.0,
            reserved_mb=0.0,
            max_allocated_mb=0.0,
            fragmentation_ratio=1.0,
            utilization=0.0,
            headroom_mb=float("inf"),
            oom_risk_score=0.0,
            seed_param_mb=0.0,
        )

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2

    # Fragmentation ratio
    fragmentation = reserved / max(allocated, 1.0)

    # Utilization
    utilization = allocated / total

    # Headroom
    headroom = total - reserved

    # Seed param memory
    seed_param_mb = 0.0
    if seed_module is not None:
        seed_param_mb = sum(
            p.numel() * p.element_size() for p in seed_module.parameters()
        ) / 1024**2

    # OOM risk score (heuristic)
    oom_risk = 0.0
    if reserved > 0.9 * total:
        oom_risk += 0.5
    if fragmentation > 2.0:
        oom_risk += 0.3
    if headroom < 200:
        oom_risk += 0.2
    oom_risk = min(1.0, oom_risk)

    return MemoryMetrics(
        allocated_mb=allocated,
        reserved_mb=reserved,
        max_allocated_mb=max_alloc,
        fragmentation_ratio=fragmentation,
        utilization=utilization,
        headroom_mb=headroom,
        oom_risk_score=oom_risk,
        seed_param_mb=seed_param_mb,
    )


__all__ = ["MemoryMetrics", "collect_memory_metrics"]
