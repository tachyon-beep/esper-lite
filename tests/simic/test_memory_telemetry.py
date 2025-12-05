"""Tests for memory telemetry."""

import torch
import pytest

from esper.simic.memory_telemetry import MemoryMetrics, collect_memory_metrics


class TestMemoryMetrics:
    """Tests for MemoryMetrics."""

    def test_collect_cpu_returns_zeros(self):
        """CPU device returns zero metrics."""
        metrics = collect_memory_metrics(torch.device("cpu"))
        assert metrics.allocated_mb == 0.0
        assert metrics.reserved_mb == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_collect_cuda_returns_positive(self):
        """CUDA device returns positive metrics when memory used."""
        device = torch.device("cuda:0")
        # Allocate some memory
        x = torch.randn(1000, 1000, device=device)

        metrics = collect_memory_metrics(device)
        assert metrics.allocated_mb > 0
        assert metrics.reserved_mb >= metrics.allocated_mb
        assert metrics.headroom_mb >= 0

        del x

    def test_oom_risk_score_range(self):
        """OOM risk score is in valid range."""
        metrics = MemoryMetrics(
            allocated_mb=8000,
            reserved_mb=10000,
            max_allocated_mb=9000,
            fragmentation_ratio=1.25,
            utilization=0.8,
            headroom_mb=2000,
            oom_risk_score=0.3,
        )
        assert 0.0 <= metrics.oom_risk_score <= 1.0

    def test_is_healthy(self):
        """is_healthy detects low headroom."""
        healthy = MemoryMetrics(
            allocated_mb=5000,
            reserved_mb=6000,
            max_allocated_mb=5500,
            fragmentation_ratio=1.2,
            utilization=0.5,
            headroom_mb=4000,
            oom_risk_score=0.1,
        )
        assert healthy.is_healthy() is True

        unhealthy = MemoryMetrics(
            allocated_mb=9000,
            reserved_mb=9800,
            max_allocated_mb=9500,
            fragmentation_ratio=1.1,
            utilization=0.95,
            headroom_mb=50,
            oom_risk_score=0.8,
        )
        assert unhealthy.is_healthy() is False

    def test_is_healthy_high_fragmentation(self):
        """High fragmentation is unhealthy even with adequate headroom."""
        fragmented = MemoryMetrics(
            allocated_mb=2000,
            reserved_mb=6000,  # 3x fragmentation ratio
            max_allocated_mb=2500,
            fragmentation_ratio=3.0,  # Above default 2.5 threshold
            utilization=0.3,
            headroom_mb=4000,  # Plenty of headroom
            oom_risk_score=0.3,
        )
        assert fragmented.is_healthy() is False

    def test_is_healthy_with_custom_thresholds(self):
        """Can customize health thresholds."""
        metrics = MemoryMetrics(
            allocated_mb=5000,
            reserved_mb=6000,
            max_allocated_mb=5500,
            fragmentation_ratio=1.5,
            utilization=0.5,
            headroom_mb=150,  # Below default 200MB threshold
            oom_risk_score=0.2,
        )
        # Default thresholds - unhealthy (headroom < 200)
        assert metrics.is_healthy() is False
        # Custom thresholds - healthy
        assert metrics.is_healthy(min_headroom_mb=100.0) is True
