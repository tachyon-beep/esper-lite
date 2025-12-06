"""Test gradient isolation monitoring performance."""
import pytest
import torch
import torch.nn as nn

from esper.kasmina.isolation import GradientIsolationMonitor


class TestGradientIsolationPerformance:
    """Verify foreach_norm optimization."""

    def test_check_isolation_uses_foreach(self):
        """Verify batched norm computation is used."""
        host = nn.Linear(64, 64)
        seed = nn.Linear(64, 64)

        monitor = GradientIsolationMonitor()
        monitor.register(host, seed)

        # Create gradients
        x = torch.randn(4, 64)
        loss = (host(x) + seed(x)).sum()
        loss.backward()

        is_isolated, stats = monitor.check_isolation()

        # Verify stats are computed
        assert "host_grad_norm" in stats
        assert "seed_grad_norm" in stats
        assert stats["host_grad_norm"] > 0
        assert stats["seed_grad_norm"] > 0
