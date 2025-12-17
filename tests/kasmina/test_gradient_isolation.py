"""Test gradient health monitoring performance."""
import torch
import torch.nn as nn

from esper.kasmina.isolation import GradientHealthMonitor


class TestGradientHealthMonitor:
    """Verify foreach_norm optimization for gradient health monitoring."""

    def test_compute_gradient_health_uses_foreach(self):
        """Verify batched norm computation is used."""
        host = nn.Linear(64, 64)
        seed = nn.Linear(64, 64)

        monitor = GradientHealthMonitor()
        monitor.register(host, seed)

        # Create gradients
        x = torch.randn(4, 64)
        loss = (host(x) + seed(x)).sum()
        loss.backward()

        stats = monitor.compute_gradient_health()

        # Verify stats are computed
        assert "host_grad_norm" in stats
        assert "seed_grad_norm" in stats
        assert stats["host_grad_norm"] > 0
        assert stats["seed_grad_norm"] > 0
