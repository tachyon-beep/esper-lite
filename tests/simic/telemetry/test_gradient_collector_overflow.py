"""Tests for gradient_collector numerical stability.

B7-PT-01 regression tests: verify float64 overflow protection works
for extreme gradient values that would overflow float32.
"""

import math

import torch
from torch import nn

from esper.simic.telemetry.gradient_collector import (
    SeedGradientCollector,
    collect_seed_gradients,
    collect_host_gradients_async,
    collect_seed_gradients_only_async,
    materialize_grad_stats,
)


class TestGradientCollectorOverflowProtection:
    """B7-PT-01: Verify float64 prevents overflow in squared norm aggregation."""

    def test_collector_handles_extreme_gradients(self):
        """SeedGradientCollector.collect_async should not overflow with 1e20 gradients."""
        collector = SeedGradientCollector()
        model = nn.Linear(10, 5)

        # Set gradients to 1e20 - would overflow float32 when squared
        for param in model.parameters():
            param.grad = torch.full_like(param, 1e20)

        async_stats = collector.collect_async(model.parameters())

        # The squared sum should be finite (not inf)
        assert "_total_squared_norm" in async_stats
        squared_norm = async_stats["_total_squared_norm"]
        assert torch.isfinite(squared_norm), f"Squared norm overflowed: {squared_norm}"

        # Materialize and verify final gradient_norm is finite
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        stats = materialize_grad_stats(async_stats)
        assert math.isfinite(stats["gradient_norm"]), f"Gradient norm: {stats['gradient_norm']}"
        assert stats["gradient_norm"] > 1e19  # Should be ~7e20

    def test_collect_seed_gradients_handles_extreme_values(self):
        """collect_seed_gradients should not overflow with 1e20 gradients."""
        model = nn.Linear(10, 5)

        for param in model.parameters():
            param.grad = torch.full_like(param, 1e20)

        stats = collect_seed_gradients(model.parameters())

        assert math.isfinite(stats["gradient_norm"]), f"Got: {stats['gradient_norm']}"
        assert stats["gradient_norm"] > 1e19

    def test_collect_host_gradients_async_handles_extreme_values(self):
        """collect_host_gradients_async should not overflow with 1e20 gradients."""
        model = nn.Linear(10, 5)

        for param in model.parameters():
            param.grad = torch.full_like(param, 1e20)

        async_stats = collect_host_gradients_async(model.parameters())

        squared_sum = async_stats["_host_squared_sum"]
        assert torch.isfinite(squared_sum), f"Host squared sum overflowed: {squared_sum}"

    def test_collect_seed_gradients_only_async_handles_extreme_values(self):
        """collect_seed_gradients_only_async should not overflow with 1e20 gradients."""
        model = nn.Linear(10, 5)

        for param in model.parameters():
            param.grad = torch.full_like(param, 1e20)

        async_stats = collect_seed_gradients_only_async(model.parameters())

        squared_sum = async_stats["_seed_squared_sum"]
        assert torch.isfinite(squared_sum), f"Seed squared sum overflowed: {squared_sum}"

    def test_normal_gradients_still_work(self):
        """Verify normal gradient values produce correct results."""
        model = nn.Linear(2, 1, bias=False)
        # 3-4-5 triangle: sqrt(3^2 + 4^2) = 5
        model.weight.grad = torch.tensor([[3.0, 4.0]])

        stats = collect_seed_gradients(model.parameters())

        assert math.isclose(stats["gradient_norm"], 5.0, rel_tol=1e-5)
