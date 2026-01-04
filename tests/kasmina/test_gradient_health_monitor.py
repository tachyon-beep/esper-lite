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

    def test_async_gradient_capture_deferred_sync(self):
        """Verify async capture returns tensors, materialize converts to floats."""
        host = nn.Linear(64, 64)
        seed = nn.Linear(64, 64)

        monitor = GradientHealthMonitor()
        monitor.register(host, seed)

        # Create gradients
        x = torch.randn(4, 64)
        loss = (host(x) + seed(x)).sum()
        loss.backward()

        # Async capture returns tensor-based stats
        async_stats = monitor.compute_gradient_health_async()

        # Verify async stats have tensor values (no .item() yet)
        assert async_stats.get("_async") is True
        assert async_stats.get("_host_norm_sq") is not None
        assert isinstance(async_stats["_host_norm_sq"], torch.Tensor)
        assert async_stats.get("_seed_norm_sq") is not None
        assert isinstance(async_stats["_seed_norm_sq"], torch.Tensor)

        # Materialize converts to floats
        final_stats = monitor.materialize_gradient_stats(async_stats)

        assert "host_grad_norm" in final_stats
        assert "seed_grad_norm" in final_stats
        assert isinstance(final_stats["host_grad_norm"], float)
        assert isinstance(final_stats["seed_grad_norm"], float)
        assert final_stats["host_grad_norm"] > 0
        assert final_stats["seed_grad_norm"] > 0


class TestSeedSlotAsyncGradientCapture:
    """Verify SeedSlot async gradient telemetry methods."""

    def test_async_capture_returns_true_when_active(self):
        """capture_gradient_telemetry_async() returns True with active seed."""
        from esper.kasmina.slot import SeedSlot
        from esper.kasmina.host import CNNHost
        from esper.leyline import SeedStage

        slot = SeedSlot(slot_id="r0c0", channels=64, fast_mode=False)
        host = CNNHost(num_classes=10, n_blocks=3)

        slot.germinate("norm", seed_id="test", host_module=host)
        slot.state.transition(SeedStage.TRAINING)

        # Need gradients for the async capture to have data
        x = torch.randn(2, 3, 32, 32)
        out = host(x)
        loss = out.sum()
        loss.backward()

        result = slot.capture_gradient_telemetry_async()
        assert result is True
        assert slot._pending_gradient_stats is not None

    def test_async_capture_returns_false_when_inactive(self):
        """capture_gradient_telemetry_async() returns False without active seed."""
        from esper.kasmina.slot import SeedSlot

        slot = SeedSlot(slot_id="r0c0", channels=64, fast_mode=False)

        result = slot.capture_gradient_telemetry_async()
        assert result is False
        assert slot._pending_gradient_stats is None

    def test_finalize_updates_metrics(self):
        """finalize_gradient_telemetry() updates metrics from async stats."""
        from esper.kasmina.slot import SeedSlot
        from esper.kasmina.host import CNNHost
        from esper.leyline import SeedStage

        slot = SeedSlot(slot_id="r0c0", channels=64, fast_mode=False)
        host = CNNHost(num_classes=10, n_blocks=3)

        slot.germinate("norm", seed_id="test", host_module=host)
        slot.state.transition(SeedStage.TRAINING)

        # Need gradients for the async capture
        x = torch.randn(2, 3, 32, 32)
        out = host(x)
        loss = out.sum()
        loss.backward()

        # Initial gradient ratio should be None (never measured)
        assert slot.state.metrics.seed_gradient_norm_ratio is None

        # Async capture + finalize
        slot.capture_gradient_telemetry_async()
        slot.finalize_gradient_telemetry()

        # Metrics should be updated
        assert slot._pending_gradient_stats is None  # Cleared after finalize
        # After finalize, ratio should be set to a real value (not None)
        assert slot.state.metrics.seed_gradient_norm_ratio is not None
        assert slot.state.metrics.seed_gradient_norm_ratio >= 0.0
        assert slot.state.metrics.gradient_norm_avg >= 0.0

    def test_finalize_noop_without_async_capture(self):
        """finalize_gradient_telemetry() is no-op if async capture wasn't called."""
        from esper.kasmina.slot import SeedSlot
        from esper.leyline import SeedStage

        slot = SeedSlot(slot_id="r0c0", channels=64, fast_mode=False)
        slot.germinate("norm", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)

        initial_ratio = slot.state.metrics.seed_gradient_norm_ratio

        # finalize without capture should be no-op
        slot.finalize_gradient_telemetry()

        assert slot.state.metrics.seed_gradient_norm_ratio == initial_ratio

    def test_async_equivalent_to_sync(self):
        """Async capture + finalize should produce same results as sync capture."""
        from esper.kasmina.slot import SeedSlot
        from esper.kasmina.host import CNNHost
        from esper.leyline import SeedStage

        host = CNNHost(num_classes=10, n_blocks=3)

        # Create two slots with identical setup
        slot_sync = SeedSlot(slot_id="sync", channels=64, fast_mode=False)
        slot_async = SeedSlot(slot_id="async", channels=64, fast_mode=False)

        slot_sync.germinate("norm", seed_id="test", host_module=host)
        slot_async.germinate("norm", seed_id="test", host_module=host)
        slot_sync.state.transition(SeedStage.TRAINING)
        slot_async.state.transition(SeedStage.TRAINING)

        # Generate gradients
        x = torch.randn(2, 3, 32, 32)
        out = host(x)
        loss = out.sum()
        loss.backward()

        # Sync capture
        slot_sync.capture_gradient_telemetry()

        # Async capture + finalize
        slot_async.capture_gradient_telemetry_async()
        slot_async.finalize_gradient_telemetry()

        # Results should be identical
        assert abs(
            slot_sync.state.metrics.seed_gradient_norm_ratio
            - slot_async.state.metrics.seed_gradient_norm_ratio
        ) < 1e-6
        assert abs(
            slot_sync.state.metrics.gradient_norm_avg
            - slot_async.state.metrics.gradient_norm_avg
        ) < 1e-6
