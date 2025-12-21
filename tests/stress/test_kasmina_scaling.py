"""Stress tests for Kasmina scaling.

Tests verify that Kasmina handles scaling scenarios efficiently:
- Large host memory usage
- Many sequential germinate/cull cycles
- Rapid stage transitions
- torch.compile overhead
- Gradient health monitor overhead

These tests are marked with @pytest.mark.stress and excluded by default.
Run with: pytest -m stress
"""

import gc
import time
import tracemalloc

import pytest
import torch
import torch.nn as nn

from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel
from esper.kasmina.slot import SeedSlot, SeedState
from esper.kasmina.isolation import blend_with_isolation, GradientHealthMonitor
from esper.leyline import SeedStage


@pytest.mark.stress
class TestLargeHostMemory:
    """Tests for memory usage with large host models."""

    def test_medium_cnn_host_memory_reasonable(self):
        """Medium CNN host should use reasonable memory."""
        gc.collect()
        tracemalloc.start()

        # Create medium CNN host (not too large)
        host = CNNHost(
            n_blocks=5,
            base_channels=64,
            memory_format=torch.contiguous_format,
        )

        # Count parameters
        param_count = sum(p.numel() for p in host.parameters())

        # Create model with 3 slots
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        # Germinate seeds
        model.germinate_seed("norm", "seed_0", slot="r0c0")
        model.germinate_seed("norm", "seed_1", slot="r0c1")
        model.germinate_seed("norm", "seed_2", slot="r0c2")

        # Force allocation with forward pass
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            _ = model(x)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak memory should be reasonable (< 500MB)
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500, f"Peak memory {peak_mb:.1f}MB exceeds 500MB limit"

    def test_transformer_host_memory_scales_with_layers(self):
        """Transformer host memory should scale linearly with layers."""
        # n_layer must be divisible by num_segments (default 3)
        layer_counts = [3, 6, 9]
        memory_usage = []

        for n_layers in layer_counts:
            gc.collect()
            tracemalloc.start()

            host = TransformerHost(
                n_layer=n_layers,
                n_embd=128,
                n_head=4,
                block_size=64,
                vocab_size=500,
            )

            # Forward pass to allocate
            x = torch.randint(0, 500, (2, 32))
            with torch.no_grad():
                _ = host(x)

            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_usage.append((n_layers, peak / (1024 * 1024)))

        # Memory should scale roughly linearly with layers
        # 9 layers should be less than 4x the memory of 3 layers
        mem_3 = memory_usage[0][1]
        mem_9 = memory_usage[2][1]

        assert mem_9 < mem_3 * 4, (
            f"9-layer memory ({mem_9:.1f}MB) > 4x 3-layer memory ({mem_3:.1f}MB)"
        )


@pytest.mark.stress
class TestGerminateCullCycles:
    """Tests for memory stability during germinate/cull cycles."""

    def test_sequential_germinate_cull_no_memory_leak(self):
        """25 germinate/cull cycles should not leak memory."""
        n_cycles = 25

        gc.collect()
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        slot = SeedSlot(slot_id="r0c0", channels=64)

        for i in range(n_cycles):
            # Germinate
            slot.germinate("noop", seed_id=f"seed_{i}")
            slot.step_epoch()  # -> TRAINING

            # Record some metrics
            slot.state.metrics.record_accuracy(50.0 + i * 0.1)

            # Cull
            slot.cull()

            # Check memory every 10 cycles
            if i % 10 == 9:
                gc.collect()

        gc.collect()
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        # Memory growth should be bounded (< 10MB for 25 cycles)
        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        assert memory_growth_mb < 10, (
            f"Memory grew by {memory_growth_mb:.1f}MB (>10MB leak suspected)"
        )

    def test_multi_slot_cycling_no_memory_leak(self):
        """Cycling through multiple slots should not leak memory."""
        n_cycles = 10

        gc.collect()
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0", "r0c1", "r0c2"])

        for cycle in range(n_cycles):
            # Germinate all slots
            for slot_id in ["r0c0", "r0c1", "r0c2"]:
                model.germinate_seed("noop", f"seed_{cycle}_{slot_id}", slot=slot_id)

            # Forward pass
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                _ = model(x)

            # Cull all slots
            for slot_id in ["r0c0", "r0c1", "r0c2"]:
                model.seed_slots[slot_id].cull()

            if cycle % 5 == 4:
                gc.collect()

        gc.collect()
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
        assert memory_growth_mb < 20, (
            f"Memory grew by {memory_growth_mb:.1f}MB during multi-slot cycling"
        )


@pytest.mark.stress
class TestRapidStageTransitions:
    """Tests for rapid stage transition performance."""

    def test_100_step_epochs_stable(self):
        """100 step_epoch calls should complete quickly and stably."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        start = time.perf_counter()

        for i in range(100):
            slot.step_epoch()

            # Record accuracy to advance stages
            if slot.state and slot.state.stage == SeedStage.TRAINING:
                slot.state.metrics.record_accuracy(50.0 + i * 0.01)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete in < 200ms
        assert elapsed_ms < 200, f"100 step_epoch calls took {elapsed_ms:.1f}ms (>200ms)"

    def test_stage_transition_throughput(self):
        """Stage transitions should achieve reasonable throughput."""
        from esper.leyline import DEFAULT_GRADIENT_RATIO_THRESHOLD, DEFAULT_MIN_BLENDING_EPOCHS

        n_transitions = 20
        slot = SeedSlot(slot_id="r0c0", channels=64)

        start = time.perf_counter()

        for i in range(n_transitions):
            # Germinate
            slot.germinate("noop", seed_id=f"seed_{i}")
            assert slot.state.stage == SeedStage.GERMINATED

            # -> TRAINING
            slot.step_epoch()
            assert slot.state.stage == SeedStage.TRAINING

            # Setup for BLENDING
            slot.state.metrics.record_accuracy(50.0)
            slot.state.metrics.record_accuracy(60.0)
            slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1
            slot.state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

            # -> BLENDING (via gate)
            slot.state.transition(SeedStage.BLENDING)
            slot.start_blending(total_steps=5)
            assert slot.state.stage == SeedStage.BLENDING

            # -> HOLDING
            slot.state.transition(SeedStage.HOLDING)
            assert slot.state.stage == SeedStage.HOLDING

            # Cull
            slot.cull()

        elapsed_s = time.perf_counter() - start
        transitions_per_sec = n_transitions / elapsed_s

        # Should achieve at least 50 full cycles per second
        assert transitions_per_sec > 50, (
            f"Stage transitions only {transitions_per_sec:.0f}/s (need >50/s)"
        )


@pytest.mark.stress
class TestGradientHealthMonitorOverhead:
    """Tests for GradientHealthMonitor overhead."""

    def test_monitor_overhead_reasonable(self):
        """GradientHealthMonitor overhead should be < 30% per forward/backward."""
        host = CNNHost(n_blocks=3, base_channels=32, memory_format=torch.contiguous_format)
        model = MorphogeneticModel(host, slots=["r0c0"])
        model.germinate_seed("noop", "seed_0", slot="r0c0")
        model.seed_slots["r0c0"].step_epoch()  # -> TRAINING

        x = torch.randn(4, 3, 32, 32)
        criterion = nn.MSELoss()

        # Without monitor
        n_iterations = 20

        without_start = time.perf_counter()
        for _ in range(n_iterations):
            output = model(x)
            # Create target matching output shape
            target = torch.randn_like(output)
            loss = criterion(output, target)
            loss.backward()
            model.zero_grad()
        without_time = (time.perf_counter() - without_start) * 1000

        # With monitor
        monitor = GradientHealthMonitor()
        slot = model.seed_slots["r0c0"]
        if slot.seed is not None:
            monitor.register(model.host, slot.seed)

        with_start = time.perf_counter()
        for _ in range(n_iterations):
            output = model(x)
            target = torch.randn_like(output)
            loss = criterion(output, target)
            loss.backward()

            # Compute gradient health
            _ = monitor.compute_gradient_health()

            model.zero_grad()
        with_time = (time.perf_counter() - with_start) * 1000

        # Overhead should be < 30%
        overhead = (with_time - without_time) / without_time
        assert overhead < 0.30, (
            f"Monitor overhead {overhead:.1%} (>30%)"
        )

    def test_monitor_compute_fast(self):
        """Individual compute_gradient_health calls should be fast."""
        # Create mock modules with parameters
        host = nn.Linear(32, 32)
        seed = nn.Linear(32, 32)

        # Do a forward/backward to generate gradients
        x = torch.randn(4, 32)
        y = host(x) + seed(x)
        y.sum().backward()

        monitor = GradientHealthMonitor()
        monitor.register(host, seed)

        # Time computes
        n_computes = 100
        start = time.perf_counter()

        for _ in range(n_computes):
            _ = monitor.compute_gradient_health()

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_compute_us = (elapsed_ms * 1000) / n_computes

        # Each compute should be < 500 microseconds
        assert per_compute_us < 500, (
            f"Compute took {per_compute_us:.1f}us (>500us)"
        )


@pytest.mark.stress
class TestBlendingPerformance:
    """Tests for blending operation performance."""

    def test_blend_operation_fast(self):
        """blend_with_isolation should be fast for typical tensor sizes."""
        host = torch.randn(8, 64, 16, 16)
        seed = torch.randn(8, 64, 16, 16)
        alpha = torch.tensor(0.5)

        # Warmup
        for _ in range(5):
            _ = blend_with_isolation(host, seed, alpha)

        # Time
        n_iterations = 100
        start = time.perf_counter()

        for _ in range(n_iterations):
            _ = blend_with_isolation(host, seed, alpha)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_blend_us = (elapsed_ms * 1000) / n_iterations

        # Each blend should be < 200 microseconds for this size
        assert per_blend_us < 200, (
            f"Blend took {per_blend_us:.1f}us (>200us)"
        )

    def test_blend_scales_with_tensor_size(self):
        """Blend time should scale linearly with tensor size."""
        sizes = [(2, 32, 8, 8), (4, 32, 16, 16), (8, 32, 16, 16)]
        times = []

        alpha = torch.tensor(0.5)

        for size in sizes:
            host = torch.randn(size)
            seed = torch.randn(size)

            # Warmup
            for _ in range(3):
                _ = blend_with_isolation(host, seed, alpha)

            # Time
            n_iterations = 50
            start = time.perf_counter()

            for _ in range(n_iterations):
                _ = blend_with_isolation(host, seed, alpha)

            elapsed_ms = (time.perf_counter() - start) * 1000 / n_iterations
            times.append((size, elapsed_ms))

        # Largest should be < 16x smallest (proportional to element count)
        time_small = times[0][1]
        time_large = times[2][1]

        assert time_large < time_small * 16, (
            f"Large blend ({time_large:.2f}ms) > 16x small blend ({time_small:.2f}ms)"
        )
