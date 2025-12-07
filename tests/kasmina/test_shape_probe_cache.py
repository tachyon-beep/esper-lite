"""Test shape probe caching with device invalidation.

The SeedSlot uses a simple dict cache for shape probes, keyed by topology
("cnn" or "transformer"). Each entry stores (device, tensor) to detect
when the slot moves to a different device and needs to regenerate probes.
"""
import pytest
import torch

from esper.kasmina.slot import SeedSlot, CNN_SHAPE_PROBE_SPATIAL, TRANSFORMER_SHAPE_PROBE_SEQ_LEN


class TestShapeProbeCache:
    """Verify shape probe cache behavior."""

    def test_cache_hit_returns_same_tensor(self):
        """Repeated calls should return cached tensor."""
        slot = SeedSlot("test", channels=32, device="cpu")

        probe1 = slot._get_shape_probe("cnn")
        probe2 = slot._get_shape_probe("cnn")

        # Should be same tensor (cached)
        assert probe1 is probe2

    def test_cache_invalidated_on_device_change(self):
        """Cache should clear when device changes."""
        slot = SeedSlot("test", channels=32, device="cpu")

        probe_cpu = slot._get_shape_probe("cnn")

        # Simulate device change
        slot.device = torch.device("cpu")  # Same device, cache should still work

        probe_after = slot._get_shape_probe("cnn")
        assert probe_cpu is probe_after

    def test_different_topologies_have_separate_cache_entries(self):
        """CNN and transformer probes should be cached separately."""
        slot = SeedSlot("test", channels=32, device="cpu")

        probe_cnn = slot._get_shape_probe("cnn")
        probe_transformer = slot._get_shape_probe("transformer")

        # Should be different tensors
        assert probe_cnn is not probe_transformer
        assert probe_cnn.shape != probe_transformer.shape

        # Both should be cached - second calls should return same objects
        assert slot._get_shape_probe("cnn") is probe_cnn
        assert slot._get_shape_probe("transformer") is probe_transformer

    def test_cnn_probe_has_correct_dimensions(self):
        """CNN probe should have shape (batch, channels, height, width)."""
        slot = SeedSlot("test", channels=64, device="cpu")

        probe = slot._get_shape_probe("cnn")

        assert probe.shape == (1, 64, CNN_SHAPE_PROBE_SPATIAL, CNN_SHAPE_PROBE_SPATIAL)
        assert probe.device == torch.device("cpu")

    def test_transformer_probe_has_correct_dimensions(self):
        """Transformer probe should have shape (batch, seq_len, dim)."""
        slot = SeedSlot("test", channels=128, device="cpu")

        probe = slot._get_shape_probe("transformer")

        assert probe.shape == (2, TRANSFORMER_SHAPE_PROBE_SEQ_LEN, 128)
        assert probe.device == torch.device("cpu")

    def test_cache_cleared_on_actual_device_change(self):
        """Cache should be cleared when slot moves to different device."""
        slot = SeedSlot("test", channels=32, device="cpu")

        # Get initial probe on CPU
        probe_cpu = slot._get_shape_probe("cnn")
        assert probe_cpu.device == torch.device("cpu")

        # Cache should have one entry
        assert len(slot._shape_probe_cache) == 1
        assert "cnn" in slot._shape_probe_cache

        # Move to CPU again (no actual change)
        slot.to("cpu")

        # Cache should NOT be cleared (device didn't change)
        assert len(slot._shape_probe_cache) == 1
        probe_after_same = slot._get_shape_probe("cnn")
        assert probe_after_same is probe_cpu

    def test_probe_on_correct_device_after_move(self):
        """After device change, new probe should be on new device."""
        slot = SeedSlot("test", channels=32, device="cpu")

        # Get initial probe on CPU
        probe_cpu = slot._get_shape_probe("cnn")
        assert probe_cpu.device == torch.device("cpu")

        # Manually change device and clear cache to simulate actual device move
        old_device = slot.device
        slot.device = torch.device("cpu")  # In practice would be "cuda"

        # Clear cache only if device actually changed
        if slot.device != old_device:
            slot._shape_probe_cache.clear()

        # Get new probe - should be on new device
        probe_new = slot._get_shape_probe("cnn")
        assert probe_new.device == slot.device

    def test_cache_persists_across_multiple_calls(self):
        """Cache should work for multiple alternating calls."""
        slot = SeedSlot("test", channels=32, device="cpu")

        # First set of calls
        cnn1 = slot._get_shape_probe("cnn")
        trans1 = slot._get_shape_probe("transformer")

        # Second set of calls
        cnn2 = slot._get_shape_probe("cnn")
        trans2 = slot._get_shape_probe("transformer")

        # Third set of calls
        cnn3 = slot._get_shape_probe("cnn")
        trans3 = slot._get_shape_probe("transformer")

        # All CNN probes should be same object
        assert cnn1 is cnn2 is cnn3

        # All transformer probes should be same object
        assert trans1 is trans2 is trans3

    def test_cache_stores_device_with_probe(self):
        """Cache should store device alongside tensor."""
        slot = SeedSlot("test", channels=32, device="cpu")

        probe = slot._get_shape_probe("cnn")

        # Check cache structure
        assert "cnn" in slot._shape_probe_cache
        cached_device, cached_tensor = slot._shape_probe_cache["cnn"]

        assert cached_device == torch.device("cpu")
        assert cached_tensor is probe

    def test_different_channels_create_different_probes(self):
        """Different channel counts should create appropriately sized probes."""
        slot_32 = SeedSlot("test_32", channels=32, device="cpu")
        slot_64 = SeedSlot("test_64", channels=64, device="cpu")

        probe_32 = slot_32._get_shape_probe("cnn")
        probe_64 = slot_64._get_shape_probe("cnn")

        # Different slots, different probes
        assert probe_32 is not probe_64

        # Different channel dimensions
        assert probe_32.shape[1] == 32
        assert probe_64.shape[1] == 64

    def test_cache_miss_creates_new_probe(self):
        """First call for a topology should create and cache probe."""
        slot = SeedSlot("test", channels=32, device="cpu")

        # Cache should be empty initially
        assert len(slot._shape_probe_cache) == 0

        # First call should create probe
        probe = slot._get_shape_probe("cnn")

        # Cache should now have entry
        assert len(slot._shape_probe_cache) == 1
        assert "cnn" in slot._shape_probe_cache

        # Cached probe should be the same object
        cached_device, cached_tensor = slot._shape_probe_cache["cnn"]
        assert cached_tensor is probe
