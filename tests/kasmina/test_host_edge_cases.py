"""Edge case tests for Kasmina Host modules.

Tests verify correct behavior at boundary conditions:
- CNNHost with prime channel counts (GroupNorm adaptation)
- Tiny and large spatial dimensions
- Transformer sequence length edge cases
- Unknown segment handling
"""

import pytest
import torch

from esper.kasmina.host import CNNHost, TransformerHost


class TestCNNPrimeChannels:
    """Tests for CNNHost with prime channel counts (GroupNorm adaptation)."""

    @pytest.mark.parametrize("prime_channels", [31, 37, 41, 47, 53])
    def test_cnn_prime_base_channels(self, prime_channels: int):
        """CNNHost should work with prime base_channels (GroupNorm adapts to num_groups=1)."""
        host = CNNHost(
            n_blocks=2,
            base_channels=prime_channels,
            memory_format=torch.contiguous_format,
        )

        # Standard CIFAR-10 input shape
        x = torch.randn(2, 3, 32, 32)
        output = host(x)

        # Output should have correct shape
        assert output.shape == (2, 10)  # default num_classes=10

    def test_cnn_channel_1_extreme(self):
        """CNNHost with base_channels=1 should work."""
        host = CNNHost(
            n_blocks=2,
            base_channels=1,
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(2, 3, 32, 32)
        output = host(x)

        # Output should have correct shape
        assert output.shape == (2, 10)


class TestCNNSpatialDimensions:
    """Tests for CNNHost with various spatial dimensions."""

    def test_cnn_tiny_spatial_4x4(self):
        """CNNHost should work with 4x4 input (minimum for 2 pools)."""
        host = CNNHost(
            n_blocks=2,
            base_channels=32,
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(2, 3, 4, 4)
        output = host(x)

        assert output.shape == (2, 10)

    def test_cnn_tiny_spatial_2x2(self):
        """CNNHost should work with 2x2 input (1 pool produces 1x1)."""
        host = CNNHost(
            n_blocks=2,
            base_channels=32,
            pool_layers=1,  # Only pool once
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(2, 3, 2, 2)
        output = host(x)

        assert output.shape == (2, 10)

    def test_cnn_tiny_spatial_1x1_no_pool(self):
        """CNNHost should work with 1x1 input when pooling is disabled."""
        host = CNNHost(
            n_blocks=2,
            base_channels=32,
            pool_layers=0,  # No pooling
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(2, 3, 1, 1)
        output = host(x)

        assert output.shape == (2, 10)

    @pytest.mark.slow
    def test_cnn_large_spatial_256x256(self):
        """CNNHost should work with large 256x256 input."""
        host = CNNHost(
            n_blocks=3,
            base_channels=16,  # Smaller channels to limit memory
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(1, 3, 256, 256)
        output = host(x)

        assert output.shape == (1, 10)

    def test_cnn_non_square_spatial(self):
        """CNNHost should work with non-square input."""
        host = CNNHost(
            n_blocks=2,
            base_channels=32,
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(2, 3, 32, 64)  # Non-square
        output = host(x)

        assert output.shape == (2, 10)


class TestCNNBlockVariations:
    """Tests for CNNHost with various block configurations."""

    def test_cnn_minimum_blocks(self):
        """CNNHost requires minimum 2 blocks."""
        with pytest.raises(ValueError, match="at least 2 blocks"):
            CNNHost(n_blocks=1)

    def test_cnn_exactly_2_blocks(self):
        """CNNHost should work with exactly 2 blocks."""
        host = CNNHost(
            n_blocks=2,
            base_channels=32,
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(2, 3, 32, 32)
        output = host(x)

        assert output.shape == (2, 10)
        assert len(host.injection_specs()) == 2

    def test_cnn_many_blocks_limited_pool(self):
        """CNNHost with many blocks but limited pooling (deep network on small images)."""
        host = CNNHost(
            n_blocks=6,
            base_channels=16,
            pool_layers=3,  # Only pool first 3 blocks
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(2, 3, 32, 32)
        output = host(x)

        assert output.shape == (2, 10)
        assert len(host.injection_specs()) == 6


class TestTransformerSequenceLength:
    """Tests for TransformerHost sequence length edge cases."""

    def test_transformer_seq_len_1(self):
        """TransformerHost should work with sequence length = 1."""
        host = TransformerHost(
            vocab_size=1000,
            n_embd=64,
            n_layer=3,
            n_head=2,
            block_size=256,
            num_segments=3,
        )

        x = torch.randint(0, 1000, (2, 1))  # Single token
        output = host(x)

        assert output.shape == (2, 1, 1000)

    def test_transformer_seq_len_max_block_size(self):
        """TransformerHost should work with seq_len = block_size (maximum)."""
        block_size = 32  # Use small block_size for test speed
        host = TransformerHost(
            vocab_size=1000,
            n_embd=64,
            n_layer=3,
            n_head=2,
            block_size=block_size,
            num_segments=3,
        )

        x = torch.randint(0, 1000, (2, block_size))  # Exactly block_size
        output = host(x)

        assert output.shape == (2, block_size, 1000)

    def test_transformer_seq_len_exceed_raises(self):
        """TransformerHost should raise ValueError when seq_len > block_size."""
        block_size = 32
        host = TransformerHost(
            vocab_size=1000,
            n_embd=64,
            n_layer=3,
            n_head=2,
            block_size=block_size,
            num_segments=3,
        )

        x = torch.randint(0, 1000, (2, block_size + 1))  # Exceed block_size

        with pytest.raises(ValueError, match="exceeds block_size"):
            host(x)

    def test_transformer_forward_to_segment_exceed_raises(self):
        """forward_to_segment should also raise ValueError when seq_len > block_size."""
        block_size = 32
        host = TransformerHost(
            vocab_size=1000,
            n_embd=64,
            n_layer=3,
            n_head=2,
            block_size=block_size,
            num_segments=3,
        )

        x = torch.randint(0, 1000, (2, block_size + 1))

        with pytest.raises(ValueError, match="exceeds block_size"):
            host.forward_to_segment("r0c0", x)


class TestTransformerLayerDivisibility:
    """Tests for TransformerHost layer divisibility requirement."""

    def test_transformer_n_layer_not_divisible_raises(self):
        """TransformerHost should raise ValueError if n_layer not divisible by num_segments."""
        with pytest.raises(ValueError, match="divisible"):
            TransformerHost(
                n_layer=5,
                num_segments=3,  # 5 % 3 != 0
            )

    def test_transformer_n_layer_divisible_works(self):
        """TransformerHost should work when n_layer is divisible by num_segments."""
        host = TransformerHost(
            n_layer=6,
            num_segments=3,  # 6 % 3 == 0
            n_embd=64,
            n_head=2,
        )

        x = torch.randint(0, 50257, (2, 8))
        output = host(x)

        assert output.shape == (2, 8, 50257)


class TestHostUnknownSegment:
    """Tests for handling unknown segment names."""

    def test_cnn_forward_to_unknown_segment_raises(self):
        """CNNHost.forward_to_segment should raise ValueError for unknown segment."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        x = torch.randn(2, 3, 32, 32)

        with pytest.raises(ValueError, match="Unknown segment"):
            host.forward_to_segment("invalid_segment", x)

    def test_cnn_forward_from_unknown_segment_raises(self):
        """CNNHost.forward_from_segment should raise ValueError for unknown segment."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)

        # Get valid features first
        x = torch.randn(2, 3, 32, 32)
        features = host.forward_to_segment("r0c0", x)

        with pytest.raises(ValueError, match="Unknown segment"):
            host.forward_from_segment("invalid_segment", features)

    def test_transformer_forward_to_unknown_segment_raises(self):
        """TransformerHost.forward_to_segment should raise ValueError for unknown segment."""
        host = TransformerHost(n_layer=6, num_segments=3, n_embd=64, n_head=2)
        x = torch.randint(0, 50257, (2, 8))

        with pytest.raises(ValueError, match="Unknown segment"):
            host.forward_to_segment("invalid_segment", x)

    def test_transformer_forward_from_unknown_segment_raises(self):
        """TransformerHost.forward_from_segment should raise ValueError for unknown segment."""
        host = TransformerHost(n_layer=6, num_segments=3, n_embd=64, n_head=2)
        x = torch.randint(0, 50257, (2, 8))

        # Get valid features first
        features = host.forward_to_segment("r0c0", x)

        with pytest.raises(ValueError, match="Unknown segment"):
            host.forward_from_segment("invalid_segment", features)


# TestHostRegistrationErrors removed - hosts no longer have register_slot/unregister_slot


class TestHostOutputConsistency:
    """Tests for output consistency across different paths."""

    def test_cnn_forward_vs_segment_forward(self):
        """Full forward should match segment-by-segment forward."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        host.eval()

        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            # Full forward
            full_output = host(x)

            # Segment-by-segment (through last segment)
            segments = list(host.segment_channels.keys())
            last_segment = segments[-1]

            partial = host.forward_to_segment(last_segment, x)
            partial_output = host.forward_from_segment(last_segment, partial)

            torch.testing.assert_close(partial_output, full_output)

    def test_transformer_forward_vs_segment_forward(self):
        """Full forward should match segment-by-segment forward."""
        host = TransformerHost(n_layer=6, num_segments=3, n_embd=64, n_head=2)
        host.eval()

        x = torch.randint(0, 50257, (2, 8))

        with torch.no_grad():
            # Full forward
            full_output = host(x)

            # Segment-by-segment (through last segment)
            segments = list(host.segment_channels.keys())
            last_segment = segments[-1]

            partial = host.forward_to_segment(last_segment, x)
            partial_output = host.forward_from_segment(last_segment, partial)

            torch.testing.assert_close(partial_output, full_output)


class TestForwardToSegmentBackwardsRouting:
    """Test that backwards segment routing raises ValueError."""

    def test_cnn_backwards_routing_raises(self):
        """Attempting to route backwards (from later to earlier segment) should raise."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        x = torch.randn(2, 3, 32, 32)

        # Forward to r0c2 first
        h = host.forward_to_segment("r0c2", x)

        # Attempt backwards routing should raise
        with pytest.raises(ValueError, match="Cannot route backwards"):
            host.forward_to_segment("r0c0", h, from_segment="r0c2")

    def test_cnn_backwards_routing_same_segment_raises(self):
        """Routing from a segment to itself should raise (range would be empty)."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        x = torch.randn(2, 3, 32, 32)

        # Forward to r0c1 first
        h = host.forward_to_segment("r0c1", x)

        # Attempt same-segment routing should raise
        with pytest.raises(ValueError, match="Cannot route backwards"):
            host.forward_to_segment("r0c1", h, from_segment="r0c1")

    def test_cnn_forward_routing_works(self):
        """Forward routing (earlier to later segment) should work."""
        host = CNNHost(n_blocks=3, memory_format=torch.contiguous_format)
        x = torch.randn(2, 3, 32, 32)

        # Forward to r0c0 first
        h = host.forward_to_segment("r0c0", x)

        # Forward routing to r0c2 should work
        h2 = host.forward_to_segment("r0c2", h, from_segment="r0c0")
        assert h2.shape[0] == 2  # Batch dimension preserved

    def test_transformer_backwards_routing_raises(self):
        """Attempting to route backwards (from later to earlier segment) should raise."""
        host = TransformerHost(
            n_layer=6,
            num_segments=3,
            n_embd=64,
            n_head=2,
        )
        x = torch.randint(0, 50257, (2, 16))

        # Forward to r0c2 first
        h = host.forward_to_segment("r0c2", x)

        # Attempt backwards routing should raise
        with pytest.raises(ValueError, match="Cannot route backwards"):
            host.forward_to_segment("r0c0", h, from_segment="r0c2")

    def test_transformer_backwards_routing_same_segment_raises(self):
        """Routing from a segment to itself should raise (range would be empty)."""
        host = TransformerHost(
            n_layer=6,
            num_segments=3,
            n_embd=64,
            n_head=2,
        )
        x = torch.randint(0, 50257, (2, 16))

        # Forward to r0c1 first
        h = host.forward_to_segment("r0c1", x)

        # Attempt same-segment routing should raise
        with pytest.raises(ValueError, match="Cannot route backwards"):
            host.forward_to_segment("r0c1", h, from_segment="r0c1")

    def test_transformer_forward_routing_works(self):
        """Forward routing (earlier to later segment) should work."""
        host = TransformerHost(
            n_layer=6,
            num_segments=3,
            n_embd=64,
            n_head=2,
        )
        x = torch.randint(0, 50257, (2, 16))

        # Forward to r0c0 first
        h = host.forward_to_segment("r0c0", x)

        # Forward routing to r0c2 should work
        h2 = host.forward_to_segment("r0c2", h, from_segment="r0c0")
        assert h2.shape == (2, 16, 64)  # (batch, seq_len, n_embd)


class TestSeedMetricsSerialization:
    """Test SeedMetrics serialization contract enforcement."""

    def test_missing_schema_version_raises(self):
        """SeedMetrics.from_dict should raise KeyError if _schema_version is missing."""
        from esper.kasmina.slot import SeedMetrics

        metrics = SeedMetrics()
        data = metrics.to_dict()
        del data["_schema_version"]

        with pytest.raises(KeyError):
            SeedMetrics.from_dict(data)

    def test_wrong_schema_version_raises(self):
        """SeedMetrics.from_dict should raise ValueError on version mismatch."""
        from esper.kasmina.slot import SeedMetrics

        metrics = SeedMetrics()
        data = metrics.to_dict()
        data["_schema_version"] = 9999  # Invalid version

        with pytest.raises(ValueError, match="schema version mismatch"):
            SeedMetrics.from_dict(data)

    def test_round_trip_preserves_data(self):
        """SeedMetrics should round-trip through to_dict/from_dict."""
        from esper.kasmina.slot import SeedMetrics

        original = SeedMetrics()
        original.epochs_total = 42
        original.current_val_accuracy = 0.85
        original.gradient_norm_avg = 1.5

        data = original.to_dict()
        restored = SeedMetrics.from_dict(data)

        assert restored.epochs_total == 42
        assert restored.current_val_accuracy == 0.85
        assert restored.gradient_norm_avg == 1.5
