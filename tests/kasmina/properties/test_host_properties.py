"""Property-based tests for Kasmina Host invariants.

These tests use Hypothesis to verify critical invariants hold for
CNNHost and TransformerHost:

1. segment_channels values are always positive
2. segment_channels[slot_id] matches actual tensor dimensions
3. forward_to_segment + forward_from_segment = full forward
4. Host output shape matches expected dimensions
"""

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.kasmina.host import CNNHost, TransformerHost


class TestSegmentChannelsPositive:
    """All segment channel values must be positive."""

    @given(
        n_blocks=st.integers(min_value=2, max_value=6),
        base_channels=st.integers(min_value=8, max_value=64),
    )
    @settings(max_examples=50)
    def test_cnn_segment_channels_all_positive(self, n_blocks: int, base_channels: int):
        """CNNHost.segment_channels values are all > 0."""
        host = CNNHost(
            n_blocks=n_blocks,
            base_channels=base_channels,
            memory_format=torch.contiguous_format,  # Skip channels_last for simpler tests
        )

        for slot_id, channels in host.segment_channels.items():
            assert channels > 0, f"Slot {slot_id} has non-positive channels: {channels}"

    @given(
        n_embd=st.sampled_from([64, 128, 256, 384]),
        n_layer=st.sampled_from([3, 6, 9, 12]),
        num_segments=st.sampled_from([1, 3]),
    )
    @settings(max_examples=30)
    def test_transformer_segment_channels_all_positive(
        self, n_embd: int, n_layer: int, num_segments: int
    ):
        """TransformerHost.segment_channels values are all > 0."""
        # Ensure n_layer divisible by num_segments
        assume(n_layer % num_segments == 0)

        host = TransformerHost(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=max(1, n_embd // 64),  # At least 1 head
            num_segments=num_segments,
        )

        for slot_id, channels in host.segment_channels.items():
            assert channels > 0, f"Slot {slot_id} has non-positive channels: {channels}"
            # For transformer, all segments have same embedding dimension
            assert channels == n_embd


class TestSegmentChannelsMatchTensors:
    """segment_channels[slot_id] must match actual tensor dimensions."""

    @given(
        n_blocks=st.integers(min_value=2, max_value=4),
        base_channels=st.sampled_from([16, 32]),
    )
    @settings(max_examples=20)
    def test_cnn_segment_channels_match_forward(self, n_blocks: int, base_channels: int):
        """CNNHost segment_channels matches actual feature dimensions."""
        host = CNNHost(
            n_blocks=n_blocks,
            base_channels=base_channels,
            memory_format=torch.contiguous_format,
        )

        # Create sample input
        x = torch.randn(2, 3, 32, 32)

        # Check each segment
        for slot_id, expected_channels in host.segment_channels.items():
            features = host.forward_to_segment(slot_id, x)
            actual_channels = features.shape[1]  # (B, C, H, W)
            assert actual_channels == expected_channels, \
                f"Slot {slot_id}: expected {expected_channels} channels, got {actual_channels}"

    @given(
        n_layer=st.sampled_from([3, 6]),
        n_embd=st.sampled_from([64, 128]),
    )
    @settings(max_examples=15)
    def test_transformer_segment_channels_match_forward(self, n_layer: int, n_embd: int):
        """TransformerHost segment_channels matches actual hidden dimensions."""
        host = TransformerHost(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=max(1, n_embd // 64),
            num_segments=3,
        )

        # Create sample input (token indices)
        x = torch.randint(0, 100, (2, 8))

        # Check each segment
        for slot_id, expected_channels in host.segment_channels.items():
            features = host.forward_to_segment(slot_id, x)
            actual_channels = features.shape[2]  # (B, T, n_embd)
            assert actual_channels == expected_channels, \
                f"Slot {slot_id}: expected {expected_channels} channels, got {actual_channels}"


class TestPartialForwardComposable:
    """forward_to_segment + forward_from_segment should equal full forward."""

    @given(
        n_blocks=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=20)
    def test_cnn_partial_forward_equals_full(self, n_blocks: int):
        """CNN: forward_to + forward_from = full forward."""
        host = CNNHost(
            n_blocks=n_blocks,
            base_channels=32,
            memory_format=torch.contiguous_format,
        )
        host.eval()

        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            # Full forward
            full_output = host(x)

            # Compose through each segment boundary
            segments = list(host.segment_channels.keys())
            for segment in segments:
                partial_to = host.forward_to_segment(segment, x)
                partial_from = host.forward_from_segment(segment, partial_to)

                # Should match full forward
                torch.testing.assert_close(
                    partial_from, full_output,
                    msg=f"Partial forward through {segment} differs from full"
                )

    @given(
        n_layer=st.sampled_from([3, 6]),
    )
    @settings(max_examples=15)
    def test_transformer_partial_forward_equals_full(self, n_layer: int):
        """Transformer: forward_to + forward_from = full forward."""
        host = TransformerHost(
            n_embd=64,
            n_layer=n_layer,
            n_head=2,
            num_segments=3,
        )
        host.eval()

        x = torch.randint(0, 100, (2, 8))

        with torch.no_grad():
            # Full forward
            full_output = host(x)

            # Compose through each segment boundary
            segments = list(host.segment_channels.keys())
            for segment in segments:
                partial_to = host.forward_to_segment(segment, x)
                partial_from = host.forward_from_segment(segment, partial_to)

                # Should match full forward
                torch.testing.assert_close(
                    partial_from, full_output,
                    msg=f"Partial forward through {segment} differs from full"
                )


class TestHostShapePreservation:
    """Host output shape should match expected dimensions."""

    @given(
        batch=st.integers(min_value=1, max_value=4),
        n_blocks=st.integers(min_value=2, max_value=4),
        num_classes=st.sampled_from([10, 100]),
    )
    @settings(max_examples=20)
    def test_cnn_output_shape(self, batch: int, n_blocks: int, num_classes: int):
        """CNNHost output shape is (batch, num_classes)."""
        host = CNNHost(
            n_blocks=n_blocks,
            num_classes=num_classes,
            memory_format=torch.contiguous_format,
        )

        x = torch.randn(batch, 3, 32, 32)
        output = host(x)

        assert output.shape == (batch, num_classes)

    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=32),
        vocab_size=st.sampled_from([100, 1000, 50257]),
    )
    @settings(max_examples=20)
    def test_transformer_output_shape(self, batch: int, seq_len: int, vocab_size: int):
        """TransformerHost output shape is (batch, seq_len, vocab_size)."""
        host = TransformerHost(
            vocab_size=vocab_size,
            n_embd=64,
            n_layer=3,
            n_head=2,
            block_size=256,
            num_segments=3,
        )

        x = torch.randint(0, vocab_size, (batch, seq_len))
        output = host(x)

        assert output.shape == (batch, seq_len, vocab_size)


class TestInjectionSpecsConsistency:
    """injection_specs should be consistent with segment_channels."""

    @given(
        n_blocks=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=20)
    def test_cnn_injection_specs_match_segment_channels(self, n_blocks: int):
        """CNNHost injection_specs matches segment_channels."""
        host = CNNHost(n_blocks=n_blocks, memory_format=torch.contiguous_format)

        specs = host.injection_specs()
        segment_channels = host.segment_channels

        # Each spec slot_id should be in segment_channels
        for spec in specs:
            assert spec.slot_id in segment_channels, \
                f"Spec {spec.slot_id} not in segment_channels"
            assert spec.channels == segment_channels[spec.slot_id], \
                f"Spec channels {spec.channels} != segment_channels {segment_channels[spec.slot_id]}"

    @given(
        n_layer=st.sampled_from([3, 6, 9]),
        num_segments=st.sampled_from([1, 3]),
    )
    @settings(max_examples=15)
    def test_transformer_injection_specs_match_segment_channels(
        self, n_layer: int, num_segments: int
    ):
        """TransformerHost injection_specs matches segment_channels."""
        assume(n_layer % num_segments == 0)

        host = TransformerHost(
            n_layer=n_layer,
            n_head=2,
            n_embd=64,
            num_segments=num_segments,
        )

        specs = host.injection_specs()
        segment_channels = host.segment_channels

        # Number of specs should equal num_segments
        assert len(specs) == num_segments

        # Each spec slot_id should be in segment_channels
        for spec in specs:
            assert spec.slot_id in segment_channels
            assert spec.channels == segment_channels[spec.slot_id]


class TestHostChannelDoubling:
    """CNNHost should double channels per block."""

    @given(
        n_blocks=st.integers(min_value=2, max_value=5),
        base_channels=st.sampled_from([16, 32, 64]),
    )
    @settings(max_examples=20)
    def test_cnn_channel_doubling(self, n_blocks: int, base_channels: int):
        """CNNHost doubles channels with each block."""
        host = CNNHost(
            n_blocks=n_blocks,
            base_channels=base_channels,
            memory_format=torch.contiguous_format,
        )

        specs = host.injection_specs()

        for i, spec in enumerate(specs):
            expected_channels = base_channels * (2 ** i)
            assert spec.channels == expected_channels, \
                f"Block {i}: expected {expected_channels} channels, got {spec.channels}"


class TestLayerDivisibility:
    """TransformerHost should validate layer divisibility."""

    def test_transformer_layer_divisibility_enforced(self):
        """TransformerHost raises ValueError if n_layer not divisible by num_segments."""
        with pytest.raises(ValueError, match="divisible"):
            TransformerHost(n_layer=5, num_segments=3)

    @given(
        layers_per_seg=st.integers(min_value=1, max_value=4),
        num_segments=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=20)
    def test_transformer_valid_divisibility_works(
        self, layers_per_seg: int, num_segments: int
    ):
        """TransformerHost works when n_layer is divisible by num_segments."""
        n_layer = layers_per_seg * num_segments

        # Should not raise
        host = TransformerHost(
            n_layer=n_layer,
            num_segments=num_segments,
            n_head=2,
            n_embd=64,
        )

        assert len(host.injection_specs()) == num_segments
