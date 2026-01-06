"""Tests for Host networks (CNNHost and TransformerHost).

This module tests:
- Segment boundaries and channel mappings
- Forward pass through segments (forward_to_segment, forward_from_segment)
- Multi-slot model creation and forward passes
- Round-trip consistency (full forward vs segmented forward)
"""

import torch

from esper.kasmina.host import CNNHost, TransformerHost, MorphogeneticModel
from esper.leyline import InjectionSpec


# =============================================================================
# CNNHost Segment Tests
# =============================================================================


def test_host_segment_channels():
    """CNNHost should expose channel counts at each segment boundary."""
    host = CNNHost()

    # Should expose channels at each injection point (access directly - AttributeError if missing)
    # With multichannel: PRE_POOL (r0cX) and POST_POOL (r1cX) surfaces
    assert host.segment_channels == {
        "r0c0": 32,   # PRE_POOL after block0
        "r0c1": 64,   # PRE_POOL after block1
        "r0c2": 128,  # PRE_POOL after block2
        "r1c0": 32,   # POST_POOL after pool0
        "r1c1": 64,   # POST_POOL after pool1
        "r1c2": 128,  # POST_POOL after pool2
    }


def test_host_forward_segments():
    """CNNHost should support segmented forward pass with PRE_POOL and POST_POOL surfaces."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to PRE_POOL segments (before pooling)
    x_pre0 = host.forward_to_segment("r0c0", x)
    assert x_pre0.shape == (2, 32, 32, 32)  # After block0, before pool

    x_pre1 = host.forward_to_segment("r0c1", x)
    assert x_pre1.shape == (2, 64, 16, 16)  # After block1, before pool

    x_pre2 = host.forward_to_segment("r0c2", x)
    assert x_pre2.shape == (2, 128, 8, 8)  # After block2, before pool

    # Forward to POST_POOL segments (after pooling)
    x_post0 = host.forward_to_segment("r1c0", x)
    assert x_post0.shape == (2, 32, 16, 16)  # After block0 + pool

    x_post1 = host.forward_to_segment("r1c1", x)
    assert x_post1.shape == (2, 64, 8, 8)  # After block1 + pool

    x_post2 = host.forward_to_segment("r1c2", x)
    assert x_post2.shape == (2, 128, 4, 4)  # After block2 + pool

    # Forward from various segments to output
    out = host.forward_from_segment("r1c2", x_post2)
    assert out.shape == (2, 10)

    out_pre = host.forward_from_segment("r0c2", x_pre2)
    assert out_pre.shape == (2, 10)


def test_forward_from_early_segment():
    """Should be able to forward from early segment through rest of network."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to early, then from early to output
    x_early = host.forward_to_segment("r0c0", x)
    out = host.forward_from_segment("r0c0", x_early)
    assert out.shape == (2, 10)


def test_forward_from_mid_segment():
    """Should be able to forward from mid segment through rest of network."""
    host = CNNHost()
    x = torch.randn(2, 3, 32, 32)

    # Forward to mid, then from mid to output
    x_mid = host.forward_to_segment("r0c1", x)
    out = host.forward_from_segment("r0c1", x_mid)
    assert out.shape == (2, 10)


def test_injection_points_alias_segment_channels():
    """injection_points should be an alias for segment_channels."""
    host = CNNHost()

    # Both should expose the same canonical IDs and channels
    assert host.injection_points == host.segment_channels
    assert "r0c0" in host.injection_points
    assert "r0c1" in host.injection_points
    assert "r0c2" in host.injection_points


# =============================================================================
# Multi-Slot Model Tests
# =============================================================================


def test_multislot_model_creation():
    """MorphogeneticModel should support multiple slots."""
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    assert len(model.seed_slots) == 3
    assert "r0c0" in model.seed_slots
    assert "r0c1" in model.seed_slots
    assert "r0c2" in model.seed_slots

    # Each slot should have correct channels
    assert model.seed_slots["r0c0"].channels == 32
    assert model.seed_slots["r0c1"].channels == 64
    assert model.seed_slots["r0c2"].channels == 128


def test_multislot_forward_pass():
    """Multi-slot model forward should pass through all slots."""
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_multislot_germinate_specific_slot():
    """Should germinate seed in specific slot."""
    host = CNNHost()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])

    # Germinate in mid slot (use actual blueprint name)
    model.germinate_seed("conv_light", "test_seed", slot="r0c1")

    assert model.seed_slots["r0c1"].is_active
    assert not model.seed_slots["r0c0"].is_active
    assert not model.seed_slots["r0c2"].is_active


# =============================================================================
# TransformerHost Segment Tests
# =============================================================================


class TestTransformerHostSegments:
    """Test TransformerHost segment_channels and segment methods."""

    def test_segment_channels_exists(self):
        """TransformerHost must expose segment_channels attribute."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        assert isinstance(host.segment_channels, dict)
        assert set(host.segment_channels.keys()) == {"r0c0", "r0c1", "r0c2"}

    def test_segment_channels_values(self):
        """All segments should map to n_embd dimension."""
        n_embd = 128
        host = TransformerHost(vocab_size=100, n_embd=n_embd, n_head=4, n_layer=6, block_size=32)
        for segment, dim in host.segment_channels.items():
            assert dim == n_embd, f"Segment {segment} should have dim {n_embd}, got {dim}"

    def test_forward_to_segment_returns_embeddings(self):
        """forward_to_segment should return hidden states at segment boundary."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        x = torch.randint(0, 100, (2, 16))  # batch=2, seq=16

        h = host.forward_to_segment("r0c1", x)
        assert h.shape == (2, 16, 64)  # (batch, seq, n_embd)

    def test_forward_from_segment_returns_logits(self):
        """forward_from_segment should return logits from hidden states."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        h = torch.randn(2, 16, 64)  # (batch, seq, n_embd)

        logits = host.forward_from_segment("r0c1", h)
        assert logits.shape == (2, 16, 100)  # (batch, seq, vocab_size)

    def test_segment_round_trip_matches_forward(self):
        """forward_to_segment + forward_from_segment should match full forward."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        host.eval()  # Disable dropout for deterministic comparison
        x = torch.randint(0, 100, (2, 16))

        # Full forward
        with torch.no_grad():
            full_out = host(x)

        # Segment round-trip through "r0c1"
        with torch.no_grad():
            h = host.forward_to_segment("r0c1", x)
            segment_out = host.forward_from_segment("r0c1", h)

        # Should be identical (deterministic with eval mode)
        torch.testing.assert_close(full_out, segment_out, rtol=1e-5, atol=1e-5)


class TestCNNHostSegments:
    """Test CNNHost segment routing consistency."""

    def test_cnn_segment_consistency(self):
        """Segment round-trips should match full forward pass."""
        host = CNNHost(num_classes=10, base_channels=8)
        host.eval()
        x = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            full_out = host(x)
            mid_h = host.forward_to_segment("r0c1", x)
            segment_out = host.forward_from_segment("r0c1", mid_h)

            early_h = host.forward_to_segment("r0c0", x)
            late_h_direct = host.forward_to_segment("r0c2", x)
            late_h_from_early = host.forward_to_segment("r0c2", early_h, from_segment="r0c0")
            late_out_from_early = host.forward_from_segment("r0c2", late_h_from_early)

        torch.testing.assert_close(segment_out, full_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(late_h_from_early, late_h_direct, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(late_out_from_early, full_out, rtol=1e-5, atol=1e-5)


# =============================================================================
# TransformerHost with MorphogeneticModel Tests
# =============================================================================


def test_transformer_forward_matches_host():
    """MorphogeneticModel with TransformerHost should match host output when no seeds active."""
    # Create host and model with multiple slots
    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=6, block_size=32, dropout=0.0)
    host.eval()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r0c2"])
    model.eval()

    # Test input
    x = torch.randint(0, 1000, (2, 16))

    # Forward through both
    with torch.no_grad():
        host_out = host(x)
        model_out = model(x)

    # Should match exactly when no seeds are active
    assert torch.allclose(host_out, model_out, atol=1e-6), \
        f"Outputs differ: max diff = {(host_out - model_out).abs().max().item()}"


def test_transformer_single_slot_forward():
    """MorphogeneticModel with single slot should process all layers."""
    # Create host and model with only mid slot
    host = TransformerHost(vocab_size=1000, n_embd=64, n_head=2, n_layer=6, block_size=32, dropout=0.0)
    host.eval()
    model = MorphogeneticModel(host, device="cpu", slots=["r0c1"])
    model.eval()

    # Test input
    x = torch.randint(0, 1000, (2, 16))

    # Forward through both
    with torch.no_grad():
        host_out = host(x)
        model_out = model(x)

    # Should match exactly - all layers must be processed
    assert torch.allclose(host_out, model_out, atol=1e-6), \
        f"Outputs differ: max diff = {(host_out - model_out).abs().max().item()}"


# =============================================================================
# CNNHost Dynamic Injection Specs Tests
# =============================================================================


class TestCNNHostInjectionSpecs:
    """Test CNNHost.injection_specs() method for dynamic slot discovery."""

    def test_default_3_block_has_6_specs(self):
        """Default 3-block CNNHost should return 6 specs (3 PRE_POOL + 3 POST_POOL)."""
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        assert len(specs) == 6
        assert all(isinstance(s, InjectionSpec) for s in specs)

    def test_specs_have_correct_slot_ids(self):
        """Specs should have canonical slot IDs in row-major order for action stability."""
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        slot_ids = [s.slot_id for s in specs]
        # Row-major order: all row 0 (PRE_POOL), then row 1 (POST_POOL)
        assert slot_ids == ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"]

    def test_specs_have_increasing_positions(self):
        """Specs should have positions sorted by order field."""
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        # Positions are for visualization; order field determines sort
        orders = [s.order for s in specs]
        assert orders == sorted(orders)
        # All positions should be in (0, 1]
        assert all(0 < s.position <= 1.0 for s in specs)

    def test_specs_have_correct_channels(self):
        """Specs should reflect actual block channel counts (same for PRE and POST_POOL)."""
        host = CNNHost(n_blocks=3, base_channels=32)
        specs = host.injection_specs()
        # Row-major order: (all PRE_POOL then all POST_POOL)
        # Row 0 channels: 32, 64, 128 (blocks 0, 1, 2)
        # Row 1 channels: 32, 64, 128 (blocks 0, 1, 2)
        channels = [s.channels for s in specs]
        assert channels == [32, 64, 128, 32, 64, 128]

    def test_5_block_host_has_10_specs(self):
        """5-block CNNHost should return 10 injection specs (5 PRE_POOL + 5 POST_POOL)."""
        host = CNNHost(n_blocks=5, base_channels=16)
        specs = host.injection_specs()
        assert len(specs) == 10
        slot_ids = [s.slot_id for s in specs]
        # Row-major order: all row 0, then all row 1
        assert slot_ids == [
            "r0c0", "r0c1", "r0c2", "r0c3", "r0c4",  # row 0 (PRE_POOL)
            "r1c0", "r1c1", "r1c2", "r1c3", "r1c4",  # row 1 (POST_POOL)
        ]

    def test_specs_layer_ranges(self):
        """Each spec should have correct layer range (same block for PRE and POST_POOL)."""
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        # Row-major order: r0c0, r0c1, r0c2, r1c0, r1c1, r1c2
        # Row 0 (PRE_POOL): blocks 0, 1, 2
        assert specs[0].layer_range == (0, 1)  # r0c0 (block 0)
        assert specs[1].layer_range == (1, 2)  # r0c1 (block 1)
        assert specs[2].layer_range == (2, 3)  # r0c2 (block 2)
        # Row 1 (POST_POOL): blocks 0, 1, 2
        assert specs[3].layer_range == (0, 1)  # r1c0 (block 0)
        assert specs[4].layer_range == (1, 2)  # r1c1 (block 1)
        assert specs[5].layer_range == (2, 3)  # r1c2 (block 2)

    def test_specs_have_correct_surface_types(self):
        """Specs should have correct surface type annotations."""
        from esper.leyline import SurfaceType

        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        # Row-major order: all PRE_POOL (row 0), then all POST_POOL (row 1)
        # r0c0, r0c1, r0c2 (PRE), r1c0, r1c1, r1c2 (POST)
        assert specs[0].surface == SurfaceType.PRE_POOL   # r0c0
        assert specs[1].surface == SurfaceType.PRE_POOL   # r0c1
        assert specs[2].surface == SurfaceType.PRE_POOL   # r0c2
        assert specs[3].surface == SurfaceType.POST_POOL  # r1c0
        assert specs[4].surface == SurfaceType.POST_POOL  # r1c1
        assert specs[5].surface == SurfaceType.POST_POOL  # r1c2

    def test_specs_have_row_col_coordinates(self):
        """Specs should have explicit row/col grid coordinates."""
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        # Check row/col match slot_id format
        for spec in specs:
            assert spec.row is not None
            assert spec.col is not None
            expected_id = f"r{spec.row}c{spec.col}"
            assert spec.slot_id == expected_id

    def test_specs_order_field_row_major(self):
        """Order field should be row-major for stable action indices."""
        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        # Row-major order: r0c0(0), r0c1(1), r0c2(2), r1c0(3), r1c1(4), r1c2(5)
        assert specs[0].order == 0  # r0c0
        assert specs[1].order == 1  # r0c1
        assert specs[2].order == 2  # r0c2
        assert specs[3].order == 3  # r1c0
        assert specs[4].order == 4  # r1c1
        assert specs[5].order == 5  # r1c2

    def test_pool_layers_controls_post_pool_count(self):
        """POST_POOL specs should only exist where pooling is applied."""
        # With pool_layers=2, only blocks 0 and 1 have pooling
        host = CNNHost(n_blocks=3, pool_layers=2)
        specs = host.injection_specs()
        # 3 PRE_POOL + 2 POST_POOL = 5 specs
        assert len(specs) == 5
        slot_ids = [s.slot_id for s in specs]
        # Row-major order: all row 0 (PRE_POOL), then row 1 (POST_POOL where present)
        assert slot_ids == ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1"]


# =============================================================================
# TransformerHost Dynamic Injection Specs Tests
# =============================================================================


class TestTransformerHostInjectionSpecs:
    """Test TransformerHost.injection_specs() method for dynamic slot discovery."""

    def test_default_3_segments_has_3_specs(self):
        """Default TransformerHost should return 3 injection specs."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        specs = host.injection_specs()
        assert len(specs) == 3
        assert all(isinstance(s, InjectionSpec) for s in specs)

    def test_specs_have_correct_slot_ids(self):
        """Specs should have canonical slot IDs in order."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        specs = host.injection_specs()
        slot_ids = [s.slot_id for s in specs]
        assert slot_ids == ["r0c0", "r0c1", "r0c2"]

    def test_specs_have_increasing_positions(self):
        """Specs should have increasing positions from 0 to 1."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        specs = host.injection_specs()
        positions = [s.position for s in specs]
        assert positions == sorted(positions)
        assert all(0 < p <= 1.0 for p in positions)

    def test_specs_have_uniform_channels(self):
        """All specs should reflect uniform n_embd dimension."""
        n_embd = 128
        host = TransformerHost(vocab_size=100, n_embd=n_embd, n_head=4, n_layer=6, block_size=32)
        specs = host.injection_specs()
        channels = [s.channels for s in specs]
        assert all(c == n_embd for c in channels)

    def test_specs_layer_ranges_cover_all_layers(self):
        """Specs should partition all layers into segments."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        specs = host.injection_specs()
        # For 6 layers divided into 3 segments: [0-2), [2-4), [4-6)
        assert specs[0].layer_range == (0, 2)
        assert specs[1].layer_range == (2, 4)
        assert specs[2].layer_range == (4, 6)

    def test_specs_positions_match_network_depth(self):
        """Positions should be evenly distributed across network depth."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        specs = host.injection_specs()
        # For 3 segments: positions should be 1/3, 2/3, 3/3
        positions = [s.position for s in specs]
        assert positions[0] == 1/3
        assert positions[1] == 2/3
        assert positions[2] == 1.0

    def test_segment_channels_derived_from_injection_specs(self):
        """segment_channels property should derive from injection_specs()."""
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=6, block_size=32)
        # segment_channels should match injection_specs mapping
        specs_dict = {spec.slot_id: spec.channels for spec in host.injection_specs()}
        assert host.segment_channels == specs_dict

    def test_different_layer_counts(self):
        """Should work with different layer counts."""
        # 9 layers -> 3 segments of 3 layers each
        host = TransformerHost(vocab_size=100, n_embd=64, n_head=2, n_layer=9, block_size=32)
        specs = host.injection_specs()
        assert len(specs) == 3
        assert specs[0].layer_range == (0, 3)
        assert specs[1].layer_range == (3, 6)
        assert specs[2].layer_range == (6, 9)


# =============================================================================
# Blueprint Resolution Invariance Tests (ChatGPT recommendation)
# =============================================================================


class TestBlueprintResolutionInvariance:
    """Verify all blueprints work at different spatial resolutions.

    PRE_POOL and POST_POOL injection points have different spatial dimensions.
    All blueprints MUST be resolution-agnostic to work at both surfaces.
    """

    def test_cnn_blueprints_are_resolution_invariant(self):
        """All CNN blueprints should work at 8x8, 16x16, and 32x32 spatial sizes."""
        from esper.kasmina.blueprints import BlueprintRegistry

        channels = 64
        batch_size = 2

        for spec in BlueprintRegistry.list_for_topology("cnn"):
            seed = spec.factory(channels)
            for spatial in [8, 16, 32]:
                x = torch.randn(batch_size, channels, spatial, spatial)
                y = seed(x)
                assert y.shape == x.shape, (
                    f"Blueprint '{spec.name}' not resolution-invariant: "
                    f"input {tuple(x.shape)} -> output {tuple(y.shape)} at {spatial}x{spatial}"
                )

    def test_seeds_preserve_batch_and_channels(self):
        """Seeds must preserve batch size and channel count at all resolutions."""
        from esper.kasmina.blueprints import BlueprintRegistry

        for channels in [32, 64, 128]:
            for spec in BlueprintRegistry.list_for_topology("cnn"):
                seed = spec.factory(channels)
                for spatial in [8, 16]:
                    for batch in [1, 4]:
                        x = torch.randn(batch, channels, spatial, spatial)
                        y = seed(x)
                        assert y.shape[0] == batch, f"{spec.name}: batch size changed"
                        assert y.shape[1] == channels, f"{spec.name}: channels changed"


# =============================================================================
# Channels Last + Fused Forward Smoke Tests (ChatGPT recommendation)
# =============================================================================


class TestChannelsLastInvariance:
    """Verify channels_last memory format is preserved through forward passes.

    This guards against regressions like BUG-005 where detach operations
    could cause segfaults with channels_last tensors.
    """

    def test_cnn_host_preserves_channels_last(self):
        """CNNHost should preserve channels_last format through forward pass."""
        host = CNNHost(n_blocks=3, memory_format=torch.channels_last)
        host.eval()

        x = torch.randn(2, 3, 32, 32).to(memory_format=torch.channels_last)
        assert x.is_contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            # Test forward_to_segment for both PRE_POOL and POST_POOL
            for segment in ["r0c0", "r0c1", "r1c0", "r1c1"]:
                out = host.forward_to_segment(segment, x)
                # Result should still be channels_last (MaxPool2d preserves it)
                assert out.is_contiguous(memory_format=torch.channels_last), (
                    f"forward_to_segment('{segment}') broke channels_last format"
                )

    def test_morphogenetic_model_channels_last_smoke(self):
        """MorphogeneticModel should work with channels_last without errors."""
        host = CNNHost(n_blocks=3, memory_format=torch.channels_last)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r1c0"])
        model.eval()

        x = torch.randn(2, 3, 32, 32).to(memory_format=torch.channels_last)

        with torch.no_grad():
            # Should complete without exceptions
            out = model(x)
            assert out.shape == (2, 10)

    def test_morphogenetic_model_with_active_seed_channels_last(self):
        """MorphogeneticModel with active seed should work with channels_last."""
        host = CNNHost(n_blocks=3, memory_format=torch.channels_last)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1"])
        model.eval()

        # Germinate a seed
        model.germinate_seed("conv_light", "test_seed", slot="r0c0")

        x = torch.randn(2, 3, 32, 32).to(memory_format=torch.channels_last)

        with torch.no_grad():
            out = model(x)
            assert out.shape == (2, 10)

    def test_fused_forward_channels_last_smoke(self):
        """fused_forward should work with channels_last tensors without errors."""
        host = CNNHost(n_blocks=3, memory_format=torch.channels_last)
        model = MorphogeneticModel(host, device="cpu", slots=["r0c0", "r0c1", "r1c0"])
        model.eval()

        # Germinate seeds in multiple slots
        model.germinate_seed("conv_light", "seed1", slot="r0c0")
        model.germinate_seed("conv_light", "seed2", slot="r0c1")

        # Create fused input: K=3 alpha configs, B=2 batch -> [6, 3, 32, 32]
        K, B = 3, 2
        x = torch.randn(K * B, 3, 32, 32).to(memory_format=torch.channels_last)

        # Create alpha overrides for active slots
        alpha_overrides = {
            "r0c0": torch.ones(K * B, 1, 1, 1),
            "r0c1": torch.ones(K * B, 1, 1, 1),
        }

        with torch.no_grad():
            out = model.fused_forward(x, alpha_overrides)
            assert out.shape == (K * B, 10)


# =============================================================================
# Action Index Stability Tests (ChatGPT recommendation)
# =============================================================================


class TestActionIndexStability:
    """Verify SlotConfig.from_specs() produces stable action indices.

    With row-major ordering, indices are grouped by row:
    r0c0(0), r0c1(1), r0c2(2), r1c0(3), r1c1(4), r1c2(5)

    This keeps action indices stable as surfaces are added/removed.
    For forward pass execution order, use host.execution_order().
    """

    def test_row_major_indices(self):
        """Slots should have indices matching row-major order."""
        from esper.leyline import SlotConfig

        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        config = SlotConfig.from_specs(specs)

        # Row-major order: all row 0 first, then row 1
        assert config.index_for_slot_id("r0c0") == 0  # row 0 block 0
        assert config.index_for_slot_id("r0c1") == 1  # row 0 block 1
        assert config.index_for_slot_id("r0c2") == 2  # row 0 block 2
        assert config.index_for_slot_id("r1c0") == 3  # row 1 block 0
        assert config.index_for_slot_id("r1c1") == 4  # row 1 block 1
        assert config.index_for_slot_id("r1c2") == 5  # row 1 block 2

    def test_execution_order_is_interleaved(self):
        """Host execution_order() should return interleaved order for routing."""
        host = CNNHost(n_blocks=3)

        # Execution order is interleaved by block (for forward pass routing)
        assert host.execution_order() == [
            "r0c0", "r1c0",  # block 0: PRE_POOL then POST_POOL
            "r0c1", "r1c1",  # block 1
            "r0c2", "r1c2",  # block 2
        ]

    def test_slot_sort_key_is_row_major(self):
        """slot_sort_key should produce row-major ordering."""
        from esper.leyline import slot_sort_key

        slot_ids = ["r1c0", "r0c2", "r1c2", "r0c0", "r0c1", "r1c1"]
        sorted_ids = sorted(slot_ids, key=slot_sort_key)

        # Row-major: all row 0 first, then all row 1
        assert sorted_ids == ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2"]

    def test_slot_config_order_matches_spec_order(self):
        """SlotConfig slot_ids should match InjectionSpec order field ordering."""
        from esper.leyline import SlotConfig

        host = CNNHost(n_blocks=3)
        specs = host.injection_specs()
        config = SlotConfig.from_specs(specs)

        # Specs are already sorted by order field
        expected_ids = tuple(s.slot_id for s in sorted(specs, key=lambda s: s.order))
        assert config.slot_ids == expected_ids
