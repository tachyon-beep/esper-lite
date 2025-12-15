"""Comprehensive torch.compile compatibility tests for PyTorch 2.9.

This test file verifies that all tensor operations intended for compilation
work correctly with torch.compile fullgraph=True mode.

PyTorch Expert Notes:
- Static shapes work with default compilation
- Dynamic shapes may need torch._constrain_as_size hints for value ranges
- Control flow (if/else) causes graph specialization, not breaks
- Python builtins like zip() on tuples are traced without graph breaks

Coverage:
- isolation.py: ste_forward, blend_with_isolation (confirmed compile-safe)
- host.py: CNNHost.forward, TransformerHost.forward (confirmed compile-safe)
- slot.py: SeedSlot.forward uses @torch.compiler.disable (REQUIRED for CUDA graphs)
- networks.py: MaskedCategorical validation uses @torch.compiler.disable (intentional)
"""
import torch

from esper.kasmina.isolation import blend_with_isolation, ste_forward
from esper.kasmina.host import CNNHost, TransformerHost


class TestIsolationCompileCompatibility:
    """Verify isolation tensor ops compile without graph breaks."""

    def test_ste_forward_compiles_with_dynamic_shapes(self):
        """STE forward should handle various batch sizes under compilation."""
        compiled_ste = torch.compile(ste_forward, fullgraph=True)

        # Test multiple batch sizes - compiler handles dynamic shapes
        for batch_size in [1, 4, 16]:
            host = torch.randn(batch_size, 32, 8, 8, requires_grad=True)
            seed = torch.randn(batch_size, 32, 8, 8, requires_grad=True)

            result = compiled_ste(host, seed)
            assert result.shape == host.shape
            assert torch.allclose(result, host, atol=1e-6)

    def test_ste_forward_maintains_gradients_when_compiled(self):
        """Compiled STE should maintain correct gradient flow."""
        compiled_ste = torch.compile(ste_forward, fullgraph=True)

        host = torch.randn(2, 32, requires_grad=True)
        seed = torch.randn(2, 32, requires_grad=True)

        result = compiled_ste(host, seed)
        loss = result.sum()
        loss.backward()

        # Gradients should flow to both
        assert host.grad is not None
        assert seed.grad is not None
        # Host grad should be all ones (dL/d(result) = 1, d(result)/d(host) = 1)
        assert torch.allclose(host.grad, torch.ones_like(host))
        # Seed grad should also be ones (STE: backward treats seed as if it was used)
        assert torch.allclose(seed.grad, torch.ones_like(seed))

    def test_blend_compiles_with_different_alpha_values(self):
        """Blend should compile and handle various alpha values."""
        compiled_blend = torch.compile(blend_with_isolation, fullgraph=True)

        host = torch.randn(4, 64)
        seed = torch.randn(4, 64)

        # Test boundary and mid values
        for alpha in [0.0, 0.5, 1.0]:
            result = compiled_blend(host, seed, alpha)
            assert result.shape == host.shape

            # Verify blend formula: host + alpha * (seed - host)
            expected = torch.lerp(host, seed, alpha)
            assert torch.allclose(result, expected, atol=1e-6)

    def test_blend_gradient_flow_to_both_inputs(self):
        """Blend should allow gradients to flow to both host and seed.

        This verifies the fix for the gradient flow bug where detach_host=True
        previously zeroed ALL host gradients. Now blend_with_isolation always
        allows gradients proportional to contribution: host gets (1-alpha),
        seed gets alpha.
        """
        compiled_blend = torch.compile(blend_with_isolation, fullgraph=True)

        host = torch.randn(2, 32, requires_grad=True)
        seed = torch.randn(2, 32, requires_grad=True)

        result = compiled_blend(host, seed, 0.5)
        loss = result.sum()
        loss.backward()

        # Both host and seed should receive gradients
        assert host.grad is not None, "Host should receive gradients"
        assert seed.grad is not None, "Seed should receive gradients"

        # Gradients should be proportional to alpha weighting
        # With alpha=0.5, both should receive equal gradients
        assert torch.allclose(host.grad, seed.grad, atol=1e-6)


class TestHostCompileCompatibility:
    """Verify host network forward passes compile efficiently."""

    def test_cnn_host_compiles_with_dynamic_batch_size(self):
        """CNNHost should compile and handle various batch sizes."""
        host = CNNHost(num_classes=10, n_blocks=3)
        compiled_host = torch.compile(host, fullgraph=True)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            result = compiled_host(x)
            assert result.shape == (batch_size, 10)

    def test_transformer_host_compiles_with_dynamic_sequence_length(self):
        """TransformerHost should compile and handle various sequence lengths."""
        host = TransformerHost(
            vocab_size=100, n_embd=64, n_head=2, n_layer=2, block_size=32
        )
        compiled_host = torch.compile(host, fullgraph=True)

        for seq_len in [8, 16, 32]:
            x = torch.randint(0, 100, (2, seq_len))
            result = compiled_host(x)
            assert result.shape == (2, seq_len, 100)

    def test_cnn_host_slot_injection_compiles(self):
        """CNNHost with identity slots should compile without graph breaks."""
        host = CNNHost(num_classes=10, n_blocks=4)

        # Slots are Identity by default - should compile
        compiled_host = torch.compile(host, fullgraph=True)

        x = torch.randn(2, 3, 32, 32)
        result = compiled_host(x)
        assert result.shape == (2, 10)

    def test_transformer_host_slot_injection_compiles(self):
        """TransformerHost with identity slots should compile without graph breaks."""
        host = TransformerHost(
            vocab_size=100, n_embd=64, n_head=2, n_layer=3, block_size=16
        )

        # Slots are Identity by default - should compile
        compiled_host = torch.compile(host, fullgraph=True)

        x = torch.randint(0, 100, (2, 12))
        result = compiled_host(x)
        assert result.shape == (2, 12, 100)


class TestPrecomputedKeysNoGraphBreaks:
    """Verify pre-computed slot keys don't cause graph breaks."""

    def test_cnn_slot_keys_are_tuples(self):
        """CNNHost slot keys should be tuples (inlined by compiler)."""
        host = CNNHost(num_classes=10, n_blocks=5)

        # Tuples are compile-friendly data structures
        assert isinstance(host._slot_keys, tuple)
        assert isinstance(host._slot_indices, tuple)
        assert len(host._slot_keys) == 4  # n_blocks - 1

    def test_transformer_slot_keys_are_tuples(self):
        """TransformerHost slot keys should be tuples."""
        host = TransformerHost(
            vocab_size=100, n_embd=64, n_head=2, n_layer=4, block_size=16
        )

        assert isinstance(host._slot_keys, tuple)
        assert len(host._slot_keys) == 4  # n_layer


class TestDynamicShapeHandling:
    """Test dynamic shape handling in compiled functions.

    PyTorch 2.9 handles most dynamic shapes automatically via Dynamo.
    torch._constrain_as_size would only be needed for manual hinting
    of valid size ranges, which is not required for our current use cases.
    """

    def test_dynamic_batch_without_explicit_constraints(self):
        """Standard dynamic batch compilation should work without explicit constraints."""
        def simple_forward(x: torch.Tensor) -> torch.Tensor:
            # B, C = x.shape  # Shape extraction - Dynamo handles this
            return x * 2

        compiled_fn = torch.compile(simple_forward, fullgraph=True)

        for batch in [1, 4, 16, 64]:
            x = torch.randn(batch, 32)
            result = compiled_fn(x)
            assert result.shape == (batch, 32)

    def test_arange_with_dynamic_size(self):
        """torch.arange with dynamic size should compile."""
        def positional_encoding(x: torch.Tensor) -> torch.Tensor:
            B, T = x.shape
            pos = torch.arange(T, device=x.device)
            return x + pos.unsqueeze(0)  # Broadcast

        compiled_fn = torch.compile(positional_encoding, fullgraph=True)

        for seq_len in [8, 16, 32]:
            x = torch.randn(2, seq_len)
            result = compiled_fn(x)
            assert result.shape == (2, seq_len)


class TestCompilerDisabledPaths:
    """Document compile strategy for control-flow-heavy methods.

    These tests verify compile decorators are correctly applied (or intentionally
    omitted) based on PyTorch expert analysis.
    """

    def test_seed_slot_forward_allows_compilation(self):
        """SeedSlot.forward should NOT have @torch.compiler.disable.

        Design decision (updated 2025-12-10):
        - @torch.compiler.disable completely opts out of compilation, which is
          WORSE than allowing Dynamo to specialize into multiple graphs
        - Stage-dependent control flow causes graph specialization (~6-8 graphs),
          but stage transitions are rare (once per epoch)
        - After warmup, execution stays within a single specialized graph
        - End-to-end compilation provides fusion benefits that outweigh
          warmup specialization costs

        DO NOT add @torch.compiler.disable to SeedSlot.forward.
        """
        from esper.kasmina.slot import SeedSlot

        # Verify forward is NOT disabled from compilation
        forward_method = SeedSlot.forward
        assert not getattr(forward_method, '_torchdynamo_disable', False), \
            "SeedSlot.forward should NOT have @torch.compiler.disable - allow graph specialization"

    def test_validate_action_mask_has_compiler_disable(self):
        """_validate_action_mask should have @torch.compiler.disable decorator."""
        from esper.simic.action_masks import _validate_action_mask

        assert getattr(_validate_action_mask, '_torchdynamo_disable', False), \
            "_validate_action_mask should have @torch.compiler.disable"
