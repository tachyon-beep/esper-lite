"""Phase 3 pre-implementation: final-layer zero-init helpers."""

from __future__ import annotations

import torch

from esper.kasmina.blend_ops import blend_gate, blend_multiply
from esper.kasmina.blueprints import BlueprintRegistry
from esper.kasmina.blueprints.initialization import zero_init_final_layer


class TestBlueprintFinalLayerZeroInit:
    def test_zero_init_final_layer_makes_conv_light_identity(self) -> None:
        seed = BlueprintRegistry.create("cnn", "conv_light", dim=16)
        x = torch.randn(2, 16, 8, 8)

        zero_init_final_layer(seed)
        seed_out = seed(x)
        assert torch.allclose(seed_out, x, atol=0.0, rtol=0.0)

        # MULTIPLY/GATE should not perturb when seed is identity-like.
        out_mul = blend_multiply(x, seed_out, torch.tensor(0.9))
        out_gate = blend_gate(x, seed_out, torch.tensor(0.9), torch.ones(2, 1, 1, 1))
        assert torch.allclose(out_mul, x, atol=0.0, rtol=0.0)
        assert torch.allclose(out_gate, x, atol=0.0, rtol=0.0)

    def test_zero_init_final_layer_makes_conv_heavy_identity(self) -> None:
        seed = BlueprintRegistry.create("cnn", "conv_heavy", dim=16)
        x = torch.randn(2, 16, 8, 8)

        zero_init_final_layer(seed)
        seed_out = seed(x)
        assert torch.allclose(seed_out, x, atol=0.0, rtol=0.0)

        out_mul = blend_multiply(x, seed_out, torch.tensor(1.0))
        out_gate = blend_gate(x, seed_out, torch.tensor(1.0), torch.ones(2, 1, 1, 1))
        assert torch.allclose(out_mul, x, atol=0.0, rtol=0.0)
        assert torch.allclose(out_gate, x, atol=0.0, rtol=0.0)

    def test_zero_init_final_layer_allow_missing_is_noop(self) -> None:
        seed = BlueprintRegistry.create("cnn", "noop", dim=16)
        zero_init_final_layer(seed, allow_missing=True)

