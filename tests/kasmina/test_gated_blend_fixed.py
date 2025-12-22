"""Test fixed GatedBlend behavior (post-M2).

Tests the corrected GatedBlend.get_alpha() implementation that tracks
step-based progress for lifecycle compatibility.
"""

import pytest
import torch

from esper.kasmina.blending import GatedBlend


class TestGatedBlendFixed:
    """Test fixed GatedBlend behavior (post-M2)."""

    def test_get_alpha_tracks_step_progress(self):
        """get_alpha() should return step-based progress, not constant 0.5."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        # At step 0, alpha should be low
        assert gate.get_alpha(0) < 0.2

        # At step 5, alpha should be ~0.5
        alpha_mid = gate.get_alpha(5)
        assert 0.4 < alpha_mid < 0.6

        # At step 10, alpha should be ~1.0
        assert gate.get_alpha(10) >= 0.95

    def test_get_alpha_at_step_zero(self):
        """get_alpha() at step 0 should be 0.0."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)
        assert gate.get_alpha(0) == 0.0

    def test_get_alpha_at_final_step(self):
        """get_alpha() at final step should be 1.0."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)
        assert gate.get_alpha(10) == 1.0

    def test_get_alpha_linear_progression(self):
        """get_alpha() should increase linearly with steps."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=100)

        # Test various points
        assert abs(gate.get_alpha(25) - 0.25) < 1e-6
        assert abs(gate.get_alpha(50) - 0.50) < 1e-6
        assert abs(gate.get_alpha(75) - 0.75) < 1e-6

    def test_get_alpha_caps_at_one(self):
        """get_alpha() should not exceed 1.0 even for steps > total_steps."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        assert gate.get_alpha(15) == 1.0
        assert gate.get_alpha(100) == 1.0

    def test_get_alpha_with_none_step_uses_current(self):
        """get_alpha(None) should use _current_step."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        gate.step(5)
        assert gate.get_alpha(None) == 0.5

        gate.step(10)
        assert gate.get_alpha(None) == 1.0

    def test_total_steps_parameter_required(self):
        """GatedBlend should accept total_steps parameter."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=20)

        assert gate.total_steps == 20

    def test_get_alpha_with_different_total_steps(self):
        """get_alpha() should scale correctly with different total_steps."""
        gate_short = GatedBlend(channels=64, topology="cnn", total_steps=5)
        gate_long = GatedBlend(channels=64, topology="cnn", total_steps=20)

        # Same step, different progress
        assert gate_short.get_alpha(2) == 0.4  # 2/5
        assert gate_long.get_alpha(2) == 0.1   # 2/20

    def test_get_alpha_for_blend_still_uses_gate_network(self):
        """get_alpha_for_blend() should still use learned gate network."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        x = torch.randn(2, 64, 8, 8)
        alpha_tensor = gate.get_alpha_for_blend(x)

        # Should return per-sample tensor, not scalar progress
        assert alpha_tensor.shape == (2, 1, 1, 1)

        # Values should be from gate network, not step-based
        # (this is the key difference between get_alpha and get_alpha_for_blend)

    def test_lifecycle_compatibility(self):
        """get_alpha() progression should allow G3 gate to pass naturally."""
        gate = GatedBlend(channels=64, topology="cnn", total_steps=10)

        # Early in blending: G3 should not pass (alpha < 0.95)
        assert gate.get_alpha(0) < 0.95
        assert gate.get_alpha(5) < 0.95
        assert gate.get_alpha(9) < 0.95

        # At completion: G3 should pass (alpha >= 0.95)
        assert gate.get_alpha(10) >= 0.95
