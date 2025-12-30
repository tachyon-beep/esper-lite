"""Tests for alpha_curve field in telemetry payloads."""

import pytest
from esper.leyline.telemetry import (
    SeedTelemetry,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
)


class TestSeedTelemetryAlphaCurve:
    """Test alpha_curve field in SeedTelemetry."""

    def test_default_alpha_curve_is_linear(self):
        """SeedTelemetry should default to LINEAR curve."""
        telemetry = SeedTelemetry(seed_id="test", blueprint_id="conv_l", layer_id="l0")
        assert telemetry.alpha_curve == "LINEAR"

    def test_to_dict_includes_alpha_curve(self):
        """to_dict() should include alpha_curve field."""
        telemetry = SeedTelemetry(
            seed_id="test",
            blueprint_id="conv_l",
            layer_id="l0",
            alpha_curve="SIGMOID",
        )
        data = telemetry.to_dict()
        assert "alpha_curve" in data
        assert data["alpha_curve"] == "SIGMOID"

    def test_from_dict_restores_alpha_curve(self):
        """from_dict() should restore alpha_curve field."""
        original = SeedTelemetry(
            seed_id="test",
            blueprint_id="conv_l",
            layer_id="l0",
            alpha_curve="SIGMOID_SHARP",
        )
        data = original.to_dict()
        restored = SeedTelemetry.from_dict(data)
        assert restored.alpha_curve == "SIGMOID_SHARP"

    def test_from_dict_requires_alpha_curve(self):
        """from_dict() should require alpha_curve field - no silent defaults."""
        incomplete_data = {
            "seed_id": "test",
            "blueprint_id": "conv_l",
            "layer_id": "l0",
            "gradient_norm": 1.0,
            "gradient_health": 1.0,
            "has_vanishing": False,
            "has_exploding": False,
            "accuracy": 0.0,
            "accuracy_delta": 0.0,
            "epochs_in_stage": 0,
            "stage": 1,
            "alpha": 0.0,
            "alpha_target": 0.0,
            "alpha_mode": 0,
            "alpha_steps_total": 0,
            "alpha_steps_done": 0,
            "time_to_target": 0,
            "alpha_velocity": 0.0,
            "alpha_algorithm": 1,  # AlphaAlgorithm.ADD
            "epoch": 0,
            "max_epochs": 25,
            "blend_tempo_epochs": 5,
            "blending_velocity": 0.0,
            "captured_at": "2025-01-01T00:00:00",
            # No alpha_curve - should fail
        }
        with pytest.raises(KeyError):
            SeedTelemetry.from_dict(incomplete_data)


class TestSeedGerminatedPayloadAlphaCurve:
    """Test alpha_curve field in SeedGerminatedPayload."""

    def test_default_alpha_curve_is_linear(self):
        """SeedGerminatedPayload should default to LINEAR curve."""
        payload = SeedGerminatedPayload(
            slot_id="slot_0",
            env_id=0,
            blueprint_id="conv_l",
            params=1000,
        )
        assert payload.alpha_curve == "LINEAR"

    def test_from_dict_restores_alpha_curve(self):
        """from_dict() should restore alpha_curve field."""
        data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "blueprint_id": "conv_l",
            "params": 1000,
            "alpha_curve": "COSINE",
        }
        payload = SeedGerminatedPayload.from_dict(data)
        assert payload.alpha_curve == "COSINE"

    def test_from_dict_requires_alpha_curve(self):
        """from_dict() should require alpha_curve field - no silent defaults."""
        incomplete_data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "blueprint_id": "conv_l",
            "params": 1000,
            # No alpha_curve - should fail
        }
        with pytest.raises(KeyError):
            SeedGerminatedPayload.from_dict(incomplete_data)


class TestSeedStageChangedPayloadAlphaCurve:
    """Test alpha_curve field in SeedStageChangedPayload.

    Note: alpha_curve is always present (not None) because the policy always
    samples a curve. However, the curve only causally affects outcomes during
    BLENDING (SET_ALPHA_TARGET/PRUNE operations). The advantage masking in
    simic/agent/advantages.py handles causal attribution - see design rationale in plan.
    """

    def test_default_alpha_curve_is_linear(self):
        """SeedStageChangedPayload should default to LINEAR curve."""
        payload = SeedStageChangedPayload(
            slot_id="slot_0",
            env_id=0,
            from_stage="TRAINING",
            to_stage="BLENDING",
        )
        assert payload.alpha_curve == "LINEAR"

    def test_from_dict_restores_alpha_curve(self):
        """from_dict() should restore alpha_curve field."""
        data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "from": "TRAINING",
            "to": "BLENDING",
            "alpha_curve": "SIGMOID_GENTLE",
        }
        payload = SeedStageChangedPayload.from_dict(data)
        assert payload.alpha_curve == "SIGMOID_GENTLE"

    def test_from_dict_requires_alpha_curve(self):
        """from_dict() should require alpha_curve field - no silent defaults."""
        incomplete_data = {
            "slot_id": "slot_0",
            "env_id": 0,
            "from": "TRAINING",
            "to": "BLENDING",
            # No alpha_curve - should fail
        }
        with pytest.raises(KeyError):
            SeedStageChangedPayload.from_dict(incomplete_data)
