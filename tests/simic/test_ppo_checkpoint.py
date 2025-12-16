"""Tests for PPOAgent checkpoint save/load.

Tests the checkpoint round-trip functionality, including:
- Default 3-slot configuration
- Custom 5-slot configuration (the bug case)
- Validation of corrupted checkpoints
- Backwards compatibility with legacy checkpoints
"""

import warnings
from pathlib import Path

import pytest
import torch

from esper.leyline.slot_config import SlotConfig
from esper.simic.ppo import PPOAgent, CHECKPOINT_VERSION


class TestPPOCheckpointRoundTrip:
    """Round-trip tests for save/load."""

    def test_roundtrip_default_3_slot_config(self, tmp_path: Path):
        """Default 3-slot config survives round-trip."""
        original = PPOAgent(
            slot_config=SlotConfig.default(),
            device="cpu",
            compile_network=False,
        )
        original.train_steps = 42

        original.save(tmp_path / "agent.pt")
        loaded = PPOAgent.load(tmp_path / "agent.pt", device="cpu")

        assert loaded.slot_config == original.slot_config
        assert loaded.train_steps == 42
        assert torch.allclose(
            original._base_network.slot_head[2].weight,
            loaded._base_network.slot_head[2].weight,
        )

    def test_roundtrip_custom_5_slot_config(self, tmp_path: Path):
        """Custom 5-slot config survives round-trip (the bug case)."""
        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
        original = PPOAgent(
            slot_config=slot_config,
            device="cpu",
            compile_network=False,
        )

        original.save(tmp_path / "agent.pt")
        loaded = PPOAgent.load(tmp_path / "agent.pt", device="cpu")

        assert loaded.slot_config == slot_config
        assert loaded._base_network.num_slots == 5

    def test_roundtrip_preserves_all_hyperparams(self, tmp_path: Path):
        """All hyperparameters survive round-trip."""
        original = PPOAgent(
            slot_config=SlotConfig.default(),
            device="cpu",
            compile_network=False,
            # Non-default values
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.1,
            entropy_coef=0.02,
            value_coef=0.25,
            n_epochs=5,
            batch_size=32,
            max_grad_norm=1.0,
            target_kl=0.02,
        )

        original.save(tmp_path / "agent.pt")
        loaded = PPOAgent.load(tmp_path / "agent.pt", device="cpu")

        assert loaded.gamma == original.gamma
        assert loaded.gae_lambda == original.gae_lambda
        assert loaded.clip_ratio == original.clip_ratio
        assert loaded.entropy_coef == original.entropy_coef
        assert loaded.value_coef == original.value_coef
        assert loaded.n_epochs == original.n_epochs
        assert loaded.batch_size == original.batch_size
        assert loaded.max_grad_norm == original.max_grad_norm
        assert loaded.target_kl == original.target_kl


class TestPPOCheckpointValidation:
    """Validation catches mismatches."""

    def test_corrupted_checkpoint_fails_with_clear_error(self, tmp_path: Path):
        """Corrupted checkpoint (missing slot_ids) fails gracefully."""
        # Save 5-slot agent
        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
        agent = PPOAgent(slot_config=slot_config, device="cpu", compile_network=False)
        agent.save(tmp_path / "agent.pt")

        # Corrupt: remove slot_ids (simulates legacy or corrupted checkpoint)
        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        del checkpoint['architecture']['slot_ids']
        torch.save(checkpoint, tmp_path / "corrupted.pt")

        # Should fail with clear error (legacy defaults to 3 slots)
        with pytest.raises(RuntimeError, match="slot count mismatch"):
            PPOAgent.load(tmp_path / "corrupted.pt", device="cpu")


class TestPPOCheckpointBackwardsCompatibility:
    """Legacy checkpoint handling."""

    def test_legacy_checkpoint_loads_with_warning(self, tmp_path: Path):
        """Legacy checkpoint (3 slots) loads with deprecation warning."""
        # Create checkpoint with current format
        agent = PPOAgent(device="cpu", compile_network=False)
        agent.save(tmp_path / "agent.pt")

        # Strip to legacy format
        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        checkpoint.pop('checkpoint_version', None)
        checkpoint['architecture'].pop('slot_ids', None)
        checkpoint['architecture'].pop('num_slots', None)
        torch.save(checkpoint, tmp_path / "legacy.pt")

        # Should load with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = PPOAgent.load(tmp_path / "legacy.pt", device="cpu")

            assert any("legacy checkpoint" in str(x.message).lower() for x in w)

        assert loaded.slot_config == SlotConfig.default()


class TestPPOCheckpointVersion:
    """Checkpoint version handling."""

    def test_checkpoint_contains_version(self, tmp_path: Path):
        """Checkpoint includes version field."""
        agent = PPOAgent(device="cpu", compile_network=False)
        agent.save(tmp_path / "agent.pt")

        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        assert 'checkpoint_version' in checkpoint
        assert checkpoint['checkpoint_version'] == CHECKPOINT_VERSION

    def test_checkpoint_contains_slot_ids(self, tmp_path: Path):
        """Checkpoint includes slot_ids in architecture."""
        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
        agent = PPOAgent(slot_config=slot_config, device="cpu", compile_network=False)
        agent.save(tmp_path / "agent.pt")

        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        assert 'slot_ids' in checkpoint['architecture']
        assert checkpoint['architecture']['slot_ids'] == ("r0c0", "r0c1", "r0c2", "r1c0", "r1c1")
        assert checkpoint['architecture']['num_slots'] == 5
