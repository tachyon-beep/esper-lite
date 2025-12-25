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
from esper.simic.agent import PPOAgent, CHECKPOINT_VERSION
from esper.tamiyo.policy.factory import create_policy
from esper.tamiyo.policy.features import get_feature_size


class TestPPOCheckpointRoundTrip:
    """Round-trip tests for save/load."""

    def test_roundtrip_default_3_slot_config(self, tmp_path: Path):
        """Default 3-slot config survives round-trip."""
        slot_config = SlotConfig.default()
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        original = PPOAgent(
            policy=policy,
            slot_config=slot_config,
            device="cpu",
        )
        original.train_steps = 42

        original.save(tmp_path / "agent.pt")
        loaded = PPOAgent.load(tmp_path / "agent.pt", device="cpu")

        assert loaded.slot_config == original.slot_config
        assert loaded.train_steps == 42
        assert torch.allclose(
            original.policy.network.slot_head[2].weight,
            loaded.policy.network.slot_head[2].weight,
        )

    def test_roundtrip_custom_5_slot_config(self, tmp_path: Path):
        """Custom 5-slot config survives round-trip (the bug case)."""
        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        original = PPOAgent(
            policy=policy,
            slot_config=slot_config,
            device="cpu",
        )

        original.save(tmp_path / "agent.pt")
        loaded = PPOAgent.load(tmp_path / "agent.pt", device="cpu")

        assert loaded.slot_config == slot_config
        assert loaded.policy.network.num_slots == 5

    def test_roundtrip_preserves_all_hyperparams(self, tmp_path: Path):
        """All hyperparameters survive round-trip."""
        slot_config = SlotConfig.default()
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        original = PPOAgent(
            policy=policy,
            slot_config=slot_config,
            device="cpu",
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

    def test_corrupted_checkpoint_missing_slot_ids_fails(self, tmp_path: Path):
        """Corrupted checkpoint (missing slot_ids) fails with clear error."""
        # Save 5-slot agent
        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, slot_config=slot_config, device="cpu")
        agent.save(tmp_path / "agent.pt")

        # Corrupt: remove slot_ids
        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        del checkpoint['architecture']['slot_ids']
        torch.save(checkpoint, tmp_path / "corrupted.pt")

        # Should fail with clear error about missing slot_ids
        with pytest.raises(RuntimeError, match="architecture.slot_ids is required"):
            PPOAgent.load(tmp_path / "corrupted.pt", device="cpu")


class TestPPOCheckpointNoBackwardsCompatibility:
    """Verify legacy checkpoints are rejected (No Legacy Code Policy)."""

    def test_legacy_checkpoint_fails_fast(self, tmp_path: Path):
        """Legacy checkpoint (missing checkpoint_version) fails with clear error."""
        # Create checkpoint with current format
        slot_config = SlotConfig.default()
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, device="cpu")
        agent.save(tmp_path / "agent.pt")

        # Strip to legacy format (missing required fields)
        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        checkpoint.pop('checkpoint_version', None)
        torch.save(checkpoint, tmp_path / "legacy.pt")

        # Should fail with clear error - no backwards compatibility
        with pytest.raises(RuntimeError, match="missing required field"):
            PPOAgent.load(tmp_path / "legacy.pt", device="cpu")

    def test_wrong_checkpoint_version_fails(self, tmp_path: Path):
        """Checkpoint with wrong version fails with clear error."""
        # Create checkpoint with current format
        slot_config = SlotConfig.default()
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, device="cpu")
        agent.save(tmp_path / "agent.pt")

        # Change version to invalid
        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        checkpoint['checkpoint_version'] = 999
        torch.save(checkpoint, tmp_path / "wrong_version.pt")

        # Should fail with version mismatch error
        with pytest.raises(RuntimeError, match="Checkpoint version mismatch"):
            PPOAgent.load(tmp_path / "wrong_version.pt", device="cpu")


class TestPPOCheckpointVersion:
    """Checkpoint version handling."""

    def test_checkpoint_contains_version(self, tmp_path: Path):
        """Checkpoint includes version field."""
        slot_config = SlotConfig.default()
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, device="cpu")
        agent.save(tmp_path / "agent.pt")

        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        assert 'checkpoint_version' in checkpoint
        assert checkpoint['checkpoint_version'] == CHECKPOINT_VERSION

    def test_checkpoint_contains_slot_ids(self, tmp_path: Path):
        """Checkpoint includes slot_ids in architecture."""
        slot_config = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r1c0", "r1c1"))
        policy = create_policy(
            policy_type="lstm",
            state_dim=get_feature_size(slot_config),
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        agent = PPOAgent(policy=policy, slot_config=slot_config, device="cpu")
        agent.save(tmp_path / "agent.pt")

        checkpoint = torch.load(tmp_path / "agent.pt", weights_only=False)
        assert 'slot_ids' in checkpoint['architecture']
        assert checkpoint['architecture']['slot_ids'] == ("r0c0", "r0c1", "r0c2", "r1c0", "r1c1")
        assert checkpoint['architecture']['num_slots'] == 5
