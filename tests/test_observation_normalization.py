"""Tests for observation normalization."""

import pytest


def test_normalize_observation_basic():
    """normalize_observation returns dict with same keys."""
    from esper.simic.features import normalize_observation, TaskConfig

    config = TaskConfig.for_cifar10()
    obs = {
        "epoch": 10,
        "global_step": 1000,
        "train_loss": 1.5,
        "val_loss": 1.8,
        "loss_delta": -0.1,
        "plateau_epochs": 3,
        "seed_alpha": 0.5,
        "has_active_seed": 1,
        "seed_stage": 3,
    }

    normalized = normalize_observation(obs, config)

    assert "epoch" in normalized
    assert "val_loss" in normalized


def test_normalize_observation_epoch_range():
    """Epoch normalized to [0, 1]."""
    from esper.simic.features import normalize_observation, TaskConfig

    config = TaskConfig.for_cifar10()

    obs_start = {
        "epoch": 0,
        "global_step": 0,
        "train_loss": 2.0,
        "val_loss": 2.0,
        "loss_delta": 0,
        "plateau_epochs": 0,
        "seed_alpha": 0,
        "has_active_seed": 0,
        "seed_stage": 0,
    }
    obs_end = {
        "epoch": 25,
        "global_step": 10000,
        "train_loss": 0.5,
        "val_loss": 0.5,
        "loss_delta": 0,
        "plateau_epochs": 0,
        "seed_alpha": 0,
        "has_active_seed": 0,
        "seed_stage": 0,
    }

    norm_start = normalize_observation(obs_start, config)
    norm_end = normalize_observation(obs_end, config)

    assert norm_start["epoch"] == 0.0
    assert norm_end["epoch"] == 1.0


def test_normalize_observation_loss_centered():
    """Loss normalized relative to task baseline."""
    from esper.simic.features import normalize_observation, TaskConfig

    config = TaskConfig.for_cifar10()

    obs = {
        "epoch": 10,
        "global_step": 1000,
        "train_loss": 2.3,
        "val_loss": 2.3,
        "loss_delta": 0,
        "plateau_epochs": 0,
        "seed_alpha": 0,
        "has_active_seed": 0,
        "seed_stage": 0,
    }

    normalized = normalize_observation(obs, config)

    assert 0.9 < normalized["val_loss"] < 1.1


def test_task_config_cifar10():
    """TaskConfig has CIFAR-10 preset."""
    from esper.simic.features import TaskConfig

    config = TaskConfig.for_cifar10()

    assert config.max_epochs == 25
    assert config.baseline_loss == pytest.approx(2.3, rel=0.1)


def test_task_config_tinystories():
    """TaskConfig has TinyStories preset."""
    from esper.simic.features import TaskConfig

    config = TaskConfig.for_tinystories()

    assert config.max_epochs == 50
    assert config.baseline_loss > 5.0
