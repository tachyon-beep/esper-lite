"""Tests for topology safety guard."""

import pytest


def test_germinate_wrong_topology_raises():
    """Germinating transformer blueprint on CNN slot raises."""
    from esper.kasmina.slot import SeedSlot
    from esper.simic.features import TaskConfig

    config = TaskConfig.for_cifar10()

    slot = SeedSlot(
        slot_id="block2_post",
        channels=64,
        task_config=config,
    )

    with pytest.raises(ValueError, match="not available for topology"):
        slot.germinate("lora", "bad-seed")


def test_germinate_correct_topology_succeeds():
    """Germinating matching topology blueprint succeeds."""
    from esper.kasmina.slot import SeedSlot
    from esper.simic.features import TaskConfig

    config = TaskConfig.for_cifar10()

    slot = SeedSlot(
        slot_id="block2_post",
        channels=64,
        task_config=config,
    )

    state = slot.germinate("norm", "good-seed")
    assert state is not None


def test_germinate_unknown_topology_fails_loudly():
    """Unknown topology should raise before blueprint lookup."""
    from esper.kasmina.slot import SeedSlot
    from esper.simic.features import TaskConfig

    config = TaskConfig(
        task_type="classification",
        topology="weird",
        baseline_loss=1.0,
        target_loss=0.5,
        typical_loss_delta_std=0.1,
        max_epochs=5,
    )

    slot = SeedSlot(
        slot_id="weird_slot",
        channels=8,
        task_config=config,
    )

    with pytest.raises(ValueError, match="Unknown topology"):
        slot.germinate("norm", "weird-seed")
