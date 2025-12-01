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

    with pytest.raises(AssertionError, match="topology"):
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
