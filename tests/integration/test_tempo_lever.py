import pytest
import torch

from esper.kasmina.slot import SeedSlot, SeedStage
from esper.leyline.factored_actions import TempoAction, TEMPO_TO_EPOCHS
from esper.leyline import HEAD_NAMES


@pytest.fixture
def slot():
    """Create a test slot."""
    return SeedSlot(
        slot_id="test_slot",
        channels=64,
        device="cpu",
    )


def test_tempo_stored_in_state(slot):
    """Verify tempo is persisted in SeedState."""
    slot.germinate(
        blueprint_id="norm",
        blend_tempo_epochs=8,  # SLOW
    )

    # Check state storage
    assert slot.state.blend_tempo_epochs == 8

    # Check serialization roundtrip
    state_dict = slot.state.to_dict()
    assert state_dict["blend_tempo_epochs"] == 8


def test_tempo_default_is_standard(slot):
    """Default tempo should be STANDARD (5 epochs)."""
    slot.germinate(blueprint_id="norm")
    assert slot.state.blend_tempo_epochs == 5


def test_head_names_includes_tempo():
    """Verify HEAD_NAMES was updated (critical for PPO training)."""
    assert "tempo" in HEAD_NAMES, "CRITICAL: tempo missing from HEAD_NAMES!"
    assert HEAD_NAMES.index("tempo") == 3  # Before "op"


def test_tempo_to_epochs_values():
    """Verify epoch mappings are correct."""
    assert TEMPO_TO_EPOCHS[TempoAction.FAST] == 3
    assert TEMPO_TO_EPOCHS[TempoAction.STANDARD] == 5
    assert TEMPO_TO_EPOCHS[TempoAction.SLOW] == 8
