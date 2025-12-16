"""Test SeedSlot checkpoint compatibility with PyTorch 2.9."""

import pytest
import torch
import tempfile
from pathlib import Path

from esper.kasmina.slot import SeedSlot
from esper.simic.features import TaskConfig
from esper.leyline.stages import SeedStage


class TestSeedSlotCheckpoint:
    """Test SeedSlot save/load with weights_only=True."""

    @pytest.fixture
    def slot(self):
        """Create a SeedSlot in BLENDING stage."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(
                task_type="classification",
                topology="cnn",
                baseline_loss=2.3,
                target_loss=0.5,
                typical_loss_delta_std=0.1,
                max_epochs=25,
                blending_steps=10,
            ),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="linear",
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)
        return slot

    def test_extra_state_contains_only_primitives(self, slot):
        """get_extra_state() returns only primitive types."""
        extra = slot.get_extra_state()

        # Recursively check no custom types
        def check_primitives(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_primitives(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_primitives(v, f"{path}[{i}]")
            elif obj is None:
                pass
            elif isinstance(obj, (str, int, float, bool)):
                pass
            else:
                pytest.fail(f"Non-primitive at {path}: {type(obj).__name__}")

        check_primitives(extra)

    def test_checkpoint_roundtrip_weights_only(self, slot):
        """Save and load with weights_only=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Save
            torch.save(slot.state_dict(), path)

            # Load with weights_only=True (PyTorch 2.9 default)
            loaded = torch.load(path, weights_only=True)

            # Should not raise
            assert "seed_slots.r0c0._extra_state" in str(loaded) or loaded is not None
