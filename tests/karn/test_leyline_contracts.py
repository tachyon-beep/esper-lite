"""Tests for Karn's use of leyline contracts."""

import pytest


class TestSeedStageContract:
    """Tests for SeedStage contract compliance."""

    def test_karn_uses_leyline_seedstage(self):
        """Karn should use leyline.SeedStage, not define its own."""
        from esper.leyline import SeedStage as LeylineSeedStage
        from esper.karn.store import SlotSnapshot

        # SlotSnapshot.stage should use the leyline enum
        slot = SlotSnapshot(slot_id="mid")
        assert type(slot.stage).__module__ == "esper.leyline.stages"
        assert isinstance(slot.stage.value, int)  # IntEnum, not Enum

    def test_karn_exports_leyline_seedstage(self):
        """Karn's re-export should be the same as leyline's."""
        from esper.leyline import SeedStage as LeylineSeedStage
        from esper.karn import SeedStage as KarnSeedStage

        assert LeylineSeedStage is KarnSeedStage
