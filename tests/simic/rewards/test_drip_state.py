"""Tests for FossilizedSeedDripState drip calculation."""

import pytest
from esper.simic.rewards.contribution import FossilizedSeedDripState


class TestFossilizedSeedDripState:
    """Unit tests for drip state calculation."""

    def test_remaining_epochs_property(self) -> None:
        """remaining_epochs computed correctly from max_epochs - fossilize_epoch."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=20,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=1.96 / 130,
        )
        assert state.remaining_epochs == 130

    def test_compute_epoch_drip_positive_contribution(self) -> None:
        """Positive contribution yields positive drip, capped at max_drip."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=20,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.015,  # ~1.96 / 130
        )

        drip = state.compute_epoch_drip(
            current_contribution=3.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.015 * 3.0 = 0.045, below cap
        assert drip == pytest.approx(0.045, abs=0.001)

    def test_compute_epoch_drip_positive_clipping(self) -> None:
        """Large positive drip is clipped to max_drip."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=140,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.196,  # 1.96 / 10 (late fossilization)
        )

        drip = state.compute_epoch_drip(
            current_contribution=5.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.196 * 5.0 = 0.98, clipped to 0.1
        assert drip == pytest.approx(0.1, abs=0.001)

    def test_compute_epoch_drip_negative_contribution(self) -> None:
        """Negative contribution yields negative drip (penalty)."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=20,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.015,
        )

        drip = state.compute_epoch_drip(
            current_contribution=-2.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.015 * (-2.0) = -0.03, below asymmetric cap
        assert drip == pytest.approx(-0.03, abs=0.001)
        assert drip < 0, "Negative contribution should produce negative drip"

    def test_compute_epoch_drip_asymmetric_clipping(self) -> None:
        """Large negative drip is clipped asymmetrically (tighter than positive)."""
        state = FossilizedSeedDripState(
            seed_id="test-seed",
            slot_id="r0c1",
            fossilize_epoch=140,
            max_epochs=150,
            drip_total=1.96,
            drip_scale=0.196,
        )

        drip = state.compute_epoch_drip(
            current_contribution=-5.0,
            max_drip=0.1,
            negative_drip_ratio=0.5,
        )

        # 0.196 * (-5.0) = -0.98, clipped to -0.05 (asymmetric)
        assert drip == pytest.approx(-0.05, abs=0.001)
