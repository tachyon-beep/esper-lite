# tests/simic/properties/test_warning_signals.py
"""Warning signal property tests.

These properties verify that warning signals (blending_warning, holding_warning)
fire correctly and provide proper credit assignment.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    compute_contribution_reward,
    SeedInfo,
    STAGE_BLENDING,
    STAGE_HOLDING,
)



@pytest.mark.property
class TestBlendingWarning:
    """Blending warning provides early signal to CULL bad seeds."""

    @given(
        total_improvement=st.floats(-3.0, -0.1, allow_nan=False),
        epochs_in_stage=st.integers(1, 10),
    )
    @settings(max_examples=200)
    def test_negative_trajectory_warned(self, total_improvement, epochs_in_stage):
        """Seeds with negative trajectory in BLENDING should get warning."""
        seed_info = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=-0.5,
            total_improvement=total_improvement,
            epochs_in_stage=epochs_in_stage,
            seed_params=50_000,
            previous_stage=SeedStage.TRAINING.value,
            previous_epochs_in_stage=3,
            seed_age_epochs=epochs_in_stage + 5,
        )

        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=1.0,  # Some contribution
            val_acc=70.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=72.0,  # Was better before
            acc_delta=-0.1,
            return_components=True,
        )

        assert components.blending_warning < 0, (
            f"Negative trajectory in BLENDING should warn, got {components.blending_warning}"
        )

    @given(epochs_in_stage=st.integers(1, 10))
    @settings(max_examples=100)
    def test_warning_escalates_with_time(self, epochs_in_stage):
        """Warning should escalate the longer negative trajectory persists."""
        def get_warning(epochs: int) -> float:
            seed_info = SeedInfo(
                stage=STAGE_BLENDING,
                improvement_since_stage_start=-0.5,
                total_improvement=-1.0,
                epochs_in_stage=epochs,
                seed_params=50_000,
                previous_stage=SeedStage.TRAINING.value,
                previous_epochs_in_stage=3,
                seed_age_epochs=epochs + 5,
            )

            _, components = compute_contribution_reward(
                action=LifecycleOp.WAIT,
                seed_contribution=1.0,
                val_acc=70.0,
                seed_info=seed_info,
                epoch=10,
                max_epochs=25,
                total_params=150_000,
                host_params=100_000,
                acc_at_germination=72.0,
                acc_delta=-0.1,
                return_components=True,
            )
            return components.blending_warning

        # Warning should be more negative with more epochs
        warning_early = get_warning(1)
        warning_late = get_warning(min(epochs_in_stage + 3, 10))

        assert warning_late <= warning_early, (
            f"Warning should escalate: epoch 1 = {warning_early}, "
            f"epoch {epochs_in_stage + 3} = {warning_late}"
        )


@pytest.mark.property
class TestHoldingWarning:
    """Holding warning creates urgency to make FOSSILIZE/CULL decision."""

    @given(epochs_in_stage=st.integers(2, 8))
    @settings(max_examples=100)
    def test_wait_in_holding_penalized(self, epochs_in_stage):
        """WAITing in HOLDING with positive attribution should be penalized."""
        seed_info = SeedInfo(
            stage=STAGE_HOLDING,
            improvement_since_stage_start=0.5,
            total_improvement=2.0,  # Positive trajectory
            epochs_in_stage=epochs_in_stage,
            seed_params=50_000,
            previous_stage=STAGE_BLENDING,
            previous_epochs_in_stage=5,
            seed_age_epochs=epochs_in_stage + 10,
        )

        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=3.0,  # Good seed
            val_acc=75.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=65.0,
            acc_delta=0.3,
            return_components=True,
        )

        # Warning should be negative (penalty for indecision)
        assert components.holding_warning < 0, (
            f"WAIT in HOLDING epoch {epochs_in_stage} should be penalized, "
            f"got {components.holding_warning}"
        )

    @given(epochs_in_stage=st.integers(2, 6))
    @settings(max_examples=100)
    def test_holding_warning_exponential(self, epochs_in_stage):
        """Holding warning should escalate exponentially."""
        def get_warning(epochs: int) -> float:
            seed_info = SeedInfo(
                stage=STAGE_HOLDING,
                improvement_since_stage_start=0.5,
                total_improvement=2.0,
                epochs_in_stage=epochs,
                seed_params=50_000,
                previous_stage=STAGE_BLENDING,
                previous_epochs_in_stage=5,
                seed_age_epochs=epochs + 10,
            )

            _, components = compute_contribution_reward(
                action=LifecycleOp.WAIT,
                seed_contribution=3.0,
                val_acc=75.0,
                seed_info=seed_info,
                epoch=15,
                max_epochs=25,
                total_params=150_000,
                host_params=100_000,
                acc_at_germination=65.0,
                acc_delta=0.3,
                return_components=True,
            )
            return components.holding_warning

        # Compare consecutive epochs
        warning_n = get_warning(epochs_in_stage)
        warning_n1 = get_warning(epochs_in_stage + 1)

        # Should be more negative (exponential escalation)
        if warning_n < 0 and warning_n1 < 0:
            ratio = warning_n1 / warning_n
            # Exponential: should be roughly 3x per epoch (capped at -10)
            assert ratio >= 1.5 or warning_n1 <= -10.0, (
                f"Warning escalation ratio {ratio:.2f} too low "
                f"(epoch {epochs_in_stage}: {warning_n}, epoch {epochs_in_stage+1}: {warning_n1})"
            )
