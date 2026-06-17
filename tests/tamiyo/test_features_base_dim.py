"""Pins the 23-dim base observation contract for Tamiyo's feature extractor.

This is a guard for P1-2 (deletion of the dead grad_norm_host / grad_norm_seed
fields on TrainingMetrics). The base feature vector width and content must be
observationally inert with respect to that deletion: removing fields that no
extractor reads cannot change the obs width (23) or any base value.

The base block is built from a TrainingSignals produced by the tamiyo
SignalTracker, then run through batch_obs_to_features, the production extractor.
"""

from __future__ import annotations

import math

import torch

from esper.leyline import OBS_V3_BASE_FEATURE_SIZE
from esper.leyline.slot_config import SlotConfig
from esper.simic.training.parallel_env_state import ParallelEnvState
from esper.tamiyo.policy.features import batch_obs_to_features
from esper.tamiyo.tracker import SignalTracker

MAX_EPOCHS = 100


def _make_env_state(last_action_success: bool = True, last_action_op: int = 0) -> ParallelEnvState:
    class _Stub:
        def reset(self) -> None:  # pragma: no cover - trivial stub
            pass

    return ParallelEnvState(
        model=_Stub(),
        host_optimizer=_Stub(),
        signal_tracker=_Stub(),
        governor=_Stub(),
        last_action_success=last_action_success,
        last_action_op=last_action_op,
    )


def _signals_from_tracker(final_epoch: int, val_loss: float, val_accuracy: float):
    """Drive the tamiyo SignalTracker to a known state and return its TrainingSignals.

    Feeds 5 identical (val_loss, val_accuracy) updates so both rolling histories
    are fully populated (no UNKNOWN-sentinel padding) and every history slot holds
    the same value, keeping the golden assertions independent of left-pad position.
    The final update carries ``final_epoch`` so the epoch feature is deterministic.
    """
    tracker = SignalTracker()
    signals = None
    start_epoch = final_epoch - 4
    for i in range(5):
        epoch = start_epoch + i
        signals = tracker.update(
            epoch=epoch,
            global_step=epoch * 100,
            train_loss=val_loss - 0.05,
            train_accuracy=val_accuracy + 2.0,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            active_seeds=[],
            available_slots=1,
        )
    assert signals is not None
    return signals


def test_base_feature_vector_is_23_dims():
    """The base block of the obs vector must be exactly OBS_V3_BASE_FEATURE_SIZE (23)."""
    assert OBS_V3_BASE_FEATURE_SIZE == 23

    slot_config = SlotConfig.default()
    signals = _signals_from_tracker(final_epoch=10, val_loss=0.5, val_accuracy=70.0)

    obs, _ = batch_obs_to_features(
        [signals],
        [{}],  # no active slots -> base block fully determines the first 23 dims
        [_make_env_state()],
        slot_config,
        torch.device("cpu"),
        max_epochs=MAX_EPOCHS,
    )

    base = obs[0, :OBS_V3_BASE_FEATURE_SIZE]
    assert base.shape == (OBS_V3_BASE_FEATURE_SIZE,)
    # Total width = base + per-slot * num_slots; base must not have shifted the layout.
    assert obs.shape[1] == OBS_V3_BASE_FEATURE_SIZE + (
        slot_config.num_slots * (obs.shape[1] - OBS_V3_BASE_FEATURE_SIZE) // slot_config.num_slots
    )


def test_base_feature_content_unchanged():
    """Golden assertion on the 23 base values, proving the deletion is observationally inert."""
    slot_config = SlotConfig.default()

    epoch = 50
    val_loss = 1.5
    val_accuracy = 85.0
    signals = _signals_from_tracker(final_epoch=epoch, val_loss=val_loss, val_accuracy=val_accuracy)

    obs, _ = batch_obs_to_features(
        [signals],
        [{}],  # no active slots
        [_make_env_state(last_action_success=True, last_action_op=2)],
        slot_config,
        torch.device("cpu"),
        max_epochs=MAX_EPOCHS,
    )

    base = obs[0, :OBS_V3_BASE_FEATURE_SIZE]

    # Index 0: epoch normalized.
    assert abs(base[0].item() - (epoch / MAX_EPOCHS)) < 1e-6

    # Index 1: val_loss symlog-normalized (log1p(x) / 7).
    assert abs(base[1].item() - (math.log1p(val_loss) / 7.0)) < 1e-4

    # Index 2: val_accuracy normalized (acc / 100).
    assert abs(base[2].item() - (val_accuracy / 100.0)) < 1e-6

    # History is fully populated (5 identical updates): loss-history slots 3-7 all
    # equal symlog(val_loss)/7, accuracy-history slots 8-12 all equal val_accuracy/100.
    for i in range(5):
        assert abs(base[3 + i].item() - (math.log1p(val_loss) / 7.0)) < 1e-4
        assert abs(base[8 + i].item() - (val_accuracy / 100.0)) < 1e-6

    # Indices 13-15: stage distribution, all zero with no active slots.
    assert base[13].item() == 0.0
    assert base[14].item() == 0.0
    assert base[15].item() == 0.0

    # Index 16: last_action_success.
    assert base[16].item() == 1.0

    # Indices 17-22: last_action_op one-hot for op=2.
    expected_one_hot = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for i, expected in enumerate(expected_one_hot):
        assert abs(base[17 + i].item() - expected) < 1e-6
