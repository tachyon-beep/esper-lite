"""Property-based tests for Tamiyo feature extraction (Obs V3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.leyline import (
    DEFAULT_GAMMA,
    NUM_OPS,
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_SLOT_FEATURE_SIZE,
)
from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.reports import SeedMetrics, SeedStateReport
from esper.leyline.signals import TrainingMetrics, TrainingSignals
from esper.leyline.stages import SeedStage
from esper.leyline.telemetry import SeedTelemetry
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.policy.features import _pad_history, batch_obs_to_features, symlog, symlog_tensor

pytestmark = pytest.mark.property


@given(x=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@settings(max_examples=100)
def test_symlog_is_odd(x: float) -> None:
    assert symlog(-x) == pytest.approx(-symlog(x), rel=1e-7, abs=1e-7)


@given(
    a=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_symlog_is_monotone(a: float, b: float) -> None:
    lo, hi = (a, b) if a <= b else (b, a)
    assert symlog(lo) <= symlog(hi)


@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=64,
    )
)
@settings(max_examples=75)
def test_symlog_tensor_matches_scalar(values: list[float]) -> None:
    x = torch.tensor(values, dtype=torch.float32)
    expected = torch.tensor([symlog(float(v)) for v in values], dtype=torch.float32)
    assert torch.allclose(symlog_tensor(x), expected, atol=1e-6)


@given(
    history=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=10,
    ),
    length=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_pad_history_returns_fixed_length(history: list[float], length: int) -> None:
    padded = _pad_history(history, length)
    assert len(padded) == length

    if len(history) >= length:
        assert padded == history[-length:]
    else:
        assert padded[: length - len(history)] == [0.0] * (length - len(history))
        assert padded[length - len(history) :] == history


@dataclass(slots=True)
class EnvStateStub:
    last_action_success: bool
    last_action_op: int
    gradient_health_prev: dict[str, float] = field(default_factory=dict)
    epochs_since_counterfactual: dict[str, int] = field(default_factory=dict)


@st.composite
def _small_slot_config(draw: st.DrawFn) -> SlotConfig:
    rows, cols = draw(
        st.sampled_from(
            [
                (1, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 1),
                (2, 2),
                (3, 1),
                (4, 1),
            ]
        )
    )
    return SlotConfig.for_grid(rows, cols)


@st.composite
def _obs_v3_batches(draw: st.DrawFn):
    slot_config = draw(_small_slot_config())
    num_slots = slot_config.num_slots
    n_envs = draw(st.integers(min_value=1, max_value=4))
    max_epochs = draw(st.integers(min_value=1, max_value=200))

    batch_signals: list[TrainingSignals] = []
    batch_slot_reports: list[dict[str, SeedStateReport]] = []
    batch_env_states: list[EnvStateStub] = []

    for env_idx in range(n_envs):
        epoch = draw(st.integers(min_value=0, max_value=max_epochs))
        val_loss = draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        )
        val_accuracy = draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        )

        metrics = TrainingMetrics(
            epoch=epoch,
            global_step=epoch,
            train_loss=val_loss,
            val_loss=val_loss,
            loss_delta=0.0,
            train_accuracy=val_accuracy,
            val_accuracy=val_accuracy,
            accuracy_delta=0.0,
            plateau_epochs=0,
            host_stabilized=0,
            best_val_accuracy=val_accuracy,
            best_val_loss=val_loss,
        )

        signals = TrainingSignals(
            metrics=metrics,
            loss_history=draw(
                st.lists(
                    st.floats(
                        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
                    ),
                    min_size=0,
                    max_size=10,
                )
            ),
            accuracy_history=draw(
                st.lists(
                    st.floats(
                        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
                    ),
                    min_size=0,
                    max_size=10,
                )
            ),
            available_slots=num_slots,
        )

        env_state = EnvStateStub(
            last_action_success=draw(st.booleans()),
            last_action_op=draw(st.integers(min_value=0, max_value=NUM_OPS - 1)),
        )

        reports: dict[str, SeedStateReport] = {}
        for slot_id in slot_config.slot_ids:
            is_active = draw(st.booleans())
            if not is_active:
                continue

            stage = draw(
                st.sampled_from(
                    [
                        SeedStage.GERMINATED,
                        SeedStage.TRAINING,
                        SeedStage.BLENDING,
                        SeedStage.HOLDING,
                        SeedStage.FOSSILIZED,
                        SeedStage.PRUNED,
                        SeedStage.EMBARGOED,
                        SeedStage.RESETTING,
                    ]
                )
            )

            seed_alpha = draw(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            improvement = draw(
                st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
            )
            velocity = draw(
                st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
            )
            interaction_sum = draw(
                st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
            )

            counterfactual = draw(
                st.one_of(
                    st.none(),
                    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
                )
            )

            epochs_in_stage = draw(st.integers(min_value=0, max_value=max_epochs))
            seed_age = draw(st.integers(min_value=0, max_value=max_epochs))

            seed_metrics = SeedMetrics(
                epochs_total=seed_age,
                epochs_in_current_stage=epochs_in_stage,
                current_alpha=seed_alpha,
                counterfactual_contribution=counterfactual,
                improvement_since_stage_start=improvement,
                contribution_velocity=velocity,
                interaction_sum=interaction_sum,
            )

            alpha_target = draw(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
            )
            alpha_steps_total = draw(st.integers(min_value=0, max_value=max_epochs))
            alpha_steps_done = draw(st.integers(min_value=0, max_value=alpha_steps_total))
            time_to_target = alpha_steps_total - alpha_steps_done
            alpha_velocity = draw(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
            )

            telemetry = None
            include_telemetry = draw(st.booleans())
            if include_telemetry:
                telemetry = SeedTelemetry(
                    seed_id=f"seed_{env_idx}_{slot_id}",
                    blueprint_id="test",
                    gradient_norm=draw(
                        st.floats(
                            min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False
                        )
                    ),
                    gradient_health=draw(
                        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
                    ),
                    has_vanishing=draw(st.booleans()),
                    has_exploding=draw(st.booleans()),
                )

            report = SeedStateReport(
                seed_id=f"seed_{env_idx}_{slot_id}",
                slot_id=slot_id,
                blueprint_id="test",
                blueprint_index=draw(st.integers(min_value=0, max_value=128)),
                stage=stage,
                alpha_mode=draw(st.sampled_from([m.value for m in AlphaMode])),
                alpha_target=alpha_target,
                alpha_steps_total=alpha_steps_total,
                alpha_steps_done=alpha_steps_done,
                time_to_target=time_to_target,
                alpha_velocity=alpha_velocity,
                alpha_algorithm=draw(st.sampled_from([a.value for a in AlphaAlgorithm])),
                blend_tempo_epochs=draw(st.integers(min_value=0, max_value=20)),
                metrics=seed_metrics,
                telemetry=telemetry,
            )
            reports[slot_id] = report

            if draw(st.booleans()):
                env_state.gradient_health_prev[slot_id] = draw(
                    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
                )
            if draw(st.booleans()):
                env_state.epochs_since_counterfactual[slot_id] = draw(
                    st.integers(min_value=0, max_value=max_epochs)
                )

        batch_signals.append(signals)
        batch_slot_reports.append(reports)
        batch_env_states.append(env_state)

    return batch_signals, batch_slot_reports, batch_env_states, slot_config, max_epochs


@given(batch=_obs_v3_batches())
@settings(max_examples=25)
def test_batch_obs_to_features_shape_and_core_invariants(batch) -> None:
    batch_signals, batch_slot_reports, batch_env_states, slot_config, max_epochs = batch

    obs, blueprint_indices = batch_obs_to_features(
        batch_signals,
        batch_slot_reports,
        batch_env_states,
        slot_config,
        torch.device("cpu"),
        max_epochs=max_epochs,
    )

    n_envs = len(batch_signals)
    num_slots = slot_config.num_slots
    expected_obs_dim = OBS_V3_BASE_FEATURE_SIZE + OBS_V3_SLOT_FEATURE_SIZE * num_slots

    assert obs.shape == (n_envs, expected_obs_dim)
    assert blueprint_indices.shape == (n_envs, num_slots)
    assert blueprint_indices.dtype == torch.int64
    assert torch.isfinite(obs).all()

    for env_idx in range(n_envs):
        reports = batch_slot_reports[env_idx]
        env_state = batch_env_states[env_idx]

        num_training = sum(1 for r in reports.values() if r.stage == SeedStage.TRAINING)
        num_blending = sum(1 for r in reports.values() if r.stage == SeedStage.BLENDING)
        num_holding = sum(
            1
            for r in reports.values()
            if r.stage in (SeedStage.HOLDING, SeedStage.FOSSILIZED)
        )

        assert obs[env_idx, 13].item() == pytest.approx(num_training / num_slots)
        assert obs[env_idx, 14].item() == pytest.approx(num_blending / num_slots)
        assert obs[env_idx, 15].item() == pytest.approx(num_holding / num_slots)

        assert obs[env_idx, 16].item() == (1.0 if env_state.last_action_success else 0.0)

        one_hot = obs[env_idx, 17:23]
        assert one_hot.sum().item() == pytest.approx(1.0)
        assert one_hot[env_state.last_action_op].item() == pytest.approx(1.0)

        for slot_idx, slot_id in enumerate(slot_config.slot_ids):
            slot_offset = OBS_V3_BASE_FEATURE_SIZE + slot_idx * OBS_V3_SLOT_FEATURE_SIZE
            if slot_id not in reports:
                assert blueprint_indices[env_idx, slot_idx].item() == -1
                assert obs[env_idx, slot_offset : slot_offset + OBS_V3_SLOT_FEATURE_SIZE].sum().item() == 0.0
                continue

            report = reports[slot_id]
            assert blueprint_indices[env_idx, slot_idx].item() == report.blueprint_index
            assert obs[env_idx, slot_offset].item() == pytest.approx(1.0)

            stage_one_hot = obs[env_idx, slot_offset + 1 : slot_offset + 11]
            assert stage_one_hot.sum().item() == pytest.approx(1.0)

            improvement_norm = obs[env_idx, slot_offset + 12].item()
            velocity_norm = obs[env_idx, slot_offset + 13].item()
            assert -1.0 <= improvement_norm <= 1.0
            assert -1.0 <= velocity_norm <= 1.0

            expected_prev = (
                env_state.gradient_health_prev[slot_id]
                if slot_id in env_state.gradient_health_prev
                else 1.0
            )
            assert obs[env_idx, slot_offset + 27].item() == pytest.approx(expected_prev)

            expected_epochs_since_cf = (
                env_state.epochs_since_counterfactual[slot_id]
                if slot_id in env_state.epochs_since_counterfactual
                else 0
            )
            assert obs[env_idx, slot_offset + 29].item() == pytest.approx(
                DEFAULT_GAMMA ** expected_epochs_since_cf
            )

