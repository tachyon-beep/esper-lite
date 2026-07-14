"""Obs V4 canonical contribution-state representation (L1: make permanence visible).

Covers the round-13 corrected representation and the byte-identical-V3 guarantee:
- V3 (default) is byte-identical: the slot block is 32 dims, dim-12 keeps the None->0
  coercion, and the presence of the canonical ContributionState does NOT change any V3 value.
- V4 grows the slot block to 35 dims, changes dim-12 to shrink toward the UNKNOWN sentinel as
  staleness rises, and appends explicit observed/frozen/age_norm status dims reaching the policy.
- The four states (never-measured / true-0 / stale / frozen) are distinguishable.
- Lifecycle clearing via seed_generation_id prevents last_valid carry-forward across occupants.
"""

from __future__ import annotations

import pytest
import torch

from esper.leyline import (
    OBS_V3_BASE_FEATURE_SIZE,
    OBS_V3_SLOT_FEATURE_SIZE,
    OBS_V3_UNKNOWN_SENTINEL,
    OBS_V4_SLOT_FEATURE_SIZE,
    OBS_V4_SLOT_OFFSET_CF_AGE_NORM,
    OBS_V4_SLOT_OFFSET_CF_FROZEN,
    OBS_V4_SLOT_OFFSET_CF_OBSERVED,
    ContributionState,
    CounterfactualStatus,
)
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.policy.features import batch_obs_to_features, get_feature_size

MAX_EPOCHS = 100
_CF_VALUE_DIM = 12  # per-slot value dim (counterfactual contribution)


def _make_signals(epoch: int = 10):
    from esper.leyline.signals import TrainingMetrics, TrainingSignals

    metrics = TrainingMetrics(
        epoch=epoch,
        global_step=epoch * 100,
        train_loss=0.45,
        val_loss=0.5,
        loss_delta=-0.02,
        train_accuracy=72.0,
        val_accuracy=70.0,
        accuracy_delta=0.5,
        plateau_epochs=2,
        best_val_accuracy=70.0,
        best_val_loss=0.4,
    )
    return TrainingSignals(
        metrics=metrics,
        loss_history=[0.6, 0.55, 0.5, 0.52, 0.5],
        accuracy_history=[65.0, 66.0, 67.0, 68.0, 70.0],
    )


def _make_report(slot_id: str, *, stage: int = 8, contribution: float | None = 2.5):
    from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
    from esper.leyline.reports import SeedMetrics, SeedStateReport
    from esper.leyline.stages import SeedStage
    from esper.leyline.telemetry import SeedTelemetry

    metrics = SeedMetrics(
        epochs_total=20,
        current_alpha=0.5,
        counterfactual_contribution=contribution,
        contribution_velocity=0.5,
        improvement_since_stage_start=2.5,
        interaction_sum=1.5,
        epochs_in_current_stage=3,
    )
    telemetry = SeedTelemetry(
        seed_id=f"seed_{slot_id}",
        blueprint_id="conv_light",
        gradient_norm=2.5,
        gradient_health=0.95,
        has_vanishing=False,
        has_exploding=False,
    )
    return SeedStateReport(
        seed_id=f"seed_{slot_id}",
        slot_id=slot_id,
        blueprint_id="conv_light",
        blueprint_index=1,
        stage=SeedStage(stage),
        alpha_mode=AlphaMode.UP.value,
        alpha_target=1.0,
        alpha_steps_total=10,
        alpha_steps_done=4,
        time_to_target=6,
        alpha_velocity=0.2,
        alpha_algorithm=AlphaAlgorithm.GATE.value,
        blend_tempo_epochs=8,
        metrics=metrics,
        telemetry=telemetry,
    )


def _make_env_state(slot_ids, *, contribution_states=None, epochs_since=None):
    from esper.simic.training.parallel_env_state import ParallelEnvState

    class _Mock:
        def reset(self):
            pass

    env_state = ParallelEnvState(
        model=_Mock(),
        host_optimizer=_Mock(),
        signal_tracker=_Mock(),
        governor=_Mock(),
    )
    env_state.escrow_credit = {slot_id: 0.0 for slot_id in slot_ids}
    if epochs_since is not None:
        env_state.epochs_since_counterfactual = dict(epochs_since)
    if contribution_states is not None:
        env_state.contribution_states = dict(contribution_states)
    return env_state


def _build(slot_config, reports, env_state, *, obs_v4):
    signals = [_make_signals()]
    return batch_obs_to_features(
        batch_signals=signals,
        batch_slot_reports=[reports],
        batch_env_states=[env_state],
        slot_config=slot_config,
        device=torch.device("cpu"),
        max_epochs=MAX_EPOCHS,
        obs_v4=obs_v4,
    )


# ---------------------------------------------------------------------------
# get_feature_size delta
# ---------------------------------------------------------------------------


def test_get_feature_size_v3_unchanged_v4_delta():
    sc = SlotConfig.default()  # 3 slots
    assert get_feature_size(sc) == 120  # V3, byte-identical
    assert get_feature_size(sc, obs_v4=False) == 120
    assert get_feature_size(sc, obs_v4=True) == 129  # +9 (3 dims x 3 slots)
    assert OBS_V4_SLOT_FEATURE_SIZE == OBS_V3_SLOT_FEATURE_SIZE + 3


# ---------------------------------------------------------------------------
# V3 byte-identical guarantee
# ---------------------------------------------------------------------------


def test_v3_shape_and_none_to_zero_preserved():
    """V3 (default) keeps the historical None->0 coercion in dim-12, shape unchanged."""
    sc = SlotConfig.default()
    reports = {"r0c0": _make_report("r0c0", contribution=None)}
    env = _make_env_state(sc.slot_ids)
    obs, _ = _build(sc, reports, env, obs_v4=False)
    assert obs.shape == (1, 120)
    # None counterfactual -> 0.0 (the V3 bug this replaces, gated OFF and preserved here)
    assert obs[0, OBS_V3_BASE_FEATURE_SIZE + _CF_VALUE_DIM].item() == 0.0


def test_v3_ignores_canonical_state():
    """Populating ContributionState must NOT change any V3 output (byte-identical)."""
    sc = SlotConfig.default()
    reports = {"r0c0": _make_report("r0c0", contribution=2.5)}

    env_empty = _make_env_state(sc.slot_ids)
    obs_empty, _ = _build(sc, reports, env_empty, obs_v4=False)

    # Same inputs but with a fully-populated (and deliberately CONTRADICTORY) canonical state.
    env_pop = _make_env_state(
        sc.slot_ids,
        contribution_states={
            "r0c0": ContributionState(9, CounterfactualStatus.FROZEN_AT_FOSSILIZE, -7.7, 55, False)
        },
    )
    obs_pop, _ = _build(sc, reports, env_pop, obs_v4=False)

    assert torch.equal(obs_empty, obs_pop), "V3 obs must be independent of the canonical state"
    # And dim-12 still reads the report's value via the V3 formula, not the canonical state.
    expected = max(-1.0, min(2.5 / 10.0, 1.0))
    assert obs_pop[0, OBS_V3_BASE_FEATURE_SIZE + _CF_VALUE_DIM].item() == pytest.approx(expected)


def test_v4_reduces_to_v3_on_shared_dims_when_fresh():
    """For a FRESH state, V4 dims 0-11 and 13-31 equal V3, and dim-12 agrees; only 32-34 are new."""
    sc = SlotConfig.default()
    reports = {"r0c0": _make_report("r0c0", contribution=2.5)}
    env_v3 = _make_env_state(sc.slot_ids)
    obs_v3, _ = _build(sc, reports, env_v3, obs_v4=False)

    env_v4 = _make_env_state(
        sc.slot_ids,
        contribution_states={
            "r0c0": ContributionState(1, CounterfactualStatus.FRESH, 2.5, 0, True)
        },
    )
    obs_v4, _ = _build(sc, reports, env_v4, obs_v4=True)
    assert obs_v4.shape == (1, 129)

    base = OBS_V3_BASE_FEATURE_SIZE
    # Base (non-slot) features identical.
    assert torch.equal(obs_v3[:, :base], obs_v4[:, :base])
    # Per slot: dims 0-11 and 13-31 identical; dim-12 agrees (fresh -> last_norm == V3 clamp).
    for i in range(sc.num_slots):
        off_v3 = base + i * OBS_V3_SLOT_FEATURE_SIZE
        off_v4 = base + i * OBS_V4_SLOT_FEATURE_SIZE
        assert torch.allclose(obs_v3[:, off_v3:off_v3 + 12], obs_v4[:, off_v4:off_v4 + 12])
        assert torch.allclose(obs_v3[:, off_v3 + 13:off_v3 + 32], obs_v4[:, off_v4 + 13:off_v4 + 32])
        # dim-12 fresh agreement
        assert obs_v4[0, off_v4 + 12].item() == pytest.approx(obs_v3[0, off_v3 + 12].item())


# ---------------------------------------------------------------------------
# V4 four-state distinction
# ---------------------------------------------------------------------------


def test_v4_four_states_distinguishable():
    sc = SlotConfig.default()  # r0c0, r0c1, r0c2
    reports = {
        "r0c0": _make_report("r0c0", contribution=None),  # never measured
        "r0c1": _make_report("r0c1", contribution=0.0),   # true measured 0
        "r0c2": _make_report("r0c2", contribution=5.0),   # stale
    }
    env = _make_env_state(
        sc.slot_ids,
        contribution_states={
            "r0c0": ContributionState(1, CounterfactualStatus.NEVER_MEASURED, None, 0, False),
            "r0c1": ContributionState(1, CounterfactualStatus.FRESH, 0.0, 0, True),
            "r0c2": ContributionState(1, CounterfactualStatus.STALE, 5.0, 10, False),
        },
    )
    obs, _ = _build(sc, reports, env, obs_v4=True)
    base = OBS_V3_BASE_FEATURE_SIZE
    ss = OBS_V4_SLOT_FEATURE_SIZE

    def slot(i):
        off = base + i * ss
        return (
            obs[0, off + _CF_VALUE_DIM].item(),
            obs[0, off + OBS_V4_SLOT_OFFSET_CF_OBSERVED].item(),
            obs[0, off + OBS_V4_SLOT_OFFSET_CF_FROZEN].item(),
            obs[0, off + OBS_V4_SLOT_OFFSET_CF_AGE_NORM].item(),
        )

    never = slot(0)
    true0 = slot(1)
    stale = slot(2)

    # never-measured -> UNKNOWN sentinel, observed=0, age=1
    assert never == (pytest.approx(OBS_V3_UNKNOWN_SENTINEL), 0.0, 0.0, pytest.approx(1.0))
    # true measured 0 -> value 0, observed=1, age=0 (DISTINCT from never-measured)
    assert true0 == (pytest.approx(0.0), 1.0, 0.0, pytest.approx(0.0))
    assert true0 != never
    # stale -> value shrunk toward sentinel (below the fresh 0.5), observed=1, age in (0,1)
    assert stale[1] == 1.0 and stale[2] == 0.0
    assert 0.0 < stale[3] < 1.0
    assert stale[0] < 0.5  # shrunk from the fresh last_norm=0.5 toward -1


def test_v4_frozen_value_shrinks_and_flags():
    """A frozen value ages toward the sentinel (mirror-bug fix) and sets the frozen flag."""
    sc = SlotConfig.default()
    reports = {"r0c0": _make_report("r0c0", stage=9, contribution=None)}  # fossil: metric None
    base = OBS_V3_BASE_FEATURE_SIZE
    off = base + 0 * OBS_V4_SLOT_FEATURE_SIZE

    def frozen_value_at(epochs):
        env = _make_env_state(
            sc.slot_ids,
            contribution_states={
                "r0c0": ContributionState(1, CounterfactualStatus.FROZEN_AT_FOSSILIZE, 5.0, epochs, False)
            },
        )
        obs, _ = _build(sc, reports, env, obs_v4=True)
        return (
            obs[0, off + _CF_VALUE_DIM].item(),
            obs[0, off + OBS_V4_SLOT_OFFSET_CF_FROZEN].item(),
            obs[0, off + OBS_V4_SLOT_OFFSET_CF_AGE_NORM].item(),
        )

    v_commit, frozen_commit, age_commit = frozen_value_at(0)
    v_old, frozen_old, age_old = frozen_value_at(70)

    assert frozen_commit == 1.0 and frozen_old == 1.0  # frozen flag reaches policy
    assert v_commit == pytest.approx(0.5)  # full magnitude at commit (staleness 0)
    assert v_old < v_commit  # NOT frozen-at-full-magnitude-forever (the mirror bug)
    assert age_old > age_commit  # uncertainty rises with post-commit age


# ---------------------------------------------------------------------------
# ContributionState transitions + lifecycle clearing (no carry-forward)
# ---------------------------------------------------------------------------


def test_contribution_state_transitions_ordering():
    """advance-then-record yields FRESH (measured epoch); advance-only yields STALE (ageing)."""
    s = ContributionState.never_measured(1)
    # measured epoch: legacy increments then measurement resets -> FRESH, epochs 0
    s.advance_epoch()
    s.record_measurement(3.0)
    assert s.counterfactual_status == CounterfactualStatus.FRESH
    assert s.epochs_since_counterfactual == 0
    assert s.last_valid_counterfactual_contribution == 3.0
    # unmeasured epoch: advance only -> STALE, epochs incremented, value held
    s.advance_epoch()
    assert s.counterfactual_status == CounterfactualStatus.STALE
    assert s.epochs_since_counterfactual == 1
    assert s.last_valid_counterfactual_contribution == 3.0


def test_lifecycle_generation_clears_carry_forward():
    """germinate/prune/reuse via seed_generation_id must clear the cached last_valid."""
    env = _make_env_state(["r0c0"])

    # Germinate: NEVER_MEASURED, generation 1.
    env.init_obs_v3_slot_tracking("r0c0")
    st = env.contribution_states["r0c0"]
    assert st.seed_generation_id == 1
    assert st.counterfactual_status == CounterfactualStatus.NEVER_MEASURED
    assert st.last_valid_counterfactual_contribution is None

    # Measure -> FRESH with a value.
    env.record_counterfactual_measurement("r0c0", 6.0)
    assert env.contribution_states["r0c0"].last_valid_counterfactual_contribution == 6.0

    # Prune / slot empties -> canonical state dropped.
    env.clear_obs_v3_slot_tracking("r0c0")
    assert "r0c0" not in env.contribution_states

    # New occupant germinates -> generation 2, NO carry-forward of the +6.0.
    env.init_obs_v3_slot_tracking("r0c0")
    st2 = env.contribution_states["r0c0"]
    assert st2.seed_generation_id == 2
    assert st2.counterfactual_status == CounterfactualStatus.NEVER_MEASURED
    assert st2.last_valid_counterfactual_contribution is None


def test_freeze_at_fossilize_preserves_last_valid():
    env = _make_env_state(["r0c0"])
    env.init_obs_v3_slot_tracking("r0c0")
    env.record_counterfactual_measurement("r0c0", 4.2)
    env.advance_counterfactual_epoch("r0c0")  # STALE, epochs 1
    env.freeze_contribution_at_fossilize("r0c0")
    st = env.contribution_states["r0c0"]
    assert st.counterfactual_status == CounterfactualStatus.FROZEN_AT_FOSSILIZE
    assert st.last_valid_counterfactual_contribution == 4.2  # pre-commit LOO preserved
    assert st.epochs_since_counterfactual == 0  # staleness reset to the commit anchor
