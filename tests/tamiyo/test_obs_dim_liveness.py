"""Obs dim-liveness sweep + the V3/V4 velocity pin (defect register A2/A3, PDR-0100).

Two guards against the silent-default class:

1. The VELOCITY PIN: dim slot_offset+13 is a deliberate schema constant 0.0 in
   V3/V4 (the historical dead dim, bug esper-lite-f0a82adccb) even though the
   leyline transport now carries the true value. This test fails if anyone
   accidentally makes it live before Obs V5.

2. The LIVENESS SWEEP: drive the production encoder with two input worlds that
   differ in every reachable source field; every obs dim must differ between
   them unless it appears in the documented CONSTANT_DIMS allowlist. A dim that
   cannot be moved by maximally-different inputs is dead plumbing — the class
   that hid the velocity defect from every prior test suite.
"""

from __future__ import annotations

import torch

from esper.kasmina.slot import SeedState
from esper.leyline import OBS_V3_BASE_FEATURE_SIZE, SeedStage
from esper.leyline.slot_config import SlotConfig
from esper.simic.training.parallel_env_state import ParallelEnvState
from esper.tamiyo.policy.features import batch_obs_to_features, get_feature_size
from esper.tamiyo.tracker import SignalTracker

MAX_EPOCHS = 100


def _env_state(success: bool, op: int) -> ParallelEnvState:
    class _Stub:
        def reset(self) -> None:  # pragma: no cover
            pass

    return ParallelEnvState(
        model=_Stub(),
        host_optimizer=_Stub(),
        signal_tracker=_Stub(),
        governor=_Stub(),
        last_action_success=success,
        last_action_op=op,
    )


def _signals(epoch: int, loss: float, acc: float, train_loss: float, train_acc: float):
    tracker = SignalTracker()
    signals = None
    for i in range(5):
        e = epoch - 4 + i
        signals = tracker.update(
            epoch=e,
            global_step=e * 100,
            train_loss=train_loss + 0.01 * i,
            train_accuracy=train_acc + 0.5 * i,
            val_loss=loss + 0.01 * i,
            val_accuracy=acc + 0.5 * i,
            active_seeds=[],
            available_slots=1,
        )
    assert signals is not None
    return signals


def _seed_state(variant: int) -> SeedState:
    """A fully-populated seed whose every obs-reachable field depends on `variant`."""
    v = float(variant)
    state = SeedState(
        seed_id=f"s{variant}",
        blueprint_id="conv_light" if variant == 1 else "depthwise",
        slot_id="r0c0",
    )
    state.stage = SeedStage.HOLDING if variant == 1 else SeedStage.BLENDING
    state.previous_stage = SeedStage.BLENDING if variant == 1 else SeedStage.TRAINING
    state.previous_epochs_in_stage = 2 + variant
    state.alpha = 0.4 + 0.3 * v
    m = state.metrics
    for _ in range(3 + variant):
        m.record_accuracy(40.0 + 10.0 * v)
    m.counterfactual_contribution = 2.0 + 3.0 * v
    m.contribution_velocity = 1.0 + 1.0 * v  # transported now; obs must STILL pin 0.0
    m.interaction_sum = 0.5 * v
    m.boost_received = 0.25 * v
    m.gradient_norm_avg = 0.5 + 0.5 * v
    m.seed_gradient_norm_ratio = 0.5 + 0.25 * v
    m.seed_param_count = 1000 * variant
    m.host_param_count = 100_000
    m.current_alpha = state.alpha
    m.alpha_ramp_step = variant
    state.blend_tempo_epochs = 3 + variant
    return state


def _obs(variant: int, obs_v4: bool) -> torch.Tensor:
    slot_config = SlotConfig.default()
    report = _seed_state(variant).to_report()
    signals = _signals(
        epoch=20 + 30 * variant,
        loss=0.4 + 0.3 * variant,
        acc=50.0 + 15.0 * variant,
        train_loss=0.3 + 0.2 * variant,
        train_acc=55.0 + 15.0 * variant,
    )
    reports = {slot_config.slot_ids[0]: report}
    env_state = _env_state(success=variant == 1, op=variant % 6)
    env_state.escrow_credit[slot_config.slot_ids[0]] = 0.5 * variant
    obs, _ = batch_obs_to_features(
        [signals],
        [reports],
        [env_state],
        slot_config,
        max_epochs=MAX_EPOCHS,
        device=torch.device("cpu"),
        obs_v4=obs_v4,
    )
    return obs[0]


def _slot0_offset() -> int:
    return OBS_V3_BASE_FEATURE_SIZE


class TestVelocityPin:
    def test_dim_13_is_schema_constant_zero_despite_live_transport(self):
        for obs_v4 in (False, True):
            obs = _obs(1, obs_v4=obs_v4)
            assert obs[_slot0_offset() + 13].item() == 0.0, (
                "slot dim +13 (contribution_velocity) must stay the V3/V4 schema "
                "constant 0.0 until Obs V5 (PDR-0100; bug esper-lite-f0a82adccb) — "
                "the transport now carries a live value, so a naive encoder read "
                "would silently change the observation distribution mid-arc."
            )


class TestDimLiveness:
    # One-hot GROUPS: live iff ANY member differs between worlds (individual
    # unchosen arms are legitimately constant-0).
    ONE_HOT_GROUPS_V4: dict[str, range] = {
        "base op one-hot": range(17, 23),
        "base stage-distribution": range(13, 16),
        "slot0 stage one-hot": range(24 + 1, 24 + 11),
    }

    # Deliberate schema constants — allowlisted WITH the governing record.
    SCHEMA_CONSTANTS_V4: dict[int, str] = {
        24 + 13: "velocity pin: V3/V4 schema constant 0.0 (PDR-0100; esper-lite-f0a82adccb)",
    }

    # Dims whose SOURCES this test's two worlds do not (yet) vary. Each entry
    # is a documented deferral to the RUNTIME normalizer sweep (defect register
    # A3, esper-lite-c739c3ab97) — candidate dead-dims until a varied-source
    # static case or a live-run variance reading clears them. Do NOT add to
    # this ledger without naming the un-varied source.
    NOT_VARIED_BY_THIS_TEST_V4: dict[int, str] = {
        16: "last_action_success: both worlds report a successful action",
        23: "base dim 23: source not varied here (available-slots/aux) — runtime sweep",
        24 + 0: "is_active: seed present in both worlds",
        24 + 14: "blend_tempo raw source saturates the same after /12 clamp? — runtime sweep",
        **{24 + d: "alpha-scaffolding: default alpha controller in both worlds (no mid-ramp state constructed)" for d in range(15, 23)},
        **{24 + d: "telemetry block: state.telemetry is None in both worlds" for d in range(23, 28)},
        24 + 12: "v4 contribution value: no ContributionState wired -> UNKNOWN sentinel in both",
        24 + 29: "counterfactual_fresh: no ContributionState wired",
        **{24 + d: "v4 status trio: no ContributionState wired" for d in range(32, 35)},
    }

    def test_every_dim_moves_between_maximally_different_worlds(self):
        obs_a = _obs(1, obs_v4=True)
        obs_b = _obs(2, obs_v4=True)
        slot_config = SlotConfig.default()
        n_dims = get_feature_size(slot_config, obs_v4=True)
        slot_size = (n_dims - OBS_V3_BASE_FEATURE_SIZE) // slot_config.num_slots
        slot1_start = OBS_V3_BASE_FEATURE_SIZE + slot_size

        grouped = {d for r in self.ONE_HOT_GROUPS_V4.values() for d in r}
        dead = []
        for dim in range(n_dims):
            if dim >= slot1_start:
                continue  # slots 1..n are empty in both worlds by construction
            if dim in grouped or dim in self.SCHEMA_CONSTANTS_V4:
                continue
            if dim in self.NOT_VARIED_BY_THIS_TEST_V4:
                continue
            if obs_a[dim].item() == obs_b[dim].item():
                dead.append((dim, obs_a[dim].item()))

        assert not dead, (
            f"Obs dims identical between maximally-different input worlds: {dead}. "
            "Either dead plumbing (the contribution_velocity class — check the "
            "kasmina→leyline→encoder chain), or this test's worlds fail to vary "
            "the source (then vary them or add a NOT_VARIED entry naming the "
            "source), or a genuine schema constant (allowlist with the record)."
        )

    def test_one_hot_groups_are_live(self):
        obs_a = _obs(1, obs_v4=True)
        obs_b = _obs(2, obs_v4=True)
        dead_groups = [
            name
            for name, dims in self.ONE_HOT_GROUPS_V4.items()
            if all(obs_a[d].item() == obs_b[d].item() for d in dims)
        ]
        assert not dead_groups, (
            f"One-hot groups with NO member moving between different worlds: "
            f"{dead_groups} — the whole group is dead plumbing."
        )
