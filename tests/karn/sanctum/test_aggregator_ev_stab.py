"""EV-stab Stage 0/2 telemetry: aggregator -> schema -> snapshot_copy round trips.

Covers the Sanctum mirror of the branch's new PPOUpdatePayload fields:
- per-stream EV + cf-head loss (HRA leg: ev_main / ev_cf / cf_value_loss)
- value-free return-variance shares (RVT leg: return_var_*_share)
- per-head advantage normalization observability
- leg-identity latches derived from field presence
- proof_profile capture on RunConfig

Deliberately NOT mirrored (PDR-0028): cov_rcf_return_share / r_main_cov are
V_cf-contaminated Karn-only diagnostics and must never reach TamiyoState.
"""

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import PPOUpdatePayload, TrainingStartedPayload


def _ppo_event(**overrides: object) -> TelemetryEvent:
    """Minimal valid PPO_UPDATE_COMPLETED event with optional field overrides."""
    kwargs: dict[str, object] = dict(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        grad_norm=1.0,
        kl_divergence=0.01,
        clip_fraction=0.1,
        nan_grad_count=0,
        explained_variance=0.3,
    )
    kwargs.update(overrides)
    return TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(**kwargs),  # type: ignore[arg-type]
    )


def _training_started_event(**overrides: object) -> TelemetryEvent:
    kwargs: dict[str, object] = dict(
        n_envs=1,
        max_epochs=25,
        max_batches=100,
        task="mnist",
        host_params=1_000_000,
        slot_ids=("r0c0",),
        seed=42,
        n_episodes=100,
        lr=3e-4,
        clip_ratio=0.2,
        entropy_coef=0.01,
        param_budget=500_000,
        policy_device="cuda:0",
        env_devices=("cuda:0",),
        reward_mode="shaped",
    )
    kwargs.update(overrides)
    return TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data=TrainingStartedPayload(**kwargs),  # type: ignore[arg-type]
    )


def test_ev_stab_fields_round_trip_through_snapshot_copy():
    """HRA + RVT fields survive aggregator -> schema -> snapshot_copy intact."""
    agg = SanctumAggregator(num_envs=4)

    agg.process_event(
        _ppo_event(
            cf_value_loss=0.021,
            ev_main=0.42,
            ev_cf=0.11,
            return_var_cf_share=0.53,
            return_var_main_share=0.46,
            return_var_residual_share=0.01,
        )
    )

    tamiyo = agg.get_snapshot().tamiyo

    assert tamiyo.cf_value_loss is not None and abs(tamiyo.cf_value_loss - 0.021) < 1e-9
    assert tamiyo.ev_main is not None and abs(tamiyo.ev_main - 0.42) < 1e-9
    assert tamiyo.ev_cf is not None and abs(tamiyo.ev_cf - 0.11) < 1e-9
    assert tamiyo.return_var_cf_share is not None
    assert abs(tamiyo.return_var_cf_share - 0.53) < 1e-9
    assert tamiyo.return_var_main_share is not None
    assert abs(tamiyo.return_var_main_share - 0.46) < 1e-9
    assert tamiyo.return_var_residual_share is not None
    assert abs(tamiyo.return_var_residual_share - 0.01) < 1e-9

    # Histories for sparklines
    assert list(tamiyo.ev_main_history) == [0.42]
    assert list(tamiyo.ev_cf_history) == [0.11]
    assert list(tamiyo.cf_value_loss_history) == [0.021]
    assert list(tamiyo.return_var_cf_share_history) == [0.53]

    # Presence latches leg identity
    assert tamiyo.hra_leg_active is True
    assert tamiyo.rvt_leg_active is True


def test_ev_stab_fields_absent_on_off_legs():
    """Flags-off runs show None scalars, empty histories, and inactive legs."""
    agg = SanctumAggregator(num_envs=4)
    agg.process_event(_ppo_event())

    tamiyo = agg.get_snapshot().tamiyo

    assert tamiyo.cf_value_loss is None
    assert tamiyo.ev_main is None
    assert tamiyo.ev_cf is None
    assert tamiyo.return_var_cf_share is None
    assert tamiyo.return_var_main_share is None
    assert tamiyo.return_var_residual_share is None
    assert len(tamiyo.ev_main_history) == 0
    assert len(tamiyo.return_var_cf_share_history) == 0
    assert tamiyo.hra_leg_active is False
    assert tamiyo.rvt_leg_active is False


def test_leg_flags_latch_across_updates():
    """Once a leg emits, its identity flag stays latched even if a later event omits fields."""
    agg = SanctumAggregator(num_envs=4)

    agg.process_event(_ppo_event(ev_main=0.40, ev_cf=0.10, cf_value_loss=0.02))
    agg.process_event(_ppo_event())  # e.g. an event predating the fields

    tamiyo = agg.get_snapshot().tamiyo

    assert tamiyo.hra_leg_active is True  # latched
    assert tamiyo.ev_main is None  # latest update did not carry it (honest mirror)
    assert list(tamiyo.ev_main_history) == [0.40]  # history retains emitted values


def test_ev_stab_histories_are_isolated_copies():
    """snapshot_copy must copy the new deques, not alias them (thread safety)."""
    agg = SanctumAggregator(num_envs=4)

    agg.process_event(_ppo_event(ev_main=0.40, return_var_cf_share=0.50))
    snapshot = agg.get_snapshot()

    # Mutate aggregator state AFTER the copy
    agg.process_event(_ppo_event(ev_main=0.44, return_var_cf_share=0.52))

    assert list(snapshot.tamiyo.ev_main_history) == [0.40]
    assert list(snapshot.tamiyo.return_var_cf_share_history) == [0.50]


def test_per_head_advantage_norm_observability_round_trip():
    agg = SanctumAggregator(num_envs=4)

    agg.process_event(
        _ppo_event(
            advantage_per_head_normalized=True,
            advantage_norm_fellback_count=2,
            min_sparse_head_advantage_std=0.04,
        )
    )

    tamiyo = agg.get_snapshot().tamiyo

    assert tamiyo.advantage_per_head_normalized is True
    assert tamiyo.advantage_norm_fellback_count == 2
    assert abs(tamiyo.min_sparse_head_advantage_std - 0.04) < 1e-9


def test_contaminated_diagnostics_not_mirrored_to_tamiyo_state():
    """PDR-0028: the V_cf-contaminated shares are Karn-only, never on TamiyoState."""
    tamiyo = SanctumAggregator(num_envs=1).get_snapshot().tamiyo

    assert not hasattr(tamiyo, "cov_rcf_return_share")
    assert not hasattr(tamiyo, "r_main_cov")
    # ev_sum is redundant on-screen (== explained_variance on the ON leg): Karn-only.
    assert not hasattr(tamiyo, "ev_sum")


def test_proof_profile_captured_in_run_config():
    agg = SanctumAggregator(num_envs=1)

    agg.process_event(_training_started_event(proof_profile="ev-stab-s2"))

    assert agg.get_snapshot().run_config.proof_profile == "ev-stab-s2"


def test_proof_profile_defaults_to_none():
    agg = SanctumAggregator(num_envs=1)

    agg.process_event(_training_started_event())

    assert agg.get_snapshot().run_config.proof_profile is None
