"""Tests for blueprint-health proof baseline planning."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from esper.leyline import (
    AlphaTargetAction,
    BlueprintAction,
    DEFAULT_MIN_BLENDING_EPOCHS,
    GerminationStyle,
    LifecycleOp,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    TEMPO_TO_EPOCHS,
    TempoAction,
)
from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_EXPECTED_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
    STATIC_FINAL_SOURCE_BLEND_RAMP_EPOCHS,
    STATIC_FINAL_SOURCE_COHORT_ID,
    STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH,
    STATIC_FINAL_SOURCE_GERMINATE_EPOCH,
    STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
    STATIC_FINAL_SOURCE_MODE,
    STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH,
    STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH,
    STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH,
    STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT,
    STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
    STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS,
    STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
    STATIC_FINAL_SOURCE_TOPOLOGY_V1,
    STATIC_FINAL_SOURCE_TRAINING_DWELL_EPOCHS,
)
from esper.leyline.telemetry import TopologyManifestPayload
from esper.simic.rewards import RewardMode
from esper.simic.training.proof_baselines import (
    ProofBaselineMode,
    build_blueprint_health_proof_plan,
    missing_required_baseline_modes,
    train_blueprint_health_proof_baselines,
)
from esper.simic.training.vectorized_trainer import apply_proof_baseline_action_controls


def _all_valid_masks(batch_size: int = 2, num_slots: int = 3) -> dict[str, torch.Tensor]:
    return {
        "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool),
        "slot": torch.ones(batch_size, num_slots, dtype=torch.bool),
        "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool),
        "style": torch.ones(batch_size, NUM_STYLES, dtype=torch.bool),
        "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool),
        "alpha_target": torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool),
        "alpha_speed": torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool),
        "alpha_curve": torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool),
        "slot_by_op": torch.ones(batch_size, NUM_OPS, num_slots, dtype=torch.bool),
    }


def _assert_only_index(mask: torch.Tensor, index: int) -> None:
    assert mask[:, index].all()
    without_index = mask.clone()
    without_index[:, index] = False
    assert not without_index.any()


def test_blueprint_health_plan_declares_required_control_modes() -> None:
    plan = build_blueprint_health_proof_plan(
        primary_reward_mode=RewardMode.SHAPED,
        comparison_reward_mode=RewardMode.SIMPLIFIED,
        base_seed=100,
    )

    modes = tuple(cohort.mode for cohort in plan.cohorts)
    assert modes == (
        ProofBaselineMode.OFF_SWITCH,
        ProofBaselineMode.STATIC_INITIAL,
        ProofBaselineMode.STATIC_FINAL,
        ProofBaselineMode.FIXED_SCHEDULE,
        ProofBaselineMode.LOCKSTEP_REWARD_AB,
        ProofBaselineMode.LOCKSTEP_REWARD_AB,
    )
    assert tuple(cohort.cohort_id for cohort in plan.cohorts) == (
        "off_switch",
        "static_initial",
        "static_final",
        "fixed_schedule",
        "lockstep_reward_ab_A",
        "lockstep_reward_ab_B",
    )
    assert plan.cohorts[-2].training_seed == plan.cohorts[-1].training_seed
    assert plan.cohorts[-2].reward_mode == RewardMode.SHAPED.value
    assert plan.cohorts[-1].reward_mode == RewardMode.SIMPLIFIED.value


def test_missing_required_baseline_modes_reports_contract_gaps() -> None:
    missing = missing_required_baseline_modes(
        (
            "off_switch",
            "static_initial",
            "fixed_schedule",
            "lockstep_reward_ab",
        )
    )

    assert missing == ("static_final",)


def _source_manifest_record() -> dict[str, object]:
    payload = TopologyManifestPayload(
        manifest_role="source_final",
        proof_baseline_pair_id="blueprint-health-proof",
        topology_manifest_version=1,
        topology_manifest_hash="source-final-hash",
        topology_manifest_json='{"slots":[{"slot_id":"r0c0","stage":"FOSSILIZED"}]}',
        task="cifar_baseline",
        host_topology="cnn",
        slot_config_hash="slot-config-hash",
        slot_count=1,
        fossilized_seed_count=1,
        topology_delta_count=1,
    )
    return {
        "event_id": "source-final-event",
        "run_dir": "telemetry_source",
        "group_id": STATIC_FINAL_SOURCE_COHORT_ID,
        "episode_idx": 0,
        "payload": asdict(payload),
    }


def test_static_final_supported_and_fixed_schedule_has_provenance() -> None:
    """static_final is supported only through source-manifest replay orchestration."""

    plan = build_blueprint_health_proof_plan()
    support = {cohort.cohort_id: cohort.current_runner_supported for cohort in plan.cohorts}
    schedule_ids = {
        cohort.cohort_id: cohort.proof_baseline_schedule_id for cohort in plan.cohorts
    }
    schedule_hashes = {
        cohort.cohort_id: cohort.proof_baseline_schedule_hash for cohort in plan.cohorts
    }
    schedule_versions = {
        cohort.cohort_id: cohort.proof_baseline_schedule_version
        for cohort in plan.cohorts
    }
    schedule_counts = {
        cohort.cohort_id: cohort.proof_baseline_schedule_action_count
        for cohort in plan.cohorts
    }

    assert support["static_final"] is True
    assert support["fixed_schedule"] is True
    assert schedule_ids["fixed_schedule"] == FIXED_SCHEDULE_GERMINATE_R0C0_V1
    assert schedule_hashes["fixed_schedule"] == FIXED_SCHEDULE_GERMINATE_R0C0_HASH
    assert schedule_versions["fixed_schedule"] == FIXED_SCHEDULE_GERMINATE_R0C0_VERSION
    assert (
        schedule_counts["fixed_schedule"]
        == FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
    )
    # The genuine controls remain supported.
    assert support["off_switch"] is True
    assert support["static_initial"] is True
    assert support["lockstep_reward_ab_A"] is True
    assert support["lockstep_reward_ab_B"] is True


def test_fixed_schedule_hash_is_pinned_to_stable_contract() -> None:
    assert (
        FIXED_SCHEDULE_GERMINATE_R0C0_EXPECTED_HASH
        == "e816388d3566fb7fd2a213cd4e44610aecb791e3e3afe498048920f49a03e797"
    )
    assert (
        FIXED_SCHEDULE_GERMINATE_R0C0_HASH
        == FIXED_SCHEDULE_GERMINATE_R0C0_EXPECTED_HASH
    )


def test_proof_baseline_runner_refuses_unsupported_cohorts() -> None:
    """The plan runner fails loudly when an unsupported cohort is present."""
    plan = build_blueprint_health_proof_plan()
    unsupported_cohorts = tuple(
        replace(cohort, current_runner_supported=False)
        if cohort.mode == ProofBaselineMode.STATIC_FINAL
        else cohort
        for cohort in plan.cohorts
    )
    unsupported_plan = replace(plan, cohorts=unsupported_cohorts)

    def fake_train_cohort(**kwargs: object) -> tuple[object, list[dict[str, object]]]:
        raise AssertionError("unsupported cohorts must not reach training")

    with pytest.raises(RuntimeError, match="not supported by this runner"):
        train_blueprint_health_proof_baselines(
            plan=unsupported_plan,
            train_kwargs={"n_episodes": 1, "device": "cpu"},
            train_cohort=fake_train_cohort,
        )


def test_proof_baseline_runner_invokes_supported_runtime_control_cohorts() -> None:
    plan = build_blueprint_health_proof_plan()
    supported = tuple(
        c for c in plan.cohorts if c.mode != ProofBaselineMode.STATIC_FINAL
    )
    supported_plan = replace(plan, cohorts=supported)
    calls: list[dict[str, object]] = []

    def fake_train_cohort(**kwargs: object) -> tuple[object, list[dict[str, object]]]:
        calls.append(kwargs)
        return object(), [{"avg_accuracy": 0.0}]

    results = train_blueprint_health_proof_baselines(
        plan=supported_plan,
        train_kwargs={"n_episodes": 1, "n_envs": 1, "max_epochs": 2, "device": "cpu"},
        train_cohort=fake_train_cohort,
    )

    assert tuple(results) == tuple(cohort.cohort_id for cohort in supported_plan.cohorts)
    assert tuple(call["proof_baseline_mode"] for call in calls) == tuple(
        cohort.mode.value for cohort in supported_plan.cohorts
    )
    assert tuple(call["proof_baseline_lifecycle_policy"] for call in calls) == tuple(
        cohort.lifecycle_policy for cohort in supported_plan.cohorts
    )
    assert tuple(call["proof_baseline_schedule_id"] for call in calls) == tuple(
        cohort.proof_baseline_schedule_id for cohort in supported_plan.cohorts
    )
    assert tuple(call["proof_baseline_schedule_hash"] for call in calls) == tuple(
        cohort.proof_baseline_schedule_hash for cohort in supported_plan.cohorts
    )
    assert tuple(call["proof_baseline_schedule_version"] for call in calls) == tuple(
        cohort.proof_baseline_schedule_version for cohort in supported_plan.cohorts
    )
    assert tuple(
        call["proof_baseline_schedule_action_count"] for call in calls
    ) == tuple(
        cohort.proof_baseline_schedule_action_count
        for cohort in supported_plan.cohorts
    )
    assert calls[-2]["seed"] == calls[-1]["seed"]
    assert calls[-2]["proof_baseline_pair_id"] == calls[-1]["proof_baseline_pair_id"]


def test_proof_baseline_runner_wires_static_final_source_manifest() -> None:
    plan = build_blueprint_health_proof_plan()
    static_final_plan = replace(
        plan,
        cohorts=tuple(
            cohort
            for cohort in plan.cohorts
            if cohort.mode == ProofBaselineMode.STATIC_FINAL
        ),
    )
    calls: list[dict[str, object]] = []

    def fake_train_cohort(**kwargs: object) -> tuple[object, list[dict[str, object]]]:
        calls.append(kwargs)
        if kwargs["group_id"] == STATIC_FINAL_SOURCE_COHORT_ID:
            return object(), [{"topology_manifests": [_source_manifest_record()]}]
        return object(), [{"topology_manifests": []}]

    results = train_blueprint_health_proof_baselines(
        plan=static_final_plan,
        train_kwargs={"n_episodes": 8, "n_envs": 4, "max_epochs": 2, "device": "cpu"},
        train_cohort=fake_train_cohort,
    )

    assert tuple(results) == (STATIC_FINAL_SOURCE_COHORT_ID, "static_final")
    assert len(calls) == 2
    source_call, static_final_call = calls
    assert source_call["group_id"] == STATIC_FINAL_SOURCE_COHORT_ID
    assert source_call["proof_baseline_mode"] == STATIC_FINAL_SOURCE_MODE
    assert (
        source_call["proof_baseline_lifecycle_policy"]
        == STATIC_FINAL_SOURCE_LIFECYCLE_POLICY
    )
    assert source_call["n_envs"] == 1
    assert source_call["n_episodes"] == 1
    assert source_call["max_epochs"] == STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS
    assert source_call["proof_baseline_schedule_id"] == STATIC_FINAL_SOURCE_TOPOLOGY_V1
    assert (
        source_call["proof_baseline_schedule_hash"]
        == STATIC_FINAL_SOURCE_TOPOLOGY_HASH
    )
    assert (
        source_call["proof_baseline_schedule_version"]
        == STATIC_FINAL_SOURCE_TOPOLOGY_VERSION
    )
    assert (
        source_call["proof_baseline_schedule_action_count"]
        == STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT
    )
    assert static_final_call["group_id"] == "static_final"
    assert static_final_call["proof_baseline_mode"] == "static_final"
    assert static_final_call["max_epochs"] == 2
    assert (
        static_final_call["proof_baseline_lifecycle_policy"]
        == "freeze_replayed_final_topology"
    )
    source_manifest = static_final_call["static_final_source_manifest"]
    assert isinstance(source_manifest, TopologyManifestPayload)
    assert source_manifest.manifest_role == "source_final"
    assert static_final_call["static_final_source_run_dir"] == "telemetry_source"
    assert (
        static_final_call["static_final_source_group_id"]
        == STATIC_FINAL_SOURCE_COHORT_ID
    )
    assert static_final_call["static_final_source_episode_idx"] == 0
    assert static_final_call["static_final_source_event_id"] == "source-final-event"


def test_proof_baseline_runner_refuses_missing_static_final_source_manifest() -> None:
    plan = build_blueprint_health_proof_plan()
    static_final_plan = replace(
        plan,
        cohorts=tuple(
            cohort
            for cohort in plan.cohorts
            if cohort.mode == ProofBaselineMode.STATIC_FINAL
        ),
    )

    def fake_train_cohort(**kwargs: object) -> tuple[object, list[dict[str, object]]]:
        if kwargs["group_id"] == STATIC_FINAL_SOURCE_COHORT_ID:
            return object(), [{"topology_manifests": []}]
        raise AssertionError("static_final must not run without source evidence")

    with pytest.raises(RuntimeError, match="exactly one source-final"):
        train_blueprint_health_proof_baselines(
            plan=static_final_plan,
            train_kwargs={"n_episodes": 1, "n_envs": 1, "max_epochs": 2, "device": "cpu"},
            train_cohort=fake_train_cohort,
        )


@pytest.mark.parametrize(
    "policy",
    (
        "force_wait_only",
        "freeze_initial_topology",
    ),
)
def test_proof_baseline_action_controls_force_wait_for_frozen_lifecycle_policies(
    policy: str,
) -> None:
    masks_batch = _all_valid_masks()

    apply_proof_baseline_action_controls(
        masks_batch=masks_batch,
        lifecycle_policy=policy,
        schedule_id=None,
        epoch=1,
    )

    _assert_only_index(masks_batch["op"], LifecycleOp.WAIT.value)


def test_proof_baseline_action_controls_force_declared_fixed_schedule_action() -> None:
    masks_batch = _all_valid_masks()

    apply_proof_baseline_action_controls(
        masks_batch=masks_batch,
        lifecycle_policy="apply_declared_lifecycle_schedule",
        schedule_id=FIXED_SCHEDULE_GERMINATE_R0C0_V1,
        schedule_hash=FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
        schedule_version=FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
        schedule_action_count=FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
        epoch=1,
    )

    _assert_only_index(masks_batch["op"], LifecycleOp.GERMINATE.value)
    _assert_only_index(masks_batch["slot"], 0)
    _assert_only_index(
        masks_batch["slot_by_op"][:, LifecycleOp.GERMINATE.value, :],
        0,
    )
    _assert_only_index(masks_batch["blueprint"], BlueprintAction.NORM.value)
    _assert_only_index(masks_batch["style"], GerminationStyle.SIGMOID_ADD.value)
    _assert_only_index(masks_batch["tempo"], TempoAction.STANDARD.value)
    _assert_only_index(masks_batch["alpha_target"], AlphaTargetAction.FULL.value)


def test_proof_baseline_action_controls_force_fixed_schedule_wait_after_declared_ops() -> None:
    masks_batch = _all_valid_masks()

    apply_proof_baseline_action_controls(
        masks_batch=masks_batch,
        lifecycle_policy="apply_declared_lifecycle_schedule",
        schedule_id=FIXED_SCHEDULE_GERMINATE_R0C0_V1,
        schedule_hash=FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
        schedule_version=FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
        schedule_action_count=FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
        epoch=2,
    )

    _assert_only_index(masks_batch["op"], LifecycleOp.WAIT.value)


def test_static_final_source_schedule_respects_lifecycle_gate_timing() -> None:
    assert STATIC_FINAL_SOURCE_GERMINATE_EPOCH == 1
    assert STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH == 2
    assert STATIC_FINAL_SOURCE_TRAINING_DWELL_EPOCHS == DEFAULT_MIN_BLENDING_EPOCHS
    assert STATIC_FINAL_SOURCE_BLEND_RAMP_EPOCHS == TEMPO_TO_EPOCHS[
        TempoAction.STANDARD
    ]
    assert (
        STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH
        == STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH
        + DEFAULT_MIN_BLENDING_EPOCHS
    )
    assert (
        STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH
        == STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH
        + TEMPO_TO_EPOCHS[TempoAction.STANDARD]
    )
    assert STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH == (
        STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH + 1
    )
    assert STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS == (
        STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH
    )


@pytest.mark.parametrize(
    ("epoch", "expected_op"),
    (
        (STATIC_FINAL_SOURCE_GERMINATE_EPOCH, LifecycleOp.GERMINATE),
        (STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH, LifecycleOp.ADVANCE),
        (STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH + 1, LifecycleOp.WAIT),
        (STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH - 1, LifecycleOp.WAIT),
        (STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH, LifecycleOp.ADVANCE),
        (STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH - 1, LifecycleOp.WAIT),
        (STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH, LifecycleOp.ADVANCE),
        (STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH, LifecycleOp.FOSSILIZE),
        (STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH + 1, LifecycleOp.WAIT),
    ),
)
def test_static_final_source_policy_forces_fossilizing_source_topology(
    epoch: int,
    expected_op: LifecycleOp,
) -> None:
    masks_batch = _all_valid_masks()

    apply_proof_baseline_action_controls(
        masks_batch=masks_batch,
        lifecycle_policy=STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
        schedule_id=STATIC_FINAL_SOURCE_TOPOLOGY_V1,
        schedule_hash=STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
        schedule_version=STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
        schedule_action_count=STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT,
        epoch=epoch,
    )

    _assert_only_index(masks_batch["op"], expected_op.value)
    if expected_op != LifecycleOp.WAIT:
        _assert_only_index(masks_batch["slot"], 0)
        _assert_only_index(
            masks_batch["slot_by_op"][:, expected_op.value, :],
            0,
        )


def test_static_final_source_policy_refuses_missing_source_schedule_provenance() -> None:
    masks_batch = _all_valid_masks()

    with pytest.raises(ValueError, match=STATIC_FINAL_SOURCE_LIFECYCLE_POLICY):
        apply_proof_baseline_action_controls(
            masks_batch=masks_batch,
            lifecycle_policy=STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
            schedule_id=None,
            epoch=STATIC_FINAL_SOURCE_GERMINATE_EPOCH,
        )


def test_proof_baseline_action_controls_refuse_fixed_schedule_without_schedule_id() -> None:
    masks_batch = _all_valid_masks()

    with pytest.raises(ValueError, match="proof_baseline_schedule_id"):
        apply_proof_baseline_action_controls(
            masks_batch=masks_batch,
            lifecycle_policy="apply_declared_lifecycle_schedule",
            schedule_id=None,
            epoch=1,
        )


def test_proof_baseline_action_controls_refuse_invalid_scheduled_action_without_partial_masks() -> None:
    masks_batch = _all_valid_masks()
    masks_before = {head: mask.clone() for head, mask in masks_batch.items()}
    masks_batch["blueprint"][:, BlueprintAction.NORM.value] = False

    with pytest.raises(RuntimeError, match="Fixed-schedule action is invalid"):
        apply_proof_baseline_action_controls(
            masks_batch=masks_batch,
            lifecycle_policy="apply_declared_lifecycle_schedule",
            schedule_id=FIXED_SCHEDULE_GERMINATE_R0C0_V1,
            schedule_hash=FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
            schedule_version=FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
            schedule_action_count=FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
            epoch=1,
        )

    expected = dict(masks_before)
    expected["blueprint"][:, BlueprintAction.NORM.value] = False
    assert tuple(masks_batch) == tuple(expected)
    for head, expected_mask in expected.items():
        assert torch.equal(masks_batch[head], expected_mask)


@pytest.mark.parametrize(
    "policy",
    ("freeze_replayed_final_topology",),
)
def test_proof_baseline_action_controls_refuse_fake_wait_only_controls(
    policy: str,
) -> None:
    """static_final must fail loudly, not degrade to WAIT-only.

    Masking the op head to WAIT for this policy would produce an impostor
    control (frozen at INITIAL topology, applying no ops), invalidating the
    morphogenesis proof. The action-control layer refuses rather than silently
    degrade.
    """
    masks_batch = _all_valid_masks()

    with pytest.raises(RuntimeError, match="not supported by this runner"):
        apply_proof_baseline_action_controls(
            masks_batch=masks_batch,
            lifecycle_policy=policy,
            schedule_id=None,
            epoch=1,
        )

    # Masks must be left untouched when the policy is refused.
    assert masks_batch["op"].all()


def test_proof_baseline_action_controls_freeze_after_validated_static_final_replay() -> None:
    masks_batch = _all_valid_masks()

    apply_proof_baseline_action_controls(
        masks_batch=masks_batch,
        lifecycle_policy="freeze_replayed_final_topology",
        schedule_id=None,
        epoch=1,
        static_final_replay_validated=True,
    )

    _assert_only_index(masks_batch["op"], LifecycleOp.WAIT.value)
