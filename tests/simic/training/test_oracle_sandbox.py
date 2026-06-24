from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import torch

from esper.karn.mcp.views import create_views, scan_ingestion_integrity
from esper.leyline import (
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    SlotConfig,
    TempoAction,
)
from esper.simic.training.oracle_sandbox import (
    ORACLE_CI_FIXTURE_ID,
    ORACLE_CI_SLOT_ID,
    ORACLE_PROOF_PROFILE,
    OracleSchedule,
    OracleScheduleStep,
    build_default_oracle_schedule,
    build_oracle_ci_fixture,
    run_oracle_schedule,
)


def _wait_action(*, slot_idx: int = 0) -> FactoredAction:
    return FactoredAction(
        slot_idx=slot_idx,
        blueprint=BlueprintAction.NOOP,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.INSTANT,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=LifecycleOp.WAIT,
    )


def test_default_oracle_schedule_is_stable_and_covers_lifecycle_contract() -> None:
    schedule = build_default_oracle_schedule()
    rebuilt = build_default_oracle_schedule()

    assert schedule.name == "oracle-ci-lifecycle-v1"
    assert schedule.proof_profile == "oracle-sandbox"
    assert schedule.fixture_id == ORACLE_CI_FIXTURE_ID
    assert schedule.identity == rebuilt.identity
    assert schedule.identity.startswith("oracle-sandbox:oracle-ci-lifecycle-v1:")

    ops = tuple(step.action.op for step in schedule.steps)
    assert ops == (
        LifecycleOp.GERMINATE,
        LifecycleOp.ADVANCE,
        LifecycleOp.ADVANCE,
        LifecycleOp.SET_ALPHA_TARGET,
        LifecycleOp.ADVANCE,
        LifecycleOp.FOSSILIZE,
        LifecycleOp.PRUNE,
    )
    assert schedule.steps[-1].expect_success is False

    manifest = schedule.to_manifest()
    assert manifest["name"] == schedule.name
    assert manifest["fixture_id"] == ORACLE_CI_FIXTURE_ID
    assert manifest["steps"][0]["op"] == "GERMINATE"
    assert manifest["steps"][-1]["expect_success"] is False


def test_oracle_schedule_validation_fails_loudly_for_malformed_entries() -> None:
    valid_step = OracleScheduleStep(step_id="wait", action=_wait_action())

    with pytest.raises(ValueError, match="at least one step"):
        OracleSchedule(name="empty", steps=())

    with pytest.raises(ValueError, match="duplicate step_id"):
        OracleSchedule(name="duplicate", steps=(valid_step, valid_step))

    with pytest.raises(ValueError, match="step_id must be non-empty"):
        OracleScheduleStep(step_id="", action=_wait_action())

    with pytest.raises(ValueError, match="slot_idx must be non-negative"):
        OracleScheduleStep(step_id="negative-slot", action=_wait_action(slot_idx=-1))


def test_oracle_ci_fixture_is_cpu_first_and_deterministic() -> None:
    fixture = build_oracle_ci_fixture()
    rebuilt = build_oracle_ci_fixture()

    assert fixture.fixture_id == ORACLE_CI_FIXTURE_ID
    assert fixture.slot_config == SlotConfig(slot_ids=(ORACLE_CI_SLOT_ID,))
    assert fixture.enabled_slots == (ORACLE_CI_SLOT_ID,)
    assert fixture.inputs.device.type == "cpu"
    assert fixture.targets.device.type == "cpu"
    assert fixture.inputs.shape == (2, 3, 8, 8)
    assert fixture.targets.equal(torch.tensor([0, 1]))
    assert fixture.model.training is True
    assert torch.equal(fixture.inputs, rebuilt.inputs)
    assert fixture.model.total_seeds() == 0


def test_oracle_schedule_runner_uses_masks_governor_and_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_oracle_ci_fixture()
    schedule = build_default_oracle_schedule()
    handler_calls: list[LifecycleOp] = []

    from esper.simic.training.handlers import get_handler as real_get_handler

    def recording_get_handler(op_idx: int):
        handler = real_get_handler(op_idx)

        def wrapped_handler(*args, **kwargs):
            handler_calls.append(LifecycleOp(op_idx))
            return handler(*args, **kwargs)

        return wrapped_handler

    monkeypatch.setattr(
        "esper.simic.training.oracle_sandbox.get_handler",
        recording_get_handler,
    )

    result = run_oracle_schedule(schedule=schedule, fixture=fixture)

    assert result.success is True
    assert tuple(step.op for step in result.steps) == tuple(
        step.action.op for step in schedule.steps
    )
    assert tuple(step.stage_after for step in result.steps) == (
        "GERMINATED",
        "TRAINING",
        "BLENDING",
        "BLENDING",
        "HOLDING",
        "FOSSILIZED",
        "FOSSILIZED",
    )
    assert handler_calls == [
        LifecycleOp.GERMINATE,
        LifecycleOp.ADVANCE,
        LifecycleOp.ADVANCE,
        LifecycleOp.SET_ALPHA_TARGET,
        LifecycleOp.ADVANCE,
        LifecycleOp.FOSSILIZE,
    ]
    assert all(step.mask_checked for step in result.steps)
    assert all(step.governor_checked for step in result.steps[:-1])
    assert result.steps[-1].blocked_by == "mask"
    assert result.steps[-1].handler_called is False
    assert result.steps[-1].success is False


def test_oracle_schedule_runner_blocks_governor_veto_before_handler() -> None:
    fixture = build_oracle_ci_fixture()
    germinate_step = build_default_oracle_schedule().steps[0]
    schedule = OracleSchedule(
        name="oracle-governor-veto",
        steps=(germinate_step,),
    )

    result = run_oracle_schedule(
        schedule=schedule,
        fixture=fixture,
        val_loss=float("nan"),
    )

    assert result.success is False
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.mask_checked is True
    assert step.governor_checked is True
    assert step.governor_approved is False
    assert step.blocked_by == "governor"
    assert step.handler_called is False
    assert fixture.model.total_seeds() == 0


def test_oracle_schedule_writes_karn_ingestable_proof_telemetry(
    tmp_path: Path,
) -> None:
    fixture = build_oracle_ci_fixture()
    schedule = build_default_oracle_schedule()

    result = run_oracle_schedule(
        schedule=schedule,
        fixture=fixture,
        telemetry_dir=tmp_path,
    )

    assert result.success is True
    assert result.telemetry_dir is not None
    assert Path(result.telemetry_dir).is_dir()
    assert scan_ingestion_integrity(str(tmp_path)).is_clean

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    run_row = conn.execute(
        """
        SELECT
            proof_profile,
            proof_baseline_mode,
            proof_baseline_lifecycle_policy,
            proof_baseline_schedule_id,
            proof_baseline_schedule_hash,
            proof_baseline_schedule_action_count,
            policy_device,
            amp_enabled,
            compile_enabled,
            host_params,
            n_envs
        FROM runs
        """
    ).fetchone()
    assert run_row == (
        ORACLE_PROOF_PROFILE,
        ORACLE_PROOF_PROFILE,
        "oracle_schedule",
        schedule.identity,
        schedule.identity[-16:],
        len(schedule.steps),
        "cpu",
        False,
        False,
        fixture.model.total_params - fixture.model.active_seed_params,
        1,
    )

    terminal_rows = conn.execute(
        """
        SELECT phase, operation, slot_id, governor_approved
        FROM morphology_causal_log
        WHERE phase IN ('commit', 'fossilization')
        ORDER BY epoch
        """
    ).fetchall()
    assert terminal_rows == [
        ("commit", "GERMINATE", ORACLE_CI_SLOT_ID, True),
        ("commit", "ADVANCE", ORACLE_CI_SLOT_ID, True),
        ("commit", "ADVANCE", ORACLE_CI_SLOT_ID, True),
        ("commit", "SET_ALPHA_TARGET", ORACLE_CI_SLOT_ID, True),
        ("commit", "ADVANCE", ORACLE_CI_SLOT_ID, True),
        ("fossilization", "FOSSILIZE", ORACLE_CI_SLOT_ID, True),
    ]

    (oracle_policy_evidence_count,) = conn.execute(
        """
        SELECT count(*)
        FROM morphology_causal_log
        WHERE phase = 'audit'
          AND operation = 'ORACLE_POLICY'
          AND message = 'Oracle policy evidence'
        """
    ).fetchone()
    assert oracle_policy_evidence_count == 1

    outcome_row = conn.execute(
        """
        SELECT
            final_accuracy,
            param_ratio,
            num_fossilized,
            num_contributing_fossilized,
            reward_mode,
            outcome_type,
            episode_length,
            germinate_count,
            prune_count,
            fossilize_count
        FROM episode_outcomes
        """
    ).fetchone()
    assert outcome_row == (
        2.0,
        pytest.approx(fixture.model.total_params / run_row[9]),
        1,
        1,
        "oracle",
        "success",
        len(schedule.steps),
        1,
        0,
        1,
    )
