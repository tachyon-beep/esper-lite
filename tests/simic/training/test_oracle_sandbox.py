from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import torch

from esper.karn.mcp.views import create_views, scan_ingestion_integrity
from esper.leyline import (
    AlphaMode,
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    SeedStage,
    SlotConfig,
    TempoAction,
)
import esper.simic.training.oracle_sandbox as oracle_sandbox_module
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


def _action(
    op: LifecycleOp,
    *,
    slot_idx: int = 0,
    blueprint: BlueprintAction = BlueprintAction.NOOP,
    alpha_target: AlphaTargetAction = AlphaTargetAction.FULL,
    alpha_speed: AlphaSpeedAction = AlphaSpeedAction.INSTANT,
    alpha_curve: AlphaCurveAction = AlphaCurveAction.LINEAR,
) -> FactoredAction:
    return FactoredAction(
        slot_idx=slot_idx,
        blueprint=blueprint,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=alpha_target,
        alpha_speed=alpha_speed,
        alpha_curve=alpha_curve,
        op=op,
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
    assert fixture.device.type == "cpu"
    assert fixture.model.training is True
    # Determinism: rebuilding the fixture yields a bit-identical model.
    fixture_params = list(fixture.model.parameters())
    rebuilt_params = list(rebuilt.model.parameters())
    assert len(fixture_params) == len(rebuilt_params)
    assert all(torch.equal(a, b) for a, b in zip(fixture_params, rebuilt_params))
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


def test_oracle_schedule_runner_blocks_out_of_range_slot_by_mask() -> None:
    fixture = build_oracle_ci_fixture()
    schedule = OracleSchedule(
        name="oracle-out-of-range-slot",
        steps=(
            OracleScheduleStep(
                step_id="wait_missing_slot",
                action=_wait_action(slot_idx=fixture.slot_config.num_slots),
            ),
        ),
    )

    result = run_oracle_schedule(schedule=schedule, fixture=fixture)

    assert result.success is False
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.mask_checked is True
    assert step.governor_checked is False
    assert step.blocked_by == "mask"
    assert step.handler_called is False


def test_oracle_schedule_runner_checks_masks_before_settling_evidence() -> None:
    fixture = build_oracle_ci_fixture()
    setup_schedule = OracleSchedule(
        name="oracle-enter-blending",
        steps=(
            OracleScheduleStep(
                step_id="germinate_norm",
                action=_action(LifecycleOp.GERMINATE, blueprint=BlueprintAction.NORM),
            ),
            OracleScheduleStep(
                step_id="advance_to_training",
                action=_action(LifecycleOp.ADVANCE),
            ),
            OracleScheduleStep(
                step_id="advance_to_blending",
                action=_action(LifecycleOp.ADVANCE),
            ),
        ),
    )

    setup_result = run_oracle_schedule(schedule=setup_schedule, fixture=fixture)
    assert setup_result.success is True
    assert setup_result.steps[-1].stage_after == "BLENDING"
    slot = fixture.model.seed_slots[ORACLE_CI_SLOT_ID]
    assert slot.state is not None
    assert slot.state.stage == SeedStage.BLENDING
    slot.state.alpha_controller.alpha_mode = AlphaMode.UP

    retarget_schedule = OracleSchedule(
        name="oracle-retarget-before-hold",
        steps=(
            OracleScheduleStep(
                step_id="set_alpha_before_hold",
                action=_action(
                    LifecycleOp.SET_ALPHA_TARGET,
                    alpha_target=AlphaTargetAction.SEVENTY,
                    alpha_speed=AlphaSpeedAction.SLOW,
                    alpha_curve=AlphaCurveAction.COSINE,
                ),
                expect_success=False,
            ),
        ),
    )

    result = run_oracle_schedule(schedule=retarget_schedule, fixture=fixture)

    assert result.success is True
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.success is False
    assert step.blocked_by == "mask"
    assert step.handler_called is False
    assert step.stage_before == "BLENDING"


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


def test_oracle_episode_outcome_counts_only_contributing_fossils(
    tmp_path: Path,
) -> None:
    fixture = build_oracle_ci_fixture()
    setup_schedule = OracleSchedule(
        name="oracle-seed-for-noncontributing-fossilize",
        steps=(
            OracleScheduleStep(
                step_id="germinate_norm",
                action=_action(LifecycleOp.GERMINATE, blueprint=BlueprintAction.NORM),
            ),
        ),
    )
    setup_result = run_oracle_schedule(schedule=setup_schedule, fixture=fixture)
    assert setup_result.success is True

    slot = fixture.model.seed_slots[ORACLE_CI_SLOT_ID]
    assert slot.state is not None
    assert slot.state.transition(SeedStage.TRAINING)
    assert slot.state.transition(SeedStage.BLENDING)
    assert slot.state.transition(SeedStage.HOLDING)
    slot.set_alpha(1.0)
    slot.state.alpha_controller.alpha_target = 1.0
    slot.state.alpha_controller.alpha_mode = AlphaMode.HOLD
    slot.state.metrics.initial_val_accuracy = 0.0
    slot.state.metrics.current_val_accuracy = 0.5
    slot.state.metrics.counterfactual_contribution = 2.0
    schedule = OracleSchedule(
        name="oracle-noncontributing-fossilize",
        steps=(
            OracleScheduleStep(
                step_id="fossilize_seed",
                action=_action(LifecycleOp.FOSSILIZE),
            ),
        ),
    )

    result = run_oracle_schedule(
        schedule=schedule,
        fixture=fixture,
        telemetry_dir=tmp_path,
    )

    assert result.success is True

    conn = duckdb.connect(":memory:")
    create_views(conn, str(tmp_path))

    outcome_row = conn.execute(
        """
        SELECT
            num_fossilized,
            num_contributing_fossilized
        FROM episode_outcomes
        """
    ).fetchone()
    assert outcome_row == (1, 0)


def test_oracle_sandbox_exposes_single_schedule_runner_dispatch() -> None:
    assert "apply_oracle_factored_action" not in oracle_sandbox_module.__all__
    assert "apply_oracle_factored_action" not in oracle_sandbox_module.__dict__


def test_emit_oracle_episode_outcome_survives_zero_host_params() -> None:
    # A degenerate fixture where total_params == active_seed_params yields
    # host_params == 0; the episode-outcome emit must clamp the divisor (matching
    # action_execution.py's canonical max(1, host_params)) rather than raise
    # ZeroDivisionError mid-telemetry.
    fixture = build_oracle_ci_fixture()
    schedule = build_default_oracle_schedule()
    captured: list = []

    class _CaptureOutput:
        def emit(self, event: object) -> None:
            captured.append(event)

    oracle_sandbox_module._emit_oracle_episode_outcome(
        telemetry_output=_CaptureOutput(),
        schedule=schedule,
        fixture=fixture,
        step_results=(),
        host_params=0,
        val_accuracy=1.0,
        run_success=True,
        num_fossilized=0,
        num_contributing_fossilized=0,
    )

    (event,) = captured
    assert event.data.param_ratio == pytest.approx(fixture.model.total_params)


def test_oracle_host_has_no_vestigial_slots_mechanism() -> None:
    # OracleSandboxHost never applies seeds itself -- the MorphogeneticModel applies them
    # via seed_slots between forward_to_segment/forward_from_segment (matching CNNHost, whose
    # forward is "no slot application"). The host carried a never-populated _slots dict, making
    # the `_slots[...] is not None` guards dead code.
    host = oracle_sandbox_module.OracleSandboxHost()
    assert not hasattr(host, "_slots")
    # Backbone still runs end to end (conv -> pool -> fc), num_classes default = 2.
    out = host(torch.randn(2, 3, 8, 8))
    assert out.shape == (2, 2)


def test_oracle_fixture_does_not_carry_unused_eval_tensors() -> None:
    # The sandbox drives proof telemetry from a synthetic val_accuracy (default 2.0); it
    # never runs a forward pass on inputs/targets -- only the device was ever read. Those
    # tensors must not be carried as misleading dead state implying real evaluation.
    fixture = build_oracle_ci_fixture()
    assert not hasattr(fixture, "inputs")
    assert not hasattr(fixture, "targets")
    # The only real use of the old tensors (their device) is preserved.
    assert fixture.device.type == "cpu"


def test_emit_oracle_causal_event_has_no_unused_step_result_param() -> None:
    # _emit_oracle_causal_event never reads step_result; a passed-but-unused parameter
    # is dead surface that call sites must not be forced to thread through.
    import inspect

    params = inspect.signature(
        oracle_sandbox_module._emit_oracle_causal_event
    ).parameters
    assert "step_result" not in params


def test_oracle_reuses_canonical_resolve_target_slot() -> None:
    # The oracle sandbox must reuse vectorized's canonical slot resolver, not
    # duplicate it (the duplicate carried the same enabled_slots[0] crash).
    from esper.simic.training.vectorized import _resolve_target_slot

    assert oracle_sandbox_module._resolve_target_slot is _resolve_target_slot
    assert not hasattr(oracle_sandbox_module, "_resolve_oracle_target_slot")
