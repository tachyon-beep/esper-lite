"""Committed-Shapley top-up end-to-end (WI-11c/d + WI-12).

Drives the production static-final-source fixed schedule (germinate r0c0 →
advance ×3 → FOSSILIZE) through the real trainer on CPU with mock dataloaders,
and observes the delivery seam via spies:

- scale > 0: the terminal fused pass evaluates the 2^k coalition family, the
  fossilize step record maps the credit to EXACTLY the FOSSILIZE transition in
  the buffer (drl plan-review F-B: the highest-risk detail), and the k=1
  structural identity holds (phi == c_paid => top_up == 0, nothing written).
  (use_telemetry is True here only because the static-final-source proof
  machinery requires a telemetry run dir for its evidence manifests; the
  coalition eval itself is gated ONLY on shapley_synergy_scale — the WI-2
  condition never reads use_telemetry, per the esper-lite-4fe98055f7 posture.)
- scale == 0 (F5 no-op): neither the Shapley computation nor the delivery
  function is ever invoked — the entire apparatus is skipped.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from esper.leyline import LifecycleOp
from esper.leyline.proof_baselines import (
    STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
    STATIC_FINAL_SOURCE_MODE,
    STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT,
    STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
    STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS,
    STATIC_FINAL_SOURCE_TOPOLOGY_V1,
    STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
)
from esper.simic.training.config import TrainingConfig
from esper.simic.training.vectorized import train_ppo_vectorized

MAX_EPOCHS = STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS + 2


def _gate_two_slot_schedule() -> dict[int, Any]:
    """Two SERIAL all-GATE lifecycles (slot_idx 0 then 1), STATIC_FINAL spacing.

    Serial because the D3 germination rule masks GERMINATE while any seed is in
    GERMINATED/TRAINING; FOSSILIZED does not block. Spacing mirrors the declared
    schedule: 10-epoch TRAINING dwell (permissive G2), 5-epoch blend ramp,
    FOSSILIZE one epoch after HOLDING entry.
    """
    from esper.leyline.factored_actions import (
        AlphaCurveAction,
        AlphaSpeedAction,
        AlphaTargetAction,
        BlueprintAction,
        FactoredAction,
        GerminationStyle,
        TempoAction,
    )

    def _act(slot: int, op: LifecycleOp, blueprint: BlueprintAction) -> FactoredAction:
        return FactoredAction(
            slot_idx=slot,
            blueprint=blueprint,
            style=GerminationStyle.GATED_GATE,
            tempo=TempoAction.STANDARD,
            alpha_target=AlphaTargetAction.FULL,
            alpha_speed=AlphaSpeedAction.INSTANT,
            alpha_curve=AlphaCurveAction.LINEAR,
            op=op,
        )

    def _lifecycle(slot: int, start: int) -> dict[int, FactoredAction]:
        return {
            start: _act(slot, LifecycleOp.GERMINATE, BlueprintAction.NORM),
            start + 1: _act(slot, LifecycleOp.ADVANCE, BlueprintAction.NOOP),
            start + 11: _act(slot, LifecycleOp.ADVANCE, BlueprintAction.NOOP),
            start + 16: _act(slot, LifecycleOp.ADVANCE, BlueprintAction.NOOP),
            start + 17: _act(slot, LifecycleOp.FOSSILIZE, BlueprintAction.NOOP),
        }

    return {**_lifecycle(0, 1), **_lifecycle(1, 19)}


GATE_TWO_SLOT_MAX_EPOCHS = 38


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    shapley_on: bool,
    force_top_up: float | None = None,
    gate_style: bool = False,
    custom_schedule: dict[int, Any] | None = None,
    max_epochs: int = MAX_EPOCHS,
    slots: list[str] | None = None,
    null_gate_schedule_at_terminal: bool = False,
    observed_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import esper.runtime as runtime
    import esper.simic.training.ppo_coordinator as ppo_coordinator_mod
    import esper.simic.training.vectorized_trainer as vt_mod

    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str) -> Any:
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)

    observed: dict[str, Any] = (
        observed_out if observed_out is not None else {}
    )
    observed.update(
        {
            "apply_calls": [],
            "compute_calls": 0,
            "blend_create_calls": 0,
        }
    )

    if null_gate_schedule_at_terminal:
        # pytorch-review H1 window: after the coalition family is built (the
        # GATE fossil is admitted) but before the fused batch loop, null the
        # fossil's schedule — the exact state where the pre-fix P4-FIX block
        # would have materialized a fresh untrained gate silently.
        real_build = vt_mod.VectorizedPPOTrainer._build_fused_val_configs

        def nulling_build(self: Any, *, env_states: Any, slots: Any, epoch: int, envs_this_batch: Any) -> Any:
            result = real_build(
                self,
                env_states=env_states,
                slots=slots,
                epoch=epoch,
                envs_this_batch=envs_this_batch,
            )
            if epoch == self.max_epochs:
                for env_state in env_states:
                    for sid in slots:
                        if not env_state.model.has_active_seed_in_slot(sid):
                            continue
                        slot = env_state.model.seed_slots[sid]
                        if slot.state is not None and slot.state.stage.name == "FOSSILIZED":
                            slot.alpha_schedule = None
            return result

        monkeypatch.setattr(
            vt_mod.VectorizedPPOTrainer,
            "_build_fused_val_configs",
            nulling_build,
        )

    if gate_style:
        # Rewrite the declared schedule's GERMINATE to the GATED_GATE style at
        # the mask-forcing seam (the declared schedules are hash-pinned, so the
        # action is rewritten in flight, not in the registry). This reproduces
        # the ON-arm crash class: an ordinary policy germination style whose
        # fossilized seed reaches the terminal coalition family
        # (esper-lite-fbeead4efc).
        from esper.leyline.factored_actions import GerminationStyle

        real_force = vt_mod._force_scheduled_action_masks

        def gate_forcing(*, masks_batch: Any, action: Any, epoch: int) -> Any:
            if action.op == LifecycleOp.GERMINATE:
                action = replace(action, style=GerminationStyle.GATED_GATE)
            return real_force(
                masks_batch=masks_batch, action=action, epoch=epoch
            )

        monkeypatch.setattr(
            vt_mod, "_force_scheduled_action_masks", gate_forcing
        )

    # Object-identity capture (owner ratification §1): at the terminal
    # config build, record each fossilized GATE slot's schedule object id and
    # a snapshot of its parameters so tests can assert the fused pass neither
    # replaced nor mutated the trained gate.
    observed["gate_capture"] = []
    real_build_capture = vt_mod.VectorizedPPOTrainer._build_fused_val_configs

    def capturing_build(self: Any, *, env_states: Any, slots: Any, epoch: int, envs_this_batch: Any) -> Any:
        result = real_build_capture(
            self,
            env_states=env_states,
            slots=slots,
            epoch=epoch,
            envs_this_batch=envs_this_batch,
        )
        if epoch == self.max_epochs:
            from esper.leyline import AlphaAlgorithm, SeedStage

            for env_state in env_states:
                for sid in slots:
                    if not env_state.model.has_active_seed_in_slot(sid):
                        continue
                    slot = env_state.model.seed_slots[sid]
                    if (
                        slot.state is not None
                        and slot.state.stage == SeedStage.FOSSILIZED
                        and slot.state.alpha_algorithm == AlphaAlgorithm.GATE
                        and slot.alpha_schedule is not None
                    ):
                        observed["gate_capture"].append(
                            (
                                slot,
                                id(slot.alpha_schedule),
                                [
                                    p.detach().clone()
                                    for p in slot.alpha_schedule.parameters()
                                ],
                            )
                        )
        return result

    monkeypatch.setattr(
        vt_mod.VectorizedPPOTrainer, "_build_fused_val_configs", capturing_build
    )

    if custom_schedule is not None:
        # Substitute an entire in-flight action sequence at the same seam:
        # scheduled epochs take the custom action, all other epochs WAIT.
        from esper.leyline.proof_baselines import WAIT_FIXED_SCHEDULE_ACTION

        real_force_full = vt_mod._force_scheduled_action_masks

        def schedule_override(*, masks_batch: Any, action: Any, epoch: int) -> Any:
            action = custom_schedule.get(epoch, WAIT_FIXED_SCHEDULE_ACTION)
            return real_force_full(
                masks_batch=masks_batch, action=action, epoch=epoch
            )

        monkeypatch.setattr(
            vt_mod, "_force_scheduled_action_masks", schedule_override
        )

    # Probe for the fused-pass materialization hazard (plan review, pytorch
    # F3): BlendCatalog.create must run exactly once for a gated lifecycle
    # (at BLENDING entry) and never inside the terminal fused pass.
    from esper.kasmina.blending import BlendCatalog

    real_create = BlendCatalog.create.__func__

    def counting_create(cls: Any, algorithm_id: str, **kwargs: Any) -> Any:
        observed["blend_create_calls"] += 1
        return real_create(cls, algorithm_id, **kwargs)

    monkeypatch.setattr(
        BlendCatalog, "create", classmethod(counting_create)
    )

    real_apply = ppo_coordinator_mod.apply_committed_shapley_credits

    def spying_apply(**kwargs: Any) -> Any:
        buffer = kwargs["buffer"]
        env_credits = kwargs["env_credits"]
        rewards_before = buffer.rewards.clone()
        payloads = real_apply(**kwargs)
        observed["apply_calls"].append(
            {
                "env_credits": env_credits,
                "payloads": payloads,
                "rewards_before": rewards_before,
                "rewards_after": buffer.rewards.clone(),
                "effective_op_actions": buffer.effective_op_actions.clone(),
                "step_counts": list(buffer.step_counts),
            }
        )
        return payloads

    monkeypatch.setattr(
        ppo_coordinator_mod, "apply_committed_shapley_credits", spying_apply
    )

    real_compute = vt_mod.compute_committed_shapley_topup

    def spying_compute(*args: Any, **kwargs: Any) -> Any:
        observed["compute_calls"] += 1
        result = real_compute(*args, **kwargs)
        if force_top_up is None:
            return result
        # Reviewer re-pass MEDIUM finding: the k=1 schedule pays zero by
        # construction, so a nonzero credit never crosses the live seam.
        # Force a paying result HERE (the math module is exhaustively
        # unit-tested); everything downstream — records handoff, run_update
        # param, config-value reads, normalizer std, the buffer write —
        # stays fully live.
        from esper.simic.rewards.committed_shapley import (
            CommittedShapleyResult,
            SlotTopUp,
        )

        per_slot = {
            slot: SlotTopUp(
                phi=entry.phi,
                c_paid=entry.c_paid,
                gap=force_top_up,
                raw=force_top_up,
                top_up=force_top_up,
            )
            for slot, entry in result.per_slot.items()
        }
        return CommittedShapleyResult(
            per_slot=per_slot,
            g=force_top_up * len(per_slot),
            sum_raw=force_top_up * len(per_slot),
            clamp_binding=False,
            k=result.k,
        )

    monkeypatch.setattr(vt_mod, "compute_committed_shapley_topup", spying_compute)

    config = TrainingConfig.for_cifar_minimal()
    config.n_envs = 1
    config.n_episodes = 1
    config.max_epochs = max_epochs
    config.chunk_length = max_epochs
    if slots is not None:
        config.slots = slots
    config.seed = 42
    config.use_telemetry = True
    config.gradient_telemetry_stride = 1
    if shapley_on:
        config.shapley_synergy_scale = 0.5
        config.shapley_synergy_cap = 3.0
        config.shapley_synergy_normalized_cap = 5.0

    train_ppo_vectorized(
        **config.to_train_kwargs(),
        device="cpu",
        devices=["cpu"],
        num_workers=0,
        quiet_analytics=True,
        telemetry_dir=str(tmp_path),
        proof_baseline_mode=STATIC_FINAL_SOURCE_MODE,
        proof_baseline_pair_id="committed-shapley-integration",
        proof_baseline_lifecycle_policy=STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
        proof_baseline_schedule_id=STATIC_FINAL_SOURCE_TOPOLOGY_V1,
        proof_baseline_schedule_hash=STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
        proof_baseline_schedule_version=STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
        proof_baseline_schedule_action_count=(
            STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT
        ),
    )
    return observed


@pytest.mark.integration
def test_scale_on_full_loop_credits_the_fossilize_transition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    observed = _run(monkeypatch, tmp_path, shapley_on=True)

    # The terminal coalition eval ran exactly once (the terminal epoch).
    assert observed["compute_calls"] == 1
    assert len(observed["apply_calls"]) == 1
    call = observed["apply_calls"][0]
    (credits,) = call["env_credits"]
    assert credits.env_idx == 0
    assert credits.result.k == 1
    # The schedule germinates in slot_idx 0 = the first ENABLED slot; derive
    # the slot id from the credit records rather than hardcoding it.
    (slot_id,) = credits.t_f_by_slot

    # drl F-B (the highest-risk detail): the recorded t_f IS the FOSSILIZE
    # transition in the buffer — not merely "a valid step".
    t_f = credits.t_f_by_slot[slot_id]
    assert 0 <= t_f < int(call["step_counts"][0])
    assert int(call["effective_op_actions"][0, t_f]) == int(LifecycleOp.FOSSILIZE)

    # k=1 structural identity: phi == c_paid exactly -> zero top-up, and the
    # delivery writes nothing (the term only fires on k >= 2 coalitions).
    slot_result = credits.result.per_slot[slot_id]
    assert slot_result.phi == pytest.approx(slot_result.c_paid)
    assert slot_result.top_up == 0.0
    assert torch.equal(call["rewards_before"], call["rewards_after"])

    # The payload books are complete even for a zero credit.
    (payload,) = call["payloads"]
    assert payload.slot_ids == (slot_id,)
    assert payload.t_f == (t_f,)
    assert payload.credit_buf == (0.0,)


@pytest.mark.integration
def test_scale_on_forced_nonzero_credit_lands_in_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Reviewer re-pass (MEDIUM): a NONZERO credit through the fully live
    seam — trainer fold -> run_update handoff -> config-value reads ->
    normalizer std -> buffer write at t_f. Only the credit magnitude is
    forced (the math module is exhaustively unit-tested); the delivery
    arithmetic asserted here uses the run's real running std."""
    top_up = 2.0
    observed = _run(monkeypatch, tmp_path, shapley_on=True, force_top_up=top_up)

    assert len(observed["apply_calls"]) == 1
    call = observed["apply_calls"][0]
    (credits,) = call["env_credits"]
    (slot_id,) = credits.t_f_by_slot
    t_f = credits.t_f_by_slot[slot_id]
    (payload,) = call["payloads"]

    # The write landed at exactly the FOSSILIZE transition, nowhere else.
    delta = call["rewards_after"] - call["rewards_before"]
    expected = min(5.0, top_up / payload.std_used)  # normalized_cap=5.0
    assert payload.dropped_no_std is False
    assert payload.std_used > 0.0
    assert delta[0, t_f].item() == pytest.approx(expected)
    assert payload.credit_buf == (pytest.approx(expected),)
    mask = torch.ones_like(delta, dtype=torch.bool)
    mask[0, t_f] = False
    assert torch.all(delta[mask] == 0.0)


@pytest.mark.integration
def test_scale_on_gate_fossilized_slot_completes_and_credits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """ON-arm crash regression (esper-lite-fbeead4efc): a GATED_GATE-germinated
    seed that fossilizes must be admitted to the terminal coalition family.

    The 6dd80716 A/B ON wave died 5/5 on a blanket rejection of GATE-algorithm
    fossilized slots, but GATE seeds keep their trained alpha_schedule for life
    (the GATE forward requires it at every stage), so the coalition eval is
    well-defined: alpha amplitude multiplies the gated contribution, exactly
    the force_alpha contract. The blend_create_calls probe pins the hazard the
    old guard protected against — the fused pass must never materialize a
    fresh (untrained) schedule for a fossilized slot."""
    observed = _run(monkeypatch, tmp_path, shapley_on=True, gate_style=True)

    assert observed["compute_calls"] == 1
    (call,) = observed["apply_calls"]
    (credits,) = call["env_credits"]
    (slot_id,) = credits.t_f_by_slot

    # k=1 structural identity holds through the gated blend path too.
    slot_result = credits.result.per_slot[slot_id]
    assert credits.result.k == 1
    assert slot_result.phi == pytest.approx(slot_result.c_paid)
    assert slot_result.top_up == 0.0
    assert torch.equal(call["rewards_before"], call["rewards_after"])

    # Exactly one gated schedule ever created (BLENDING entry) — the fused
    # pass reused the trained gate and materialized nothing.
    assert observed["blend_create_calls"] == 1

    # Gate-identity telemetry (addendum §5): the payload carries the fossil's
    # real alpha algorithm and the tau applied — the scoring-side
    # stratification key, riding the scale>0-only TOPUP path.
    (payload,) = call["payloads"]
    assert payload.alpha_algorithms == ("GATE",)
    assert payload.tau_used == 0.0

    # Owner-ratified identity pin: the fossil's schedule is the SAME OBJECT
    # after the terminal pass, with bit-identical parameters — the fused pass
    # neither replaced nor mutated the trained gate.
    (capture,) = observed["gate_capture"]
    slot_ref, sched_id, params_before = capture
    assert id(slot_ref.alpha_schedule) == sched_id
    params_after = list(slot_ref.alpha_schedule.parameters())
    assert len(params_after) == len(params_before)
    for before, after in zip(params_before, params_after):
        assert torch.equal(before, after)


@pytest.mark.integration
def test_fused_pass_fails_loud_when_fossilized_gate_schedule_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """pytorch-review H1 differential pin for the stage-scope edit.

    A fossilized GATE slot whose schedule has (impossibly) gone missing must
    FAIL LOUD in the GATE forward — the P4-FIX materialization is scoped away
    from FOSSILIZED, so the fused pass must not fabricate a fresh untrained
    gate. Without the `stage != FOSSILIZED` clause this run would complete
    silently with a junk gate (blend_create_calls would tick to 2 and no error
    would surface) — both assertions below would fail, making edit #2 visible
    to CI for the first time."""
    observed_out: dict[str, Any] = {}
    with pytest.raises(
        RuntimeError, match="alpha_schedule is required when alpha_algorithm=GATE"
    ):
        _run(
            monkeypatch,
            tmp_path,
            shapley_on=True,
            gate_style=True,
            null_gate_schedule_at_terminal=True,
            observed_out=observed_out,
        )
    # Exactly the one legitimate creation (BLENDING entry) — nothing
    # materialized after the schedule was nulled.
    assert observed_out["blend_create_calls"] == 1


@pytest.mark.integration
def test_scale_on_k2_all_gate_coalition_pays_through_live_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """drl-review coverage clause (addendum §6): a k=2 all-GATE coalition
    through the live trainer path — 2^2 coalition family evaluated with both
    gated fossils, credit delivered at BOTH fossilize transitions.

    The credit magnitude is forced (the existing harness pattern — the G-clamp
    arithmetic under real values is exhaustively unit-tested in the math
    module); everything else is live: the fused pass builds and evaluates the
    4-config family over two gated fossils, real compute runs on the real
    coalition accs, and the delivery writes through the run's real normalizer.
    """
    top_up = 1.5
    observed = _run(
        monkeypatch,
        tmp_path,
        shapley_on=True,
        force_top_up=top_up,
        custom_schedule=_gate_two_slot_schedule(),
        max_epochs=GATE_TWO_SLOT_MAX_EPOCHS,
        slots=["r0c0", "r0c1"],
    )

    assert observed["compute_calls"] == 1
    (call,) = observed["apply_calls"]
    (credits,) = call["env_credits"]

    # Both gated fossils are coalition members; the full 2^2 family was
    # evaluated (no silent exclusion — fix (a) must not degrade to fix (c)).
    assert credits.result.k == 2
    assert set(credits.t_f_by_slot) == {"r0c0", "r0c1"}
    assert len(credits.coalition_accs) == 4

    # Each slot's forced credit lands at exactly its own FOSSILIZE transition.
    delta = call["rewards_after"] - call["rewards_before"]
    (payload,) = call["payloads"]
    assert payload.dropped_no_std is False
    nonzero = delta[0].nonzero().flatten().tolist()
    assert sorted(nonzero) == sorted(credits.t_f_by_slot.values())
    for slot_id, t_f in credits.t_f_by_slot.items():
        assert int(call["effective_op_actions"][0, t_f]) == int(
            LifecycleOp.FOSSILIZE
        )
        assert delta[0, t_f].item() != 0.0

    # Exactly two gated schedules ever created (one per lifecycle, at
    # BLENDING entry) — the fused pass materialized nothing for the fossils.
    assert observed["blend_create_calls"] == 2

    # Gate-identity telemetry: both fossils stratify as GATE in the payload.
    assert payload.alpha_algorithms == ("GATE", "GATE")

    # Identity pin for both fossils: same schedule objects, bit-identical
    # parameters, through the full 2^2 coalition evaluation.
    assert len(observed["gate_capture"]) == 2
    for slot_ref, sched_id, params_before in observed["gate_capture"]:
        assert id(slot_ref.alpha_schedule) == sched_id
        for before, after in zip(
            params_before, slot_ref.alpha_schedule.parameters()
        ):
            assert torch.equal(before, after)


@pytest.mark.integration
def test_scale_zero_apparatus_never_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """F5 no-op: at scale=0 the same fossilizing run must never touch the
    coalition eval or the delivery seam."""
    observed = _run(monkeypatch, tmp_path, shapley_on=False)
    assert observed["compute_calls"] == 0
    assert observed["apply_calls"] == []
