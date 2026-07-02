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


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    shapley_on: bool,
    force_top_up: float | None = None,
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

    observed: dict[str, Any] = {
        "apply_calls": [],
        "compute_calls": 0,
    }

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
    config.max_epochs = MAX_EPOCHS
    config.chunk_length = MAX_EPOCHS
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
def test_scale_zero_apparatus_never_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """F5 no-op: at scale=0 the same fossilizing run must never touch the
    coalition eval or the delivery seam."""
    observed = _run(monkeypatch, tmp_path, shapley_on=False)
    assert observed["compute_calls"] == 0
    assert observed["apply_calls"] == []
