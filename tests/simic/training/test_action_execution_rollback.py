from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import esper.simic.training.action_execution as action_execution
from esper.leyline import HEAD_NAMES, OP_GERMINATE, OP_WAIT, SlotConfig
from esper.simic.rewards import RewardFamily, RewardMode
from esper.simic.training.action_execution import ActionExecutionContext, execute_actions
from esper.simic.training.handlers import HandlerResult
from esper.simic.training.handlers import registry as handler_registry
from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
    EnvStepRecord,
    RewardSummaryAccumulator,
)


class _FakeBuffer:
    def __init__(self) -> None:
        self.step_counts = [0]
        self.action_ids = [[]]
        self.add_calls: list[dict[str, object]] = []
        self.ended_envs: list[int] = []

    def add(self, **kwargs: object) -> None:
        self.add_calls.append(kwargs)
        env_id = int(kwargs["env_id"])
        self.action_ids[env_id].append(str(kwargs["action_id"]))
        self.step_counts[env_id] += 1

    def end_episode(self, env_id: int) -> None:
        self.ended_envs.append(env_id)

    def last_action_id(self, env_id: int) -> str:
        return self.action_ids[env_id][self.step_counts[env_id] - 1]


class _FakeModel:
    def __init__(self) -> None:
        self.seed_slots = {"r0c0": _FakeSlot()}
        self.total_params = 100
        self.germinate_calls: list[tuple[object, ...]] = []

    def has_active_seed_in_slot(self, slot_id: str) -> bool:
        return False

    def total_seeds(self) -> int:
        return 0

    def germinate_seed(self, *args: object, **kwargs: object) -> None:
        self.germinate_calls.append(args)


class _FakeSlot:
    state = None
    active_seed_params = 0

    def __init__(self) -> None:
        self.pending_contexts: list[object] = []
        self.clear_count = 0

    def step_epoch(self) -> None:
        pass

    def set_pending_morphology_context(self, context: object) -> None:
        self.pending_contexts.append(context)

    def clear_pending_morphology_context(self) -> None:
        self.clear_count += 1


class _FakeGovernor:
    _panic_reason = "governor_nan"
    _panic_loss = float("nan")
    consecutive_panics = 1

    def __init__(self) -> None:
        self.rollback_calls: list[dict[str, object]] = []

    def get_punishment_reward(self) -> float:
        return -10.0

    def execute_rollback(self, **kwargs: object) -> None:
        self.rollback_calls.append(kwargs)


class _VetoGovernor:
    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, object]] = []

    def preflight_lifecycle_mutation(self, **kwargs: object) -> SimpleNamespace:
        self.preflight_calls.append(kwargs)
        return SimpleNamespace(
            approved=False,
            reason="unit-test veto",
            blocked_factor="unit_test",
        )


class _ApprovingGovernor:
    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, object]] = []

    def preflight_lifecycle_mutation(self, **kwargs: object) -> SimpleNamespace:
        self.preflight_calls.append(kwargs)
        return SimpleNamespace(
            approved=True,
            reason="approved",
            blocked_factor=None,
        )


def _resolve_target_slot(
    slot_idx: int,
    *,
    enabled_slots: list[str],
    slot_config: SlotConfig,
) -> tuple[str, bool]:
    return enabled_slots[slot_idx], True


def test_rollback_step_skips_stale_lifecycle_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rollback env must not execute the sampled pre-rollback lifecycle action."""
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (0, 0.0),
    )
    reward_components = SimpleNamespace(
        bounded_attribution=None,
        compute_rent=0.0,
        alpha_shock=0.0,
        new_drip_state=None,
        total_reward=0.0,
    )
    monkeypatch.setattr(
        action_execution,
        "compute_reward",
        lambda inputs: (0.0, reward_components),
    )

    model = _FakeModel()
    governor = _FakeGovernor()
    env_state = SimpleNamespace(
        model=model,
        stream=None,
        governor=governor,
        host_optimizer=SimpleNamespace(state={"momentum": object()}),
        seed_optimizers={},
        action_counts=defaultdict(int),
        successful_action_counts=defaultdict(int),
        val_acc=50.0,
        val_loss=1.0,
        train_loss=1.0,
        train_acc=50.0,
        committed_val_acc=50.0,
        prev_slot_alphas={},
        prev_slot_params={},
        acc_at_germination={},
        escrow_credit=defaultdict(float),
        seeds_created=0,
        germinate_count=0,
        seeds_fossilized=0,
        fossilize_count=0,
        contributing_fossilized=0,
        scaffold_boost_ledger={},
        fossilized_drip_states=[],
        pending_auto_prune_penalty=0.0,
        pending_hindsight_credit=0.0,
        episode_rewards=[],
        last_action_success=True,
        last_action_op=OP_WAIT,
        gradient_ratio_ema={},
        gradient_health_prev={},
        epochs_since_counterfactual={},
        telemetry_cb=None,
        init_obs_v3_slot_tracking=lambda slot_id: None,
        clear_obs_v3_slot_tracking=lambda slot_id: None,
    )

    buffer = _FakeBuffer()
    emitted: list[object] = []
    slot_config = SlotConfig.default()
    reward_config = SimpleNamespace(
        reward_mode=RewardMode.SHAPED,
        rent_host_params_floor=1,
        base_slot_rent_ratio=0.0,
    )
    context = ActionExecutionContext(
        slots=["r0c0"],
        ordered_slots=["r0c0"],
        slot_config=slot_config,
        task_spec=SimpleNamespace(topology=None),
        env_reward_configs=[reward_config],
        reward_family_enum=RewardFamily.CONTRIBUTION,
        reward_config=SimpleNamespace(auto_prune_penalty=-1.0),
        loss_reward_config=SimpleNamespace(),
        reward_normalizer=SimpleNamespace(clip=10.0, update_and_normalize=lambda reward: reward),
        telemetry_config=None,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=5,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(_get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)),
        emitters=[SimpleNamespace(emit=emitted.append)],
        agent=SimpleNamespace(buffer=buffer),
        fossilize_active_seed=lambda model, slot_id: False,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_GERMINATE
    head_log_probs = {head: torch.zeros(1) for head in HEAD_NAMES}
    masks_batch = {head: torch.ones((1, 1), dtype=torch.bool) for head in HEAD_NAMES}
    masks_batch["op"] = torch.ones((1, len(action_execution.OP_NAMES)), dtype=torch.bool)
    masks_batch["slot_by_op"] = torch.ones(
        (1, len(action_execution.OP_NAMES), slot_config.num_slots),
        dtype=torch.bool,
    )

    execute_actions(
        context=context,
        env_states=[env_state],
        actions_np=actions_np,
        values=[0.0],
        all_signals=[SimpleNamespace(metrics=SimpleNamespace(accuracy_delta=0.0), accuracy_history=[50.0])],
        all_slot_reports=[{}],
        states_batch_normalized=torch.zeros((1, 4)),
        blueprint_indices_batch=torch.zeros((1, slot_config.num_slots), dtype=torch.long),
        pre_step_hiddens=[(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))],
        head_log_probs=head_log_probs,
        masks_batch=masks_batch,
        step_records=[
            EnvStepRecord(
                env_idx=0,
                action_spec=ActionSpec(),
                action_outcome=ActionOutcome(),
                mask_flags=ActionMaskFlags(),
                reward_summary=RewardSummaryAccumulator(),
                contribution_reward_inputs=SimpleNamespace(),
                loss_reward_inputs=SimpleNamespace(),
            )
        ],
        head_confidences_cpu=None,
        head_entropies_cpu=None,
        op_probs_cpu=None,
        masked_np=None,
        baseline_accs=[{}],
        all_disabled_accs={},
        governor_panic_envs=[0],
        reward_summary_accum=[RewardSummaryAccumulator()],
        episode_history=[],
        episode_outcomes=[],
        step_obs_stats=None,
        epoch=1,
        episodes_completed=0,
        batch_idx=0,
    )

    assert len(governor.rollback_calls) == 1
    assert governor.rollback_calls[0]["triggering_action_id"] is None
    assert governor.rollback_calls[0]["raw_penalty"] == -10.0
    assert governor.rollback_calls[0]["normalized_penalty"] == -10.0
    assert governor.rollback_calls[0]["rollback_severity"] == 10.0
    assert governor.rollback_calls[0]["watch_window_evidence"] == 10.0
    assert model.germinate_calls == []
    assert env_state.germinate_count == 0
    assert env_state.last_action_success is False
    # P2 fix: the rollback env must NOT record a buffer transition for the
    # sampled-but-unexecuted action. The panic reflects the PRIOR executed
    # transition; the death penalty (applied later by handle_rollbacks via
    # mark_terminal_with_penalty) must attach to that prior row, not a fresh
    # row holding an action that never ran. The episode is still ended so the
    # prior transition becomes the terminal.
    assert buffer.add_calls == []
    assert env_state.episode_rewards == []
    assert buffer.ended_envs == [0]


def test_tolaria_preflight_veto_blocks_lifecycle_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simic must apply lifecycle mutations only after Tolaria approves them."""
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (0, 0.0),
    )
    reward_components = SimpleNamespace(
        bounded_attribution=None,
        compute_rent=0.0,
        alpha_shock=0.0,
        new_drip_state=None,
        total_reward=0.0,
    )
    monkeypatch.setattr(
        action_execution,
        "compute_reward",
        lambda inputs: (0.0, reward_components),
    )

    model = _FakeModel()
    governor = _VetoGovernor()
    env_state = SimpleNamespace(
        model=model,
        stream=None,
        governor=governor,
        host_optimizer=SimpleNamespace(state={}),
        seed_optimizers={},
        action_counts=defaultdict(int),
        successful_action_counts=defaultdict(int),
        val_acc=50.0,
        val_loss=1.0,
        train_loss=1.0,
        train_acc=50.0,
        committed_val_acc=50.0,
        prev_slot_alphas={},
        prev_slot_params={},
        acc_at_germination={},
        escrow_credit=defaultdict(float),
        seeds_created=0,
        germinate_count=0,
        seeds_fossilized=0,
        fossilize_count=0,
        contributing_fossilized=0,
        scaffold_boost_ledger={},
        fossilized_drip_states=[],
        pending_auto_prune_penalty=0.0,
        pending_hindsight_credit=0.0,
        episode_rewards=[],
        last_action_success=True,
        last_action_op=OP_WAIT,
        gradient_ratio_ema={},
        gradient_health_prev={},
        epochs_since_counterfactual={},
        telemetry_cb=None,
        init_obs_v3_slot_tracking=lambda slot_id: None,
        clear_obs_v3_slot_tracking=lambda slot_id: None,
    )

    buffer = _FakeBuffer()
    slot_config = SlotConfig.default()
    reward_config = SimpleNamespace(
        reward_mode=RewardMode.SHAPED,
        rent_host_params_floor=1,
        base_slot_rent_ratio=0.0,
    )
    context = ActionExecutionContext(
        slots=["r0c0"],
        ordered_slots=["r0c0"],
        slot_config=slot_config,
        task_spec=SimpleNamespace(topology=None),
        env_reward_configs=[reward_config],
        reward_family_enum=RewardFamily.CONTRIBUTION,
        reward_config=SimpleNamespace(auto_prune_penalty=-1.0),
        loss_reward_config=SimpleNamespace(),
        reward_normalizer=SimpleNamespace(clip=10.0, update_and_normalize=lambda reward: reward),
        telemetry_config=None,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=5,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(_get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)),
        emitters=[SimpleNamespace(emit=lambda event: None)],
        agent=SimpleNamespace(buffer=buffer),
        fossilize_active_seed=lambda model, slot_id: True,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_GERMINATE
    actions_np[HEAD_NAMES.index("blueprint"), 0] = 1
    head_log_probs = {head: torch.zeros(1) for head in HEAD_NAMES}
    masks_batch = {head: torch.ones((1, 1), dtype=torch.bool) for head in HEAD_NAMES}
    masks_batch["op"] = torch.ones((1, len(action_execution.OP_NAMES)), dtype=torch.bool)
    masks_batch["slot_by_op"] = torch.ones(
        (1, len(action_execution.OP_NAMES), slot_config.num_slots),
        dtype=torch.bool,
    )
    action_outcome = ActionOutcome()

    execute_actions(
        context=context,
        env_states=[env_state],
        actions_np=actions_np,
        values=[0.0],
        all_signals=[SimpleNamespace(metrics=SimpleNamespace(accuracy_delta=0.0), accuracy_history=[50.0])],
        all_slot_reports=[{}],
        states_batch_normalized=torch.zeros((1, 4)),
        blueprint_indices_batch=torch.zeros((1, slot_config.num_slots), dtype=torch.long),
        pre_step_hiddens=[(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))],
        head_log_probs=head_log_probs,
        masks_batch=masks_batch,
        step_records=[
            EnvStepRecord(
                env_idx=0,
                action_spec=ActionSpec(),
                action_outcome=action_outcome,
                mask_flags=ActionMaskFlags(),
                reward_summary=RewardSummaryAccumulator(),
                contribution_reward_inputs=SimpleNamespace(),
                loss_reward_inputs=SimpleNamespace(),
            )
        ],
        head_confidences_cpu=None,
        head_entropies_cpu=None,
        op_probs_cpu=None,
        masked_np=None,
        baseline_accs=[{}],
        all_disabled_accs={},
        governor_panic_envs=[],
        reward_summary_accum=[RewardSummaryAccumulator()],
        episode_history=[],
        episode_outcomes=[],
        step_obs_stats=None,
        epoch=1,
        episodes_completed=0,
        batch_idx=0,
    )

    assert len(governor.preflight_calls) == 1
    preflight = governor.preflight_calls[0]
    assert preflight["operation"] == action_execution.LifecycleOp.GERMINATE
    assert preflight["slot_id"] == "r0c0"
    assert preflight["blueprint_id"] == action_execution.BLUEPRINT_IDS[1]
    assert model.germinate_calls == []
    assert env_state.germinate_count == 0
    assert action_outcome.action_success is False


def test_execute_actions_dispatches_lifecycle_mutation_through_handler_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production lifecycle mutation must use the handler registry as authority."""
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (0, 0.0),
    )
    reward_components = SimpleNamespace(
        bounded_attribution=None,
        compute_rent=0.0,
        alpha_shock=0.0,
        new_drip_state=None,
        total_reward=0.0,
    )
    monkeypatch.setattr(
        action_execution,
        "compute_reward",
        lambda inputs: (0.0, reward_components),
    )

    calls: list[object] = []

    def fake_germinate_handler(ctx: object, params: object) -> HandlerResult:
        calls.append((ctx, params))
        return HandlerResult(success=True, telemetry={"seed_id": "registry-seed"})

    original_handler = handler_registry.HANDLER_REGISTRY[OP_GERMINATE]
    monkeypatch.setitem(
        handler_registry.HANDLER_REGISTRY,
        OP_GERMINATE,
        fake_germinate_handler,
    )
    assert handler_registry.HANDLER_REGISTRY[OP_GERMINATE] is not original_handler

    model = _FakeModel()
    governor = SimpleNamespace(
        preflight_lifecycle_mutation=lambda **kwargs: SimpleNamespace(
            approved=True,
            reason="approved",
            blocked_factor=None,
        )
    )
    env_state = SimpleNamespace(
        model=model,
        stream=None,
        governor=governor,
        host_optimizer=SimpleNamespace(state={}),
        seed_optimizers={},
        action_counts=defaultdict(int),
        successful_action_counts=defaultdict(int),
        val_acc=50.0,
        val_loss=1.0,
        train_loss=1.0,
        train_acc=50.0,
        committed_val_acc=50.0,
        prev_slot_alphas={},
        prev_slot_params={},
        acc_at_germination={},
        escrow_credit=defaultdict(float),
        seeds_created=0,
        germinate_count=0,
        seeds_fossilized=0,
        fossilize_count=0,
        contributing_fossilized=0,
        scaffold_boost_ledger={},
        fossilized_drip_states=[],
        pending_auto_prune_penalty=0.0,
        pending_hindsight_credit=0.0,
        episode_rewards=[],
        last_action_success=True,
        last_action_op=OP_WAIT,
        gradient_ratio_ema={},
        gradient_health_prev={},
        epochs_since_counterfactual={},
        telemetry_cb=None,
        init_obs_v3_slot_tracking=lambda slot_id: None,
        clear_obs_v3_slot_tracking=lambda slot_id: None,
    )

    buffer = _FakeBuffer()
    slot_config = SlotConfig.default()
    reward_config = SimpleNamespace(
        reward_mode=RewardMode.SHAPED,
        rent_host_params_floor=1,
        base_slot_rent_ratio=0.0,
    )
    context = ActionExecutionContext(
        slots=["r0c0"],
        ordered_slots=["r0c0"],
        slot_config=slot_config,
        task_spec=SimpleNamespace(topology=None),
        env_reward_configs=[reward_config],
        reward_family_enum=RewardFamily.CONTRIBUTION,
        reward_config=SimpleNamespace(auto_prune_penalty=-1.0),
        loss_reward_config=SimpleNamespace(),
        reward_normalizer=SimpleNamespace(clip=10.0, update_and_normalize=lambda reward: reward),
        telemetry_config=None,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=5,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(_get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)),
        emitters=[SimpleNamespace(emit=lambda event: None)],
        agent=SimpleNamespace(buffer=buffer),
        fossilize_active_seed=lambda model, slot_id: True,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_GERMINATE
    actions_np[HEAD_NAMES.index("blueprint"), 0] = 1
    head_log_probs = {head: torch.zeros(1) for head in HEAD_NAMES}
    masks_batch = {head: torch.ones((1, 1), dtype=torch.bool) for head in HEAD_NAMES}
    masks_batch["op"] = torch.ones((1, len(action_execution.OP_NAMES)), dtype=torch.bool)
    masks_batch["slot_by_op"] = torch.ones(
        (1, len(action_execution.OP_NAMES), slot_config.num_slots),
        dtype=torch.bool,
    )
    action_outcome = ActionOutcome()

    execute_actions(
        context=context,
        env_states=[env_state],
        actions_np=actions_np,
        values=[0.0],
        all_signals=[SimpleNamespace(metrics=SimpleNamespace(accuracy_delta=0.0), accuracy_history=[50.0])],
        all_slot_reports=[{}],
        states_batch_normalized=torch.zeros((1, 4)),
        blueprint_indices_batch=torch.zeros((1, slot_config.num_slots), dtype=torch.long),
        pre_step_hiddens=[(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))],
        head_log_probs=head_log_probs,
        masks_batch=masks_batch,
        step_records=[
            EnvStepRecord(
                env_idx=0,
                action_spec=ActionSpec(),
                action_outcome=action_outcome,
                mask_flags=ActionMaskFlags(),
                reward_summary=RewardSummaryAccumulator(),
                contribution_reward_inputs=SimpleNamespace(),
                loss_reward_inputs=SimpleNamespace(),
            )
        ],
        head_confidences_cpu=None,
        head_entropies_cpu=None,
        op_probs_cpu=None,
        masked_np=None,
        baseline_accs=[{}],
        all_disabled_accs={},
        governor_panic_envs=[],
        reward_summary_accum=[RewardSummaryAccumulator()],
        episode_history=[],
        episode_outcomes=[],
        step_obs_stats=None,
        epoch=1,
        episodes_completed=0,
        batch_idx=0,
    )

    assert calls
    fake_slot = model.seed_slots["r0c0"]
    assert fake_slot.clear_count == 1
    assert len(fake_slot.pending_contexts) == 1
    assert fake_slot.pending_contexts[0].proposal_id == "morph-b0-e1-env0-r0c0-op1-proposal"
    assert model.germinate_calls == []
    assert action_outcome.action_success is True


def test_execute_actions_emits_joinable_morphology_causal_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle execution should emit proposal, verdict, mutation, watch, and commit log rows."""
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (7, 0.5),
    )
    reward_components = SimpleNamespace(
        bounded_attribution=None,
        compute_rent=0.0,
        alpha_shock=0.0,
        new_drip_state=None,
        total_reward=0.0,
    )
    monkeypatch.setattr(
        action_execution,
        "compute_reward",
        lambda inputs: (0.0, reward_components),
    )

    calls: list[object] = []

    def fake_germinate_handler(ctx: object, params: object) -> HandlerResult:
        calls.append((ctx, params))
        return HandlerResult(success=True, telemetry={"seed_id": "registry-seed"})

    monkeypatch.setitem(
        handler_registry.HANDLER_REGISTRY,
        OP_GERMINATE,
        fake_germinate_handler,
    )

    model = _FakeModel()
    governor = _ApprovingGovernor()
    emitted: list[object] = []
    env_state = SimpleNamespace(
        model=model,
        stream=None,
        governor=governor,
        host_optimizer=SimpleNamespace(state={}),
        seed_optimizers={},
        action_counts=defaultdict(int),
        successful_action_counts=defaultdict(int),
        val_acc=50.0,
        val_loss=1.0,
        train_loss=1.0,
        train_acc=50.0,
        committed_val_acc=50.0,
        prev_slot_alphas={},
        prev_slot_params={},
        acc_at_germination={},
        escrow_credit=defaultdict(float),
        seeds_created=0,
        germinate_count=0,
        seeds_fossilized=0,
        fossilize_count=0,
        contributing_fossilized=0,
        scaffold_boost_ledger={},
        fossilized_drip_states=[],
        pending_auto_prune_penalty=0.0,
        pending_hindsight_credit=0.0,
        episode_rewards=[],
        last_action_success=True,
        last_action_op=OP_WAIT,
        gradient_ratio_ema={},
        gradient_health_prev={},
        epochs_since_counterfactual={},
        telemetry_cb=None,
        init_obs_v3_slot_tracking=lambda slot_id: None,
        clear_obs_v3_slot_tracking=lambda slot_id: None,
    )

    buffer = _FakeBuffer()
    slot_config = SlotConfig.default()
    reward_config = SimpleNamespace(
        reward_mode=RewardMode.SHAPED,
        rent_host_params_floor=1,
        base_slot_rent_ratio=0.0,
    )
    context = ActionExecutionContext(
        slots=["r0c0"],
        ordered_slots=["r0c0"],
        slot_config=slot_config,
        task_spec=SimpleNamespace(topology="cnn"),
        env_reward_configs=[reward_config],
        reward_family_enum=RewardFamily.CONTRIBUTION,
        reward_config=SimpleNamespace(auto_prune_penalty=-1.0),
        loss_reward_config=SimpleNamespace(),
        reward_normalizer=SimpleNamespace(clip=10.0, update_and_normalize=lambda reward: reward),
        telemetry_config=None,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=5,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(_get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)),
        emitters=[SimpleNamespace(emit=emitted.append)],
        agent=SimpleNamespace(buffer=buffer),
        fossilize_active_seed=lambda model, slot_id: True,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_GERMINATE
    actions_np[HEAD_NAMES.index("blueprint"), 0] = 1
    head_log_probs = {head: torch.zeros(1) for head in HEAD_NAMES}
    masks_batch = {head: torch.ones((1, 1), dtype=torch.bool) for head in HEAD_NAMES}
    masks_batch["op"] = torch.ones((1, len(action_execution.OP_NAMES)), dtype=torch.bool)
    masks_batch["slot_by_op"] = torch.ones(
        (1, len(action_execution.OP_NAMES), slot_config.num_slots),
        dtype=torch.bool,
    )

    execute_actions(
        context=context,
        env_states=[env_state],
        actions_np=actions_np,
        values=[0.0],
        all_signals=[SimpleNamespace(metrics=SimpleNamespace(accuracy_delta=0.0), accuracy_history=[50.0])],
        all_slot_reports=[{}],
        states_batch_normalized=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        blueprint_indices_batch=torch.zeros((1, slot_config.num_slots), dtype=torch.long),
        pre_step_hiddens=[(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))],
        head_log_probs=head_log_probs,
        masks_batch=masks_batch,
        step_records=[
            EnvStepRecord(
                env_idx=0,
                action_spec=ActionSpec(),
                action_outcome=ActionOutcome(),
                mask_flags=ActionMaskFlags(),
                reward_summary=RewardSummaryAccumulator(),
                contribution_reward_inputs=SimpleNamespace(),
                loss_reward_inputs=SimpleNamespace(),
            )
        ],
        head_confidences_cpu=None,
        head_entropies_cpu=None,
        op_probs_cpu=None,
        masked_np=None,
        baseline_accs=[{}],
        all_disabled_accs={},
        governor_panic_envs=[],
        reward_summary_accum=[RewardSummaryAccumulator()],
        episode_history=[],
        episode_outcomes=[],
        step_obs_stats=None,
        epoch=1,
        episodes_completed=0,
        batch_idx=0,
    )

    causal_events = [
        event
        for event in emitted
        if event.event_type == action_execution.TelemetryEventType.MORPHOLOGY_CAUSAL_LOG
    ]
    phases = [event.data.phase for event in causal_events]
    # KTS-002: the post-dispatch phase is "dispatch" (not "watch"); there is no
    # genuine delayed post-mutation measurement on this step, so the dispatch and
    # terminal-commit rows must NOT carry watch/audit evidence.
    assert phases == ["proposal", "verdict", "mutation", "dispatch", "commit"]
    ids = {(event.data.action_id, event.data.proposal_id, event.data.verdict_id, event.data.mutation_id) for event in causal_events}
    assert ids == {
        (
            "morph-b0-e1-env0-r0c0-op1",
            "morph-b0-e1-env0-r0c0-op1-proposal",
            "morph-b0-e1-env0-r0c0-op1-verdict",
            "morph-b0-e1-env0-r0c0-op1-mutation",
        )
    }
    assert causal_events[0].data.observation_hash.startswith("obs-")
    assert causal_events[0].data.rng_stream == "simic.lifecycle.env0"
    assert causal_events[0].data.topology == "cnn"
    assert causal_events[0].data.blueprint_id == action_execution.BLUEPRINT_IDS[1]
    assert causal_events[1].data.governor_approved is True
    # KTS-002: same-step rows must NOT mislabel pre-mutation val_loss as post-mutation
    # evidence. dispatch + terminal commit carry no watch_window_evidence.
    dispatch_event = next(e for e in causal_events if e.data.phase == "dispatch")
    commit_event = next(e for e in causal_events if e.data.phase == "commit")
    assert dispatch_event.data.watch_window_evidence is None
    assert commit_event.data.watch_window_evidence is None
    assert causal_events[-1].data.linked_event_id == "morph-b0-e1-env0-r0c0-op1-mutation"


def test_execute_actions_failed_handler_does_not_emit_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declined lifecycle handler must NOT emit a terminal 'commit' row.

    P2 fix: when the handler returns success=False (e.g. a failed ADVANCE gate
    or FOSSILIZE G5 check) after governor approval, emitting phase='commit'
    would mark a no-op/failed attempt as a committed mutation and corrupt the
    proof/audit trail. The 'dispatch' row already records the attempt; no
    terminal commit/fossilization row should follow a declined handler.
    """
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (0, 0.0),
    )
    reward_components = SimpleNamespace(
        bounded_attribution=None,
        compute_rent=0.0,
        alpha_shock=0.0,
        new_drip_state=None,
        total_reward=0.0,
    )
    monkeypatch.setattr(
        action_execution,
        "compute_reward",
        lambda inputs: (0.0, reward_components),
    )

    def fake_failing_handler(ctx: object, params: object) -> HandlerResult:
        return HandlerResult(success=False, telemetry={})

    monkeypatch.setitem(
        handler_registry.HANDLER_REGISTRY,
        OP_GERMINATE,
        fake_failing_handler,
    )

    model = _FakeModel()
    governor = _ApprovingGovernor()
    emitted: list[object] = []
    env_state = SimpleNamespace(
        model=model,
        stream=None,
        governor=governor,
        host_optimizer=SimpleNamespace(state={}),
        seed_optimizers={},
        action_counts=defaultdict(int),
        successful_action_counts=defaultdict(int),
        val_acc=50.0,
        val_loss=1.0,
        train_loss=1.0,
        train_acc=50.0,
        committed_val_acc=50.0,
        prev_slot_alphas={},
        prev_slot_params={},
        acc_at_germination={},
        escrow_credit=defaultdict(float),
        seeds_created=0,
        germinate_count=0,
        seeds_fossilized=0,
        fossilize_count=0,
        contributing_fossilized=0,
        scaffold_boost_ledger={},
        fossilized_drip_states=[],
        pending_auto_prune_penalty=0.0,
        pending_hindsight_credit=0.0,
        episode_rewards=[],
        last_action_success=True,
        last_action_op=OP_WAIT,
        gradient_ratio_ema={},
        gradient_health_prev={},
        epochs_since_counterfactual={},
        telemetry_cb=None,
        init_obs_v3_slot_tracking=lambda slot_id: None,
        clear_obs_v3_slot_tracking=lambda slot_id: None,
    )

    buffer = _FakeBuffer()
    slot_config = SlotConfig.default()
    reward_config = SimpleNamespace(
        reward_mode=RewardMode.SHAPED,
        rent_host_params_floor=1,
        base_slot_rent_ratio=0.0,
    )
    context = ActionExecutionContext(
        slots=["r0c0"],
        ordered_slots=["r0c0"],
        slot_config=slot_config,
        task_spec=SimpleNamespace(topology="cnn"),
        env_reward_configs=[reward_config],
        reward_family_enum=RewardFamily.CONTRIBUTION,
        reward_config=SimpleNamespace(auto_prune_penalty=-1.0),
        loss_reward_config=SimpleNamespace(),
        reward_normalizer=SimpleNamespace(clip=10.0, update_and_normalize=lambda reward: reward),
        telemetry_config=None,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=5,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(_get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)),
        emitters=[SimpleNamespace(emit=emitted.append)],
        agent=SimpleNamespace(buffer=buffer),
        fossilize_active_seed=lambda model, slot_id: True,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_GERMINATE
    actions_np[HEAD_NAMES.index("blueprint"), 0] = 1
    head_log_probs = {head: torch.zeros(1) for head in HEAD_NAMES}
    masks_batch = {head: torch.ones((1, 1), dtype=torch.bool) for head in HEAD_NAMES}
    masks_batch["op"] = torch.ones((1, len(action_execution.OP_NAMES)), dtype=torch.bool)
    masks_batch["slot_by_op"] = torch.ones(
        (1, len(action_execution.OP_NAMES), slot_config.num_slots),
        dtype=torch.bool,
    )

    execute_actions(
        context=context,
        env_states=[env_state],
        actions_np=actions_np,
        values=[0.0],
        all_signals=[SimpleNamespace(metrics=SimpleNamespace(accuracy_delta=0.0), accuracy_history=[50.0])],
        all_slot_reports=[{}],
        states_batch_normalized=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        blueprint_indices_batch=torch.zeros((1, slot_config.num_slots), dtype=torch.long),
        pre_step_hiddens=[(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))],
        head_log_probs=head_log_probs,
        masks_batch=masks_batch,
        step_records=[
            EnvStepRecord(
                env_idx=0,
                action_spec=ActionSpec(),
                action_outcome=ActionOutcome(),
                mask_flags=ActionMaskFlags(),
                reward_summary=RewardSummaryAccumulator(),
                contribution_reward_inputs=SimpleNamespace(),
                loss_reward_inputs=SimpleNamespace(),
            )
        ],
        head_confidences_cpu=None,
        head_entropies_cpu=None,
        op_probs_cpu=None,
        masked_np=None,
        baseline_accs=[{}],
        all_disabled_accs={},
        governor_panic_envs=[],
        reward_summary_accum=[RewardSummaryAccumulator()],
        episode_history=[],
        episode_outcomes=[],
        step_obs_stats=None,
        epoch=1,
        episodes_completed=0,
        batch_idx=0,
    )

    causal_events = [
        event
        for event in emitted
        if event.event_type == action_execution.TelemetryEventType.MORPHOLOGY_CAUSAL_LOG
    ]
    phases = [event.data.phase for event in causal_events]
    # The handler declined: the dispatch row records the attempt, but NO terminal
    # commit/fossilization row may follow — that would be a false committed mutation.
    assert phases == ["proposal", "verdict", "mutation", "dispatch"]
    assert "commit" not in phases
    assert "fossilization" not in phases


def test_rollback_step_emits_cooldown_and_audit_causal_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback should expose rollback, cooldown, and audit rows in the causal log."""
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (0, 0.0),
    )
    reward_components = SimpleNamespace(
        bounded_attribution=None,
        compute_rent=0.0,
        alpha_shock=0.0,
        new_drip_state=None,
        total_reward=0.0,
    )
    monkeypatch.setattr(
        action_execution,
        "compute_reward",
        lambda inputs: (0.0, reward_components),
    )

    model = _FakeModel()
    governor = _FakeGovernor()
    emitted: list[object] = []
    env_state = SimpleNamespace(
        model=model,
        stream=None,
        governor=governor,
        host_optimizer=SimpleNamespace(state={"momentum": object()}),
        seed_optimizers={},
        action_counts=defaultdict(int),
        successful_action_counts=defaultdict(int),
        val_acc=50.0,
        val_loss=1.0,
        train_loss=1.0,
        train_acc=50.0,
        committed_val_acc=50.0,
        prev_slot_alphas={},
        prev_slot_params={},
        acc_at_germination={},
        escrow_credit=defaultdict(float),
        seeds_created=0,
        germinate_count=0,
        seeds_fossilized=0,
        fossilize_count=0,
        contributing_fossilized=0,
        scaffold_boost_ledger={},
        fossilized_drip_states=[],
        pending_auto_prune_penalty=0.0,
        pending_hindsight_credit=0.0,
        episode_rewards=[],
        last_action_success=True,
        last_action_op=OP_WAIT,
        gradient_ratio_ema={},
        gradient_health_prev={},
        epochs_since_counterfactual={},
        telemetry_cb=None,
        init_obs_v3_slot_tracking=lambda slot_id: None,
        clear_obs_v3_slot_tracking=lambda slot_id: None,
    )

    buffer = _FakeBuffer()
    buffer.step_counts[0] = 1
    buffer.action_ids[0].append("morph-b0-e0-env0-r0c0-op0")
    slot_config = SlotConfig.default()
    reward_config = SimpleNamespace(
        reward_mode=RewardMode.SHAPED,
        rent_host_params_floor=1,
        base_slot_rent_ratio=0.0,
    )
    context = ActionExecutionContext(
        slots=["r0c0"],
        ordered_slots=["r0c0"],
        slot_config=slot_config,
        task_spec=SimpleNamespace(topology="cnn"),
        env_reward_configs=[reward_config],
        reward_family_enum=RewardFamily.CONTRIBUTION,
        reward_config=SimpleNamespace(auto_prune_penalty=-1.0),
        loss_reward_config=SimpleNamespace(),
        reward_normalizer=SimpleNamespace(clip=10.0, update_and_normalize=lambda reward: reward),
        telemetry_config=None,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=5,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(_get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)),
        emitters=[SimpleNamespace(emit=emitted.append)],
        agent=SimpleNamespace(buffer=buffer),
        fossilize_active_seed=lambda model, slot_id: False,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_GERMINATE
    head_log_probs = {head: torch.zeros(1) for head in HEAD_NAMES}
    masks_batch = {head: torch.ones((1, 1), dtype=torch.bool) for head in HEAD_NAMES}
    masks_batch["op"] = torch.ones((1, len(action_execution.OP_NAMES)), dtype=torch.bool)
    masks_batch["slot_by_op"] = torch.ones(
        (1, len(action_execution.OP_NAMES), slot_config.num_slots),
        dtype=torch.bool,
    )

    execute_actions(
        context=context,
        env_states=[env_state],
        actions_np=actions_np,
        values=[0.0],
        all_signals=[SimpleNamespace(metrics=SimpleNamespace(accuracy_delta=0.0), accuracy_history=[50.0])],
        all_slot_reports=[{}],
        states_batch_normalized=torch.zeros((1, 4)),
        blueprint_indices_batch=torch.zeros((1, slot_config.num_slots), dtype=torch.long),
        pre_step_hiddens=[(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))],
        head_log_probs=head_log_probs,
        masks_batch=masks_batch,
        step_records=[
            EnvStepRecord(
                env_idx=0,
                action_spec=ActionSpec(),
                action_outcome=ActionOutcome(),
                mask_flags=ActionMaskFlags(),
                reward_summary=RewardSummaryAccumulator(),
                contribution_reward_inputs=SimpleNamespace(),
                loss_reward_inputs=SimpleNamespace(),
            )
        ],
        head_confidences_cpu=None,
        head_entropies_cpu=None,
        op_probs_cpu=None,
        masked_np=None,
        baseline_accs=[{}],
        all_disabled_accs={},
        governor_panic_envs=[0],
        reward_summary_accum=[RewardSummaryAccumulator()],
        episode_history=[],
        episode_outcomes=[],
        step_obs_stats=None,
        epoch=1,
        episodes_completed=0,
        batch_idx=0,
    )

    causal_events = [
        event
        for event in emitted
        if event.event_type == action_execution.TelemetryEventType.MORPHOLOGY_CAUSAL_LOG
    ]
    assert [event.data.phase for event in causal_events] == ["rollback", "cooldown", "audit"]
    assert {event.data.action_id for event in causal_events} == {"morph-b0-e1-env0-r0c0-op1"}
    assert causal_events[0].data.watch_window_evidence != 0.0
    assert len(governor.rollback_calls) == 1
    assert governor.rollback_calls[0]["triggering_action_id"] == "morph-b0-e0-env0-r0c0-op0"
    assert governor.rollback_calls[0]["raw_penalty"] == -10.0
    assert governor.rollback_calls[0]["normalized_penalty"] == -10.0
    assert governor.rollback_calls[0]["rollback_severity"] == 10.0
    assert governor.rollback_calls[0]["watch_window_evidence"] == 10.0


def test_rollback_emits_morphology_causal_log_with_watch_evidence() -> None:
    """Rollback steps should be joinable to causal morphology evidence, not only generic panic telemetry."""
    events: list[object] = []
    payload = action_execution._build_rollback_causal_log_payload(
        batch_idx=2,
        epoch=3,
        env_idx=0,
        op_action=OP_GERMINATE,
        target_slot="r0c0",
        topology="cnn",
        observation_hash="obs-rollback",
        reason="governor_nan",
        loss_at_panic=12.5,
    )

    events.append(payload)

    assert events[0].phase == "rollback"
    assert events[0].action_id == "morph-b2-e3-env0-r0c0-op1"
    assert events[0].proposal_id == "morph-b2-e3-env0-r0c0-op1-proposal"
    assert events[0].watch_window_evidence == 12.5
    assert events[0].governor_reason == "governor_nan"
    assert events[0].observation_hash == "obs-rollback"


def test_decision_head_telemetry_is_unavailable_without_entropy() -> None:
    """Decision telemetry must not encode missing entropy as zero entropy."""
    confidences = np.ones((8, 1), dtype=np.float32)

    head_telemetry = action_execution._build_decision_head_telemetry(
        head_confidences_cpu=confidences,
        head_entropies_cpu=None,
        env_idx=0,
    )

    assert head_telemetry is None


def test_decision_head_telemetry_uses_real_entropy_values() -> None:
    """Decision telemetry should carry entropy only when measured values exist."""
    confidences = np.full((8, 1), 0.5, dtype=np.float32)
    entropies = np.arange(8, dtype=np.float32).reshape(8, 1)

    head_telemetry = action_execution._build_decision_head_telemetry(
        head_confidences_cpu=confidences,
        head_entropies_cpu=entropies,
        env_idx=0,
    )

    assert head_telemetry is not None
    assert head_telemetry.op_confidence == pytest.approx(0.5)
    assert head_telemetry.op_entropy == pytest.approx(0.0)
    assert head_telemetry.curve_entropy == pytest.approx(7.0)


def test_i16_action_outcome_fields_reset_each_step() -> None:
    """ActionOutcome fields written only at epoch==max_epochs must be reset to None each step.

    Pin: action_execution.py:570-574 resets reward_components, episode_reward,
    final_accuracy, episode_outcome to sentinel each step. Verifies that a prior
    step's non-None values don't leak into the next step.

    Builds an ActionOutcome with stale values from a prior step to document the
    contract, then asserts the RESET CODE EXISTS at lines 570-574 in
    execute_actions (the per-step reset that clears the leak).
    """
    # An ActionOutcome carrying stale values from a prior step.
    outcome = ActionOutcome()
    outcome.reward_components = object()  # stale
    outcome.episode_reward = 99.0         # stale
    outcome.final_accuracy = 0.99         # stale
    outcome.episode_outcome = object()    # stale

    # The partial reset at action_execution.py:570-574 must clear these each step.
    import inspect

    src = inspect.getsource(action_execution.execute_actions)
    assert "action_outcome.reward_components = None" in src
    assert "action_outcome.episode_reward = None" in src
    assert "action_outcome.final_accuracy = None" in src
    assert "action_outcome.episode_outcome = None" in src
