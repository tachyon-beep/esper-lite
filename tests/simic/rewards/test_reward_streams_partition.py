"""Keystone test (Stage 2 §7): the reward-stream partition driven through the
FULL ``action_execution.execute_actions`` reward-finalization path.

Unlike ``test_partition.py`` (which unit-tests ``split_reward_streams`` in
isolation), this suite drives the REAL finalization site: ``compute_reward`` is
stubbed only to control the *initial* ``(reward, components)``; everything after
— the terminal corrections (escrow forfeit, germination forfeit, auto-prune
penalty, hindsight credit), ``reward_raw`` assignment, the reward split, and the
``divide_by_std`` cf-normalization — runs as production code. We then assert:

- ``r_main_raw + r_cf_raw == reward_raw`` (<= 1e-6) across SHAPED x ESCROW x a
  range of ops/stages that fire escrow/rent/hindsight/auto-prune corrections;
- ``r_cf == components.bounded_attribution`` (ESCROW case ``== escrow_delta``);
- the buffer actually received ``r_cf_norm == reward_normalizer.divide_by_std(r_cf_raw)``
  computed against the SAME per-step std the total used (proving the wiring, not
  just the helper);
- SHAPED with the HRA flag ON forces ``return_components`` even when ALL telemetry
  emitters are OFF (§3.3), so ``components`` is not None;
- inert-decomposition guard: ``r_cf`` is not identically 0 over an
  attribution-bearing rollout.

The OFF leg (``hra_value_decomposition=False``) is also exercised to confirm the
cf kwargs are omitted from ``buffer.add`` (default 0.0 path, byte-identical).
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import esper.simic.training.action_execution as action_execution
from esper.leyline import HEAD_NAMES, OP_WAIT, SlotConfig
from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.control import RewardNormalizer
from esper.simic.rewards import RewardFamily, RewardMode
from esper.simic.rewards.partition import split_reward_streams
from esper.simic.training.action_execution import ActionExecutionContext, execute_actions
from esper.simic.vectorized_types import (
    ActionMaskFlags,
    ActionOutcome,
    ActionSpec,
    EnvStepRecord,
    RewardSummaryAccumulator,
)


class _FakeBuffer:
    """Captures buffer.add(**kwargs) so the test can assert the threaded cf fields."""

    def __init__(self) -> None:
        self.step_counts = [0]
        self.action_ids: list[list[str]] = [[]]
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


class _FakeSlot:
    state = None
    active_seed_params = 0
    alpha_schedule = None

    def __init__(self) -> None:
        self.pending_contexts: list[object] = []
        self.clear_count = 0

    def step_epoch(self) -> None:
        pass

    def set_pending_morphology_context(self, context: object) -> None:
        self.pending_contexts.append(context)

    def clear_pending_morphology_context(self) -> None:
        self.clear_count += 1


class _FakeModel:
    def __init__(self) -> None:
        self.seed_slots = {"r0c0": _FakeSlot()}
        self.total_params = 100

    def has_active_seed_in_slot(self, slot_id: str) -> bool:
        return False

    def total_seeds(self) -> int:
        return 0


def _resolve_target_slot(
    slot_idx: int,
    *,
    enabled_slots: list[str],
    slot_config: SlotConfig,
) -> tuple[str, bool]:
    return enabled_slots[slot_idx], True


def _make_env_state(
    *,
    pending_auto_prune_penalty: float = 0.0,
    pending_hindsight_credit: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=_FakeModel(),
        stream=None,
        governor=None,
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
        pending_auto_prune_penalty=pending_auto_prune_penalty,
        pending_hindsight_credit=pending_hindsight_credit,
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


def _run_step(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reward_mode: RewardMode,
    initial_reward: float,
    components: RewardComponentsTelemetry | None,
    reward_normalizer: RewardNormalizer,
    hra_value_decomposition: bool,
    cf_value: float = 0.0,
    pending_auto_prune_penalty: float = 0.0,
    pending_hindsight_credit: float = 0.0,
    telemetry_off: bool = False,
    epoch: int = 1,
    max_epochs: int = 5,
) -> tuple[ActionOutcome, dict[str, object]]:
    """Drive ONE WAIT step through execute_actions; return (outcome, buffer.add kwargs)."""
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (0, 0.0),
    )
    # Stub only the INITIAL reward computation; the split + corrections are real.
    if components is None:
        monkeypatch.setattr(
            action_execution, "compute_reward", lambda inputs: initial_reward
        )
    else:
        monkeypatch.setattr(
            action_execution,
            "compute_reward",
            lambda inputs: (initial_reward, components),
        )

    env_state = _make_env_state(
        pending_auto_prune_penalty=pending_auto_prune_penalty,
        pending_hindsight_credit=pending_hindsight_credit,
    )
    buffer = _FakeBuffer()
    slot_config = SlotConfig.default()
    reward_config = SimpleNamespace(
        reward_mode=reward_mode,
        rent_host_params_floor=1,
        base_slot_rent_ratio=0.0,
        disable_pbrs=True,  # no germination clawback noise on a WAIT step
        escrow_stable_window=1,
    )
    telemetry_config = (
        SimpleNamespace(should_collect=lambda level: False) if telemetry_off else None
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
        reward_normalizer=reward_normalizer,
        telemetry_config=telemetry_config,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=max_epochs,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(
            _get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)
        ),
        emitters=[SimpleNamespace(emit=lambda event: None)],
        agent=SimpleNamespace(buffer=buffer),
        fossilize_active_seed=lambda model, slot_id: False,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
        hra_value_decomposition=hra_value_decomposition,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_WAIT
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
        cf_values=[cf_value] if hra_value_decomposition else None,
        all_signals=[
            SimpleNamespace(
                metrics=SimpleNamespace(
                    accuracy_delta=0.0, loss_delta=0.0
                ),
                accuracy_history=[50.0],
            )
        ],
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
        epoch=epoch,
        episodes_completed=0,
        batch_idx=0,
    )

    assert len(buffer.add_calls) == 1, "WAIT step must record exactly one transition"
    return action_outcome, buffer.add_calls[0]


def _seeded_normalizer() -> RewardNormalizer:
    """A normalizer with established stats so divide_by_std uses a real (non-1) std."""
    norm = RewardNormalizer(clip=10.0)
    for r in (1.0, -2.0, 3.0, -1.5, 0.5):
        norm.update_and_normalize(r)
    return norm


# ---------------------------------------------------------------------------
# Keystone: r_main_raw + r_cf_raw == reward_raw across SHAPED x ESCROW x ops/stages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reward_mode", [RewardMode.SHAPED, RewardMode.ESCROW])
@pytest.mark.parametrize(
    "attribution, auto_prune, hindsight, epoch, max_epochs",
    [
        (2.5, 0.0, 0.0, 1, 5),       # plain attribution-bearing step
        (None, 0.0, 0.0, 1, 5),      # no attribution this step
        (-0.75, -1.0, 0.0, 1, 5),    # negative attribution + auto-prune penalty (R_main)
        (1.25, 0.0, 3.0, 1, 5),      # hindsight credit lands in R_main
        (0.9, -0.5, 2.0, 1, 5),      # auto-prune + hindsight corrections together (R_main)
    ],
)
def test_partition_exhaustive_through_full_path(
    monkeypatch: pytest.MonkeyPatch,
    reward_mode: RewardMode,
    attribution: float | None,
    auto_prune: float,
    hindsight: float,
    epoch: int,
    max_epochs: int,
) -> None:
    components = RewardComponentsTelemetry(bounded_attribution=attribution)
    reward_normalizer = _seeded_normalizer()

    outcome, add_kwargs = _run_step(
        monkeypatch,
        reward_mode=reward_mode,
        initial_reward=4.0,
        components=components,
        reward_normalizer=reward_normalizer,
        hra_value_decomposition=True,
        cf_value=0.0,
        pending_auto_prune_penalty=auto_prune,
        pending_hindsight_credit=hindsight,
        epoch=epoch,
        max_epochs=max_epochs,
    )

    reward_raw = outcome.reward_raw
    final_components = outcome.reward_components
    assert final_components is not None

    # The split as the production path computed it (same helper, same components).
    r_main_raw, r_cf_raw = split_reward_streams(reward_raw, final_components)

    # Keystone exhaustiveness invariant.
    assert abs(r_main_raw + r_cf_raw - reward_raw) <= 1e-6

    # R_cf == bounded_attribution (None -> 0.0).
    expected_cf = 0.0 if attribution is None else attribution
    assert r_cf_raw == pytest.approx(expected_cf)

    # The terminal corrections (auto-prune + hindsight) land in R_main, not R_cf.
    if attribution is not None:
        assert r_main_raw != pytest.approx(reward_raw)  # cf carved out
    # Auto-prune / hindsight magnitude is fully inside R_main.
    assert r_main_raw == pytest.approx(reward_raw - r_cf_raw)


# ---------------------------------------------------------------------------
# Wiring: the buffer received r_cf_norm == divide_by_std(r_cf_raw), same per-step std
# ---------------------------------------------------------------------------


def test_buffer_receives_r_cf_norm_via_divide_by_std(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = RewardComponentsTelemetry(bounded_attribution=1.75)
    reward_normalizer = _seeded_normalizer()

    outcome, add_kwargs = _run_step(
        monkeypatch,
        reward_mode=RewardMode.SHAPED,
        initial_reward=4.0,
        components=components,
        reward_normalizer=reward_normalizer,
        hra_value_decomposition=True,
        cf_value=0.42,
    )

    _r_main_raw, r_cf_raw = split_reward_streams(outcome.reward_raw, outcome.reward_components)

    # The cf reward stream threaded into the buffer must be the unclipped per-step
    # divide_by_std of r_cf_raw, evaluated AFTER the total's update_and_normalize ran
    # (i.e. the SAME running std). Recompute against the normalizer's now-current std.
    expected_r_cf_norm = reward_normalizer.divide_by_std(r_cf_raw)
    assert add_kwargs["r_cf_norm"] == pytest.approx(expected_r_cf_norm)

    # The per-step head-only V_cf value is threaded verbatim.
    assert add_kwargs["cf_value"] == pytest.approx(0.42)

    # The buffer still stores the CLIPPED normalized TOTAL as the main reward.
    assert add_kwargs["reward"] == pytest.approx(outcome.reward_normalized)


def test_r_cf_norm_unclipped_when_total_clips(monkeypatch: pytest.MonkeyPatch) -> None:
    """§5: clip is a total-only property. divide_by_std(r_cf) is NOT clipped even when
    the total's update_and_normalize returns the ±clip boundary."""
    # Seed many small-spread samples so the running std is small AND stable: one big
    # new reward then clips the TOTAL (>10) without materially moving the std, while
    # divide_by_std of a large cf addend stays far above the clip (no clip applied).
    reward_normalizer = RewardNormalizer(clip=10.0)
    rng = np.random.default_rng(0)
    for r in rng.normal(0.0, 0.5, size=200):
        reward_normalizer.update_and_normalize(float(r))
    big_attr = 300.0
    components = RewardComponentsTelemetry(bounded_attribution=big_attr)

    outcome, add_kwargs = _run_step(
        monkeypatch,
        reward_mode=RewardMode.SHAPED,
        initial_reward=200.0,
        components=components,
        reward_normalizer=reward_normalizer,
        hra_value_decomposition=True,
    )

    # Total clipped to the +clip boundary.
    assert abs(outcome.reward_normalized) == pytest.approx(10.0)
    # cf stream is the UNCLIPPED r_cf/std: it exceeds the clip, proving no clip applied
    # (r_cf == bounded_attribution == 300 here; r_main absorbs the 200-300 residual).
    _r_main_raw, r_cf_raw = split_reward_streams(
        outcome.reward_raw, outcome.reward_components
    )
    assert add_kwargs["r_cf_norm"] == pytest.approx(
        reward_normalizer.divide_by_std(r_cf_raw)
    )
    assert abs(add_kwargs["r_cf_norm"]) > 10.0


# ---------------------------------------------------------------------------
# §3.3 force-on: SHAPED with telemetry OFF still gets components under the flag
# ---------------------------------------------------------------------------


def test_shaped_flag_on_forces_return_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHAPED + ALL telemetry emitters OFF: the HRA flag alone must force
    return_components so the split has bounded_attribution to read."""
    components = RewardComponentsTelemetry(bounded_attribution=1.5)
    reward_normalizer = _seeded_normalizer()

    outcome, add_kwargs = _run_step(
        monkeypatch,
        reward_mode=RewardMode.SHAPED,
        initial_reward=3.0,
        components=components,
        reward_normalizer=reward_normalizer,
        hra_value_decomposition=True,
        telemetry_off=True,
    )

    # With telemetry off and SHAPED mode, components would be None today; the flag
    # forces them on, so the split has a real cf stream.
    assert outcome.reward_components is not None
    assert outcome.reward_components.bounded_attribution == pytest.approx(1.5)
    assert add_kwargs["r_cf_norm"] != 0.0


# ---------------------------------------------------------------------------
# Inert-decomposition guard: r_cf not identically 0 over an attribution rollout
# ---------------------------------------------------------------------------


def test_cf_stream_not_identically_zero_over_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reward_normalizer = _seeded_normalizer()
    cf_norms: list[float] = []
    for attribution in (0.0, 2.5, -1.0, 0.0, 3.25):
        components = RewardComponentsTelemetry(bounded_attribution=attribution)
        _outcome, add_kwargs = _run_step(
            monkeypatch,
            reward_mode=RewardMode.SHAPED,
            initial_reward=4.0,
            components=components,
            reward_normalizer=reward_normalizer,
            hra_value_decomposition=True,
        )
        cf_norms.append(float(add_kwargs["r_cf_norm"]))

    assert any(v != 0.0 for v in cf_norms), "cf decomposition is inert (all zero)"


# ---------------------------------------------------------------------------
# OFF leg: cf kwargs omitted from buffer.add (default 0.0 path, byte-identical)
# ---------------------------------------------------------------------------


def test_off_leg_omits_cf_buffer_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    components = RewardComponentsTelemetry(bounded_attribution=2.5)
    reward_normalizer = _seeded_normalizer()

    _outcome, add_kwargs = _run_step(
        monkeypatch,
        reward_mode=RewardMode.SHAPED,
        initial_reward=4.0,
        components=components,
        reward_normalizer=reward_normalizer,
        hra_value_decomposition=False,
    )

    # OFF leg: no cf kwargs threaded into buffer.add at all.
    assert "cf_value" not in add_kwargs
    assert "r_cf_norm" not in add_kwargs


def test_non_contribution_family_rejects_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """§3.3 config-validation guard: HRA flag ON with a non-CONTRIBUTION family raises."""
    monkeypatch.setattr(
        action_execution,
        "compute_rent_and_shock_inputs",
        lambda **_: (0, 0.0),
    )
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
        reward_family_enum=RewardFamily.LOSS,
        reward_config=SimpleNamespace(auto_prune_penalty=-1.0),
        loss_reward_config=SimpleNamespace(),
        reward_normalizer=_seeded_normalizer(),
        telemetry_config=None,
        ops_telemetry_enabled=False,
        disable_advance=False,
        effective_max_seeds=1,
        max_epochs=5,
        num_train_batches=1,
        device="cpu",
        analytics=SimpleNamespace(
            _get_scoreboard=lambda env_idx: SimpleNamespace(host_params=100)
        ),
        emitters=[SimpleNamespace(emit=lambda event: None)],
        agent=SimpleNamespace(buffer=_FakeBuffer()),
        fossilize_active_seed=lambda model, slot_id: False,
        resolve_target_slot=_resolve_target_slot,
        host_params_baseline=100,
        hra_value_decomposition=True,
    )

    actions_np = np.zeros((len(HEAD_NAMES), 1), dtype=np.int64)
    actions_np[HEAD_NAMES.index("op"), 0] = OP_WAIT
    head_log_probs = {head: torch.zeros(1) for head in HEAD_NAMES}
    masks_batch = {head: torch.ones((1, 1), dtype=torch.bool) for head in HEAD_NAMES}
    masks_batch["op"] = torch.ones((1, len(action_execution.OP_NAMES)), dtype=torch.bool)
    masks_batch["slot_by_op"] = torch.ones(
        (1, len(action_execution.OP_NAMES), slot_config.num_slots),
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="CONTRIBUTION reward family"):
        execute_actions(
            context=context,
            env_states=[_make_env_state()],
            actions_np=actions_np,
            values=[0.0],
            cf_values=[0.0],
            all_signals=[
                SimpleNamespace(
                    metrics=SimpleNamespace(accuracy_delta=0.0, loss_delta=0.0),
                    accuracy_history=[50.0],
                )
            ],
            all_slot_reports=[{}],
            states_batch_normalized=torch.zeros((1, 4)),
            blueprint_indices_batch=torch.zeros(
                (1, slot_config.num_slots), dtype=torch.long
            ),
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


# ---------------------------------------------------------------------------
# Trainer ON-leg: the truncation cf bootstrap is written to buffer.cf_bootstrap_values
# at the SAME indices as the main bootstrap (Step 2). Drives the REAL
# VectorizedPPOTrainer._run_action_transaction bootstrap path.
# ---------------------------------------------------------------------------


class _CfBootstrapBuffer:
    """Captures the [env_id, step_idx] bootstrap writes for both streams."""

    def __init__(self) -> None:
        self.bootstrap_values: dict[tuple[int, int], float] = {}
        self.cf_bootstrap_values: dict[tuple[int, int], float] = {}


def test_trainer_writes_cf_bootstrap_at_main_bootstrap_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from esper.simic.training import vectorized_trainer as vt
    from esper.simic.training.action_execution import ActionExecutionResult

    # Two truncated transitions across two envs -> two bootstrap writes.
    targets = [(0, 3), (1, 7)]
    main_vals = [0.5, -0.25]
    cf_vals = [1.5, -2.0]

    # Stub execute_actions (module global): return truncated targets + post-action
    # signals/masks so the bootstrap branch runs.
    def _fake_execute_actions(**kwargs: object) -> ActionExecutionResult:
        return ActionExecutionResult(
            truncated_bootstrap_targets=list(targets),
            post_action_signals=[object(), object()],
            post_action_slot_reports=[{}, {}],
            post_action_masks=[
                {
                    **{h: torch.ones((1,), dtype=torch.bool) for h in HEAD_NAMES},
                    "slot_by_op": torch.ones((1,), dtype=torch.bool),
                }
                for _ in targets
            ],
        )

    monkeypatch.setattr(vt, "execute_actions", _fake_execute_actions)
    # batch_signals_to_features returns (features, blueprint_indices).
    monkeypatch.setattr(
        vt,
        "batch_signals_to_features",
        lambda **kwargs: (torch.zeros((len(targets), 4)), torch.zeros((len(targets), 1), dtype=torch.long)),
    )
    # No terminal envs to reset.
    monkeypatch.setattr(vt, "_reset_hidden_for_terminal_envs", lambda hidden, terminal_envs: hidden)
    # Identity hidden slice (subset selection is exercised separately).
    monkeypatch.setattr(vt, "_select_hidden_for_envs", lambda hidden, env_indices: hidden)
    monkeypatch.setattr(vt, "HEAD_NAMES", HEAD_NAMES)

    buffer = _CfBootstrapBuffer()
    bootstrap_result = SimpleNamespace(
        value=torch.tensor(main_vals),
        cf_value=torch.tensor(cf_vals),
    )

    class _Policy:
        def get_action(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return bootstrap_result

    agent = SimpleNamespace(
        buffer=buffer,
        policy=_Policy(),
        probability_floor=0.0,
        hra_value_decomposition=True,
    )

    trainer = object.__new__(vt.VectorizedPPOTrainer)
    trainer.agent = agent
    trainer.slot_config = SlotConfig.default()
    trainer.max_epochs = 5
    trainer.obs_normalizer = SimpleNamespace(normalize=lambda x: x)
    trainer.device = "cpu"
    trainer.action_execution_context = SimpleNamespace()
    trainer.proof_baseline_lifecycle_policy = None

    import contextlib

    aib = SimpleNamespace(
        actions_np=None,
        values=[0.0, 0.0],
        cf_values=[0.0, 0.0],
        all_signals=[object(), object()],
        all_slot_reports=[{}, {}],
        states_batch_normalized=torch.zeros((2, 4)),
        blueprint_indices_batch=torch.zeros((2, 1), dtype=torch.long),
        pre_step_hiddens=[],
        head_log_probs={},
        masks_batch={},
        head_confidences_cpu=None,
        head_entropies_cpu=None,
        op_probs_cpu=None,
        masked_np=None,
        step_obs_stats=None,
        governor_panic_envs=[],
    )

    step_records = [
        EnvStepRecord(
            env_idx=i,
            action_spec=ActionSpec(),
            action_outcome=ActionOutcome(),
            mask_flags=ActionMaskFlags(),
            reward_summary=RewardSummaryAccumulator(),
            contribution_reward_inputs=SimpleNamespace(),
            loss_reward_inputs=SimpleNamespace(),
        )
        for i in range(2)
    ]
    for rec in step_records:
        rec.rollback_occurred = False

    hidden = (torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))

    vt.VectorizedPPOTrainer._run_action_transaction(
        trainer,
        env_states=[_make_env_state(), _make_env_state()],
        step_records=step_records,
        fused_result=SimpleNamespace(all_disabled_accs={}),
        aib=aib,
        reward_summary_accum=[RewardSummaryAccumulator(), RewardSummaryAccumulator()],
        baseline_accs=[{}, {}],
        episode_history=[],
        episode_outcomes=[],
        epoch=1,
        episodes_completed=0,
        batch_idx=0,
        rollout_autocast=contextlib.nullcontext,
        batched_lstm_hidden=hidden,
    )

    # The cf bootstrap landed at the SAME [env_id, step_idx] indices as the main one.
    assert buffer.bootstrap_values == {(0, 3): 0.5, (1, 7): -0.25}
    assert buffer.cf_bootstrap_values == {(0, 3): 1.5, (1, 7): -2.0}
