"""Estimand-invariant core of the causal-contribution harness (R1 pilot).

CPU-only unit tests for the three provable pieces (design §5.1, §5.5, §5.6):

  T1  op-mask repair routes "suppressed-slot-was-the-only-option" steps to
      WAIT-only, never leaving an all-false slot row under a True op (no silent
      invalid sample, no empty-mask crash), and splits forced steps by reason.
  T2  the RNG split keeps controller sampling IDENTICAL regardless of how many
      blueprint/host draws happen between controller draws (pairing preserved),
      and constructing the split does not perturb the global default generator
      (controller-init / data-order pairing pillars).
  T3  the compare-point proves intervention-OFF == control at every step (CRN
      no-op), would CATCH an RNG offset, and blueprint-init is content-addressed
      (order-independent weights) and a literal no-op on the global CPU stream.
"""

from __future__ import annotations

import torch

from esper.leyline import LifecycleOp
from esper.simic.training.experiment_rng import ExperimentRngDomains, derive_child_seed
from esper.simic.training.intervention import (
    OffsetParityError,
    apply_suppress_slot_mask,
    assert_constant_controller_stride,
    assert_offset_free_parity,
    assert_target_slot_never_committed,
    split_forced_step_reasons,
)

WAIT = LifecycleOp.WAIT.value
GERMINATE = LifecycleOp.GERMINATE.value
ADVANCE = LifecycleOp.ADVANCE.value
NUM_OPS = len(LifecycleOp)
NUM_SLOTS = 3
R0C0 = 0  # SlotConfig("r0c0","r0c1","r0c2").index_for_slot_id("r0c0")


def _blank_masks(num_envs: int) -> dict[str, torch.Tensor]:
    op = torch.zeros(num_envs, NUM_OPS, dtype=torch.bool)
    op[:, WAIT] = True  # WAIT always valid (matches action_masks.py:271)
    slot_by_op = torch.zeros(num_envs, NUM_OPS, NUM_SLOTS, dtype=torch.bool)
    return {"op": op, "slot_by_op": slot_by_op}


def _assert_no_orphan_op(masks: dict[str, torch.Tensor]) -> None:
    """No op may be valid (True) while its slot_by_op row is all-false.

    WAIT is exempt: its slot_by_op row is intentionally never populated
    (action_masks.py), and WAIT acts on no slot.
    """
    op = masks["op"]
    slot_by_op = masks["slot_by_op"]
    has_slot = slot_by_op.any(dim=-1)  # [E, O]
    for o in range(NUM_OPS):
        if o == WAIT:
            continue
        orphan = op[:, o] & ~has_slot[:, o]
        assert not bool(orphan.any()), f"op {o} valid with all-false slot row"


# ---------------------------------------------------------------------------
# T1 — op-mask repair
# ---------------------------------------------------------------------------
def test_repair_routes_r0c0_only_germinate_to_wait() -> None:
    # Env 0: r0c0 is the ONLY germinable slot -> GERMINATE must die, WAIT remains.
    masks = _blank_masks(1)
    masks["op"][0, GERMINATE] = True
    masks["slot_by_op"][0, GERMINATE, R0C0] = True

    res = apply_suppress_slot_mask(masks, suppressed_slot_index=R0C0)

    assert not bool(masks["op"][0, GERMINATE]), "GERMINATE must be repaired off"
    assert bool(masks["op"][0, WAIT]), "WAIT must stay valid"
    assert int(masks["op"][0].sum()) == 1, "env must be WAIT-only"
    assert bool(res.intervention_forced[0]) and res.intervention_forced_count == 1
    assert not bool(res.wait_saturation_forced[0])
    assert not bool(masks["slot_by_op"][:, :, R0C0].any()), "r0c0 column must be zeroed"
    _assert_no_orphan_op(masks)


def test_repair_keeps_germinate_when_other_slot_available() -> None:
    # Env 0: GERMINATE can go to r0c0 OR r0c1 -> survives at r0c1, never r0c0.
    masks = _blank_masks(1)
    masks["op"][0, GERMINATE] = True
    masks["slot_by_op"][0, GERMINATE, R0C0] = True
    masks["slot_by_op"][0, GERMINATE, 1] = True

    res = apply_suppress_slot_mask(masks, suppressed_slot_index=R0C0)

    assert bool(masks["op"][0, GERMINATE]), "GERMINATE keeps a valid (non-r0c0) slot"
    assert not bool(masks["slot_by_op"][0, GERMINATE, R0C0])
    assert bool(masks["slot_by_op"][0, GERMINATE, 1])
    assert res.intervention_forced_count == 0
    _assert_no_orphan_op(masks)


def test_repair_multi_env_multi_op_invariant() -> None:
    # Env0: GERMINATE only-r0c0 (forced). Env1: ADVANCE r0c1 (untouched).
    # Env2: already WAIT-only (wait-saturation). Env3: GERMINATE r0c0+r0c2 (survives).
    masks = _blank_masks(4)
    masks["op"][0, GERMINATE] = True
    masks["slot_by_op"][0, GERMINATE, R0C0] = True
    masks["op"][1, ADVANCE] = True
    masks["slot_by_op"][1, ADVANCE, 1] = True
    # env2: only WAIT
    masks["op"][3, GERMINATE] = True
    masks["slot_by_op"][3, GERMINATE, R0C0] = True
    masks["slot_by_op"][3, GERMINATE, 2] = True

    res = apply_suppress_slot_mask(masks, suppressed_slot_index=R0C0)

    assert res.intervention_forced.tolist() == [True, False, False, False]
    assert res.wait_saturation_forced.tolist() == [False, False, True, False]
    assert bool(masks["op"][1, ADVANCE])  # untouched op survives
    assert bool(masks["op"][3, GERMINATE])  # survives via r0c2
    assert not bool(masks["slot_by_op"][:, :, R0C0].any())
    _assert_no_orphan_op(masks)


def test_wait_saturation_is_mask_derivable_for_parity_check() -> None:
    # The trainer computes wait_saturation = total_wait_only - intervention_forced
    # from the POST-hook op mask, so it is defined for EVERY arm (incl. control/OFF
    # where there is no SuppressSlotResult). This pins that identity to the result.
    masks = _blank_masks(4)
    masks["op"][0, GERMINATE] = True  # only-r0c0 -> intervention-forced
    masks["slot_by_op"][0, GERMINATE, R0C0] = True
    masks["op"][1, GERMINATE] = True  # r0c1 available -> not forced
    masks["slot_by_op"][1, GERMINATE, 1] = True
    # env2, env3: naturally WAIT-only (wait-saturation)
    res = apply_suppress_slot_mask(masks, suppressed_slot_index=R0C0)

    total_wait_only = int((masks["op"].sum(dim=-1) == 1).sum().item())
    derived_wait_saturation = total_wait_only - res.intervention_forced_count
    assert derived_wait_saturation == res.wait_saturation_count == 2

    # Control-like mask (no suppression): every WAIT-only env is natural saturation.
    ctrl = _blank_masks(3)
    ctrl["op"][0, GERMINATE] = True
    ctrl["slot_by_op"][0, GERMINATE, 1] = True  # env0 has a real op (not forced)
    ctrl_total_wait_only = int((ctrl["op"].sum(dim=-1) == 1).sum().item())
    assert ctrl_total_wait_only == 2  # envs 1,2 are WAIT-only naturally


def test_split_forced_step_reasons_pure() -> None:
    pre = torch.zeros(3, NUM_OPS, dtype=torch.bool)
    pre[:, WAIT] = True
    pre[0, GERMINATE] = True  # had a non-WAIT op
    pre[1, GERMINATE] = True
    # env2 already WAIT-only
    post = pre.clone()
    post[0, GERMINATE] = False  # env0 flipped to WAIT-only by intervention
    # env1 keeps GERMINATE (not forced)
    interv, wait_sat = split_forced_step_reasons(pre, post, wait_idx=WAIT)
    assert interv.tolist() == [True, False, False]
    assert wait_sat.tolist() == [False, False, True]


def test_suppress_slot_index_out_of_range_raises() -> None:
    masks = _blank_masks(1)
    try:
        apply_suppress_slot_mask(masks, suppressed_slot_index=NUM_SLOTS)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for out-of-range slot index")


# ---------------------------------------------------------------------------
# T2 — RNG split: controller insulation + pairing preservation
# ---------------------------------------------------------------------------
def _draw_controller(rng: ExperimentRngDomains, n: int) -> list[int]:
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    return [
        int(torch.multinomial(probs, 1, generator=rng.controller_generator).item())
        for _ in range(n)
    ]


def test_controller_insulated_from_blueprint_and_host_draws() -> None:
    # Arm A: controller draws with NOTHING else touching RNG.
    rng_a = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    seq_a = _draw_controller(rng_a, 8)

    # Arm B: same master seed, but blueprint-init forks + host draws interleaved
    # between every controller draw (the exact thing that desyncs a shared stream).
    rng_b = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    host = rng_b.host_generator(0, env_seed=41, device="cpu")
    seq_b: list[int] = []
    probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    for i in range(8):
        with rng_b.blueprint_init(content_seed=1000 + i):
            torch.randn(5)  # CPU weight-init style draw
        torch.randn(3, generator=host)  # host shape-probe style draw
        seq_b.append(
            int(torch.multinomial(probs, 1, generator=rng_b.controller_generator).item())
        )

    assert seq_a == seq_b, "controller sampling must be insulated from B/C draw counts"


def test_construction_does_not_perturb_global_default_generator() -> None:
    # Controller init + data order key off the global default; the split must not
    # touch it (no torch.manual_seed mid-run).
    torch.manual_seed(123)
    before = torch.get_rng_state()
    _rng = ExperimentRngDomains(master_seed=999, controller_device="cpu")
    _ = _rng.host_generator(0, env_seed=7, device="cpu")
    after = torch.get_rng_state()
    assert torch.equal(before, after), "split construction perturbed global default RNG"


def test_distinct_master_seeds_give_distinct_controller_streams() -> None:
    a = _draw_controller(ExperimentRngDomains(master_seed=41, controller_device="cpu"), 16)
    b = _draw_controller(ExperimentRngDomains(master_seed=42, controller_device="cpu"), 16)
    assert a != b


# ---------------------------------------------------------------------------
# T3 — compare-point: CRN no-op proof, offset detection, content addressing
# ---------------------------------------------------------------------------
def test_compare_point_off_is_crn_identical_to_control() -> None:
    # "control" and "intervention-OFF" run the identical sequence (OFF never
    # suppresses, so it never changes the controller draw COUNT). Per-step
    # compare-point hashes must match at EVERY step -> whole-run no-op proof.
    control = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    off = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    for _ in range(20):
        _draw_controller(control, 3)
        _draw_controller(off, 3)
        assert control.compare_point() == off.compare_point()


def test_compare_point_detects_rng_offset() -> None:
    # If one arm's controller draws a DIFFERENT number of times (the offset bug),
    # the controller hash diverges -> the compare-point catches it.
    a = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    b = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    _draw_controller(a, 3)
    _draw_controller(b, 4)  # one extra draw == an offset
    assert a.controller_state_hash() != b.controller_state_hash()


def test_controller_hash_independent_of_sampled_values() -> None:
    # multinomial consumes fixed entropy regardless of probs, so legitimate
    # action divergence (different masked distribution) does NOT move the hash;
    # only a draw-count offset does. This underpins the compare-point claim.
    a = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    b = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    sharp = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
    flat = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    for _ in range(5):
        torch.multinomial(sharp, 1, generator=a.controller_generator)
        torch.multinomial(flat, 1, generator=b.controller_generator)
    assert a.controller_state_hash() == b.controller_state_hash()


def test_blueprint_init_is_content_addressed_and_order_independent() -> None:
    rng = ExperimentRngDomains(master_seed=41, controller_device="cpu")

    def make_weight(content_seed: int) -> torch.Tensor:
        with rng.blueprint_init(content_seed=content_seed):
            lin = torch.nn.Linear(4, 4)  # stock reset_parameters draws CPU default
        return lin.weight.detach().clone()

    w_first = make_weight(777)
    # Advance the world arbitrarily (different number of intervening draws/germs).
    for s in range(5):
        make_weight(900 + s)
    torch.randn(123)
    w_again = make_weight(777)

    assert torch.equal(w_first, w_again), "same content seed must give identical weights"


def test_blueprint_init_restores_global_cpu_stream_exactly() -> None:
    # The fork is a literal no-op on the global CPU default generator: a run with
    # the manager but a germination must leave the global stream byte-identical.
    rng = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    torch.manual_seed(555)
    before = torch.get_rng_state()
    with rng.blueprint_init(content_seed=12345):
        torch.nn.Linear(8, 8)
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    assert rng.blueprint_germination_count == 1


def test_derive_child_seed_is_positive_and_deterministic() -> None:
    s1 = derive_child_seed(41, 0x5EED_C0DE)
    s2 = derive_child_seed(41, 0x5EED_C0DE)
    assert s1 == s2 and 0 <= s1 < (1 << 63)
    assert derive_child_seed(41, 1) != derive_child_seed(42, 1)


# ---------------------------------------------------------------------------
# Hardening: device-portable draw count + strided hashes + analysis gates
# ---------------------------------------------------------------------------
def test_controller_draw_count_is_device_portable_and_monotone() -> None:
    rng = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    assert rng.controller_draw_count == 0
    for i in range(1, 6):
        rng.record_controller_invocation()
        assert rng.controller_draw_count == i


def test_compare_point_strides_state_hashes_but_not_counts() -> None:
    rng = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    rng.record_controller_invocation()
    cheap = rng.compare_point(include_state_hashes=False)
    assert cheap["controller_draw_count"] == 1  # cheap signal always present
    assert cheap["controller_state_hash"] is None  # GPU-sync skipped
    assert cheap["host_state_hash"] is None
    full = rng.compare_point(include_state_hashes=True)
    assert full["controller_state_hash"] is not None


def test_blueprint_count_not_inflated_by_failed_germination() -> None:
    rng = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    try:
        with rng.blueprint_init(content_seed=123):
            raise RuntimeError("blueprint construction blew up")
    except RuntimeError:
        pass
    assert rng.blueprint_germination_count == 0  # finally→try: no overcount
    # ...and the global CPU stream is still restored despite the failure.
    import torch as _t

    _t.manual_seed(7)
    before = _t.get_rng_state()
    try:
        with rng.blueprint_init(content_seed=999):
            raise RuntimeError("again")
    except RuntimeError:
        pass
    assert _t.equal(before, _t.get_rng_state())


def test_within_run_stride_constancy_and_break_detection() -> None:
    assert assert_constant_controller_stride([0, 1, 2, 3, 4]) == 1
    assert assert_constant_controller_stride([0, 2, 4, 6]) == 2
    try:
        assert_constant_controller_stride([0, 1, 2, 4])  # offset at last step
    except OffsetParityError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected OffsetParityError on stride break")


def test_offset_free_parity_gate() -> None:
    assert_offset_free_parity([0, 1, 2, 3], [0, 1, 2, 3])  # paired: OK
    assert_offset_free_parity(
        [0, 1, 2], [0, 1, 2],
        control_state_hashes=["a", None, "c"],
        other_state_hashes=["a", None, "c"],
    )
    try:
        assert_offset_free_parity([0, 1, 2, 3], [0, 1, 2, 4])  # offset divergence
    except OffsetParityError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected OffsetParityError on draw-count mismatch")


def test_target_slot_never_committed_gate() -> None:
    assert_target_slot_never_committed(["r0c1", "r0c2", "r0c1"])  # OK
    try:
        assert_target_slot_never_committed(["r0c1", "r0c0"])  # suppression breach
    except OffsetParityError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected OffsetParityError on r0c0 commit")


# ---------------------------------------------------------------------------
# Hook integration — the trainer's actual mask path + the literal OFF gate
# ---------------------------------------------------------------------------
from esper.leyline import SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY  # noqa: E402
from esper.leyline.slot_config import SlotConfig  # noqa: E402
from esper.simic.training.vectorized_trainer import (  # noqa: E402
    apply_proof_baseline_action_controls,
)

_SLOT_CONFIG = SlotConfig.default()  # ("r0c0","r0c1","r0c2")


def _germinable_masks(num_envs: int) -> dict[str, torch.Tensor]:
    masks = _blank_masks(num_envs)
    masks["op"][:, GERMINATE] = True
    masks["slot_by_op"][:, GERMINATE, :] = True  # all slots germinable
    return masks


def test_hook_control_is_noop() -> None:
    masks = _germinable_masks(2)
    ref = {k: v.clone() for k, v in masks.items()}
    out = apply_proof_baseline_action_controls(
        masks_batch=masks, lifecycle_policy=None, schedule_id=None, epoch=1
    )
    assert out is None
    assert all(torch.equal(masks[k], ref[k]) for k in ref)


def test_hook_suppress_slot_off_is_literal_noop_equal_to_control() -> None:
    # SUPPRESS-SLOT arm with suppression OFF must equal the control mask exactly
    # (byte-identical) — the CRN no-op proof at the hook level.
    masks_off = _germinable_masks(2)
    masks_ctrl = _germinable_masks(2)
    out = apply_proof_baseline_action_controls(
        masks_batch=masks_off,
        lifecycle_policy=SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY,
        schedule_id=None,
        epoch=1,
        slot_config=_SLOT_CONFIG,
        suppression_enabled=False,
    )
    assert out is None
    assert all(torch.equal(masks_off[k], masks_ctrl[k]) for k in masks_ctrl)


def test_hook_suppress_slot_on_mutates_and_returns_result() -> None:
    masks = _germinable_masks(2)
    out = apply_proof_baseline_action_controls(
        masks_batch=masks,
        lifecycle_policy=SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY,
        schedule_id=None,
        epoch=1,
        slot_config=_SLOT_CONFIG,
        suppression_enabled=True,
    )
    assert out is not None
    assert out.suppressed_slot_index == 0  # r0c0
    assert not bool(masks["slot_by_op"][:, :, 0].any())  # r0c0 column zeroed
    assert bool(masks["op"][:, GERMINATE].all())  # still germinable via r0c1/r0c2
    _assert_no_orphan_op(masks)


def test_hook_suppress_slot_rejects_schedule_provenance() -> None:
    masks = _germinable_masks(1)
    try:
        apply_proof_baseline_action_controls(
            masks_batch=masks,
            lifecycle_policy=SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY,
            schedule_id="bogus",
            epoch=1,
            slot_config=_SLOT_CONFIG,
            suppression_enabled=True,
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for schedule provenance on suppress-slot")


def test_hook_suppress_slot_on_requires_slot_config() -> None:
    masks = _germinable_masks(1)
    try:
        apply_proof_baseline_action_controls(
            masks_batch=masks,
            lifecycle_policy=SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY,
            schedule_id=None,
            epoch=1,
            slot_config=None,
            suppression_enabled=True,
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when slot_config missing")


# ---------------------------------------------------------------------------
# Slot germination wiring — content-addressed blueprint-init reads the pending
# morphology context's rng_seed (domains B + C end-to-end through SeedSlot).
# ---------------------------------------------------------------------------
from esper.kasmina.slot import SeedSlot  # noqa: E402
from esper.leyline.lifecycle_mutation import LifecycleMutationCausalContext  # noqa: E402


def _germinate_with_content_seed(rng: ExperimentRngDomains, content_seed: int) -> torch.Tensor:
    slot = SeedSlot(slot_id="r0c0", channels=16)
    slot.attach_experiment_rng(rng, env_idx=0, env_seed=41)
    ctx = LifecycleMutationCausalContext(
        action_id="a",
        proposal_id="p",
        verdict_id="v",
        mutation_id="m",
        observation_hash="o",
        rng_stream="s",
        rng_seed=content_seed,
        topology="cnn",
        slot_id="r0c0",
        operation="GERMINATE",
    )
    slot.set_pending_morphology_context(ctx)
    slot.germinate("norm", "seed", topology="cnn")
    slot.clear_pending_morphology_context()
    return torch.cat([p.detach().flatten() for p in slot.seed.parameters()])


def test_slot_germination_is_content_addressed_through_pending_context() -> None:
    rng = ExperimentRngDomains(master_seed=41, controller_device="cpu")
    w_first = _germinate_with_content_seed(rng, 0xABCDEF)
    # Different intervening germinations with other content seeds.
    for s in range(3):
        _germinate_with_content_seed(rng, 1000 + s)
    w_again = _germinate_with_content_seed(rng, 0xABCDEF)
    assert w_first.numel() > 0
    assert torch.equal(w_first, w_again), "same content seed -> identical seed weights"


def test_slot_germination_without_harness_is_unchanged() -> None:
    # No attached experiment_rng => global-default path (no fork, no crash).
    slot = SeedSlot(slot_id="r0c0", channels=16)
    state = slot.germinate("norm", "seed", topology="cnn")
    assert state is not None and slot.seed is not None


# ---------------------------------------------------------------------------
# Run-start INTERVENTION_CONFIGURED payload builder (§5.7)
# ---------------------------------------------------------------------------
from esper.simic.training.intervention import (  # noqa: E402
    build_intervention_configured_payload,
)


def test_configured_payload_for_suppress_slot_on() -> None:
    p = build_intervention_configured_payload(
        lifecycle_policy=SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY,
        suppression_enabled=True,
        slot_config=_SLOT_CONFIG,
        rng_split_enabled=True,
        master_seed=41,
    )
    assert p.proof_baseline_mode == "suppress_slot"
    assert p.suppressed_slot_id == "r0c0" and p.suppressed_slot_index == 0
    assert p.suppression_enabled is True
    assert p.determinism_class == "crn_statistical"
    assert p.rng_split_enabled is True and p.master_seed == 41


def test_configured_payload_for_control_is_split_on_suppression_off() -> None:
    p = build_intervention_configured_payload(
        lifecycle_policy=None,
        suppression_enabled=False,
        slot_config=_SLOT_CONFIG,
        rng_split_enabled=True,
        master_seed=42,
    )
    assert p.proof_baseline_mode == "control"
    assert p.suppressed_slot_id == "none" and p.suppressed_slot_index == -1
    assert p.suppression_enabled is False
    assert p.rng_split_enabled is True
