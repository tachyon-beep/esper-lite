"""Anchored reference pass: epoch-0 importance ratio must be exactly one at theta_0.

This is the RED test for the multi-epoch recurrent PPO "anchored reference pass"
design. The PPO importance-ratio baseline (old_log_probs) must be sourced from an
anchor forward pass over the unrolled rollout — the same forward that produces the
epoch-0 scored log_probs — rather than from the per-step rollout sampling forward.

When old_log_probs come from the anchor, the epoch-0 ratio = exp(scored - old) is
identically 1.0 (within FP tolerance) because, at theta_0, the anchor forward and
the epoch-0 scored forward are the SAME forward over the SAME parameters. Today the
baseline is sourced from the rollout sampling forward, whose per-step log_probs
differ from the unrolled scored log_probs, so the epoch-0 ratio is NOT 1.0.
"""

from __future__ import annotations

import pytest
import torch

from esper.leyline import HEAD_NAMES
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.simic.training.helpers import policy_amp_context
from esper.tamiyo.policy import create_policy

from tests.simic.test_ppo_update_golden import _fill_buffer


def _build_agent() -> tuple[PPOAgent, SlotConfig]:
    torch.manual_seed(123)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=None,
        recurrent_n_epochs=1,
    )
    return agent, slot_config


def test_epoch0_ratio_is_one_at_theta0() -> None:
    """At theta_0 with real rollout log_probs stored, the epoch-0 ratio is 1.0.

    The buffer stores the GENUINE per-head sampling log_probs 
    With the anchored reference pass, old_log_probs are re-derived from the anchor
    forward over the unrolled rollout, so ratio == 1.0 and approx_kl == 0.0 exactly.
    """
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    assert metrics["finiteness_gate_skip_count"] == 0

    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-5), (
        f"epoch-0 ratio_mean expected 1.0, got {metrics['ratio_mean']!r} "
        "(old_log_probs still sourced from the rollout sampling forward, "
        "not the anchored reference pass)"
    )
    assert metrics["ratio_max"] == pytest.approx(1.0, abs=1e-5), (
        f"epoch-0 ratio_max expected 1.0, got {metrics['ratio_max']!r}"
    )
    assert metrics["ratio_min"] == pytest.approx(1.0, abs=1e-5), (
        f"epoch-0 ratio_min expected 1.0, got {metrics['ratio_min']!r}"
    )
    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-6), (
        f"epoch-0 approx_kl expected 0.0, got {metrics['approx_kl']!r}"
    )


def _build_cuda_agent() -> tuple[PPOAgent, SlotConfig]:
    torch.manual_seed(123)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cuda",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cuda",
        target_kl=None,
        recurrent_n_epochs=1,
    )
    return agent, slot_config


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_epoch0_per_head_grad_norms_nonzero_under_amp() -> None:
    """Anchored reference pass must NOT poison the BF16 cast cache (Acceptance #2).

    The anchor forward is the cache-priming FIRST BF16 touch under the caller's
    grad-capable autocast (entered in production at vectorized.py:447). It runs
    under plain ``torch.no_grad()`` so it inherits that autocast exactly like the
    epoch-0 scored forward at ppo_agent.py:898 -- the only difference is no_grad +
    detach. If the anchor instead disabled autocast (autocast(enabled=False)), the
    no_grad forward would cache an FP32 weight that the epoch-0 grad forward then
    reuses, breaking the BF16 cast-weight cache and ZEROING per-head epoch-0
    gradients.

    This is the LOAD-BEARING discovery gate: under BF16 AMP, EVERY head's gradient
    norm must be strictly positive. A zero norm on any head means the anchor
    architecture is unsound under AMP and must not be merged.
    """
    agent, slot_config = _build_cuda_agent()
    _fill_buffer(agent, slot_config)

    # Mirror production: the CALLER enters BF16 autocast and update() inherits it.
    # Pass the torch.bfloat16 DTYPE OBJECT -- a string silently degrades to nullcontext.
    with policy_amp_context(amp_enabled=True, resolved_amp_dtype=torch.bfloat16):
        metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    assert metrics["finiteness_gate_skip_count"] == 0

    head_grad_norms = metrics["head_grad_norms"]  # nested dict {head: [norms...]}
    for head_name, norms in head_grad_norms.items():
        assert len(norms) > 0 and all(n > 0 for n in norms), (
            f"{head_name} grad norm is zero: {norms!r} -- the no_grad anchor poisoned "
            "the BF16 cast cache and zeroed epoch-0 head gradients (HARD GATE TRIP)."
        )


_ROLLOUT_LOG_PROB_KEYS = tuple(f"{head}_log_probs" for head in HEAD_NAMES)


def test_loss_path_reads_only_ref_log_probs() -> None:
    """The PPO loss path must consume the anchor baseline, never the rollout log_probs.

    Acceptance #4 (no stale-baseline consumption). We POISON every rollout per-head
    log_prob tensor (the buffer's stored self.{head}_log_probs) with NaN before
    update() runs. update() obtains its 'data' via agent.buffer.get_batched_sequences
    (ppo_agent.py:596), which copies those buffer attributes into data["{head}_log_probs"].
    The anchor forward re-derives ref_log_probs from data["states"]/actions/masks/
    initial_hidden, none of which we poison, so a correct loss path is UNAFFECTED.

    If any importance-ratio baseline read site still sourced old_log_probs from the
    rollout (the pre-anchor behaviour), the NaN poison would propagate into ratio_mean,
    policy_loss, value_loss and approx_kl. We assert ratio_mean == 1.0 and all losses
    finite -> the loss path read ONLY the anchor. We then DISCRIMINATE: we show the
    poison is "live" by re-deriving the network's epoch-0 log_probs and computing a
    manual ratio against the poisoned rollout baseline, asserting THAT ratio is
    non-finite (i.e. had the old path survived, the loss would be NaN).
    """
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    # --- Poison the rollout log_probs in place via the buffer's public surface ---
    # The buffer stores each head's rollout log_probs as self.{head}_log_probs.
    # get_batched_sequences() (the call site update() uses) copies these into
    # data["{head}_log_probs"]; NaN-filling them here corrupts ONLY the rollout
    # baseline the old (pre-anchor) path would have read. The anchor reads
    # states/actions/masks/initial_hidden, which we leave pristine.
    for head in HEAD_NAMES:
        rollout_lp = getattr(agent.buffer, f"{head}_log_probs")
        rollout_lp.fill_(float("nan"))

    # Snapshot the data update() will see (same buffer surface), for the discriminating
    # control below. Taken BEFORE update() so it predates the clear_buffer=True reset.
    poisoned_data = agent.buffer.get_batched_sequences(device=agent.device)

    metrics = agent.update(clear_buffer=True)

    # --- Primary assertions: loss path consumed the anchor, not the poison ---
    assert metrics["ppo_update_performed"] is True
    # The poison must NOT trip the finiteness gate: a tripped gate would mean some
    # read site still sourced the rollout log_probs.
    finiteness_failures = (
        metrics["finiteness_gate_failures"] if "finiteness_gate_failures" in metrics else None
    )
    assert metrics["finiteness_gate_skip_count"] == 0, (
        "finiteness gate tripped under poisoned rollout log_probs -- a baseline read "
        f"site still sources data['*_log_probs']: {finiteness_failures!r}"
    )

    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-5), (
        f"ratio_mean expected 1.0 (anchor baseline), got {metrics['ratio_mean']!r} -- "
        "loss path consumed the poisoned rollout log_probs, not ref_log_probs."
    )
    for loss_key in ("policy_loss", "value_loss", "approx_kl"):
        loss_val = torch.tensor(metrics[loss_key])
        assert torch.isfinite(loss_val).all(), (
            f"{loss_key} is non-finite ({metrics[loss_key]!r}) -- the poisoned rollout "
            "log_probs leaked into the loss path."
        )

    # --- Sanity: the poison really was NaN on every head in the data update() saw ---
    for key in _ROLLOUT_LOG_PROB_KEYS:
        assert torch.isnan(poisoned_data[key]).all(), (
            f"poison sanity check failed: {key} is not all-NaN"
        )

    # --- DISCRIMINATING CONTROL: prove the poison is live ---
    # Re-derive the network's epoch-0 log_probs at theta_0 (the same anchor forward
    # update() ran), then compute a manual importance ratio using the POISONED rollout
    # log_probs as the baseline. Had the loss path used the rollout baseline, this is
    # the ratio it would have seen: exp(new - NaN) -> non-finite.
    valid_mask = poisoned_data["valid_mask"]
    anchor_actions = {head: poisoned_data[f"{head}_actions"] for head in HEAD_NAMES}
    anchor_masks = {head: poisoned_data[f"{head}_masks"] for head in HEAD_NAMES}
    with torch.no_grad():
        epoch0 = agent.policy.evaluate_actions(
            poisoned_data["states"],
            poisoned_data["blueprint_indices"],
            anchor_actions,
            anchor_masks,
            hidden=(poisoned_data["initial_hidden_h"], poisoned_data["initial_hidden_c"]),
            probability_floor=agent.probability_floor,
            aux_stop_gradient=agent.aux_stop_gradient,
        )
    for head in HEAD_NAMES:
        new_lp = epoch0.log_prob[head][valid_mask]
        # The network's epoch-0 log_probs are themselves finite (anchor is healthy)...
        assert torch.isfinite(new_lp).all(), (
            f"epoch-0 {head} log_probs unexpectedly non-finite -- anchor forward is unhealthy"
        )
        poisoned_old_lp = poisoned_data[f"{head}_log_probs"][valid_mask]
        manual_ratio = torch.exp(new_lp - poisoned_old_lp)
        # ...but the ratio against the POISONED rollout baseline IS non-finite, proving
        # the poison would corrupt the loss had the old (rollout-sourced) path survived.
        assert not torch.isfinite(manual_ratio).all(), (
            f"discriminating control failed for {head}: ratio against poisoned rollout "
            "baseline was finite -- the poison is not live, so the GREEN result above is "
            "not actually discriminating between anchor and rollout baselines."
        )

    # --- DIAGNOSTIC COVERAGE: no non-finite attribution to the rollout ---
    # If the finiteness slow-path had fired it would attribute to "ref_log_probs[...]"
    # (the anchor), never to "old_log_probs[...]" / "*_log_probs" (the rollout). With a
    # healthy anchor it does not fire at all; the key is only present on failure, so its
    # ABSENCE is itself the proof that no rollout-sourced NaN was attributed.
    if "finiteness_gate_failures" in metrics:
        for failure in metrics["finiteness_gate_failures"]:
            for source in failure["sources"]:
                assert not source.startswith("old_log_probs["), (
                    f"finiteness slow-path attributed a non-finite to the rollout baseline: {source!r}"
                )
                # The rollout per-head buffer keys must never appear as a non-finite source.
                for key in _ROLLOUT_LOG_PROB_KEYS:
                    assert key not in source, (
                        f"finiteness slow-path attributed a non-finite to rollout key {key!r}: {source!r}"
                    )

    # The ratio-explosion diagnostic (key present only when a ratio actually exploded)
    # reads old_log_probs['op'], which is the anchor baseline (ref_log_probs). With
    # ratio == 1.0 it does not fire; if it did, its statistics would be finite (anchor),
    # not the poisoned rollout NaN.
    if "ratio_diagnostic" in metrics:
        diag = metrics["ratio_diagnostic"]
        for diag_key, diag_val in diag.items():
            if isinstance(diag_val, (int, float)) and not isinstance(diag_val, bool):
                assert torch.isfinite(torch.tensor(float(diag_val))).all(), (
                    f"ratio diagnostic field {diag_key!r} is non-finite ({diag_val!r}) -- "
                    "diagnostic sourced the poisoned rollout baseline, not ref_log_probs."
                )


def _build_agent_no_aux() -> tuple[PPOAgent, SlotConfig]:
    """K=1 CPU LSTM agent with the contribution-aux head disabled.

    enable_contribution_aux=False removes the Dropout-bearing auxiliary
    supervision path from the optimised loss, keeping the update fully
    deterministic (no RNG) so the behaviour-invariance assertions below are
    not perturbed by a stochastic aux term. The anchor forward already runs
    under no_grad with detached aux, so the reference baseline is unaffected
    either way.
    """
    torch.manual_seed(123)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=None,
        recurrent_n_epochs=1,
        enable_contribution_aux=False,
    )
    return agent, slot_config


def test_k1_finiteness_and_update_performed_unchanged() -> None:
    """At K=1 the anchored reference pass preserves the single-epoch contract.

    Acceptance #9 @ K=1 (behaviour invariance). With the anchor in place, a K=1
    update over a buffer filled with GENUINE rollout log_probs must still:
      * perform the optimizer step (ppo_update_performed is True),
      * record ZERO finiteness-gate skips (the anchor baseline is finite, so no
        epoch is dropped), and
      * produce finite value_loss / return_mean / return_std.
    Together these prove the anchor did not regress the established single-epoch
    behaviour: it neither suppressed the update nor injected non-finite values
    into the value/return path.
    """
    agent, slot_config = _build_agent_no_aux()
    _fill_buffer(agent, slot_config)

    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True, (
        "K=1 update did not perform an optimizer step -- the anchor suppressed the "
        "single-epoch update."
    )
    # No finiteness-gate failures: zero skips and an empty failure list. The key is
    # always present (mirrors the defaultdict(list) source); a TRIPPED gate is signalled
    # by a NON-EMPTY list, so emptiness is the invariant, not the key's absence.
    assert metrics["finiteness_gate_skip_count"] == 0, (
        "finiteness gate tripped at K=1 with genuine rollout log_probs: "
        f"{metrics['finiteness_gate_skip_count']} skip(s)."
    )
    assert metrics["finiteness_gate_failures"] == [], (
        "finiteness_gate_failures recorded at K=1 despite a healthy anchor baseline: "
        f"{metrics['finiteness_gate_failures']!r}"
    )

    for key in ("value_loss", "return_mean", "return_std"):
        value = torch.tensor(float(metrics[key]))
        assert torch.isfinite(value).all(), (
            f"{key} is non-finite ({metrics[key]!r}) at K=1 -- the anchor perturbed "
            "the value/return path."
        )


def _run_one_seeded_update() -> dict[str, float]:
    """Seed RNG, build a fresh K=1 CPU/FP32 agent, fill its buffer, and update().

    Captures the FULL deterministic sequence (seed -> construct -> fill -> update)
    behind a single torch.manual_seed so two back-to-back invocations are bit-for-bit
    identical. The invariant under test is DETERMINISM-GIVEN-SEED, not RNG-freedom: at
    the production default (enable_contribution_aux=True) the network stays in training
    mode, so the aux contribution-predictor's Dropout(0.1) draws one mask in the no_grad
    anchor and one per epoch -- the update is NOT RNG-free. But those draws are identical
    across two identically-seeded runs, so the metrics still match. The MAIN policy/value
    path re-scores already-chosen actions with the residual-LSTM dropout at 0.0 (Identity),
    so it draws no RNG and the ratio==1.0 / value path are independent of the anchor's
    aux draw. A divergence here would mean a genuinely unseeded entropy source, not the
    anchor "leaking" entropy.
    """
    torch.manual_seed(20240617)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=None,
        recurrent_n_epochs=1,
    )
    _fill_buffer(agent, slot_config)
    return agent.update(clear_buffer=True)


def test_two_identical_seed_updates_match_k1() -> None:
    """Two identical seeded K=1 updates produce bit-stable scalar metrics (Acceptance #3).

    Precursor to the multi-epoch determinism guarantee: with the anchored reference
    pass in place, the K=1 update must be fully deterministic. We run the IDENTICAL
    sequence (same seed, same construction, same fill, same update) twice and assert
    the headline scalar metrics agree to <1e-6.

    The load-bearing invariant is DETERMINISM-GIVEN-SEED, not RNG-freedom. The update
    does draw RNG -- at the production default the aux contribution-predictor's Dropout
    draws once in the no_grad anchor and once per epoch (the net is in training mode) --
    but those draws are identical across two identically-seeded runs, so the metrics
    match. The main policy/value re-score (fixed actions, residual-LSTM dropout=0.0)
    draws no RNG, so ratio==1.0 and the value path are unaffected by the anchor's aux
    draw. A divergence here is a real non-determinism bug (an unseeded entropy source),
    NOT a tolerance to be widened.
    """
    metrics_a = _run_one_seeded_update()
    metrics_b = _run_one_seeded_update()

    assert metrics_a["ppo_update_performed"] is True
    assert metrics_b["ppo_update_performed"] is True

    for key in (
        "policy_loss",
        "value_loss",
        "approx_kl",
        "ratio_mean",
        "explained_variance",
    ):
        assert metrics_b[key] == pytest.approx(metrics_a[key], abs=1e-6), (
            f"{key} diverged between two identical seeded K=1 updates: "
            f"A={metrics_a[key]!r} vs B={metrics_b[key]!r}. The K=1 update must be "
            "deterministic given the seed -- a divergence means an unseeded entropy "
            "source desynced the two runs (real determinism bug, do NOT widen tolerance)."
        )


# ---------------------------------------------------------------------------
# K=4 multi-epoch tests (PR2). The anchored reference pass makes multi-epoch
# recurrent PPO mathematically exact: every scored forward is a full-recompute
# TBPTT unroll from theta_0's anchor baseline, so epoch 0 still has ratio==1.0
# and epochs 1..K-1 measure GENUINE policy drift driven by the optimizer steps.
# ---------------------------------------------------------------------------

# Fixed seed for all K=4 fixtures below. Shared with _run_one_seeded_update so the
# K=1-vs-K=4 comparison (2.9) is over the SAME rollout/initialisation.
_K4_SEED = 20240617


def _run_one_seeded_update_k(
    recurrent_n_epochs: int,
    *,
    target_kl: float | None = None,
    lr: float = 3e-4,
    enable_contribution_aux: bool = True,
) -> dict[str, float]:
    """Seed RNG, build a fresh CPU/FP32 agent at the given K, fill, and update().

    Mirrors ``_run_one_seeded_update`` (the K=1 helper) but parameterises the
    epoch count, target_kl, learning rate and aux flag. Pinned to CPU/FP32 with a
    single ``torch.manual_seed`` so the full sequence (seed -> construct -> fill ->
    update) is bit-for-bit reproducible across invocations. At the production
    default (enable_contribution_aux=True) the aux Dropout draws RNG in the no_grad
    anchor and once per epoch, but those draws are identical across identically
    seeded runs, so the scalar metrics still match exactly.
    """
    torch.manual_seed(_K4_SEED)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=target_kl,
        recurrent_n_epochs=recurrent_n_epochs,
        lr=lr,
        enable_contribution_aux=enable_contribution_aux,
    )
    _fill_buffer(agent, slot_config)
    return agent.update(clear_buffer=True)


def test_two_identical_seed_updates_match_k4() -> None:
    """Two identical seeded K=4 updates produce bit-stable scalar metrics (Acceptance #3).

    The K=4 anchored update runs four full-recompute TBPTT epochs in sequence: each
    epoch re-scores the buffer from theta_0's hidden state under the CURRENT weights,
    measures the importance ratio against the frozen theta_0 anchor baseline, and steps
    the optimizer. The whole loop -- the anchor forward, the four epoch forwards, and
    every optimizer step -- is deterministic GIVEN THE SEED: the only RNG is the aux
    contribution-predictor's Dropout (production default), whose draws are identical
    across two identically seeded runs because the seed, construction, fill and update
    order are byte-identical.

    We run the IDENTICAL sequence twice and assert the headline scalar metrics agree to
    <1e-6. A divergence here is a real non-determinism bug in the multi-epoch loop (an
    unseeded entropy source desyncing the two runs), NOT a tolerance to be widened.
    """
    metrics_a = _run_one_seeded_update_k(4)
    metrics_b = _run_one_seeded_update_k(4)

    assert metrics_a["ppo_update_performed"] is True
    assert metrics_b["ppo_update_performed"] is True

    for key in (
        "policy_loss",
        "value_loss",
        "approx_kl",
        "ratio_mean",
        "explained_variance",
    ):
        assert metrics_b[key] == pytest.approx(metrics_a[key], abs=1e-6), (
            f"{key} diverged between two identical seeded K=4 updates: "
            f"A={metrics_a[key]!r} vs B={metrics_b[key]!r}. The K=4 loop must be "
            "deterministic given the seed -- a divergence means an unseeded entropy "
            "source desynced the two runs (real determinism bug, do NOT widen tolerance)."
        )


def _run_k4_with_epoch1_poisoned(target_kl: float | None) -> dict[str, float]:
    """Run a K=4 update with the epoch-1 scored forward NaN-poisoned (interleave case).

    ``evaluate_actions`` is called K+1 times per update(): once for the no_grad anchor
    (call index 0), then once per epoch (call index epoch_i+1). To corrupt ONLY the
    epoch_i==1 forward we wrap ``evaluate_actions`` with a call counter and NaN the
    returned ``value`` tensor on the third call (index 2). That trips the finiteness
    gate (ppo_agent.py ~:1006), which records a finiteness_gate_failures entry for
    epoch 1 and ``continue``s WITHOUT stepping the optimizer -- so epoch 1 contributes
    no policy drift and no approx_kl. The anchor (call 0) and epochs 0/2/3 are pristine.

    Deterministic: the poison is keyed on a deterministic call counter, not on values.
    """
    torch.manual_seed(_K4_SEED)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=target_kl,
        recurrent_n_epochs=4,
        lr=3e-4,
        enable_contribution_aux=False,
    )
    _fill_buffer(agent, slot_config)

    original_evaluate_actions = agent.policy.evaluate_actions
    call_counter = {"n": 0}
    poison_call_index = 2  # anchor=0, epoch0=1, epoch1=2

    def poisoned_evaluate_actions(*args: object, **kwargs: object) -> object:
        result = original_evaluate_actions(*args, **kwargs)
        if call_counter["n"] == poison_call_index:
            result.value[:] = float("nan")
        call_counter["n"] += 1
        return result

    agent.policy.evaluate_actions = poisoned_evaluate_actions
    return agent.update(clear_buffer=True)


def test_kl_early_stop_fires_sanely_k4() -> None:
    """K=4 early-stop fires on REAL drift -- neither always nor never -- and the
    finiteness ``continue`` attributes early-stop to epochs that RAN, not wall epochs.

    The buffer carries a high-variance reward sequence ([0.2, -0.1, 0.3, 0.0] in
    _fill_buffer), so GAE produces high-variance advantages and each epoch's optimizer
    step moves theta enough to drive a GENUINE (non-trivial) policy ratio away from 1.0.
    The drift is real: at lr=3e-4 the aggregate approx_kl over K=4 is ~1e-2, with the
    per-epoch KL growing monotonically as the policy walks away from theta_0.

    THREE cases, all deterministic and fixed-seed:

    1. NON-TRIPPING (target_kl=None): early stopping is disabled, so the loop runs all
       four epochs and ``early_stop_epoch`` is ABSENT. This is the "never" guard: a test
       that only ever saw early-stop firing could pass with a hair-trigger bug.

    2. TRIPPING mid-loop (target_kl=0.002): the accumulated drift crosses 1.5*target_kl
       partway through the loop, so early stopping fires at a wall epoch >=1 (here epoch
       1) -- NOT at epoch 0. This is the "always 0" guard: a test that only ever saw
       early-stop at epoch 0 could pass with a fires-immediately bug.

    3. INTERLEAVING (the Systems case): we NaN-poison the epoch-1 scored forward to trip
       the finiteness gate's ``continue`` (ppo_agent.py ~:1045). The skipped epoch steps
       NO optimizer and contributes NO drift, so with a FIXED target_kl the trust region
       is crossed one wall epoch LATER than the clean run. We assert early_stop_epoch is
       attributed to an epoch that actually RAN by comparing against the clean run: the
       poisoned run early-stops STRICTLY LATER (clean: epoch 2; poisoned: epoch 3) and
       records the skipped epoch as a finiteness failure, never as the early-stop epoch.
    """
    # --- Case 1: NON-TRIPPING (early stopping disabled) ---
    metrics_never = _run_one_seeded_update_k(4, target_kl=None, enable_contribution_aux=False)
    assert metrics_never["ppo_update_performed"] is True
    assert "early_stop_epoch" not in metrics_never, (
        "early_stop_epoch present with target_kl=None -- early stopping fired despite "
        f"being disabled: {metrics_never.get('early_stop_epoch')!r}"
    )
    # Sanity: the drift the early-stop guards on is REAL, not trivially zero.
    assert abs(metrics_never["approx_kl"]) > 1e-4, (
        "K=4 approx_kl is ~zero -- the fixture produced no real policy drift, so the "
        f"early-stop cases below would be vacuous: approx_kl={metrics_never['approx_kl']!r}"
    )

    # --- Case 2: TRIPPING mid-loop (fires at a wall epoch >= 1, not 0) ---
    metrics_trip = _run_one_seeded_update_k(
        4, target_kl=0.002, enable_contribution_aux=False
    )
    assert metrics_trip["ppo_update_performed"] is True
    assert "early_stop_epoch" in metrics_trip, (
        "early_stop_epoch absent with a finite target_kl over a drifting K=4 update -- "
        "early stopping never fired despite real drift crossing the trust region."
    )
    early_stop_epoch = metrics_trip["early_stop_epoch"]
    assert early_stop_epoch >= 1, (
        f"early_stop_epoch={early_stop_epoch} -- fired at epoch 0, before ANY drift could "
        "accumulate. Epoch 0 is the anchor (ratio==1.0, approx_kl==0), so it must never "
        "trip early-stop; a fire-at-0 means a hair-trigger / sign bug in the KL guard."
    )
    assert early_stop_epoch < 4, (
        f"early_stop_epoch={early_stop_epoch} is out of range for K=4 (valid wall epochs "
        "0..3); the telemetry recorded a non-existent epoch."
    )

    # --- Case 3: INTERLEAVING -- finiteness continue desyncs wall index from RAN epochs ---
    # Clean reference at a FIXED target_kl: epoch 1 runs and contributes drift.
    metrics_clean = _run_one_seeded_update_k(
        4, target_kl=0.005, enable_contribution_aux=False
    )
    assert "early_stop_epoch" in metrics_clean
    clean_stop_epoch = metrics_clean["early_stop_epoch"]
    assert metrics_clean["finiteness_gate_skip_count"] == 0, (
        "clean reference run unexpectedly skipped an epoch -- the interleave comparison "
        "below is only meaningful if the clean run ran every epoch."
    )

    # Same target_kl, but epoch 1's scored forward is NaN-poisoned so it hits the
    # finiteness `continue` and steps NO optimizer.
    metrics_poison = _run_k4_with_epoch1_poisoned(target_kl=0.005)
    assert metrics_poison["ppo_update_performed"] is True

    # The poisoned epoch is recorded as a finiteness failure for epoch 1, and as a skip.
    assert metrics_poison["finiteness_gate_skip_count"] == 1, (
        "epoch-1 poison did not trip the finiteness gate exactly once: "
        f"skip_count={metrics_poison['finiteness_gate_skip_count']}"
    )
    failure_epochs = [f["epoch"] for f in metrics_poison["finiteness_gate_failures"]]
    assert failure_epochs == [1], (
        f"finiteness failure attributed to wrong epoch(s): {failure_epochs!r} "
        "(expected exactly the poisoned epoch 1)."
    )

    # The LOAD-BEARING attribution assertion: because the skipped epoch 1 stepped no
    # optimizer and contributed no drift, the trust region is crossed STRICTLY LATER in
    # the poisoned run than in the clean run. early_stop_epoch therefore tracks the
    # epochs that RAN (accumulated drift), not the wall-clock epoch slots. If early-stop
    # were attributed to wall epochs independent of whether they executed, the poisoned
    # and clean runs would early-stop at the SAME epoch.
    poison_stop_epoch = metrics_poison["early_stop_epoch"]
    assert poison_stop_epoch > clean_stop_epoch, (
        f"poisoned early_stop_epoch={poison_stop_epoch} did not move LATER than the "
        f"clean run's {clean_stop_epoch}. The finiteness `continue` skipped epoch 1's "
        "optimizer step, so the policy drifted less and must cross the trust region at a "
        "later wall epoch. Equal epochs would mean early-stop telemetry is attributed to "
        "wall epochs rather than the epochs that actually executed an update."
    )
    # And the skipped epoch is NEVER the recorded early-stop epoch.
    assert poison_stop_epoch != 1, (
        f"early_stop_epoch={poison_stop_epoch} equals the skipped epoch 1 -- early-stop "
        "was attributed to an epoch that hit the finiteness `continue` and never ran."
    )


def test_value_loss_same_order_k1_vs_k4() -> None:
    """value_loss / return_mean / return_std stay the same order of magnitude K=1 vs K=4.

    The value path is untouched by the multi-epoch change: GAE and the running value
    normalizer both execute exactly ONCE per update() (in the pre-loop), so the return
    targets are identical at K=1 and K=4. value_loss is recomputed each epoch against
    those fixed targets; under K=4 the critic takes additional steps, so value_loss may
    drift modestly, but it must stay on the SAME ORDER OF MAGNITUDE -- a K=4 value_loss
    that exploded or collapsed by orders of magnitude would signal the value path was
    perturbed by the epoch loop (e.g. a re-normalisation leak or a stale-target bug).

    return_mean / return_std come straight from the once-per-update normaliser, so they
    are EXPECTED to be identical (not merely same-order) between K=1 and K=4; we assert
    finiteness and same-order on all three, with a generous ~10x band on value_loss.
    """
    metrics_k1 = _run_one_seeded_update_k(1, target_kl=None)
    metrics_k4 = _run_one_seeded_update_k(4, target_kl=None)

    assert metrics_k1["ppo_update_performed"] is True
    assert metrics_k4["ppo_update_performed"] is True

    for key in ("value_loss", "return_mean", "return_std"):
        v1 = float(metrics_k1[key])
        v4 = float(metrics_k4[key])
        assert torch.isfinite(torch.tensor(v1)).all(), f"K=1 {key} non-finite: {v1!r}"
        assert torch.isfinite(torch.tensor(v4)).all(), f"K=4 {key} non-finite: {v4!r}"

        # Same order of magnitude: generous ~10x band in each direction. Guards against an
        # explosion/collapse from the epoch loop while tolerating modest multi-epoch drift.
        # Both values are non-trivially non-zero here, so a plain ratio test is well-posed.
        assert abs(v1) > 1e-9 and abs(v4) > 1e-9, (
            f"{key} is ~zero (K1={v1!r}, K4={v4!r}); the order-of-magnitude check is "
            "ill-posed -- the fixture degenerated."
        )
        ratio = abs(v4) / abs(v1)
        assert 0.1 <= ratio <= 10.0, (
            f"{key} changed by more than ~10x between K=1 and K=4: K1={v1!r}, K4={v4!r}, "
            f"ratio={ratio:.3f}. The value path runs GAE + the normaliser once per update "
            "in both, so an order-of-magnitude shift means the epoch loop perturbed the "
            "value/return path."
        )
