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
