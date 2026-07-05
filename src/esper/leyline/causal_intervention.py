"""Shared contracts for the causal-contribution run harness.

These contracts back the parallel-control causal-contribution experiment
(``docs/plans/concepts/2026-06-28-causal-contribution-run-design-v2.md``). They
are intentionally minimal and estimand-invariant: the only intervention wired
here is SUPPRESS-SLOT (a stationary mask that makes one slot un-committable for
the whole run). SUPPRESS-COMMIT (force-prune) is deliberately NOT defined here.

The RNG-domain contracts encode the three-domain split (§5.5) that keeps the
intervention-OFF run common-random-number (CRN) identical to the matched control
up to the first suppression event. The determinism class is declared as
CRN/statistical, NOT bit-exact (§5.6): no GPU-determinism configuration exists in
this repo, so cross-build bit-exactness is unachievable; the harness proves the
weaker-but-sufficient CRN property via the per-domain compare-point hash.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# SUPPRESS-SLOT intervention
# ---------------------------------------------------------------------------

#: Lifecycle policy string keyed by ``apply_proof_baseline_action_controls``.
#: A stationary mask making the target slot un-committable for the whole run.
SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY = "suppress_slot_r0c0"

#: The slot the R1 pilot suppresses. The neg per-seed LOO marginal early-conv
#: stem (see design §0/§1). Keyed via ``slot_config.index_for_slot_id``.
SUPPRESS_SLOT_TARGET_SLOT_ID = "r0c0"

#: Reported mode for the matched-control arm (RNG split ON, no stationary mask).
#: A named constant so telemetry consumers join on it instead of a bare literal.
CONTROL_MODE = "control"


class ForcedStepReason(str, Enum):
    """Why a rollout step was forced (telemetry split — §5.7).

    A single pooled forced-step ratio conflates two distinct mechanisms; the
    causal read needs them separated. ``WAIT_SATURATION`` is the natural state
    (only WAIT was ever valid this step) and must show parity across arms;
    ``INTERVENTION`` is a step the SUPPRESS-SLOT mask flipped to WAIT-only
    because the suppressed slot was the env's only remaining option.
    """

    INTERVENTION = "intervention"
    WAIT_SATURATION = "wait_saturation"


# ---------------------------------------------------------------------------
# RNG three-domain split (§5.5) + determinism class (§5.6)
# ---------------------------------------------------------------------------


class RngDomain(str, Enum):
    """The three logically-distinct RNG consumers split apart in §5.5.

    Today a single global generator is seeded once and shared by all three; the
    rollout samples actions, so suppressing a slot changes how many blueprint
    /host draws happen between arms, offsets the shared stream, and injects a
    pure RNG artifact straight into Δ_struct (the primary discriminator). The
    split gives each consumer an independent, separately-hashable stream.
    """

    #: Policy/controller sampling (torch.multinomial: op draw + per-head draws).
    CONTROLLER = "controller"
    #: Host stochasticity (the germination shape-probe randn). Hygiene only: the
    #: probe output is discarded (a shape smoke test), so this domain's *values*
    #: never touch training. Splitting it keeps the controller stream and the
    #: compare-point clean, it does NOT remove a training-affecting artifact.
    HOST = "host"
    #: Blueprint weight-init draws at germination. Content-addressed per
    #: germination so a blueprint gets identical weights in both arms regardless
    #: of how many germinations preceded it — the only thing that survives the
    #: divergence point.
    BLUEPRINT_INIT = "blueprint_init"


class DeterminismClass(str, Enum):
    """Declared determinism guarantee for the harness."""

    #: Common-random-number / statistical determinism. Shared exogenous inputs
    #: and shared controller stream up to the first divergent masked draw; NOT
    #: bit-exact across builds/GPU.
    CRN_STATISTICAL = "crn_statistical"
    #: Bit-exact reproduction. NOT offered here (no GPU-determinism config).
    BIT_EXACT = "bit_exact"


#: The class the harness declares and the compare-point proves.
INTERVENTION_DETERMINISM_CLASS = DeterminismClass.CRN_STATISTICAL

# Deterministic child-seed offsets so each domain is derived from the master
# seed but advanced independently. Distinct from AUGMENT_SEED_OFFSET (=1) so the
# host domain never collides with the augmentation generator. Large primes keep
# domains well-separated even for adjacent master seeds.
CONTROLLER_RNG_SEED_OFFSET = 0x5EED_C0DE  # 1592706270
HOST_RNG_SEED_OFFSET = 0x4057_0057  # 1079185505


__all__ = [
    "SUPPRESS_SLOT_R0C0_LIFECYCLE_POLICY",
    "SUPPRESS_SLOT_TARGET_SLOT_ID",
    "CONTROL_MODE",
    "ForcedStepReason",
    "RngDomain",
    "DeterminismClass",
    "INTERVENTION_DETERMINISM_CLASS",
    "CONTROLLER_RNG_SEED_OFFSET",
    "HOST_RNG_SEED_OFFSET",
]
