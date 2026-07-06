# PDR-0015 — PIN-E measurement redesigned k=1→k=3 during build; harness BUILT and ACCEPTED; runs in flight

Date: 2026-07-03   Status: accepted (within grant; design corrections dual-reviewed by
drl-expert + pytorch-expert, both APPROVED_WITH_CHANGES, all REQUIRED changes folded)
Author: Claude (agent)
Related: PDR-0014 (gate opened — its *call* stands; this corrects the measurement design it
sketched), esper-lite-94869250f1, docs/plans/ready/2026-07-03-pin-e-placebo-harness.md

## Context
PDR-0014 opened the gate with the phase-0 §2 sketch: a single placebo, terminal LOO ≡ φ at
k=1, tau from the φ spread. Dual specialist review during DISPATCH falsified two elements
**before any code existed**:

1. **drl R1 (decisive):** at k=1, φ(s) ≡ c_paid(s) = v({s})−v(∅) *identically*, so the gate's
   actual estimand — the signed excess (φ − c_paid) that leaks through the
   max(0, φ − c_paid − tau) deadband — is zero by construction and uncalibratable at k=1.
2. **pytorch (decisive):** registry-string blueprints crash the observation encoder
   (slot.py:548-551), so the placebo requires a `BlueprintAction.PLACEBO` enum member
   (Option A). Cost accepted: NUM_BLUEPRINTS 13→14 shifts the policy-head dim and RNG stream —
   pre-change checkpoints don't load, cross-version bit-comparability is broken, paired A/Bs
   must be same-commit; four golden tests re-baselined with documented procedure.

WI-5 (CPU go/no-go) then falsified two schedule assumptions empirically: the D3 mask rule
forces a SERIAL stagger (no concurrent germination), and the BLENDING alpha ramp is real
(`alpha_speed=INSTANT` does not skip the 5-epoch sigmoid).

## Options
1. Keep k=1 and report tau from φ alone (rejected — measures the wrong estimand; the deadband
   operand is the excess, not φ).
2. k=3 co-resident placebos, offline 2³ Shapley assembly from the fused val pass's factorial
   (already logged at scale=0 every epoch), tau = P99 of SIGNED terminal (φ − c_paid),
   episode-block bootstrap CI (chosen).

## The call
k=3 placebos on a serial declared schedule (`fixed-schedule-hold-placebo-3slot-v1`), held at
HOLDING α=1 forever — **never fossilized** (fossilized slots are ablation-invisible at
scale=0, which would silence the measurement; the Rec3 masking-equivalence tests prove tau
transfers from the HOLDING regime to the live fossil regime, commit 1c05ba4f). Placebo =
3×3 depthwise-conv residual, std=1e-3 init, seed_lr_override=0 (frozen delta, G2 still
measurable). Downward epsilon ladder {1e-3, 1e-4} with plateau acceptance. tau stays
CONSERVATIVE-PROVISIONAL until the first ON calibration run.

**Build ACCEPTED:** WI-1..7 delivered TDD-first across commits 4d1e4c54..9be132c1 (blueprint +
enum + declared-schedule registry + lr-override + serial schedule + gate test + run driver
with hard measurement preflight + offline D1/D2 analyzer, ~60 new tests, suites green).
Smoke run (24 episodes) verified the full chain on GPU and returned a **non-degenerate**
floor: 89.3% of factorial events and 23/24 terminal episodes show nonzero spread —
**PDR-0014's reversal trigger did NOT trip at std=1e-3.** Preliminary smoke tau =
P99(signed excess) = +0.252 pp, CI [+0.106, +0.380], N=72. Full 2-arm runs (3 seeds ×
{1e-3, 1e-4}) in flight on both GPUs.

## Rationale
The estimand correction is not optional — a k=1 tau would have calibrated the deadband
against a distribution that is identically zero, silently admitting every real excess. The
enum cost (broken cross-version comparability) is bounded because all enablement A/Bs were
already required to be same-commit paired runs.

## Reversal trigger
Same as PDR-0014 for degeneracy, now sharpened: if the FULL runs (≈1080 episodes/arm) return
a degenerate floor, or the epsilon arms fail plateau (tau at 1e-4 differs from 1e-3 beyond
their bootstrap CIs, indicating the floor tracks the placebo's own magnitude rather than
measurement noise), the tau method reverts to methodology §5 option (a) — K resampled val
minibatches — and esper-lite-94869250f1 re-scopes. If the full-run P99 CI is too wide to
bound a false-positive budget, extend seeds before accepting a vacuous tau.
