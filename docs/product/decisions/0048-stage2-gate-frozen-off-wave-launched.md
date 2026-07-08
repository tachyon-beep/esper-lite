# PDR-0048 — Stage-2 MAJOR-1 gate formulas frozen; OFF wave launched; ON remains owner-gated

Date: 2026-07-08   Status: accepted
Author: Claude (agent)   Owner sign-off: YES — owner ratified the threshold set and the
posture amendment in-session (AskUserQuestion, 2026-07-08) and confirmed the authority
grant unchanged. Related: PDR-0044 (harness closed), PDR-0047 (escrow fix), PDR-0021
(prior A/B seed convention), PDR-0026 (rollback asymmetry = RCA-not-gate), epic
esper-lite-f25b71c165.

## Context

The Stage-2 MAJOR-1 acceptance harness was code-complete and tracker-closed (PDR-0044)
but the gate doc's §11 owner-set threshold slots were unfrozen and no A/B had run. The
owner directed this session at freeze + launch. A drl-expert SME review of the proposed
slot values ran before ratification
(`docs/analysis/2026-07-08-stage2-hra-major1-gate-threshold-review.md`).

## Options

- (a) Freeze the proposed slots as-is and launch.
- (b) Freeze with the reviewer's adjustments folded in, commit all resolution FORMULAS
  before any OFF data exists, then launch the OFF wave. **Chosen.**
- (c) Defer freeze until after the OFF arms run. Rejected: choosing formula shapes after
  seeing OFF data is a soft peek; the commitment moment must precede the data.

## The call

1. **Threshold set ratified** (owner, 2026-07-08) with two review adjustments:
   `g4_abs_floor` 5→2 (a uniform floor of 5 would silence ON-specific alarm-channel
   elevations of 3–4 events; code semantics `(on−off)>floor AND on>ratio·off`, plus the
   hardcoded `reward_hacking>0` ON-arm hard trip, verified in code); and a mandatory
   step-3 ε_rel validation (OFF seed-to-seed relative spread of `IQR_u(√(1−ev))` must be
   < 0.10 or LEG-B gates inside its own noise floor → escalate). Also added: δ tier
   semantics (recompute from all 10 OFF arms at claim tier; bootstrap deterministic —
   2000 resamples, rng 0), a precise `plateau()` for W (trailing 8-update window, max
   relative change < 5%, +5 cushion), an OFF-arm stationarity pre-check, and the
   noise-floor calibration principle. Full table now in the gate doc §11.
2. **Pre-data amendment (unit error corrected).** The first ratification set the budget
   as "400 episodes/run" through an env-episodes-vs-updates unit error (the prior A/B's
   "200 episodes/run" means 200 rounds = 200 PPO updates) and estimated ~2h/run. Both
   launches under that plan died at config-load (`task=None`) before producing any data,
   so the amendment is a clean pre-registration correction, not a peek. Corrected:
   **200 rounds/run (2400 env-episodes), min 150 scored updates** post burn-in + §8B
   exclusion, posture = prior-A/B base (`config-3slot-3seed-baseline-shaped.json`:
   3-slot, 12 envs, entropy 0.15→0.08, amp on) + `return_variance_telemetry: true` +
   `per_head_advantage_norm: false` pinned + the HRA toggle. True cost ~9–13h/run
   (measured from prior A/B logs); owner accepted the ~5× larger spend.
3. **Arm configs generated from one source** (`configs/ablations/stage2-ab-{off,on}.json`,
   diff = the flag only) so §10 config-equality holds by construction.
4. **OFF wave launched** (within grant: "launch GPU experiment runs within the active
   bet"): seeds 41/43/45 sequential on cuda:0, 42/44 on cuda:1. Both first runs verified
   training (correct headers, events flowing, lifecycle active). Aborted first-launch
   telemetry shells MOVED (not deleted) to `telemetry/stage2_ab_off/aborted-launch-1/`.
5. **Same-commit discipline:** no training-path code changes between the OFF and ON
   waves; docs-only commits acceptable.
6. **ON launch remains owner-gated** on §10 step-3 scalar resolution (δ, Δparam_max, W,
   ε_rel validation, stationarity check) + explicit approval.

## Rationale

The gate exists to make the Stage-2 read falsifiable and peek-free. Committing formulas
before data closes the last soft-peek channel (choosing formula shapes post-hoc). Every
threshold in the set fails toward the owner (INVALID/INCONCLUSIVE/escalation), never
toward a silent ACCEPT. Running the A/B in the prior-A/B posture keeps the churn/guard
anchors meaningful, keeps the k≥2 organic-coverage watch signal defined (multi-slot),
and doubles LEG-B's per-run sample (200 vs ~96 updates) over the mis-ratified plan.

## Reversal triggers

- If any OFF arm dies or the wave stalls (repeat of the Stage-0 reap), diagnose before
  relaunch; a second unexplained death at a similar point reopens the
  leak-vs-external-kill question (PDR-0033).
- If step-3 validation fails (ε_rel inside OFF noise, non-stationary OFF window, or
  degenerate-zero `added_params_off`), do NOT proceed to ON — escalate to owner with the
  specific failed check.
- If the ON wave cannot run at the same commit (a training-path fix becomes necessary),
  the OFF arms must be re-run at the new commit; do not score cross-commit pairs.
