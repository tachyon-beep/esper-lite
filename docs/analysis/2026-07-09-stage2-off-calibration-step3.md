# Stage-2 OFF calibration — §10 step-3 record (2026-07-09)

Evidence: 5/5 OFF arms complete and §1-valid (200/200 PPO updates each, 0 skipped,
2400/2400 episode outcomes, 0 tracebacks, rc=0). Runs `telemetry/stage2_ab_off/seed{41..45}`
at commit `fe177844` (2 training-inert commits atop frozen `04b58ed4`; PDR-0048 discipline
intact). Scoring surface: depth-1 symlink root `telemetry/stage2_ab_off/packet_root/`
(run_dirs `off_s41..off_s45`; ingestion clean; stdout logs linked into §1 traceback scope).
Provenance note: first wave (launched 11:34 2026-07-08) was externally killed at 16:04 by
launcher-session close — partials quarantined `killed-by-session-reap-1/`, never scored;
this wave is a clean fresh relaunch (23:16 → 17:40).

## Resolved scalars (frozen §11 formulas, applied as written)

| Symbol | Rule | Resolution |
|---|---|---|
| `W` | `max(10, plateau(value_target_scale)+5)`; plateau = first u with trailing-8-window max rel-change < 5% | **17** (per-arm plateaus 8/8/12/8/8; worst 12) |
| `δ` | `max(0.05, paired_bootstrap_SE(OFF ev-level medians))` via `calibrate_off` | **0.0503** |
| `Δparam_max` | `max(0.10·median_s(added_params_off), IQR_s(added_params_off))` | **1527.8** (added_params/seed: 2637.6, 1308.3, 3373.0, 879.0, 2836.1; median 2637.6 — NOT degenerate-zero) |
| `τ_acc` | fixed | 0.3pp (unchanged) |
| `ε_rel` | fixed 0.10 **subject to step-3 validation** | 0.10 — **VALIDATION FAILED, see escalation 1** |

OFF per-arm: ev levels [0.1316, 0.1138, 0.1936, 0.0449, 0.2823];
ev IQRs [0.3866, 0.4124, 0.4678, 0.2569, 0.4603]; scored updates 183/183/183/182/183
(≥150 budget); §8B floored fractions ~0 (max 0.005).

## Pre-registered checks

- **G3 churn spread confirmation: PASS.** Seed-to-seed max/min: germinate 1.006,
  prune 1.007, fossilize 1.379 — all < 1.5.
- **ε_rel OFF-spread validation: FAIL → ESCALATE (pre-registered).** OFF seed-to-seed
  relative spread of `IQR_u(sqrt(1−ev))` = **0.5716 ≥ 0.10**. LEG-B's fixed floor sits
  deep inside its own noise floor at n=5.
- **Stationarity pre-check: MATERIAL NON-STATIONARITY → ESCALATE (pre-registered).**
  Second-half vs first-half `IQR_u(sqrt(1−ev))` of the scored window, per arm:
  8.41× / 7.70× / 4.50× / 13.55× / 2.38× (all arms elevated). Cause visible in the
  series: EV sits near 0 for the first half then lifts (final-window levels 0.04–0.28)
  while Var(returns) grows ~10–25× within-run — the residual is *moving* late, so the
  LEG-B statistic conflates learning-drift with noise on this horizon.

## Harness finding (spec-vs-code, owner adjudication)

`off_calibration_validity_reasons` rejects the OFF set: `_FROZEN_RUN_CONFIG_COLUMNS`
includes `policy_device`/`env_devices_json`, and the ratified launch plan split seeds
across GPUs (41/43/45 cuda:0; 42/44 cuda:1). §10's frozen list does NOT include device
placement; every non-device field is identical across the 5 arms. Proposed amendment
(read-path only, freeze-safe): treat placement fields as reported provenance, excluded
from the cross-seed homogeneity check, while RETAINING per-pair device equality at score
time (each ON arm must run on the same device as its OFF partner) + regression test.
δ/ε_rel-anchor above were computed via the library `calibrate_off` with device fields
set aside (all other validity reasons empty).

## Disposition

Escalated to owner same-day with the specific failed checks; **owner ruled 2026-07-09**:
(1) LEG-B demoted to descriptive at the n=5 screen — composite predicate becomes
LEG-A ∧ MECH ∧ G1–G4; volatility claim deferred to n=10 or a redesigned pre-registered
metric. (2) Device placement fields are provenance, not frozen config — harness amended
read-path-only (RunMeta.placement split + per-pair equality in validate_pair, regression
tests added, 182 harness tests green); each ON arm must run on its OFF partner's device.
Rulings recorded in gate doc §11.1 and PDR-0050. ON wave cleared for launch.
