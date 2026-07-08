# Current State — Esper        Checkpoint: 2026-07-08 (afternoon) · checkpoint #34 (PDR-0048; on `feat/ev-stab-stage2-hra`)

## The bet right now
**EV-stabilization** (esper-lite-f25b71c165, Now bet) has moved from "harness built" to
**experiment running**: the Stage-2 MAJOR-1 gate's threshold formulas are FROZEN
(owner-ratified 2026-07-08 after a drl-expert review) and the **5-seed OFF wave is in
flight** (seeds 41/43/45 on cuda:0, 42/44 on cuda:1; prior-A/B posture, 200 rounds/run,
~9–13h each, wave ≈ 1.5 days). The metric this leg moves is the Stage-2 MAJOR-1
composite (LEG-A `ev_sum` non-inferiority ∧ LEG-B advantage-path volatility ↓ ∧ MECH ∧
G1–G4), read only through the pre-registered packet.

## In flight
- **Stage-2 OFF wave (running):** logs at `telemetry/stage2_ab_off/logs/`; run dirs
  `telemetry/stage2_ab_off/seed<NN>/telemetry_*`. Queue scripts notify on completion or
  first failure. Aborted first-launch shells quarantined in
  `telemetry/stage2_ab_off/aborted-launch-1/` (moved, not deleted).
- **After the OFF wave (next session's work):** §10 step-3 calibration —
  `scripts/stage2_packet.py calibrate` for δ; resolve `Δparam_max` and `W` via the
  PDR-0048 formulas; run the ε_rel OFF-spread validation and the stationarity
  pre-check. ALL results go to the owner for scalar freeze + ON-launch approval.
- **Reward-efficiency statistics (esper-lite-a2abff5ec5, Next bet):** unchanged —
  esper-lite-9678c6d05a (ROI verdict semantics) is the serial pick;
  esper-lite-c3cb5338c4 (CI-safe proof lane) parallel-startable. Good filler while the
  OFF wave runs, BUT observe the same-commit discipline (below) — training-path code is
  frozen until the ON wave completes; Karn/proof-layer work is fine.
- **Branch survivor pick (esper-lite-1f1e55f58f):** unchanged (PDR-0038); owner-gated.

## Facts the next session must not relitigate
- **Gate freeze is DONE — PDR-0048.** Formulas committed pre-data; do not reopen
  threshold choices. The drl-expert review artifact is
  `docs/analysis/2026-07-08-stage2-hra-major1-gate-threshold-review.md`; the frozen
  table is §11 of the gate doc.
- **Budget units:** the pre-registered budget is **200 rounds/run = 200 PPO updates**
  (2400 env-episodes), min 150 scored updates. A "400 episodes" figure anywhere is the
  corrected pre-data unit error — the gate doc §11 records the amendment.
- **Posture:** arm configs `configs/ablations/stage2-ab-{off,on}.json`, generated from
  the prior-A/B base; only `hra_value_decomposition` differs. Regenerate via one source
  if ever touched — never hand-edit one arm.
- **SAME-COMMIT DISCIPLINE (PDR-0048): no training-path code changes until the ON wave
  completes.** Docs-only commits acceptable. A forced training fix invalidates the OFF
  arms (re-run at the new commit; never score cross-commit pairs).
- **Escrow shaping state settled — PDR-0047.** Obs V3 schema v2, checkpoint v4,
  120/132-dim contract stand.
- **Branch:** stay on `feat/ev-stab-stage2-hra`. No push without an explicit owner ask.

## Open questions / blocked-on-owner
- **ON launch (standing gate):** after step-3 calibration, the owner must freeze the
  resolved scalars (δ, Δparam_max, W) and the validation outcomes (ε_rel spread,
  stationarity, degenerate-zero param guard), then explicitly approve the ON wave.
- **Step-3 failure paths:** if ε_rel sits inside OFF noise, the OFF window is
  non-stationary, or `added_params_off` is degenerate-zero → escalate with the specific
  failed check; do NOT proceed to ON (PDR-0048 reversal triggers).
- **North-star target/date, rent ceiling, host-accuracy floor:** still owner-unset
  (unchanged; not blocking the A/B — τ_acc=0.3pp guards accuracy inside the gate).
- **Standing owner gates:** no push, tag, release, branch deletion, telemetry deletion,
  or remote action without explicit approval.

## Last checkpoint did (checkpoint #34)
- Ratified + froze the Stage-2 §11 threshold set (drl-expert review; g4_abs_floor 5→2,
  ε_rel step-3 validation added; δ tier semantics, plateau() definition, stationarity
  check pre-registered) — PDR-0048.
- Caught and corrected the budget unit error pre-data (400 env-episodes → 200 rounds);
  re-ratified posture as prior-A/B faithful at ~5× the stated cost.
- Generated single-source arm configs; launched + verified the 5-seed OFF wave on both
  GPUs; quarantined aborted launch shells; updated the epic tracker with provenance.
- Committed the gate-doc freeze, review artifact, arm configs, and workspace.

## Previous checkpoint did (checkpoint #33)
- Closed stale verification-only task esper-lite-e6382020d2; committed the escrow
  telescoping fix as `89b50a16` and closed esper-lite-3defe42928 (PDR-0047).
- Removed `escrow_delta_clip` from contribution reward shaping, added escrow stable
  accuracy + per-slot escrow credit to Obs V3 schema v2, bumped checkpoint compat to v4.

## Next session, start here
Check the OFF wave: `tail telemetry/stage2_ab_off/logs/off-s4*.log` and the queue task
notifications. If all 5 OFF arms completed clean → run §10 step-3 calibration
(`scripts/stage2_packet.py calibrate` + the PDR-0048 formula resolutions + ε_rel /
stationarity / degenerate-zero checks) and bring the owner the scalar-freeze +
ON-launch decision. If a run died → diagnose per PDR-0048 reversal triggers before any
relaunch. While waiting, reward-efficiency work (esper-lite-9678c6d05a) is startable —
Karn/proof layer only, training-path code frozen.
