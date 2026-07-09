# PDR-0050 — Step-3 scalars frozen; LEG-B demoted to descriptive; ON wave cleared

Date: 2026-07-09
Status: decided (owner-ratified via explicit two-question decision)
Decider: owner (john); analysis and options by owner-agent

## Context

The 5-seed OFF wave completed clean (200/200 updates per arm, 0 skips, 0 tracebacks;
relaunched wave at `fe177844` after the 2026-07-08 session-reap kill). §10 step-3
calibration ran per the frozen §11 formulas. Two pre-registered escalation triggers
fired, plus one harness spec-vs-code finding. Full evidence:
`docs/analysis/2026-07-09-stage2-off-calibration-step3.md` (+ harness report
`2026-07-09-stage2-off-calibration-report.txt`); amendment text: gate doc §11.1.

## Frozen scalars

W = 17 · δ = 0.050311 · Δparam_max = 1527.8 · τ_acc = 0.3pp (unchanged) ·
G3 spread confirm PASS. ε_rel = 0.10 remains recorded but LEG-B no longer gates
(below).

## The calls

1. **LEG-B → descriptive at the n=5 screen** (chosen over raising ε_rel to ~0.6 or
   pre-registering a windowed region). Grounds: OFF seed-to-seed volatility spread
   0.5716 ≥ 0.10 (ε_rel validation FAIL) and material within-run non-stationarity
   (second/first-half vol ratios 2.4–13.6×) — LEG-B cannot be powered at n=5 on this
   horizon; first half is flat-dead, second half carries the spread, so no window is
   both stationary and informative. Screen predicate = LEG-A ∧ MECH ∧ G1–G4; the
   scorer's emitted composite verdict (original §7) is superseded at the screen tier —
   read leg_a/mech/g1–g4 fields. Volatility claims deferred to n=10 or a redesigned
   pre-registered metric.
2. **Device placement = provenance, not frozen config** (chosen over rerunning 42/44
   on cuda:0 and a single-GPU ON wave). §10 never froze device; the ratified launch
   plan split seeds across GPUs. Harness amended read-path-only: `RunMeta.placement`
   split out of `frozen_config`; cross-seed homogeneity ignores placement; per-pair
   ON==OFF placement equality added to `validate_pair`; TDD (3 new tests), 182
   harness tests green, ruff+mypy clean. Freeze-safe: proof-layer only, no
   training-path change; same-commit discipline intact.

## Consequence

ON wave (seeds 41–45, `configs/ablations/stage2-ab-on.json` — verified single-key
diff `hra_value_decomposition: true`) launches with device assignments mirroring OFF
partners (41/43/45 → cuda:0, 42/44 → cuda:1), session-proof launcher
(setsid/nohup; the 2026-07-08 lesson). Amendment recorded before any ON telemetry
exists. Run-commit provenance: ON launches from the SAME commit as OFF (`fe177844`)
with the working tree carrying only the read-path harness amendment + docs/tests —
no training-path file differs from the OFF launch tree; the amendment is committed
separately post-launch so both arms' §0 provenance records match.

## Reversal trigger

If the ON wave's packet shows LEG-B would have PASSED cleanly under the original
predicate (both legs, all pairs, wide margin), note it in the read but do NOT
retro-promote — the demotion stands for the screen; promotion happens only at a
pre-registered n=10 tier.
