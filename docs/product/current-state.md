# Current State — Esper        Checkpoint: 2026-07-03 (checkpoint #7 — PIN-E harness BUILT; noise-floor runs IN FLIGHT)

## The bet right now
**Reward credit-assignment redesign.** Term BUILT default-OFF (PDR-0013); **enablement gate
OPENED (PDR-0014)** and its first leg — the PIN-E placebo noise-floor harness — is **BUILT,
ACCEPTED, and measuring** (PDR-0015): full 2-arm GPU runs in flight to set tau.
`shapley_synergy_scale=0.0` everywhere. Metric: committed-J / corr(reward,J) in the
enablement A/B; immediate sub-metric: tau (see metrics.md, preliminary +0.252 pp).

## In flight
- **PIN-E noise-floor runs** (esper-lite-94869250f1, in_progress): 3 seeds × std=1e-3 on
  cuda:0 + 3 seeds × std=1e-4 (epsilon arm) on cuda:1 → `telemetry/pin_e/`. Smoke verified
  NON-degenerate (PDR-0014 reversal trigger NOT tripped). On completion: run
  `scripts/pin_e_placebo_analyze.py telemetry/pin_e` (D1 GATE-0 line + D2 tau, committed
  9be132c1), write `docs/analysis/2026-07-03-pin-e-placebo-noise-floor.md`, post the tau
  memo to esper-lite-f22a1d48a7, close the task, move the plan to completed/.
- **Enablement gate** (esper-lite-f22a1d48a7, blocked_by the above): after tau — F2
  scale/cap calibration, hard off-switch-J efficiency + fossilize-count gate, entrenchment
  monitor, ON-run dormancy recheck, paired ≥5-seed OFF/ON A/B. Enabling scale>0 stays
  owner-gated.
- **Telemetry hygiene** (esper-lite-425dcc4ca2): unchanged. (The fixed-schedule runs print
  per-head entropy-collapse warnings — expected: the policy is a masked passenger; alarm
  miscalibration is this issue's scope.)

## Open questions / blocked-on-owner
- **n=10 vs bank-at-n=5 (MAGNITUDE, PDR-0009):** still unanswered; independent of this leg.
- **metrics.md TARGET numbers** still `<owner-set>` placeholders.
- **Owner's working tree:** phase-0 doc/CLAUDE.md/AGENTS.md/.gitignore edits remain
  uncommitted (untouched by the agent); seed_residency test gap = esper-lite-obs-7240279be3.
- **NUM_BLUEPRINTS 13→14 cost (PDR-0015):** pre-change checkpoints don't load; all paired
  A/Bs must be same-commit. Standing constraint, not a question.

## Last checkpoint did (checkpoint #7)
- **PDR-0015:** k=1→k=3 measurement redesign (dual-review caught φ≡c_paid at k=1 before any
  code), Option A enum cost accepted, HOLDING-forever schedule; build ACCEPTED.
- **Delivered WI-1..7** (commits 4d1e4c54..9be132c1, TDD-first): placebo blueprint + PLACEBO
  action + declared-schedule registry + seed_lr_override + serial schedule + WI-5 gate test
  + run driver with hard measurement preflight + offline D1/D2 analyzer (agent-built,
  independently reproduced, ~60 new tests). Rec3 masking-equivalence closed (1c05ba4f).
- **Metrics:** first tau reading (preliminary smoke) added; non-degeneracy verdict recorded.
- Tracker: esper-lite-94869250f1 progress comments #95/#96; gate description corrected
  (pointer mis-binding fixed per PDR-0014).

## Next session, start here
**If the runs finished:** analyze → analysis doc → tau memo on esper-lite-f22a1d48a7 →
close esper-lite-94869250f1 → plan to completed/. **If a run died:** driver is resumable
per-seed (`--seeds`), preflight refuses corrupted configs. Then the gate's next leg (F2
calibration) is DECIDE-ready.
