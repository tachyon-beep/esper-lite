# PDR-0067 — cf-earns-its-keep screen (design) + pre-launch read-review & external-relevance gates

Date: 2026-07-13
Status: accepted for the GATES + no-GPU findings (within grant); the cf-off SCREEN and the
epic reframe are PROPOSED — owner-gated (new direction / next Now bet). Synthesizes advisor +
3 external-peer reviews (GPT last-say). Builds on PDR-0066.

## No-GPU findings (done 2026-07-13)

- **Block-bootstrap hardens the retractions:** `Var(G_cf)` slopes 250–600 are −53 (90% CI
  [−50,+53]) and +41 (CI [−45,+43]) — both cross zero. No "plateau-1 split," no drift.
  `ev_cf` ~flat. "ceiling/still-rising/split" language retracted with uncertainty.
- **Path-C probe NOT feasible from retained data** — only aggregates persisted, no Obs V3
  per-timestep sequences. A ceiling probe needs a fresh instrumented run; off the cheap list.
- **+0.04 recalibration lift** corrected: matched pooled lift ≈ 0 (was a pooled-vs-median
  artifact); any residual mismatch's cause is UNIDENTIFIED (normalizer-lag was an unsupported
  attribution — `a=1.40` is on `V_main`, immaterial since `Var(e_main)≈0`).

## Process gates adopted (amend the run-authorization discipline)

**(G1) Pre-launch instrument-validity review** — a decision-bearing GPU run's pre-committed
read is independently (advisor) reviewed BEFORE launch. Triggers: expected GPU cost over
threshold ∨ read selects between architectural branches ∨ can terminate/authorize an epic ∨
newly-constructed metric ∨ non-obvious population/denominator. One-page packet answers:
(1) what hypotheses does the read separate; (2) value/distribution under each; (3) can both
produce the same read; (4) what can it structurally NOT establish; (5) population/denominator/
missingness; (6) is the null informative at expected coverage; (7) what downstream decision
changes if it passes; (8) can existing telemetry answer it without a run. **Plus one synthetic
counterexample:** construct a world where A holds, one where B holds, show the read separates
them. (This would have caught the frozen-head discriminator pre-launch.)

**(G2) Program external-relevance gate** — no critic architecture is banked merely because EV
improves; at least one downstream quantity must improve or be causally protected (final
val-acc; all-on − host-alone contribution; acc per added M-param; compute-adjusted efficiency;
compact-motif frequency; destructive-intervention frequency). EV / return-variance / advantage
stay MECHANISM telemetry, not endpoints.

**(G3) Stop rule** — no new metric-specific optimization epic begins until the metric's
marginal relationship to a product outcome is demonstrated OR explicitly declared as the
experiment being run.

## PROPOSED next experiment — paired cf-off earn-its-keep screen (owner-gated)

Does the CURRENT DENSE `R_cf` improve the accuracy–efficiency frontier enough to justify its
target noise? Uses the ACCEPTED direct-`V_total` critic in BOTH arms (does NOT reintroduce
rejected Objective A).

- **Arm 1 (current reward):** `R = R_main + R_cf`, direct `V_total` baseline/bootstrap.
- **Arm 2 (dense cf removed):** `R = R_main`; `R_cf` still computed + logged but does NOT
  enter reward or return; direct `V_total`.
- **Held identical:** PBRS, terminal-accuracy term, parameter rent, intervention costs,
  state-machine penalties, observation schema, entropy schedules, critic architecture, seeds.

Primary outcomes (downstream — NOT EV, NOT J as sole endpoint; J is cf-derived and misses
synergy per Phase-0): terminal val-accuracy; all-on − host-alone contribution; added-param
footprint; compute multiplier; acc per M added param; compact high-value-motif frequency;
destructive/tanking-intervention frequency. EV/return-var/advantage = mechanism reads.
Paired n=5 first leg; marginal/heterogeneous ⇒ n=10 discipline.

Pre-committed reading:
- cf-off preserves accuracy AND improves efficiency ⇒ the current dense cf term does not earn
  its keep ⇒ favour true de-shaping / auxiliary-only cf (`V_main`/clean return as PPO
  baseline, cf as telemetry/aux target).
- cf-off materially lowers accuracy or destroys efficient motifs ⇒ cf load-bearing ⇒ a safer
  critic treatment (heavy-A′ vs paired direct-`V_total` control) regains priority.
- Seed-heterogeneous ⇒ cf regime-dependent ⇒ third arm: cf auxiliary/gated/hindsight-only.

This screen is subject to G1 (its pre-committed reading gets the pre-launch advisor pass
before any launch).

## Owner DECIDE pending
- Frame the cf-off screen as the epic's new Now bet, or as a smaller screen before deciding
  the epic's fate?
- Confirm the primary downstream endpoint priority (GPT proposes accuracy + contribution +
  efficiency over J/EV).

## Reversal trigger
If the cf-off screen shows cf is load-bearing, the epic returns to a critic/de-shaping angle —
but now with a downstream baseline (G2 satisfied), not blind.
